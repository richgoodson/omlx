# SPDX-License-Identifier: Apache-2.0
"""
Process-level memory enforcer for oMLX.

Monitors total Metal memory usage via mx.get_active_memory() and enforces
the max_process_memory limit by unloading LRU models from EnginePool.

The enforcer runs as a background asyncio task that polls memory usage at
a configurable interval (default: 1 second). When usage exceeds the limit,
it immediately unloads the least-recently-used non-pinned model. If the
model is mid-inference, the inference is aborted as part of engine shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import mlx.core as mx

from .utils.proc_memory import get_phys_footprint

if TYPE_CHECKING:
    from .engine_pool import EnginePool
    from .model_settings import ModelSettingsManager
    from .settings import GlobalSettings

logger = logging.getLogger(__name__)


def _format_gb(b: int) -> str:
    """Format bytes as GB string."""
    return f"{b / 1024**3:.1f}GB"


class ProcessMemoryEnforcer:
    """
    Background task that enforces process-level memory limits.

    Polls mx.get_active_memory() every poll_interval seconds and unloads
    LRU models from EnginePool when the limit is exceeded.
    """

    def __init__(
        self,
        engine_pool: EnginePool,
        max_bytes: int,
        poll_interval: float = 1.0,
        settings_manager: ModelSettingsManager | None = None,
        prefill_memory_guard: bool = True,
        global_settings: GlobalSettings | None = None,
        soft_threshold: float = 0.85,
        hard_threshold: float = 0.95,
        user_explicit_max: bool = False,
        prefill_safe_zone_ratio: float = 0.80,
        prefill_min_chunk_tokens: int = 32,
    ):
        """
        Initialize the process memory enforcer.

        Args:
            engine_pool: The engine pool to evict models from.
            max_bytes: Maximum allowed process memory in bytes (compared
                against max(mx.get_active_memory(), phys_footprint)).
            poll_interval: Seconds between memory checks.
            settings_manager: Optional settings manager for TTL checks.
            prefill_memory_guard: Whether to enable pre-flight memory
                estimation to reject requests that would exceed limits.
            global_settings: Optional global settings for idle timeout.
            soft_threshold: Fraction of max_bytes that triggers soft action
                (LRU non-pinned eviction + admission pause; in-flight allowed).
            hard_threshold: Fraction of max_bytes that triggers hard action
                (also abort in-flight when all loaded models are pinned).
            user_explicit_max: True when max_bytes came from a user-set value
                (CLI / env / settings.json), False when it was derived from
                the "auto" default. Controls whether the scheduler hard
                ceiling honors max_bytes (explicit) or system_ram - 4GB
                (auto, with system protection priority).
            prefill_safe_zone_ratio: Fraction of hard cap below which prefill
                runs at full chunk size; above triggers adaptive shrink.
            prefill_min_chunk_tokens: Floor for adaptive shrink. If even this
                size would exceed the cap, prefill is aborted.
        """
        self._engine_pool = engine_pool
        self._max_bytes = max_bytes
        self._poll_interval = poll_interval
        self._settings_manager = settings_manager
        self._prefill_memory_guard = prefill_memory_guard
        self._global_settings = global_settings
        self._soft_threshold = soft_threshold
        self._hard_threshold = hard_threshold
        self._user_explicit_max = user_explicit_max
        self._prefill_safe_zone_ratio = prefill_safe_zone_ratio
        self._prefill_min_chunk_tokens = prefill_min_chunk_tokens
        self._task: asyncio.Task | None = None
        self._running = False
        # Most recently observed pressure level, consumed by scheduler /
        # admission control. Updated on every poll iteration.
        self._pressure_level: str = "ok"
        # Engine types we've already complained about in
        # _propagate_memory_limit's "scheduler unreachable" path. Prevents
        # the per-poll warning from spamming logs while keeping the first
        # occurrence loud enough to alert CI / oncall.
        self._scheduler_resolve_warned: set[str] = set()

    @property
    def max_bytes(self) -> int:
        """Maximum allowed Metal memory in bytes."""
        return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, value: int) -> None:
        old = self._max_bytes
        self._max_bytes = value
        if self._running:
            self._propagate_memory_limit()
            self._set_metal_memory_limit()
        logger.info(
            f"Process memory limit changed: "
            f"{_format_gb(old)} -> {_format_gb(value)}"
        )

    @property
    def is_running(self) -> bool:
        """Whether the enforcement loop is active."""
        return self._running

    def start(self) -> None:
        """Start the background enforcement loop."""
        if self._running:
            return
        self._running = True
        self._propagate_memory_limit()
        self._set_metal_memory_limit()
        self._task = asyncio.create_task(self._enforcement_loop())
        logger.info(
            f"Process memory enforcer started "
            f"(limit: {_format_gb(self._max_bytes)}, "
            f"interval: {self._poll_interval}s)"
        )

    def _get_hard_limit_bytes(self) -> int:
        """Hard limit propagated to scheduler._memory_hard_limit_bytes for the
        prefill peak check.

        - User-explicit max (CLI/env/settings.json): the user value IS the
          ceiling. They asked for this number to be respected, even on big
          systems where ``system_ram - 4GB`` would otherwise win.
        - Auto mode: ``system_ram - 4GB`` so the kernel keeps headroom for
          itself and prefill gets room above the enforcer's soft/hard
          watermarks. The adaptive prefill throttle
          (``_prefill_safe_zone_ratio`` / ``_prefill_min_chunk_tokens``)
          shrinks the per-chunk transient as we approach the cap, so the
          head_dim>128 SDPA-fallback bursts that previously motivated
          unconditional ``max_bytes`` are now bounded by the throttle
          before they reach Metal.

        Returns 0 if enforcement is disabled (max_bytes <= 0).
        """
        if self._max_bytes <= 0:
            return 0
        if self._user_explicit_max:
            return self._max_bytes
        from .settings import get_system_memory

        return max(get_system_memory() - 4 * 1024**3, self._max_bytes)

    @property
    def _soft_bytes(self) -> int:
        """Soft watermark: max_bytes * soft_threshold."""
        if self._max_bytes <= 0:
            return 0
        return int(self._max_bytes * self._soft_threshold)

    @property
    def _hard_bytes(self) -> int:
        """Hard watermark: max_bytes * hard_threshold."""
        if self._max_bytes <= 0:
            return 0
        return int(self._max_bytes * self._hard_threshold)

    def _current_usage_bytes(self) -> int:
        """Process memory usage as seen by macOS jetsam.

        Combines MLX-reported active memory and the kernel phys_footprint
        ledger. phys_footprint covers anonymous + IOAccelerator + dirty
        file-backed, so it usually dominates; we take max() so MLX-internal
        cache that hasn't been mirrored into phys yet still triggers.
        """
        return max(mx.get_active_memory(), get_phys_footprint())

    def get_pressure_level(self) -> str:
        """Return cached pressure level: 'ok', 'soft', or 'hard'.

        Consumed by scheduler `_schedule_waiting` and HTTP admission control.
        Updated on every enforcer poll iteration.
        """
        return self._pressure_level if self._running else "ok"

    def _set_metal_memory_limit(self) -> None:
        """No-op. Metal-level limits removed to prevent model load swap.

        mx.set_memory_limit() causes MLX to aggressively reclaim cached
        buffers during model loading, creating alloc/free churn that
        pushes the system into swap. All memory enforcement is handled
        by mx.get_active_memory() polling instead. (#429)
        """
        pass

    def _clear_metal_memory_limit(self) -> None:
        """No-op. See _set_metal_memory_limit."""
        pass

    @property
    def prefill_memory_guard(self) -> bool:
        """Whether prefill memory guard is enabled."""
        return self._prefill_memory_guard

    @prefill_memory_guard.setter
    def prefill_memory_guard(self, value: bool) -> None:
        self._prefill_memory_guard = value
        if self._running:
            self._propagate_memory_limit()
            if value:
                self._set_metal_memory_limit()
            else:
                self._clear_metal_memory_limit()
        logger.info(f"Prefill memory guard: {'enabled' if value else 'disabled'}")

    @staticmethod
    def _resolve_scheduler(engine):
        """Return the real Scheduler instance for an EnginePool entry.

        Both BatchedEngine and VLMBatchedEngine in the live engine pool
        store the scheduler at ``self._engine.engine.scheduler`` (the outer
        wrapper holds an AsyncEngineCore at ``_engine`` whose ``.engine``
        is the EngineCore that actually owns the scheduler). Neither
        exposes a top-level ``.scheduler`` attribute, so the previous
        ``getattr(engine, "scheduler", None)`` always returned None for
        real engines and the propagation silently no-op'd — including the
        prefill memory guard flag, which meant the guard was dead at
        runtime regardless of the user's setting. Test mocks set
        ``.scheduler`` directly, so the wrapper-traversal fallback only
        kicks in for real engines.
        """
        sched = getattr(engine, "scheduler", None)
        if sched is not None:
            return sched
        inner = getattr(engine, "_engine", None)
        if inner is None:
            return None
        return getattr(getattr(inner, "engine", None), "scheduler", None)

    def _propagate_memory_limit(self) -> None:
        """Propagate soft/hard memory limits to schedulers for inline prefill checking.

        Invariant: this method is synchronous (no ``await``) so it runs to
        completion within a single event-loop tick. ``EnginePool._load_engine``
        / ``_unload_engine`` / ``EnginePool.discover_models()`` mutate the
        mapping but all run on the same event loop and cannot interleave
        with this loop today. The iteration uses ``list(values())`` so a
        future refactor that moves an EnginePool mutator to a worker
        thread cannot trigger ``RuntimeError: dictionary changed size``
        or — worse — silently miss an engine and leave it without the
        propagated guard / hard limit, regressing the dead-guard bug
        this method exists to fix. The snapshot cost is one cheap copy
        of value references.

        The four guard fields (``_memory_limit_bytes``,
        ``_memory_hard_limit_bytes``, ``_prefill_memory_guard``,
        ``_admission_paused``) are written as direct attributes. On
        current CPython the GIL serializes the per-attribute STORE_ATTRs
        within a single enforcer poll, so the hot-path reader
        (``Scheduler._preflight_memory_check_tokens``) sees a coherent
        (guard, hard_limit) pair.
        """
        hard_limit = self._get_hard_limit_bytes()
        admission_paused = self._pressure_level != "ok"
        guard_enabled = self._prefill_memory_guard
        # Snapshot to a list so a future EnginePool mutator on a worker
        # thread cannot interleave — see method docstring.
        for entry in list(self._engine_pool._entries.values()):
            if entry.engine is not None:
                scheduler = self._resolve_scheduler(entry.engine)
                if scheduler is None:
                    # Rate-limited per-engine-type so a wrapper-chain
                    # change is loud once instead of every poll. Silent
                    # no-op was the failure mode that originally hid the
                    # dead memory guard — surface it now.
                    engine_type = type(entry.engine).__name__
                    if engine_type not in self._scheduler_resolve_warned:
                        self._scheduler_resolve_warned.add(engine_type)
                        logger.warning(
                            "ProcessMemoryEnforcer: could not resolve "
                            "scheduler for engine type %s — prefill memory "
                            "guard will not propagate to this engine. "
                            "Verify the wrapper chain "
                            "(engine._engine.engine.scheduler) still holds.",
                            engine_type,
                        )
                    continue
                scheduler._memory_limit_bytes = self._max_bytes
                scheduler._memory_hard_limit_bytes = hard_limit
                scheduler._prefill_memory_guard = guard_enabled
                scheduler._admission_paused = admission_paused
                scheduler._prefill_safe_zone_ratio = self._prefill_safe_zone_ratio
                scheduler._prefill_min_chunk_tokens = self._prefill_min_chunk_tokens
                bg = getattr(scheduler, "batch_generator", None)
                if bg is not None and hasattr(bg, "_memory_limit_bytes"):
                    bg._memory_limit_bytes = self._max_bytes
                    bg._memory_hard_limit_bytes = hard_limit

    def _walk_store_cache_caps(self) -> None:
        """Walk each scheduler's store-cache gate one step per poll (#1383).

        Driven on every enforcement tick, not just on pressure transitions,
        so the cap converges ±1 per poll toward its pressure-driven target
        (ok -> max_num_seqs, soft/hard -> 1). Decoupled from
        `_propagate_memory_limit` to avoid double-stepping the cap when
        a transition fires.
        """
        for entry in list(self._engine_pool._entries.values()):
            if entry.engine is None:
                continue
            scheduler = self._resolve_scheduler(entry.engine)
            if scheduler is None:
                continue
            adjust = getattr(scheduler, "adjust_store_cache_cap", None)
            if adjust is not None:
                adjust(self._pressure_level)

    async def stop(self) -> None:
        """Stop the background enforcement loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Process memory enforcer stopped")

    async def _enforcement_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self._check_and_enforce()
                await self._check_ttl()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process memory enforcer error: {e}")
            await asyncio.sleep(self._poll_interval)

    async def _check_ttl(self) -> None:
        """Check and unload models that exceeded their TTL."""
        if self._settings_manager is None:
            return
        await self._engine_pool.check_ttl_expirations(
            self._settings_manager,
            global_idle_timeout_seconds=(
                self._global_settings.idle_timeout.idle_timeout_seconds
                if self._global_settings else None
            ),
        )

    async def _check_and_enforce(self) -> None:
        """Check current memory and enforce 2-watermark policy.

        Pressure levels:
        - ok (current < soft): no action, ensure admission unpaused.
        - soft (soft <= current < hard): LRU non-pinned eviction + signal
          schedulers to pause new admissions (in-flight requests proceed).
        - hard (current >= hard): full enforcement — LRU evict, abort
          in-flight when only pinned remain, abort in-progress model loads.

        Pressure target on recovery is the soft threshold (always evict
        back below soft to avoid oscillation when single eviction lands
        just under hard).
        """
        if self._max_bytes <= 0:
            self._pressure_level = "ok"
            return

        current = self._current_usage_bytes()
        soft = self._soft_bytes
        hard = self._hard_bytes
        prev_level = self._pressure_level

        if current < soft:
            new_level = "ok"
        elif current < hard:
            new_level = "soft"
        else:
            new_level = "hard"

        # Propagate every poll iteration so engines loaded AFTER enforcer.start()
        # (i.e. lazy-loaded on first request, which is the normal case for the
        # paged-SSD engine pool) pick up the limits and prefill_memory_guard
        # flag within one poll interval — not only when pressure level happens
        # to change. Without this the first heavy request arrives at a
        # scheduler with _prefill_memory_guard=False / _memory_hard_limit_bytes=0
        # (their defaults) and the guard short-circuits, letting the request
        # enter prefill and hit the underlying Apple IOGPUFamily kernel bug
        # (#1146 / FB22091885). Observed 2026-05-15: 110k-token Qwen3.6-VL
        # prefill rebooted the host because propagation hadn't reached the
        # freshly-loaded engine yet.
        self._pressure_level = new_level
        self._propagate_memory_limit()
        if new_level != prev_level:
            logger.info(
                f"Memory pressure level: {prev_level} -> {new_level} "
                f"(current={_format_gb(current)}, "
                f"soft={_format_gb(soft)}, hard={_format_gb(hard)})"
            )

        if new_level == "ok":
            # Still walk the store-cache cap so it can recover toward
            # max_num_seqs while pressure stays low (#1383).
            self._walk_store_cache_caps()
            return

        # Recover below soft regardless of level — prevents oscillation
        # at the boundary.
        target = soft

        async with self._engine_pool._lock:
            while self._current_usage_bytes() > target:
                victim = self._engine_pool._find_lru_victim()
                if victim is not None:
                    loaded_non_pinned = [
                        mid
                        for mid, e in self._engine_pool._entries.items()
                        if e.engine is not None and not e.is_pinned
                    ]
                    if len(loaded_non_pinned) > 1:
                        # Multiple non-pinned: evict LRU victim cleanly.
                        # abort_all_requests is fired before _unload_engine
                        # so clients receive proper error responses instead
                        # of silent disconnect.
                        entry = self._engine_pool._entries.get(victim)
                        if entry and entry.engine is not None:
                            if hasattr(entry.engine, "abort_all_requests"):
                                aborted = await entry.engine.abort_all_requests()
                                if aborted > 0:
                                    logger.warning(
                                        f"Aborted {aborted} requests on "
                                        f"'{victim}' before eviction"
                                    )
                        logger.warning(
                            f"Evicting model '{victim}' (pressure={new_level})"
                        )
                        await self._engine_pool._unload_engine(victim)
                        continue

                    # Only one non-pinned model remains.
                    if new_level == "hard":
                        # Abort in-flight requests, keep model loaded —
                        # frees KV blocks so short-context follow-ups work.
                        entry = self._engine_pool._entries.get(victim)
                        if entry and entry.engine is not None:
                            if hasattr(entry.engine, "abort_all_requests"):
                                aborted = await entry.engine.abort_all_requests()
                                if aborted > 0:
                                    logger.warning(
                                        f"Aborted {aborted} requests on "
                                        f"'{victim}' due to hard memory "
                                        f"pressure (model kept loaded)"
                                    )
                    # soft: leave in-flight alone — admission pause already
                    # signaled, eviction can't help further without aborts.
                    break

                # No non-pinned victim. All loaded models are pinned.
                if new_level == "hard":
                    # Hard only: abort any in-progress model loads.
                    aborted_any = False
                    for entry in self._engine_pool._entries.values():
                        if entry.is_loading and not entry.abort_loading:
                            logger.warning(
                                f"Aborting in-progress load of "
                                f"'{entry.model_id}' (hard memory pressure)"
                            )
                            entry.abort_loading = True
                            aborted_any = True
                    if not aborted_any:
                        has_loaded = any(
                            e.engine is not None
                            for e in self._engine_pool._entries.values()
                        )
                        if has_loaded:
                            logger.warning(
                                "Hard memory pressure but all loaded models "
                                "are pinned and no loads in progress."
                            )
                        else:
                            logger.warning(
                                "Hard memory pressure but no models loaded."
                            )
                # soft + all pinned: nothing to do beyond admission pause.
                break

        # Re-evaluate level after eviction completes so admission state
        # reflects post-eviction reality on the next propagate.
        post_current = self._current_usage_bytes()
        if post_current < soft:
            post_level = "ok"
        elif post_current < hard:
            post_level = "soft"
        else:
            post_level = "hard"
        if post_level != self._pressure_level:
            self._pressure_level = post_level
            self._propagate_memory_limit()
            logger.info(
                f"Memory pressure post-eviction: {new_level} -> {post_level} "
                f"(current={_format_gb(post_current)})"
            )

        # Walk each scheduler's store-cache gate ±1 toward its
        # pressure-driven target every poll (#1383).
        self._walk_store_cache_caps()

    def get_status(self) -> dict:
        """Get enforcer status for monitoring endpoints.

        Reports the same `max(active, phys_footprint)` value the enforcer
        uses internally so admin UI / /health utilization matches the
        watermark the enforcer is actually comparing against.
        """
        current = self._current_usage_bytes() if self._running else 0
        return {
            "enabled": self._running,
            "max_bytes": self._max_bytes,
            "max_formatted": _format_gb(self._max_bytes),
            "soft_threshold": self._soft_threshold,
            "hard_threshold": self._hard_threshold,
            "soft_bytes": self._soft_bytes,
            "soft_formatted": _format_gb(self._soft_bytes),
            "hard_bytes": self._hard_bytes,
            "hard_formatted": _format_gb(self._hard_bytes),
            "current_bytes": current,
            "current_formatted": _format_gb(current),
            "pressure_level": self._pressure_level if self._running else "ok",
            "utilization": (
                current / self._max_bytes if self._max_bytes > 0 else 0.0
            ),
        }
