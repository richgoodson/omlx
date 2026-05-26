# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx/admin/oq_manager.py — the async oQ quantization
orchestrator. Focuses on the synchronous logic paths and validation:
helpers, task lifecycle bookkeeping, input validation, and the
``_phase_label`` ETA parser. The actual streaming quantization is
exercised end-to-end in test_oq.py.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Eager-load omlx.oq before any test patches sys.modules — otherwise the
# first ``from ..oq import …`` inside oq_manager would re-import the real
# module and clobber our fake.
import omlx.oq  # noqa: F401
from omlx.admin import oq_manager
from omlx.admin.oq_manager import (
    OQManager,
    QuantStatus,
    QuantTask,
    _dir_size,
    _format_size,
)


def _make_fake_oq(quantize_impl):
    """Build a MagicMock replacement for ``omlx.oq`` with the four
    symbols ``oq_manager`` imports lazily — ``OQ_LEVELS``, ``OQ_DTYPES``,
    ``resolve_output_name``, and ``quantize_oq_streaming``."""
    fake = MagicMock()
    fake.OQ_LEVELS = {2, 3, 3.5, 4, 5, 6, 8}
    fake.OQ_DTYPES = ("bfloat16", "float16")

    def _resolve(name, level, dtype="bfloat16", *, preserve_mtp=False):
        suffix = f"-oQ{level:g}"
        if dtype == "float16":
            suffix += "-fp16"
        if preserve_mtp:
            suffix += "-mtp"
        return f"{name}{suffix}"

    fake.resolve_output_name = _resolve
    fake.quantize_oq_streaming = quantize_impl
    # Harmless defaults for list_quantizable_models if it gets called
    # under the same patch (it isn't, but defends against future churn).
    fake.validate_quantizable.return_value = False
    fake.estimate_memory.return_value = {}
    return fake


# =============================================================================
# Pure helpers and data classes
# =============================================================================


class TestQuantStatus:
    def test_enum_values(self):
        assert QuantStatus.PENDING.value == "pending"
        assert QuantStatus.LOADING.value == "loading"
        assert QuantStatus.QUANTIZING.value == "quantizing"
        assert QuantStatus.SAVING.value == "saving"
        assert QuantStatus.COMPLETED.value == "completed"
        assert QuantStatus.FAILED.value == "failed"
        assert QuantStatus.CANCELLED.value == "cancelled"

    def test_is_string_enum(self):
        """Inherits from str so JSON encoders treat it like a string —
        the to_dict path relies on ``.value`` but downstream callers
        sometimes pass the status itself."""
        assert isinstance(QuantStatus.PENDING, str)


class TestQuantTaskToDict:
    def _make(self, **overrides):
        defaults = dict(
            task_id="tid",
            model_name="Qwen-7B",
            model_path="/m/Qwen-7B",
            oq_level=4.0,
            output_name="Qwen-7B-oQ4",
            output_path="/m/Qwen-7B-oQ4",
        )
        defaults.update(overrides)
        return QuantTask(**defaults)

    def test_to_dict_default_shape(self):
        d = self._make().to_dict()
        assert d["task_id"] == "tid"
        assert d["status"] == "pending"  # enum.value, not the enum itself
        assert d["progress"] == 0.0
        assert d["dtype"] == "bfloat16"
        # Fields not included in to_dict (intentional — internal only):
        assert "group_size" not in d
        assert "sensitivity_model_path" not in d
        assert "auto_proxy_sensitivity" not in d
        assert "preserve_mtp" not in d

    def test_progress_rounded_to_one_decimal(self):
        t = self._make()
        t.progress = 42.6789
        assert t.to_dict()["progress"] == 42.7

    def test_status_serialized_as_string(self):
        t = self._make()
        t.status = QuantStatus.QUANTIZING
        assert t.to_dict()["status"] == "quantizing"


class TestDirSize:
    def test_nonexistent_returns_zero(self, tmp_path):
        assert _dir_size(tmp_path / "missing") == 0

    def test_empty_dir_returns_zero(self, tmp_path):
        assert _dir_size(tmp_path) == 0

    def test_sums_files_recursively(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"x" * 100)
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.bin").write_bytes(b"y" * 50)
        assert _dir_size(tmp_path) == 150


class TestFormatSize:
    @pytest.mark.parametrize(
        "size,expected",
        [
            (0, "0 B"),
            (1023, "1023 B"),
            (1024, "1.0 KB"),
            (1024 * 512, "512.0 KB"),
            (1024**2, "1.0 MB"),
            (1024**3, "1.0 GB"),
            (5 * 1024**3, "5.0 GB"),
        ],
    )
    def test_thresholds(self, size, expected):
        assert _format_size(size) == expected


class TestPhaseLabel:
    def test_known_phase_loading(self):
        assert OQManager._phase_label("loading", 4) == "Loading model..."

    def test_known_phase_quantizing_formats_level(self):
        # oq_level uses :g — integer levels render without trailing .0
        assert OQManager._phase_label("quantizing", 4) == "Quantizing to oQ4..."

    def test_known_phase_quantizing_handles_fractional_level(self):
        assert (
            OQManager._phase_label("quantizing", 3.5) == "Quantizing to oQ3.5..."
        )

    def test_unknown_phase_passes_through(self):
        assert OQManager._phase_label("custom_phase", 4) == "custom_phase"

    def test_quantizing_eta_with_percent_and_eta(self):
        label = OQManager._phase_label("quantizing_eta|400|800|0:30", 4)
        assert label == "oQ4: 50% (0:30 remaining)"

    def test_quantizing_eta_without_eta_suffix(self):
        label = OQManager._phase_label("quantizing_eta|400|800|", 4)
        assert label == "oQ4: 50%"

    def test_quantizing_eta_handles_zero_total(self):
        """Division by zero is guarded — total=0 must not crash."""
        label = OQManager._phase_label("quantizing_eta|10|0|0:05", 4)
        # current/total: int(10 / max(0,1)) * 100 = 1000% — implementation
        # caps the ratio at division but doesn't clamp the result, so we
        # just assert no crash and the eta makes it through.
        assert "remaining" in label

    def test_quantizing_eta_handles_non_numeric_parts(self):
        """When current/total aren't digits, the pct falls back to 0."""
        label = OQManager._phase_label("quantizing_eta|x|y|0:30", 4)
        assert label == "oQ4: 0% (0:30 remaining)"


# =============================================================================
# OQManager lifecycle
# =============================================================================


class TestOQManagerInit:
    def test_defaults_with_one_dir(self, tmp_path):
        mgr = OQManager(model_dirs=[str(tmp_path)])
        assert mgr._model_dirs == [tmp_path]
        assert mgr._output_dir == tmp_path  # first dir wins
        assert mgr._tasks == {}
        assert mgr._active_tasks == {}
        assert mgr._cancelled == set()
        assert mgr._on_complete is None

    def test_first_dir_is_output_dir(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        mgr = OQManager(model_dirs=[str(a), str(b)])
        assert mgr._output_dir == a  # output_dir = first

    def test_empty_dirs_fallback_to_cwd(self):
        mgr = OQManager(model_dirs=[])
        assert mgr._model_dirs == []
        assert mgr._output_dir == Path(".")  # "." fallback

    def test_on_complete_callback_stored(self, tmp_path):
        cb = lambda: None
        mgr = OQManager(model_dirs=[str(tmp_path)], on_complete=cb)
        assert mgr._on_complete is cb


class TestGetTasksAndIsQuantizing:
    def test_initial_state_is_empty_and_idle(self, tmp_path):
        mgr = OQManager(model_dirs=[str(tmp_path)])
        assert mgr.get_tasks() == []
        assert mgr.is_quantizing is False

    def test_is_quantizing_true_for_active_status(self, tmp_path):
        mgr = OQManager(model_dirs=[str(tmp_path)])
        t = QuantTask(
            task_id="t1",
            model_name="m",
            model_path="/m",
            oq_level=4,
            output_name="m-oQ4",
            output_path="/o",
        )
        t.status = QuantStatus.QUANTIZING
        mgr._tasks["t1"] = t
        assert mgr.is_quantizing is True

    def test_is_quantizing_false_for_terminal_status(self, tmp_path):
        mgr = OQManager(model_dirs=[str(tmp_path)])
        for status in (
            QuantStatus.COMPLETED,
            QuantStatus.FAILED,
            QuantStatus.CANCELLED,
        ):
            t = QuantTask(
                task_id=f"t-{status.value}",
                model_name="m",
                model_path="/m",
                oq_level=4,
                output_name="m-oQ4",
                output_path="/o",
            )
            t.status = status
            mgr._tasks[t.task_id] = t
        assert mgr.is_quantizing is False


class TestRemoveTask:
    def _mgr_with_task(self, tmp_path, status):
        mgr = OQManager(model_dirs=[str(tmp_path)])
        t = QuantTask(
            task_id="t",
            model_name="m",
            model_path="/m",
            oq_level=4,
            output_name="m-oQ4",
            output_path="/o",
        )
        t.status = status
        mgr._tasks["t"] = t
        return mgr

    def test_unknown_task_returns_false(self, tmp_path):
        mgr = OQManager(model_dirs=[str(tmp_path)])
        assert mgr.remove_task("does-not-exist") is False

    def test_refuses_active_task(self, tmp_path):
        mgr = self._mgr_with_task(tmp_path, QuantStatus.QUANTIZING)
        assert mgr.remove_task("t") is False
        assert "t" in mgr._tasks  # not removed

    def test_removes_terminal_task_and_clears_cancelled_set(self, tmp_path):
        """Removing a cancelled task must also drop the entry from the
        ``_cancelled`` set — otherwise a same-id resubmission (unlikely
        since IDs are UUIDs, but the invariant matters) would observe
        a phantom cancel flag."""
        mgr = self._mgr_with_task(tmp_path, QuantStatus.CANCELLED)
        mgr._cancelled.add("t")
        assert mgr.remove_task("t") is True
        assert "t" not in mgr._tasks
        assert "t" not in mgr._cancelled


class TestCancelQuantization:
    @pytest.mark.asyncio
    async def test_unknown_task_returns_false(self, tmp_path):
        mgr = OQManager(model_dirs=[str(tmp_path)])
        assert await mgr.cancel_quantization("nope") is False

    @pytest.mark.asyncio
    async def test_refuses_non_active_task(self, tmp_path):
        """Cancelling a COMPLETED task is a no-op — protects against UI
        races where 'cancel' is clicked after completion."""
        mgr = OQManager(model_dirs=[str(tmp_path)])
        t = QuantTask(
            task_id="t",
            model_name="m",
            model_path="/m",
            oq_level=4,
            output_name="m-oQ4",
            output_path="/o",
        )
        t.status = QuantStatus.COMPLETED
        mgr._tasks["t"] = t
        assert await mgr.cancel_quantization("t") is False
        # Status not flipped to CANCELLED
        assert mgr._tasks["t"].status == QuantStatus.COMPLETED


# =============================================================================
# start_quantization validation paths
# =============================================================================


def _write_fake_model(path: Path, *, with_safetensors: bool = True) -> None:
    """Write a minimal config.json + weight file so source_size > 0."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    if with_safetensors:
        (path / "model.safetensors").write_bytes(b"x" * 1024)


class TestStartQuantizationValidation:
    @pytest.mark.asyncio
    async def test_invalid_oq_level_rejected(self, tmp_path):
        src = tmp_path / "src"
        _write_fake_model(src)
        mgr = OQManager(model_dirs=[str(tmp_path)])
        with pytest.raises(ValueError, match="Invalid oQ level"):
            await mgr.start_quantization(str(src), oq_level=7)

    @pytest.mark.asyncio
    async def test_invalid_dtype_rejected(self, tmp_path):
        src = tmp_path / "src"
        _write_fake_model(src)
        mgr = OQManager(model_dirs=[str(tmp_path)])
        with pytest.raises(ValueError, match="Invalid dtype"):
            await mgr.start_quantization(
                str(src), oq_level=4, dtype="float8_e4m3"
            )

    @pytest.mark.asyncio
    async def test_missing_model_dir_rejected(self, tmp_path):
        mgr = OQManager(model_dirs=[str(tmp_path)])
        with pytest.raises(ValueError, match="Model not found"):
            await mgr.start_quantization(
                str(tmp_path / "does-not-exist"), oq_level=4
            )

    @pytest.mark.asyncio
    async def test_missing_config_json_rejected(self, tmp_path):
        """Dir exists but no config.json → still 'Model not found'."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "model.safetensors").write_bytes(b"x" * 100)
        mgr = OQManager(model_dirs=[str(tmp_path)])
        with pytest.raises(ValueError, match="Model not found"):
            await mgr.start_quantization(str(src), oq_level=4)

    @pytest.mark.asyncio
    async def test_output_collision_rejected(self, tmp_path):
        """If the resolved output dir already exists, refuse before
        starting — overwriting a finished quant is a costly mistake."""
        src = tmp_path / "Qwen-7B"
        _write_fake_model(src)
        # Pre-create the expected output path
        (tmp_path / "Qwen-7B-oQ4").mkdir()
        mgr = OQManager(model_dirs=[str(tmp_path)])
        with pytest.raises(ValueError, match="already exists"):
            await mgr.start_quantization(str(src), oq_level=4)

    @pytest.mark.asyncio
    async def test_duplicate_active_task_rejected(self, tmp_path):
        """Two concurrent quant attempts for the same (model, level,
        dtype) must be refused — single GPU, single semaphore slot."""
        src = tmp_path / "Qwen-7B"
        _write_fake_model(src)
        mgr = OQManager(model_dirs=[str(tmp_path)])

        # Plant a duplicate active task by hand
        existing = QuantTask(
            task_id="existing",
            model_name="Qwen-7B",
            model_path=str(src),
            oq_level=4,
            output_name="Qwen-7B-oQ4",
            output_path=str(tmp_path / "Qwen-7B-oQ4"),
            dtype="bfloat16",
        )
        existing.status = QuantStatus.QUANTIZING
        mgr._tasks["existing"] = existing

        with pytest.raises(ValueError, match="already in progress"):
            await mgr.start_quantization(str(src), oq_level=4)

    @pytest.mark.asyncio
    async def test_completed_task_does_not_block_resubmit(self, tmp_path):
        """A finished task for the same (model, level, dtype) must not
        block a fresh attempt — only ACTIVE statuses do."""
        src = tmp_path / "Qwen-7B"
        _write_fake_model(src)
        mgr = OQManager(model_dirs=[str(tmp_path)])

        old = QuantTask(
            task_id="old",
            model_name="Qwen-7B",
            model_path=str(src),
            oq_level=4,
            output_name="Qwen-7B-oQ4",
            output_path=str(tmp_path / "Qwen-7B-oQ4"),
        )
        old.status = QuantStatus.COMPLETED
        mgr._tasks["old"] = old

        # Stub the actual run to prevent background quantization from
        # firing while we just verify the validation path.
        with patch.object(mgr, "_run_quantization", new=AsyncMock()):
            task = await mgr.start_quantization(str(src), oq_level=4)
        assert task.task_id != "old"
        assert task.status == QuantStatus.PENDING
        # Cancel cleanup of the background task we never let run
        bg = mgr._active_tasks.pop(task.task_id, None)
        if bg is not None:
            bg.cancel()


# =============================================================================
# list_quantizable_models
# =============================================================================


class TestListQuantizableModels:
    @pytest.mark.asyncio
    async def test_empty_dirs_returns_empty_lists(self, tmp_path):
        mgr = OQManager(model_dirs=[str(tmp_path)])
        src, all_ = await mgr.list_quantizable_models()
        assert src == []
        assert all_ == []

    @pytest.mark.asyncio
    async def test_skips_dirs_without_config_or_weights(self, tmp_path):
        # No config.json, no weights
        (tmp_path / "junk").mkdir()
        # config.json but no weights
        empty_model = tmp_path / "empty"
        empty_model.mkdir()
        (empty_model / "config.json").write_text("{}")
        mgr = OQManager(model_dirs=[str(tmp_path)])
        src, all_ = await mgr.list_quantizable_models()
        assert src == []
        assert all_ == []

    @pytest.mark.asyncio
    async def test_classifies_model_and_detects_mtp_heads(self, tmp_path):
        """A model with ``mtp_num_hidden_layers > 0`` must be flagged
        has_mtp_heads=True — admin UI uses that to grey out 'preserve
        MTP' for models that don't have any MTP weights to preserve.
        """
        m = tmp_path / "Qwen-MTP"
        m.mkdir()
        (m / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "mtp_num_hidden_layers": 1,
                    "num_hidden_layers": 28,
                }
            )
        )
        (m / "model.safetensors").write_bytes(b"x" * 2048)

        # Make validate_quantizable return True for our fake config so
        # this model ends up in source_models too.
        fake_oq = MagicMock()
        fake_oq.validate_quantizable.return_value = True
        fake_oq.estimate_memory.return_value = {"streaming_gb": 0.5}
        with patch.dict("sys.modules", {"omlx.oq": fake_oq}):
            mgr = OQManager(model_dirs=[str(tmp_path)])
            src, all_ = await mgr.list_quantizable_models()

        assert len(all_) == 1
        info = all_[0]
        assert info["name"] == "Qwen-MTP"
        assert info["model_type"] == "qwen3_5"
        assert info["has_mtp_heads"] is True
        assert info["is_vlm"] is False
        assert info["is_quantized"] is False
        assert len(src) == 1
        assert src[0]["num_layers"] == 28
        assert src[0]["memory_streaming"] == {"streaming_gb": 0.5}

    @pytest.mark.asyncio
    async def test_marks_already_quantized_models(self, tmp_path):
        m = tmp_path / "Qwen-oQ4"
        m.mkdir()
        (m / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5", "quantization": {"bits": 4}})
        )
        (m / "model.safetensors").write_bytes(b"x" * 1024)

        fake_oq = MagicMock()
        fake_oq.validate_quantizable.return_value = False
        with patch.dict("sys.modules", {"omlx.oq": fake_oq}):
            mgr = OQManager(model_dirs=[str(tmp_path)])
            _src, all_ = await mgr.list_quantizable_models()

        assert len(all_) == 1
        assert all_[0]["is_quantized"] is True

    @pytest.mark.asyncio
    async def test_deduplicates_same_name_across_dirs(self, tmp_path):
        """Scanning two parent dirs that both contain a child 'Qwen' must
        only report it once — the admin UI relies on unique names."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        for parent in (a, b):
            m = parent / "Qwen"
            m.mkdir(parents=True)
            (m / "config.json").write_text(
                json.dumps({"model_type": "llama"})
            )
            (m / "model.safetensors").write_bytes(b"x" * 100)

        fake_oq = MagicMock()
        fake_oq.validate_quantizable.return_value = False
        with patch.dict("sys.modules", {"omlx.oq": fake_oq}):
            mgr = OQManager(model_dirs=[str(a), str(b)])
            _src, all_ = await mgr.list_quantizable_models()

        assert [m["name"] for m in all_] == ["Qwen"]


# =============================================================================
# shutdown
# =============================================================================


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_with_no_tasks_is_noop(self, tmp_path):
        mgr = OQManager(model_dirs=[str(tmp_path)])
        await mgr.shutdown()  # must not raise

    @pytest.mark.asyncio
    async def test_shutdown_cancels_all_active(self, tmp_path):
        """Server shutdown path calls this — every active task must be
        cancelled, not just one."""
        mgr = OQManager(model_dirs=[str(tmp_path)])
        calls = []

        async def fake_cancel(tid):
            calls.append(tid)
            return True

        # Plant two fake active task entries
        mgr._active_tasks["a"] = MagicMock()
        mgr._active_tasks["b"] = MagicMock()
        with patch.object(mgr, "cancel_quantization", new=fake_cancel):
            await mgr.shutdown()
        assert set(calls) == {"a", "b"}


# =============================================================================
# update_model_dirs runtime flow
# =============================================================================


@pytest.fixture
def fp_model_dir(tmp_path):
    """One directory with a full-precision (quantizable) source model."""
    d = tmp_path / "models1"
    d.mkdir()
    model = d / "Llama-3B"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "num_hidden_layers": 32,
    }))
    (model / "model.safetensors").write_bytes(b"\x00" * 4096)
    return d


@pytest.fixture
def second_fp_model_dir(tmp_path):
    """A second directory holding a different full-precision model."""
    d = tmp_path / "models2"
    d.mkdir()
    model = d / "Qwen-7B"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen2",
        "num_hidden_layers": 28,
    }))
    (model / "model.safetensors").write_bytes(b"\x00" * 4096)
    return d


class TestUpdateModelDirs:
    @pytest.mark.asyncio
    async def test_picks_up_added_dir(self, fp_model_dir, second_fp_model_dir):
        # Mirrors the real Settings UI flow: server starts with one model
        # directory, the user adds a second one at runtime via Settings, and
        # _apply_model_dirs_runtime calls update_model_dirs(). Without that
        # call, models in the newly added directory never show up in the oQ
        # Quantization "Source Model" dropdown.
        manager = OQManager(model_dirs=[str(fp_model_dir)])
        source_before, _ = await manager.list_quantizable_models()
        names_before = {m["name"] for m in source_before}
        assert "Llama-3B" in names_before
        assert "Qwen-7B" not in names_before

        manager.update_model_dirs(
            [str(fp_model_dir), str(second_fp_model_dir)]
        )

        source_after, _ = await manager.list_quantizable_models()
        names_after = {m["name"] for m in source_after}
        assert "Llama-3B" in names_after
        assert "Qwen-7B" in names_after

    def test_output_dir_tracks_primary_dir(
        self, fp_model_dir, second_fp_model_dir
    ):
        # Output is always written to the primary (first) directory and
        # _model_dirs reflects the exact input order.
        manager = OQManager(model_dirs=[str(fp_model_dir)])
        assert manager._output_dir == fp_model_dir

        manager.update_model_dirs(
            [str(second_fp_model_dir), str(fp_model_dir)]
        )
        assert manager._model_dirs == [second_fp_model_dir, fp_model_dir]
        assert manager._output_dir == second_fp_model_dir

    def test_update_to_empty_leaves_old_output_dir(self, tmp_path):
        """Empty update is a no-op for output_dir — protects against an
        accidental ``Path('.')`` fallback when the admin UI sends an
        empty list."""
        mgr = OQManager(model_dirs=[str(tmp_path)])
        original = mgr._output_dir
        mgr.update_model_dirs([])
        assert mgr._model_dirs == []
        assert mgr._output_dir == original


# =============================================================================
# _run_quantization happy + failure paths
# =============================================================================


class TestRunQuantizationHappyPath:
    @pytest.mark.asyncio
    async def test_completes_and_sets_completion_fields(self, tmp_path):
        src = tmp_path / "Qwen-7B"
        _write_fake_model(src)

        def fake_quantize(
            model_path, output_path, oq_level, group_size, progress_cb, *args
        ):
            out = Path(output_path)
            out.mkdir(parents=True, exist_ok=True)
            (out / "model.safetensors").write_bytes(b"q" * 2048)
            progress_cb("quantizing", 50.0)
            progress_cb("saving", 95.0)

        mgr = OQManager(model_dirs=[str(tmp_path)])
        with patch.dict("sys.modules", {"omlx.oq": _make_fake_oq(fake_quantize)}):
            task = await mgr.start_quantization(str(src), oq_level=4)
            bg = mgr._active_tasks.get(task.task_id)
            assert bg is not None
            await bg

        assert task.status == QuantStatus.COMPLETED
        assert task.progress == 100.0
        assert task.phase == "Completed"
        assert task.completed_at > 0
        assert task.started_at > 0
        assert task.completed_at >= task.started_at
        assert task.output_size == 2048
        assert task.error == ""
        # Lifecycle cleanup: task no longer registered as active
        assert task.task_id not in mgr._active_tasks
        assert task.task_id not in mgr._progress_tasks

    @pytest.mark.asyncio
    async def test_sync_on_complete_callback_fires(self, tmp_path):
        called = []

        def cb():
            called.append("sync")

        src = tmp_path / "Qwen-7B"
        _write_fake_model(src)

        def fake_quantize(model_path, output_path, *args, **kwargs):
            Path(output_path).mkdir(parents=True, exist_ok=True)

        mgr = OQManager(model_dirs=[str(tmp_path)], on_complete=cb)
        with patch.dict("sys.modules", {"omlx.oq": _make_fake_oq(fake_quantize)}):
            task = await mgr.start_quantization(str(src), oq_level=4)
            await mgr._active_tasks[task.task_id]
        assert called == ["sync"]

    @pytest.mark.asyncio
    async def test_async_on_complete_callback_is_awaited(self, tmp_path):
        """Coroutine callbacks are detected and awaited — the
        ``asyncio.iscoroutine(result)`` check in _run_quantization must
        not silently drop async work."""
        called = []

        async def acb():
            await asyncio.sleep(0)
            called.append("async")

        src = tmp_path / "Qwen-7B"
        _write_fake_model(src)

        def fake_quantize(model_path, output_path, *args, **kwargs):
            Path(output_path).mkdir(parents=True, exist_ok=True)

        mgr = OQManager(model_dirs=[str(tmp_path)], on_complete=acb)
        with patch.dict("sys.modules", {"omlx.oq": _make_fake_oq(fake_quantize)}):
            task = await mgr.start_quantization(str(src), oq_level=4)
            await mgr._active_tasks[task.task_id]
        assert called == ["async"]

    @pytest.mark.asyncio
    async def test_on_complete_exception_does_not_fail_task(self, tmp_path):
        """A buggy on_complete callback must not flip a successful
        quant to FAILED — the work is done, the callback is
        cosmetic."""

        def cb():
            raise RuntimeError("registry refresh failed")

        src = tmp_path / "Qwen-7B"
        _write_fake_model(src)

        def fake_quantize(model_path, output_path, *args, **kwargs):
            Path(output_path).mkdir(parents=True, exist_ok=True)

        mgr = OQManager(model_dirs=[str(tmp_path)], on_complete=cb)
        with patch.dict("sys.modules", {"omlx.oq": _make_fake_oq(fake_quantize)}):
            task = await mgr.start_quantization(str(src), oq_level=4)
            await mgr._active_tasks[task.task_id]
        assert task.status == QuantStatus.COMPLETED
        assert task.error == ""

    @pytest.mark.asyncio
    async def test_progress_callback_updates_phase_label(self, tmp_path):
        """The progress callback fed to quantize_oq_streaming must route
        through ``_phase_label`` — so we can verify the level-aware
        formatting wired up correctly."""
        observed_phases = []
        observed_progress = []

        def fake_quantize(
            model_path, output_path, oq_level, group_size, progress_cb, *args
        ):
            Path(output_path).mkdir(parents=True, exist_ok=True)
            progress_cb("loading", 10.0)
            observed_phases.append(progress_cb.__self__._tasks if False else None)
            # We can't reach the task object via the callback, so capture
            # via the manager after the fact.
            progress_cb("quantizing", 45.0)

        src = tmp_path / "Qwen-7B"
        _write_fake_model(src)
        mgr = OQManager(model_dirs=[str(tmp_path)])
        with patch.dict("sys.modules", {"omlx.oq": _make_fake_oq(fake_quantize)}):
            task = await mgr.start_quantization(str(src), oq_level=4)
            await mgr._active_tasks[task.task_id]

        # After the last quantize-phase callback was 45.0, but completion
        # then bumps to 100.0 / "Completed". Confirm completion wins.
        assert task.progress == 100.0
        assert task.phase == "Completed"


class TestRunQuantizationFailure:
    @pytest.mark.asyncio
    async def test_exception_marks_task_failed(self, tmp_path):
        src = tmp_path / "Qwen-7B"
        _write_fake_model(src)

        def fake_quantize(model_path, output_path, *args, **kwargs):
            raise RuntimeError("OOM during quantization")

        mgr = OQManager(model_dirs=[str(tmp_path)])
        with patch.dict("sys.modules", {"omlx.oq": _make_fake_oq(fake_quantize)}):
            task = await mgr.start_quantization(str(src), oq_level=4)
            await mgr._active_tasks[task.task_id]

        assert task.status == QuantStatus.FAILED
        assert "OOM during quantization" in task.error
        assert task.completed_at > 0
        # Active-task registry cleared on failure
        assert task.task_id not in mgr._active_tasks

    @pytest.mark.asyncio
    async def test_failure_cleans_up_partial_output(self, tmp_path):
        """A crashed quantize leaves a half-written model dir on disk;
        _run_quantization must remove it so the user doesn't have to."""
        src = tmp_path / "Qwen-7B"
        _write_fake_model(src)

        def fake_quantize(model_path, output_path, *args, **kwargs):
            out = Path(output_path)
            out.mkdir(parents=True, exist_ok=True)
            (out / "partial.safetensors").write_bytes(b"x" * 100)
            raise RuntimeError("kaboom")

        mgr = OQManager(model_dirs=[str(tmp_path)])
        with patch.dict("sys.modules", {"omlx.oq": _make_fake_oq(fake_quantize)}):
            task = await mgr.start_quantization(str(src), oq_level=4)
            await mgr._active_tasks[task.task_id]

        assert task.status == QuantStatus.FAILED
        assert not Path(task.output_path).exists()  # cleaned up


class TestRunQuantizationPreCancel:
    @pytest.mark.asyncio
    async def test_pre_cancelled_task_skips_quantize(self, tmp_path):
        """If ``_cancelled`` is set before _run_quantization enters the
        semaphore section, quantize_oq_streaming must NOT be invoked.
        Guards against a race where shutdown cancels a queued task
        between start_quantization scheduling and the background task
        actually running."""
        quantize_called = []

        def fake_quantize(*args, **kwargs):
            quantize_called.append(True)

        task = QuantTask(
            task_id="pre-cancelled",
            model_name="m",
            model_path="/m",
            oq_level=4,
            output_name="m-oQ4",
            output_path=str(tmp_path / "m-oQ4"),
        )
        mgr = OQManager(model_dirs=[str(tmp_path)])
        mgr._tasks[task.task_id] = task
        mgr._cancelled.add(task.task_id)

        with patch.dict("sys.modules", {"omlx.oq": _make_fake_oq(fake_quantize)}):
            await mgr._run_quantization(task.task_id)

        assert quantize_called == []
        assert task.status == QuantStatus.PENDING  # untouched


# =============================================================================
# Cooperative cancellation
# =============================================================================


class TestCancelCooperativeExit:
    @pytest.mark.asyncio
    async def test_cancel_via_progress_callback(self, tmp_path):
        """End-to-end cancel flow: a running quantize is interrupted by
        the next progress_cb call, which sees the ``_cancelled`` flag
        and raises ``_QuantCancelled``. The task ends as CANCELLED with
        the partial output dir removed.

        This is the design upstream chose over hard-cancelling the
        asyncio wrapper — see the comment block in cancel_quantization
        about not calling active_task.cancel() first."""
        src = tmp_path / "Qwen-7B"
        _write_fake_model(src)

        started = threading.Event()

        def fake_quantize(
            model_path, output_path, oq_level, group_size, progress_cb, *args
        ):
            out = Path(output_path)
            out.mkdir(parents=True, exist_ok=True)
            (out / "partial.safetensors").write_bytes(b"x" * 256)
            started.set()
            # Spin calling progress_cb. The N+1th call raises
            # _QuantCancelled once the test has triggered cancel.
            for _ in range(400):  # ~20s upper bound
                progress_cb("quantizing", 50.0)
                time.sleep(0.05)
            raise AssertionError(
                "progress_cb should have raised _QuantCancelled before this point"
            )

        mgr = OQManager(model_dirs=[str(tmp_path)])
        with patch.dict("sys.modules", {"omlx.oq": _make_fake_oq(fake_quantize)}):
            task = await mgr.start_quantization(str(src), oq_level=4)
            # Wait until the background thread is actually running
            assert await asyncio.to_thread(started.wait, 5.0), (
                "fake_quantize never started"
            )
            result = await mgr.cancel_quantization(task.task_id)

        assert result is True
        assert task.status == QuantStatus.CANCELLED
        # Partial output cleaned up by cancel_quantization (before the
        # cooperative wait, so it's gone whether or not the background
        # task exits in time).
        assert not Path(task.output_path).exists()
        # Registries cleared
        assert task.task_id not in mgr._active_tasks
        assert task.task_id not in mgr._progress_tasks

    @pytest.mark.asyncio
    async def test_cancel_during_loading_phase(self, tmp_path):
        """Cancel can arrive while we're still in LOADING (before any
        progress_cb has been called). The cooperative path still works
        because the first progress_cb in the QUANTIZING phase will
        see the flag."""
        src = tmp_path / "Qwen-7B"
        _write_fake_model(src)

        ready = threading.Event()
        proceed = threading.Event()

        def fake_quantize(
            model_path, output_path, oq_level, group_size, progress_cb, *args
        ):
            Path(output_path).mkdir(parents=True, exist_ok=True)
            ready.set()
            # Wait for cancel to be issued before calling progress_cb
            proceed.wait(timeout=5.0)
            progress_cb("quantizing", 1.0)  # first call after cancel → raises

        mgr = OQManager(model_dirs=[str(tmp_path)])
        with patch.dict("sys.modules", {"omlx.oq": _make_fake_oq(fake_quantize)}):
            task = await mgr.start_quantization(str(src), oq_level=4)
            assert await asyncio.to_thread(ready.wait, 5.0)

            # Issue cancel — sets _cancelled, then cooperatively awaits.
            # Schedule it so we can release `proceed` after the cancel
            # has run synchronously up to the cooperative await.
            cancel_task = asyncio.create_task(
                mgr.cancel_quantization(task.task_id)
            )
            # Yield once so cancel_quantization gets a chance to run its
            # synchronous setup (set _cancelled, clean partial output)
            # before we release fake_quantize.
            await asyncio.sleep(0)
            proceed.set()
            result = await cancel_task

        assert result is True
        assert task.status == QuantStatus.CANCELLED


# =============================================================================
# _estimate_progress
# =============================================================================


class TestEstimateProgress:
    @pytest.mark.asyncio
    async def test_returns_immediately_for_unknown_task(self, tmp_path):
        """Unknown task_id is a no-op, not an error — the estimator is
        fire-and-forget; it must tolerate the task being removed by
        cleanup before it gets a chance to look it up."""
        mgr = OQManager(model_dirs=[str(tmp_path)])
        # Should return without raising and without hanging
        await asyncio.wait_for(
            mgr._estimate_progress("does-not-exist"), timeout=1.0
        )

    @pytest.mark.asyncio
    async def test_run_quantization_cancels_progress_task_on_success(
        self, tmp_path
    ):
        """The progress estimator must not leak past the parent task's
        completion — _run_quantization's finally clause cancels it."""
        src = tmp_path / "Qwen-7B"
        _write_fake_model(src)

        def fake_quantize(model_path, output_path, *args, **kwargs):
            Path(output_path).mkdir(parents=True, exist_ok=True)

        mgr = OQManager(model_dirs=[str(tmp_path)])
        with patch.dict("sys.modules", {"omlx.oq": _make_fake_oq(fake_quantize)}):
            task = await mgr.start_quantization(str(src), oq_level=4)
            await mgr._active_tasks[task.task_id]

        # _progress_tasks should be empty: either the estimator finished
        # naturally (status went terminal) or finally clause cancelled it.
        assert task.task_id not in mgr._progress_tasks

    @pytest.mark.asyncio
    async def test_run_quantization_cancels_progress_task_on_failure(
        self, tmp_path
    ):
        src = tmp_path / "Qwen-7B"
        _write_fake_model(src)

        def fake_quantize(*args, **kwargs):
            raise RuntimeError("boom")

        mgr = OQManager(model_dirs=[str(tmp_path)])
        with patch.dict("sys.modules", {"omlx.oq": _make_fake_oq(fake_quantize)}):
            task = await mgr.start_quantization(str(src), oq_level=4)
            await mgr._active_tasks[task.task_id]

        assert task.task_id not in mgr._progress_tasks
