# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx/admin/oq_manager.py — the async oQ quantization
orchestrator. Focuses on the synchronous logic paths and validation:
helpers, task lifecycle bookkeeping, input validation, and the
``_phase_label`` ETA parser. The actual streaming quantization is
exercised end-to-end in test_oq.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.admin import oq_manager
from omlx.admin.oq_manager import (
    OQManager,
    QuantStatus,
    QuantTask,
    _dir_size,
    _format_size,
)


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


class TestUpdateModelDirs:
    def test_changes_output_dir_to_new_first(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        mgr = OQManager(model_dirs=[str(a)])
        assert mgr._output_dir == a

        mgr.update_model_dirs([str(b), str(a)])
        assert mgr._model_dirs == [b, a]
        assert mgr._output_dir == b  # output_dir tracks the head

    def test_update_to_empty_leaves_old_output_dir(self, tmp_path):
        """Empty update is a no-op for output_dir — protects against an
        accidental ``Path('.')`` fallback when the admin UI sends an
        empty list."""
        mgr = OQManager(model_dirs=[str(tmp_path)])
        original = mgr._output_dir
        mgr.update_model_dirs([])
        assert mgr._model_dirs == []
        assert mgr._output_dir == original


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


class TestOQManagerUpdateModelDirs:
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
        # Output is always written to the primary (first) directory.
        manager = OQManager(model_dirs=[str(fp_model_dir)])
        assert manager._output_dir == fp_model_dir

        manager.update_model_dirs(
            [str(second_fp_model_dir), str(fp_model_dir)]
        )
        assert manager._output_dir == second_fp_model_dir
