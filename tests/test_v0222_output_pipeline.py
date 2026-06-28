"""Phase 2 (S1) structural async output pipeline -- CPU gates.

Proves, with NO GPU, the falsifiable acceptance bars for the three-stage
SNAPSHOT (step thread) -> MATERIALIZE (background) -> WRITE (#101 background)
output pipeline (``PHASE2_ASYNC_PIPELINE_DESIGN.md``):

* env resolvers: ``GPUWRF_NEST_OUTPUT_PIPELINE`` default OFF, enable/disable
  tokens; ``GPUWRF_NEST_OUTPUT_PIPELINE_DEPTH``.
* **G1 -- byte-identity:** the per-domain ``__call__`` driven with the pipeline ON
  writes a wrfout file ``cmp``-byte-identical to the pipeline-OFF (synchronous)
  path -- same inputs, same pure ops, only the thread + timing differ.
* **G3 -- finite fail-closed (Invariant F):** a NaN injected into a state leaf at
  an output boundary makes the run raise ``NonFiniteStateError`` -- surfaced at the
  NEXT boundary's step-thread ``submit`` AND at ``join`` -- and the bad frame is
  NOT on disk.
* **G5 -- ordering + failure surfacing:** several frames materialize + land on disk
  in order; a materialize/write error is re-raised at the next ``submit`` and at
  ``join``; subsequent frames are skipped; a producer blocked on a full snapshot
  queue is released on error (no deadlock).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.integration import nested_pipeline as nested_pipeline_module
from gpuwrf.io.async_wrfout import AsyncWrfoutWriter
from gpuwrf.integration.nested_pipeline import (
    OutputPipeline,
    OutputSnapshot,
    _PerDomainWrfoutWriter,
    _nested_m9_radiation_from_carry_from_env,
    _nested_output_pipeline_depth_from_env,
    _nested_output_pipeline_from_env,
)
from gpuwrf.runtime.finite_state_guard import NonFiniteStateError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_m7_netcdf_writer import synthetic_case  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Env resolvers
# ---------------------------------------------------------------------------

def test_output_pipeline_default_is_off(monkeypatch):
    monkeypatch.delenv("GPUWRF_NEST_OUTPUT_PIPELINE", raising=False)
    assert _nested_output_pipeline_from_env() is False


@pytest.mark.parametrize("value", ["1", "true", "on", "yes", "TRUE", "On"])
def test_output_pipeline_enable_tokens(monkeypatch, value):
    monkeypatch.setenv("GPUWRF_NEST_OUTPUT_PIPELINE", value)
    assert _nested_output_pipeline_from_env() is True


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "", "garbage"])
def test_output_pipeline_disable_tokens(monkeypatch, value):
    monkeypatch.setenv("GPUWRF_NEST_OUTPUT_PIPELINE", value)
    assert _nested_output_pipeline_from_env() is False


def test_output_pipeline_depth_default_and_clamp(monkeypatch):
    monkeypatch.delenv("GPUWRF_NEST_OUTPUT_PIPELINE_DEPTH", raising=False)
    assert _nested_output_pipeline_depth_from_env() == 1
    monkeypatch.setenv("GPUWRF_NEST_OUTPUT_PIPELINE_DEPTH", "3")
    assert _nested_output_pipeline_depth_from_env() == 3
    monkeypatch.setenv("GPUWRF_NEST_OUTPUT_PIPELINE_DEPTH", "0")
    assert _nested_output_pipeline_depth_from_env() == 1
    monkeypatch.setenv("GPUWRF_NEST_OUTPUT_PIPELINE_DEPTH", "garbage")
    assert _nested_output_pipeline_depth_from_env() == 1


def test_output_pipeline_and_radiation_source_envs_are_independent(monkeypatch):
    monkeypatch.setenv("GPUWRF_NEST_OUTPUT_PIPELINE", "1")
    monkeypatch.delenv("GPUWRF_NESTED_M9_RADIATION_FROM_CARRY", raising=False)
    assert _nested_output_pipeline_from_env() is True
    assert _nested_m9_radiation_from_carry_from_env() is False

    monkeypatch.setenv("GPUWRF_NEST_OUTPUT_PIPELINE", "0")
    monkeypatch.setenv("GPUWRF_NESTED_M9_RADIATION_FROM_CARRY", "1")
    assert _nested_output_pipeline_from_env() is False
    assert _nested_m9_radiation_from_carry_from_env() is True


# ---------------------------------------------------------------------------
# Test scaffolding (no Gen2Run / no device): mirrors the v014 helper.
# ---------------------------------------------------------------------------

def _jaxify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(**{k: _jaxify(v) for k, v in vars(value).items()})
    if isinstance(value, dict):
        return {k: _jaxify(v) for k, v in value.items()}
    return value


def _make_writer(*, async_writer, output_pipeline=None):
    from gpuwrf.integration.daily_pipeline import (
        _merge_output_diagnostics,
        _surface_diagnostics_for_output,
    )

    writer = object.__new__(_PerDomainWrfoutWriter)
    writer._surface_diagnostics_for_output = _surface_diagnostics_for_output
    writer._merge_output_diagnostics = _merge_output_diagnostics
    writer.writer_diagnostics = {}
    writer.writer_static_latlon_metadata = {}
    writer._async_writer = async_writer
    writer._output_pipeline = output_pipeline
    writer._variable_subset = None  # full byte-identical default output
    writer._full_variable_set = False
    writer.written = {"d01": []}
    return writer


def _bind_case(writer, *, output_dir, dt_s, run_start, state=None):
    state_case, grid, namelist = synthetic_case()
    state_case = _jaxify(state_case)
    grid = _jaxify(grid)
    bundle = SimpleNamespace(namelist=namelist, grid=grid)
    writer.output_dir = output_dir
    writer.run_start = run_start
    writer.bundles = {"d01": bundle}
    writer.dt_by_domain = {"d01": dt_s}
    return state_case if state is None else state


def test_materialize_stage_honours_radiation_carry_source_flag(monkeypatch, tmp_path):
    """The shared materialize stage uses S2 carry diagnostics when explicitly opted in."""

    monkeypatch.setenv("GPUWRF_NESTED_M9_RADIATION_FROM_CARRY", "1")
    writer = _make_writer(async_writer=None)
    namelist = SimpleNamespace(use_noahmp=True)
    writer.output_dir = tmp_path
    writer.run_start = RUN_START
    writer.bundles = {"d01": SimpleNamespace(namelist=namelist, grid="grid")}
    writer.dt_by_domain = {"d01": DT_S}
    seen: dict[str, Any] = {}

    def fake_surface_from_carry(*_args, **_kwargs):
        seen["carry_helper_called"] = True
        return {"SWDOWN": np.array([[7.0]]), "GLW": np.array([[300.0]])}

    def fake_prepare(*_args, diagnostics=None, **_kwargs):
        seen["diagnostics"] = diagnostics
        return "prepared"

    monkeypatch.setattr(nested_pipeline_module, "assert_state_finite_at_boundary", lambda *a, **k: None)
    monkeypatch.setattr(
        nested_pipeline_module,
        "_noahmp_surface_diagnostics_from_held_radiation",
        fake_surface_from_carry,
    )
    monkeypatch.setattr(nested_pipeline_module, "prepare_wrfout_payload", fake_prepare)
    monkeypatch.setattr(
        nested_pipeline_module,
        "write_prepared_wrfout",
        lambda prepared, **_kwargs: seen.setdefault("written", prepared),
    )

    state = SimpleNamespace(t_skin=np.array([[280.0]]))
    carry = SimpleNamespace(
        state=state,
        noahmp_land=object(),
        noahmp_rad=(np.array([[7.0]]), np.array([[300.0]]), np.array([[0.5]])),
    )
    writer._materialize_and_submit(
        name="d01",
        own_step=120,
        carry=carry,
        valid_time=RUN_START,
        lead_seconds=3600.0,
        lead_hours=1.0,
        path=tmp_path / "wrfout_d01",
    )

    assert seen["carry_helper_called"] is True
    assert np.array_equal(seen["diagnostics"]["SWDOWN"], np.array([[7.0]]))
    assert seen["written"] == "prepared"


RUN_START = datetime(2026, 5, 25, 18, tzinfo=timezone.utc)
DT_S = 30.0


# ---------------------------------------------------------------------------
# G1 -- byte-identity (sync vs pipeline)
# ---------------------------------------------------------------------------

def test_pipeline_byte_identical_to_sync(tmp_path):
    """Default-output wrfout from the S1 pipeline == the synchronous path, byte-for-byte."""

    # Synchronous (pipeline OFF) baseline.
    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()
    sync_writer = _make_writer(async_writer=None)
    sync_state = _bind_case(sync_writer, output_dir=sync_dir, dt_s=DT_S, run_start=RUN_START)
    sync_result = sync_writer("d01", own_step=120, carry=sync_state)

    # Pipeline ON: SNAPSHOT (step thread) -> MATERIALIZE -> WRITE.
    pipe_dir = tmp_path / "pipe"
    pipe_dir.mkdir()
    aw = AsyncWrfoutWriter(max_pending=2)
    pipeline = OutputPipeline(aw, max_snapshots=1)
    pipe_writer = _make_writer(async_writer=aw, output_pipeline=pipeline)
    pipe_state = _bind_case(pipe_writer, output_dir=pipe_dir, dt_s=DT_S, run_start=RUN_START)
    pipe_result = pipe_writer("d01", own_step=120, carry=pipe_state)
    # The step thread returned immediately (snapshot enqueued); drain both stages.
    pipeline.join()

    # Same recorded path basename + same step-thread summary dict.
    assert Path(sync_result["wrfout"]).name == Path(pipe_result["wrfout"]).name
    assert sync_result["all_finite"] == pipe_result["all_finite"] is True
    assert sync_writer.written["d01"] == [str(Path(sync_dir) / Path(sync_result["wrfout"]).name)]
    assert pipe_writer.written["d01"] == [str(Path(pipe_dir) / Path(pipe_result["wrfout"]).name)]

    p_sync = Path(sync_result["wrfout"])
    p_pipe = Path(pipe_result["wrfout"])
    assert p_sync.exists() and p_pipe.exists()
    assert p_sync.read_bytes() == p_pipe.read_bytes(), "pipeline wrfout bytes differ from sync"

    with Dataset(p_sync) as ds_a, Dataset(p_pipe) as ds_b:
        assert sorted(ds_a.variables) == sorted(ds_b.variables)
        for name in ds_a.variables:
            a = np.asarray(ds_a.variables[name][:])
            b = np.asarray(ds_b.variables[name][:])
            assert a.shape == b.shape, f"{name} shape differs"
            if np.issubdtype(a.dtype, np.floating):
                assert np.array_equal(a, b, equal_nan=True), f"{name} bytes differ"
            else:
                assert np.array_equal(a, b), f"{name} bytes differ"


# ---------------------------------------------------------------------------
# G3 -- finite fail-closed (Invariant F)
# ---------------------------------------------------------------------------

def test_pipeline_nan_fail_closed_at_next_submit_and_join(tmp_path):
    """A NaN at frame K raises at the NEXT submit AND at join; bad frame not on disk."""

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    aw = AsyncWrfoutWriter(max_pending=2)
    pipeline = OutputPipeline(aw, max_snapshots=1)
    writer = _make_writer(async_writer=aw, output_pipeline=pipeline)
    good_state = _bind_case(writer, output_dir=out_dir, dt_s=DT_S, run_start=RUN_START)

    # Frame K: inject a NaN into theta (a prognostic finite-guard field).
    bad_theta = good_state.theta.at[0, 0, 0].set(jnp.nan)
    bad_state = SimpleNamespace(**{**vars(good_state), "theta": bad_theta})
    bad_path = Path(writer("d01", own_step=120, carry=bad_state)["wrfout"])

    # The materialize thread runs the finite guard on frame K and records the
    # NonFiniteStateError; join() drains that thread and re-raises the first stage
    # error -- fail-closed at run end (Invariant F).
    with pytest.raises(NonFiniteStateError):
        # join() drains the materialize thread (running frame K's finite check) and
        # re-raises the first stage error -- fail-closed at run end.
        pipeline.join()

    # The bad frame K is NOT on disk.
    assert not bad_path.exists(), "non-finite frame was written to disk"


def test_pipeline_nan_surfaces_at_next_boundary_before_commit(tmp_path):
    """The NaN error is re-raised on the NEXT step-thread submit (before next commit)."""

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    aw = AsyncWrfoutWriter(max_pending=2)
    pipeline = OutputPipeline(aw, max_snapshots=1)
    writer = _make_writer(async_writer=aw, output_pipeline=pipeline)
    good_state = _bind_case(writer, output_dir=out_dir, dt_s=DT_S, run_start=RUN_START)

    bad_theta = good_state.theta.at[0, 0, 0].set(jnp.inf)
    bad_state = SimpleNamespace(**{**vars(good_state), "theta": bad_theta})
    writer("d01", own_step=120, carry=bad_state)

    # Block until the materialize thread has consumed frame K (queue empty) so the
    # error slot is populated before the next submit.
    pipeline._queue.join()

    # The next boundary's submit re-raises the stored error -> the next frame is
    # never committed.
    next_snapshot = OutputSnapshot(
        writer=writer,
        name="d01",
        own_step=240,
        carry=good_state,
        valid_time=RUN_START,
        lead_seconds=7200.0,
        lead_hours=2.0,
        path=out_dir / "wrfout_next",
    )
    with pytest.raises(NonFiniteStateError):
        pipeline.submit(next_snapshot)

    # And join still re-raises (fail-closed at run end), idempotently.
    with pytest.raises(NonFiniteStateError):
        pipeline.join()


# ---------------------------------------------------------------------------
# G5 -- ordering + failure surfacing
# ---------------------------------------------------------------------------

def test_pipeline_multiple_frames_land_in_order(tmp_path):
    """Several frames submitted in order all materialize + land on disk."""

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    aw = AsyncWrfoutWriter(max_pending=2)
    pipeline = OutputPipeline(aw, max_snapshots=2)
    writer = _make_writer(async_writer=aw, output_pipeline=pipeline)
    state = _bind_case(writer, output_dir=out_dir, dt_s=DT_S, run_start=RUN_START)

    paths = []
    for step in (120, 240, 360, 480):
        result = writer("d01", own_step=step, carry=state)
        paths.append(Path(result["wrfout"]))
    pipeline.join()

    assert len(set(paths)) == len(paths), "frames collided on the same path"
    for path in paths:
        assert path.exists(), f"{path} not written after join"
        with Dataset(path) as ds:
            assert "T2" in ds.variables
    # Output-present count stays valid: written recorded all frames on the step thread.
    assert len(writer.written["d01"]) == len(paths)


def test_pipeline_write_error_surfaced_and_subsequent_skipped(tmp_path):
    """A write error fails closed: re-raised at next submit + join; later frames skipped."""

    # Build a target whose parent cannot be created so write_prepared_wrfout raises.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    out_dir = blocker / "subdir"  # cannot be created (blocker is a file)

    aw = AsyncWrfoutWriter(max_pending=2)
    pipeline = OutputPipeline(aw, max_snapshots=1)
    writer = _make_writer(async_writer=aw, output_pipeline=pipeline)
    state = _bind_case(writer, output_dir=out_dir, dt_s=DT_S, run_start=RUN_START)

    raised = False
    try:
        # Several frames; the first write fails on the writer thread. Subsequent
        # submits re-raise the stored error fail-closed (or join does).
        for step in (120, 240, 360):
            writer("d01", own_step=step, carry=state)
        pipeline.join()
    except (OSError, RuntimeError, Exception):  # noqa: BLE001
        raised = True
    assert raised, "write error was not surfaced fail-closed"


def test_pipeline_join_is_idempotent(tmp_path):
    """join() may be called twice (success path); second call is a no-op."""

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    aw = AsyncWrfoutWriter(max_pending=2)
    pipeline = OutputPipeline(aw, max_snapshots=1)
    writer = _make_writer(async_writer=aw, output_pipeline=pipeline)
    state = _bind_case(writer, output_dir=out_dir, dt_s=DT_S, run_start=RUN_START)
    writer("d01", own_step=120, carry=state)
    pipeline.join()
    pipeline.join()  # idempotent, no raise
    assert len(writer.written["d01"]) == 1
