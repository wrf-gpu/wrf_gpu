"""SI39: under carry donation, nothing in nested_pipeline reads a carry an advance has deleted."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.integration import nested_pipeline as pipeline


@pytest.fixture
def no_donation(monkeypatch):
    for name in [n for n in __import__("os").environ if n.startswith("GPUWRF_") and "DONATE" in n]:
        monkeypatch.delenv(name)


def test_donation_predicate(no_donation, monkeypatch):
    assert pipeline._carry_donation_enabled() is False
    monkeypatch.setenv("GPUWRF_CARRY_DONATE", "0")
    assert pipeline._carry_donation_enabled() is False
    monkeypatch.setenv("GPUWRF_CARRY_DONATE", "1")
    assert pipeline._carry_donation_enabled() is True
    monkeypatch.setenv("GPUWRF_CARRY_DONATE", "0")
    monkeypatch.setenv("GPUWRF_FORCEDOWN_DONATE", "1")  # any future donation flag counts
    assert pipeline._carry_donation_enabled() is True


def test_device_copy_tree_gives_distinct_equal_buffers():
    carry = {"u": jnp.arange(6.0).reshape(2, 3), "n": np.ones(2), "tag": "x", "none": None}
    out = pipeline._device_copy_tree(carry)
    assert out["u"].unsafe_buffer_pointer() != carry["u"].unsafe_buffer_pointer()
    assert np.array_equal(np.asarray(out["u"]), np.asarray(carry["u"]))
    assert out["n"] is carry["n"] and out["tag"] == "x" and out["none"] is None


class _State(NamedTuple):  # pytree stand-ins (OperationalCarry/State are pytrees too)
    theta: object


class _Carry(NamedTuple):
    state: _State


class _NoopAsyncWriter:
    def join(self):
        return None


def _drive_pipeline(tmp_path):
    """One S1 snapshot; the 'advance' deletes the carry before the materializer reads it."""
    read_gate, seen = threading.Event(), []
    writer = object.__new__(pipeline._PerDomainWrfoutWriter)
    writer.output_dir, writer.run_start = tmp_path, datetime(2026, 2, 28, tzinfo=timezone.utc)
    writer.dt_by_domain, writer.written = {"d01": 54.0}, {"d01": []}
    writer._output_pipeline = pipeline.OutputPipeline(_NoopAsyncWriter())
    writer._full_variable_set = False

    def _materialize(*, carry, **_kwargs):
        read_gate.wait(10)
        seen.append(np.asarray(carry.state.theta).copy())

    writer._materialize_and_submit = _materialize
    carry = _Carry(_State(jnp.full((2, 3), 300.0)))
    writer("d01", 67, carry)
    carry.state.theta.delete()  # what a donating advance does to its input carry
    read_gate.set()
    return writer._output_pipeline, seen


def test_output_snapshot_owns_a_copy_under_donation(tmp_path, no_donation, monkeypatch):
    monkeypatch.setenv("GPUWRF_CARRY_DONATE", "1")
    out_pipeline, seen = _drive_pipeline(tmp_path)
    out_pipeline.join()
    assert len(seen) == 1 and np.all(seen[0] == 300.0)


def test_output_snapshot_by_reference_without_donation(tmp_path, no_donation):
    """Flag off: unchanged by-reference snapshot (no copy) -> a deleted carry is visible."""
    out_pipeline, seen = _drive_pipeline(tmp_path)
    with pytest.raises(Exception, match="deleted"):
        out_pipeline.join()
    assert seen == []


def test_forcedown_warmup_never_deletes_live_init_carries_under_donation(no_donation, monkeypatch):
    import gpuwrf.runtime.domain_tree as dt

    def _donating_force(edge, parent, child):
        child["u_bdy"].delete()  # a force-down that donates the child's *_bdy records
        return SimpleNamespace(state=SimpleNamespace(u_bdy=jnp.zeros(1)))

    monkeypatch.setattr(dt, "_operational_force", _donating_force)
    monkeypatch.setenv("GPUWRF_CARRY_DONATE", "1")
    carries = {"d01": {"u_bdy": jnp.ones(3)}, "d02": {"u_bdy": jnp.ones(3)}}
    tree = SimpleNamespace(edges={"d01": [SimpleNamespace(parent="d01", child="d02", coupled_forcedown=True)]})
    report = pipeline._start_forcedown_warmup(tree, carries)
    report["thread"].join()
    assert report["error"] is None
    assert not carries["d02"]["u_bdy"].is_deleted() and np.all(np.asarray(carries["d02"]["u_bdy"]) == 1.0)
