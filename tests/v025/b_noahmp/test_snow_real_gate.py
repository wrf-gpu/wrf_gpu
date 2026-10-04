"""NS01: native REAL Noah-MP snow vs pristine WRF snow routines at REAL kind 4 (14 frozen scenarios)."""
import importlib.util
from pathlib import Path

import jax
import pytest


def _gate():
    path = Path(__file__).with_name("snow_real_gate.py")
    spec = importlib.util.spec_from_file_location("ns01_snow_real_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_real_snow_matches_pristine_wrf_real4(monkeypatch):
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")
    summary = _gate().run_gate()
    assert summary["n"] == 14
    assert {d for s in summary["scenarios"] for d in s["dtypes"]} == {"float32"}
    failed = [(s["scenario"], {k: v for k, v in s["checks"].items() if not v["pass"]})
              for s in summary["scenarios"] if not s["pass"]]
    assert summary["all_pass"], failed


@pytest.mark.parametrize("factor", [0.999, 1.001])
def test_gate_rejects_a_compaction_mutant(monkeypatch, factor):
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")
    gate = _gate()
    original = gate.snowmod._compact
    monkeypatch.setattr(gate.snowmod, "_compact", lambda *a, **k: original(*a, **k) * factor)
    jax.clear_caches()  # noahmp_snow is jitted; the canonical test's trace would hide the mutant
    try:
        assert not gate.run_gate()["all_pass"]
    finally:
        monkeypatch.undo()
        jax.clear_caches()
