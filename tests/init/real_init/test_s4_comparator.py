"""S4 comparator self-tests — the harness must be UNGAMEABLE.

These tests run on cores 0-3, CPU only (NO GPU), against the real.exe oracle
fixtures on disk. They prove four things about the comparator:

1. **sanity (ungameable):** a verbatim oracle->product copy scored against the
   SAME oracle file gives ~0 error and PASSES every field. (A comparator that
   reads the wrong variable or mis-shapes would show large error here.)
2. **fail mechanics:** a candidate perturbed ABOVE tolerance on a hour-0-critical
   field (T) FAILS that field — so the comparator is not a rubber stamp.
3. **pass mechanics:** a candidate perturbed strictly BELOW tolerance still
   PASSES — so the tolerance band is honored, not zero-tolerance.
4. **wrfbdy parity:** the coupled-value + tendency comparison runs and the
   sanity candidate passes the wrfbdy gate too.

If the oracle corpus is not mounted, the tests skip (the proof script is the
authoritative artifact); CI without /mnt still imports + unit-checks the metric.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpuwrf.init.real_init import comparator as C
from gpuwrf.init.real_init.types import WRFINPUT_TOLS

from tests.init.real_init._oracle_product import build_product_from_oracle


ORACLE_ROOT = Path("/mnt/data/canairy_meteo/runs")


def _first_case():
    cases = C.discover_oracle_cases(
        ORACLE_ROOT, require_domains=("d01", "d02", "d03"),
        require_wrfbdy=True, limit=1)
    if not cases:
        pytest.skip("real.exe oracle corpus not mounted at /mnt/data")
    return cases[0]


# --------------------------------------------------------------------------
# Pure-metric unit checks (no corpus needed).
# --------------------------------------------------------------------------
def test_masked_rmse_excludes_nonfinite_and_mask():
    native = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    oracle = np.array([[1.0, 2.5, 99.0], [4.0, 5.0, 6.0]])
    # no mask: the NaN pair is dropped; diffs are {0, .5, _, 0,0,0}
    rmse, maxabs, n = C._masked_rmse_maxabs(native, oracle, None)
    assert n == 5
    assert maxabs == pytest.approx(0.5)
    # mask out the second column entirely -> only the .0-diff points remain
    mask = np.array([[True, False, True], [True, False, True]])
    rmse2, maxabs2, n2 = C._masked_rmse_maxabs(native, oracle, mask)
    assert n2 == 3  # (0,0) (1,0) (1,2); (0,2) is NaN
    assert maxabs2 == pytest.approx(0.0)


def test_score_field_shape_mismatch_fails():
    r = C._score_field("T", np.zeros((3, 3)), np.zeros((4, 4)), WRFINPUT_TOLS)
    assert not r.passed and r.status == "SHAPE_MISMATCH"


def test_score_field_missing_native_fails():
    r = C._score_field("T", None, np.zeros((3, 3)), WRFINPUT_TOLS)
    assert not r.passed and r.status == "MISSING_NATIVE"


def test_forecast_gate_is_scaffold_only():
    # The plan dict must be returned without touching the GPU; execute=True
    # must raise (S5/manager owns the GPU serialization point).
    plan = C.run_forecast_gate(execute=False, cases=[])
    assert plan["executed"] is False and plan["gpu_bound"] is True
    with pytest.raises(NotImplementedError):
        C.run_forecast_gate(execute=True, cases=[])


# --------------------------------------------------------------------------
# Corpus-backed comparator self-tests.
# --------------------------------------------------------------------------
def test_sanity_oracle_vs_itself_is_zero_error():
    oc = _first_case()
    product = build_product_from_oracle(
        oc.wrfinput["d01"], domain="d01", wrfbdy_path=oc.wrfbdy_d01)
    res = C.compare_wrfinput(product, oc.wrfinput["d01"])
    # Every OK field must be (near) bit-zero error and PASS.
    ok = [f for f in res.fields if f.status == "OK"]
    assert ok, "no fields scored OK — extraction map is broken"
    for f in ok:
        assert f.rmse == pytest.approx(0.0, abs=1e-9), f"{f.name} rmse={f.rmse}"
        assert f.maxabs == pytest.approx(0.0, abs=1e-6), f"{f.name} maxabs={f.maxabs}"
        assert f.passed, f"{f.name} failed its own oracle (status {f.status})"
    assert res.passed


def test_sanity_wrfbdy_vs_itself_is_zero_error():
    oc = _first_case()
    product = build_product_from_oracle(
        oc.wrfinput["d01"], domain="d01", wrfbdy_path=oc.wrfbdy_d01)
    res = C.compare_wrfbdy(product, oc.wrfbdy_d01)
    ok = [f for f in res.fields if f.status == "OK"]
    assert ok, "no wrfbdy fields scored OK"
    for f in ok:
        assert f.rmse == pytest.approx(0.0, abs=1e-6), f"{f.name} rmse={f.rmse}"
        assert f.passed
    assert res.passed


def test_fail_mechanics_large_T_perturbation_fails():
    oc = _first_case()
    # +5 K on T is 50x the 0.10 K rmse tol -> must FAIL T (and the case).
    product = build_product_from_oracle(
        oc.wrfinput["d01"], domain="d01", perturb={"T": 5.0})
    res = C.compare_wrfinput(product, oc.wrfinput["d01"])
    t = next(f for f in res.fields if f.name == "T")
    assert not t.passed, "comparator rubber-stamped a 5 K T offset"
    assert t.rmse == pytest.approx(5.0, abs=1e-6)
    assert not res.passed


def test_pass_mechanics_subtol_perturbation_passes():
    oc = _first_case()
    # +0.05 K on T is below the 0.10 K rmse tol AND below the 1.0 K maxabs cap.
    product = build_product_from_oracle(
        oc.wrfinput["d01"], domain="d01", perturb={"T": 0.05})
    res = C.compare_wrfinput(product, oc.wrfinput["d01"])
    t = next(f for f in res.fields if f.name == "T")
    assert t.passed, f"sub-tol T offset wrongly failed: rmse={t.rmse}"
    assert t.rmse == pytest.approx(0.05, abs=1e-6)


def test_fail_mechanics_categorical_exact():
    oc = _first_case()
    # Any nonzero offset on a categorical (ISLTYP, exact tol 0.0) must FAIL.
    product = build_product_from_oracle(
        oc.wrfinput["d01"], domain="d01", perturb={"ISLTYP": 1.0})
    res = C.compare_wrfinput(product, oc.wrfinput["d01"])
    isl = next(f for f in res.fields if f.name == "ISLTYP")
    assert not isl.passed and isl.rmse > 0
