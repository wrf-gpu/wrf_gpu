"""New-Tiedtke JAX kernel (cu_physics=16) parity gates.

Three acceptance gates for :mod:`gpuwrf.physics.cumulus_ntiedtke_jax`, the
jit/vmap-traceable transcription of the validated NumPy reference
:mod:`gpuwrf.physics.cumulus_ntiedtke`:

1. vs the 5 fp64 WRF oracle savepoints
   (``proofs/v013/savepoints/cumulus/ntiedtke_case_{1..5}.json``) with the
   same predeclared bands as the reference oracle test
   (``tests/test_ntiedtke_cumulus_oracle.py``).
2. vs the NumPy reference on the 5 savepoint inputs AND >= 20 perturbed
   variants (fixed-seed multiplicative fp64 perturbations of T/QV/W and the
   QVFTEN/THFTEN/QFX/HFX forcings, spanning triggering and non-triggering
   regimes): agreement <= 1e-12 max_abs on every output.
3. jit compiles once; vmap over an (8, kx) batch built from the savepoint
   columns matches the per-column jit calls bitwise.

Run CPU-only:
  env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=src \
      taskset -c 4-31 pytest -q tests/test_ntiedtke_jax_parity.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

from gpuwrf.physics.cumulus_ntiedtke import ntiedtke_column  # noqa: E402
from gpuwrf.physics.cumulus_ntiedtke_jax import ntiedtke_column_jax  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / "proofs" / "v013" / "savepoints" / "cumulus"
CASES = (1, 2, 3, 4, 5)
TENDENCY_FIELDS = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN")
MOMENTUM_FIELDS = ("RUCUTEN", "RVCUTEN")
ALL_FIELDS = TENDENCY_FIELDS + MOMENTUM_FIELDS

# Gate 1 bands: identical to tests/test_ntiedtke_cumulus_oracle.py.
TEND_ABS = 1.0e-12
MOM_ABS = 1.0e-12
NONTRIV_REL = 1.0e-10
REL_SCALE_FLOOR = 1.0e-13
RAINCV_ABS = 1.0e-12

# Gate 2 band: JAX kernel vs NumPy reference, every output.
JAX_VS_REF_ABS = 1.0e-12

# Regimes the five savepoints exercise (per the oracle savepoint scalars).
PRECIP_CASES = (1, 3)
ZERO_PRECIP_CASES = (2, 4, 5)

# Fixed-seed perturbation ensemble (5 cases x 5 amplitudes = 25 variants).
PERTURB_SEED = 20260702
PERTURB_SCALES = (1.0e-6, 1.0e-4, 1.0e-3, 5.0e-3, 2.0e-2)

# jit trace counter (gate 3: exactly one compile for repeated same-shape calls)
_TRACE_COUNT = {"n": 0}


def _column_fn(*args):
    _TRACE_COUNT["n"] += 1
    return ntiedtke_column_jax(*args, stepcu=5, itimestep=2)


_JF = jax.jit(_column_fn)


def _load(case: int) -> dict:
    with (SAVE / f"ntiedtke_case_{case}.json").open() as fh:
        return json.load(fh)


def _arr(columns: dict, name: str) -> np.ndarray:
    return np.asarray(columns[name], dtype=np.float64)


def _case_inputs(d: dict):
    """(array_args, scalar_args) in the frozen kernel argument order."""
    s = d["scalars"]
    c = d["columns"]
    arrays = [_arr(c, n) for n in
              ("T", "QV", "QC", "QI", "P", "P8W", "DZ", "RHO", "PI",
               "U", "V", "W", "QVFTEN", "THFTEN")]
    scalars = [np.float64(s["QFX"]), np.float64(s["HFX"]),
               np.float64(s["XLAND"]), np.float64(s["DX"]),
               np.float64(s["DT"])]
    assert int(s["STEPCU"]) == 5 and int(s["ITIMESTEP"]) == 2
    return arrays, scalars


def _run_ref(arrays, scalars):
    qfx, hfx, xland, dx, dt = scalars
    return ntiedtke_column(*arrays, qfx, hfx, xland, dt, dx=dx,
                           stepcu=5, itimestep=2)


def _run_jax(arrays, scalars):
    out = _JF(*arrays, *scalars)
    return {k: np.asarray(v) for k, v in out.items()}


def _perturbed_inputs():
    """25 fixed-seed multiplicative perturbations of the savepoint columns.

    Perturbs T, QV, W (state) and QVFTEN, THFTEN, QFX, HFX (forcings) --
    the trigger-relevant fields -- at amplitudes from 1e-6 to 2e-2 so the
    ensemble spans triggering and non-triggering regimes.
    """
    rng = np.random.default_rng(PERTURB_SEED)
    variants = []
    for case in CASES:
        d = _load(case)
        for scale in PERTURB_SCALES:
            arrays, scalars = _case_inputs(d)
            (t, qv, qc, qi, p, p8w, dz, rho, pi, u, v, w,
             qvften, thften) = arrays
            t = t * (1.0 + scale * rng.uniform(-1.0, 1.0, t.shape))
            qv = qv * (1.0 + scale * rng.uniform(-1.0, 1.0, qv.shape))
            w = w * (1.0 + scale * rng.uniform(-1.0, 1.0, w.shape))
            qvften = qvften * (1.0 + scale * rng.uniform(-1.0, 1.0,
                                                         qvften.shape))
            thften = thften * (1.0 + scale * rng.uniform(-1.0, 1.0,
                                                         thften.shape))
            qfx, hfx, xland, dx, dt = scalars
            qfx = qfx * np.float64(1.0 + scale * rng.uniform(-1.0, 1.0))
            hfx = hfx * np.float64(1.0 + scale * rng.uniform(-1.0, 1.0))
            arrays = [t, qv, qc, qi, p, p8w, dz, rho, pi, u, v, w,
                      qvften, thften]
            scalars = [qfx, hfx, xland, dx, dt]
            variants.append((case, scale, arrays, scalars))
    return variants


# ---------------------------------------------------------------------------
# Gate 1: JAX kernel vs fp64 WRF oracle savepoints (reference test bands).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_ntiedtke_jax_fp64_savepoint_parity(case: int) -> None:
    d = _load(case)
    arrays, scalars = _case_inputs(d)
    out = _run_jax(arrays, scalars)
    c = d["columns"]
    failures = []

    for field, abs_band in (
        [(f, TEND_ABS) for f in TENDENCY_FIELDS]
        + [(f, MOM_ABS) for f in MOMENTUM_FIELDS]
    ):
        oracle = _arr(c, field)
        mad = float(np.max(np.abs(out[field] - oracle)))
        scale = float(np.max(np.abs(oracle)))
        if mad > abs_band:
            failures.append(f"{field} max_abs={mad:.3e} > {abs_band:.1e}")
        if scale > REL_SCALE_FLOOR and mad / scale > NONTRIV_REL:
            failures.append(
                f"{field} max_rel={mad / scale:.3e} > {NONTRIV_REL:.1e}")

    rain_abs = abs(float(out["RAINCV"]) - float(d["scalars"]["RAINCV"]))
    if rain_abs > RAINCV_ABS:
        failures.append(f"RAINCV abs={rain_abs:.3e} > {RAINCV_ABS:.1e}")
    prate_abs = abs(float(out["PRATEC"]) - float(d["scalars"]["PRATEC"]))
    if prate_abs > RAINCV_ABS:
        failures.append(f"PRATEC abs={prate_abs:.3e} > {RAINCV_ABS:.1e}")

    assert not failures, f"case {case}: " + "; ".join(failures)


def test_ntiedtke_jax_savepoints_cover_real_regimes() -> None:
    """The savepoint evidence spans triggering AND non-triggering regimes."""
    for case in PRECIP_CASES:
        d = _load(case)
        assert float(d["scalars"]["RAINCV"]) > 0.0
        out = _run_jax(*_case_inputs(d))
        assert float(out["RAINCV"]) > 0.0
    for case in ZERO_PRECIP_CASES:
        d = _load(case)
        assert float(d["scalars"]["RAINCV"]) == 0.0
        out = _run_jax(*_case_inputs(d))
        assert float(out["RAINCV"]) == 0.0


# ---------------------------------------------------------------------------
# Gate 2: JAX kernel vs NumPy reference (savepoints + perturbation ensemble).
# ---------------------------------------------------------------------------
def _assert_jax_matches_ref(tag, arrays, scalars, failures):
    ref = _run_ref(arrays, scalars)
    out = _run_jax(arrays, scalars)
    for field in ALL_FIELDS:
        mad = float(np.max(np.abs(out[field] - ref[field])))
        if mad > JAX_VS_REF_ABS:
            failures.append(f"{tag} {field} max_abs={mad:.3e}")
    for field in ("RAINCV", "PRATEC"):
        dv = abs(float(out[field]) - float(ref[field]))
        if dv > JAX_VS_REF_ABS:
            failures.append(f"{tag} {field} abs={dv:.3e}")
    return ref


@pytest.mark.parametrize("case", CASES)
def test_ntiedtke_jax_vs_numpy_reference_savepoints(case: int) -> None:
    failures = []
    d = _load(case)
    arrays, scalars = _case_inputs(d)
    _assert_jax_matches_ref(f"case{case}", arrays, scalars, failures)
    assert not failures, "; ".join(failures)


def test_ntiedtke_jax_vs_numpy_reference_perturbed() -> None:
    """25 perturbed variants; both regimes must actually occur."""
    failures = []
    regimes = {"deep": 0, "shallow": 0, "mid": 0, "none": 0}
    for case, scale, arrays, scalars in _perturbed_inputs():
        ref = _assert_jax_matches_ref(f"case{case}/s{scale:g}", arrays,
                                      scalars, failures)
        if not ref["LDCUM"]:
            regimes["none"] += 1
        elif ref["KTYPE"] == 1:
            regimes["deep"] += 1
        elif ref["KTYPE"] == 2:
            regimes["shallow"] += 1
        else:
            regimes["mid"] += 1
    assert not failures, "; ".join(failures)
    n_trig = regimes["deep"] + regimes["shallow"] + regimes["mid"]
    # both regimes must actually occur in the ensemble (not happy-path only)
    assert n_trig > 0, f"no triggering variant: {regimes}"
    assert regimes["none"] > 0, f"no non-triggering variant: {regimes}"
    print(f"\nperturbed regime counts: {regimes} "
          f"(triggering={n_trig}, non-triggering={regimes['none']})")


# ---------------------------------------------------------------------------
# Gate 3: single jit compile + vmap batch == per-column calls, bitwise.
# ---------------------------------------------------------------------------
def test_ntiedtke_jax_jit_compiles_once() -> None:
    d1 = _load(1)
    d2 = _load(2)
    before = _TRACE_COUNT["n"]
    _run_jax(*_case_inputs(d1))
    _run_jax(*_case_inputs(d2))
    _run_jax(*_case_inputs(d1))
    after = _TRACE_COUNT["n"]
    # same shapes/dtypes/statics -> at most ONE trace in this window (zero if
    # an earlier test already compiled it)
    assert after - before <= 1, f"retraced {after - before} times"
    assert _TRACE_COUNT["n"] == 1, f"compiled {_TRACE_COUNT['n']} times total"


# The vmap TRANSFORM is value-preserving: a (1, kx) vmap batch is bitwise
# identical to the per-column jit call (asserted below).  Across SIMD widths,
# however, XLA:CPU lowers exp/log/pow with vector-width-dependent rounding
# (<= 1 ulp per call), so an (8, kx) batch differs from width-1 execution at
# the ulp level (observed worst 2.3e-16 on RTHCUTEN after the deep-closure
# amplification; 1.6e-21 on the moisture tendencies).  That is backend
# codegen, not transcription -- the jaxpr is identical -- so the width-8
# comparison uses the same predeclared 1e-12 band as the reference parity
# gate, while bitwise identity is asserted at width 1.
VMAP_WIDTH_ABS = 1.0e-12


def _vmap_fn(*args):
    return ntiedtke_column_jax(*args, stepcu=5, itimestep=2)


_VMAP_AXES = (0,) * 14 + (0, 0, 0, 0, None)
_VF = jax.jit(jax.vmap(_vmap_fn, in_axes=_VMAP_AXES))


def _vmap_batch():
    """(8, kx) batch: the 5 savepoint columns + 3 perturbed variants."""
    variants = _perturbed_inputs()
    batch = [_case_inputs(_load(case)) for case in CASES]
    batch += [(arrays, scalars) for (case, scale, arrays, scalars)
              in variants[:3]]
    assert len(batch) == 8
    dts = {float(sc[4]) for _, sc in batch}
    assert len(dts) == 1, "vmap batch assumes a shared DT"
    arr_stack = [np.stack([b[0][i] for b in batch]) for i in range(14)]
    sc_stack = [np.stack([b[1][j] for b in batch]) for j in range(4)]
    dt = batch[0][1][4]
    return batch, arr_stack, sc_stack, dt


def test_ntiedtke_jax_vmap_transform_is_bitwise() -> None:
    """vmap over a (1, kx) batch == the per-column jit call, bitwise."""
    batch, arr_stack, sc_stack, dt = _vmap_batch()
    for i, (arrays, scalars) in enumerate(batch):
        v1 = _VF(*[a[i:i + 1] for a in arr_stack],
                 *[s[i:i + 1] for s in sc_stack], dt)
        v1 = {k: np.asarray(v) for k, v in v1.items()}
        sout = _run_jax(arrays, scalars)
        for field in ALL_FIELDS:
            assert np.array_equal(v1[field][0], sout[field]), (
                f"vmap(1) row {i} {field} not bitwise-identical")
        for field in ("RAINCV", "PRATEC"):
            assert float(v1[field][0]) == float(sout[field]), (
                f"vmap(1) row {i} {field} not bitwise-identical")


def test_ntiedtke_jax_vmap_batch8_matches_per_column() -> None:
    """(8, kx) vmap batch vs per-column calls: within the SIMD-width band."""
    batch, arr_stack, sc_stack, dt = _vmap_batch()
    vout = _VF(*arr_stack, *sc_stack, dt)
    vout = {k: np.asarray(v) for k, v in vout.items()}

    worst = 0.0
    for i, (arrays, scalars) in enumerate(batch):
        sout = _run_jax(arrays, scalars)
        for field in ALL_FIELDS:
            mad = float(np.max(np.abs(vout[field][i] - sout[field])))
            worst = max(worst, mad)
            assert mad <= VMAP_WIDTH_ABS, (
                f"vmap row {i} {field} max_abs={mad:.3e} > "
                f"{VMAP_WIDTH_ABS:.1e}")
        for field in ("RAINCV", "PRATEC"):
            dv = abs(float(vout[field][i]) - float(sout[field]))
            worst = max(worst, dv)
            assert dv <= VMAP_WIDTH_ABS, (
                f"vmap row {i} {field} abs={dv:.3e} > {VMAP_WIDTH_ABS:.1e}")
    print(f"\nvmap(8) vs per-column worst max_abs = {worst:.3e} "
          f"(band {VMAP_WIDTH_ABS:.1e}; width-1 vmap is bitwise)")


def test_ntiedtke_catalog_classification_implemented():
    """cu=16 graduates to IMPLEMENTED in v0.23 F2 (WSM7-graduation pattern).

    The machine-precision fp64 kernel (this file + the NumPy-reference oracle
    gate) + the ntiedtke_adapter scan wiring flip New-Tiedtke from
    REFERENCE_ONLY to IMPLEMENTED. The catalog must stay internally consistent
    and the neighbouring cumulus options unchanged.
    """
    from gpuwrf.coupling.scan_adapters import (
        CU_SCAN_ADAPTERS,
        CU_STATELESS_SCAN_ADAPTERS,
        ntiedtke_adapter,
    )
    from gpuwrf.io.scheme_catalog import (
        SupportStatus,
        assert_catalog_consistent,
        classify_scheme,
    )
    from gpuwrf.runtime.operational_mode import (
        _SCAN_UNWIRED_REASON,
        _SCAN_WIRED_OPTIONS,
    )

    assert_catalog_consistent()
    assert classify_scheme("cu_physics", 16).status is SupportStatus.IMPLEMENTED
    # neighbours unchanged: cu=6 implemented, SAS family still reference-only.
    assert classify_scheme("cu_physics", 6).status is SupportStatus.IMPLEMENTED
    assert classify_scheme("cu_physics", 4).status is SupportStatus.REFERENCE_ONLY
    assert CU_SCAN_ADAPTERS[16] is ntiedtke_adapter
    assert CU_STATELESS_SCAN_ADAPTERS[16] is ntiedtke_adapter
    assert 16 in _SCAN_WIRED_OPTIONS["cu_physics"]
    assert "cu_physics=16" not in _SCAN_UNWIRED_REASON
