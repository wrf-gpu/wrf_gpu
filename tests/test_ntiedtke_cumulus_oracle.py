"""New-Tiedtke cumulus (cu_physics=16) fp64 parity vs WRF oracle savepoints.

Oracle: single-column driver linked against the UNMODIFIED WRF
``phys/module_cu_ntiedtke.F`` + ``phys/physics_mmm/cu_ntiedtke.F90`` compiled
fp64 (``-DDOUBLE_PRECISION``), committed at
``proofs/v013/savepoints/cumulus/ntiedtke_case_{1..5}.json``
(built by ``proofs/v013/oracle/cumulus/ntiedtke_build_and_run.sh``).

The port is the line-faithful NumPy reference
:mod:`gpuwrf.physics.cumulus_ntiedtke`.  Because BOTH sides are fp64, the
predeclared band is a machine-precision faithfulness gate (not a physical
tolerance): the port reproduces WRF's unsuffixed default-REAL literals by
rounding them through fp32 and reproduces the ``amax1`` REAL*4 demotion of
the precip accumulator.

Run CPU-only:
  env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=src \
      taskset -c 4-31 pytest -q tests/test_ntiedtke_cumulus_oracle.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / "proofs" / "v013" / "savepoints" / "cumulus"
CASES = (1, 2, 3, 4, 5)
TENDENCY_FIELDS = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN")
MOMENTUM_FIELDS = ("RUCUTEN", "RVCUTEN")

# Predeclared machine-precision faithfulness band (fp64 port vs fp64 oracle).
TEND_ABS = 1.0e-12          # K/s or kg/kg/s -- observed worst ~1e-15
MOM_ABS = 1.0e-12           # m/s^2         -- observed worst ~2e-16
NONTRIV_REL = 1.0e-10       # relative band on fields with real signal
REL_SCALE_FLOOR = 1.0e-13   # below this max|oracle| a field is roundoff dust
RAINCV_ABS = 1.0e-12        # mm per cumulus call (bit-exact up to 1 ulp)

# Regimes the five savepoints must exercise (deep + shallow + non-triggering),
# asserted via precipitation activity so the parity claim is not happy-path.
PRECIP_CASES = (1, 3)
ZERO_PRECIP_CASES = (2, 4, 5)


def _load(case: int) -> dict:
    with (SAVE / f"ntiedtke_case_{case}.json").open() as fh:
        return json.load(fh)


def _arr(columns: dict, name: str) -> np.ndarray:
    return np.asarray(columns[name], dtype=np.float64)


def _run_case(d: dict) -> dict:
    from gpuwrf.physics.cumulus_ntiedtke import ntiedtke_column

    s = d["scalars"]
    c = d["columns"]
    return ntiedtke_column(
        _arr(c, "T"), _arr(c, "QV"), _arr(c, "QC"), _arr(c, "QI"),
        _arr(c, "P"), _arr(c, "P8W"), _arr(c, "DZ"), _arr(c, "RHO"),
        _arr(c, "PI"), _arr(c, "U"), _arr(c, "V"), _arr(c, "W"),
        _arr(c, "QVFTEN"), _arr(c, "THFTEN"),
        s["QFX"], s["HFX"], s["XLAND"], s["DT"], dx=s["DX"],
        stepcu=int(s["STEPCU"]), itimestep=int(s["ITIMESTEP"]))


@pytest.mark.parametrize("case", CASES)
def test_ntiedtke_fp64_savepoint_parity(case: int) -> None:
    d = _load(case)
    out = _run_case(d)
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
            failures.append(f"{field} max_rel={mad / scale:.3e} > {NONTRIV_REL:.1e}")

    rain_abs = abs(out["RAINCV"] - float(d["scalars"]["RAINCV"]))
    if rain_abs > RAINCV_ABS:
        failures.append(f"RAINCV abs={rain_abs:.3e} > {RAINCV_ABS:.1e}")
    prate_abs = abs(out["PRATEC"] - float(d["scalars"]["PRATEC"]))
    if prate_abs > RAINCV_ABS:
        failures.append(f"PRATEC abs={prate_abs:.3e} > {RAINCV_ABS:.1e}")

    assert not failures, f"case {case}: " + "; ".join(failures)


def test_ntiedtke_savepoints_cover_real_regimes() -> None:
    """The parity evidence spans triggering AND non-triggering regimes."""
    for case in PRECIP_CASES:
        d = _load(case)
        assert float(d["scalars"]["RAINCV"]) > 0.0, f"case {case} should rain"
        out = _run_case(d)
        assert out["LDCUM"] and out["KTYPE"] == 1
        assert out["RAINCV"] > 0.0
    for case in ZERO_PRECIP_CASES:
        d = _load(case)
        assert float(d["scalars"]["RAINCV"]) == 0.0
        out = _run_case(d)
        assert out["RAINCV"] == 0.0


def test_ntiedtke_oracle_provenance() -> None:
    """The committed savepoints came from the checksummed pristine WRF source."""
    wrf_root = Path(os.environ.get("GPUWRF_WRF_ROOT", "<USER_HOME>/src/wrf_pristine/WRF"))
    core = wrf_root / "phys" / "physics_mmm" / "cu_ntiedtke.F90"
    wrapper = wrf_root / "phys" / "module_cu_ntiedtke.F"
    if not core.exists():
        pytest.skip("pristine WRF tree not available on this host")
    checksums_path = ROOT / "proofs" / "v013" / "t3_cumulus_source_checksums.json"
    checksums = json.loads(checksums_path.read_text())
    flat = json.dumps(checksums)
    for src in (core, wrapper):
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        assert digest in flat, f"{src.name} sha256 {digest} not in {checksums_path.name}"
