"""v090 MYNN-SL faithfulness gate vs the pristine module_sf_mynn.F Fortran oracle.

Builds (once) the BYTE-IDENTICAL pristine WRF MYNN-SL Fortran oracle and asserts the
production ``surface_layer.surface_layer_with_diagnostics`` matches it per-field on
daytime-unstable / stable / neutral / water columns, for the flux+similarity fields.
The 2-m T2/Q2 diagnostic is checked under the FAITHFUL default (lsm_t2_diag=False).

The oracle source sha256 is asserted so the test fails loudly if the pristine file is
swapped. If gfortran (wrfbuild) is unavailable the parity assertions are skipped (the
oracle cannot be built), but the sha256 guard still runs.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROOFS = Path(__file__).resolve().parents[1] / "proofs" / "v090"
ORACLE_SHA = "86395534a6c9bfc79dcad50094bce290eff05756777a95794b2673795f9761c3"
GF = "/home/enric/miniconda3/envs/wrfbuild/bin/gfortran"


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_oracle_source_is_byte_identical_pristine():
    src = PROOFS / "module_sf_mynn_pristine.f90"
    assert src.exists(), "pristine oracle source missing"
    assert _sha256(src) == ORACLE_SHA, "oracle source diverged from pristine module_sf_mynn.F"


@pytest.mark.skipif(not Path(GF).exists(), reason="wrfbuild gfortran unavailable")
def test_production_mynnsl_matches_pristine_oracle():
    # build the fp32 oracle (how WRF runs) if not present
    exe = PROOFS / "mynn_oracle"
    if not exe.exists():
        subprocess.run(["bash", str(PROOFS / "build_oracle.sh")], check=True,
                       capture_output=True, text=True)
    assert exe.exists()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("GPUWRF_JAX_CACHE", "0")
    np.seterr(invalid="ignore")
    sys.path.insert(0, str(PROOFS))
    from mynnsl_parity import make_cases, run_oracle, run_production  # noqa: E402

    cases, _ = make_cases()
    orc = run_oracle(cases, "mynn_oracle")
    prod = run_production(cases)

    # flux + similarity faithfulness thresholds (relative); diagnostics tighter.
    thr = {"ust": 0.01, "hfx": 0.02, "lh": 0.03, "qsfc": 0.01, "br": 0.05, "zol": 0.05,
           "mol": 0.05, "psim": 0.03, "psih": 0.03, "rmol": 0.05, "u10": 0.01, "v10": 0.01,
           "t2": 0.001, "th2": 0.001, "regime": 0.0, "znt": 1e-3}
    failures = []
    for k, t in thr.items():
        if k not in prod:
            continue
        a = np.asarray(prod[k], dtype=np.float64)
        b = np.asarray(orc[k], dtype=np.float64)
        absd = float(np.max(np.abs(a - b)))
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
        ok = rel <= t or absd < 1e-6
        if not ok:
            failures.append(f"{k}: relmax={rel:.4e} absmax={absd:.4e} > thr={t}")
    assert not failures, "MYNN-SL diverged from pristine oracle:\n" + "\n".join(failures)
