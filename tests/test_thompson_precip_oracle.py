"""Regression for the PRECIPITATING WRF Thompson oracle validation (0.1.0 #32).

Validates the shipped FAITHFUL-EXPLICIT GPU Thompson against a real WRF
`mp_gt_driver` precipitating savepoint (active rain/snow/graupel/ice), and
asserts the implicit-sed gate is default-OFF so the shipped default is unchanged.

Skips when the oracle savepoint or a JAX GPU backend is unavailable (the
savepoint lives outside git at /mnt/data; regenerate via
proofs/thompson_perf/precip_oracle_harness/precip_column_oracle.F).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ORACLE = Path("/mnt/data/wrf_gpu2/physics_oracle/microphysics_precip")
ROOT = Path(__file__).resolve().parents[1]


def _have_oracle() -> bool:
    return (ORACLE / "thompson_in.sidecar.txt").is_file()


@pytest.mark.skipif(not _have_oracle(), reason="precip oracle savepoint absent")
def test_faithful_thompson_validates_against_precip_oracle():
    sys.path.insert(0, str(ROOT / "proofs" / "thompson_perf"))
    import gpuwrf.physics.thompson_column as tc
    from precip_oracle_validate import run_scheme, wrf_reference_precip

    # default must be faithful explicit (implicit gate OFF)
    assert tc._implicit_sed_nsub() == 0

    wrf = wrf_reference_precip()
    r = run_scheme("faithful_explicit")
    pf = r["per_field"]
    pm = r["precip_mass"]

    # #32 tolerance band (precip oracle, one 18 s step):
    #  - water closure mass-conserving
    assert pm["water_closure_max_rel_residual"] < 1e-5, pm
    #  - qv faithful (<1% mean), qr faithful (<10% mean) -- profiles match WRF
    assert pf["qv"]["mean_rel"] < 0.01, pf["qv"]
    assert pf["qr"]["mean_rel"] < 0.10, pf["qr"]
    #  - it actually precipitates and is within order of WRF (functional gate,
    #    NOT RAINNCV parity -- see PRECIP_ORACLE_AND_IMPLICIT_SED.md caveat)
    jax_total = pm["total_surface_precip_mm"]
    wrf_total = wrf["wrf_total_rainncv_mm"]
    assert jax_total > 0.0
    assert wrf_total > 0.0
    assert jax_total < 4.0 * wrf_total, (jax_total, wrf_total)


@pytest.mark.skipif(not _have_oracle(), reason="precip oracle savepoint absent")
def test_implicit_sed_is_more_diffusive_and_gated_off_by_default():
    sys.path.insert(0, str(ROOT / "proofs" / "thompson_perf"))
    from precip_oracle_validate import run_scheme
    from implicit_sedimentation_prototype import sedimentation_implicit

    faithful = run_scheme("faithful_explicit")
    implicit = run_scheme("implicit", lambda s, dt: sedimentation_implicit(s, dt, nsub=1))

    # implicit single-sweep BE over-precipitates relative to faithful explicit
    # (more diffusive: smears the falling front to the surface) -> documented
    # reason it is REJECTED as a default.
    assert implicit["precip_mass"]["total_surface_precip_mm"] > \
        faithful["precip_mass"]["total_surface_precip_mm"]
