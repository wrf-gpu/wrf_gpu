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

    # P1-5 PRECIP/WATER PARITY tolerance band (precip oracle, one 18 s step).
    # PREDECLARED and tightened from the old #32 *functional* gate after the three
    # WRF-faithful fixes: (1) adaptive per-species nstep (NSED_MAX masked scan,
    # nstep = MAX_k INT(DT/(dz/vt)+1)); (2) the rr(kts)>1e-9 surface-accumulation
    # threshold; (3) cloud-water sedimentation (single full-DT pass below 500 m
    # AGL, module_mp_thompson.F:3824-3837 -- NOT counted as surface precip).
    # Fixes (1)+(2) collapsed the +13% surface-precip bias to within +-3% and the
    # rain-field error to <1%; fix (3) collapsed the qc error (max_rel 1.0 -> ~1e-4,
    # mean_rel 7e-3 -> ~8e-6) and is reported separately as a water-budget sink.
    #  - water closure mass-conserving (FULL budget incl the cloud-w sink)
    assert pm["water_closure_max_rel_residual"] < 1e-5, pm
    #  - qv faithful (<1% mean), qr now WRF-PARITY (<1% mean / <2% max)
    assert pf["qv"]["mean_rel"] < 0.01, pf["qv"]
    assert pf["qr"]["mean_rel"] < 0.01, pf["qr"]
    assert pf["qr"]["max_rel"] < 0.02, pf["qr"]
    #  - qc now WRF-PARITY after cloud-water sedimentation (fix 3): the prior
    #    neglect of the cloud-w fall term left qc max_rel == 1.0 on the surface
    #    layers; the faithful term closes it to <1% (predeclared band).
    assert pf["qc"]["mean_rel"] < 0.001, pf["qc"]
    assert pf["qc"]["max_rel"] < 0.01, pf["qc"]
    #  - snow profile placement (mean_rel) within the WRF single-mode closure band
    assert pf["qs"]["mean_rel"] < 0.15, pf["qs"]
    #  - cloud-w sed is a water SINK, never surface precip: must be present, small,
    #    and separate from the rain/snow/graupel/ice surface total.
    assert "cloudw" in r["precip_by_species_mm"], r["precip_by_species_mm"]
    assert 0.0 <= r["precip_by_species_mm"]["cloudw"] < 0.05 * pm["total_surface_precip_mm"], r["precip_by_species_mm"]
    #  - SURFACE PRECIP PARITY: total within +-3% of WRF RAINNCV (was +13%).
    jax_total = pm["total_surface_precip_mm"]
    wrf_total = wrf["wrf_total_rainncv_mm"]
    assert jax_total > 0.0
    assert wrf_total > 0.0
    ratio = jax_total / wrf_total
    assert 0.97 <= ratio <= 1.03, (jax_total, wrf_total, ratio)
    #  - per-column surface precip tracks WRF within 3% on every column.
    jper = pm["surface_precip_mm_per_col"]
    wper = wrf["wrf_rainncv_mm_per_col"]
    for jc, wc in zip(jper, wper):
        assert abs(jc - wc) <= 0.03 * wc + 1e-6, (jc, wc)


@pytest.mark.skipif(not _have_oracle(), reason="precip oracle savepoint absent")
def test_implicit_sed_is_more_diffusive_and_gated_off_by_default():
    sys.path.insert(0, str(ROOT / "proofs" / "thompson_perf"))
    from precip_oracle_validate import run_scheme
    from implicit_sedimentation_prototype import sedimentation_implicit

    faithful = run_scheme("faithful_explicit")
    # **kw absorbs the v0.15 ``vts_boost`` keyword the production body now
    # threads into ``_sedimentation`` (the prototype has no riming boost).
    implicit = run_scheme("implicit", lambda s, dt, **kw: sedimentation_implicit(s, dt, nsub=1))

    # implicit single-sweep BE over-precipitates relative to faithful explicit
    # (more diffusive: smears the falling front to the surface) -> documented
    # reason it is REJECTED as a default.
    assert implicit["precip_mass"]["total_surface_precip_mm"] > \
        faithful["precip_mass"]["total_surface_precip_mm"]
