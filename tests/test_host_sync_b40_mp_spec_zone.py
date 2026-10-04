"""B40: time-split microphysics skips the specified zone (WRF solve_em.F:3693-3707).

Pristine WRF bounds moist_physics_prep_em / microphysics_driver /
moist_physics_finish_em to ``ids+sz..ide-1-sz`` (``sz = spec_zone`` when
``specified .or. nested``).  CPU-WRF PROD output confirms it: ring-0 RAINNC is
exactly 0 on d01 and d02 at +6 h and +30 h while d02 ring 1 rains.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling.physics_couplers import thompson_adapter
from gpuwrf.runtime.operational_mode import (
    _apply_post_rk_microphysics,
    _microphysics_interior_only,
    _microphysics_spec_zone,
)
from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state

RAIN01_D03 = Path("<USER_HOME>/wrf_gpu2_lanes/wn3/W3/ni_probe/rain01/d03_first_exceed.pkl")
# Read-only CPU-WRF WN3 0227 d03 frame with active hydrometeors at the nest edge.
WN3_D03_H18 = Path(
    "<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run/wrfout_d03_2026-02-28_18:00:00"
)


def _namelist(*, run_boundary=True, source="wrfbdy", force_geopotential=False, spec_zone=1):
    return SimpleNamespace(
        run_boundary=run_boundary,
        grid=SimpleNamespace(bc=SimpleNamespace(source=source)),
        boundary_config=SimpleNamespace(
            force_geopotential=force_geopotential, spec_zone=spec_zone
        ),
    )


def _ring(a, sz):
    a = np.asarray(a)
    mask = np.ones(a.shape[-2:], bool)
    mask[sz:a.shape[-2] - sz, sz:a.shape[-1] - sz] = False
    return a[..., mask]


def _interior(a, sz):
    return np.asarray(a)[..., sz:-sz, sz:-sz]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    (
        ({}, 1),                                   # nested (force_geopotential False)
        ({"force_geopotential": True}, 1),         # specified root with wrfbdy
        ({"spec_zone": 2}, 2),
        ({"run_boundary": False}, 0),              # periodic/ideal: no restriction
        ({"source": "ideal"}, 0),
    ),
)
def test_spec_zone_follows_solve_em_predicate(kwargs, expected):
    assert _microphysics_spec_zone(_namelist(**kwargs)) == expected


def test_interior_only_restores_ring_and_keeps_untouched_leaves():
    grid = build_flat_grid(nx=9, ny=8, nz=3, dx_m=1000.0)
    before = build_neutral_state(grid)
    after = before.replace(
        qc=before.qc + 1.0e-4,
        theta=before.theta + 0.5,
    )
    out = _microphysics_interior_only(before, after, _namelist())
    for name in ("qc", "theta"):
        np.testing.assert_array_equal(_ring(getattr(out, name), 1), _ring(getattr(before, name), 1))
        np.testing.assert_array_equal(_interior(getattr(out, name), 1), _interior(getattr(after, name), 1))
    assert out.qv is after.qv and out.u is after.u
    assert _microphysics_interior_only(before, after, _namelist(run_boundary=False)) is after


def _cpu_wrf_frame(state, path):
    """Weather of a real CPU-WRF frame on the same d03 grid/State template."""

    from netCDF4 import Dataset

    names = dict(qv="QVAPOR", qc="QCLOUD", qr="QRAIN", qi="QICE", qs="QSNOW",
                 qg="QGRAUP", Ni="QNICE", Nr="QNRAIN")
    with Dataset(path) as nc:
        raw = {k: np.asarray(nc.variables[v][0]) for k, v in names.items()}
        theta = np.asarray(nc.variables["T"][0]) + 300.0
        p_pert = np.asarray(nc.variables["P"][0])
        p_total = p_pert + np.asarray(nc.variables["PB"][0])
    updates = {k: jnp.asarray(v) for k, v in raw.items()}
    updates.update(theta=jnp.asarray(theta), p_total=jnp.asarray(p_total),
                   p_perturbation=jnp.asarray(p_pert))
    return state.replace(**updates)


@pytest.mark.skipif(not RAIN01_D03.is_file(), reason="WN3 Rain01 operands not on this host")
@pytest.mark.parametrize("weather", ("rain01", "cpu_wrf_h18"))
def test_real_wn3_d03_nest_edge_columns_get_no_microphysics(weather):
    with RAIN01_D03.open("rb") as fh:
        payload = pickle.load(fh)
    state = payload["before"].state
    namelist = payload["namelist"]
    if weather == "cpu_wrf_h18":
        if not WN3_D03_H18.is_file():
            pytest.skip("WN3 CPU-WRF d03 frame not mounted")
        state = _cpu_wrf_frame(state, WN3_D03_H18)
    sz = _microphysics_spec_zone(namelist)
    assert sz == int(namelist.boundary_config.spec_zone) == 1
    raw = thompson_adapter(state, float(namelist.dt_s))
    fixed = _apply_post_rk_microphysics(state, namelist)
    changed_ring = []
    for name in State_slots(state):
        before, unrestricted, got = (getattr(x, name) for x in (state, raw, fixed))
        if unrestricted is before or unrestricted is None:
            assert got is before or np.array_equal(np.asarray(got), np.asarray(before)), name
            continue
        # Interior = the unrestricted column physics, bit for bit.
        np.testing.assert_array_equal(_interior(got, sz), _interior(unrestricted, sz), err_msg=name)
        # Spec zone = exactly the pre-microphysics value (no tendency/precip).
        np.testing.assert_array_equal(_ring(got, sz), _ring(before, sz), err_msg=name)
        if not np.array_equal(_ring(unrestricted, sz), _ring(before, sz)):
            changed_ring.append(name)
    # Deletion sensitivity: on this real active nest edge the unrestricted
    # adapter does change ring-0 prognostics, so removing B40 fails above.
    assert {"theta", "qc"} & set(changed_ring), changed_ring
    if weather == "cpu_wrf_h18":
        # Active edge: hydrometeor/number prognostics themselves would change.
        assert len({"qc", "qr", "qi", "qs", "qg", "Ni", "Nr"} & set(changed_ring)) >= 2, changed_ring


def State_slots(state):
    return [name for name in type(state).__slots__ if hasattr(getattr(state, name), "shape")]
