"""B45: WRF specified root without have_bcs_moist/have_bcs_scalar (Registry .false.).

Pristine solve_em spec+relaxes only QV on a specified root and applies
flow_dep_bdy to the other moist species and the scalars (Ni, Nr) after every
rk_update_scalar; the port's end-of-step pass must do the same when the root
config carries have_bcs_* = False.  The pristine-operator gate is
tests/v025/b_diff/probe_flow_dep.py (real d01 edge columns, bit-exact).
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.state import State
from gpuwrf.coupling.boundary_apply import BoundaryConfig, apply_lateral_boundaries, flow_dep_bdy
from gpuwrf.integration import nested_pipeline

NZ, NY, NX = 5, 18, 22


def _leaf(z_len, value, dtype=jnp.float64):
    return jnp.full((2, 4, 5, z_len, max(NX + 1, NY + 1)), value, dtype)


def _state():
    rng = np.random.default_rng(3)
    f64 = jnp.float64
    q = lambda s: jnp.asarray(s * (1.0 + 0.3 * rng.random((NZ, NY, NX))), f64)
    # Mixed-sign edge winds so both inflow and outflow cells occur on every side.
    u = jnp.asarray(rng.standard_normal((NZ, NY, NX + 1)), f64)
    v = jnp.asarray(rng.standard_normal((NZ, NY + 1, NX)), f64)
    fields = dict(
        theta=jnp.full((NZ, NY, NX), 300.0, f64), qv=q(0.01), qc=q(1e-4), qr=q(2e-5), Ni=q(1e3), Nr=q(50.0),
        u=u, v=v, w=jnp.zeros((NZ + 1, NY, NX), f64),
        p_total=jnp.ones((NZ, NY, NX), f64) * 1000.0, p_perturbation=jnp.zeros((NZ, NY, NX), f64),
        ph_total=jnp.ones((NZ + 1, NY, NX), f64) * 100.0, ph_perturbation=jnp.zeros((NZ + 1, NY, NX), f64),
        mu_total=jnp.ones((NY, NX), f64) * 1000.0, mu_perturbation=jnp.zeros((NY, NX), f64),
        theta_bdy=_leaf(NZ, 301.0), qv_bdy=_leaf(NZ, 0.012), qc_bdy=_leaf(NZ, 3e-4), qr_bdy=_leaf(NZ, 1e-4),
        Ni_bdy=_leaf(NZ, 5e3), Nr_bdy=_leaf(NZ, 90.0), u_bdy=_leaf(NZ, 3.0), v_bdy=_leaf(NZ, 4.0),
        w_bdy=_leaf(NZ + 1, 0.0), p_bdy=_leaf(NZ, 8.0), pb_bdy=_leaf(NZ, 900.0), ph_bdy=_leaf(NZ + 1, 5.0),
        phb_bdy=_leaf(NZ + 1, 95.0), mu_bdy=_leaf(1, 6.0), mub_bdy=_leaf(1, 990.0),
    )
    return State.tree_unflatten(None, [fields.get(name) for name in State.__slots__])


@pytest.mark.parametrize("dry_spec_only", [True, False])
def test_root_without_have_bcs_flow_deps_non_qv_species(dry_spec_only):
    state = _state()
    wrf = BoundaryConfig(have_bcs_moist=False, have_bcs_scalar=False)
    released = apply_lateral_boundaries(state, 0.0, 54.0, BoundaryConfig(), dry_spec_only=dry_spec_only)
    got = apply_lateral_boundaries(state, 0.0, 54.0, wrf, dry_spec_only=dry_spec_only)
    ring = np.zeros((NY, NX), bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    for name in ("qc", "qr", "Ni", "Nr"):
        field = np.asarray(getattr(state, name))
        expect = np.maximum(np.asarray(flow_dep_bdy(getattr(state, name), state.u, state.v, wrf)), 0.0)
        out = np.asarray(getattr(got, name))
        np.testing.assert_array_equal(out, expect)
        # No relaxation rows: everything off the spec ring is untouched.
        np.testing.assert_array_equal(out[:, ~ring], field[:, ~ring])
        # Inflow cells are zero, outflow cells copy the first interior cell.
        assert np.any(out[:, ring] == 0.0) and np.any(out[:, ring] > 0.0)
        # The released path spec+relaxes from the wrfbdy record instead.
        assert not np.array_equal(np.asarray(getattr(released, name)), out)
    np.testing.assert_array_equal(np.asarray(got.qv), np.asarray(released.qv))


def test_have_bcs_true_keeps_record_spec_relax():
    state = _state()
    on = BoundaryConfig(have_bcs_moist=True, have_bcs_scalar=True)
    released = apply_lateral_boundaries(state, 0.0, 54.0, BoundaryConfig(), dry_spec_only=True)
    got = apply_lateral_boundaries(state, 0.0, 54.0, on, dry_spec_only=True)
    for name in ("qv", "qc", "qr", "Ni", "Nr"):
        np.testing.assert_array_equal(np.asarray(getattr(got, name)), np.asarray(getattr(released, name)))


def test_root_bdy_flags_follow_wrf_namelist_defaults():
    flag = nested_pipeline._root_bdy_flag
    assert flag(None, "have_bcs_moist") is False
    assert flag({"bdy_control": {"spec_bdy_width": 5}}, "have_bcs_moist") is False
    assert flag({"bdy_control": {"have_bcs_moist": [True, False]}}, "have_bcs_moist") is True
    assert flag({"bdy_control": {"have_bcs_scalar": True}}, "have_bcs_scalar") is True

    @dataclasses.dataclass(frozen=True)
    class _Namelist:
        specified_bdy_cadence: bool = False
        specified_adv_degrade: bool = False
        boundary_config: BoundaryConfig = BoundaryConfig()

    meta = {"boundary": {"interval_seconds": 21600}}
    out = nested_pipeline._root_boundary_cadence_override(_Namelist(), meta, {"bdy_control": {"specified": [True]}})
    assert out.boundary_config.have_bcs_moist is False and out.boundary_config.have_bcs_scalar is False
    out = nested_pipeline._root_boundary_cadence_override(
        _Namelist(), meta, {"bdy_control": {"have_bcs_moist": [True], "have_bcs_scalar": [True]}})
    assert out.boundary_config.have_bcs_moist is True and out.boundary_config.have_bcs_scalar is True
