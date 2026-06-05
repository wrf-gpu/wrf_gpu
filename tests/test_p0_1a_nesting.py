"""P0-1a nesting unit tests (GPU-free, CPU JAX).

Covers the three deliverables against analytic oracles + the State boundary-leaf
interface:

  1. interpolation operators -- WRF cell-centered registration reproduces a linear
     field exactly; the node-aligned bilinear is off by exactly the WRF -1/3-cell
     registration; the full monotone-TR4 sint reference equals the linear gather on
     a linear field and stays monotone (no new extrema) on a step field.
  2. boundary-VALUE construction -- the two-time [old, new] leaf has the frozen
     State.*_bdy shape and reproduces the WRF bdy_*/bdy_t* cadence when fed through
     the existing boundary_apply.interpolate_boundary_leaf consumer.
  3. scheduler cadence -- 9->3->1 km subcycle counts and the
     parent-step -> force-children -> recurse-children event ordering match WRF.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")

import numpy as np
import jax.numpy as jnp
import pytest

from gpuwrf.nesting import interp as I
from gpuwrf.nesting import boundary_construction as BC
from gpuwrf.nesting import scheduler as S
from gpuwrf.coupling.boundary_apply import interpolate_boundary_leaf, SIDE_INDEX


# ---------------------------------------------------------------------------
# 1. interpolation operators
# ---------------------------------------------------------------------------


def _linear_parent(pny, pnx, a=10.0, bx=2.0, by=3.0):
    ii, jj = np.meshgrid(np.arange(pnx), np.arange(pny))
    return (a + bx * ii + by * jj).astype(np.float64)


def test_sint_linear_reproduces_linear_field_exactly():
    ratio, ips, jps = 3, 5, 4
    pny = pnx = 40
    cny = cnx = 12
    parent = _linear_parent(pny, pnx)
    w = I.build_sint_weights(parent_grid_ratio=ratio, i_parent_start=ips, j_parent_start=jps,
                             parent_ny=pny, parent_nx=pnx, child_ny=cny, child_nx=cnx)
    child = np.asarray(I.interp_sint_linear(jnp.asarray(parent), w))
    xc = (ips - 1) + (np.arange(cnx) - ratio // 2) / ratio
    yc = (jps - 1) + (np.arange(cny) - ratio // 2) / ratio
    XX, YY = np.meshgrid(xc, yc)
    expected = 10.0 + 2.0 * XX + 3.0 * YY
    assert np.max(np.abs(child - expected)) < 1e-10


def test_sint_tr4_reference_reproduces_linear_field():
    ratio, ips, jps = 3, 5, 4
    pny = pnx = 40
    cny = cnx = 12
    parent = _linear_parent(pny, pnx)
    child = I.sint_to_child_reference(parent, ratio=ratio, i_parent_start=ips,
                                      j_parent_start=jps, child_ny=cny, child_nx=cnx)
    w = I.build_sint_weights(parent_grid_ratio=ratio, i_parent_start=ips, j_parent_start=jps,
                             parent_ny=pny, parent_nx=pnx, child_ny=cny, child_nx=cnx)
    lin = np.asarray(I.interp_sint_linear(jnp.asarray(parent), w))
    # the TR4 limiter vanishes for a linear field => identical to the linear gather
    assert np.max(np.abs(child - lin)) < 1e-9


def test_bilinear_offset_from_sint_is_exactly_wrf_registration():
    ratio, ips, jps = 3, 5, 4
    pny = pnx = 40
    cny = cnx = 12
    parent = _linear_parent(pny, pnx, bx=2.0, by=3.0)
    ws = I.build_sint_weights(parent_grid_ratio=ratio, i_parent_start=ips, j_parent_start=jps,
                              parent_ny=pny, parent_nx=pnx, child_ny=cny, child_nx=cnx)
    wb = I.build_bilinear_weights(parent_grid_ratio=ratio, i_parent_start=ips, j_parent_start=jps,
                                  parent_ny=pny, parent_nx=pnx, child_ny=cny, child_nx=cnx)
    sint = np.asarray(I.interp_sint_linear(jnp.asarray(parent), ws))
    bil = np.asarray(I.interp_bilinear(jnp.asarray(parent), wb))
    # node-aligned bilinear sits +1/3 cell from cell-centered => +grad/3 each axis
    expected_shift = (2.0 + 3.0) / ratio
    assert abs(np.max(np.abs(bil - sint)) - expected_shift) < 1e-9


def test_sint_tr4_is_monotone_on_step_field():
    # a sharp step => the monotone limiter must NOT create values outside [lo, hi]
    ratio, ips, jps = 3, 5, 4
    pny = pnx = 40
    cny = cnx = 18
    parent = np.zeros((pny, pnx))
    parent[:, pnx // 2:] = 5.0  # step in x
    child = I.sint_to_child_reference(parent, ratio=ratio, i_parent_start=ips,
                                      j_parent_start=jps, child_ny=cny, child_nx=cnx)
    assert np.nanmin(child) >= -1e-9
    assert np.nanmax(child) <= 5.0 + 1e-9


# ---------------------------------------------------------------------------
# 2. boundary-VALUE construction
# ---------------------------------------------------------------------------


class _DuckState:
    """Minimal duck-typed state exposing only what build_child_boundary_package reads.

    Avoids constructing a full GridSpec/State (terrain/projection/metrics) for a
    pure boundary-package structural test.
    """

    _FIELDS = ("u", "v", "w", "theta", "qv", "p_perturbation", "p_total",
               "ph_perturbation", "ph_total", "mu_perturbation", "mu_total")
    _BDY = ("u_bdy", "v_bdy", "w_bdy", "theta_bdy", "qv_bdy", "ph_bdy", "phb_bdy",
            "p_bdy", "pb_bdy", "mu_bdy", "mub_bdy")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def replace(self, **updates):
        d = {f: getattr(self, f) for f in self._FIELDS}
        d.update({b: getattr(self, b) for b in self._BDY})
        d.update(updates)
        return _DuckState(**d)


def _make_state(nz, ny, nx, width, side_len, fill):
    z3 = jnp.full((nz, ny, nx), fill)
    z3f = jnp.full((nz + 1, ny, nx), fill)
    z2 = jnp.full((ny, nx), fill)
    bdy3 = jnp.zeros((1, 4, width, nz, side_len))
    bdy3f = jnp.zeros((1, 4, width, nz + 1, side_len))
    bdy2 = jnp.zeros((1, 4, width, 1, side_len))
    return _DuckState(
        u=jnp.full((nz, ny, nx + 1), fill), v=jnp.full((nz, ny + 1, nx), fill),
        w=z3f, theta=z3, qv=z3,
        p_perturbation=z3, p_total=z3 * 2,
        ph_perturbation=z3f, ph_total=z3f * 2,
        mu_perturbation=z2, mu_total=z2 * 2,
        u_bdy=jnp.zeros((1, 4, width, nz, side_len)),
        v_bdy=jnp.zeros((1, 4, width, nz, side_len)),
        w_bdy=bdy3f, theta_bdy=bdy3, qv_bdy=bdy3,
        ph_bdy=bdy3f, phb_bdy=bdy3f,
        p_bdy=bdy3, pb_bdy=bdy3, mu_bdy=bdy2, mub_bdy=bdy2,
    )


class _Grid:
    def __init__(self, ny, nx):
        self.ny = ny
        self.nx = nx


def test_boundary_package_two_time_leaf_shape_and_cadence():
    ratio, ips, jps = 3, 5, 4
    nz = 6
    pny, pnx = 40, 40
    cny, cnx = 12, 12
    width = 5
    side_len = max(cny, cnx) + 1
    weights = BC.build_nest_force_weights(
        parent_grid_ratio=ratio, i_parent_start=ips, j_parent_start=jps,
        parent_grid=_Grid(pny, pnx), child_grid=_Grid(cny, cnx), registration="sint",
    )
    # parent constant field => child ring is that constant (mass-coupling term vanishes)
    parent = _make_state(nz, pny, pnx, width, side_len, fill=7.0)
    child = _make_state(nz, cny, cnx, width, side_len, fill=1.0)
    out = BC.build_child_boundary_package(child, parent, weights, bdy_width=width)
    # leaf shape: (time=2, side=4, width, z, side_len)
    assert out.theta_bdy.shape == (2, 4, width, nz, side_len)
    assert out.mu_bdy.shape == (2, 4, width, 1, side_len)
    # WRF cadence: old ring == child current (1.0); new ring == parent target (7.0)
    th = np.asarray(out.theta_bdy)
    # west side, outer width 0, level 0, first interior column of the ring
    assert abs(th[0, SIDE_INDEX["W"], 0, 0, 0] - 1.0) < 1e-9   # old = child current
    assert abs(th[1, SIDE_INDEX["W"], 0, 0, 0] - 7.0) < 1e-9   # new = parent target


def test_boundary_package_rejects_unknown_registration_and_width():
    ratio, ips, jps = 3, 5, 4
    nz = 6
    pny, pnx = 40, 40
    cny, cnx = 12, 12
    width = 5
    side_len = max(cny, cnx) + 1
    with pytest.raises(ValueError, match="unknown nest interpolation registration"):
        BC.build_nest_force_weights(
            parent_grid_ratio=ratio, i_parent_start=ips, j_parent_start=jps,
            parent_grid=_Grid(pny, pnx), child_grid=_Grid(cny, cnx), registration="nearest",
        )
    weights = BC.build_nest_force_weights(
        parent_grid_ratio=ratio, i_parent_start=ips, j_parent_start=jps,
        parent_grid=_Grid(pny, pnx), child_grid=_Grid(cny, cnx), registration="sint",
    )
    parent = _make_state(nz, pny, pnx, width, side_len, fill=7.0)
    child = _make_state(nz, cny, cnx, width, side_len, fill=1.0)
    with pytest.raises(ValueError, match="bdy_width must be positive"):
        BC.build_child_boundary_package(child, parent, weights, bdy_width=0)


def test_boundary_leaf_time_interp_matches_wrf_bdy_plus_dtbc_tend():
    # feed the two-time leaf through the EXISTING consumer; at lead=0 -> old,
    # at lead=parent_dt -> new, at lead=parent_dt/2 -> midpoint (WRF linear cadence).
    ratio, ips, jps = 3, 5, 4
    nz = 6
    pny, pnx = 40, 40
    cny, cnx = 12, 12
    width = 5
    side_len = max(cny, cnx) + 1
    parent_dt = 30.0
    weights = BC.build_nest_force_weights(
        parent_grid_ratio=ratio, i_parent_start=ips, j_parent_start=jps,
        parent_grid=_Grid(pny, pnx), child_grid=_Grid(cny, cnx), registration="sint",
    )
    parent = _make_state(nz, pny, pnx, width, side_len, fill=7.0)
    child = _make_state(nz, cny, cnx, width, side_len, fill=1.0)
    out = BC.build_child_boundary_package(child, parent, weights, bdy_width=width)
    leaf = out.theta_bdy
    f0 = np.asarray(interpolate_boundary_leaf(leaf, 0.0, parent_dt))
    fmid = np.asarray(interpolate_boundary_leaf(leaf, parent_dt / 2, parent_dt))
    f1 = np.asarray(interpolate_boundary_leaf(leaf, parent_dt, parent_dt))
    w = SIDE_INDEX["W"]
    assert abs(f0[w, 0, 0, 0] - 1.0) < 1e-9          # bdy_* at dtbc=0
    assert abs(f1[w, 0, 0, 0] - 7.0) < 1e-9          # bdy_*+cdt*bdy_t* at dtbc=cdt
    assert abs(fmid[w, 0, 0, 0] - 4.0) < 1e-9        # linear midpoint = WRF bdy_*+0.5*cdt*bdy_t*


def test_field_sides_ring_layout_outer_to_inner():
    # the bdy_width axis must run outer (index 0 = domain edge) -> inner.
    nz, ny, nx = 3, 8, 8
    width = 3
    side_len = nx
    f = jnp.asarray(np.arange(ny * nx, dtype=np.float64).reshape(ny, nx)[None].repeat(nz, 0))
    sides = np.asarray(BC.field_sides_3d(f, width, side_len))
    # west side index 0 (outer) must be column 0 of the field; index 1 -> column 1
    w = SIDE_INDEX["W"]
    assert np.allclose(sides[w, 0, 0, :ny], np.asarray(f)[0, :, 0])
    assert np.allclose(sides[w, 1, 0, :ny], np.asarray(f)[0, :, 1])
    # east side index 0 (outer) must be the LAST column (flipped)
    e = SIDE_INDEX["E"]
    assert np.allclose(sides[e, 0, 0, :ny], np.asarray(f)[0, :, nx - 1])


# ---------------------------------------------------------------------------
# 3. scheduler cadence
# ---------------------------------------------------------------------------


def _tower():
    return S.NestTower.from_edges(
        ["d01", "d02", "d03"],
        [S.NestEdge("d01", "d02", 3, 22, 20), S.NestEdge("d02", "d03", 3, 56, 18)],
    )


def test_subcycle_counts_9_3_1():
    counts = S.expected_substep_counts(_tower(), root_steps=4)
    assert counts == {"d01": 4, "d02": 12, "d03": 36}


def test_forcedown_ordering_advance_then_force_then_recurse():
    log = S.forcedown_event_log(_tower(), root_steps=1)
    # first events: advance d01, then force d02, then recurse d02
    assert log[0] == ("advance", "d01", 1)
    assert log[1] == ("force", "d02", 1)
    assert log[2] == ("recurse", "d02", 1)
    # per root step: 1 d01 advance, 3 d02 advances, 9 d03 advances
    from collections import Counter
    adv = Counter(d for k, d, _ in log if k == "advance")
    assert adv["d01"] == 1 and adv["d02"] == 3 and adv["d03"] == 9
    # every advance of a parent is immediately followed by force(s) of its children
    for idx, (k, d, _) in enumerate(log):
        if k == "advance" and d == "d01":
            assert log[idx + 1][0] == "force" and log[idx + 1][1] == "d02"


def test_run_host_tower_drives_correct_advance_counts():
    tower = _tower()
    calls = {"d01": 0, "d02": 0, "d03": 0}
    forces = {"d02": 0, "d03": 0}

    def advance(name, st, local):
        calls[name] += 1
        return st

    def force(child, parent_st, child_st):
        forces[child] += 1
        return child_st

    S.run_host_tower(tower, {"d01": 0, "d02": 0, "d03": 0}, root_steps=2,
                     advance=advance, force=force)
    assert calls == {"d01": 2, "d02": 6, "d03": 18}
    # force fires once per parent step per child: d02 = 2 (d01 steps), d03 = 6 (d02 steps)
    assert forces == {"d02": 2, "d03": 6}


def test_runtime_hook_spec_documents_no_inloop_transfer():
    spec = S.runtime_hook_spec()
    assert "ZERO host transfer" in spec
    assert "build_child_boundary_package" in spec
    assert "update_cadence_s = parent_dt" in spec
