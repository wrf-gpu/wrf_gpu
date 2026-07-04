"""v0.23 G2 moving-nest + adaptive-dt OPERATIONAL CPU gates.

These gates exercise the REAL moving-nest driver
(:mod:`gpuwrf.nesting.moving_driver`) against the live operational runtime --
full RK3 + acoustic dycore, live nested forcedown, per-move weight rebuild --
on the flat-terrain idealized testbed
(:mod:`gpuwrf.validation.moving_nest_testbed`).  CPU-only by design (the G2
brief pins the GPU to another lane); every check is one of the brief's oracles:

* bit-level move semantics (overlap carried bitwise, exposed cells re-derived
  from the parent bitwise-vs-direct-interp);
* conservation of dry mass / energy proxies across a move;
* an exact analytic advection-through-moves oracle (CFL=1 upwind translation is
  the exact PDE solution; the moved-window field must match the analytic field);
* vortex-following (analytic feature tracking + the WRF z500 tracker on a real
  hydrostatic state);
* zero-move runs stay BIT-IDENTICAL to the static-nest path;
* unsupported configurations fail closed with named reasons;
* adaptive-dt drives the live tree from state CFL diagnostics and re-pins the
  child dt/boundary cadence.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
os.environ.setdefault("GPUWRF_NESTED_AOT", "0")
os.environ.setdefault("GPUWRF_NESTED_FUSE", "0")

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import jax.numpy as jnp

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.nesting.adaptive_timestep import AdaptiveTimeStepConfig
from gpuwrf.nesting.adaptive_driver import AdaptiveDtDriver, cfl_diagnostics_from_state
from gpuwrf.nesting.boundary_construction import interp_parent_field_to_child
from gpuwrf.nesting.moving_driver import (
    MovingNestConfig,
    MovingNestDriver,
    MovingNestError,
    PrescribedMove,
    vortex_center_from_state,
)
from gpuwrf.runtime.domain_tree import (
    DomainEdge,
    run_domain_tree_callbacks,
    run_operational_domain_tree,
)
from gpuwrf.runtime.operational_mode import _initial_carry_for_run
from gpuwrf.validation.moving_nest_testbed import (
    build_nested_pair,
    cell_center_coords_m,
    gaussian_low,
)

RATIO = 3


def _tree(**kwargs):
    return build_nested_pair(
        u0_m_s=kwargs.pop("u0_m_s", 5.0),
        theta_anomaly=kwargs.pop(
            "theta_anomaly", gaussian_low(x0_m=30000.0, y0_m=36000.0, radius_m=9000.0)
        ),
        **kwargs,
    )


def _carries(tree):
    return {
        name: _initial_carry_for_run(bundle.state, bundle.namelist)
        for name, bundle in tree.domains.items()
    }


def _driver(tree, moves, **cfg_kwargs):
    return MovingNestDriver.for_tree(
        tree,
        [
            MovingNestConfig(
                child="d02",
                mode="prescribed",
                prescribed_moves=tuple(moves),
                corral_dist=cfg_kwargs.pop("corral_dist", 1),
                **cfg_kwargs,
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1. Bit-level move semantics on the REAL operational carry
# ---------------------------------------------------------------------------


def test_unit_move_bitwise_overlap_and_parent_fill():
    tree = _tree()
    driver = _driver(tree, [PrescribedMove(2, 1, 0)])
    move_fn = driver.make_move_fn()
    carries = _carries(tree)
    edge = tree.edges["d01"][0]

    assert move_fn(edge, carries["d01"], carries["d02"], 1) is None  # not yet

    result = move_fn(edge, carries["d01"], carries["d02"], 2)
    assert isinstance(result, tuple)
    new_edge, moved = result
    assert (edge.spec.i_parent_start, new_edge.spec.i_parent_start) == (5, 6)
    assert new_edge.spec.j_parent_start == 5

    child, parent = carries["d02"], carries["d01"]
    shift = RATIO  # +1 parent cell * ratio fine cells along x

    # Overlap carried BITWISE for prognostics, staggered winds, 2-D mass, and
    # the persistent cross-step carry scratch (WRF shift_domain_em semantics).
    for name in ("theta", "qv", "w", "p_total", "ph_perturbation"):
        old = np.asarray(getattr(child.state, name))
        new = np.asarray(getattr(moved.state, name))
        assert np.array_equal(new[..., : old.shape[-1] - shift], old[..., shift:]), name
    old_u, new_u = np.asarray(child.state.u), np.asarray(moved.state.u)
    assert np.array_equal(new_u[..., : old_u.shape[-1] - shift], old_u[..., shift:])
    old_v, new_v = np.asarray(child.state.v), np.asarray(moved.state.v)
    assert np.array_equal(new_v[..., : old_v.shape[-1] - shift], old_v[..., shift:])
    old_mu, new_mu = np.asarray(child.state.mu_total), np.asarray(moved.state.mu_total)
    assert np.array_equal(new_mu[..., : old_mu.shape[-1] - shift], old_mu[..., shift:])
    for name in ("ww", "rthraten", "u_save", "t_2ave", "muts"):
        old = np.asarray(getattr(child, name))
        new = np.asarray(getattr(moved, name))
        assert np.array_equal(new[..., : old.shape[-1] - shift], old[..., shift:]), name

    # Newly exposed cells are the parent re-interpolation at the NEW position,
    # bitwise identical to calling the forcedown interp operator directly.
    for name, stagger in (("theta", "mass"), ("u", "u"), ("v", "v")):
        fill = np.asarray(
            interp_parent_field_to_child(
                getattr(parent.state, name), new_edge.weights, staggering=stagger
            )
        )
        new = np.asarray(getattr(moved.state, name))
        assert np.array_equal(new[..., -shift:], fill[..., -shift:]), name

    # Gather plans were REBUILT for the new position (not the stale ones).
    assert new_edge.weights is not edge.weights
    assert new_edge.feedback_weights is not edge.feedback_weights
    got = np.asarray(new_edge.weights.mass.x0[:3])
    want = np.asarray(
        tree.edges["d01"][0].weights.mass.x0[:3]
    )  # old plan, old start
    assert not np.array_equal(got, want)

    assert driver.move_log and driver.move_log[0]["outcome"] == "moved"


def test_move_conserves_dry_mass_and_energy_proxies():
    tree = _tree()
    driver = _driver(tree, [PrescribedMove(1, 1, 0)])
    carries = _carries(tree)
    edge = tree.edges["d01"][0]
    _, moved = driver.make_move_fn()(edge, carries["d01"], carries["d02"], 1)

    child = carries["d02"]
    shift = RATIO
    old_mu = np.asarray(child.state.mu_total, dtype=np.float64)
    new_mu = np.asarray(moved.state.mu_total, dtype=np.float64)
    old_th = np.asarray(child.state.theta, dtype=np.float64)
    new_th = np.asarray(moved.state.theta, dtype=np.float64)

    # (a) Overlap dry-mass and mu*theta energy-proxy integrals conserved EXACTLY
    # (the overlap is a pure roll -- bit-identical values, identical sums).
    assert float(new_mu[:, :-shift].sum() if new_mu.ndim == 2 else 0) or True
    np.testing.assert_array_equal(new_mu[:, : new_mu.shape[1] - shift], old_mu[:, shift:])
    overlap_mass_old = old_mu[:, shift:].sum()
    overlap_mass_new = new_mu[:, : new_mu.shape[1] - shift].sum()
    assert overlap_mass_new == overlap_mass_old
    overlap_energy_old = (old_mu[None, :, shift:] * old_th[:, :, shift:]).sum()
    overlap_energy_new = (
        new_mu[None, :, : new_mu.shape[1] - shift] * new_th[:, :, : new_th.shape[2] - shift]
    ).sum()
    assert overlap_energy_new == overlap_energy_old

    # (b) The uniform-mu column mass is reproduced exactly by the exposed-cell
    # re-interpolation (interpolation of a constant is exact): total child dry
    # mass after the move equals the uniform analytic total to fp roundoff.
    mu0 = float(old_mu[0, 0])
    total = float(new_mu.sum())
    assert abs(total - mu0 * new_mu.size) / (mu0 * new_mu.size) < 1.0e-14


def test_exposed_fill_reproduces_linear_fields_exactly():
    # SINT cell-centered registration reproduces a linear (plane) field exactly;
    # with theta' = a + b*x + c*y the re-derived exposed cells must match the
    # analytic plane -- conservation-grade re-interpolation, no smoothing loss.
    a_k, bx_k_per_m, cy_k_per_m = 1.0, 2.0e-5, -1.0e-5

    def plane(x2d, y2d):
        return a_k + bx_k_per_m * x2d + cy_k_per_m * y2d

    tree = build_nested_pair(u0_m_s=0.0, theta_anomaly=plane)
    driver = _driver(tree, [PrescribedMove(1, 1, 0)])
    carries = _carries(tree)
    edge = tree.edges["d01"][0]
    new_edge, moved = driver.make_move_fn()(edge, carries["d01"], carries["d02"], 1)

    child_nx = int(tree.domains["d02"].grid.nx)
    child_ny = int(tree.domains["d02"].grid.ny)
    xs = cell_center_coords_m(
        n=child_nx, dx_m=1000.0, parent_dx_m=3000.0,
        parent_start_1based=new_edge.spec.i_parent_start, ratio=RATIO,
    )
    ys = cell_center_coords_m(
        n=child_ny, dx_m=1000.0, parent_dx_m=3000.0,
        parent_start_1based=new_edge.spec.j_parent_start, ratio=RATIO,
    )
    x2d, y2d = np.meshgrid(xs, ys)
    analytic = 300.0 + plane(x2d, y2d)
    new_th = np.asarray(moved.state.theta)
    np.testing.assert_allclose(new_th[:, :, -RATIO:], np.broadcast_to(analytic, new_th.shape)[:, :, -RATIO:], rtol=0.0, atol=1.0e-9)


# ---------------------------------------------------------------------------
# 2. Zero-move bit-identity with the static path + full-run integration
# ---------------------------------------------------------------------------


def test_zero_move_run_bit_identical_to_static_path():
    tree = _tree()
    baseline = run_operational_domain_tree(tree, root_steps=2, block_between=False)
    driver = _driver(tree, [])  # configured but never moves
    with_driver = run_operational_domain_tree(
        tree, root_steps=2, move=driver.make_move_fn(), block_between=False
    )
    assert not driver.move_log
    for name in ("d01", "d02"):
        base = baseline.states[name]
        got = with_driver.states[name]
        for field in ("theta", "u", "v", "w", "qv", "p_total", "ph_total", "mu_total"):
            assert np.array_equal(
                np.asarray(getattr(base, field)), np.asarray(getattr(got, field))
            ), (name, field)


def test_full_run_with_move_stays_finite_and_records_move():
    tree = _tree()
    driver = _driver(tree, [PrescribedMove(2, 1, 0)])
    result = run_operational_domain_tree(
        tree, root_steps=3, move=driver.make_move_fn(), block_between=False
    )
    moves = [event for event in result.events if event[0] == "move"]
    assert moves == [("move", "d01", "d02", 2, 5, 5, 6, 5)]
    for name in ("d01", "d02"):
        theta = np.asarray(result.states[name].theta)
        assert np.isfinite(theta).all(), name
        assert 280.0 < theta.min() and theta.max() < 320.0, name
    assert driver.move_log[0]["to"] == (6, 5)


def test_moving_with_two_way_feedback_stays_finite():
    tree = _tree(feedback_enabled=True)
    driver = _driver(tree, [PrescribedMove(1, 1, 0)])
    result = run_operational_domain_tree(
        tree,
        root_steps=2,
        move=driver.make_move_fn(),
        feedback_enabled=True,
        block_between=False,
    )
    kinds = [event[0] for event in result.events]
    assert "feedback" in kinds and "move" in kinds
    for name in ("d01", "d02"):
        assert np.isfinite(np.asarray(result.states[name].theta)).all(), name


def test_feedback_after_move_uses_rebuilt_edge_not_stale_weights():
    # Callback-level pin of the resolve_edge fix: after `move` returns a
    # repositioned edge with REBUILT weights, the feedback callback must receive
    # THAT edge -- the pre-v0.23 fallback paired the moved spec with the static
    # tree's stale weights.
    hierarchy = DomainHierarchy.from_edges(
        ("d01", "d02"), (DomainNest("d01", "d02", 3, 4, 4),)
    )
    static_weights = object()
    moved_weights = object()
    static_edge = DomainEdge(DomainNest("d01", "d02", 3, 4, 4), weights=static_weights)
    moved_edge = DomainEdge(DomainNest("d01", "d02", 3, 5, 4), weights=moved_weights)
    seen: list[DomainEdge] = []

    def move(edge, parent, child, parent_step):
        return (moved_edge, child) if parent_step == 1 else None

    def feedback(edge, parent, child):
        seen.append(edge)
        return parent

    run_domain_tree_callbacks(
        hierarchy,
        {"d01": SimpleNamespace(), "d02": SimpleNamespace()},
        root_steps=2,
        advance=lambda name, carry, start, n: carry,
        move=move,
        feedback=feedback,
        feedback_enabled=True,
        edge_lookup=lambda spec: static_edge,
        block_between=False,
    )
    assert len(seen) == 2
    assert seen[0].weights is moved_weights
    assert seen[1].weights is moved_weights


# ---------------------------------------------------------------------------
# 3. Analytic advection-through-moves oracle (exact CFL=1 upwind translation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AdvCarry:
    theta: jnp.ndarray

    def replace(self, **updates) -> "_AdvCarry":
        return _AdvCarry(updates.get("theta", self.theta))


def _run_analytic_advection(field_fn, *, moves=None, tracker=None, root_steps=6):
    """Drive the REAL MovingNestDriver with an exact advection stepper.

    CFL=1 upwind advection is the EXACT solution of the 1-D advection PDE: each
    child step translates the field one child cell in +x, each parent step one
    parent cell, with the inflow column evaluated analytically.  The nest window
    moves (prescribed or feature-tracked); at every time the field on the
    CURRENT window must equal the analytic solution F(x - c t).
    """

    parent_nx, parent_ny = 46, 20
    child_nx, child_ny = 24, 15
    dxp = 3000.0
    dxc = dxp / RATIO
    dtp = 60.0
    speed = dxp / dtp  # exactly one parent cell per parent step
    i0, j0 = 8, 3
    geometry = {
        "d01": SimpleNamespace(nx=parent_nx, ny=parent_ny),
        "d02": SimpleNamespace(nx=child_nx, ny=child_ny),
    }
    config = MovingNestConfig(
        child="d02",
        mode="prescribed" if tracker is None else "vortex_following",
        prescribed_moves=tuple(moves or ()),
        move_interval_steps=1,
        corral_dist=1,
        tracker=tracker,
        search_radius_child_cells=6.0,
    )
    driver = MovingNestDriver(geometry=geometry, configs=[config])
    initial_spec = DomainNest("d01", "d02", RATIO, i0, j0)

    def window_coords(spec):
        xs = cell_center_coords_m(
            n=child_nx, dx_m=dxc, parent_dx_m=dxp,
            parent_start_1based=spec.i_parent_start, ratio=RATIO,
        )
        ys = cell_center_coords_m(
            n=child_ny, dx_m=dxc, parent_dx_m=dxp,
            parent_start_1based=spec.j_parent_start, ratio=RATIO,
        )
        return np.meshgrid(xs, ys)

    def current_spec():
        edge = driver._edges.get(("d01", "d02"))
        return edge.spec if edge is not None else initial_spec

    parent_x = np.arange(parent_nx) * dxp
    parent_y = np.arange(parent_ny) * dxp
    px2d, py2d = np.meshgrid(parent_x, parent_y)

    def sample(x2d, y2d, t_s):
        return np.asarray(field_fn(x2d - speed * t_s, y2d), dtype=np.float64)

    x2d0, y2d0 = window_coords(initial_spec)
    carries = {
        "d01": _AdvCarry(jnp.asarray(sample(px2d, py2d, 0.0)[None])),
        "d02": _AdvCarry(jnp.asarray(sample(x2d0, y2d0, 0.0)[None])),
    }

    def advance(name, carry, start_step, n_steps):
        theta = np.asarray(carry.theta)
        if name == "d01":
            dt, x_axis, y_axis = dtp, parent_x, parent_y
        else:
            dt = dtp / RATIO
            spec = current_spec()
            x2d, y2d = window_coords(spec)
            x_axis, y_axis = x2d[0, :], y2d[:, 0]
        for local in range(int(n_steps)):
            t_new = (int(start_step) + local) * dt
            theta = np.roll(theta, 1, axis=-1)
            xx, yy = np.meshgrid(np.asarray([x_axis[0]]), y_axis)
            theta[..., :, 0:1] = sample(xx, yy, t_new)[None]
        return _AdvCarry(jnp.asarray(theta))

    hierarchy = DomainHierarchy.from_edges(
        ("d01", "d02"), (initial_spec,)
    )
    result = run_domain_tree_callbacks(
        hierarchy,
        carries,
        root_steps=root_steps,
        advance=advance,
        move=driver.make_move_fn(),
        block_between=False,
    )
    final_spec = current_spec()
    t_final = root_steps * dtp
    x2d, y2d = window_coords(final_spec)
    analytic = sample(x2d, y2d, t_final)[None]
    return np.asarray(result.carries["d02"].theta), analytic, driver, final_spec


def test_analytic_linear_field_through_moves_is_machine_exact():
    def linear(x, y):
        return 250.0 + 3.0e-4 * x + 1.0e-4 * y

    moves = [PrescribedMove(k, 1, 0) for k in range(1, 7)]
    got, analytic, driver, final_spec = _run_analytic_advection(linear, moves=moves)
    assert len(driver.move_log) == 6
    assert final_spec.i_parent_start == 8 + 6
    np.testing.assert_allclose(got, analytic, rtol=0.0, atol=1.0e-9)


def test_analytic_vortex_following_tracks_the_feature():
    radius = 5000.0

    def gaussian(x, y):
        x0, y0 = (8 - 1 + (24 / 2 - RATIO // 2) / RATIO) * 3000.0, (3 - 1 + (15 / 2 - RATIO // 2) / RATIO) * 3000.0
        # feature initially centered in the child window
        return 300.0 - 5.0 * np.exp(-(((x - x0) ** 2) + ((y - y0) ** 2)) / (2.0 * radius**2))

    def tracker(state):
        theta = np.asarray(state.theta)[0]
        j, i = np.unravel_index(np.argmin(theta), theta.shape)
        return float(i + 1), float(j + 1)

    got, analytic, driver, final_spec = _run_analytic_advection(
        gaussian, tracker=tracker, root_steps=6
    )
    # The feature moves +1 parent cell per parent step; the tracked nest must
    # follow move-for-move and the moved-window field must match the analytic
    # translated Gaussian (roll-exact interior; exposed cells parent-derived).
    applied = [entry["applied"] for entry in driver.move_log if entry["outcome"] == "moved"]
    assert applied and all(move == (1, 0) for move in applied)
    assert final_spec.i_parent_start == 8 + len(applied)
    np.testing.assert_allclose(got, analytic, rtol=0.0, atol=5.0e-2)
    # exact in the never-exposed interior:
    np.testing.assert_allclose(got[..., : -RATIO * 2], analytic[..., : -RATIO * 2], rtol=0.0, atol=1.0e-9)


def test_vortex_tracker_finds_low_on_real_hydrostatic_state():
    # A cold column anomaly hydrostatically depresses the 500-hPa surface; the
    # WRF time_for_move2-style tracker must find it from p_total/ph_total.
    # Feature placed inside the tracker's search interior (the outer 5-cell
    # forced ring is excluded from the search, exactly as WRF never tracks into
    # the specified frame): child window spans x,y in [11,34]x[11,31] km.
    x0, y0 = 24000.0, 21000.0
    tree = build_nested_pair(theta_anomaly=gaussian_low(x0_m=x0, y0_m=y0, radius_m=6000.0))
    state = tree.domains["d02"].state
    i, j = vortex_center_from_state(state, level_pa=50000.0, exclude_ring_cells=5)
    xs = cell_center_coords_m(n=24, dx_m=1000.0, parent_dx_m=3000.0, parent_start_1based=5, ratio=RATIO)
    ys = cell_center_coords_m(n=21, dx_m=1000.0, parent_dx_m=3000.0, parent_start_1based=5, ratio=RATIO)
    assert abs(xs[int(i) - 1] - x0) <= 1500.0  # within 1.5 child cells
    assert abs(ys[int(j) - 1] - y0) <= 1500.0


# ---------------------------------------------------------------------------
# 4. Fail-closed named reasons
# ---------------------------------------------------------------------------


def test_fail_closed_unknown_mode_and_bad_prescriptions():
    with pytest.raises(MovingNestError, match="unknown mode"):
        MovingNestDriver(
            geometry={"d02": SimpleNamespace(nx=9, ny=9)},
            configs=[MovingNestConfig(child="d02", mode="teleport")],
        )
    with pytest.raises(MovingNestError, match="strictly increasing"):
        MovingNestDriver(
            geometry={"d02": SimpleNamespace(nx=9, ny=9)},
            configs=[
                MovingNestConfig(
                    child="d02",
                    prescribed_moves=(PrescribedMove(2, 1, 0), PrescribedMove(2, 0, 1)),
                )
            ],
        )
    with pytest.raises(MovingNestError, match="one parent cell"):
        MovingNestDriver(
            geometry={"d02": SimpleNamespace(nx=9, ny=9)},
            configs=[
                MovingNestConfig(child="d02", prescribed_moves=(PrescribedMove(1, 2, 0),))
            ],
        )
    with pytest.raises(MovingNestError, match="no geometry"):
        MovingNestDriver(geometry={}, configs=[MovingNestConfig(child="d02")])


def test_fail_closed_non_flat_terrain():
    tree = _tree()
    bumpy = np.zeros((int(tree.domains["d01"].grid.ny), int(tree.domains["d01"].grid.nx)))
    bumpy[3, 3] = 100.0
    import dataclasses

    grid = dataclasses.replace(tree.domains["d01"].grid, terrain_height=jnp.asarray(bumpy))
    tree.domains["d01"] = dataclasses.replace(tree.domains["d01"], grid=grid)
    with pytest.raises(MovingNestError, match="non-flat terrain"):
        MovingNestDriver.for_tree(
            tree, [MovingNestConfig(child="d02", prescribed_moves=(), corral_dist=1)]
        )


def test_fail_closed_corral_violation_at_initial_placement():
    tree = _tree()
    with pytest.raises(MovingNestError, match="corral_dist"):
        MovingNestDriver.for_tree(
            tree,
            [MovingNestConfig(child="d02", prescribed_moves=(), corral_dist=8)],
        )


def test_fail_closed_active_land_carry():
    tree = _tree()
    driver = _driver(tree, [PrescribedMove(1, 1, 0)])
    carries = _carries(tree)
    bad_child = carries["d02"].replace(noahmp_land=object())
    with pytest.raises(MovingNestError, match="noahmp_land"):
        driver.make_move_fn()(tree.edges["d01"][0], carries["d01"], bad_child, 1)


def test_corral_blocks_move_at_the_fence_without_error():
    tree = _tree()
    driver = _driver(tree, [PrescribedMove(1, -1, 0)], corral_dist=4)
    # start i=5, span ceil(24/3)=8 -> corral_dist=4 allows i in [5, 30-8-4+1=19]:
    # a -1 move would leave the corral -> blocked, run continues unmoved.
    carries = _carries(tree)
    result = driver.make_move_fn()(tree.edges["d01"][0], carries["d01"], carries["d02"], 1)
    assert result is None
    assert driver.move_log[0]["outcome"] == "corral_blocked"


# ---------------------------------------------------------------------------
# 5. Adaptive-dt on the live operational tree
# ---------------------------------------------------------------------------


def test_adaptive_dt_drives_root_and_pins_child_cadence():
    tree = _tree(u0_m_s=10.0)
    dt0 = float(tree.domains["d01"].namelist.dt_s)
    config = AdaptiveTimeStepConfig(
        target_cfl=1.2,
        target_hcfl=0.84,
        min_time_step_s=1.0,
        max_time_step_s=24.0,
        max_step_increase_pct=20.0,
        precision=10,
    )
    driver = AdaptiveDtDriver(tree, config)
    result = run_operational_domain_tree(
        tree, root_steps=3, adaptive_dt=driver.make_adaptive_dt_fn(), block_between=False
    )
    dt_events = [event for event in result.events if event[0] == "dt"]
    assert len(dt_events) == 3
    log = driver.dt_log
    assert log[0]["dt_s"] == dt0  # WRF starting_time_step keeps the initial dt
    # CFL is far below target (u0=10, dx=3000, dt=6 -> hcfl ~0.02): dt must GROW,
    # capped at max_step_increase_pct per step.
    assert log[1]["dt_s"] > dt0
    assert log[1]["dt_s"] <= dt0 * 1.2 + 1.0e-12
    assert log[2]["dt_s"] <= log[1]["dt_s"] * 1.2 + 1.0e-12
    # The namelist write-through cascaded: child dt == parent dt / ratio and the
    # child boundary cadence == parent dt.
    parent_dt = float(tree.domains["d01"].namelist.dt_s)
    child_nl = tree.domains["d02"].namelist
    assert abs(float(child_nl.dt_s) - parent_dt / RATIO) < 1.0e-12
    assert abs(float(child_nl.boundary_config.update_cadence_s) - parent_dt) < 1.0e-12
    for name in ("d01", "d02"):
        assert np.isfinite(np.asarray(result.states[name].theta)).all(), name


def test_cfl_diagnostics_match_hand_computation():
    tree = _tree(u0_m_s=12.0)
    state = tree.domains["d01"].state
    horiz, vert = cfl_diagnostics_from_state(state, dt_s=6.0, dx_m=3000.0, dy_m=3000.0)
    assert abs(horiz - 12.0 / 3000.0 * 6.0) < 1.0e-9  # v=0, |u|=12 everywhere
    assert vert == 0.0  # w=0 initial state
