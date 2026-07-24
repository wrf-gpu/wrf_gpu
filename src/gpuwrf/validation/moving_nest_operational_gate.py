"""v0.23 G2 OPERATIONAL moving-nest gate: real driver, real dycore, CPU oracles.

Runs the headline G2 oracles end-to-end on the live operational runtime and
emits a JSON proof object with the measured numbers (the pytest suite
``tests/test_v023_moving_nest_operational.py`` asserts the same properties;
this gate records the quantities).

CPU-only: run with ``JAX_PLATFORMS=cpu``.  Usage::

    python -m gpuwrf.validation.moving_nest_operational_gate <output.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _bitwise_overlap(old: np.ndarray, new: np.ndarray, shift: int) -> bool:
    return bool(np.array_equal(new[..., : new.shape[-1] - shift], old[..., shift:]))


def run_gate(*, output: str | Path | None = None) -> dict[str, Any]:
    from gpuwrf.contracts.grid import DomainNest
    from gpuwrf.nesting.adaptive_timestep import AdaptiveTimeStepConfig
    from gpuwrf.nesting.adaptive_driver import AdaptiveDtDriver
    from gpuwrf.nesting.boundary_construction import interp_parent_field_to_child
    from gpuwrf.nesting.moving_driver import (
        MovingNestConfig,
        MovingNestDriver,
        MovingNestError,
        PrescribedMove,
        vortex_center_from_state,
    )
    from gpuwrf.runtime.domain_tree import run_operational_domain_tree
    from gpuwrf.runtime.operational_mode import _initial_carry_for_run
    from gpuwrf.validation.moving_nest_testbed import (
        build_nested_pair,
        cell_center_coords_m,
        gaussian_low,
    )

    ratio = 3
    payload: dict[str, Any] = {
        "schema": "gpuwrf.v023.moving_nest_operational_gate",
        "schema_version": 1,
        "wrf_reference": {
            "root": "<USER_HOME>/src/wrf_pristine",
            "moving_driver": "share/mediation_nest_move.F::med_nest_move/time_for_move2",
            "state_shift": "dyn_em/shift_domain_em.F (registry-wide array shift)",
            "exposed_fill": "nest re-initialization interpolation from parent (SINT registration)",
            "adaptive_dt": "dyn_em/adapt_timestep_em.F::adapt_timestep/calc_dt",
            "corral": "namelist corral_dist",
        },
        "transfer_audit": {
            "gpu_used": False,
            "host_device_transfer_inside_timestep_loop": False,
            "note": "CPU-lane gate; moves are host choreography (WRF-faithful), "
            "vortex tracker reads 3 fields at vortex_interval cadence only.",
        },
    }
    checks: dict[str, Any] = {}

    # ---- 1. unit move on the live OperationalCarry --------------------------
    tree = build_nested_pair(
        u0_m_s=5.0, theta_anomaly=gaussian_low(x0_m=30000.0, y0_m=36000.0, radius_m=9000.0)
    )
    driver = MovingNestDriver.for_tree(
        tree,
        [MovingNestConfig(child="d02", prescribed_moves=(PrescribedMove(1, 1, 0),), corral_dist=1)],
    )
    carries = {
        name: _initial_carry_for_run(bundle.state, bundle.namelist)
        for name, bundle in tree.domains.items()
    }
    edge = tree.edges["d01"][0]
    new_edge, moved = driver.make_move_fn()(edge, carries["d01"], carries["d02"], 1)
    child, parent = carries["d02"], carries["d01"]
    overlap = {
        name: _bitwise_overlap(
            np.asarray(getattr(child.state, name)), np.asarray(getattr(moved.state, name)), ratio
        )
        for name in ("theta", "u", "v", "w", "qv", "p_total", "ph_perturbation", "mu_total")
    }
    scratch = {
        name: _bitwise_overlap(np.asarray(getattr(child, name)), np.asarray(getattr(moved, name)), ratio)
        for name in ("ww", "rthraten", "u_save", "t_2ave", "muts")
    }
    exposed = {}
    for name, stagger in (("theta", "mass"), ("u", "u"), ("v", "v")):
        fill = np.asarray(
            interp_parent_field_to_child(getattr(parent.state, name), new_edge.weights, staggering=stagger)
        )
        new = np.asarray(getattr(moved.state, name))
        exposed[name] = bool(np.array_equal(new[..., -ratio:], fill[..., -ratio:]))
    checks["unit_move"] = {
        "start": [5, 5],
        "moved_start": [int(new_edge.spec.i_parent_start), int(new_edge.spec.j_parent_start)],
        "overlap_bitwise": overlap,
        "carry_scratch_overlap_bitwise": scratch,
        "exposed_equals_parent_interp_bitwise": exposed,
        "force_weights_rebuilt": new_edge.weights is not edge.weights,
        "feedback_weights_rebuilt": new_edge.feedback_weights is not edge.feedback_weights,
    }

    # ---- 2. conservation across the move ------------------------------------
    old_mu = np.asarray(child.state.mu_total, dtype=np.float64)
    new_mu = np.asarray(moved.state.mu_total, dtype=np.float64)
    old_th = np.asarray(child.state.theta, dtype=np.float64)
    new_th = np.asarray(moved.state.theta, dtype=np.float64)
    mass_old = float(old_mu[:, ratio:].sum())
    mass_new = float(new_mu[:, : new_mu.shape[1] - ratio].sum())
    energy_old = float((old_mu[None, :, ratio:] * old_th[:, :, ratio:]).sum())
    energy_new = float(
        (new_mu[None, :, : new_mu.shape[1] - ratio] * new_th[:, :, : new_th.shape[2] - ratio]).sum()
    )
    mu0 = float(old_mu[0, 0])
    checks["conservation"] = {
        "overlap_dry_mass_delta_pa": mass_new - mass_old,
        "overlap_mu_theta_energy_delta": energy_new - energy_old,
        "total_dry_mass_rel_err_vs_uniform_analytic": abs(float(new_mu.sum()) - mu0 * new_mu.size)
        / (mu0 * new_mu.size),
    }

    # ---- 3. linear-plane exposed-fill exactness ------------------------------
    a_k, bx, cy = 1.0, 2.0e-5, -1.0e-5

    def plane(x2d, y2d):
        return a_k + bx * x2d + cy * y2d

    tree_lin = build_nested_pair(u0_m_s=0.0, theta_anomaly=plane)
    driver_lin = MovingNestDriver.for_tree(
        tree_lin,
        [MovingNestConfig(child="d02", prescribed_moves=(PrescribedMove(1, 1, 0),), corral_dist=1)],
    )
    carries_lin = {
        name: _initial_carry_for_run(bundle.state, bundle.namelist)
        for name, bundle in tree_lin.domains.items()
    }
    new_edge_lin, moved_lin = driver_lin.make_move_fn()(
        tree_lin.edges["d01"][0], carries_lin["d01"], carries_lin["d02"], 1
    )
    xs = cell_center_coords_m(
        n=24, dx_m=1000.0, parent_dx_m=3000.0,
        parent_start_1based=new_edge_lin.spec.i_parent_start, ratio=ratio,
    )
    ys = cell_center_coords_m(
        n=21, dx_m=1000.0, parent_dx_m=3000.0,
        parent_start_1based=new_edge_lin.spec.j_parent_start, ratio=ratio,
    )
    x2d, y2d = np.meshgrid(xs, ys)
    analytic = 300.0 + plane(x2d, y2d)
    got = np.asarray(moved_lin.state.theta)
    checks["linear_field_exposed_fill_max_abs_err_k"] = float(
        np.max(np.abs(got[:, :, -ratio:] - np.broadcast_to(analytic, got.shape)[:, :, -ratio:]))
    )

    # ---- 4. zero-move bit-identity + full-run finiteness ---------------------
    baseline = run_operational_domain_tree(tree, root_steps=2, block_between=False)
    driver_zero = MovingNestDriver.for_tree(
        tree, [MovingNestConfig(child="d02", prescribed_moves=(), corral_dist=1)]
    )
    zero = run_operational_domain_tree(
        tree, root_steps=2, move=driver_zero.make_move_fn(), block_between=False
    )
    identical = all(
        np.array_equal(
            np.asarray(getattr(baseline.states[name], field)),
            np.asarray(getattr(zero.states[name], field)),
        )
        for name in ("d01", "d02")
        for field in ("theta", "u", "v", "w", "qv", "p_total", "ph_total", "mu_total")
    )
    driver_run = MovingNestDriver.for_tree(
        tree, [MovingNestConfig(child="d02", prescribed_moves=(PrescribedMove(2, 1, 0),), corral_dist=1)]
    )
    moved_run = run_operational_domain_tree(
        tree, root_steps=3, move=driver_run.make_move_fn(), block_between=False
    )
    checks["zero_move_bit_identical_to_static_path"] = bool(identical)
    checks["full_run_with_move"] = {
        "move_events": [list(e) for e in moved_run.events if e[0] == "move"],
        "finite": {
            name: bool(np.isfinite(np.asarray(moved_run.states[name].theta)).all())
            for name in ("d01", "d02")
        },
        "child_theta_range_k": [
            float(np.asarray(moved_run.states["d02"].theta).min()),
            float(np.asarray(moved_run.states["d02"].theta).max()),
        ],
    }

    # ---- 5. vortex tracker on a real hydrostatic state -----------------------
    x0, y0 = 24000.0, 21000.0
    tree_v = build_nested_pair(theta_anomaly=gaussian_low(x0_m=x0, y0_m=y0, radius_m=6000.0))
    vi, vj = vortex_center_from_state(tree_v.domains["d02"].state, level_pa=50000.0)
    xs_v = cell_center_coords_m(n=24, dx_m=1000.0, parent_dx_m=3000.0, parent_start_1based=5, ratio=ratio)
    ys_v = cell_center_coords_m(n=21, dx_m=1000.0, parent_dx_m=3000.0, parent_start_1based=5, ratio=ratio)
    checks["vortex_tracker_real_state"] = {
        "true_center_m": [x0, y0],
        "found_center_m": [float(xs_v[int(vi) - 1]), float(ys_v[int(vj) - 1])],
        "abs_err_m": [abs(float(xs_v[int(vi) - 1]) - x0), abs(float(ys_v[int(vj) - 1]) - y0)],
    }

    # ---- 6. adaptive dt on the live tree --------------------------------------
    tree_dt = build_nested_pair(
        u0_m_s=10.0, theta_anomaly=gaussian_low(x0_m=30000.0, y0_m=36000.0, radius_m=9000.0)
    )
    adriver = AdaptiveDtDriver(
        tree_dt,
        AdaptiveTimeStepConfig(
            target_cfl=1.2, target_hcfl=0.84, min_time_step_s=1.0,
            max_time_step_s=24.0, max_step_increase_pct=20.0, precision=10,
        ),
    )
    res_dt = run_operational_domain_tree(
        tree_dt, root_steps=3, adaptive_dt=adriver.make_adaptive_dt_fn(), block_between=False
    )
    checks["adaptive_dt"] = {
        "dt_log": adriver.dt_log,
        "final_parent_dt_s": float(tree_dt.domains["d01"].namelist.dt_s),
        "final_child_dt_s": float(tree_dt.domains["d02"].namelist.dt_s),
        "final_child_boundary_cadence_s": float(
            tree_dt.domains["d02"].namelist.boundary_config.update_cadence_s
        ),
        "finite": {
            name: bool(np.isfinite(np.asarray(res_dt.states[name].theta)).all())
            for name in ("d01", "d02")
        },
    }

    # ---- 7. fail-closed named reasons -----------------------------------------
    reasons = []
    for build in (
        lambda: MovingNestDriver(
            geometry={"d02": type("G", (), {"nx": 9, "ny": 9})()},
            configs=[MovingNestConfig(child="d02", mode="teleport")],
        ),
        lambda: MovingNestDriver.for_tree(
            tree, [MovingNestConfig(child="d02", prescribed_moves=(), corral_dist=8)]
        ),
        lambda: MovingNestDriver(
            geometry={"d02": type("G", (), {"nx": 9, "ny": 9})()},
            configs=[MovingNestConfig(child="d02", prescribed_moves=(PrescribedMove(1, 2, 0),))],
        ),
    ):
        try:
            build()
            reasons.append(None)
        except MovingNestError as exc:
            reasons.append(str(exc))
    checks["fail_closed_named_reasons"] = reasons

    ok = (
        all(overlap.values())
        and all(scratch.values())
        and all(exposed.values())
        and checks["unit_move"]["force_weights_rebuilt"]
        and checks["unit_move"]["feedback_weights_rebuilt"]
        and checks["conservation"]["overlap_dry_mass_delta_pa"] == 0.0
        and checks["conservation"]["overlap_mu_theta_energy_delta"] == 0.0
        and checks["conservation"]["total_dry_mass_rel_err_vs_uniform_analytic"] < 1.0e-14
        and checks["linear_field_exposed_fill_max_abs_err_k"] < 1.0e-9
        and checks["zero_move_bit_identical_to_static_path"]
        and all(checks["full_run_with_move"]["finite"].values())
        and max(checks["vortex_tracker_real_state"]["abs_err_m"]) <= 1500.0
        and all(checks["adaptive_dt"]["finite"].values())
        and all(reason is not None for reason in reasons)
    )
    payload["checks"] = checks
    payload["verdict"] = "PASS" if ok else "FAIL"
    payload["argv"] = sys.argv

    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("GPUWRF_NESTED_AOT", "0")
    os.environ.setdefault("GPUWRF_NESTED_FUSE", "0")
    out = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_gate(output=out)
    print(json.dumps({"verdict": result["verdict"]}, indent=2))
    sys.exit(0 if result["verdict"] == "PASS" else 1)
