"""Two-way nesting feedback (feedback=1) — runtime 1-way vs 2-way validation.

Drives the GENUINE runtime feedback path -- :func:`run_domain_tree_callbacks`
with the operational :func:`gpuwrf.runtime.domain_tree._operational_feedback`
callback over real :class:`State` carries on a 2-domain (d01->d02, ratio 3) nest
-- twice:

  * one-way  : ``feedback_enabled=False`` (the v0.11.0/v0.12.0-validated wiring);
  * two-way  : ``feedback_enabled=True``  (child overlap fed back to the parent
               via WRF ``copy_fcn`` area-average + the ``sm121`` feedback-zone
               smoother).

The ``advance`` callback is the IDENTITY on the parent and a deterministic
structured stamp on the child, so the ONLY thing that can change the parent
interior between the two runs is the feedback operator.  This isolates and proves:

  (a) the parent INTERIOR is actually updated by the child (2-way != 1-way over
      the nest-overlap region, and ONLY there);
  (b) the result stays finite (no blow-up / NaN);
  (c) the fed-back overlap is consistent (copy_fcn area-average then sm121),
      checked against an explicit re-derivation + the WRF overlap conservation
      diagnostic on the raw (pre-smoother) feedback.

Run::

    JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-3 \
        python proofs/v0120/two_way_feedback_validation.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import (
    BCMetadata,
    DomainHierarchy,
    DomainNest,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling.boundary_feedback import (
    apply_feedback,
    feedback_mask,
    feedback_overlap_conservation,
    sm121_smooth,
)
from gpuwrf.runtime.domain_tree import (
    DomainBundle,
    DomainTree,
    _operational_feedback,
    run_domain_tree_callbacks,
)


jax.config.update("jax_enable_x64", True)

RATIO = 3
I_PARENT_START = 4
J_PARENT_START = 4
PARENT_NX, PARENT_NY = 20, 18
CHILD_NX, CHILD_NY = 24, 24  # 8x8 parent cells covered
NZ = 8


def _make_grid(nx: int, ny: int, dx_m: float) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, NZ + 1, dtype=jnp.float64)
    metrics = DycoreMetrics.flat(
        ny=ny, nx=nx, nz=NZ, eta_levels=eta, top_pressure_pa=5000.0,
        provenance="two-way-feedback-validation",
    )
    return GridSpec(
        projection=Projection("lambert", 40.0, -3.0, dx_m, dx_m, nx, ny),
        terrain=TerrainProvenance(
            source_path="validation:two-way-feedback",
            sha256="analytic",
            shape=(ny, nx),
            units="m",
            projection_transform="flat",
            max_elevation_m=0.0,
            coastline_sanity_check_passed=True,
        ),
        vertical=VerticalCoord("hybrid_eta", NZ, 5000.0, eta),
        bc=BCMetadata(
            source="ideal",
            fields=("u", "v", "w", "theta", "p", "ph", "mu"),
            update_cadence_h=999,
            interpolation="linear",
            restart_compatible=False,
        ),
        eta_levels=eta,
        terrain_height=jnp.zeros((ny, nx), dtype=jnp.float64),
        metrics=metrics,
        halo_width=2,
        staggering="c-grid",
    )


def _structured(shape: tuple[int, ...], seed: float) -> jnp.ndarray:
    """Deterministic smooth structured field (a tilted sinusoid) for one leaf."""
    idx = np.indices(shape, dtype=np.float64)
    val = np.zeros(shape, dtype=np.float64)
    for ax in range(len(shape)):
        val = val + np.sin((idx[ax] + seed) * (0.3 + 0.07 * ax))
    return jnp.asarray(val)


def _cpu_zeros_state(grid: GridSpec) -> State:
    """Allocate a zero ``State`` on the host (``State.zeros`` forces a GPU device)."""
    return State(
        **{
            field: jnp.zeros(shape, dtype=jnp.float64)
            for field, shape in _state_field_shapes(grid).items()
        }
    )


def _seed_state(grid: GridSpec, *, base: float, amp: float, seed: float) -> State:
    """A finite, structured State on ``grid`` (perturbations + matching totals)."""
    s = _cpu_zeros_state(grid)
    ny, nx, nz = grid.ny, grid.nx, grid.nz
    theta = base + amp * _structured((nz, ny, nx), seed)
    qv = 0.005 + 0.001 * _structured((nz, ny, nx), seed + 1.0)
    p_pert = amp * _structured((nz, ny, nx), seed + 2.0)
    ph_pert = amp * _structured((nz + 1, ny, nx), seed + 3.0)
    mu_pert = amp * _structured((ny, nx), seed + 4.0)
    u = amp * _structured((nz, ny, nx + 1), seed + 5.0)
    v = amp * _structured((nz, ny + 1, nx), seed + 6.0)
    w = amp * _structured((nz + 1, ny, nx), seed + 7.0)
    p_base = jnp.full((nz, ny, nx), 90000.0)
    ph_base = jnp.full((nz + 1, ny, nx), 100000.0)
    mu_base = jnp.full((ny, nx), 95000.0)
    return s.replace(
        _cast=False,
        u=u, v=v, w=w, theta=theta, qv=qv,
        p_perturbation=p_pert, p_total=p_base + p_pert, p=p_base + p_pert,
        ph_perturbation=ph_pert, ph_total=ph_base + ph_pert, ph=ph_base + ph_pert,
        mu_perturbation=mu_pert, mu_total=mu_base + mu_pert, mu=mu_base + mu_pert,
        qke=0.5 + 0.1 * _structured((nz, ny, nx), seed + 8.0),
    )


def _identity_advance(name, carry, start_step, n_steps):
    """No-op advance: isolate the feedback as the sole parent-mutating operator."""
    return carry


def _finite(state: State) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(getattr(state, f))))
        for f in ("u", "v", "w", "theta", "qv", "p_total", "ph_total", "mu_total")
    )


def main() -> dict:
    parent_grid = _make_grid(PARENT_NX, PARENT_NY, dx_m=9000.0)
    child_grid = _make_grid(CHILD_NX, CHILD_NY, dx_m=3000.0)

    hierarchy = DomainHierarchy.from_edges(
        ("d01", "d02"),
        (DomainNest("d01", "d02", RATIO, I_PARENT_START, J_PARENT_START),),
    )
    parent_state = _seed_state(parent_grid, base=300.0, amp=2.0, seed=0.0)
    child_state = _seed_state(child_grid, base=305.0, amp=4.0, seed=11.0)

    bundles = {
        "d01": DomainBundle("d01", parent_state, None, grid=parent_grid, metrics=parent_grid.metrics),
        "d02": DomainBundle("d02", child_state, None, grid=child_grid, metrics=child_grid.metrics),
    }
    tree = DomainTree.from_domains(hierarchy, bundles)
    edge = tree.children("d01")[0]
    fb_weights = edge.feedback_weights

    # Genuine runtime carries: the operational feedback callback reads carry.state.
    class _Carry:
        def __init__(self, state):
            self.state = state

        def replace(self, *, state):
            return _Carry(state)

    def edge_lookup(spec):
        return edge

    def run(feedback_enabled: bool) -> dict[str, State]:
        carries = {"d01": _Carry(parent_state), "d02": _Carry(child_state)}
        result = run_domain_tree_callbacks(
            hierarchy,
            carries,
            root_steps=1,
            advance=_identity_advance,
            force=None,
            feedback=_operational_feedback,
            feedback_enabled=feedback_enabled,
            block_between=False,
            edge_lookup=edge_lookup,
        )
        return {k: v.state for k, v in result.carries.items()}

    one_way = run(False)
    two_way = run(True)

    p1 = one_way["d01"]
    p2 = two_way["d01"]

    # (a) parent interior actually changed by the child, ONLY in the overlap.
    mask = np.asarray(feedback_mask(fb_weights.mass), dtype=bool)  # (pny, pnx)
    diff_fields = {}
    changed_outside_overlap = False
    for leaf in ("theta", "qv", "w", "p_perturbation", "ph_perturbation", "mu_perturbation"):
        a = np.asarray(getattr(p1, leaf))
        b = np.asarray(getattr(p2, leaf))
        d = np.abs(b - a)
        diff_fields[leaf] = {
            "max_abs_diff": float(d.max()),
            "mean_abs_diff_overlap": float(
                d[..., mask].mean() if d.ndim == 3 else d[mask].mean()
            ),
        }
        outside = d[..., ~mask] if d.ndim == 3 else d[~mask]
        if float(np.max(outside)) > 1e-12:
            changed_outside_overlap = True

    # u/v use their own staggered masks.
    for leaf, w in (("u", fb_weights.u), ("v", fb_weights.v)):
        a = np.asarray(getattr(p1, leaf))
        b = np.asarray(getattr(p2, leaf))
        d = np.abs(b - a)
        m = np.asarray(feedback_mask(w), dtype=bool)
        diff_fields[leaf] = {
            "max_abs_diff": float(d.max()),
            "mean_abs_diff_overlap": float(d[..., m].mean()),
        }
        if float(np.max(d[..., ~m])) > 1e-12:
            changed_outside_overlap = True

    theta_changed = diff_fields["theta"]["max_abs_diff"] > 1e-6

    # (c) consistency: re-derive copy_fcn-then-sm121 for theta and match the run.
    fed_only = apply_feedback(parent_state.theta, child_state.theta, fb_weights.mass, feedback=True)
    expected = sm121_smooth(fed_only, fb_weights.mass_smooth)
    theta_match = bool(np.allclose(np.asarray(p2.theta), np.asarray(expected), atol=1e-10))
    smoother_changed_theta = not bool(
        np.allclose(np.asarray(fed_only), np.asarray(expected), atol=1e-12)
    )

    # WRF overlap conservation on the RAW (pre-smoother) copy_fcn feedback.
    cons = feedback_overlap_conservation(child_state.theta, fb_weights.mass, leaf="theta")

    # (b) finite/stable.
    finite_ok = _finite(p2) and _finite(two_way["d02"])

    verdict = (
        "TWO_WAY_FEEDBACK_VALIDATED"
        if (theta_changed and not changed_outside_overlap and theta_match
            and smoother_changed_theta and finite_ok and cons.conserved)
        else "TWO_WAY_FEEDBACK_PARTIAL"
    )

    payload = {
        "schema": "V0120TwoWayFeedbackValidation",
        "verdict": verdict,
        "operator": "WRF copy_fcn odd-ratio area-average + sm121 1-2-1 feedback-zone smoother",
        "runtime_path": "run_domain_tree_callbacks + _operational_feedback (genuine operational callback)",
        "geometry": {
            "parent_grid_ratio": RATIO,
            "i_parent_start": I_PARENT_START,
            "j_parent_start": J_PARENT_START,
            "parent_nx_ny": [PARENT_NX, PARENT_NY],
            "child_nx_ny": [CHILD_NX, CHILD_NY],
            "nz": NZ,
            "overlap_parent_mass_cells": int(mask.sum()),
        },
        "a_parent_interior_changed_by_child": bool(theta_changed),
        "a_changed_only_in_overlap": bool(not changed_outside_overlap),
        "b_two_way_state_finite": bool(finite_ok),
        "c_theta_matches_copyfcn_then_sm121": theta_match,
        "c_smoother_actually_changed_copyfcn_result": smoother_changed_theta,
        "c_raw_feedback_overlap_conserved": bool(cons.conserved),
        "c_overlap_conservation_rel_residual": float(cons.rel_residual),
        "one_way_vs_two_way_diff": diff_fields,
        "note": (
            "CPU runtime validation of the feedback OPERATOR + SMOOTHER on the genuine "
            "operational callback path. Full 24h real-data 2-way nested equivalence "
            "(GPU, real.exe init) is deferred to v0.13."
        ),
    }
    return payload


if __name__ == "__main__":
    out = main()
    proof_dir = Path(__file__).resolve().parent
    proof_path = proof_dir / "two_way_feedback_validation.json"
    proof_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {proof_path}")
