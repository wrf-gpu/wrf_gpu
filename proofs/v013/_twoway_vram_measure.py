"""Standalone peak-VRAM probe for the two-way feedback step (d02->d03 scale).

Run on GPU with::

    PYTHONPATH=src python proofs/v013/_twoway_vram_measure.py <label>

Prints a JSON line with the peak device bytes consumed by ONE
``apply_state_feedback`` call on a realistic d02 (parent) / d03 (child) 1km nest
edge.  Used to compare BEFORE vs AFTER the VRAM reduction.  Kept as a private
helper (underscore) -- the public proof is ``twoway_vram.py``.
"""

from __future__ import annotations

import gc
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gpuwrf.contracts.grid import (
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling.boundary_feedback import (
    apply_state_feedback,
    build_state_feedback_weights,
)

# Real 9/3/1 km Canary nest (wrf_l3 namelist); the 1km feedback edge is d02->d03.
# d02 (parent, 3km): e_we=160, e_sn=67 ; d03 (child, 1km): e_we=94, e_sn=76.
# Mass-grid horizontal extents are (e_we-1, e_sn-1).
RATIO = 3
NZ = 44  # e_vert=45 -> 44 mass layers
PARENT_NX, PARENT_NY = 159, 66
CHILD_NX, CHILD_NY = 93, 75
I_PARENT_START = 52
J_PARENT_START = 20


def _make_grid(nx: int, ny: int, dx_m: float) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, NZ + 1, dtype=jnp.float64)
    metrics = DycoreMetrics.flat(
        ny=ny, nx=nx, nz=NZ, eta_levels=eta, top_pressure_pa=5000.0,
        provenance="twoway-vram-probe",
    )
    return GridSpec(
        projection=Projection("lambert", 28.0, -16.0, dx_m, dx_m, nx, ny),
        terrain=TerrainProvenance(
            source_path="probe:twoway-vram",
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


def _seed_state(grid: GridSpec, *, base: float, seed: int) -> State:
    rng = np.random.default_rng(seed)
    shapes = _state_field_shapes(grid)
    leaves = {}
    for field, shape in shapes.items():
        if field in ("p_total", "p", "ph_total", "ph", "mu_total", "mu"):
            continue  # filled from base + perturbation below
        leaves[field] = jnp.asarray(rng.standard_normal(shape), dtype=jnp.float64)
    # build totals consistent with base + perturbation so feedback rebuild is valid
    p_base = jnp.full(shapes["p_perturbation"], base)
    ph_base = jnp.full(shapes["ph_perturbation"], base + 1000.0)
    mu_base = jnp.full(shapes["mu_perturbation"], base + 2000.0)
    leaves["theta"] = base + leaves["theta"]
    leaves["p_total"] = p_base + leaves["p_perturbation"]
    leaves["p"] = p_base + leaves["p_perturbation"]
    leaves["ph_total"] = ph_base + leaves["ph_perturbation"]
    leaves["ph"] = ph_base + leaves["ph_perturbation"]
    leaves["mu_total"] = mu_base + leaves["mu_perturbation"]
    leaves["mu"] = mu_base + leaves["mu_perturbation"]
    return State(**leaves)


def _peak_bytes(dev) -> int:
    stats = dev.memory_stats() or {}
    return int(stats.get("peak_bytes_in_use", 0))


def _measure_one(dev, fn, parent, child, weights) -> dict:
    """Run ``fn(parent, child, weights)`` once and report the transient peak.

    The allocator tracks a RUNNING peak, so we (a) materialise + block on the
    resident inputs, (b) read the peak just before the op, (c) run + block, and
    (d) read the peak after; the delta is the transient working set the feedback
    op adds on top of the resident states (the marginal cost the nested run pays).
    """

    jax.block_until_ready((parent.theta, child.theta))
    gc.collect()
    resident_before = int((dev.memory_stats() or {}).get("bytes_in_use", 0))
    peak_pre = _peak_bytes(dev)
    out = fn(parent, child, weights)
    jax.block_until_ready(out.theta)
    peak_post = _peak_bytes(dev)
    resident_after = int((dev.memory_stats() or {}).get("bytes_in_use", 0))
    # keep `out` alive only long enough to read peak; drop it before returning
    del out
    gc.collect()
    return {
        "resident_before_bytes": resident_before,
        "resident_after_bytes": resident_after,
        "peak_pre_bytes": peak_pre,
        "peak_post_bytes": peak_post,
        "feedback_transient_bytes": int(peak_post - peak_pre),
        "feedback_transient_gib": float((peak_post - peak_pre) / 2**30),
        "peak_post_gib": float(peak_post / 2**30),
    }


def main(mode: str) -> dict:
    """Measure ONE feedback path's transient peak in a FRESH process.

    The device peak counter is monotonic (running max), so eager and jitted must
    each be measured in their own process to get an honest transient.  ``mode`` is
    ``eager`` (BEFORE: op-by-op ``apply_state_feedback``) or ``jitted`` (AFTER: the
    production ``_operational_feedback`` -> ``_feedback_state_jit`` path).
    """

    dev = jax.devices()[0]
    parent_grid = _make_grid(PARENT_NX + 1, PARENT_NY + 1, dx_m=3000.0)
    child_grid = _make_grid(CHILD_NX + 1, CHILD_NY + 1, dx_m=1000.0)

    weights = build_state_feedback_weights(
        parent_grid_ratio=RATIO,
        i_parent_start=I_PARENT_START,
        j_parent_start=J_PARENT_START,
        parent_grid=parent_grid,
        child_grid=child_grid,
        spec_zone=1,
    )

    parent = _seed_state(parent_grid, base=300.0, seed=1)
    child = _seed_state(child_grid, base=305.0, seed=2)

    if mode == "jitted":
        from gpuwrf.runtime.domain_tree import _feedback_state_jit

        # Warm the jit cache (compile) so the measured peak is the RUN peak, not
        # the one-time compile scratch; compile here, reset gc, then measure.
        _w = _feedback_state_jit(parent, child, weights)
        jax.block_until_ready(_w.theta)
        del _w
        gc.collect()
        result = _measure_one(dev, _feedback_state_jit, parent, child, weights)
    else:
        mode = "eager"

        def _eager(p, c, w):
            return apply_state_feedback(p, c, w, feedback=True)

        result = _measure_one(dev, _eager, parent, child, weights)

    payload = {
        "mode": mode,
        "device": str(dev.device_kind),
        "geometry": {
            "ratio": RATIO,
            "nz": NZ,
            "parent_nx_ny": [PARENT_NX, PARENT_NY],
            "child_nx_ny": [CHILD_NX, CHILD_NY],
        },
        **result,
    }
    print(json.dumps(payload))
    return payload


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eager")
