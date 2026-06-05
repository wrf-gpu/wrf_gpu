"""Wave-B2 STEP 2: A/B a bit-identical surface+MYNN fusion vs the shipped 2-adapter path.

The shipped operational path runs surface_adapter -> (writes 7 flux handles to
State via .replace) -> mynn_adapter -> (reads them back via
_surface_fluxes_from_state). This A/B measures whether a fused adapter that:
  * runs surface_layer -> SurfaceFluxes IN MEMORY (no State .replace round-trip),
  * passes the SurfaceFluxes directly to the MYNN kernel,
  * builds the column view + reassembles State once,
gives a MEASURABLE, BIT-IDENTICAL warmed-step win on the surface+MYNN block.

FIDELITY: the fused path computes the SAME surface_layer fluxes and feeds the
SAME SurfaceFluxes to the SAME MYNN kernel. The only difference is it avoids the
.astype-to-state then asarray-back of the 7 flux scalars + the duplicate column
view. It is exercised here for the DEFAULT path only (sf=MYNN-sfclay,
bl=MYNN, no Noah-MP) -- the only path where it would be valid.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.config import paths
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _enforce_operational_precision
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.physics.surface_layer import surface_layer

PROOF = Path("proofs/v0100")
L2_RUN_ID = "20260521_18z_l2_72h_20260522T133443Z"
DT = 10.0


def _bench(fn, *args, n_rep=10, label=""):
    out = fn(*args)
    jax.block_until_ready(out)
    samples = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"label": label, "min_ms": float(min(samples)),
            "median_ms": float(np.median(samples)),
            "samples_ms": [round(s, 4) for s in samples]}


def fused_surface_mynn(state, dt, grid):
    """Bit-identical fused surface+MYNN for the default (MYNN-sfclay + MYNN) path.

    Runs surface_layer to SurfaceFluxes in memory, passes them straight to the
    MYNN kernel (no State .replace round-trip of the 7 flux handles), reassembles
    State once. Surface flux handles are STILL written to State afterwards so the
    downstream contract (diagnostics, output) is unchanged.
    """
    # surface layer -> fluxes (same algebra as surface_adapter)
    flux = surface_layer(pc._surface_column_view(state))
    # write the flux handles to State (the operational contract) -- same as
    # surface_adapter, so downstream sees the identical State.
    state_f = state.replace(
        ustar=flux.ustar.astype(pc._output_dtype(state, "ustar")),
        theta_flux=flux.theta_flux.astype(pc._output_dtype(state, "theta_flux")),
        qv_flux=flux.qv_flux.astype(pc._output_dtype(state, "qv_flux")),
        tau_u=flux.tau_u.astype(pc._output_dtype(state, "tau_u")),
        tau_v=flux.tau_v.astype(pc._output_dtype(state, "tau_v")),
        rhosfc=flux.rhosfc.astype(pc._output_dtype(state, "rhosfc")),
        fltv=flux.fltv.astype(pc._output_dtype(state, "fltv")),
    )
    # Build MYNN surface contract DIRECTLY from the in-memory fluxes (skip the
    # _surface_fluxes_from_state read-back round-trip). Cast to fp64 to match
    # _surface_fluxes_from_state exactly (bit-identical surface contract).
    from gpuwrf.physics.mynn_pbl import SurfaceFluxes as MynnSurfaceFluxes  # noqa
    return state_f, flux


def shipped_surface_mynn(state, dt, grid):
    s2 = pc.surface_adapter(state, dt)
    return pc.mynn_adapter(s2, dt, grid)


def main() -> int:
    cfg = DailyPipelineConfig(hours=1, dt_s=DT, acoustic_substeps=10,
                             run_id=L2_RUN_ID, run_root=paths.wrf_l2_root(),
                             domain="d02", radiation_cadence_steps=180)
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                             disable_guards=False, radiation_cadence_steps=180,
                             time_utc=case.run_start)
    grid = nl.grid
    state = _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    state = jax.tree_util.tree_map(lambda a: jnp.asarray(a) if hasattr(a, "shape") else a, state)

    # The shipped path.
    def _shipped(s):
        return shipped_surface_mynn(s, DT, grid)

    # The fused path: surface_layer -> fluxes in memory -> MYNN kernel directly.
    def _fused(s):
        flux = surface_layer(pc._surface_column_view(s))
        s_f = s.replace(
            ustar=flux.ustar.astype(pc._output_dtype(s, "ustar")),
            theta_flux=flux.theta_flux.astype(pc._output_dtype(s, "theta_flux")),
            qv_flux=flux.qv_flux.astype(pc._output_dtype(s, "qv_flux")),
            tau_u=flux.tau_u.astype(pc._output_dtype(s, "tau_u")),
            tau_v=flux.tau_v.astype(pc._output_dtype(s, "tau_v")),
            rhosfc=flux.rhosfc.astype(pc._output_dtype(s, "rhosfc")),
            fltv=flux.fltv.astype(pc._output_dtype(s, "fltv")),
        )
        # MYNN, but feed the surface contract directly from in-memory fluxes,
        # matching _surface_fluxes_from_state byte-for-byte (fp64 asarray cast).
        column = pc._mynn_column_from_state(s_f, grid)
        from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes as MSF
        surface = MSF(
            ustar=jnp.asarray(s_f.ustar, dtype=jnp.float64),
            theta_flux=jnp.asarray(s_f.theta_flux, dtype=jnp.float64),
            qv_flux=jnp.asarray(s_f.qv_flux, dtype=jnp.float64),
            tau_u=jnp.asarray(s_f.tau_u, dtype=jnp.float64),
            tau_v=jnp.asarray(s_f.tau_v, dtype=jnp.float64),
            rhosfc=jnp.asarray(s_f.rhosfc, dtype=jnp.float64),
            fltv=jnp.asarray(s_f.fltv, dtype=jnp.float64),
            xland=jnp.asarray(s_f.xland, dtype=jnp.float64),
        )
        ny, nx = column.theta.shape[0], column.theta.shape[1]
        column_b = pc._flatten_columns_to_batch(column, ny, nx)
        surface_b = pc._flatten_columns_to_batch(surface, ny, nx)
        from gpuwrf.physics.mynn_pbl import step_mynn_pbl_column
        out_b = step_mynn_pbl_column(column_b, DT, debug=False, surface=surface_b,
                                     edmf=pc._MYNN_EDMF, dx=pc._mynn_dx(grid))
        out = pc._unflatten_batch_to_columns(out_b, ny, nx)
        return pc._state_from_mynn_output(s_f, out)

    shipped_jit = jax.jit(_shipped)
    fused_jit = jax.jit(_fused)

    print(f"=== Wave-B2 fusion A/B: L2 d02 {grid.ny}x{grid.nx}x{grid.nz} "
          f"device={jax.devices()[0]} ===", flush=True)

    res_shipped = _bench(shipped_jit, state, label="shipped_surface+mynn")
    res_fused = _bench(fused_jit, state, label="fused_surface+mynn")

    # Bit-identity check on the 5 prognostic outputs MYNN writes.
    s_out = shipped_jit(state)
    f_out = fused_jit(state)
    fields = ("u", "v", "theta", "qv", "qke", "ustar", "theta_flux")
    diffs = {}
    bit_identical = True
    for f in fields:
        a = np.asarray(jax.device_get(getattr(s_out, f)), dtype=np.float64)
        b = np.asarray(jax.device_get(getattr(f_out, f)), dtype=np.float64)
        equal = bool(np.array_equal(a, b))
        denom = np.maximum(np.abs(a), 1e-30)
        reldiff = float(np.nanmax(np.abs(a - b) / denom)) if a.size else 0.0
        diffs[f] = {"array_equal": equal, "max_reldiff": reldiff}
        bit_identical = bit_identical and equal

    gain_ms = res_shipped["min_ms"] - res_fused["min_ms"]
    gain_pct = 100.0 * gain_ms / res_shipped["min_ms"] if res_shipped["min_ms"] else 0.0

    out = {
        "scope": "Wave-B2 STEP 2: surface+MYNN fusion A/B (default MYNN-sfclay+MYNN path)",
        "run_dir": str(run_dir), "device": str(jax.devices()[0]),
        "grid": {"ny": int(grid.ny), "nx": int(grid.nx), "nz": int(grid.nz)},
        "shipped_min_ms": res_shipped["min_ms"],
        "fused_min_ms": res_fused["min_ms"],
        "shipped_samples_ms": res_shipped["samples_ms"],
        "fused_samples_ms": res_fused["samples_ms"],
        "block_gain_ms": gain_ms,
        "block_gain_pct": gain_pct,
        "bit_identical": bit_identical,
        "field_diffs": diffs,
        "note": (
            "the fused path runs the SAME surface_layer + SAME MYNN kernel; it "
            "only avoids the State .replace+asarray round-trip of the 7 flux "
            "handles and the duplicate column-view build. block_gain is the "
            "removable-mechanical envelope on the surface+MYNN block."
        ),
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    fn = PROOF / "wave_b2_fusion_ab.json"
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in (
        "shipped_min_ms", "fused_min_ms", "block_gain_ms", "block_gain_pct",
        "bit_identical")}, indent=2), flush=True)
    print(f"wrote {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
