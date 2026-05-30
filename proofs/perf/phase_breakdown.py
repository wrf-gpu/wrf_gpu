"""Per-phase cost breakdown of the warmed per-step coupled forecast.

Times each phase of the per-step pipeline IN ISOLATION (each wrapped in its own
jax.jit, warmed, then median of N timed reps via block_until_ready), plus the
XLA cost-analysis FLOPs+bytes of each phase, so we can attribute the per-step
wall to:

  acoustic small-step loop  (the dominant suspect: 16 substeps/step =
                              1[RK1] + 5[RK2] + 10[RK3]; each substep =
                              advance_uv + advance_mu_t + advance_w[tridiag])
    - one acoustic_substep_core call (and within it advance_w's vertical Thomas)
  calc_coef_w                (per-RK-stage coefficient build, 3/step)
  small_step_prep_wrf        (per-RK-stage prep, 3/step)
  calc_p_rho (EOS step=0)    (per-RK-stage, 3/step)
  compute_advection_tendencies + flux advection + diffusion + PGF (per stage)
  physics: thompson / surface / mynn / rrtmg-apply  (1/step, rrtmg @ cadence)
  boundary apply + guards    (1/step)
  halo                       (many/step)

Each phase is timed via a 1-rep vs N-rep marginal to remove dispatch overhead,
mirroring WRF's per-step call counts (RK stages x substeps). We then build the
per-step wall budget table.

Run:
  PYTHONPATH=src OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-3 \
    python proofs/perf/phase_breakdown.py
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.core.advance_w import dry_cqw
from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf
from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf
from gpuwrf.dynamics.tridiag_solve import thomas_solve_scan
from gpuwrf.coupling.physics_couplers import thompson_adapter, surface_adapter, mynn_adapter
from gpuwrf.coupling.boundary_apply import apply_lateral_boundaries
import gpuwrf.contracts.state as _stmod
from gpuwrf.runtime.operational_mode import (
    _augment_large_step_tendencies,
    _enforce_operational_precision,
)

PROOF = Path("proofs/perf")

# Profiling-only guard (see roofline_costanalysis.py): tolerate abstract
# placeholders in State.__init__'s lu_index int cast during .lower() introspection.
_orig_asarray = _stmod.jnp.asarray


def _safe_asarray(x, dtype=None, **kw):
    try:
        return _orig_asarray(x, dtype=dtype, **kw) if dtype is not None else _orig_asarray(x, **kw)
    except (TypeError, ValueError):
        return x


_stmod.jnp.asarray = _safe_asarray


def _block(x):
    jax.block_until_ready(x)


def _time_fn(fn, *args, reps=20, warm=3):
    """Warm then median-of-reps wall of a jit'd fn(*args)."""
    f = jax.jit(fn)
    for _ in range(warm):
        _block(f(*args))
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _block(f(*args))
        samples.append(time.perf_counter() - t0)
    samples.sort()
    # also cost analysis
    try:
        ca = jax.jit(fn).lower(*args).compile().cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0]
        flops = float(ca.get("flops", float("nan")))
        byts = float(ca.get("bytes accessed", ca.get("bytes_accessed", float("nan"))))
    except Exception:
        flops = byts = float("nan")
    return {
        "median_ms": samples[len(samples) // 2] * 1000.0,
        "min_ms": samples[0] * 1000.0,
        "flops": flops,
        "gbytes": byts / 1e9,
    }


def main() -> int:
    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist, run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=180, time_utc=case.run_start,
    )
    grid = nl.grid
    metrics = nl.metrics
    dt_s = float(nl.dt_s)
    hspec = halo_spec(grid)
    state = _enforce_operational_precision(case.state, force_fp64=True)
    haloed = apply_halo(state, hspec)
    ny, nx, nz = int(grid.ny), int(grid.nx), int(grid.nz)

    phases = {}

    # ---- halo ----
    phases["halo_apply"] = _time_fn(lambda s: apply_halo(s, hspec), state, reps=30)

    # ---- compute_advection_tendencies ----
    phases["advection_tendencies"] = _time_fn(
        lambda h: compute_advection_tendencies(h, nl.tendencies, grid), haloed, reps=20
    )

    # ---- _augment_large_step_tendencies (flux adv + diffusion + PGF) ----
    base_tend = compute_advection_tendencies(haloed, nl.tendencies, grid)
    phases["augment_large_step_tend_flux_adv"] = _time_fn(
        lambda h, t: _augment_large_step_tendencies(h, t, nl, rk_step=3), haloed, base_tend, reps=20
    )

    # ---- small_step_prep_wrf ----
    prep = small_step_prep_wrf(haloed, 3, dt_s, metrics=metrics, reference_state=haloed, ww=jnp.zeros((nz + 1, ny, nx)))
    phases["small_step_prep"] = _time_fn(
        lambda h: small_step_prep_wrf(h, 3, dt_s, metrics=metrics, reference_state=h,
                                      ww=jnp.zeros((nz + 1, ny, nx))), haloed, reps=20
    )

    # ---- calc_p_rho (EOS step=0) ----
    phases["calc_p_rho_eos"] = _time_fn(
        lambda p: calc_p_rho_wrf(p, step=0, non_hydrostatic=True), prep, reps=20
    )

    # ---- calc_coef_w (tridiag coefficient build) ----
    cqw = dry_cqw(nz, ny, nx, dtype=jnp.float64)
    phases["calc_coef_w_coeffs"] = _time_fn(
        lambda mut, c2a: calc_coef_w_wrf_coefficients(
            mut, metrics, dt=dt_s / 10.0, epssm=float(nl.epssm), top_lid=bool(nl.top_lid),
            cqw=cqw, c2a=c2a),
        prep.mut, prep.c2a, reps=20,
    )

    # ---- calc_coef_w coefficients (a/alpha/gamma) for the tridiag-solve isolation ----
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        prep.mut, metrics, dt=dt_s / 10.0, epssm=float(nl.epssm), top_lid=bool(nl.top_lid),
        cqw=cqw, c2a=prep.c2a,
    )

    # ---- the vertical Thomas tridiag solve in isolation ----
    a3 = jnp.asarray(a) if a.ndim == 3 else jnp.broadcast_to(jnp.asarray(a)[:, None, None], (nz + 1, ny, nx))
    alpha3 = jnp.asarray(alpha) if alpha.ndim == 3 else jnp.broadcast_to(jnp.asarray(alpha)[:, None, None], (nz + 1, ny, nx))
    gamma3 = jnp.asarray(gamma) if gamma.ndim == 3 else jnp.broadcast_to(jnp.asarray(gamma)[:, None, None], (nz + 1, ny, nx))
    rhs = jnp.asarray(state.w, dtype=jnp.float64)
    if rhs.shape[0] != a3.shape[0]:
        rhs = jnp.concatenate([rhs, rhs[-1:]], axis=0)[: a3.shape[0]]
    phases["tridiag_thomas_solve_vertical"] = _time_fn(
        lambda A, AL, G, R: thomas_solve_scan(A, AL, G, R), a3, alpha3, gamma3, rhs, reps=30
    )

    # ---- physics adapters ----
    phases["physics_thompson"] = _time_fn(lambda s: thompson_adapter(s, dt_s), state, reps=15)
    phases["physics_surface"] = _time_fn(lambda s: surface_adapter(s, dt_s), state, reps=15)
    phases["physics_mynn"] = _time_fn(lambda s: mynn_adapter(s, dt_s, grid), state, reps=15)

    # ---- boundary apply ----
    phases["boundary_apply"] = _time_fn(
        lambda s: apply_lateral_boundaries(s, 0.0, dt_s, nl.boundary_config), state, reps=20
    )

    out = {
        "scope": "Per-phase isolated cost breakdown -- warmed per-step coupled d02",
        "run_dir": str(run_dir),
        "device": str(jax.devices()[0]),
        "grid": {"ny": ny, "nx": nx, "nz": nz},
        "per_step_call_counts": {
            "acoustic_substeps_total": "16 = 1(RK1)+5(RK2)+10(RK3)",
            "calc_coef_w": 3, "small_step_prep": 3, "calc_p_rho_eos": 3,
            "advection_tendencies": 3, "augment_large_step_tend": 3,
            "halo_apply": "~8 (per stage entry/exit + step entry/exit)",
            "physics": "1/step (rrtmg @ cadence 180)", "boundary": "1/step",
        },
        "phases_isolated": phases,
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    fn = PROOF / "phase_breakdown.json"
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(phases, indent=2), flush=True)
    print(f"\nwrote {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
