# F7C — rk_addtend_dry + large-step PGF cadence reconciliation

This proof object documents (1) the WRF verification of the large-step PGF /
`rk_addtend_dry` split, (2) the JAX cadence fix that makes the integrated `u/v`
move (flow circulates), and (3) the honest residual instability that remains.

## 1. WRF ground truth (verified against source)

- `rk_tendency` puts the horizontal pressure-gradient force in the **large-step
  coupled** `ru/rv_tend` via `horizontal_pressure_gradient`
  (`dyn_em/module_em.F:1325`; body `dyn_em/module_big_step_utilities_em.F:2459-2466`
  for u, `:2379-2386` for v). WRF: `ru_tend(i,k,j) = ru_tend(i,k,j) - cqu*dpx`,
  where `dpx` carries the coupled `(c1h*muu+c2h)` mass factor.
- The large-step PGF uses the **absolute** `rk_step_prep` diagnostics
  (`ph`, `alt`, `p'`, `pb`, `al`, `php` = full geopotential on mass levels).
- `advance_uv` consumes `ru_tend` once per acoustic substep:
  `u(i,k,j) = u(i,k,j) + dts*ru_tend(i,k,j)` (`dyn_em/module_small_step_em.F:805`),
  then adds the **small-step** PGF `dpxy` (`:828-868`) built from the *work-array*
  perturbation pressure (which restarts ~0 each RK stage).
- `rk_addtend_dry` (`dyn_em/module_em.F:1711-1786`) merges the RK1-fixed physics
  tendencies `*_tendf` into `*_tend` with field-specific coupling: u `/msfuy`,
  v `*msfvx_inv`, w/ph/theta `/msfty`, mu uncoupled, plus the final-RK theta
  diabatic term `(c1*mut+c2)*h_diabatic/msfty`. With physics off + periodic
  boundaries every `*_tendf == 0`, so the merge is numerically identity.

**Conclusion (corrects Sprint A):** the large-step PGF and the small-step
`advance_uv` PGF are *distinct split terms*, not a double-count. Sprint A wrongly
dropped the large-step PGF. WRF keeps both.

## 2. JAX implementation + the cadence fix

- New module `src/gpuwrf/dynamics/core/rk_addtend_dry.py`:
  - `large_step_horizontal_pgf(...)` builds the WRF coupled `ru/rv_tend = -cqu*dpx`
    from the absolute diagnostics (full `php`, `alt`, absolute `p'`, `pb`, `al'`),
    on the C-grid u/v faces (matching `advance_uv_wrf`).
  - `rk_addtend_dry(...)` the field-specific map/mass-coupled merge.
- `runtime/operational_mode.py`:
  - `_augment_large_step_tendencies` now builds **coupled** large-step tendencies
    (advection, diffusion, flux-form theta, and the large-step PGF) so they net
    correctly with the coupled small-step work arrays, then applies
    `rk_addtend_dry`.
  - **Double-application removed**: the previous cadence applied
    `add_scaled_tendencies(origin, tendencies, dt_rk)` AND the per-substep
    `advance_uv` tendency, double-counting the momentum forcing and preventing
    `u/v` from moving. WRF advances the prognostics ONLY through the acoustic
    substeps; `candidate` is now the carried-forward stage-entry state (the WRF
    `u_2`), `rk1_reference` is the RK reference (`u_1`).
  - `theta_tend`/`mu_tend` in `_acoustic_core_state_from_prep` were wired from the
    base ZERO tendency; now wired from the computed large-step tendency so the
    flux-form theta advection actually enters `advance_mu_t`.
- `scripts/f6_transaction_audit.py` updated to the same cadence (no
  `add_scaled_tendencies`; uses `_augment_large_step_tendencies`).

### Evidence the PGF is correct and the flow now circulates

- **IC PGF structure** (warm bubble, `top_lid`): physical u-acceleration peaks at
  **0.649 m/s² at the bubble flanks (x≈9/11 km)**, **exactly 0 at the bubble
  center (x=10 km)** and **0 in the far field** — the correct horizontal-gradient
  structure (matches Sprint B's ~0.6 m/s² prototype). The earlier broken version
  (using `diagnose_pressure_al_alt(state, base_state=None)` and perturbation-only
  `php`) gave a constant nonzero value at the center and zero at the edge — the
  cause of the spurious uniform-column convergence.
- **Flow circulates**: integrated warm-bubble `max|u|` 0 → 3.96 → 7.89 → 11.9 →
  16.2 → 20.2 m/s at t = 10/20/30/40/50 s (baseline before the fix: `max|u|`
  stayed ≈ 0; the bubble rose 0.66 m in 500 s with θ′ untransported). Buoyant
  `max|w|` rises 5.9 → 12 → 18 → 24 → 30 m/s over the same window.
- **No regression (AC4)**: `flat_rest` exact 0 on all fields (confirms the
  large-step PGF produces zero tendency at the reference rest state — i.e. it is
  balanced), analytic acoustic dipole sign+order PASS, 300-step conservation
  dry-drift 0 / theta-drift 0. `proofs/f7c/regression_recheck.json`.

## 3. Honest residual: the absolute-buoyancy ↔ delta-reference acoustic mismatch

Once the flow circulates, a **coherent large-scale instability** appears:
`max|u|` and `max|w|` grow ~linearly in lockstep (u: 8→16→24→30; w: 12→24→36→48
m/s at 20/40/60/80 s) with θ′ pinned at 2.0 K, going non-finite at ~80-100 s
(warm bubble) — independent of acoustic substeps (10 vs 20), `epssm` (0.1 vs 0.5),
and explicit diffusion (ν up to 200 m²/s changed the blow-up time by < 20 s).
This is **not** grid-scale noise (diffusion does not damp it) and **not** an
acoustic-CFL violation (vertical CFL ≈ 0.003).

Localized root cause (bisection):
- The dycore uses a **delta-from-RK-reference** acoustic formulation: the
  small-step perturbation-pressure work array `p` (from `calc_p_rho`) is relative
  to the stage reference, which for a slowly-rising bubble is ≈ the current state,
  so the work-`p` carries ~no buoyancy. Sprint B therefore feeds `pg_buoy_w` the
  **absolute** `p_buoy` (rk_step_prep p′) so a statically-imbalanced parcel rises.
  Forcing `pg_buoy_w` back to the substep `p` kills the rise entirely
  (`max|w| → 0.01`), confirming the absolute p′ is required for buoyancy.
- But the absolute `p_buoy` is a **once-per-RK-stage constant** forcing that does
  **not** receive the acoustic pressure-adjustment feedback (the work-`p` that
  `advance_uv`/`advance_w` use for the restoring PGF stays ≈ 0). So buoyancy and
  the PGF pump momentum into the circulation with no acoustic restoring closing
  the loop → linear runaway. The local `mu'` initially drifted under this (relieved
  once the PGF face-grid bug was fixed; the runaway then moved into u/w).

This is a **fundamental acoustic-core formulation issue** (the delta-from-reference
split is incompatible with the absolute-buoyancy term), **not** the
`rk_addtend_dry`/PGF cadence gap this sprint scoped. The proper fix is to diagnose
the small-step pressure from the **absolute small-step total** state (WRF's actual
`mu_save`+`mu_work` totals through `calc_p_rho`), so buoyancy and pressure
adjustment are consistent — a dedicated acoustic-core sprint.
