# F7 Sprint C — Worker Report (Opus 4.8 frontrunner)

## Objective

Close the single localized Block-2 gap from Sprint B: implement WRF
`rk_addtend_dry` and restore the large-step horizontal pressure-gradient force
into the per-RK-stage dry tendency, fixing the operational cadence so the
integrated `u/v` actually move (flow circulates → θ′ transported), and make the
published idealized cases (Straka density current, Skamarock warm bubble) PASS —
the F7 dynamical-core close. WRF Fortran source treated as ground truth.

## Files changed

- `src/gpuwrf/dynamics/core/rk_addtend_dry.py` (NEW) — `large_step_horizontal_pgf`
  (WRF `module_em.F:1325` → `module_big_step_utilities_em.F:2459-2466`/`:2379-2386`,
  built from the **absolute** rk_step_prep diagnostics: full `php`, `alt`,
  absolute `p'`, `pb`, `al'`; returns the **coupled** `ru/rv_tend = -cqu*dpx`) and
  `rk_addtend_dry` (WRF `module_em.F:1711-1786`: field-specific map/mass-coupled
  merge of RK1-fixed physics tendencies; identity when physics-off + periodic).
- `src/gpuwrf/runtime/operational_mode.py`:
  - `_augment_large_step_tendencies` now builds all large-step tendencies
    **coupled** (advection × face mass, diffusion × face mass, flux-form coupled
    theta, large-step PGF), then applies `rk_addtend_dry`.
  - **Removed the double-application**: the previous `advance_stage` applied
    `add_scaled_tendencies(origin, tendencies, dt_rk)` AND the per-substep
    `advance_uv` tendency. WRF advances the prognostics ONLY through the acoustic
    substeps (`u += dts*ru_tend`, `module_small_step_em.F:805`); `candidate` is now
    the carried-forward stage-entry state (WRF `u_2`), `rk1_reference` is the RK
    reference (`u_1`).
  - `_acoustic_core_state_from_prep`: `theta_tend`/`mu_tend` were wired from the
    base ZERO tendency; now wired from the computed large-step tendency so the
    flux-form theta advection enters `advance_mu_t`.
- `src/gpuwrf/ic_generators/idealized.py` — idealized harness now runs with a
  rigid lid + WRF upper Rayleigh damping (`top_lid=True`, `w_damping=1`,
  `damp_opt=3`, `dampcoef=0.2`, `zdamp=3000`; cited WRF coefficients), removing the
  spurious open-top w-face mode. (Minimal, documented.)
- `scripts/f6_transaction_audit.py` — audit RK cadence updated to match production
  (uses `_augment_large_step_tendencies`; no `add_scaled_tendencies`).
- `proofs/f7c/**` — proof objects.

## Commands run (all `taskset -c 0-3`, `cuda:0`, fp64)

- `scripts/f7a_oracles.py --conservation-steps 300 --epssm 0.5` (AC4 regression)
- `scripts/f6_transaction_audit.py --steps 12 --dt-s 6 --acoustic-substeps 4
  --epssm 0.5 --combination a --damping` (AC3)
- `run_warm_bubble_case` / `run_density_current_case` (require_gpu=True) → proofs/f7c
- single-compile `_physics_boundary_step` traces (warm bubble) for localization
- IC `large_step_horizontal_pgf` structure check
- `pytest` on the AC5 red test + acoustic/mu_t/PGF regression subset

## Acceptance gate status

- **AC1 Straka density current — NOT MET** (RAN_TO_COMPLETION, verdict FAIL).
  Runs end-to-end but goes non-finite from the same large-scale instability as
  AC2. `proofs/f7c/straka_density_current_*`.
- **AC2 Skamarock warm bubble — NOT MET** (RAN_TO_COMPLETION, verdict FAIL).
  **Major progress**: the flow now genuinely circulates — `max|u|` grows
  0→3.96→7.89→11.9→16.2→20.2 m/s over t=10..50 s (baseline before the fix: u≈0,
  bubble rose 0.66 m in 500 s with θ′ untransported), with buoyant `max|w|`
  5.9→12→18→24→30. But `max|u|`/`max|w|` then grow ~linearly in lockstep and the
  run goes non-finite at ~80-100 s. `proofs/f7c/skamarock_bubble_*`.
- **AC3 12-step operational-dt audit — NOT MET.** First critical moved from
  Sprint B's step 6-7 to **step 8** (RK2/substep1, `advance_mu_t`,
  `pressure_bounded`, abs_p/base=3.92). No masking clamp. `proofs/f7c/audit_*`.
- **AC4 no regression — PASS.** flat-rest exact 0 on all fields (confirms the
  large-step PGF is balanced at the reference rest state), analytic acoustic
  dipole sign+order PASS, 300-step conservation dry-drift 0 / theta-drift 0.
  `proofs/f7c/regression_recheck.json`.
- **AC5 last red test — documented precisely, not the cadence bug.**
  `test_step2_operational_theta_stays_finite` exercises the **legacy non-prep**
  `_operational_acoustic_substep_core` single-substep path on the real fp32 d02 IC.
  Its NaN comes from the legacy `_acoustic_core_state` builder feeding
  inconsistent work-array inputs (`al=0`, `mu=full_pert` with `muts=mut`,
  `theta_1=t_save`) into `advance_mu_t`/`advance_w` — a data-consistency defect in
  the legacy builder, NOT the rk_addtend_dry/PGF cadence this sprint scoped (the
  production forecast uses the prep-path `_acoustic_scan`, not this path). All
  acoustic-core inputs (`u/v/w/theta/mu/c2a/alt`) and the `calc_coef_w`
  `a/alpha/gamma` are finite; the NaN is downstream in the legacy substep math.
  No tolerance widened, no xfail.

## What was fixed (the localized gap, verified)

The large-step horizontal PGF was missing from the per-RK-stage tendency, and the
operational cadence double-applied the momentum tendency (`add_scaled_tendencies`
+ acoustic `advance_uv`), so integrated `u/v` never moved. Both are now WRF-faithful:
- IC large-step PGF physical u-acceleration = **0.649 m/s² at the bubble flanks
  (x≈9/11 km), exactly 0 at the bubble center and far field** — the correct
  horizontal-gradient structure (matches Sprint B's ~0.6 m/s² prototype). A first
  implementation using `diagnose_pressure_al_alt(state, base_state=None)` (which
  returns `al=0` and perturbation-only `php`) gave the structure *backwards* and
  drove a spurious uniform-column mass convergence; the corrected version uses the
  full absolute diagnostics.
- The PGF and small-step `advance_uv` acoustic PGF are confirmed **distinct split
  terms, not a double-count** (corrects Sprint A). `rk_addtend_dry` is implemented
  faithfully (identity for the physics-off periodic gate).

## Root-cause of the residual instability (precisely localized)

A coherent large-scale runaway remains once circulation is active: `max|u|`/`max|w|`
grow ~linearly with θ′ pinned at 2 K, non-finite at ~80-100 s — independent of
acoustic substeps (10 vs 20), `epssm` (0.1 vs 0.5), and explicit diffusion (ν up to
200 m²/s shifts blow-up < 20 s). Not grid-scale noise, not acoustic CFL.

Bisection localized it to a **fundamental acoustic-core formulation mismatch**, NOT
the F7C cadence gap:
- The dycore uses a **delta-from-RK-reference** acoustic split: the small-step
  perturbation-pressure work array `p` (from `calc_p_rho`) is relative to the stage
  reference, so for a slowly-rising parcel it carries ~no buoyancy. Sprint B
  therefore feeds `pg_buoy_w` the **absolute** `p_buoy` (rk_step_prep p′) to make a
  static parcel rise; forcing it back to the substep `p` kills the rise entirely
  (`max|w|→0.01`), confirming the absolute p′ is required.
- But the absolute `p_buoy` is a **once-per-RK-stage constant** forcing that does
  NOT receive the acoustic pressure-adjustment feedback (the work-`p` used by the
  restoring small-step PGF stays ≈ 0), so buoyancy + PGF pump momentum into the
  circulation with no acoustic restoring closing the loop → linear runaway. The
  same weakness produces the AC3 step-8 `pressure_bounded` violation on the real IC.

The proper fix is to diagnose the small-step pressure from the **absolute small-step
total** state (WRF's actual `mu_save`+`mu_work` totals through `calc_p_rho`) so
buoyancy and pressure adjustment are consistent — a dedicated acoustic-core
reformulation sprint, beyond the scoped `rk_addtend_dry`/PGF cadence.

## Unresolved risks / next decision

- The acoustic-core delta-from-reference vs absolute-buoyancy mismatch is the
  remaining blocker to AC1/AC2/AC3. It needs an acoustic-core sprint (diagnose
  small-step pressure from absolute totals), not another cadence patch.
- Legacy non-prep `_operational_acoustic_substep_core` builder is inconsistent
  (AC5) and should be retired or fixed to mirror `small_step_prep_wrf`.
- `w_t` (large-step w-advection tendency) is computed but not consumed by the
  acoustic core (rw_tend is buoyancy-only) — a secondary gap, harmless here.

## Verdict

**F7C_PARTIAL.** The scoped localized gap is genuinely fixed and WRF-verified:
`rk_addtend_dry` + the large-step horizontal PGF are implemented (PGF structure
proven physically correct at the bubble flanks), the double-application cadence bug
is removed, and the integrated flow now **circulates** (max|u| 0→20 m/s) and
transports buoyantly — versus the Sprint-B baseline where u stayed ≈0 and the bubble
rose 0.66 m. AC4 fully holds (flat-rest exact 0 proves the PGF is balanced at rest).
The idealized cases and the 12-step audit do **not** PASS: a pre-existing
acoustic-core formulation mismatch (absolute-buoyancy forcing without the
delta-from-reference acoustic pressure-restoring feedback) drives a coherent
large-scale u/w runaway, precisely localized via bisection and documented for a
dedicated acoustic-core sprint. Honest partial: the Block-2 cadence is closed; the
deeper acoustic restoring-balance reformulation is the next gate.

F7C_PARTIAL
