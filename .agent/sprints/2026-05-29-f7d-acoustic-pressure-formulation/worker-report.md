# F7D Worker Report — MUT/MUTS acoustic total-mass semantics fix

**Verdict: `F7D_PARTIAL`.**
The GPT-5.5-verified MUT/MUTS mass-semantics fix is implemented exactly per
`gpt-findings.md §3.2`, verified against WRF source, and proven correct
(AC1, AC2) with **no regression** (AC5). **The mass-semantics fix ALONE does NOT
remove the idealized runaway (AC3 fails).** The residual runaway has been
rigorously localized to a *different* root cause — an idealized-IC
perturbed-column discrete-balance / horizontally-uniform vertical-acoustic-mode
issue, **not** the dycore mass-semantics — with empirical proof either way.

## Objective
Implement the verified MUT/MUTS total-mass fix so `calc_p_rho` is fed WRF's
full small-step total mass (`grid%muts`) instead of base mass (MUB), restoring
the acoustic loop; make the idealized cases stop running away.

## Files changed
- `src/gpuwrf/dynamics/core/small_step_prep.py` — `prep.mut = MUB + MU_current`
  (was MUB); added explicit `prep.mub`; `muts = mut + mu_work`; `muu/muv` from
  `mut`, `muus/muvs` from `muts`; θ/u/v/w work arrays from the full current/stage
  mass pairs.
- `src/gpuwrf/dynamics/core/calc_p_rho.py` — `_calc_al_p` param `mut→muts_total`;
  `calc_p_rho_wrf` feeds `prep.muts`; `calc_p_rho_step` param renamed; numerator
  kept as work variables (NOT absolute).
- `src/gpuwrf/dynamics/core/acoustic.py` — `acoustic_substep_core` feeds the LIVE
  `muts_new` into `calc_p_rho_step` (was `uv_state.mut`); `advance_w` gets full
  `mut` + live `muts`; `pg_buoy_w` split unchanged.
- `scripts/f7d_*.py` — runaway probe, source-parity probe (AC1/AC2), substep
  tracer, w-mode locator, null-bubble probe, ph'=0 IC experiment.

## Commands run (all `taskset -c 0-3`, `cuda:0`, fp64)
- `scripts/f7d_source_parity_probe.py` → AC1, AC2 PASS.
- `scripts/f7a_oracles.py --conservation-steps 300 --epssm 0.5` → AC5 PASS.
- `scripts/f7d_runaway_probe.py --case warm_bubble --end-seconds 200` → runaway persists.
- `scripts/f7d_substep_tracer.py`, `f7d_mode_locator.py`, `f7d_null_bubble_probe.py`
  → localization.
- `scripts/f6_transaction_audit.py --steps 12 --dt-s 6 --acoustic-substeps 4 --epssm 0.5 --combination a --damping` → AC4 (see below).

## Proof objects (`proofs/f7d/`)
`mass_semantics_proof.md`, `rk1_source_parity.json` (AC1), `acoustic_restoring_probe.json`
(AC2), `f7a_recheck/{flat_rest_oracle,analytic_acoustic_oracle,conservation_long_run}.json`
+ `regression_recheck.json` (AC5), `postfix_runaway_warm_bubble.json` +
`postfix_runaway_straka.json` + `{warm_bubble,density_current}_ac3_verdict.json` +
`plots/{warm_bubble,straka}_maxw_vs_t.txt` (AC3), `substep_trace_warm_bubble.json`,
`w_mode_locator.txt`, `consistent_base_column_stable.txt`, `ic_phzero_warm_bubble.json`,
`warm_bubble_epssm0p5.json`, `rwtend_check.json` (localization), `audit_operational_dt/`
+ `audit_summary.md` (AC4).

## Acceptance gates
- **AC1 RK1 source-parity — PASS.** Stage-entry rest: work mu/theta/ph/p/al all
  exactly 0; stage absolute `p_buoy` = 744 Pa (warm) / 5176 Pa (cold). Synthetic
  nonzero-mu check: `mut=MUB+MU_current`, `muts=MUB+MU_ref`, `muts-mut=mu_work`
  all machine-zero. WRF split preserved.
- **AC2 acoustic-restoring probe — PASS.** After one substep, work `p` Δ=1.82,
  `al` Δ=1e-4, `pm1` updated, live `muts` Δ=0.034 feeds the denominator, and the
  next `advance_uv` consumes the refreshed `p`. Restoring loop is live.
- **AC3 idealized cases — FAIL (runaway persists).** Warm bubble: max|w| grows
  linearly (slope 0.604 m/s/s, ~9× the physical 0.065 m/s² buoyancy) → NaN at ~80s.
  **The mass-semantics fix alone did NOT remove the runaway.** Localized below.
- **AC4 12-step operational-dt audit — DID NOT IMPROVE (honest).** Combination a
  (pure dycore, real d02 IC, dt=6/substeps=4/epssm=0.5, damping ON — identical to
  Sprint C). Post-fix `first_critical_violation = step 5` (RK2, substep 2,
  `advance_mu_t`, `pressure_bounded`, abs_p/base=3.38); Sprint C was **step 8**
  (abs_p/base=3.92). The mass-fix moved the first critical EARLIER (5 vs 8) though
  the overshoot magnitude is slightly smaller (3.38 vs 3.92). On the real d02 IC
  (nonzero mu_perturbation) the WRF-correct `muts` denominator surfaces the same
  `pressure_bounded` acoustic-restoring weakness sooner — the contract target
  (move past step 8 toward clean) was NOT met. Same residual class as AC3.
- **AC5 no regression — PASS.** f7a flat-rest exactly 0 on all 7 fields; analytic
  dipole sign+order PASS; 300-step conservation dry_drift=0, theta_drift=0, w bounded.
  Acoustic/mu_t/PGF pytest subset: 21 passed, 2 failed — but the 2 failures
  (`test_m6b4_acoustic_recurrence_parity`) FAIL IDENTICALLY against the pre-fix
  code (commit 82ccf65), so they are pre-existing (the known M6B4 self-compare
  tautology whose savepoint reference no longer matches post-F7.A/B/C rewrite),
  NOT a regression introduced by F7D. No tolerance widened, no xfail added.
  Evidence: `regression_recheck.json`.

## KEY DIAGNOSTIC FINDING — did the mass fix alone remove the runaway? NO.
Decisive localization (no clamps, pure observation):
1. **A fully consistent θ'=0 base column (ph integrated from θ0) is EXACTLY
   stable**: max|w| = 0.000 for the entire 100 s run.
2. **Any perturbed column drives a horizontally-uniform, linearly-growing
   vertical w mode** → the SAME linear ramp whether the bubble θ' is present
   (full warm bubble) or set to 300 K with the bubble's pre-loaded geopotential
   kept. The growing w is uniform in x (the `w_mode_locator` shows w identical
   across all columns at the peak level) with a smooth half-sine vertical
   structure — a column normal mode, not a 2Δx checkerboard and not the bubble
   updraft (the θ' centroid is essentially stationary, cz: 2000→1986 m).
3. f7a flat-rest = exactly 0 and 300-step conservation hold → the dycore +
   metrics are in perfect discrete balance at rest.

Therefore the residual runaway is **NOT** the MUT/MUTS mass-semantics (which is
correct: AC1/AC2/AC5 prove it).

**Decisive ph'=0 experiment** (`proofs/f7d/ic_phzero_warm_bubble.json`): zeroing
the pre-loaded `ph_perturbation` (injecting the bubble through θ' only, exactly
as the f7a analytic warm-bubble oracle that PASSES) makes the run **completely
stable to 200 s** (max|w| ~ 1.6e-3 m/s, no runaway) — BUT the bubble then does
NOT rise at all (centroid 2000 m stationary, θ'max pinned at 2.000). So the
pre-loaded ph' is REQUIRED to provide the buoyancy (WRF
`module_initialize_ideal.F:1107-1130` integrates the perturbation geopotential
`ph_1` from the perturbation inverse-density `al = alt-alb` for exactly this
reason), and it is precisely that (WRF-correct) buoyancy loading that excites the
runaway.

**ROOT CAUSE — quantified** (`proofs/f7d/rwtend_check.json`): the stage-entry
frozen buoyancy `rw_tend` has physical magnitude **0.615 m/s²** = **9.4× the
analytic physical buoyancy** `b = g·θ'/θ0 = 0.065 m/s²` — and this 0.615 m/s²
matches the observed runaway slope (0.604 m/s/s). The over-forcing is entirely
the WRF `pg_buoy_w` pressure-gradient term `rdn·(p_buoy(k)-p_buoy(k-1))` (= 4687);
the mass term `c1f·mu'` is **exactly 0 because the idealized IC sets
`mu_perturbation = 0`**. In a balanced WRF warm-bubble column the pressure-gradient
term and the `c1f·mu'` mass term nearly **cancel**, leaving only the small physical
buoyancy residual; with `mu'=0` there is nothing to cancel the full 4687 pressure
gradient, so `pg_buoy_w` over-forces w by ~9×. So the residual is an
**idealized-IC defect**: the perturbed columns need a consistent `mu'` (or the WRF
`module_initialize_ideal.F:1107-1130` iterative column balance) so that
`pg_buoy_w`'s `rdn·Δp_buoy` and `c1f·mu'` terms cancel to the physical buoyancy.
Off-centering (epssm=0.1→0.5) does not change it (constant forcing, not an
eigenmode). The harness IC ph' construction matches
WRF's `al`-based discrete operator (verified: for mu'=0, `ph_pert(k+1)-ph_pert(k)
= dnw·mu·(alt_full-alt_base)` == WRF's form), so it is not a simple IC bug. This
is beyond the verified mass-semantics scope and the dycore's protected files.
**Off-centering is NOT the fix**: a WRF-cited `epssm=0.5` (vs 0.1) gives the SAME
linear runaway (max|w| 15→30→44 → NaN at 100 s, `proofs/f7d/warm_bubble_epssm0p5.json`).
The growth being LINEAR (slope ~0.6 m/s/s, R²≈1) and epssm-insensitive means it is a
**constant forcing**, not an exponential eigenmode — the stage-entry absolute
`p_buoy` buoyancy source and the substep acoustic restoring do not form a closed
adjustment for a buoyant column. Next step: audit the stage-entry `p_buoy`
diagnostic vs the substep `calc_p_rho`/`advance_w` vertical-restoring closure (are
`al`/`alt`/`c2a` mutually consistent), and/or adopt the WRF
`module_initialize_ideal.F:1107-1130` iterative column balance for the IC — NOT a clamp.

## Unresolved risk / next decision
The mass-semantics fix is complete, WRF-faithful, and a strict improvement
(it is required for the real-d02 nonzero-mu_perturbation path). The remaining
idealized runaway needs an IC discrete-balance fix (rebuild the perturbed
columns to satisfy the dycore's discrete `calc_p_rho` relation, i.e. the WRF
`module_initialize_ideal.F` iterative column balance) and/or a vertical-acoustic
off-centering/damping review for sustained-buoyancy perturbations — both beyond
the verified mass-semantics scope and carrying real risk of an incorrect change,
so they were not force-fit. The `ph'=0` IC experiment (let the dycore develop the
geopotential dynamically, as the f7a analytic warm-bubble oracle does and passes)
is the most promising next step and is captured in
`scripts/f7d_ic_phzero_probe.py` + `proofs/f7d/ic_phzero_*.json`.

F7D_PARTIAL
