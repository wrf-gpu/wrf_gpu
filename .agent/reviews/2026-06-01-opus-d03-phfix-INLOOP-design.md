# d03 T2 +1.5 K warm bias — in-acoustic-loop nested ph'/w boundary forcing (DESIGN)

Date: 2026-06-01
Agent: Opus 4.8 MAX (worker/opus/final-verdict)
Scope: faithful fix design grounded in pristine WRF, BEFORE any GPU run.
Owned files: `dynamics/core/acoustic.py`, `coupling/boundary_apply.py`,
`runtime/operational_mode.py` (acoustic staging only).

## Why the prior attempt blew up (confirmed mechanism)

The prior agent forced a re-derived hydrostatic `ph'` over the FULL spec+relax ring
at the END of each operational timestep (after `small_step_finish`). That injected a
`ph'` value inconsistent with the carried `w`/`ww` acoustic memory, excited the w-ph
acoustic resonance, and detonated at forecast hour 1 (both hard and gentle variants).
See `2026-06-01-opus-d03-t2bias-phfix-attempt.md`.

## What pristine WRF ACTUALLY does (decisive trace)

`solve_em.F` small-step loop (`small_steps : DO iteration`, lines 3065-4363) order
per acoustic substep, for `specified .OR. nested`:

1. `advance_uv` -> `spec_bdyupdate(u_2, ru_tend)` / `spec_bdyupdate(v_2, rv_tend)`
   (solve_em.F:1346-1364) — spec_zone normal-momentum pin (we already mirror this via
   `apply_normal_bdy_work` inside `advance_uv_wrf`).
2. `advance_mu_t` -> `spec_bdyupdate(t_2, mu_2, muts)` (:1462-1490) — spec_zone.
3. `advance_w` (the implicit (w,ph) Thomas solve).
4. `sumflux`.
5. **`spec_bdyupdate_ph(ph_save, grid%ph_2, ph_tend, mu_tend, muts, c1f, c2f, dts_rk, 'h')`**
   (:1587-1597) — spec_zone (outermost 1 row) ph update, mass-reweight + tendency.
6. nested branch: **`spec_bdyupdate(grid%w_2, rw_tend, dts_rk, 'h')`** (:1609-1617) —
   spec_zone w update. (specified branch uses `zero_grad_bdy(w_2)` instead.)
7. `calc_p_rho`.

The **relaxation-zone** ph/w forcing is NOT applied as a value overwrite anywhere in
the loop. It is folded into the large-step tendencies ONCE per RK stage:

- `relax_bdy_dry` (module_bc_em.F:161-346, called solve_em.F:940 before the loop)
  builds `ph_tendf` from the MASS-WEIGHTED ph `rfield = mass_weight(ph, mut, c1f, c2f)`
  relaxed toward the boundary leaf via `relax_bdytend_tile` (the standard
  `fcx*fls0 - gcx*(fls1+fls2+fls3+fls4-4*fls0)` stencil, module_bc.F:1221-1427), and
  (for `nested` only) builds `rw_tendf` from `mass_weight(w)`.
- `rk_addtend_dry` (module_em.F:1092, called solve_em.F:968) folds them into the
  carried tendencies: `ph_tend += ph_tendf/msfty` (line 110), `rw_tend += rw_tendf/msfty`
  (line 107).
- Those total `ph_tend`/`rw_tend` are then **carried unchanged through every acoustic
  substep** and applied INSIDE `advance_w` (rhs of the phi equation gets `dts*ph_tend`;
  the vertical PGF/buoyancy gets `rw_tend`). So the relaxation forcing flows THROUGH the
  implicit (w,ph) solve — it is intrinsically coupled with w, never an after-the-fact
  overwrite. THIS is the piece the prior attempt skipped.

End-of-step: `spec_bdy_final(w_2)`/`spec_bdy_final(ph_2)` (solve_em.F:4660/4687) hard-pin
ONLY the 1-cell spec_zone to the boundary value (anti-round-off-drift). Not the relax zone.

## Mapping to our architecture

Our `ph` small-step work array (`acoustic.ph`, WRF `ph_2`) is the UNCOUPLED
perturbation-geopotential delta `ph'_ref - ph'` (small_step_finish reconstructs
`ph' = ph_work + ph_save`). Our `ph_tend` (from `rhs_ph_wrf`) is the coupled large-step
geopotential tendency `(c1f*mut+c2f)*(...)/msfty`, added to the advance_w rhs as
`dts*ph_tend`. The relaxation tendency must be in the SAME units.

### TARGET CORRECTION (offline check `d03_phfix_target_check.py`, decisive)

The prior attempt relaxed toward `_hydrostatic_ph_perturbation`. The offline check at
t=0 (IC matches corpus exactly) shows that re-derived hydrostatic target DIVERGES from
the real (corpus-matching) ph' by hundreds-to-thousands of m^2/s^2 aloft
(level 20 ~ -650, level 44 ~ -21,700) — relaxing toward it would CORRUPT the upper column.
The interpolated PARENT leaf `ph_bdy` is the correct target: the state ph' already equals
it to within a few m^2/s^2 at every ring level at IC (max ~7.9 of ~21,000). This is the
WRF-faithful target — WRF `relax_bdytend_tile(mass_weight(ph), ph_bdy_leaf)` relaxes
toward the boundary leaf, NOT a re-derived hydrostatic profile.

So the prior attempt had TWO errors: (1) wrong target (re-derived hydrostatic instead of
parent leaf), AND (2) wrong mechanism (decoupled end-of-step overwrite). The hard
end-of-step overwrite toward the parent leaf is `force_geopotential=True` (the original
pump). The faithful fix = relax toward the parent leaf THROUGH the in-loop tendency.

### Primary mechanism (relax zone) — the dominant, naturally-coupled nudge

In `_acoustic_core_state_from_prep` (once per RK stage, only when the nested boundary is
active: `run_boundary AND boundary_config.force_geopotential == False AND lead_seconds
is not None`):

1. Target `ph'_target` = the time-interpolated PARENT leaf `ph_bdy` (NOT
   `_hydrostatic_ph_perturbation`, which the offline check proved diverges aloft).
2. Compute the WRF relax tendency on the ph residual using the existing
   `_apply_side_relax` stencil and `_wrf_relax_weights` (fcx/gcx), on the residual
   `(ph'_target - ph'_stage)`, mass-couple by `(c1f*mut+c2f)` and divide by `msfty`, and
   ADD it into `ph_tend_stage`. (= WRF `ph_tend += ph_tendf/msfty`.)
3. (nested) build the analogous `rw_tendf` toward a w target (the forced `w_bdy` leaf,
   default 0 perturbation) and ADD into `rw_tend_stage`. (= WRF `rw_tend += rw_tendf/msfty`.)

Because steps 2-3 enter the tendencies that `advance_w` consumes inside its implicit
solve, the ph'/w forcing is solved jointly with w every substep — no decoupled overwrite,
no resonance. This is the whole point.

### Secondary mechanism (spec zone, 1 cell) — optional, add only if relax insufficient

A `spec_bdyupdate_ph`-style mass-reweight + tendency pin of the OUTERMOST row of
`ph_next` toward `ph'_target`, applied inside `acoustic_substep_core` after
`advance_w_wrf`. Start WITHOUT this (relax-only) since the +2.6 kPa error is a near-uniform
INTERIOR effect driven by the relax-zone equilibrating to a wrong reference; add the
1-cell spec pin only if the 6 h short run shows residual drift.

## No-regression guarantees

- Idealized (warm-bubble/Straka): doubly-periodic, no lateral boundary -> the gate is
  `lead_seconds is None` / `run_boundary False` -> the entire addition is skipped ->
  byte-for-byte unchanged. (Same gating as the existing `u_work_bdy`/`v_work_bdy`.)
- d02 self-replay: `force_geopotential=True` -> skipped -> unchanged.
- Only the d03 nested path (`force_geopotential=False`, run_boundary, lead_seconds) is
  affected.

## Validation gates (STOP+report on failure)

1. Idealized warm-bubble + Straka 6/6 PASS (unchanged).
2. Short d03 6 h: finite + the +2.6 kPa psfc error collapses (`d03_psfc_t2_check.py`).
3. Full d03 24 h: T2 RMSE 1.92->~0.9-1.1, bias +1.51->~0, beats persistence, U10/V10<7.5.
4. d02 3 km no-regression vs `v010_d02_result_hfxfix.json`.

## OUTCOME (see ...-opus-d03-phfix-INLOOP-findings.md for full detail)

STABLE (finite, no blow-up — a real advance over the prior end-of-step attempt) and
WRF-faithful, but Gate 2 FAILS the bias-collapse criterion: every ph'-forcing variant
(relax / spec / both / +w) toward the DECOUPLED hourly parent leaf injects spurious
interior vertical motion (interior max|W| ~13 m/s vs corpus 7.2 / free-drift 6.2) that
adiabatically warms interior theta +5..+9 K -> d03 T2 RMSE goes UP (1.94 -> 5..8 K) even
though psfc collapses ~50%. Per the STOP+report mandate I reverted the production default
to the validated free-drift baseline (machinery committed, default-OFF, sweep-able). The
real fix is the decoupled-replay architecture (proper nested ph re-sync), not a further
dycore boundary change.
