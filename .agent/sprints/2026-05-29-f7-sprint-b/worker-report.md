# F7 Sprint B — Worker Report (Opus 4.8 frontrunner)

## Objective

Re-enable WRF damping so the dry core runs at the operational dt, add the WRF
large-step dry tendency (flux-form advection + rk_addtend_dry), and prove it with
the published idealized cases (Straka density current, Skamarock warm bubble),
re-running the Sprint A gates. WRF Fortran source treated as ground truth.

## Files changed

- `src/gpuwrf/dynamics/core/advance_w.py` — added `w_damp_vertical_cfl`
  (WRF `w_damping=1` vertical-CFL limiter, `module_big_step_utilities_em.F:2714-2774`,
  `w_alpha=0.3`/`w_beta=1.0` from `share/module_model_constants.F:88-89`) and the
  `damp_opt=3` implicit Rayleigh top-damping (`module_small_step_em.F:1559-1572`,
  `dampmag=dts*dampcoef`, `hdepth=zdamp`, sin² ramp) applied after the Thomas
  back-substitution using the uncoupled `small_step_prep` `w_save`.
- `src/gpuwrf/dynamics/core/acoustic.py` — `AcousticCoreConfig` damping fields;
  `AcousticCoreState.w_save` + `AcousticCoreState.p_buoy`; `acoustic_substep_core`
  now (a) threads damping into `advance_w_wrf`, (b) uses the absolute `p_buoy` for
  the large-step `pg_buoy_w` buoyancy when present (else the substep `p`).
- `src/gpuwrf/dynamics/mu_t_advance.py` — **bug fix**: rewrote `advance_mu_t_wrf`
  neighbour access to periodic `jnp.roll` (the haloed-interior slice arithmetic
  overran the staggered u/v faces on every non-square periodic grid → it was
  impossible to run the idealized cases at all).
- `src/gpuwrf/contracts/grid.py` — **bug fix**: `DycoreMetrics.flat` now computes
  `dn(k)=0.5*(dnw(k)+dnw(k-1))` (WRF `module_initialize_ideal.F:711-727`) instead
  of `dn=dnw` (only valid for uniform eta; wrong on the hydrostatic eta grid).
- `src/gpuwrf/ic_generators/idealized.py` — eta-hydrostatic IC builder; **pure-sigma
  metric override** (c1=1,c2=0; the `.flat` hybrid c1f=eta gave zero top-face dry
  mass → singular `calc_coef_w` → NaN); per-column hydrostatic geopotential from the
  full θ (the out-of-balance signal that drives buoyancy); `force_fp64=True`;
  constant-K ν=75 for Straka; flux-form advection on.
- `src/gpuwrf/dynamics/flux_advection.py` (NEW) — WRF flux-form mass-coupled scalar
  advection (h=5/v=3, `module_advect_em.F:3029-4359`) + periodic mass-coupled
  velocities. `flux5` verified 5th-order on 1-D linear advection.
- `src/gpuwrf/dynamics/explicit_diffusion.py` (NEW) — WRF 6th-order monotonic filter
  (`module_big_step_utilities_em.F:6504-6920`) + constant-K diffusion (Straka ν).
- `src/gpuwrf/runtime/operational_mode.py` — damping/diffusion/flux/`force_fp64`
  namelist plumbing; `_augment_large_step_tendencies`; absolute-`p_buoy` diagnostic;
  `force_fp64` precision path (F7-B is fp64-correctness-only).
- `scripts/f6_transaction_audit.py` — `--damping` + `--diff-6th-opt` flags.
- `tests/` — the 3 AC5 red tests fixed (below).
- `proofs/f7b/**` — proof objects.

## Commands run (all `taskset -c 0-3`, `cuda:0`, fp64)

- `scripts/f6_transaction_audit.py --steps 12 --dt-s 6 --acoustic-substeps 4
  --epssm 0.5 --combination a --damping [--diff-6th-opt 2]` → `proofs/f7b/audit_operational_dt*`
- `scripts/f7a_oracles.py --conservation-steps 300 --epssm 0.5` (AC4 regression, ×4 across changes)
- `run_warm_bubble_case` / `run_density_current_case` (require_gpu=True) → `proofs/f7b/`
- 1-D linear-advection convergence check of the WRF `flux5` operator
- `pytest` on the 3 AC5 red tests + acoustic/mu_t regression suites

## Acceptance gate status

- **AC1 Straka density current — NOT MET (RAN_TO_COMPLETION, verdict FAIL).** The
  case now runs end-to-end on GPU (was BLOCKED/never-ran in F2), but goes non-finite
  by 900 s (`all_snapshots_finite=False`): the cold pool needs the horizontal
  circulation to spread the front, which is the missing large-step PGF/transport
  coupling (below); over the long 9000-step integration the unbalanced vertical
  motion + const-K (ν=75) diffusion destabilises. Front position not reached.
  `proofs/f7b/straka_density_current_*`.
- **AC2 Skamarock warm bubble — PARTIAL: RAN_TO_COMPLETION, verdict FAIL.** Finite
  for the full 5000 steps, dry-mass drift ≤1e-8, bounded θ′; the buoyancy now drives
  coherent upward `w` (0→3→11→23→41 m/s). FAILS `thermal_rise` and the `max|w|≤30`
  ceiling because the warm parcel rises buoyantly but the θ′ field is **not yet
  transported** with the motion and the flow does not recirculate — the remaining
  Block-2 coupling gap (below). `proofs/f7b/skamarock_bubble_*`.
- **AC3 12-step operational-dt audit — PARTIAL.** With WRF damping ON at dt=6/epssm=0.5,
  the transients are **physical** (`w~17 m/s`, `u~25 m/s`) for the first ~5 steps —
  the O(≤100 m/s) AC3 target — vs Sprint A's immediate (step-4) detonation to `w~1.5e4`.
  A residual growing mode still escapes near step 6-7 and `first_critical_violation`
  is not null over 12 steps (theta floor at step 6, blow-up by step 8). No masking
  clamp used. The WRF 6th-order filter is wired but `diff_6th_factor=0.12` alone does
  not close it; the full operational stabiliser set (km_opt=4 Smagorinsky) +
  resolving the Block-2 PGF/transport coupling is the remaining work.
  `proofs/f7b/audit_operational_dt/`, `audit_operational_dt_diff6/`, `audit_summary.md`,
  `damping_dt_sweep.json`.
- **AC4 no regression — PASS.** Sprint A oracles re-run (flat-rest exact 0, analytic
  dipole sign+order, 300-step conservation: dry-drift 0, theta-drift 0, bounded) —
  unchanged across every commit (damping defaults OFF on the oracle path).
  `proofs/f7b/{flat_rest,analytic_acoustic_oracle,conservation_long_run}.json`.
- **AC5 three red tests — 2 of 3 fixed; #2 still red (honest, see below).**
  (a) `test_ph_tend_matches_validation_bound_theta_delta_formula` — PASS: updated to
  assert the new WRF `advance_w` geopotential finish (the 0.01·Δθ stub is gone by
  design; INV-6-compliant since the asserted code was a deleted stub).
  (c) `test_mu_persistence_two_substeps` — PASS: replaced the unphysical zero-geopotential
  fixture with a discretely-hydrostatic column + pure-sigma metrics.
  (b) `test_step2_operational_theta_stays_finite` — STILL RED. The contract expected
  damping to fix it; I enabled the WRF damping (w_damping=1, damp_opt=3, dampcoef=0.2,
  zdamp=5000) + fp64 on the comparator namelist, but it still goes NaN. This test
  exercises the LEGACY non-prep `_operational_acoustic_substep_core` single-substep
  path on the real d02 IC, which has a separate first-substep defect that damping
  (a multi-step stabiliser) does not address. Left honest — damping is wired and
  active; the failure is in the legacy non-prep path. No tolerance widened, no xfail.

## Root-cause findings (5 structural bugs, all blocked the idealized cases at step 1)

1. `advance_mu_t_wrf` periodic-indexing (broke every non-square periodic grid).
2. `DycoreMetrics.flat` `dn=dnw` (wrong mass-level spacing on non-uniform eta).
3. `.flat` hybrid c1f=eta → zero top-face dry mass → singular `calc_coef_w` (NaN).
4. fp32-gated u/θ (ADR-007) detonated the warm-bubble 2 K perturbation → `force_fp64`.
5. Static-bubble buoyancy: the acoustic small-step is delta-from-reference, so a
   balanced parcel present in both reference and current produced zero work-deltas
   and zero buoyancy; fixed by feeding `pg_buoy_w` the **absolute** diagnostic p′
   (WRF rk_step_prep), which made the warm bubble rise.

## Unresolved risks / remaining gaps (precise)

- **Block-2 large-step horizontal PGF + scalar transport (the AC1/AC2/AC3 blocker).**
  WRF `rk_tendency` puts the horizontal PGF in the large-step `ru/rv_tend`
  (`module_em.F:1325`) — Sprint A dropped it to avoid double-counting the small-step
  `advance_uv` PGF, but WRF keeps both. A prototype restoration produced a nonzero
  u-tendency (~0.6 m/s/s from the absolute p′) but did NOT move u in the integrated
  state: the operational tendency cadence applies tendencies in `add_scaled_tendencies`,
  `advance_uv`, AND reconstructs in `small_step_finish`, so without WRF `rk_addtend_dry`
  (field-specific map/mass-coupled merge) the contributions do not net to the physical
  acceleration. So the parcel rises (vertical buoyancy works) but does not circulate or
  transport θ′. **`rk_addtend_dry` + the proper RK tendency cadence is the single
  remaining Block-2 piece** and is the gate to AC1/AC2 and likely the residual AC3
  growing mode.
- `rhs_ph` (large-step ph_tend terms 1+2) not implemented; for the rest/bubble it is
  a secondary correction (the dominant geopotential terms are already in `advance_w`).
- AC3 residual instability also wants the km_opt=4 Smagorinsky path (not wired).
- Two source-string tests (`test_advance_mu_t_outputs_are_committed_by_shared_core`,
  `test_w_coefficients_and_dt_sub_follow_contracted_acoustic_cadence`) are red, but
  were **already red at this sprint's start** (commit e4c6677) — they assert
  pre-Sprint-A code patterns Sprint A removed; out of this sprint's AC5 scope.

## WRF-vs-contract discrepancies noted

- Contract Block 2 said "for the periodic dry gate [flux advection] is inert" — false;
  the idealized cases genuinely need flux-form advection + the large-step PGF/buoyancy.
- Sprint A's "horizontal PGF is small-step only, large-step double-counts" note is not
  WRF-faithful: WRF `rk_tendency` includes the PGF in the large-step (`module_em.F:1325`).
  WRF wins per the cardinal rule; documented for the rk_addtend_dry sprint.

## Verdict

**F7B_PARTIAL.** Block 1 damping is implemented WRF-faithfully (w_damping CFL +
damp_opt=3 Rayleigh, verified to engage and to make the operational-dt transients
physical for ~5 steps vs Sprint A's immediate detonation). Five structural bugs that
blocked the idealized cases were found and fixed; the dry core now runs the warm
bubble finite and mass-conserving for 5000 steps with a physically rising thermal
(buoyancy works). AC4 fully re-verified; 2 of 3 AC5 red tests fixed honestly (no
weakening). The single remaining gate to AC1/AC2/AC3-clean is the WRF large-step
dry-tendency merge (`rk_addtend_dry` + horizontal PGF cadence), which makes the
buoyantly-rising parcel circulate and transport θ′. An honest partial: the damped
stable core (Block 1 + AC3 physical-magnitude + AC4) is delivered; the large-step
transport coupling (the rest of Block 2) is precisely localized for the next sprint.

F7B_PARTIAL
