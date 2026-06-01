# d03 1 km T2 +1.5 K warm bias — in-acoustic-loop nested ph'/w boundary forcing (FINDINGS, STOP+report)

Date: 2026-06-01
Agent: Opus 4.8 MAX (worker/opus/final-verdict)
Owned: `dynamics/core/acoustic.py`, `coupling/boundary_apply.py`,
`runtime/operational_mode.py` (acoustic staging), `scripts/diag/`.

## TL;DR

The WRF-faithful in-acoustic-loop nested ph'/w boundary forcing was implemented
correctly (cross-checked line-for-line against pristine WRF AND the GPT-5.5 WRF
review, which fully agreed on the mechanism). Unlike the prior end-of-step attempt
it is **dynamically STABLE — finite through every hour, no blow-up** — a genuine
advance. It **collapses the +2.7 kPa diagnostic surface-pressure error ~50%.**
BUT it **fails the short-d03 gate**: forcing ph' (and/or w) toward the DECOUPLED
hourly parent leaf injects spurious interior vertical motion (interior max|W|
~13 m/s vs CPU-WRF corpus 7.2 and the free-drift GPU baseline 6.2 m/s) that
adiabatically warms the interior theta +5..+9 K and makes d03 T2 **worse** (hour-1
RMSE 1.94 -> 5..8 K). Per the sprint's falsifiable STOP+report mandate I did NOT
force a pass: I reverted the production default to the validated-stable free-drift
behaviour (the new machinery is committed but default-OFF, fully documented) and am
reporting the mechanism.

## The WRF-faithful approach adopted (and it matched GPT's review exactly)

Pristine WRF (`solve_em.F` small-step loop) forces nested ph_2 / w_2 INSIDE the
acoustic loop, never as a decoupled end-of-step value repair:
- relax zone: `relax_bdy_dry` (module_bc_em.F:274-344) builds a relaxation
  tendency from the MASS-WEIGHTED full-level ph (and w for nests) toward the parent
  boundary leaf; `rk_addtend_dry` (module_em.F:107-110) folds it into the carried
  `ph_tend`/`rw_tend` as `+= tendf/msfty`; `advance_w` then consumes those every
  acoustic substep (ph_tend in the phi-equation rhs, rw_tend in the buoyancy/PGF) —
  so the forcing flows THROUGH the implicit (w,ph) solve, coupled with w.
- spec zone (outer 1 row): `spec_bdyupdate_ph` (module_bc_em.F:17-157) updates ph_2
  every substep AFTER advance_w + BEFORE calc_p_rho; nests also `spec_bdyupdate(w_2)`.
- end-of-step `spec_bdy_final` pins only the 1-cell spec zone (anti-round-off).

GPT's independent review (`2026-06-01-gpt-nest-ph-boundary-wrf-review.md`) reached
the identical conclusion ("the faithful dycore fix is the in-loop mass-coupled ph
boundary tendency plus spec_bdyupdate_ph, not another end-of-step hydrostatic
overwrite") and flagged the exact pitfall this run then hit: "ph directly feeds
pressure and the implicit w solve... risky."

TARGET correction vs the prior attempt: an offline check
(`scripts/diag/d03_phfix_target_check.py`) proved the prior attempt's re-derived
`_hydrostatic_ph_perturbation` target DIVERGES from the real (corpus-matching) ph'
by hundreds-to-thousands of m^2/s^2 aloft, while the state ph' equals the parent
leaf `ph_bdy` to within a few m^2/s^2 at every ring level at the IC. So the parent
leaf is the correct WRF target (= WRF `ph_b*`). I used it.

## What was implemented (committed, default OFF)

- `coupling/boundary_apply.py`: `nested_ph_relax_tendency` (mass-coupled WRF
  relax_bdytend ph -> /msfty -> add to ph_tend), `nested_w_relax_tendency`
  (nests), `spec_bdyupdate_ph_inloop` (WRF spec_bdyupdate_ph on the uncoupled ph
  work delta), `_full_ring_target_from_leaf`, `_relax_tendency_row` /
  `_scatter_relax_tendency`; three `BoundaryConfig` toggles (`nested_ph_relax`,
  `nested_w_relax`, `nested_ph_spec`), all default False.
- `dynamics/core/acoustic.py`: `AcousticCoreState` gains `ph_bdy_target` /
  `ph_save_for_spec` (default None); `acoustic_substep_core` applies
  `spec_bdyupdate_ph_inloop` after `advance_w_wrf`, before `calc_p_rho_step`
  (WRF order) when those are staged.
- `runtime/operational_mode.py::_acoustic_core_state_from_prep`: when the nested
  boundary is active (`run_boundary AND lead_seconds AND not force_geopotential`),
  folds the relax tendencies into `ph_tend_stage`/`rw_tend_stage` and stages the
  spec target — each gated on its toggle.
- `scripts/d03_replay.py`: env overrides `D03_NESTED_PH_RELAX/W_RELAX/PH_SPEC` for
  the isolation sweep.

## Validation gates

1. Idealized (Gate 1) — PASS. Warm-bubble 6/6 (thermal_rise 1924.35 m, theta' 1.92 K,
   mass_drift 0) and Straka 6/6 (front 14150 m, 4 rotors, mass_drift 2.3e-9),
   BIT-IDENTICAL to baseline (the change is a strict no-op for the doubly-periodic
   idealized path: lead_seconds None -> all targets None). The w-ph acoustic solve
   is undisturbed.
2. Short d03 6 h (Gate 2) — **FAIL the bias-collapse criterion (STOP).** All variants
   FINITE (no blow-up). Hour-1 vs corpus, interior (away from the forced ring):

   | config | psfc_bias(int) Pa | T2 interior K | interior max\|W\| m/s | T2 RMSE K |
   |---|---:|---:|---:|---:|
   | free-drift baseline (validated) | +2717 | overall +1.75 | 6.2 | 1.94 |
   | ph-relax + spec + w-relax | +1372 | +8.95 | (>12) | 8.08 |
   | ph-relax + spec | +1582 | +7.35 | 12.3 | 6.66 |
   | spec-only (1-cell pin) | +1906 | +5.46 | 13.2 | 4.98 |
   | CPU-WRF corpus (truth) | 0 | 0 | 7.2 | 0 |

   Every ph-forcing variant ~doubles the physical interior vertical velocity and
   warms the interior theta; psfc collapses but T2 RMSE goes UP. Gate 3 (24 h) and
   Gate 4 (d02) were therefore not run (would only confirm the worse-T2 trend; d02
   is unaffected since force_geopotential=True -> all toggles inert).

## Why it pumps (root cause — architectural, not a port bug)

The in-loop relax/spec forcing is correct WRF. But in a full WRF nest the child
geopotential is PROGNOSTIC and re-synced to the parent every parent step
(`med_nest_force` -> hydrostatic rebuild + bdy_interp), so the child interior never
drifts 2.6 kPa low and the boundary-ring ph residual stays small. Our v0.1.0 d03
path is a DECOUPLED hourly side-history replay: u/v/theta/qv/mu are forced from
hourly parent wrfout leaves but ph' free-drifts, so the child ph' equilibrates ~2.6
kPa low in the interior. Relaxing the ring toward the (still-correct) parent ph leaf
then becomes a LARGE sustained residual; the implicit (w,ph) acoustic solve balances
that geopotential forcing partly with vertical velocity -> a ~2x-physical w pump ->
adiabatic interior warming. This is exactly the decoupled-replay caveat the
boundary_apply module already documents for the normal momentum (which needed a
calibrated strength-20 blend); ph feeds the pressure/implicit-w solve directly, so
the same decoupled-leaf inconsistency is far more damaging here.

## Decision / next

The faithful in-loop dycore mechanism is DONE and STABLE; it is preserved
default-OFF for the real follow-up, which is NOT a dycore boundary change but the
replay-architecture fix the bisection + GPT review both name:
- proper nested re-sync (rebuild the child ph' hydrostatically + bdy_interp the
  parent every parent step, WRF med_nest_force), OR
- a true child-prognostic-ph path (drop the decoupled hourly replay for ph), OR
- the labelled stopgap (re-reference the DIAGNOSTIC surface pressure feeding T2's
  Exner in the runtime surface-diagnostic path — OUT of this sprint's dycore
  ownership; collapses the T2 artifact at the diagnostic without touching the
  prognostic geopotential, machine-exact knockout already proven 1.94->0.93 K).

v0.1.0 d03 stays on the validated-stable free-drift baseline (T2 RMSE 1.92, no
blow-up); the per-the-principal d02 3 km island proof is unaffected.
