# d03 1 km T2 +1.5 K warm bias — hydrostatic-ph' boundary fix ATTEMPT (BLOCKED)

Date: 2026-06-01
Agent: Opus 4.8 MAX (final-verdict branch)
Scope: implement + validate the hydrostatically-consistent nested-boundary ph'
forcing prescribed by the two 2026-06-01 RCAs. File owned: `src/gpuwrf/coupling/boundary_apply.py`.

## TL;DR

The root cause from the two RCAs is CONFIRMED and the prescribed *direction* is
correct, but the prescribed *mechanism* — forcing a re-derived hydrostatic
perturbation geopotential `ph'` at the nested lateral boundary at the END of each
operational timestep — is **empirically unstable**: it blows the dycore up at
forecast hour 1 (`w -> ~1e83 m/s`, `p' -> ~1e16 Pa`, NONFINITE_STATE), in BOTH the
hard-overwrite and the gentle WRF spec+relax variants. The d03 T2 bias was NOT
collapsed because no stable run was obtained. Per the sprint instruction ("if the
short run doesn't collapse the bias, report immediately rather than proceeding"),
I reverted `boundary_apply.py` to the validated-stable free-drift behaviour and am
reporting. The verified-correct hydrostatic derivation is kept in the module (dead,
documented) for the follow-up in-acoustic-loop fix.

## What was verified WORKING

- `_hydrostatic_ph_perturbation` (new): re-derives `ph'` from the forced
  mu'/theta'/qv' as the EXACT inverse of the dycore's own `diagnose_pressure_al_alt`
  (WRF `calc_p_rho_phi`, non_hydrostatic al at `module_big_step_utilities_em.F:1029`
  + hypsometric_opt==1 geopotential integration at `:1183-1193`). Anchored at the
  surface (`ph'[0]=0`). Standalone round-trip check (CPU, machine-exact):
  feeding the re-derived `ph'` back through the dycore al/EOS reproduces the
  hydrostatic pressure to 6e-11 Pa and al to 5e-15; `ph'[0]==0` exactly; eager==jit.
- Compile cost in isolation at d03 grid (44x75x93): 0.24 s (cumsum form, NOT the
  44-deep Python recurrence which compiled pathologically slow — that was fixed).
- Idealized warm-bubble (6/6 PASS, thermal_rise 1924 m, theta' 1.92 K, mass_drift 0)
  and Straka density current (6/6 PASS, front 14150 m, 4 rotors) — UNCHANGED by the
  change (idealized callers pass no `metrics`, so the branch is a no-op). Dycore
  guards undisturbed.
- 31/31 `boundary_apply` unit tests pass (pre-existing fixture-pin mismatch in
  `test_m6_boundary_replay::..._re_pinned_when_present` is unrelated — it compares two
  hardcoded run-id constants and never calls boundary code).

## What FAILED (the falsifiable short-run gate)

Two short d03 6 h GPU runs (CPU-pinned 0-3, MEM_FRACTION 0.80, detached):

| variant | application | result |
|---|---|---|
| v2 | hard overwrite of the full spec+relax ring `ph'` | PIPELINE_BLOCKED, NONFINITE_STATE @ hour 1 (`w` 5.8e107, `p'` 2.7e16) |
| v4 | WRF spec(hard outer row)+relax(Laplacian nudge) toward the hydrostatic target | PIPELINE_BLOCKED, NONFINITE_STATE @ hour 1 (`w` 6.2e83, `p'` 2.6e16) |

Proof objects: `proofs/v010_validation/pipeline_run_d03_phfix_short6h.json` (v2),
`proofs/v010_validation/pipeline_run_d03_phfix_short6h_v4.json` (v4) — both
`detail.failure_mode = NONFINITE_STATE`, `failed_hour = 1`. The free-drift baseline
(committed `d03_validation_run24h_hfxfix4.json`) is STABLE: T2 RMSE 1.92 K, bias +1.51 K.

## Why it blows up (mechanism)

`ph'` (perturbation geopotential) is an ACOUSTIC-COUPLED prognostic: the split-
explicit small step advances `ph'` together with `w` (vertical momentum) every
acoustic substep (`advance_w`/omega-ph evolution). The operational step re-seeds
the small-step prep (`ph_save`, `ph_1`) and the al diagnostic from the incoming
`state.ph_perturbation` — so injecting an externally re-derived `ph'` value at the
END of the previous step leaves `ph'` inconsistent with the carried `w`/`ww`. That
mismatch is a forcing on the w-ph acoustic mode and excites a resonance that
diverges within one forecast hour. The other forced fields (u/v/theta/qv/mu/p)
do NOT have this acoustic-coupling, which is exactly why only `ph'` forcing blows up.
Forcing strength does not matter: a weak (relax) nudge and a hard overwrite both diverge.

WRF does NOT force the nest `ph'` this way. It (a) recomputes the child geopotential
hydrostatically at vertical-interpolation / init time (`med_interp_domain`), and
(b) at the lateral boundary forces the MASS-COUPLED variables through the in-acoustic-
loop relaxation tendency (`relax_bdy_dry` on the coupled work arrays), NOT a decoupled
end-of-step `ph'` value.

## Viable paths forward (both OUT OF this file's scope)

1. **Faithful**: fold the hydrostatic-`ph'` boundary forcing INTO the acoustic small-
   step loop, coupled with `w` (mirror the existing `apply_normal_bdy_work`
   normal-momentum in-loop boundary protection, but for the w-ph pair). This needs
   a dycore/acoustic-core change (`dynamics/...`, `runtime/operational_mode._acoustic_*`).
   The verified `_hydrostatic_ph_perturbation` helper is ready to supply the target.
2. **Stable stopgap** (the bisection's own "Alternative / cheaper interim"): leave
   `ph'` free-drifting (stable) and re-reference the DIAGNOSTIC surface pressure that
   T2's Exner conversion uses. The +2.6 kPa is a near-uniform column constant; the
   offline machine-exact Exner knockout already proved re-Exnering T2 with the correct
   psfc collapses the bias (1.94->0.93 K hour 1). This is a `runtime`/surface-diagnostic
   change (`operational_mode._psfc_from_state` / `_surface_column_view`), NOT MYNN
   physics — but it touches files outside this sprint's ownership.

## Files

- `src/gpuwrf/coupling/boundary_apply.py` — reverted to stable free-drift; verified
  `_hydrostatic_ph_perturbation` + `_relax_field_to_target` helpers retained (dead,
  documented) for path 1. `apply_lateral_boundaries` gained an optional `metrics` arg
  (default None; backward compatible; unused on the now-reverted branch).
- `src/gpuwrf/runtime/operational_mode.py` — passes `namelist.metrics` to the boundary
  call (harmless; supports the future in-loop fix).
- `scripts/diag/d03_psfc_t2_check.py` — new CPU-only wrfout PSFC+T2-vs-corpus scorer.
- Proofs: `proofs/v010_validation/pipeline_run_d03_phfix_short6h{,_v4}.json` (BLOCKED).

## Next decision needed

Choose path 1 (faithful in-acoustic-loop coupled ph'/w boundary forcing — larger
dycore sprint) vs path 2 (diagnostic psfc re-reference — small, stable, fixes the T2
artifact now but not the underlying prognostic geopotential drift). Either requires
relaxing this sprint's file-ownership constraint.
