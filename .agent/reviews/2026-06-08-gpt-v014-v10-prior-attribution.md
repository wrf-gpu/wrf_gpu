# V014 V10 / Wind-Divergence Prior Attribution

Date: 2026-06-08
Scope: read-only synthesis. No model-code edits. No GPU runs.

## Manager Takeaway

Do not rediscover these as live root causes: the normal-momentum boundary defect and the missing large-step Coriolis term were real, fixed, and are ancestors of the current line (`a6438276`, `5319b8d7`). The remaining open item is the stricter v0120/v014 WRF-equivalence / V10 grid-divergence problem, not the old v010 "winds beat persistence" problem.

The current evidence points away from a single diagnostic sign bug. V10 grid RMSE fails on 3/3 v014 cases, but the bias signs are `-, -, +`, and station-vs-grid outcomes disagree. The next useful work is current-code attribution: spatial/vertical anatomy first, then first-divergence/tendency localization.

## Already Known

| Fact | Proof / location | Manager implication |
|---|---|---|
| KI-9 remains open for 24h d02 WRF equivalence. | `docs/KNOWN_ISSUES.md`; `docs/equivalence-demo.md`; `proofs/v0120/equivalence_demo_20260509_d02_FINAL.json`. Post-PSFC-fix d02 replay is `NOT_EQUIVALENT`: U10 pooled RMSE `2.237 > 1.5`, V10 `2.441 > 1.5`, 3D U `3.167 > 1.8`, 3D V `8.130 > 1.8`, PSFC `415.3 > 120`; T2/RAINNC/W/QVAPOR pass. | Treat wind divergence as a current equivalence blocker. T2/QV/RAINNC passing argues against broad model blow-up. |
| The v0120 wind error grows with lead time, especially 3D V. | Same final proof. 3D V RMSE is about `0.17` at h1 and grows to about `11` m/s by h19-h24; U10/V10 cross tolerance later. | Look for accumulated tendency/cadence/feedback error, not just t0 initialization. |
| PSFC diagnostic offset was fixed, but residual PSFC error is dynamic. | `proofs/v0120/psfc_extrapolation_proof.json`; `proofs/v0120/equivalence_demo_20260509_d02_FINAL.json`. PSFC pooled RMSE improved `707.8 -> 415.3` Pa but still fails and tracks wind/mass divergence. | Do not spend another round on the already-fixed PSFC diagnostic offset as the wind root cause. |
| v014 grid diagnostics show V10 grid failure across cases, with mixed signs and station/grid mismatch. | `proofs/v014/v10_grid_diagnostics.md/json`. V10 grid RMSE > `1.5` on `3/3`; signs `-, -, +`. Case `20260501...` has grid RMSE `2.524`, bias `+1.036`, block sign flip from `0-6h` bias `-0.771` to `6-12h` `+1.979`; station V10 paired delta is inside margin on 2/3 cases. | This is not a clean one-sign diagnostic bug. Stratify by region, lead, mask, and vertical profile before changing physics. |
| Old wind-skill case2 boundary spike was real and fixed. | Branch `worker/opus/wind-skill`; commit `a6438276` `[wind-fix] WRF spec_zone+relax_bdy on NORMAL momentum INSIDE acoustic loop`; ancestor of current HEAD. Evidence: `proofs/wind/WIND_SKILL_ROOT_CAUSE.md`, `proofs/wind/skill_postfix_24h.txt`, `proofs/wind/gpu_wind_localize_FIXED.json`. Case2 24h V10 RMSE improved `3.275 -> 1.565`, U10 `2.539 -> 1.671`. | Do not reopen "normal boundary momentum applied only after the step" unless a current-code regression probe proves boundary-frame dominance. |
| Old case3 missing-Coriolis root cause was real and fixed. | Branch `worker/opus/v10-momentum`; commit `5c6dd380` found missing Coriolis; commit `5319b8d7` added it and is an ancestor of current HEAD. Evidence: `proofs/wind/case3_v10_momentum_budget_findings.md` on branch; `proofs/wind/coriolis_fix_verdict.md`; `proofs/wind/coriolis_case3_v10_budget.json`. Case3 water V10 skill improved `-0.132 -> +0.169`; U10 `-0.003 -> +0.361`. | Current work should not assume Coriolis is absent. If suspected, prove a current regression with a component-tendency check. |
| v010 d02 validation passed under the then-current skill gate after those fixes. | Commit `7c864fa1` `[v010] D02_VALIDATED...`; `proofs/v010_validation/v010_d02_result.json` has `all_pass`, all cases passed, base model `Coriolis-corrected dycore (HEAD 5319b8d)`. | v010 success does not close v0120/v014 strict WRF-equivalence. Keep the gate distinction explicit. |
| v040 native-init/standalone V10 work is related history but not identical to current d02 replay KI-9. | `.agent/decisions/V0.4.0-CLOSE.md`; `proofs/v040/v040_close_proof.json`; `.agent/tasks/V0.4.0-WIND-BIAS-CARRYOVER.md`. v040 closed native init + LBC parity, explicitly carrying standalone near-surface westerly bias open. | Do not mix v040 d01 native-init h1/h2 smoke fixes with current d02 replay equivalence without a bridging probe. |

## Dead Ends / Low-Return Repeats

| Attempt | Evidence | Result |
|---|---|---|
| Base-state inversion / PSFC diagnostic offset as V10 fix. | `.agent/reviews/2026-06-02-gpt-v040-baseimbalance-fix.md`; `proofs/v0120/psfc_extrapolation_proof.json`. | Fixed PSFC diagnostic component, but V10/wind divergence remained. |
| Native-init roughness placeholders, first-hour LBC leaf cadence, Charnock ZNT update, duplicate end-step normal forcing, cold-start QKE seed. | `.agent/reviews/2026-06-02-opus-v040-v10resid-fix.md`; `.agent/reviews/2026-06-03-opus-v040-v10b.md`; `.agent/reviews/2026-06-03-gpt-v040-v10spinup.md`. | Helped v040 h1/h2 native-init smoke numbers, but did not close full 24h standalone or v0120/v014 d02 equivalence. |
| Dry-mass continuity loop bounds for specified/nested LBCs. | Commit source `a1def01`, consolidated `f3a0241`; `.agent/reviews/2026-06-03-gpt-v040-mu-continuity-fix.md`; `proofs/v040/mu_continuity_savepoint_parity.json`. | Real WRF-faithful fix and savepoint pass, but forecast wind bias did not collapse; PSFC could worsen. |
| PGF `al/alb/muts` and `dpn/cfn/cfn1` fixes. | Source `b827469`, consolidated `6fb6a51`; `.agent/reviews/2026-06-03-gpt-v040-pgf-inloop.md`; `.agent/reviews/2026-06-03-opus-v040-boundary-transport.md`; `proofs/v040/boundary_transport_savepoint_parity.json`. | Real fix, but 24h interior wind impact was about `0.006` m/s; not the bias driver. |
| Momentum-advection "5x divergence" theory from full-field boundary rings. | `.agent/reviews/2026-06-03-opus-v040-boundary-transport.md`. | Ring-dominated false positive; interior advection was bit-identical to WRF at same state. |
| MYNN `s_aw` floor omission. | Source `770287a`, consolidated `adfd3a9`; `.agent/reviews/2026-06-03-gpt-v040-r4-saw.md`; `proofs/v040/r4_saw_floor_savepoint_parity.json`. | Real savepoint fix, no wind/pressure collapse. |
| Split-explicit `php` freeze. | Source `a668238`, consolidated `b6f610f`; `proofs/v040/r5_php_freeze_savepoint_parity.json`. | Real savepoint fix, no evidence it closes the wind issue. |
| KF/cumulus as near-surface wind cause. | `.agent/decisions/V0.4.0-CLOSE.md`; `proofs/v040/forecast_gate_kf2date6h_COMPARE.json`; CPU-WRF cu0/cu1 comparison noted in close doc. | JAX and CPU-WRF falsified as material U10/V10 driver. KF-eta has no momentum tendency. |
| Pure 10m surface diagnostic / neutral-log branch as dominant V10 cause. | `proofs/wind/WIND_SKILL_ROOT_CAUSE.md`; `proofs/wind/case3_wind_residual_findings.md`; `proofs/wind/gpt_sidecar_verdict.md`. | Diagnostic cannot explain prognostic k0-k4 wind vector error. In case2 za was about 25.5 m and CPU/JAX used same PSIX10/PSIX branch. |
| MYNN tuning/off switch as case3 cure. | Branch `worker/opus/v10-momentum`; `case3_v10_momentum_budget_findings.md`. | MYNN-off worsened vector skill; MYNN increments were too small / mitigating. Do not tune blindly. |
| v040 expand-dates closure. | `.agent/reviews/2026-06-03-opus-v040-expand-dates.md`; `.agent/tasks/V0.4.0-WIND-BIAS-CARRYOVER.md`. | Data-blocked by purged/dangling `met_em`; not a model result. |

## Ranked Current Root-Cause Hypotheses

1. **Current-code low-level prognostic wind tendency divergence after the already-merged boundary and Coriolis fixes.**
   Why: v0120 grows from near-short-lead agreement into large 3D V/U10/V10 divergence; PSFC residual follows dynamics; v014 grid failures are real but sign-varying. This could be remaining dycore/source-tendency assembly, cadence, mass-wind coupling, or a current regression, not the old missing-Coriolis/normal-boundary bug.

2. **Coupled surface/PBL/radiation feedback or cadence error that becomes a wind error over hours.**
   Why: v040 full-gate behavior after h2 suggested diurnal surface/PBL/radiation coupling; v013/v014 still have off-by-default or recently integrated levers such as moisture flux advection in RK3, MYJ+Janjic operational mode, and clear-sky diagnostics. This is plausible only if probe evidence shows dycore component tendencies are initially consistent.

3. **Scoring/corpus/mask stratification problem hiding multiple regimes rather than one code bug.**
   Why: v014 grid V10 fails on all 3, but station TOST fails on only 1/3 and bias signs change. The manager should know whether the failure is boundary-frame, deep-ocean/interior, land/coast, elevation, or station-representativeness before assigning a physics owner.

Low-priority unless contradicted by fresh evidence: absent Coriolis, post-step-only normal boundary enforcement, PGF `al/dpn`, KF/cumulus, pure 10m diagnostic, and old v040 t0 native-init assembly.

## Next 3 Falsifiable Probes

| Probe | Cost | Pass / fail evidence |
|---|---:|---|
| **1. Current-code spatial/vertical V10 anatomy from existing artifacts.** Recompute or extend `proofs/v014/v10_grid_diagnostics.*` to stratify V10/U10/PSFC/T2 by lead block, land/water/coast, boundary frame vs interior, Tenerife box vs rest, and k0-k5 wind profile where wrfouts exist. | CPU: minutes to <1 hour if wrfouts are local. GPU: 0. | Pass for boundary hypothesis if excluding the 5-cell frame collapses V10 RMSE or sign. Pass for prognostic-column hypothesis if k0-k5 share the V10 sign and interior/deep-ocean remains bad. Pass for scoring hypothesis if station-success cases are grid-fail only in masks not represented by stations. Proof object: `proofs/v014/v10_spatial_vertical_attribution.{json,md}`. |
| **2. Current-code first-divergence / component-tendency audit on one failing case.** Use 20260509 d02 final-equivalence case or 20260501 full-wrfout v014 case. On CPU first, compare WRF/JAX same-state large-step components around h0-h3: PGF, Coriolis, advection, diffusion, boundary/spec-relax, physics/source-tendency folding, and resulting `ru/rv/mu` updates. | CPU: hours if JAX CPU is used, less if reduced to sampled tiles/columns and stored WRF states. GPU: 0 for initial sidecar-safe audit; optional later only with manager approval. | Pass for dycore/source assembly if one component residual is >10% of WRF term before physics feedback. Pass for physics/cadence if dry dycore components match at same state and divergence appears after source-tendency folding. Fail if no component explains h1-h3 growth, forcing probe 1 to guide a different case/region. Proof object: `proofs/v014/v10_current_tendency_attribution.json`. |
| **3. Short managed A/B gate only after probes 1-2 pick the owner.** Run 6h first, 24h only if 6h moves: selected toggles such as `moist_adv_opt`, radiation/source-tendency cadence, MYJ+Janjic vs MYNN control, or a specific boundary/cadence toggle if probe 2 implicates it. | CPU: minutes for setup/scoring scripts. GPU: future manager-authorized jobs only; estimate 1 baseline plus 1 job per variant for 6h, then 24h confirmation for winners. This report did not run GPU. | Pass if V10 grid RMSE/bias improves >=30% on the implicated region/lead without T2/QV/PSFC regression and with vertical-profile movement in the expected direction. Fail if movement is <10%, sign-trades across cases, or improvement is station-only while grid error remains. Proof object: `proofs/v014/v10_ab_probe_<owner>.json`. |
