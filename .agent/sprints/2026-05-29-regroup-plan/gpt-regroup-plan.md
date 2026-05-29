# GPT-5.5 Blind Regroup Plan

## Position

The reset roadmap should be compressed and re-ordered. Once the F7 dry dycore is closed, the fastest credible path is not "M9 then M11 then M17 then M12 then M13" as mostly serial work. The fastest path is:

1. Freeze the coupled interfaces and WRF savepoint schema.
2. Turn pristine WRF v4.7.1 into a per-scheme oracle factory.
3. Run Thompson, surface/MYNN, radiation/land-driver, lateral-boundary/static-field, corpus-build, and perf-smoke lanes in parallel.
4. Recompose only after per-scheme WRF deltas are explained.
5. Run single-case skill, then the already-built 30-case TOST corpus, then final perf recertification.

The critical path to M19 is: F7 closure -> interface/oracle freeze -> per-scheme parity gates -> coupled cadence/guard-free integration -> single-case skill. The critical path to M21 is: M19 plus a validation corpus that must start immediately in parallel, not after M19.

Fastest credible wall-clock from dycore close:

- No full Noah-MP required: 12-16 weeks.
- Full/prognostic Noah-MP required: 20-28 weeks.

Do not claim a shorter path unless M19 proves land-surface prognostics are not the T2 limiter on at least a 5-case mini-ensemble.

## Revised Milestone Graph

### Gate 0: F7 Dycore Closure - 2-5 days remaining

Close the Straka touchdown residual before Phase B merges. Acceptance:

- Skamarock warm bubble passes 6/6.
- Straka density current reaches 900 s with front near published/WRF reference and min theta prime near -9 to -10 K.
- WRF per-acoustic-substep diff at touchdown identifies no unexplained continuity / horizontal-spreading discrepancy.
- GPT pre-close critique done.

Parallel work allowed before this closes: WRF oracle extraction planning, corpus discovery, CPU baseline inventory, and perf measurement scripts only. No model-code Phase B merge before F7 closure.

### Gate 1: Interface + Oracle Freeze - 3-5 days

This replaces the old M8/M9 serial start. Freeze before dispatching parallel implementers:

- State/coupler interfaces for physics tendencies, surface diagnostics, radiation diagnostics, land fields, and boundary fields.
- Savepoint schema extension for physics operators, with variables, units, staggering, precision, tolerance ladders, source run, and checksums.
- One "oracle manifest" per scheme committed; large WRF outputs stay outside git.
- Diagnostic harness expected-field scopes updated so identity on physically inactive columns is not mistaken for a missing operator.

Proof object: `phase_b_oracle_freeze.json` with schema versions, owner file sets, forbidden cross-edits, and first WRF extraction commands.

### Phase B Parallel Lanes - 3-6 weeks wall-clock

Run these lanes concurrently after Gate 1. Each lane owns its code paths and proof objects. The manager merges only after the lane passes WRF-oracle parity and a coupled diagnostic harness run.

#### Lane B1: Thompson Microphysics - 2-4 weeks

Current code is a useful source/sink subset but explicitly keeps sedimentation out of scope, and the 20260521 initial columns are cloud-free/subsaturated, so identity is not proof. Make Thompson a real WRF-oracle-driven lane.

Required WRF savepoints from pristine WRF:

- `mp_gt_driver_pre`: T, p, rho, dz, qv/qc/qr/qi/qs/qg, Ni/Nr, F_QV/F_QC/F_QR/F_QI/F_QS/F_QG if WRF uses tendency arrays, precipitation accumulators, effective radii inputs.
- Post hydrometeor clipping / number initialization.
- Post warm-rain autoconversion/accretion.
- Post ice nucleation/freezing/deposition/sublimation.
- Post saturation adjustment.
- Post rain evaporation.
- Post melt/freeze.
- Post sedimentation / fallout / surface precip fluxes.
- `mp_gt_driver_post`: all species, number concentrations, theta/T tendency, rain/snow/graupel/ice accumulators.

Acceptance:

- At least one moist/cloudy WRF column pack plus one Canary full-domain sampled slab. Do not use only the cloud-free 20260521 start.
- Species and latent-heating deltas pass predeclared tolerances against WRF savepoints.
- Sedimentation either implemented or explicitly proved irrelevant to T2/U10/V10 margins on the mini-ensemble; default should be to implement it.
- Harness shows Thompson ACTIVE only on physically active cases, and PASSIVE/INACTIVE is classified honestly on dry cases.

#### Lane B2: Surface Layer + MYNN - 2-4 weeks

This is likely the first T2/U10/V10 limiter. Existing M12 proof has HFX/LH and station skill failures; current MYNN is a dry level-2.5 path with EDMF/cloud terms disabled and surface flux coupling split across two locations.

Required WRF savepoints:

- `sfclay_pre/post`: U/V lowest level, theta/qv/p, TSK, XLAND, LAKEMASK, MAVAIL/SMOIS, ZNT/roughness, UST, MOL, PBLH, HFX, QFX/LH, T2, Q2, U10, V10, CHS/CHS2/CQS2/CM.
- `mynn_get_pblh_pre/post`: thv, qke/tke, zw/dz, pblh.
- `mym_level2_pre/post`: gradients, GM/GH, SM/SH, Richardson terms.
- `mym_length_pre/post`: EL, QKW, PBLH-related length scales.
- `mym_turbulence_pre/post`: DFM/DFH/DFQ, shear/buoyancy production.
- `mym_predict_pre/post`: QKE/TKE, dissipation, transport terms.
- `mynn_tendencies_pre/post`: U/V/theta/qv column updates, bottom BC terms, KM/KH.

Acceptance:

- HFX/LH hour-1 RMSE against WRF must drop by at least 70 percent from the stale M12 failure before single-case skill is attempted.
- U10/V10/T2 diagnostics from sfclay match WRF on nontrivial cells with predeclared tolerances.
- MYNN column-integrated theta/qv/u/v budgets close against surface flux and top flux diagnostics.
- No hidden guard fallback in accepted WRF-range states.

#### Lane B3: Radiation + Diurnal/Land Driver - 2-3 weeks for radiation, plus land decision gate

RRTMG has strong analytic/intermediate machinery and hour-1 SWDOWN looked good, but operational coupling still has gaps: model time is not fully threaded, surface albedo/emissivity are table/surrogate driven, and radiation did not move T2 skill in the stale proofs.

Required WRF savepoints:

- `radiation_driver_pre/post`: model time, JULDAY/UTC minutes, XLAT/XLONG, COSZEN, ALBEDO, EMISS, TSK, CLDFRA, qv/cloud species, pressure/temperature, RTHRATEN/RTHRATENLW, SWDOWN, SWUP, GLW, LWUP, OLR.
- `RRTMG_SWRAD_pre/post`: setcoef fields, taug/taur, cloud optical properties, MCICA mask, spcvmc fluxes, SW heating rates.
- `RRTMG_LWRAD_pre/post`: setcoef, taumol/fracs, cldprmc, rtrnmc per-g-point fluxes, LW heating rates.
- Land-driver diagnostics: ALBEDO/EMISS/ZNT/TSK source fields and their cadence.

Acceptance:

- SWDOWN and GLW hour-1 parity against WRF; radiation heating-rate vertical profiles pass WRF savepoint tolerance.
- Radiation cadence is WRF-equivalent and model time is threaded through the forecast scan, with no legacy fixed-time fallback on accepted paths.
- Diurnal T2 amplitude on land stations is within 1 K of CPU WRF on the pinned case.

Land decision gate:

- If post-B2/B3 T2 diurnal error remains >1.5 K on the pinned case or >1.0 K median on the 5-case mini-ensemble, promote M16 to the critical path immediately.
- If not, defer full Noah-MP to v0.2.0 but keep a documented prognostic/minimal lower-boundary contract for v0.1.0.

#### Lane B4: Static Fields + Lateral Boundaries - 1-3 weeks

Do this in parallel, not after physics. M19 cannot be trusted if boundaries or LU/static fields are stale.

Required WRF savepoints / comparisons:

- Static: LU_INDEX, HGT, LANDMASK, XLAND, IVGTYP, ISLTYP, VEGFRA, ALBEDO, EMISS, ZNT/roughness, LAKEMASK, SST, TSK, soil fields.
- Boundary: U, V, W, T/theta, QVAPOR, P, PB, PH, PHB, MU, MUB at every side, full spec+relax width, both interpolation endpoints and midpoint.
- First-hour boundary-strip and interior-split errors for all prognostic boundary fields.

Acceptance:

- Static fields bitwise or explainable decode-equivalent to WRF input.
- Relax-zone width and weights match WRF.
- Boundary-strip RMSE is not the dominant first-hour error source.

#### Lane B5: Validation Corpus + Statistics - 2-4 weeks, starts immediately

This is on the M21 critical path and must not wait for M19.

Deliverables:

- >=30 Canary L2/L3 cases on disk, seasonal and regime-stratified.
- CPU WRF v4 baseline run paths and hashes.
- IC/BC availability verified before GPU implementation is ready.
- Station join manifest with row counts per variable, domain, lead-hour bin, and station.
- TOST margins predeclared for T2/U10/V10 RMSE deltas, with power analysis and missing-data rules.
- 5-case mini-ensemble selected early as the anti-overfit gate before M19 is declared durable.

Acceptance:

- `case_manifest.json`, `cpu_baseline_manifest.json`, `station_join_manifest.json`, and `tost_design.json` exist before single-case skill closes.

#### Lane B6: Perf Smoke / Residency Audit - 2-3 days, then held

Run this after F7 closure and again after Phase B integration. Do not do a major optimization sprint before physics interfaces settle.

Acceptance:

- Warmed Nsight says `d2h_inter_kernel == 0` on the current operational path.
- Kernel count, compile time, memory peak, and speedup are recorded as estimates only.
- If speedup is already <7x after dycore closure, schedule a blocker perf sprint before adding more physics; otherwise wait.

## Recomposition Gate: Coupled Phase B Close - 1-2 weeks

After B1-B4 pass independently, merge through one composed operational path:

- WRF physics cadence: dycore RK bundle, microphysics, surface, MYNN, radiation, boundary order.
- Tendency accounting: theta/mu/p/ph updates and perturbation/total fields stay synchronized.
- Guards: zero hidden clips/fallbacks on valid WRF-range states; all limiter activations logged with first field/cell.
- Diagnostic harness: no unexplained MISSING/NOISY_ZERO verdicts under a case where each physics scheme should be active.
- Conservation: dry mass, water, and energy budgets produce proof objects even if final thresholds are tightened later.

This recomposition gate replaces the old serial M11/M12/M13/M14 closure pattern. It is the first time coupled skill should be judged.

## M19: Single-Case Skill Recovery - 1-2 weeks

Run after recomposition, not before.

Acceptance:

- Pinned 20260521 L2/L3 d02 replay.
- Same scorer and station mask as CPU WRF.
- T2/U10/V10 RMSE and MAE within 20 percent of CPU WRF.
- 5-case mini-ensemble median RMSE not worse than CPU by more than predeclared provisional margins.
- If the pinned case passes but mini-ensemble fails, do not advance to M21; triage by regime.

True M19 blockers are, in order:

1. Surface/MYNN flux parity and U10/V10 diagnostics.
2. Land/TSK/diurnal lower boundary.
3. Boundary completeness.
4. Moist/cloud/radiation interaction.
5. Dry dycore residuals, if F7 closure was incomplete.

## M20/M21: Corpus and TOST - 4-8 weeks after corpus is ready

M20 should be mostly complete before M19. M21 begins only after M19 plus the 5-case mini-ensemble pass.

Execution:

- Run GPU forecasts for all >=30 predeclared cases.
- Reuse CPU WRF baselines already built in Lane B5.
- Score T2/U10/V10 by domain, lead-hour bin, station mask, and aggregate.
- Apply TOST with predeclared margins; report CIs/effect sizes, not only p-values.
- Fail honestly on any variable/domain/lead-hour stratum that misses equivalence.

Do not tune margins after seeing GPU output. Do not drop hard cases unless the exclusion rule was predeclared.

## Performance Sequencing

Use three performance gates:

1. F7-perf smoke immediately after dycore close: detect catastrophic slowdown and D2H regressions only. No speed claim.
2. M19 economics gate after coupled single-case skill: if speedup is <8x, optimize before spending full 30-case GPU time.
3. M22 final recert after M21 correctness: official >=10x claim, warmed Nsight, transfer audit, final CPU denominator.

Major XLA fusion / fp32 downcast work belongs after Phase B recomposition and preferably after M19. Earlier optimization is allowed only for compile/OOM/blocker fixes and must be bitwise or tolerance-neutral against WRF savepoints.

## Noah-MP Decision

Full Noah-MP is not the first sprint on the critical path, but a prognostic land lower boundary probably is. Static/prescribed TSK/soil cannot be assumed adequate for 24-72 h T2 equivalence.

Decision:

- Do not block Thompson/MYNN/RRTMG parallel work on full M16.
- Do run a Noah-MP discriminator immediately after B2/B3 first coupled integration.
- Promote full/prognostic Noah-MP to critical path if land-station T2 diurnal amplitude or HFX/LH parity remains outside the thresholds above.
- Defer full canopy/snow/groundwater complexity to v0.2.0 only if the 5-case mini-ensemble proves T2 equivalence is achievable with the smaller verified lower-boundary model.

Minimum defensible v0.1.0:

- Closed F7 dycore with WRF/public idealized proof.
- WRF-savepoint-validated Thompson, sfclay/MYNN, RRTMG, static fields, and LBC.
- A verified prognostic or WRF-equivalent lower-boundary path sufficient for T2/U10/V10 24-72 h skill.
- 5-case mini-ensemble anti-overfit pass before M21.
- >=30-case TOST equivalence on T2/U10/V10.
- Final >=10x speedup with no timestep-loop transfers.

Cut/defer:

- Full general WRF replacement scope.
- Non-Canary domains.
- Precip/gust/RH equivalence as release gates, though precipitation accumulators must exist if needed for Thompson/land water budgets.
- ArXiv polish until after technical gates; keep release evidence first.
- Full Noah-MP only if the discriminator proves it is not needed for v0.1.0.
- Cosmetic refactors, broad API redesign, and performance hero work before correctness freeze.

Do not cut:

- Boundary completeness.
- Static-field parity.
- Surface flux / land diurnal evidence.
- 30-case corpus and TOST.
- Transfer audit.

## Biggest Risks and Early De-Risking

1. Corpus not ready when M19 closes.
   Mitigation: B5 starts now; CPU baselines and station joins are built while physics agents work.

2. Static/prescribed land surface cannot recover T2.
   Mitigation: explicit Noah-MP discriminator on pinned case plus 5-case mini-ensemble before M19 close.

3. Surface/MYNN errors dominate U10/V10 and T2.
   Mitigation: sfclay and MYNN WRF savepoints before coupled debugging; HFX/LH and U10/V10 are first-class parity fields.

4. Thompson looks inactive because the initial case is dry.
   Mitigation: moist/cloudy WRF column pack and Canary slab with nonzero hydrometeors; harness distinguishes physically inactive from missing.

5. Performance collapses after correctness.
   Mitigation: perf smoke after F7, economics gate after M19, final M22 only after M21; no stale speed claims.

6. Overfitting to 20260521.
   Mitigation: 5-case mini-ensemble before M19 close and >=30-case TOST already selected before tuning.

7. Parallel workers collide.
   Mitigation: Gate 1 file ownership. Thompson owns `physics/thompson_*` and its adapter fields; surface/MYNN owns `surface_*`, `mynn_*`, bottom-BC adapter; radiation owns `rrtmg_*` and time/radiation diagnostics; boundary/static owns IO/boundary/static loaders; no shared core edits without manager merge branch.

## Next Sprints to Dispatch After F7 Closes

### Sprint 1: Phase B Interface + Oracle Freeze

Duration: 3-5 days. One manager/frontrunner plus GPT critique.

Outputs:

- Physics savepoint schema and manifests.
- Frozen coupler/state diagnostic interfaces.
- File ownership map.
- First extraction commands against `/home/enric/src/wrf_pristine/WRF`.
- Diagnostic harness scope corrections for physically inactive operators.

### Sprint 2A: Thompson WRF-Oracled Parity

Duration: 2-4 weeks. Runs in parallel with 2B/2C/2D/2E.

Outputs:

- WRF savepoints around `mp_gt_driver` and key internal process boundaries.
- JAX-vs-WRF comparator.
- Moist/cloudy and Canary-slab proof.
- Decision on sedimentation implementation; default implement.

### Sprint 2B: Surface/SFCLAY + MYNN WRF-Oracled Parity

Duration: 2-4 weeks. Runs in parallel.

Outputs:

- SFCLAY savepoints and HFX/LH/U10/V10/T2/Q2 parity.
- MYNN subroutine savepoints and column budget proof.
- Coupled surface->MYNN bottom-BC proof with no duplicate or sign-wrong flux application.

### Sprint 2C: Radiation/Diurnal + Land Discriminator

Duration: 2-3 weeks for radiation, 1 additional week for discriminator if needed. Runs in parallel.

Outputs:

- Radiation-driver, SW, and LW savepoints.
- Model-time/radiation-cadence proof.
- SWDOWN/GLW/heating-rate parity.
- TSK/land diurnal discriminator deciding whether M16 is critical.

### Sprint 2D: Static Fields + Complete LBC

Duration: 1-3 weeks. Runs in parallel.

Outputs:

- Static-field parity manifest.
- Full U/V/W/theta/QV/P/PB/PH/PHB/MU/MUB boundary proof.
- Interior-vs-boundary first-hour split.

### Sprint 2E: Corpus + TOST Buildout

Duration: 2-4 weeks initial, then continues as runs complete. Runs in parallel.

Outputs:

- >=30-case manifest with IC/BC availability.
- CPU WRF baseline manifest.
- Station join manifest.
- Predeclared TOST margins and 5-case mini-ensemble.

### Sprint 3: Coupled Recomposition + Guard-Free 24h Pinned Case

Duration: 1-2 weeks after 2A-2D land.

Outputs:

- Fresh diagnostic harness with all expected operators active or explained.
- Conservation/closure first proof.
- Pinned 24h run without hidden guard fallbacks.
- M19 scorer ready.

### Sprint 4: M19 Skill + Mini-Ensemble Gate

Duration: 1-2 weeks.

Outputs:

- 20260521 T2/U10/V10 RMSE and MAE within 20 percent of CPU WRF.
- 5-case mini-ensemble provisional pass.
- Go/no-go decision for M21 or mandatory M16.

### Sprint 5: M19 Economics Perf Gate

Duration: 2-5 days, parallel with Sprint 4 scoring where possible.

Outputs:

- Warmed Nsight transfer audit.
- Speedup estimate under coupled code.
- Decision: proceed to 30-case M21, do targeted perf sprint, or block for correctness/perf interaction.

REGROUP_PLAN_COMPLETE
