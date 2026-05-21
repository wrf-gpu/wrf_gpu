# M6 Milestone Plan Critical Review - Codex

Reviewer: Codex gpt-5.5 xhigh
Plan under review: `worker/codex/m6-milestone-plan-scout @ 3392d04`

Line citations refer to `.agent/sprints/2026-05-21-m6-milestone-plan-scout/m6-milestone-plan.md` from the scout branch.

Decision: **RATIFY-WITH-AMENDMENTS**. The scout plan has the right milestone shape, but it needs explicit amendments before M6-S2/S8 dispatch. The load-bearing gaps are nested `d02` boundary replay, stale RRTMG/M5-S3.y treatment, underspecified proof schemas, and a Tier-3/Tier-4/operational gate that can overclaim if left as written.

## Section 1 - Scope / Completeness Audit

### M6-S0 - Prologue: M5 Debt Closure

- Scope: stale against current M5 closeout state. The scout S0 says it absorbs "three M5 deferred blockers" but names Thompson HLO, Thompson residuals, and MYNN harness only (scout-plan:21-30). Current state has Thompson/MYNN follow-ups closed and M5-S3.y RRTMG setcoef/taumol/Planck still queued as the M5-prologue Phase-2 item that blocks coupled validation. Amend S0 to "M5 prologue gate" with completed Thompson/MYNN entries and an explicit RRTMG/radiation-conditioning gate.
- Ownership: the S0 ownership set covers Thompson/MYNN and harness files (scout-plan:26), but excludes RRTMG files. That is fine only if M5-S3.y remains a separate M5-prologue sprint with disjoint ownership; it is not fine if S0 claims all prologue debt.
- Proof objects: too generic. `artifacts/m6/prologue/` is named only as a directory (scout-plan:26). S0 needs exact artifacts such as `artifacts/m6/prologue/thompson_status.json`, `mynn_status.json`, `rrtmg_status.json`, and a prologue gate report with `ready_for_s1`, `ready_for_s2`, and `ready_for_s8` booleans.
- Missing sprint: yes, unless M5-S3.y is explicitly tracked outside M6. Either include it as "M5-prologue RRTMG Phase 2, parallel with M6-S1 but blocking S2 full forecast/S8 T2" or add a radiation-conditioning sprint.

### M6-S1 - Coupled Interface and Precision Boundary Freeze

- Scope: mostly correct. Extending ADR-002 SoA state and freezing precision boundaries is necessary before parallel work (scout-plan:35-47). However, the state/coupling contract must include handles for lateral boundary forcing, prescribed external forcing, and validation output metadata, not only hydrometeors/TKE/surface state (scout-plan:42-45). Running `d02` standalone without a boundary contract is not a valid 24h forecast.
- Ownership: S1 owns `src/gpuwrf/contracts/*` and new `src/gpuwrf/coupling/` contract files (scout-plan:40). S2 also owns `src/gpuwrf/coupling/` (scout-plan:54), so the freeze must be hard: S1 defines interfaces; S2 implements driver behavior; S3 must not opportunistically edit adapters.
- Proof objects: directionally concrete but not fully specified in the scout plan. The 100-step dummy carry and spacetime budget are named conceptually (scout-plan:45); the active S1 contract should be reflected back into the milestone plan with exact JSON paths and schema fields.
- Missing sprint: not separate if S1 is amended to freeze boundary/forcing handles. If not, add an S1.x interface addendum before S2.

### M6-S2 - Coupled Forecast Driver

- Scope: necessary, but under-specified for a nested `d02` run. The plan chooses `d02` as the 3 km domain from a five-domain nested Gen2 run (scout-plan:11-13) and then asks S2 to run that domain (scout-plan:49-60). The backfill contains `wrfbdy_d01`, not `wrfbdy_d02`; a standalone GPU `d02` forecast needs explicit lateral-boundary replay/extraction from parent/nest history or it is only a closed/cropped diagnostic case.
- Ownership: S2 ownership of `src/gpuwrf/coupling/`, `src/gpuwrf/timestep/`, and the run script is plausible (scout-plan:54). It conflicts with any S3 coupling-adapter edits unless S1 freezes adapter hooks and S3 contributes only surface modules plus tests.
- Proof objects: insufficient. "1 h smoke forecast and 6 h forecast complete" plus "profiler JSON" is not a proof schema (scout-plan:57). Add exact outputs: `artifacts/m6/coupled_driver/smoke_1h.json`, `forecast_6h_summary.json`, transfer audit, output manifest, boundary-forcing manifest, and raw profiler paths.
- Missing sprint: yes. Add `M6-S2a Gen2 backfill accessor + d02 boundary replay + regridding/I/O schema` before full S2 24h or make it the first half of S2 with separate acceptance.

### M6-S3 - Surface Layer and Noah-MP Minimum Fold-In

- Scope: operationally justified but too broad as written. The objective to make `U10/V10/T2/qv2` meaningful is right (scout-plan:62-72), but "minimum needed for `sf_surface_physics = 4`" (scout-plan:69) is not a bounded implementation spec. Full Noah-MP semantics are too large for the 30-48h estimate (scout-plan:65).
- Ownership: new surface/land modules and fixtures are disjoint enough (scout-plan:67) if the sprint does not edit coupling adapters. The phrase "except coupling adapters approved by S1" creates a merge-conflict loophole with S2 (scout-plan:67, scout-plan:74).
- Proof objects: missing exact paths and schemas. Add `surface_scope_memo.md`, `surface_static_manifest.json`, `surface_flux_budget.json`, `surface_fixture_manifest.json`, `radiation_conditioning_feasibility.json`, and `surface_operational_delta.json`.
- Missing sprint: maybe. If prescribed radiation is used, S3 needs an explicit feasibility subtask. The scout says to use prescribed Gen2 radiation tendencies "if available" (scout-plan:72) and later calls this acceptable (scout-plan:157), but the plan does not prove those 3D tendencies exist in the pinned wrfout.

### M6-S4 - Tier-2 Conservation Under Coupling

- Scope: strong but needs an explicit non-self oracle. The listed dry-mass, water, positivity, latent-heating, and NaN checks are the right families (scout-plan:82-86). The warning against tautological checks is correct (scout-plan:84).
- Ownership: `src/gpuwrf/validation/tier2_coupled.py` and `scripts/m6_run_tier2_coupled.py` are disjoint from S3/S5/S6/S7 in filename terms (scout-plan:81), but common validation loaders/regridders must be owned by one sprint or frozen before parallel work.
- Proof objects: thresholds are concrete (scout-plan:86), but output schema is not. Add `artifacts/m6/tier2/tier2_coupled_invariants.json` with fields for each budget term, source/sink term, boundary term, clamp term, precision-cast term, pass/fail, and source artifact paths.
- Missing sprint: no separate sprint if S4 requires an independent budget calculator. Without that, it risks repeating the M5-S2 mean-field self-consistency problem.

### M6-S5 - ADR-007 Full-Domain Batching Verdict

- Scope: necessary. ADR-007's 4x conditional belongs here (scout-plan:90-101, scout-plan:143-147).
- Ownership: profiling scripts, `src/gpuwrf/profiling/`, and `artifacts/m6/performance/` are mostly disjoint (scout-plan:95). The generic `scripts/` ownership needs exact filenames to avoid collision with S2/S4/S6/S7.
- Proof objects: this is the best-specified sprint, with `artifacts/m6/performance/full_domain_batching_verdict.json` named (scout-plan:97). Amend the schema to match `PERFORMANCE_TARGETS.md` exactly and add CPU-denominator metadata, warmup/compile exclusion policy, raw `ncu/nsys` paths, profiler-failure fallback fields, and transfer-count/byte fields.
- Missing sprint: no, but it needs a concrete FAIL fallback ladder. The scout only says FAIL blocks M7 and triggers discussion (scout-plan:99, scout-plan:147).

### M6-S6 - Tier-3 Short-Run Convergence

- Scope: right tier, wrong proposed oracle. TSC1.0 should test timestep sensitivity under controlled equations and forcing. Comparing `wrf_l2` and `wrf_l3` 3 km domains for the same nominal run (scout-plan:111) is a useful backfill consistency check, but it is not a clean timestep-refinement envelope.
- Ownership: `tier3_coupled.py`, `m6_run_tsc.py`, and `artifacts/m6/tier3/` are disjoint (scout-plan:108).
- Proof objects: not concrete enough. Add `artifacts/m6/tier3/tsc_envelope.json` with base/refined dt, boundary mode, forcing mode, norm definitions, per-variable/lead envelope, GPU drift, CPU/backfill sanity deltas, and regridding details.
- Missing sprint: no separate sprint if S2a provides boundary/regridding infrastructure. Otherwise S6 will need its own I/O work and will collide with S8.

### M6-S7 - Tier-4 Small-Ensemble Prototype

- Scope: acceptable as an M6 prototype if it is labeled honestly. Probtest-style per-variable tolerances are a good first method for `U10/V10/T2/qv2/precip` (scout-plan:123). Ten deterministic day-members are not a perturbed ensemble and should be called a historical operational sample, not full Tier-4 proof (scout-plan:124).
- Ownership: `tier4_probtest.py`, `m6_run_tier4.py`, and `artifacts/m6/tier4/` are disjoint (scout-plan:121), subject to the shared validation-loader caveat.
- Proof objects: needs exact artifacts: `probtest_tolerances.json`, `ensemble_member_manifest.json`, `heldout_candidate.json`, `cost_model.json`, and `tolerance_freeze_report.md`. The "no tolerance after seeing candidate failure" rule is correct (scout-plan:125).
- Missing sprint: no, but add stratification by land/sea/elevation and a clear "prototype only" limitation, especially for precipitation.

### M6-S8 - Operational Gen2 Comparison and Closeout Pack

- Scope: required, but the pass gate is too loose. The comparison variables and optional observation artifact are appropriate (scout-plan:129-140). However, `GPU-vs-Gen2 RMSE <= max(CPU-vs-observation RMSE, S7 tolerance)` can pass a forecast that violates the user's operational philosophy whenever S7 tolerance is larger than CPU-vs-observation error (scout-plan:139).
- Ownership: `operational.py`, `m6_compare_gen2.py`, and `artifacts/m6/operational/` are reasonable (scout-plan:134), but the "M6 closeout draft" needs a concrete path.
- Proof objects: `gen2_comparison.json` and `obs_comparison.json` are named (scout-plan:137-138). Add schemas and `artifacts/m6/closeout/proof_index.json`, plus `.agent/decisions/MILESTONE-M6-CLOSEOUT.md`.
- Missing sprint: add an observation-source/METplus-equivalent scout before M7 dispatch if S8 cannot produce CPU-vs-observation RMSE for U10/V10/T2. `observations_unavailable.json` is acceptable for an M6 diagnostic close, not for "ready for M7 operational validation" (scout-plan:138-140).

## Section 2 - Sequencing Critique

The proposed macro-sequence is directionally right: S0 prologue, S1 interface freeze, S2 driver, then validation/performance tooling in parallel, then S8 closeout (scout-plan:17-20, scout-plan:199-210). It is not actually as parallel as the plan implies.

Hidden serialization:

- S3 is on the critical path for meaningful S4, S5, and S8. The plan says S3 can run in parallel after S2 smoke (scout-plan:73-74), but S4's final pass waits for S3 (scout-plan:87), S5's verdict waits for S3/S4 (scout-plan:100), and S8 depends on S0-S7 (scout-plan:140).
- S6 tooling can start after S2, but final pass waits for S4 (scout-plan:113-114). S7 final variable/tolerance selection depends on S6 (scout-plan:126). That makes the validation proof path S3 -> S4 -> S6 -> S7 -> S8, even if scaffolding is parallel.
- Common Gen2 I/O, regridding, domain masks, boundary forcing, and variable naming are not assigned. S4, S6, S7, and S8 each own separate validation files (scout-plan:81, scout-plan:108, scout-plan:121, scout-plan:134), but they all need the same loader semantics.
- The d02 nested-boundary problem is not sequenced. The plan pins `d02` from a nested run (scout-plan:11-13) and dispatches a d02 driver (scout-plan:51), but a standalone d02 24h run requires boundary replay before S2 can become a load-bearing forecast sprint.

Optimal amended sequence:

1. M5-prologue gate: close or explicitly classify M5-S3.y/radiation-conditioning.
2. S1 interface freeze, including state, precision, boundary-forcing handles, output manifest, and shared validation schema.
3. S2a Gen2 backfill accessor, read-only manifest, d02 boundary replay/regridding, and CPU-denominator extractor.
4. S2 driver smoke on cropped/closed case, then full d02 smoke with replayed boundaries.
5. S3 surface-layer/minimal land state integration.
6. S4/S5/S6/S7 tooling can overlap, but their final proof order is S4 before S5 verdict/S6 final, S6 before S7 final.
7. S8 closeout.

M5-S3.y should be treated as a M5-prologue Phase-2 item, not an ordinary M6 sprint. It does not need to block S1 interface freeze. It should block S2 full 24h coupled validation and S8 T2 operational pass unless the manager makes an explicit radiation-conditioned downgrade. The scout plan currently excludes full RRTMG from M6 (scout-plan:72, scout-plan:151-157) without adding the proof that prescribed radiation tendencies are available.

## Section 3 - Surface-Layer + Noah-MP Scope-Creep Risk

The recommendation to pull some surface/land work into M6 is defensible because the M5 neutral surface stub cannot support a binding `U10/V10/T2/qv2` gate (scout-plan:64, scout-plan:70, scout-plan:155-161). The scout's "minimum Noah-MP" is not yet bounded enough to be realistic.

The smallest operationally meaningful subset should be:

- Monin-Obukhov/surface-layer diagnostics for `U10`, `V10`, `T2`, `Q2`, `UST`, sensible heat flux, moisture flux, and momentum stress.
- Read-only land/static/SST provenance from `wrfinput`/wrfout: `XLAT/XLONG`, `HGT`, `LANDMASK`, `LU_INDEX`, soil category, vegetation category, `TSK/SSTSK`, and needed Noah-MP diagnostic land fields.
- Prescribed or replayed land-state evolution for the first 24h, not dynamic Noah-MP soil/snow/canopy hydrology.
- Flux insertion into the lowest-level/PBL tendencies through the MYNN interface, with unit/sign tests and a surface energy/moisture budget artifact.
- No groundwater, canopy water, snowpack evolution, irrigation, urban canopy, optional Noah-MP physics matrix, or RRTMG table/solar machinery in M6.

If S3 implements that subset, 30-48h is aggressive but plausible. If S3 tries to implement real `sf_surface_physics = 4` Noah-MP semantics (scout-plan:69), the estimate is not credible.

Prescribed Gen2 radiation tendencies are not a safe assumption. The scout says to use prescribed tendencies if available (scout-plan:72) and calls that acceptable for the first 24h comparison (scout-plan:157). A local `ncdump -h` probe on the pinned `wrfout_d02_2026-05-19_18:00:00` shows many radiation flux fields (`SWDOWN`, `GLW`, `SWUPT`, `LWDNT`, etc.) and surface fields, but no obvious `RTH*`/radiation heating tendency field. Fluxes alone do not provide 3D theta tendencies at every timestep. The plan must add one of these gates:

- M5-S3.y RRTMG parity closes before S2/S8.
- A radiation-conditioning sprint proves extractable/replayable 3D tendencies from existing Gen2 artifacts.
- M6 labels T2 as radiation-conditioned/provisional and does not use T2 to authorize M7 dispatch.

## Section 4 - ADR-007 4x Verdict (M6-S5)

The S5 metrics list is close to the required profiler target: wall time, transfer count/bytes, launch count, occupancy, registers, local memory, bandwidth, compile time, and warmup time are named (scout-plan:97). Amend the JSON schema so it includes at least the `PERFORMANCE_TARGETS.md` fields: `benchmark`, `backend`, `hardware`, `case`, `wall_time_s`, `kernel_launches`, `host_device_transfer_bytes`, `occupancy_pct`, `registers_per_thread`, `local_memory_bytes`, and `artifact_paths`. Extra fields should include `host_device_transfer_count`, `device_to_host_transfer_bytes`, `compile_time_s`, `warmup_time_s`, `warmup_excluded`, `cpu_denominator_artifact`, `correctness_gate_artifact`, and `verdict`.

The CPU denominator is pinned by run ID but not by scope. The scout says PASS requires >=4x against the Gen2 CPU denominator for the same pinned run (scout-plan:99). That pinned run is a five-domain nested WRF job, while the GPU plan targets `d02` only (scout-plan:11-13). S5 must not compare a d02-only GPU run against total five-domain Gen2 wall time. It needs `artifacts/m6/performance/gen2_cpu_denominator.json` specifying:

- source run ID, domain(s), task count, hardware/workstation, and whether I/O is included;
- whether timing is total nested-job wall time, domain-2 `rsl.error` timing, or both;
- number of steps and lead window;
- mean/p50/p95 per-step time and total comparable wall time;
- a rationale for any extrapolation from >=6h to 24h (scout-plan:99).

Fallback if S5 says FAIL must be concrete, not just "alternatives discussion" (scout-plan:99, scout-plan:147):

- `FAIL_TRANSFER`: block M7; fix residency before tuning.
- `FAIL_DENOMINATOR`: re-run denominator extraction or narrow the comparison; no architecture decision.
- `FAIL_LAUNCH_BOUND`: open a fusion/Pallas/Triton sprint for the dominant physics package.
- `FAIL_REGISTER_OR_SPILL`: split kernels or move table-heavy package to custom kernel path.
- `FAIL_MEMORY_BANDWIDTH`: revisit state layout/access pattern and compression/downcast only under ADR-007 gates.
- `FAIL_VALIDATION`: correctness sprint before performance work.
- `FAIL_HARDWARE_FP64/BLACKWELL`: manager-level decision on scope, GPU target, or backend fallback.

M6 may close as "correctness achieved, performance blocked" if validation passes but S5 fails. It must not dispatch M7 as operational v0 in that state.

## Section 5 - Tier-2/3/4 Validation Soundness

S4 is testing the right invariant families, but the plan must require an oracle that is independent of the coupled solver. The anti-tautology warning is explicit (scout-plan:84), but the acceptance criteria still allow a worker to compute source/sink terms from the same GPU update and declare closure. Add one of:

- a WRF-derived budget extractor on a small replay fixture;
- an analytic closed/cropped case with prescribed forcing and known conservation;
- a second independent NumPy/JAX-free budget implementation owned by a reviewer/tester;
- a cross-AI budget residual report that recomputes dry mass and water from pre/post arrays and boundary fluxes only.

S6's TSC target is the weakest validation point. A base `dt=18s` and refinement candidate are named (scout-plan:111), but the proposed CPU envelope from `wrf_l2` vs `wrf_l3` is not a timestep-convergence envelope. Those campaigns differ in forecast length, nesting/configuration, and likely output cadence. Use them as operational sanity only. The reduced TSC case should be a cropped/closed or boundary-replayed d02 case with identical IC/BC, identical physics scope, and dt refinement (`18s`, `9s`, possibly `6s`) on the GPU plus any available CPU/WRF small oracle. The artifact should separate "TSC envelope" from "backfill consistency envelope."

S7 is a reasonable first method choice if it is kept humble. Probtest-style per-variable tolerance derivation is better than PCA for the first operational surface variables (scout-plan:123). Ten deterministic daily Gen2 members are an operational sample, not a perturbed ensemble (scout-plan:124). With only 10 members, precipitation tolerances will be unstable and qv2 tails can be sensitive to weather-regime sampling. The S7 artifact should include sample-size uncertainty, land/sea/elevation stratification, robust statistics, and a statement that M6 Tier-4 is a prototype, not production ensemble consistency.

## Section 6 - Operational Gate (M6-S8)

The operational philosophy is "GPU forecast differs from CPU less than CPU differs from observations" for surface variables. The scout pass gate uses the larger of CPU-vs-observation RMSE or S7 tolerance (scout-plan:139). That is not defensible as the binding gate because a loose S7 tolerance can mask a GPU-vs-CPU error larger than the real observation noise floor.

Recommended gate:

- For `U10`, `V10`, and `T2` when observations are available: `GPU_vs_Gen2_RMSE <= CPU_vs_obs_RMSE` per lead and variable. S7 tolerance is reported as a statistical-consistency sanity check, not as a loosening factor.
- If observations are unavailable: M6 can close only as `PROVISIONAL_VALIDATION`, not `READY_FOR_M7_OPERATIONAL`, unless the manager explicitly accepts an observation-source blocker.
- `qv2/Q2`: allow `PARTIAL_PASS_QV2` in M6 if U10/V10/T2 pass and qv2 bias/RMSE is fully reported, but require an M7-S0 humidity/surface follow-up before public operational claims.
- Precipitation: diagnostic in M6 unless the event sample and observation truth are adequate.
- Any variable may be `FAIL_WITH_DIAGNOSIS`; failures must preserve per-variable/lead status, not only aggregate RMSE.

Closeout artifact:

- `.agent/decisions/MILESTONE-M6-CLOSEOUT.md`
- `artifacts/m6/closeout/proof_index.json`
- `artifacts/m6/operational/gen2_comparison.json`
- `artifacts/m6/operational/obs_comparison.json` or `observations_unavailable.json`
- `artifacts/m6/performance/full_domain_batching_verdict.json`

M7 dispatch decision belongs to the manager after the closeout review. Dispatch should require S5 `PASS`, S8 `GREEN` for U10/V10/T2, no unresolved transfer violations, and a reviewed M7 sprint contract. A partial M6 close may authorize remediation sprints, not M7 operational v0.

## Section 7 - Risk Register Additions

Add these risks or patch them into the M6 plan:

- **Nested d02 boundary forcing gap**: the plan pins nested `d02` (scout-plan:11-13) and runs a d02 forecast (scout-plan:51), but no d02 `wrfbdy` exists in the Gen2 run. Mitigation: S2a boundary replay/extraction proof object.
- **Radiation tendency availability**: the plan relies on prescribed Gen2 radiation tendencies if available (scout-plan:72, scout-plan:157), but existing wrfout appears to expose fluxes rather than 3D heating tendencies. Mitigation: feasibility artifact or M5-S3.y block.
- **Gen2 data integrity/read-only contract**: the plan says Gen2 is read-only (scout-plan:15, scout-plan:71, scout-plan:171), but it needs a read-only manifest with path, size, mtime/checksum, variables, domain ID, and no-write audit.
- **CPU denominator unfairness**: S5 compares to the "same pinned run" (scout-plan:99), but d02-only GPU vs five-domain CPU would inflate speedup. Mitigation: domain-scoped denominator artifact.
- **Memory and compile-pressure scale-up**: S1's 100-step dummy carry (scout-plan:45) is much smaller than full d02 (scout-plan:13). Hydrometeors, TKE, surface, RRTMG tables, and JAX compile artifacts can fail only at full shape. Mitigation: full-d02 allocation/compile preflight before 24h.
- **JAX/XLA Blackwell/profiler fragility**: S5 requires counters (scout-plan:97), but prior profiler evidence had permission limitations. Mitigation: profiler-failure artifact is allowed only if wall time, transfers, and raw logs still support the verdict; otherwise S5 is blocked.
- **Shared validation infrastructure collision**: S4/S6/S7/S8 own separate files (scout-plan:81, scout-plan:108, scout-plan:121, scout-plan:134), but all need loaders, masks, regridding, units, and lead-time selection. Mitigation: one frozen validation I/O owner.
- **Cross-AI workflow pressure**: the plan budgets Gemini quota (scout-plan:173-182) but not Codex/Claude review bandwidth across 6-9 days (scout-plan:205-208). Mitigation: manager queue, max concurrent workers, and proof-index discipline.
- **Observation-source blocker**: S8 allows `observations_unavailable.json` (scout-plan:138). That is honest, but it prevents a binding operational pass. Mitigation: observation-source scout before M7.
- **TSC false confidence**: using `wrf_l2` vs `wrf_l3` as a "CPU envelope" (scout-plan:111) can validate configuration noise rather than timestep convergence. Mitigation: controlled dt-refinement case.
- **Surface/noah scope creep**: the plan's Noah-MP minimum is not enumerated (scout-plan:69). Mitigation: scope memo with explicit included/excluded WRF features before code.
- **Partial-pass ambiguity**: qv2/precip may be diagnostic (scout-plan:139), but no state machine says what M7 can do with partial failure. Mitigation: define `GREEN`, `PARTIAL`, `BLOCKED`, and `FAIL` closeout statuses.

## Section 8 - Binding Recommendation

**RATIFY-WITH-AMENDMENTS**.

Required edits before M6-S2/S8 dispatch:

1. Update S0/prologue to current state: Thompson/MYNN closed, M5-S3.y or radiation-conditioning remains a M5-prologue Phase-2 gate. Do not let scout-plan:21-33 claim all M5 debt is covered.
2. Add S2a or equivalent: Gen2 backfill accessor, read-only manifest, d02 boundary replay/regridding, output schema, and CPU denominator extractor. This fixes the d02 standalone gap created by scout-plan:11-13 and scout-plan:51.
3. Amend S1 to freeze boundary/forcing/output metadata interfaces, not just physics state leaves (scout-plan:42-45).
4. Rewrite S3 scope to the smallest surface-layer/prescribed-land subset, with explicit included/excluded Noah-MP features and a radiation-conditioning feasibility artifact (scout-plan:64-72, scout-plan:151-161).
5. Add exact proof-object paths and JSON schemas for every S0-S8 sprint. S5/S8 are closest; S0/S2/S3/S6/S7 are not concrete enough.
6. Tighten file ownership: S3 cannot edit coupling adapters after S1/S2 ownership freeze; one sprint owns shared validation I/O and regridding.
7. Replace S6's `wrf_l2` vs `wrf_l3` TSC envelope with a controlled dt-refinement reduced case. Keep l2/l3 as secondary operational sanity only (scout-plan:111).
8. Strengthen S4 with an external/cross-implementation budget oracle so Tier-2 is not solver-self-consistency (scout-plan:83-86).
9. Extend S5 with a fair Gen2 CPU denominator artifact and a concrete FAIL fallback ladder (scout-plan:97-99, scout-plan:147).
10. Replace S8's `max(CPU_vs_obs, S7 tolerance)` pass criterion with CPU-vs-observation as the binding gate for observed U10/V10/T2, S7 as a separate statistical sanity check, and explicit partial-pass statuses (scout-plan:137-140).

With these amendments, the M6 plan is good enough to proceed through S1 and preparatory tooling. It is not yet safe to dispatch a full 24h S2/S8 operational validation path as written.
