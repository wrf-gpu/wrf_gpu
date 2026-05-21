# M7 Milestone Plan Critical Review - Codex

Reviewer: Codex gpt-5.5 xhigh
Plan under review: `.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md`

Line citations refer to the scout M7 plan unless otherwise noted.

Decision: **RATIFY-WITH-AMENDMENTS**. The scout plan has the right operational shape: 3 km first, 1 km conditional, AIFS/Gen2 reuse for v0, explicit proof objects, and no fake `00Z in / 06Z publish` SLA. The amendments below are required before M7-S0 dispatch because the current plan is stale against M6 state, overstates parallelism, underweights live-observation and JAX compile risk, and leaves several shared operational interfaces without a single owner.

## §1 - Scope/completeness audit per S0..S8

### M7-S0 - Operational Readiness Prologue

- Scope: right gate, but stale. The plan says S0 must handle ADR-011 being absent and block S1 if still absent (lines 64-68). ADR-011 is now present in this worktree; S0 should gate on ADR-011 review/closeout status and proof-schema compatibility, not file presence.
- Acceptance: mostly concrete. `m6_inheritance_gate.json`, `gen2_baseline_inventory.json`, `operational_contract.json`, and `aifs_ingest_contract.json` have useful fields (lines 53-62). Add explicit fields for M5-S3.zz/S3.zzz radiation status, M6-S5 performance status, M6-S8 operational RMSE status, and whether M6 closed `GREEN`, `PARTIAL`, `BLOCKED`, or `FAIL`.
- Ownership: unsafe if S0 runs while M6 is still being assembled. S0 may modify `src/gpuwrf/io/proof_schemas.py` append-only (line 47), while active/queued M6 work still writes proof schemas and surface artifacts. If S0 overlaps final M6 closeout as allowed (line 67), it should be read-only except for `artifacts/m7/prologue/**` until the manager freezes M6 shared I/O.
- Wall-time: 12-18 h is realistic only for audit/prologue artifacts (line 43). It is optimistic if S0 also creates operational config modules and schema code.
- Hidden dependencies: S0 must not treat "M6-S2/S3 closed" as sufficient. M7 dispatch also depends on M6 performance/operational gates and M5 radiation closeout, because M7 claims U10/V10/T2/Q2/precip products and verification (lines 31, 178, 547).

### M7-S1 - 3 km Daily Pipeline

- Scope: load-bearing and necessary. The artifacts for AIFS-to-state, run summary, transfer audit, Gen2 comparison, and cycle directory are concrete (lines 86-95), and the zero-transfer gate is correctly hard (lines 90-92).
- Ownership: mostly disjoint from other M7 lanes, but not from active M6 until M6 closes. S1 owns `src/gpuwrf/coupling/operational_driver.py` and may use `src/gpuwrf/coupling/driver.py`/`boundary_apply.py` (lines 80-82); M6-S2 still owns the underlying state, boundary leaves, driver behavior, and GridSpec threading. S1 must wait for those interfaces to be closed and reviewed.
- Wall-time: 36-60 h (line 76) is optimistic for first live AIFS ingest plus WPS/met_em reuse, real d01->d02 one-way nesting, 24 h run, atomic publish, transfer audit, and same-cycle Gen2 comparison. Either split ingest/preflight from the 24 h daily runner or budget nearer 48-84 h after M6 is truly closed.
- Hidden dependencies: S1 depends on more than S0 and M6-S2 (lines 97-101). It also depends on M6-S3 surface/T2 credibility, M5 radiation status, the fair Gen2 CPU denominator, AIFS/WPS command provenance, and same-cycle Gen2 CPU output availability.

### M7-S2 - 1 km Memory, Compile, and Residency Audit

- Scope: correct decision sprint. The memory audit and nested compile smoke artifacts are concrete (lines 118-123).
- Ownership: `src/gpuwrf/profiling/memory.py` and `src/gpuwrf/ops/nesting_plan.py` are disjoint enough (lines 111-114), but `nesting_plan.py` becomes a contract used by S3. Freeze its schema before S3 starts.
- Wall-time: 12-20 h (line 109) is plausible for allocation and compile probes, not for debugging a failing nested scan. The sprint must be allowed to return a failure artifact without attempting tiling, as the scout says (lines 127-129).
- Hidden dependency: the proposed PASS gate overweights raw HBM (`peak HBM <= 26 GB`) relative to the real risk (line 122). Persistent 1 km state is not likely to be the limiter; XLA compile size, temporary buffers, retracing, allocator fragmentation, and nested static-scan HLO blow-up are the likely blockers. Add `hlo_op_count`, `stablehlo_bytes`, `compile_cache_bytes`, `compile_retries`, `retrace_count`, `xla_temp_peak_bytes`, and allocator fragmentation evidence to the audit.

### M7-S3 - Conditional 1 km Pipeline

- Scope: reasonable if S2 passes, but the claim boundary must be sharper. The sprint may run d03 Tenerife only and include d04/d05 only if headroom permits (lines 131-134, 414). Tenerife-only cannot be reported as "Canary 1 km operational"; it is `PARTIAL_1KM_TENERIFE`.
- Acceptance: `nested_run_summary.json`, `nest_boundary_consistency.json`, and `gen2_l3_comparison.json` are concrete (lines 146-153). Add an explicit `one_km_claim_scope` field: `NONE`, `D03_TENERIFE_ONLY`, or `D03_D04_D05_CANARY_L3`.
- Ownership: new nested driver/boundary files are appropriate (line 141), but S3 also writes under the shared cycle tree and may depend on S4's common output format (line 158). The output interface must be frozen by S0/S1 before S3 writes cycle products.
- Wall-time: 36-72 h (line 137) is plausible for d03 after S2 pass; all sibling nests plus comparison and products likely exceed it.
- Hidden dependency: S3 final comparison depends on S4 conversion if S4 owns common output format (line 158). Treat S4 format freeze as a prerequisite for S3 final acceptance, even if S3 compute work overlaps.

### M7-S4 - Post-processing and Operational Products

- Scope: useful and mostly bounded. Required product manifests and diagnostics inventory are concrete (lines 175-183).
- Ownership: output writers are disjoint from forecast drivers (lines 168-171), but `station_extract.py` overlaps conceptually with S5 station matching and station QC (lines 170, 207-215). Assign station registry, vertical/elevation correction, and observation/model collocation to S5; S4 should only consume a frozen station list and write point extracts.
- Wall-time: 18-30 h (line 166) is plausible for NetCDF-like/Zarr products if the output state schema is stable. It is low if variable availability is still conditional on M6 surface/radiation debt (line 178).
- Hidden dependency: S4 feeds S5 and S7 (line 189), so product manifest schema has to freeze before S5 score computations are accepted.

### M7-S5 - Operational Verification Against Stations and Gen2

- Scope: essential and aligned with the validation philosophy. The binding v0 gate `gpu_vs_gen2_rmse <= gen2_vs_obs_rmse` for U10/V10/T2 is the right operational metric (lines 213-217, 449-453).
- Acceptance: concrete fields are present (lines 207-217), but the observation source actuals are weaker than the plan implies. The existing station cube ends on 2026-05-07 (line 439), while M7 live cycles will be later. Existing cube data can validate replay/backfill cycles, not a live operational cycle unless a live obs ingest path is proven.
- Ownership: validation files are disjoint (lines 199-203), but S5 depends on S4's product manifest and S1/S3 outputs (lines 219-223). S5 can build loaders after S0, but final scoring is not parallel.
- Wall-time: 24-36 h (line 197) is optimistic if live AEMET/GRAFCAN access, API keys, rate limits, station QC, anemometer height, wind vector conversion, elevation representativeness, and freshness rules are in scope.
- Hidden dependency: `custom METplus-equivalent first` is defensible (lines 425-431), but it must not become a mini-METplus rewrite. Limit it to RMSE/bias/count, transparent collocation, and reproducible schemas.

### M7-S6 - Restart and Crash Recovery

- Scope: correct. A restart matrix plus continuity/idempotency/cold-cache artifacts are the right proof objects (lines 240-248, 460-484).
- Ownership: restart/recovery files are disjoint (lines 233-236), but idempotency touches the same atomic publish and `latest` behavior S1 owns (lines 95, 245-246). S0/S1 must freeze the cycle lifecycle API before S6 implements recovery.
- Wall-time: 18-30 h (line 231) is plausible for project-native restart and crash injection. Full WRF-compatible `wrfrst` writing may exceed this and should be allowed to close as an explicit `DEVIATION`.
- Hidden dependency: JAX compile cache behavior is operational, not just restart metadata (lines 247-248, 479-483). S6 needs a process-restart test: kill the process, reload restart, measure cold/warm cache, and prove no cache corruption or stale shape key silently invalidates recovery.

### M7-S7 - Monitoring, Alerting, and Ops Dashboard Hooks

- Scope: needed for single-machine v0. Status schema, alert policy, examples, and `latest.json` are concrete (lines 270-277).
- Ownership: status API and monitoring are disjoint from model code (lines 263-266), but they consume S1/S4/S5/S6 state transitions. Freeze status names in S0 before S1/S6/S7 each invent cycle states.
- Wall-time: 12-24 h (line 261) is fine for local status JSON. It is not enough for real alert delivery unless the sprint explicitly defines "alerting" as file/status emission only.
- Hidden dependency: the plan says S7 can start after S0 but final examples depend on S1/S4/S5/S6 (lines 279-282). Mark S7 as scaffold-parallel, final-serial.

### M7-S8 - Milestone Soak and Closeout

- Scope: correct closeout shape. The proof index, soak, exit status, and decision document are concrete (lines 299-306).
- Ownership: closeout files are disjoint and should not touch implementation code except index plumbing (lines 292-295).
- Wall-time: 12-24 h plus soak wait (line 290) understates the calendar if "three successful cycles" means live daily cycles (lines 301-302, 538). If pinned replay cycles count, the plan must say so; if not, soak requires at least three calendar days.
- Hidden dependency: S8 is serial after required sprints (lines 308-311), and milestone closeout requires independent review. It must also encode variable-specific claim status so blocked Q2/RH2/precip does not contaminate a GREEN U10/V10/T2 decision (lines 547, 554-561).

## §2 - Sequencing critique

The macro-sequence is directionally right, but less parallel than the scout summary implies. S0 blocks implementation (lines 64-68). S1 is the real critical-path opener (lines 70-101). S4/S5/S6/S7 can scaffold in parallel, but their final proof path is serialized by output and lifecycle artifacts.

Hidden serialization:

- S4 can start on a 6 h smoke, but final acceptance waits for S1 24 h and S3 for 1 km products (lines 185-189).
- S5 depends on the S4 product manifest and S1/S3 outputs (lines 219-223). Observation loaders can be built early; operational scores cannot.
- S6 can start after a 12 h S1 run (lines 250-253), but idempotency and recovery must use the same atomic cycle lifecycle S1 publishes (lines 95, 245-246).
- S7 can start after S0 (line 281), but live examples depend on S1/S4/S5/S6 (line 282).
- S3 is not independent of S4 if S4 owns the common output format (line 158).

Amended sequence:

1. Finish or explicitly classify M5/M6 prerequisites: M5 radiation closeout, M6-S2 real 24 h d02 with zero transfers, M6-S3 surface layer, M6 performance verdict, and M6 operational RMSE gate. The scout hard blockers cover only part of this (lines 524-530).
2. Run S0 as an audit/schema-freeze sprint. If M6 is still closing, S0 may write only prologue artifacts until M6 shared I/O/proof schemas are frozen.
3. Dispatch S1 after S0 and M6 closeout. Freeze run manifest, cycle lifecycle, product/output contract, and status states before any parallel M7 implementation.
4. S4/S5/S7 may scaffold after S0/S1 interface freeze; S6 may scaffold after S1 12 h; final acceptance waits on real S1 24 h output.
5. S2 starts after S1 has real state factories; S3 starts only after S2 PASS and S1 24 h.
6. S8 is serial after S1, S4, S5, S6, S7, and S3 only if 1 km passed (lines 308-311).

The scout's 5-8 working day 3 km critical path (lines 514-517) is plausible only after M6 closes cleanly and if replay cycles count for soak. With live observation acquisition and three live cycles, calendar time is longer.

## §3 - AIFS ingest realism

The AIFS plan is realistic for v0 if it is treated as "reuse Gen2/WPS as an operational oracle," not as native ingest. The poll window and Gen2 live paths are concrete (lines 27-28, 317-323), and reusing WPS/met_em for v0 is the right pragmatic call (lines 331-337). Native AIFS regridding should remain post-v0.

Required amendments:

- S0/S1 must record exact WPS/real-equivalent command provenance, executable paths, environment, geog root checksum, met_em files, and terrain/static checksums. The current AIFS contract lists source paths and interpolation policy (lines 61-62), but not the operational binaries and command logs that make a WPS-dependent v0 reproducible.
- Add source-availability and license fields. The plan names Dynamical/ECMWF AIFS Single, S3 icechunk, ECMWF open-data GRIB, and Azure default (lines 319-322), but does not make upstream license/availability/version drift a blocking risk.
- Treat partial GRIB and key-count failures as first-class failure artifacts. The plan already blocks partial missing GRIB before GPU allocation (lines 346-350); add source version, expected keys, actual keys, missing variables/levels, and retry history.
- Soil translation from AIFS layers to WRF-facing layers (line 328) and terrain/geog provenance (lines 334-337) need explicit tolerances or deviation status, because surface errors dominate T2/Q2 validation.
- Same-cycle Gen2 comparator availability is a hidden dependency. S1 compares against Gen2 for the same AIFS cycle (lines 93-95), but the live Gen2 CPU run may not be finished or may fail. S1/S5 need `GEN2_SAME_CYCLE_MISSING` status rather than blocking all GPU cycle production.

## §4 - Operational schedule

The narrow schedule is acceptable for v0 but must be labeled narrow. One daily 18Z cycle and 24 h minimum forecast are clear (lines 25-31, 354-360). Refusing a `00Z in / 06Z publish` SLA is correct (line 33).

Issues:

- The provisional 08:00 UTC publish target (line 360) is only credible if AIFS arrives by 05:25, WPS prep is included or separately bounded, cold compile stays under 90 min, post-processing and verification hit 15 min each, and Gen2 comparison is not on the critical path (lines 364-367). Do not call it an SLA until S8 soak proves it.
- 24 h v0 is operationally useful, but it does not satisfy the user's broader 24 h/72 h validation philosophy. The plan should explicitly say M7 v0 makes no 72 h claim and that 48 h is only diagnostic (lines 29-31, 447).
- Workstation reliability is under-specified. The plan has a single-machine lock (line 359), status states (lines 270-277), and recovery artifacts (lines 245-248), but needs preflight checks for disk, GPU health, driver/JAX version, compile-cache writeability, stale lock cleanup, system reboot recovery, and power/network interruptions.

## §5 - Nesting + 1 km gate

One-way nesting is the right v0 choice because Gen2 uses `feedback = 0` (lines 390-407) and the plan explicitly excludes child-to-parent feedback (lines 409-421). Two-way nesting should stay out of M7 unless an ADR approves it.

The 1 km gate should be reframed. The plan correctly makes 1 km conditional on memory, compile, and residency (lines 20-23, 118-123), but raw persistent state is not the main risk. For the listed L3 shapes (lines 392-407), 30 FP64 fields across d03/d04/d05 are on the order of hundreds of MB, not the dominant 32 GB problem. The likely blockers are:

- XLA compile blow-up from static nested scans (lines 416-419).
- temporary buffer peaks and aliasing failures;
- retracing from shape or cycle metadata drift;
- compile-cache invalidation across JAX/driver/code changes;
- output buffering and parent-child interpolation buffers;
- hidden D2H transfers from dynamic predicates, repeating the M6 radiation-cadence failure mode the scout calls out (line 418).

Amend S2 PASS to include HLO/StableHLO size, op count, temp peak, compile retries, retrace count, cache size, and allocator fragmentation, not only `peak_hbm_bytes <= 26 GB` and `compile <= 45 min` (line 122).

Also amend S3/S8 claim language. The plan says d03 Tenerife first, d04/d05 if headroom permits (lines 133-134, 414). If only d03 ships, closeout must say `1km_status = PARTIAL_TENERIFE_ONLY`; full "Canary 1 km" requires d03/d04/d05 or an explicit manager-approved deviation.

## §6 - Verification framework

The verification philosophy is correct: operational RMSE is binding, per-cell parity is not. The scout's gate for U10/V10/T2, `gpu_vs_gen2_rmse <= gen2_vs_obs_rmse`, matches that philosophy (lines 213-217, 449-453).

Amendments:

- Existing observation assets are not automatically live. The station cube ends 2026-05-07 (line 439). M7 live cycles after 2026-05-21 need live AEMET/GRAFCAN/other ingestion or close only as replay/backfill-validated.
- S5 must define freshness windows, API key handling, rate-limit behavior, station exclusions, and license/access status. The plan lists AEMET and GRAFCAN sources and key requirements (lines 435-438), but S5 acceptance should make unavailable credentials a `BLOCKED_OBS_SOURCE` status.
- Wind observations usually arrive as speed/direction, not U/V. S5 should own vector conversion, calm-wind direction handling, anemometer height metadata, and station elevation/representativeness masks before computing U10/V10 scores (lines 207-215, 442-445).
- T2 and RH/Q2 require elevation, exposure, and land/sea masks. Humidity is already partial unless coverage is proven (lines 215, 445), which is good.
- The custom METplus-equivalent must stay intentionally small: RMSE, bias, count, collocation, and reproducible score schemas. The plan's METplus deferral is reasonable (lines 425-431), but only if score equivalence is tested on a small fixed sample before S5 closes (lines 210-212).

## §7 - Restart + crash recovery

The restart plan is directionally strong. It requires a `wrfrst` compatibility matrix, project restart, continuation, restart-vs-continuous delta, idempotency, and compile-cache timing (lines 240-248, 460-484).

Required tightening:

- S6 should treat WRF-compatible `wrfrst` writing as preferred but non-blocking if a deviation is explicit (lines 466-470). Project-native restart plus continuity is the v0 operational requirement.
- Add a process-kill/restart test, not just a 6 h + restart + 6 h numerical comparison (lines 243-248). The test should prove restart after Python process death, stale lock cleanup, cache reload, and `latest` preservation.
- Compile cache is an operational dependency. The scout measures cold and warm cache (lines 247-248, 479-483), but S1/S6/S8 should also record cache key inputs: commit, jax/jaxlib, CUDA driver, shapes, domain list, and static options. Cache miss after a code or driver update must not make the daily cycle silently miss the publish target.
- Recovery must preserve failure proof objects. The plan says this (lines 472-477); S6 acceptance should include crash injection at AIFS ingest, during forecast, during postprocess, during verification, and during atomic publish.

## §8 - M7-specific risks

The scout risk register is strong (lines 485-499), especially on AIFS lateness, WPS dependency, 1 km memory/compile, hidden D2H transfers, observation gaps, and output creep. Add these missed or underweighted risks:

- **Stale M6 premise risk**: the plan still refers to ADR-011 as possibly absent (line 68). M7-S0 must be updated to current M6 artifacts and review status before dispatch.
- **M5/M6 radiation and surface credibility risk**: the plan notes surface/Noah/RRTMG debt (line 497), but M7 variable claims depend on the in-flight M5-S3.zz/S3.zzz and M6-S3 outcomes. T2/Q2/cloud/radiation-derived products should be provisional until those close.
- **Shared operational interface collision**: S1, S6, and S7 all touch lifecycle/status/publish semantics (lines 95, 245-248, 270-277). S0 must freeze the state machine and run manifest before parallel work.
- **Station data staleness risk**: the existing station cube is stale relative to M7 live cycles (line 439). Without live obs, M7 can close only as replay/backfill validated.
- **Same-cycle Gen2 comparator risk**: S1/S5 require Gen2 same-cycle comparison (lines 93-95, 213-215), but Gen2 can be late or fail independently.
- **Compile cache invalidation risk**: cold/warm compile is measured (lines 247-248, 364-365), but driver/JAX/code changes can invalidate cache and break the 08:00 target.
- **Retention/disk risk**: retention is proposed (lines 376-386), but no quota, disk-low threshold, cleanup proof, or emergency behavior is specified.

## §9 - Risk register additions

| Risk | Impact | Mitigation |
|---|---|---|
| Disk full under `/mnt/data/wrf_gpu2/operational/` | Cycle fails mid-run or corrupts products/proof objects | S0 retention policy with byte quotas; S7 `disk_low` threshold; S6 crash test for no-space during output; S8 proof of cleanup behavior. Related scout paths/retention: lines 376-386. |
| Single-machine SPOF | No forecast during reboot, power loss, GPU fault, driver crash, or user workload contention | S7 health/status preflight, stale-last-good policy, explicit `MACHINE_UNAVAILABLE`, and no stronger SLA than proven by soak. Related scout single-machine lock/status: lines 359, 270-277. |
| Long-running JAX/driver instability | Daily cycles degrade after repeated compiles/runs, memory leaks, cache corruption, or allocator fragmentation | S8 soak records per-cycle memory, compile, wall time, process restart behavior; S6 process-kill recovery; S2 allocator fragmentation metrics. Related compile/cache lines: 247-248, 364-365, 479-483. |
| AIFS upstream license/availability/version change | Operational ingest breaks or becomes legally/technically non-reproducible | S0/S1 source license/version manifest, fallback status only to stale-last-good, no silent source substitution. Related AIFS source lines: 319-322, 346-350. |
| Live station API credentials/rate limits | Verification blocks or silently uses stale observations | S5 credential/access manifest, freshness gate, `BLOCKED_OBS_SOURCE` status, and replay/backfill-only closeout if live obs unavailable. Related station lines: 435-439, 207-215. |
| Same-cycle Gen2 CPU run unavailable | GPU run exists but v0 comparison denominator is missing | Separate `GPU_CYCLE_OK_GEN2_MISSING` from forecast failure; compare when Gen2 arrives; do not publish validation score until matched. Related comparison lines: 93-95, 213-215. |
| WPS/Gen2 tooling drift | AIFS-to-met_em path changes outside this repo and breaks reproducibility | Pin command logs, executable paths, env vars, geog checksums, and met_em checksums in S1. Related WPS reuse lines: 331-337. |
| UTC/local-time cycle confusion | Wrong AIFS cycle, wrong station matching window, or stale publish label | Require UTC-only cycle IDs and timestamp fields in S0/S1/S5/S7 schemas. Related cycle lines: 27-31, 474. |
| Nested 1 km partial-claim ambiguity | Tenerife-only d03 is marketed as full Canary 1 km | Add `one_km_claim_scope` and require d03/d04/d05 for full Canary L3 claim. Related nesting lines: 133-134, 414. |
| Output atomics across filesystems/symlinks | `latest` points to partial products or stale status | S1/S6 atomic publish contract with temp dirs on same filesystem, fsync/rename semantics, and crash injection. Related publish lines: 95, 245-246, 380-381. |

## §10 - Binding recommendation

**RATIFY-WITH-AMENDMENTS**.

Required edits before M7-S0 dispatch:

1. Update S0 to current M6 reality: ADR-011 exists; gate on reviewed M6 closeout status, M5 radiation closeout, M6-S2/S3/S5/S8 proof objects, and variable-specific GREEN/PARTIAL/BLOCKED/FAIL status. Fix the stale ADR-011 language at lines 64-68.
2. Make S0 read-only if it overlaps M6 closeout. Do not let it append to `src/gpuwrf/io/proof_schemas.py` (line 47) until M6 shared I/O/proof schemas are frozen.
3. Freeze shared M7 interfaces in S0/S1: run manifest, cycle lifecycle, status state machine, output/product manifest, proof-schema registry, station-collocation API, and atomic publish contract. Assign exactly one owning sprint for each.
4. Amend S1 to include WPS/AIFS command provenance, geog/static checksums, source license/version metadata, partial-GRIB failure schema, and `GEN2_SAME_CYCLE_MISSING` status. Revisit the 36-60 h estimate at line 76 or split ingest/preflight from full 3 km run.
5. Amend sequencing: S4/S5/S6/S7 can scaffold in parallel, but final acceptance is serial through S1 24 h output, S4 product manifest, S5 scores, S6 recovery, S7 live examples, and S8 closeout (lines 185-189, 219-223, 279-282, 308-311).
6. Rewrite S2's 1 km PASS gate to emphasize XLA compile/temporary/retrace/cache risks over raw persistent HBM. Add HLO/StableHLO size, op count, compile retries, retrace count, temp peak, cache size, and allocator fragmentation to lines 118-123.
7. Add explicit 1 km claim scope. D03-only is `PARTIAL_TENERIFE_ONLY`; full Canary 1 km requires d03/d04/d05 or an explicit deviation (lines 133-134, 414).
8. Tighten S5 live-observation requirements: freshness, API credentials, licensing, rate limits, wind speed/direction conversion, station height/elevation masks, and `BLOCKED_OBS_SOURCE`. Existing station cube data ending 2026-05-07 (line 439) cannot validate a later live cycle by itself.
9. Keep the validation gate operational: U10/V10/T2 use `gpu_vs_gen2_rmse <= gen2_vs_obs_rmse`; Q2/RH2 and precip stay partial/diagnostic unless coverage and event sample are proven (lines 213-217, 442-453, 547).
10. Amend S6 restart tests to include process death, stale lock cleanup, compile-cache reload/corruption fallback, version/hash compatibility, and crash injection at ingest/forecast/postprocess/verify/publish. WRF-compatible `wrfrst` writing may close as an explicit deviation (lines 240-248, 466-470).
11. Add workstation operations gates: disk quota/cleanup, GPU health, single-machine unavailable state, UTC-only cycle IDs, cache writeability, and no 08:00 SLA until S8 soak proves it (lines 354-368, 376-386).
12. Clarify S8 soak: whether three pinned cycles may be replay cycles or require three live daily cycles. If live, the calendar minimum is three days, not 12-24 h plus an unspecified wait (lines 290, 301-302, 538).

With these amendments, the M7 plan is sound enough for manager integration and M7-S0 contract drafting. As written, it is not yet safe to dispatch M7 implementation because it can overclaim 1 km scope, operational validation, and schedule reliability before the necessary M6/M5 evidence and live-observation plumbing exist.
