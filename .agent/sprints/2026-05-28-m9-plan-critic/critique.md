# M9 Plan Critique

Role: ADVERSARIAL PLAN CRITIC  
Sprint: `2026-05-28-m9-plan-critic`  
Date: 2026-05-28  

## Evidence Caveats

1. The current worktree branch was behind the manager ref at dispatch time. The sprint contract and M9 proof objects were read from the local `manager-2026-05-23` / M9A worktree evidence rather than from this checkout's initial tree.
2. `.agent/sprints/2026-05-28-m9a-trace-harness/worker-report.md` is stale relative to the manager close: it still says `M9A_PARTIAL`, while the later manager branch contains `proofs/m9/operational_trace_hourly.json` and `proofs/m9/divergence_map.json`. Treat the worker report as process evidence for the blocked 1000-step run, not as the final M9 verdict.
3. The M9 trace is a wrfout-level hourly comparison, not the originally requested per-operator WRF Fortran trace. It is useful for triage, but not enough to assign operator blame without comparator and raw-value audits.

## PC1

M11-M14 is still the right broad Phase B decomposition, but it is not the right immediate next action. M9 shows divergence in nearly every operational field at hour 1, and several of the largest deltas are vulnerable to variable-convention or diagnostic-comparator mistakes. Phase B should be augmented by a mandatory M9.C sprint before M10/M11:

- audit `T` / theta reference-state convention in GPU and WRF wrfout files;
- record raw min/max/percentiles for GPU and WRF, not only deltas;
- test timestamp alignment and field transforms;
- rerun `operational_trace_hourly.json` after the comparator audit.

Do not replace M11-M14 yet. The M9 manager diagnosis maps plausible defects to M10 static fields, M11 theta/guards, M12 surface flux/MYNN, M13 radiation, and M14 boundary completeness. But using the current trace quantitatively before M9.C would risk spending weeks on apparent physics defects that are partly comparator artifacts.

## PC2

TSK max_abs = 0.0 confirms one narrow fact: hourly skin-temperature data replay can reproduce WRF's emitted TSK bitwise for the pinned 20260521 d02 wrfout sequence. It does not prove prognostic Noah-MP is unnecessary for the project in general.

For the binding v0.1.0 path, M16 should become conditional. If hourly replay inputs are available for the full >=30-case L2/L3 corpus and the replay-based model passes TOST on T2/U10/V10, then carrying an 8-14 week prognostic Noah-MP port on the critical path is not justified.

The strongest argument for keeping M16 is product independence. A replay land surface depends on external hourly land-state truth, may effectively borrow from CPU WRF for the surface boundary, and is not a fully self-contained forecast from IC/BC data. It may pass a paired evaluation while still failing the professional-forkable "GPU-native regional NWP" objective. Keep M16 only if the v0.1.0 release definition requires independent 24-72 h forecasts without hourly WRF-derived land replay, or if M19/M20 shows replay cannot meet TOST.

## PC3

The premise is partly overstated because the M9 "max" values are max absolute differences, not raw physical field maxima. PBLH = 986 m is not physically impossible for a boundary-layer height, and a 986 m PBLH difference is a serious error but not a physical-bound violation. SWDOWN 1122 W/m2 as a difference also does not by itself prove either model emitted an impossible value. HFX differences near 4105 W/m2 are much more suspicious and likely indicate a real surface-flux conversion/sign/coupling bug or a gross diagnostic mismatch.

The current trace already rules out some easy artifacts: shapes match for these fields and nonfinite counts are zero. That pushes the likely explanation toward real model/comparator semantics rather than NaN handling. Specific disambiguation tests:

1. Raw-value audit at every argmax cell: store GPU value, WRF value, delta, units, timestamp, i/j/k, land/sea mask, LU_INDEX, HGT, and percentiles for SWDOWN, HFX, LH, PBLH, T2, U10, V10, theta.
2. Radiation time-alignment audit: recompute SWDOWN deltas with WRF shifted -1/0/+1 hours and compare to local solar geometry. A midday GPU near zero while WRF is near 1000 W/m2 means cadence/coszen/radiation activation, not unit mismatch.
3. Surface-flux dimensional audit: compare WRF HFX against GPU `rho * cp * theta_flux` and against any already-converted W/m2 diagnostic. This catches cp/rho multiplication twice, sign inversion, and K m/s vs W/m2 confusion.
4. PBLH vertical-coordinate audit: for the argmax cells, dump the MYNN input profiles and diagnostic PBLH computation. Verify meters above ground, not model level index, pressure thickness, or AGL/MSL mismatch.
5. Static-field conditioned split: stratify HFX/PBLH/T2/U10/V10 deltas by LU_INDEX match/mismatch, land/sea, and elevation. If errors concentrate on LU_INDEX-mismatched land cells, M10 is a prerequisite for interpreting M12/M13.

## PC4

The M9 trace deliverable needs a comparator audit before it is used to drive M11-M14 quantitatively.

The trace script applies `theta = T + 300.0` to both GPU and WRF `T`. If both outputs store WRF perturbation potential temperature, that transform cancels and is harmless. If the GPU writes absolute theta into a variable named `T`, the script wrongly adds 300 K to the GPU side and leaves a synthetic near-300 K offset. The current proof object does not record raw `T` ranges or NetCDF descriptions, so it cannot distinguish these cases.

The one-line-conversion hypothesis is plausible but not proven. M9.C should compute theta RMSE under four transforms: raw/raw, +300 both, +300 WRF only, and +300 GPU only, then choose the transform supported by raw value ranges and variable metadata. Until then, the trace is trustworthy for "many fields diverge by hour 1" and for "TSK replay is bitwise," but not trustworthy as a ranked magnitude source for Phase B.

## PC5

The blocked 1000-step dycore parity run is not strong evidence of a model defect, because the 24 h operational pipeline did produce GPU outputs and the saved `savepoint_parity_1000.json` reports no visible JAX GPU backend in the codex environment. It is a proof-process risk, not yet a project-architecture risk.

A single manager-environment rerun is necessary, but "run it once and call it good" is too weak. The rerun must produce a committed proof object with command, environment, visible devices, commit, source wrfout, depth, first divergence step, and per-field final-step deltas. If it passes once, require it as an invariant rerun after M11/M12/M14 because those milestones can change long-run stability even when 100-step parity remains green.

## PC6

The current 8-12 calendar week estimate for M11-M14 is no longer credible as the base estimate. It is an optimistic best case after M9.C passes and after M10 removes the static-field confounder.

Re-estimate:

- M9.C comparator/theta/raw-range audit: 2-4 days.
- M10 static-field parity: 3-7 days if kept narrow; 1-2 weeks if roughness/soil/state serialization expands.
- M11 theta/mu/guard accounting: 2-4 weeks, depending on whether the theta issue is just wrfout convention or a real state-update bug.
- M12 surface flux + MYNN bottom boundary: 3-6 weeks because HFX evidence points to magnitude/coupling problems, not just wiring order.
- M13 radiation + land-surface diurnal physics: 2-5 weeks, gated by the SWDOWN time/coszen audit.
- M14 lateral BC + state completeness: 2-4 weeks.

Calendar estimate with careful parallelism: 12-18 weeks for M9.C through M14, or 10-16 weeks for M11-M14 after M9.C/M10 are closed. Raw un-parallelized effort is closer to 14-22 weeks. The plan should state that the old 8-12 weeks is a low-case, not the planning median.

## PC7

Do not skip M10 as a standalone sprint. LU_INDEX has a confirmed 14-category delta, and folding it into M14 would delay a cheap known fix while polluting every surface-flux and 10 m wind diagnosis. Static fields are not just "state completeness"; they are inputs to land/roughness/surface-layer behavior and therefore must be settled before M12/M13 evidence is interpreted.

Keep M10, but constrain it aggressively: LU_INDEX, HGT, LANDMASK, XLAND, roughness, soil category, state leaf/schema, and a static parity proof. If it cannot close in roughly one week, split any extra land-category semantics into the later land milestone rather than letting M10 sprawl.

## PC8

The single most likely-to-be-wrong assumption is that M9's wrfout-level deltas are already measuring same-variable, same-time, same-convention physical errors. The theta, SWDOWN, HFX, and PBLH evidence is too comparator-sensitive to use as a quantitative work queue without M9.C.

## PC9

The strongest objection in one sentence: the plan is still scheduling physics-port work before proving that the comparator is comparing the same physical variables at the same times under the same WRF output conventions.

## PC10

If only one milestone action is allowed for maximum probability of TOST equivalence within 6 calendar months, move M16 off the v0.1.0 critical path and make it a conditional v0.2.0 milestone.

Rationale: M16 is the largest single time block, and M9 TSK proves the replay mechanism can be exact for at least the skin-temperature state. A six-month path is unlikely if an 8-14 week prognostic Noah-MP port remains mandatory before M21. The condition should be explicit: keep replay for v0.1.0 unless M19/M20 shows replay-based T2/U10/V10 cannot pass TOST, replay inputs are unavailable for the >=30-case corpus, or principal release criteria require a forecast independent of hourly WRF-derived land replay.

M9.C should still be dispatched immediately, but as a sub-sprint prerequisite to M10/M11 rather than as the one milestone-level change.

## Handoff

- objective: adversarially critique the M8/M9-amended reset plan and answer PC1-PC10.
- files changed: `.agent/sprints/2026-05-28-m9-plan-critic/critique.md`.
- commands run: initial unpinned orientation before reading the CPU-pinning rule (`pwd`, `rg --files`, `git status`, missing-path checks); then `taskset -c 0-3` reads of the sprint contract from `manager-2026-05-23`, local `conducting-blind-review` skill, required plan/ADR/M8 artifacts, M9 proof objects, M9A worker report, savepoint scaffold, proof-index excerpts, M9 trace summaries, M9 comparator script excerpts, PC-section validation, ASCII validation, and `git status`. Attempted `taskset -c 0-3 git add .agent/sprints/2026-05-28-m9-plan-critic/critique.md && taskset -c 0-3 git commit -m "[m9 plan critic] critique Phase B after M9 divergence"`, which failed because the real git metadata path is read-only in this sandbox. Attempted the required tmux notification with nonzero exit marker; tmux returned `error connecting to /tmp/tmux-1000/default (Operation not permitted)`.
- proof objects produced: this critique file.
- unresolved risks: no GPU runtime was used; the current branch is behind the manager ref that contains the sprint contract and M9 proof objects; M9A worker report is stale relative to manager close; no independent raw NetCDF value audit was executed; git add/commit is blocked by read-only worktree metadata outside the writable sandbox; auto-notify is blocked by tmux socket permissions in this environment.
- next decision needed: decide whether to dispatch M9.C comparator/theta/raw-range audit immediately and whether to reclassify M16 as conditional for v0.1.0.

CRITIQUE_COMPLETE - Keep M10-M14 but insert M9.C before using M9 quantitatively, treat M16 as conditional for v0.1.0, and replan Phase B at 12-18 weeks including comparator/static-field cleanup.
