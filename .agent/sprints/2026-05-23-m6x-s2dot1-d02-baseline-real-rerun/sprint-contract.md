# Sprint Contract — M6.x S2.1: Real d02 Baseline Replay (rerun w/ longer wall budget)

## Objective

S2's replay probe timed out at 120s before JAX/GPU produced a proof (worker reported "130.002s elapsed, no stdout/stderr"). S2 fell back to synthetic data, leaving the project without a real Gen2 d02 baseline. The 120s probe guard was too tight — likely needed JAX compile + GPU warm-up + first substep.

This sprint reruns the same `scripts/m6_d02_baseline_run_instrumented.py` orchestration **with a longer wall budget** (target: 30+ minutes for the JAX warm-up to complete + 1h forecast to run). Goal: produce a real Gen2-anchored 1h baseline that S3 can use.

## Non-Goals

- NO edits to `src/gpuwrf/`. Strictly READ-ONLY on the operator.
- No re-running the 12 sidecars from scratch — use the S1 versions.
- No new sidecars.
- No 24h forecast yet.
- No fix attempts.
- No remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_s2dot1_baseline` on branch `worker/gpt/m6x-s2dot1-d02-baseline-real-rerun`.

Write-only:
- `scripts/m6_d02_baseline_run_instrumented.py` (modify ONLY to extend the timeout — keep the orchestration logic identical to what S2 produced)
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/` — proofs + worker-report

Read-only on `src/gpuwrf/`.

## Inputs

Required reading:
- `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/findings.md` — S2's BLOCKER F01 (timeout)
- `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/worker-report.md` — what S2 actually ran
- `scripts/m6_d02_baseline_run_instrumented.py` — the orchestration with the 120s probe guard to extend
- `scripts/m6_d02_boundary_replay_1h.py` — the underlying replay (probably the long one)
- `src/gpuwrf/integration/d02_replay.py` — the replay engine
- `data/fixtures/gen2_baseline/rmse_summary.csv` — Gen2 noise anchors (T2 24h 0.628 K, U10 1.46 m/s, V10 1.59 m/s)

## Acceptance Criteria

1. **Real replay completes 1h forecast**. The orchestration script's probe timeout extended to **1800s** (30 min). The forecast itself must finish 3600 simulated seconds. Capture `proof_real_replay.txt`.

2. **All 12 S1 sidecars produce real outputs**. Each `proof_*.json` in the sprint folder contains data from the real replay, not synthetic. The aggregator `proof_s2dot1_baseline_summary.json` aggregates them.

3. **Re-classified findings**. Write `findings_real.md` superseding S2's synthetic-data findings. Use the same 4-category classification (EXPECTED-BAD / NEW / OK-WITHIN-NOISE / BLOCKER), but now backed by real Gen2 data.

4. **Updated S3 input memo**. Write `s3_input_memo_real.md` superseding S2's. Same structure (top-3 source-cited fixes), now with real baseline numbers backing the recommendations.

5. **No regression**: `pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m3_transfer_audit.py -v` — all PASS (45+ tests).

6. **Real-replay first-nonfinite-step**: documented. If the forecast goes nonfinite before 1h, that's data — document the step + the bound-violation-tracer output at that step.

7. **Worker report** at `worker-report.md` with: real run wall time, real RMSE numbers vs Gen2, top-5 numerical findings, comparison to S2's synthetic findings, recommendation for S3.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_s2dot1_baseline
# Extend the timeout in the orchestration first:
sed -i 's/PROBE_TIMEOUT_S = 120/PROBE_TIMEOUT_S = 1800/' scripts/m6_d02_baseline_run_instrumented.py  # or whatever the actual variable name is
# Rerun:
timeout 2400 python scripts/m6_d02_baseline_run_instrumented.py \
  --duration-s 3600 \
  --output-dir .agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/ \
  | tee .agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_real_replay.txt
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_no_regression.txt
```

## Performance Metrics

- Wall time for 1h forecast on RTX 5090: report informational
- Peak GPU memory: report informational
- First-nonfinite-step: report (must be null OR documented)
- Transfer audit: must be 0 H2D/D2H bytes (binding)

## Proof Object

- `proof_real_replay.txt`
- `proof_s2dot1_baseline_summary.json`
- All 12 per-sidecar `proof_*.json`
- `proof_no_regression.txt`
- `findings_real.md` + `s3_input_memo_real.md`
- `worker-report.md`

Time budget: **30-60 minutes** (the replay itself is the slow step — actual JAX compile + 1h forecast).

## Risks

- **Still times out**: if 1800s isn't enough, the issue isn't timeout — it's the replay script or GPU environment. Document and report.
- **Gen2 d02 IC/BC truly unavailable**: if `/mnt/data/canairy_meteo/runs/wrf_l2/...` doesn't have the expected files, that's a BLOCKER. Document and recommend.
- **Forecast goes nonfinite**: that's data, not failure. Capture diagnostics at the nonfinite step.
- **Spec-gaming**: do NOT report synthetic data as real. The orchestration must verify `replay_mode == "real"` in every sidecar JSON.

## Handoff Requirements

When all proof files are on disk with real-replay data + worker-report.md committed: `/exit`. Wrapper sends AGENT REPORT to manager pane (session 2).

## Failure modes the manager will reject

- Reporting synthetic-mode data as real.
- Modifying `src/gpuwrf/`.
- Skipping the no-regression run.
- Fewer than 12 sidecar JSONs.
