# Sprint Contract — M6.x S2.1-redo: REAL d02 1h Baseline (post-hang-fix)

## Objective

S2.2 fixed the d02 replay hang (commit `4ee4d31` — dynamic `jax.lax.cond` radiation predicate → static helpers). The replay now completes: d=1 in 90s wall, d=60 in 231s, d=300 in 209s.

This sprint finally runs the REAL Gen2-anchored 1h d02 baseline that S2 and S2.1 couldn't produce. Use the post-S3-narrow + post-S2.2 unified ADR-023 operator. Instrument with the 12 S1 sidecars. Produce a real `s3_input_memo.md` that S3-real can act on.

## Non-Goals

- NO edits to `src/gpuwrf/dynamics/`, `src/gpuwrf/contracts/`, `src/gpuwrf/physics/`. STRICTLY READ-ONLY on the operator.
- No new sidecars.
- No operator fix attempts.
- No 24h forecast (S5 does that).
- No remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_s2dot1redo` on branch `worker/gpt/m6x-s2dot1redo-real-baseline`.

Write-only:
- `scripts/m6_d02_baseline_run_instrumented.py` — minor adjustments only (e.g., probe timeout). NO operator changes.
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/` — proofs + worker-report

Read-only everywhere else.

## Inputs

Required reading:
- **`.agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/worker-report.md`** — what the hang fix did
- `scripts/m6_d02_boundary_replay_1h.py` — post-fix; verified working
- `src/gpuwrf/integration/d02_replay.py` — post-fix
- `scripts/m6_d02_baseline_run_instrumented.py` — orchestrator that wires sidecars to replay
- All 12 `scripts/diagnostic_*.py` (S1)
- `data/fixtures/gen2_baseline/rmse_summary.csv` — Gen2 noise floor anchors
- `.agent/decisions/source_mining_operator_table.md` — for s3_input_memo cross-references

## Acceptance Criteria

### 1. Real 1h forecast completes

Run the orchestrator at `--duration-s 3600` with the post-hang-fix replay. Per the S2.2 results, expect ~30-60min wall time (mostly JAX compile + 1h sim). Capture `proof_real_1h_run.txt`.

Verdict must be `replay_mode = "real"` (NOT synthetic) in every sidecar JSON. If any sidecar still falls back to synthetic, that's a BLOCKER.

### 2. All 12 S1 sidecars produce real data

Per-sidecar `proof_*.json` files with real-replay data. Aggregate into `proof_s2dot1redo_summary.json`.

### 3. Findings classification

`findings_real.md` (NEW — supersedes S2 and S2.1 synthetic findings):
- `EXPECTED-BAD` / `NEW-FINDING-NEEDS-S3-FIX` / `OK-WITHIN-NOISE` / `BLOCKER` per finding
- For each `OK-WITHIN-NOISE`, cite the Gen2 noise floor anchor

### 4. S3-real input memo

`s3_input_memo_real.md` (NEW):
- Top-3 operator concerns from real baseline
- Each cross-referenced to `source_mining_operator_table.md`
- Expected effect on baseline numbers when fixed
- Exit-rule status: would S3-real + 1 fix sprint plausibly produce Tier-3 PASS?

### 5. No regression

```bash
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_d02_boundary_replay.py tests/test_m6x_d02_replay_hang_debug.py tests/test_m6x_s3narrow_stabilizer_audit.py tests/test_m6x_tier3_convergence_infra.py tests/test_m3_transfer_audit.py -v
```
All PASS (50+ tests).

### 6. Worker report

`worker-report.md` with: 1h wall time, real RMSE vs Gen2 at t=15/30/45/60min, comparison to Gen2 noise floor anchors, top-5 numerical findings, files changed (just orchestrator adjustments if any), risks, handoff to S3-real.

### 7. Branch commits on `worker/gpt/m6x-s2dot1redo-real-baseline`. Multiple commits OK.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_s2dot1redo
timeout 3600 python scripts/m6_d02_baseline_run_instrumented.py \
  --duration-s 3600 \
  --output-dir .agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/ \
  | tee .agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_real_1h_run.txt
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_d02_boundary_replay.py tests/test_m6x_d02_replay_hang_debug.py tests/test_m6x_s3narrow_stabilizer_audit.py tests/test_m6x_tier3_convergence_infra.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_no_regression.txt
```

## Performance Metrics

- 1h wall time on RTX 5090: report informational
- Peak GPU memory: report informational
- First-nonfinite-step: report (must be null OR documented)
- Transfer audit: must be 0 H2D/D2H bytes (binding)

## Proof Object

- `proof_real_1h_run.txt` — main orchestration log
- `proof_s2dot1redo_summary.json` — aggregated
- Per-sidecar `proof_*.json` (real data, all 12)
- `proof_no_regression.txt`
- `findings_real.md`
- `s3_input_memo_real.md`
- `worker-report.md`

Time budget: **45-90 minutes** (the replay is now 30-60min based on S2.2 timing + sidecar overhead).

## Risks

- **Forecast goes nonfinite during 1h**: that's still data. Capture first-nonfinite step + bound-violation-tracer output at that step.
- **Sidecar still falls back to synthetic**: the orchestrator's `replay_probe` may still timeout. Disable the probe — go directly to the long replay with full timeout.
- **Spec-gaming**: every numerical claim cites the JSON proof.
- **CPU budget**: bound to cores 0-3 via dispatch_role_session2.sh wrapper; verify by checking ps aux during run.

## Handoff Requirements

When all proof files on disk + findings_real + s3_input_memo_real + worker-report committed: `/exit`. Wrapper sends AGENT REPORT to manager pane.

## Failure modes the manager will reject

- Modifying any file under `src/gpuwrf/dynamics/`.
- Falling back to synthetic data (S2.2 proved the replay works — no excuses).
- Sub-hourly RMSE claims without truth files (only hourly Gen2 leads have truth).
- Skipping the no-regression run.
