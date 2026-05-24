# Worker Report

## Summary
Summary: Extended the S2 orchestration probe timeout from 120s to 1800s in `scripts/m6_d02_baseline_run_instrumented.py` and reran the required command under `timeout 2400`. The real Gen2 d02 baseline still was not produced. The 1-second real replay probe timed out after 1810.003s including cleanup, with zero stdout/stderr and no `replay_probe.json`; the unchanged orchestrator then generated synthetic fallback sidecar inputs and outputs. All 12 sidecars executed, but they are not real replay outputs and do not satisfy the sprint's `replay_mode == "real"` acceptance gate. The required no-regression suite passed: 45 tests in 2112.81s.

Real run wall time: no real forecast wall time exists because the 3600s forecast did not start. Probe wall time was 1810.003s. Real RMSE vs Gen2: unavailable. Synthetic fallback RMSE values were T2=0.4 K, U10=0.5 m/s, V10=0.6 m/s, w_k20=0.02 m/s, theta_k20=0.3 K; these are plumbing values only.

## Files Changed
- `scripts/m6_d02_baseline_run_instrumented.py`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_real_replay.txt`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_d02_replay.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_s2_baseline_summary.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_s2dot1_baseline_summary.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_no_regression.txt`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_*.json` for the 12 S1 diagnostic sidecars
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/artifacts/sidecar-inputs/`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/artifacts/command-logs/`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/findings.md`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/findings_real.md`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/s3_input_memo.md`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/s3_input_memo_real.md`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/worker-report.md`

## Commands Run
1. Baseline orchestration:
```bash
timeout 2400 python scripts/m6_d02_baseline_run_instrumented.py \
  --duration-s 3600 \
  --output-dir .agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/ \
  | tee .agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_real_replay.txt
```

Exit code: 0. Key stdout:
```text
[probe] exit=None timeout=True elapsed_s=1810.003 stdout_bytes=0 stderr_bytes=0
[m6x-s2] wrote proof_s2_baseline_summary.json
{
  "status": "BLOCKER_SYNTHETIC_FALLBACK",
  "replay_mode": "synthetic",
  "fallback_reason": "real replay probe timed out after 1800s"
}
```

2. No-regression:
```bash
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m3_transfer_audit.py -v \
  | tee .agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_no_regression.txt
```

Exit code: 0. Key stdout:
```text
======================= 45 passed in 2112.81s (0:35:12) ========================
```

## Proof Objects
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_real_replay.txt`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_s2dot1_baseline_summary.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_s2_baseline_summary.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_d02_replay.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_bound_violation_tracer.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_sanitizer_audit.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_limiter_activation_tracker.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_field_rmse_timeline.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_spatial_divergence_map.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_conservation_tracker.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_boundary_ring_error_profiler.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_vertical_column_phase_space.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_operator_term_budget_tracer.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_transfer_launch_timeline.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_timestep_convergence_dashboard.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_stabilizer_provenance_scanner.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_no_regression.txt`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/findings_real.md`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/s3_input_memo_real.md`

Top-5 numerical findings, all from synthetic fallback only:
- Probe elapsed before timeout: 1810.003s; stdout_bytes=0; stderr_bytes=0.
- Synthetic candidate nonfinite/clip/changed totals: 0 / 0 / 0.
- Synthetic peak_w_abs_m_s: 0.0399081591550353.
- Synthetic theta range over run: 302.3 K to 302.76 K.
- Synthetic RMSEs: T2=0.4 K, U10=0.5 m/s, V10=0.6 m/s.

Comparison to S2 synthetic findings: the failure mode persisted after raising the probe guard from 120s to 1800s. S2 timed out at 130.002s; this rerun timed out at 1810.003s. The sidecar plumbing still runs and reports the same synthetic fallback shape, so there is still no real physics, transfer-audit, or performance evidence.

## Risks
- BLOCKER: the sprint did not meet the required real Gen2-anchored 1h baseline acceptance criterion.
- BLOCKER: every sidecar proof in this sprint is based on synthetic fallback inputs, not real replay data.
- The real first-nonfinite-step is unknown because the forecast did not start.
- Transfer audit is not binding for the real replay; the synthetic sidecar reported `TRANSFER_OR_CALLBACK_RISK`.
- The unchanged orchestrator has a hardcoded command-log directory from S2; I copied this run's command logs into the current sprint artifacts and removed only the untracked old-path logs generated by this run.
- I did not modify `src/gpuwrf/`.
- I did not remote-push, following the sprint contract's "No remote push" non-goal.

## Handoff
Objective: rerun the unchanged S2 d02 baseline orchestration with a 30-minute probe guard to try to produce a real Gen2 d02 1h baseline.

Files changed: timeout default in `scripts/m6_d02_baseline_run_instrumented.py`; current sprint proof files, findings, S3 memo, command logs, sidecar inputs, and this report.

Commands run: baseline orchestration with `timeout 2400`; required no-regression pytest command. Full stdout is in `proof_real_replay.txt` and `proof_no_regression.txt`.

Proof objects produced: all listed above, including the required summary alias `proof_s2dot1_baseline_summary.json`.

Unresolved risks: real replay probe still times out; no real 1h forecast, real RMSE, real first-nonfinite-step, or real transfer audit exists.

Next decision needed: run a focused infrastructure/debug sprint on why `scripts/m6_d02_boundary_replay_1h.py --duration-s 1` hangs beyond 1800s before dispatching S3 as an operator-fix sprint. S3 should not use this sprint's synthetic fallback numbers as real baseline evidence.
