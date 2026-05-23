# Worker Report

## Summary
Summary: Implemented the S2 measurement-only orchestration script and ran the required validation commands. The current real ADR-023 d02 replay did **not** produce a real 1h baseline: the preserved `scripts/m6_d02_boundary_replay_1h.py` probe timed out after 120s, with no stdout/stderr and no `replay_probe.json`. The orchestrator therefore used the contract-approved synthetic Gen2-shaped fallback and classified the result as `BLOCKER_SYNTHETIC_FALLBACK`. All 12 S1 diagnostic sidecars ran and wrote JSON proofs. The no-regression suite passed: 45 tests in 2427.82s.

1h forecast verdict: real forecast did not complete. Sanitizer masking was not measured on a real replay; the fallback sidecar reports zero synthetic sanitizer changes only. The run must not be used for physics or GPU-performance claims.

## Files Changed
- `scripts/m6_d02_baseline_run_instrumented.py`
- `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/findings.md`
- `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/s3_input_memo.md`
- `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/worker-report.md`
- Proof files under `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/`

## Commands Run
1. Baseline orchestration:
```bash
python scripts/m6_d02_baseline_run_instrumented.py \
  --duration-s 3600 \
  --output-dir .agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/ \
  | tee .agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_baseline_run.txt
```
Exit code: 0. Key output:
```text
[probe] exit=None timeout=True elapsed_s=130.002 stdout_bytes=0 stderr_bytes=0
[m6x-s2] wrote proof_s2_baseline_summary.json
{
  "status": "BLOCKER_SYNTHETIC_FALLBACK",
  "replay_mode": "synthetic",
  "fallback_reason": "real replay probe timed out after 120s"
}
```

2. No-regression:
```bash
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_no_regression.txt
```
Exit code: 0. Output:
```text
======================= 45 passed in 2427.82s (0:40:27) ========================
```

## Proof Objects
- `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_baseline_run.txt`
- `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_s2_baseline_summary.json`
- `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_d02_replay.json`
- `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_*_tracer.json`, `proof_*_audit.json`, and other per-sidecar JSON proofs
- `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_no_regression.txt`
- `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/findings.md`
- `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/s3_input_memo.md`

Top numerical findings:
- Real replay probe timeout: 120s guard, 130.002s elapsed, no stdout/stderr.
- Synthetic fallback surface RMSEs: T2 0.4 K, U10 0.5 m/s, V10 0.6 m/s; these are non-binding because no real Gen2 replay completed.
- Synthetic sanitizer counts: candidate nonfinite 0, clip 0, changed 0; not real replay evidence.
- Stabilizer scanner: 28 experiment-backed findings, 8 source-backed, 0 reject.
- Regression suite: 45/45 passed in 2427.82s.

## Risks
- BLOCKER: no real Gen2 d02 baseline was produced in this environment. The replay probe timed out before JAX/GPU execution produced a proof.
- Synthetic fallback artifacts exercise sidecar plumbing only; they cannot support physics correctness, sanitizer-masking, transfer-audit, or performance claims.
- The preserved replay proof shape does not expose raw/bounded `dmu` arrays or per-substep RHS term arrays, so limiter activation and operator term budget are marked as S3 input gaps.
- I did not modify `src/gpuwrf/` or any operator code.
- Contract conflict: role prompt asked for remote push, but sprint contract lists "No remote push." I followed the sprint contract.

## Handoff
Objective: measure the unchanged ADR-023 1h d02 replay baseline with S1 sidecars.

Files changed: new orchestration script plus sprint-folder findings, memo, proofs, and this report.

Commands run: baseline orchestration and no-regression pytest command above; full stdout is in the proof text files.

Proof objects produced: see list above.

Unresolved risks: real d02 baseline remains missing; GPU/JAX replay probe timed out; limiter raw telemetry and term-budget telemetry are not serialized by the preserved scaffold.

Next decision needed: rerun the same script on a healthy GPU/JAX replay environment before designing an S3 operator fix. If S3 proceeds without that, scope it as instrumentation/telemetry and source-backed limiter cleanup, not as a Tier-3-producing fix sprint.
