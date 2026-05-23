# Findings Real - M6.x S2.1 d02 Baseline Rerun

Replay mode: `synthetic`
Overall status: `BLOCKER_SYNTHETIC_FALLBACK`
Real Gen2 d02 1h baseline: `NOT PRODUCED`

This file supersedes the S2 synthetic findings by rerunning the same orchestration with the probe guard extended from 120s to 1800s. The rerun still did not produce `replay_mode == "real"` evidence: the 1-second real replay probe timed out after 1810.003s with zero stdout and zero stderr, so the unchanged orchestrator fell back to synthetic data.

Gen2 24h noise-floor anchors used for context only: T2 0.628 K, U10 1.46 m/s, V10 1.59 m/s. No real 1h RMSE against Gen2 was produced in this sprint.

## Classified Findings

### F01 - BLOCKER

Real Gen2 d02 baseline was not produced.

Detail: the real replay probe timed out after 1800s (`elapsed_s=1810.003` including cleanup), with no `replay_probe.json`, no probe stdout, and no probe stderr. The full 3600 simulated second forecast did not start.

Proof: `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_real_replay.txt`, `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_s2dot1_baseline_summary.json`

### F02 - BLOCKER

The 12 sidecar outputs are not real replay outputs.

Detail: all 12 sidecars executed, but their inputs came from the orchestrator's synthetic fallback payload. They are useful only as plumbing evidence, not physics, transfer, or S3-fix evidence.

Proof: `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_d02_replay.json`, `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_s2dot1_baseline_summary.json`

### F03 - BLOCKER

Transfer audit status is not binding for the real replay.

Detail: sidecar status was `TRANSFER_OR_CALLBACK_RISK` with `post_init_total_transfer_bytes=0`, but it was computed from synthetic fallback metadata. The required real replay transfer audit was not produced.

Proof: `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_transfer_launch_timeline.json`

### RMSE-REAL - BLOCKER

Real 1h RMSE against Gen2 is unavailable.

Detail: synthetic fallback RMSE values were T2=0.4 K, U10=0.5 m/s, V10=0.6 m/s, w_k20=0.02 m/s, theta_k20=0.3 K. These are generated fallback values and must not be interpreted as real Gen2-anchored errors.

Proof: `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_field_rmse_timeline.json`

### F04 - NEW-FINDING-NEEDS-S3-FIX

Limiter raw_dmu telemetry is still not exposed by the preserved replay proof.

Detail: the sidecar ran, but the current replay proof only exposes sanitizer counts, not raw/bounded dmu arrays. Because the run was synthetic fallback, this remains an instrumentation gap rather than real replay evidence.

Proof: `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_limiter_activation_tracker.json`

### F05 - NEW-FINDING-NEEDS-S3-FIX

Operator RHS term budget is still not exposed by the preserved replay proof.

Detail: no per-term replay arrays are serialized without operator-side hooks. Because the real replay did not start, S3 cannot prioritize operator fixes from real term-budget data yet.

Proof: `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_operator_term_budget_tracer.json`

### F06 - EXPECTED-BAD

Stabilizer provenance scan still finds non-source-backed stabilizer-like code.

Detail: classification counts were `{"experiment-backed": 28, "reject": 0, "source-backed": 8}`. This repeats the S2 shape and is independent of real replay completion.

Proof: `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_stabilizer_provenance_scanner.json`

## Known Gaps

- No real Gen2 d02 1h forecast completed.
- `first_nonfinite_step` is not known for the real forecast because the forecast did not start.
- The sidecar JSONs do not satisfy `replay_mode == "real"`; they are synthetic fallback plumbing outputs.
- Synthetic fallback RMSE values are not physics evidence and are not valid S3 baseline numbers.
- Raw limiter arrays and per-term RHS arrays remain unavailable in the preserved replay proof.
