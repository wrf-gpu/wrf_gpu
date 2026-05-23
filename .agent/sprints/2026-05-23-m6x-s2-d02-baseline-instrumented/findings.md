# Findings - M6.x S2 Instrumented Baseline

Replay mode: `synthetic`
Overall status: `BLOCKER_SYNTHETIC_FALLBACK`

Gen2 24h noise-floor anchors used for context only: T2 0.628 K, U10 1.46 m/s, V10 1.59 m/s.

## Classified Findings

### F01 - BLOCKER

Real Gen2 d02 baseline was not produced.

Detail: real replay probe timed out after 120s

Proof: `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_d02_replay.json`

### F02 - OK-WITHIN-NOISE

No sanitizer masking detected by available counts.

Detail: status=OK

Proof: `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_sanitizer_audit.json`

### F03 - BLOCKER

Transfer audit status.

Detail: status=TRANSFER_OR_CALLBACK_RISK; post_init_total_transfer_bytes=0

Proof: `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_transfer_launch_timeline.json`

### RMSE-T2 - OK-WITHIN-NOISE

T2 RMSE against Gen2 reference.

Detail: 1h_rmse=0.4; 24h_noise_floor=0.628406; 24h floor is an anchor, not a binding 1h threshold

Proof: `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_field_rmse_timeline.json`

### RMSE-U10 - OK-WITHIN-NOISE

U10 RMSE against Gen2 reference.

Detail: 1h_rmse=0.5; 24h_noise_floor=1.45648; 24h floor is an anchor, not a binding 1h threshold

Proof: `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_field_rmse_timeline.json`

### RMSE-V10 - OK-WITHIN-NOISE

V10 RMSE against Gen2 reference.

Detail: 1h_rmse=0.6; 24h_noise_floor=1.59097; 24h floor is an anchor, not a binding 1h threshold

Proof: `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_field_rmse_timeline.json`

### F04 - NEW-FINDING-NEEDS-S3-FIX

Limiter raw_dmu telemetry is not exposed by the preserved replay proof.

Detail: The sidecar ran, but the current replay proof only exposes sanitizer counts, not raw/bounded dmu arrays.

Proof: `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_limiter_activation_tracker.json`

### F05 - NEW-FINDING-NEEDS-S3-FIX

Operator RHS term budget is not exposed by the preserved replay proof.

Detail: No per-term replay arrays are serialized without operator-side hooks.

Proof: `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_operator_term_budget_tracer.json`

### F06 - EXPECTED-BAD

Stabilizer provenance scan found non-source-backed stabilizer-like code.

Detail: {"experiment-backed": 28, "reject": 0, "source-backed": 8}

Proof: `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_stabilizer_provenance_scanner.json`

## Known Gaps

- Real Gen2 d02 replay did not complete; synthetic fallback cannot support physics or performance claims.
- raw_dmu/bounded_dmu arrays are not serialized by the preserved replay scaffold.
- Per-substep RHS term arrays are not serialized by the preserved replay scaffold.
- Tier-3 timestep convergence remains S4 scope; S2 does not claim convergence.
