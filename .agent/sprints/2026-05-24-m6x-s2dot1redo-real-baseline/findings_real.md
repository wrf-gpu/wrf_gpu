# Findings - M6.x S2 Instrumented Baseline

Replay mode: `real`
Overall status: `NEEDS_S3_FIX`

Gen2 24h noise-floor anchors used for context only: T2 0.628 K, U10 1.46 m/s, V10 1.59 m/s.

## Classified Findings

### F01 - NEW-FINDING-NEEDS-S3-FIX

Physical-bound tracer found a post-sanitize bound violation.

Detail: {"bound": 400.0, "comparator": "<=", "field": "theta_K", "i": null, "j": null, "k": null, "step": 3600, "time_s": 3600.0, "units": "K", "value": 550.0}

Proof: `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_bound_violation_tracer.json`

### F02 - EXPECTED-BAD

Sanitizer changed pre-sanitize candidates.

Detail: {"candidate_changed_count": 20049146903, "candidate_clip_count": 2811641892, "candidate_nonfinite_count": 17237505011}

Proof: `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_sanitizer_audit.json`

### F03 - OK-WITHIN-NOISE

Transfer audit status.

Detail: status=OK; post_init_total_transfer_bytes=0

Proof: `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_transfer_launch_timeline.json`

### RMSE-T2 - NEW-FINDING-NEEDS-S3-FIX

T2 RMSE against Gen2 reference.

Detail: 1h_rmse=136.885; 24h_noise_floor=0.628406; 24h floor is an anchor, not a binding 1h threshold

Proof: `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_field_rmse_timeline.json`

### RMSE-U10 - NEW-FINDING-NEEDS-S3-FIX

U10 RMSE against Gen2 reference.

Detail: 1h_rmse=106.419; 24h_noise_floor=1.45648; 24h floor is an anchor, not a binding 1h threshold

Proof: `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_field_rmse_timeline.json`

### RMSE-V10 - NEW-FINDING-NEEDS-S3-FIX

V10 RMSE against Gen2 reference.

Detail: 1h_rmse=102.232; 24h_noise_floor=1.59097; 24h floor is an anchor, not a binding 1h threshold

Proof: `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_field_rmse_timeline.json`

### F04 - NEW-FINDING-NEEDS-S3-FIX

Limiter raw_dmu telemetry is not exposed by the preserved replay proof.

Detail: The sidecar ran, but the current replay proof only exposes sanitizer counts, not raw/bounded dmu arrays.

Proof: `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_limiter_activation_tracker.json`

### F05 - NEW-FINDING-NEEDS-S3-FIX

Operator RHS term budget is not exposed by the preserved replay proof.

Detail: No per-term replay arrays are serialized without operator-side hooks.

Proof: `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_operator_term_budget_tracer.json`

### F06 - EXPECTED-BAD

Stabilizer provenance scan found non-source-backed stabilizer-like code.

Detail: {"experiment-backed": 10, "reject": 0, "source-backed": 37}

Proof: `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_stabilizer_provenance_scanner.json`

## Known Gaps

- raw_dmu/bounded_dmu arrays are not serialized by the preserved replay scaffold.
- Per-substep RHS term arrays are not serialized by the preserved replay scaffold.
- Tier-3 timestep convergence remains S4 scope; S2 does not claim convergence.
