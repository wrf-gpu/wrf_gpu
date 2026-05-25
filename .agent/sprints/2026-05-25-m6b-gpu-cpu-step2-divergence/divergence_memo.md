# Divergence Memo

Verdict: (A)-SENTINEL-COINCIDENCE

Evidence:
- 4x5 matrix: `.agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/proof_4path_step2_matrix.json`.
- V3-vs-comparator diff: `.agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/proof_v3_vs_comparator_diff.md`.
- Step-2 rows: `{"cpu_operational": {"all_state_leaves_finite": true, "largest_bad_field": null, "max_mu": 96738.5546875, "max_theta": 492.527099609375, "min_mu": 66364.8515625, "min_theta": 288.8056640625}, "cpu_validation": {"all_state_leaves_finite": true, "largest_bad_field": null, "max_mu": 99837.05073937865, "max_theta": 931797632.0, "min_mu": 67450.75934795201, "min_theta": -549520000.0}, "gpu_operational": {"all_state_leaves_finite": true, "largest_bad_field": null, "max_mu": 96738.5546875, "max_theta": 492.527099609375, "min_mu": 66364.8515625, "min_theta": 288.8056640625}, "gpu_validation": {"all_state_leaves_finite": true, "largest_bad_field": null, "max_mu": 99837.05073937865, "max_theta": 931797632.0, "min_mu": 67450.75934795201, "min_theta": -549520000.0}}`.

Recommended next sprint: 2026-05-25-m6b-comparator-nan-sentinel-audit: audit comparator max_abs_delta arithmetic, NaN sentinel handling, and field-order reporting.

Severity for M6 close: minor
Note: this resolves only the step-2 contradiction; the V3 physical-bounds blocker remains separate evidence against closing M6.
