# Worker Report

Summary: Completed the S3-narrow stabilizer source cleanup inside the worker-owned files. The provenance scanner now reports experiment-backed 28 baseline -> 20 current and source-backed 8 baseline -> 37 current, with reject count still 0. `_mu_continuity_increment` remains in place and is explicitly marked DEFER to post-S2.1 pending real Gen2 baseline evidence.

## Per-Stabilizer Table

| Stabilizer | Change | Source citation / status | Before -> after |
|---|---|---|---|
| `MPAS_OMEGA_TO_W_METRIC = 1.35` | Removed as the production conversion. Added `_mpas_w_metric_faces` using per-face geometry and dry-column mass. Analytic slice runs import the legacy value from `gpuwrf.validation.mpas_oracles.mpas_column_slice` only. | MPAS `mpas_atm_time_integration.F:2491-2495`, `:5575-5585`. | experiment-backed -> source-backed production path; slice-only compatibility retained. |
| `MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = 0.38` | Demoted behind `SLICE_ONLY_MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE`; non-analytic production path uses `SOURCE_BACKED_COLUMN_BUOYANCY_TENDENCY_SCALE = 1.0`. | WRF `module_small_step_em.F:1451-1489`; MPAS `mpas_atm_time_integration.F:2160-2169` do not define the scalar gain. | production scalar gain removed; slice-only compatibility retained for existing oracle tests. |
| `_mu_continuity_increment` tanh cap | Not removed. Added explicit DEFER note and source lines showing WRF/MPAS do not contain this cap. | WRF `module_small_step_em.F:1102-1119`; MPAS `mpas_atm_time_integration.F:2146-2199`. | retained by contract, documented as not source-backed. |
| `smdiv` pressure memory | Added immediate WRF source comments/docstrings around config, call site, and damping helper. | WRF `module_small_step_em.F:548-563`. | experiment-backed scanner hits converted to source-backed. |
| Rayleigh damping | Added immediate WRF/MPAS source comments/docstrings around config, call site, ramp, and damping helper. | WRF `module_small_step_em.F:1559-1569`; MPAS `mpas_atm_time_integration.F:2184-2192`. | experiment-backed scanner hits converted to source-backed. |

## Files Changed

- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `src/gpuwrf/dynamics/damping.py`
- `tests/test_m6x_s3narrow_stabilizer_audit.py`
- `.agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_stabilizer_after.json`
- `.agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_stabilizer_after.txt`
- `.agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_audit_test.txt`
- `.agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_no_regression.txt`
- `.agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/worker-report.md`

## Commands Run

1. `python scripts/diagnostic_stabilizer_provenance_scanner.py --input src/gpuwrf/dynamics/ --output .agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_stabilizer_after.json | tee .agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_stabilizer_after.txt`
   - Exit 0. Stdout is empty by scanner design. JSON output: `{"experiment-backed": 20, "reject": 0, "source-backed": 37}`, status `OK`.
2. `pytest tests/test_m6x_s3narrow_stabilizer_audit.py -v | tee .agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_audit_test.txt`
   - Exit 0. Output summary: `4 passed in 0.13s`.
3. `pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_no_regression.txt`
   - Exit 0. Output summary: `45 passed in 1750.72s (0:29:10)`.
4. `python -m py_compile src/gpuwrf/dynamics/acoustic_wrf.py src/gpuwrf/dynamics/damping.py tests/test_m6x_s3narrow_stabilizer_audit.py`
   - Exit 0. No stdout.

## Proof Objects

- `.agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_stabilizer_after.json`
- `.agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_stabilizer_after.txt`
- `.agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_audit_test.txt`
- `.agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_no_regression.txt`
- `tests/test_m6x_s3narrow_stabilizer_audit.py`

## Performance / Transfer Notes

- Vertical operator launch count was not reprofiled with Nsight in this cleanup sprint. The code path remains inside the existing JIT recurrence; no new host callbacks or timestep-loop transfers were introduced.
- Transfer proof remains covered by the required regression command: `tests/test_m3_transfer_audit.py::*` all passed, including `test_transfer_audit_artifact_is_zero_post_init`.

## Risks

- The source-derived MPAS `rw`/`w` metric is active for non-analytic metric provenance, but existing analytic slice tests still need the oracle's legacy 1.35 value. This is intentionally isolated to `metrics.provenance.startswith("analytic")` and imports the value from the slice oracle module.
- `_mu_continuity_increment` is still load-bearing and not source-backed. It is explicitly deferred to the post-S2.1 sprint with real Gen2 baseline evidence.
- Remaining experiment-backed scanner hits are in `hyperdiffusion.py`, `limiters.py`, and `orchestrator.py`, which were outside this sprint's file ownership.

## Handoff

- Objective: reduce source-provenance imbalance without removing the load-bearing mass limiter or expanding carry.
- Files changed: listed above.
- Commands run: listed above with outputs.
- Proof objects produced: listed above.
- Unresolved risks: analytic slice compatibility branch; deferred `_mu_continuity_increment`; out-of-scope limiter/hyperdiffusion/orchestrator findings.
- Next decision needed: S2.1 should decide whether `_mu_continuity_increment` can be removed or source-ratified against the real Gen2 d02 baseline.
