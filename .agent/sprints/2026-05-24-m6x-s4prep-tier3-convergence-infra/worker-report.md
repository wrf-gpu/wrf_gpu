# Worker Report

## Summary
Summary: Built the M6.x Tier-3 dt-convergence infrastructure for the idealized `flat_warm_bubble_tier3` case. The case is independent of d02 replay and uses a small flat cosine-bell theta perturbation so this is a convergence harness, not the old warm-bubble amplitude gate. The runner exercises the current ADR-023 `run_acoustic_scan_carry` path without editing `src/gpuwrf/dynamics/`. The required smoke run completed finite and returned `PASS_TIER3`; this is an infrastructure smoke result only, not an attempt to promote ADR-023.

Required input note: `.agent/sprints/2026-05-21-m6-milestone-plan-scout/m6-milestone-plan.md` is absent in this worktree. I used the available `critical-review-codex.md` lines 54-59 and 151-153 plus `manager-amendments.md`, `VALIDATION_STRATEGY.md`, and the S4 critic report.

## Files Changed
- `scripts/m6_tier3_convergence_runner.py`
- `src/gpuwrf/validation/tier3_envelope.py`
- `tests/test_m6x_tier3_convergence_infra.py`
- `data/fixtures/tier3_idealized/case_definition.json`
- `.agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_tier3_smoke_current_state.json`
- `.agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_tier3_smoke.txt`
- `.agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_smoke_test.txt`
- `.agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_no_regression.txt`
- `.agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/worker-report.md`

## Commands Run
- `python scripts/m6_tier3_convergence_runner.py --case flat_warm_bubble_tier3 --dt 1.0 --output .agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_tier3_smoke_current_state.json | tee .agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_tier3_smoke.txt`
  Output: `verdict=PASS_TIER3`; rationale: all refined dt-pair RMSE/Linf norms stayed within configured growth bounds. Final 4s RMSE pair0 -> pair1: U `16.2711 -> 7.9220`, V `16.2711 -> 7.9220`, W `3.759e-4 -> 1.681e-4`, theta `3.852e-4 -> 1.442e-4`, p perturbation `0.15035 -> 0.06030`, mu perturbation `40.9207 -> 29.9023`.
- `pytest tests/test_m6x_tier3_convergence_infra.py -v | tee .agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_smoke_test.txt`
  Output: `3 passed in 6.93s`.
- `pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_s3narrow_stabilizer_audit.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_no_regression.txt`
  Output: `49 passed in 26.14s`.

## Proof Objects
- `.agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_tier3_smoke_current_state.json`
- `.agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_tier3_smoke.txt`
- `.agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_smoke_test.txt`
- `.agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_no_regression.txt`

## Risks
- The smoke case is intentionally tiny (`4x4x8`, 4 seconds) so it proves infrastructure readiness, not production Tier-3 adequacy.
- The transfer audit in the smoke JSON is static: the runner uses a single JAX scan body and transfers checkpoint arrays after `block_until_ready`; it does not include an nsys/ncu trace.
- The current tiny case returned `PASS_TIER3`; S4 should still run a stronger case/window after S2.2, S2.1-redo, and S3-real land.
- `data/fixtures/tier3_idealized/case_definition.json` sits under ignored `data/`, so it must be force-added if committed.

## Handoff
Objective complete: Tier-3 idealized dt-doubling infrastructure exists, schema is locked in the smoke test, and the current ADR-023 unified path was smoke-run honestly. No `src/gpuwrf/dynamics/` files were modified. Next decision: manager/tester should decide whether the S4 real gate uses this flat case as-is or extends it to a longer `em_hill2d_x`-style setup before making any M6 correctness claim.
