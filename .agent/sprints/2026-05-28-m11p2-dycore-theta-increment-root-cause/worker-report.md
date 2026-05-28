# Worker Report - M11.2 Dycore Theta Increment Root Cause

Summary: `M11P2_PARTIAL`; no safe model-code fix survived the required gates.

Verdict: `M11P2_PARTIAL`.

## Outcome

No model-code fix was committed. The current worktree does not reproduce the contract's theta-only first failure: the re-run 1h harness fails first on `wind_in_bounds` at step 72 and reaches dycore nonfinite cells at step 93. Candidate theta-line fixes were tested and rejected because they either worsened AC3 or did not reduce limiter activity enough.

## Files changed

- `proofs/m11p2/diagnostic_report_after_fix.json`
- `proofs/m11p2/limiter_diagnostics_24h.json`
- `proofs/m11p2/pipeline_run_20260521.json`
- `proofs/m11p2/post_m11p2_skill_diff.json`
- `proofs/m11p2/dycore_100_steps_pytest.txt`
- `.agent/sprints/2026-05-28-m11p2-dycore-theta-increment-root-cause/root_cause_analysis.md`
- `.agent/sprints/2026-05-28-m11p2-dycore-theta-increment-root-cause/worker-report.md`

## Commands run

- `taskset -c 0-3 python scripts/run_diagnostic_harness.py --hours 1 --radiation-cadence-steps 999999 --jax-platform cuda --output proofs/m11p2/diagnostic_report_after_fix.json`
  - required a runtime compatibility shim because `comprehensive_harness.py` imports stale `_limit_theta_by_level`
  - result: `wind_in_bounds` first failure at step 72; dycore nonfinite at step 93
- `taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py`
  - result: `1 passed in 439.24s`
- 24h limiter diagnostic with radiation disabled
  - result: `FAIL`; `8640/8640` limited steps, max clip `442112`, mass residual `Infinity`
- `taskset -c 0-3 python scripts/m7_daily_pipeline.py --hours 24 --output-dir /tmp/m11p2_dycore_theta_20260521 --proof-dir proofs/m11p2 --score`
  - result: `PIPELINE_BLOCKED`; nonfinite model state after forecast hour 1

## Proof objects produced

- `proofs/m11p2/diagnostic_report_after_fix.json`
- `proofs/m11p2/limiter_diagnostics_24h.json`
- `proofs/m11p2/pipeline_run_20260521.json`
- `proofs/m11p2/post_m11p2_skill_diff.json`
- `proofs/m11p2/dycore_100_steps_pytest.txt`

## Acceptance status

- AC1: partial. Candidate theta lines and WRF references are documented; final root cause is not isolated.
- AC2: fail. No safe model-code fix survived AC3.
- AC3: fail. First invariant is `wind_in_bounds` at step 72, not theta fixed/pushed past 500.
- AC4: fail. Limiter activity is `8640/8640` steps; max cells `442112`; residual `Infinity`.
- AC5: pass. 100-step parity passed.
- AC6: fail. 24h pipeline blocked at hour 1; T2 RMSE unavailable.

## Unresolved risks

- The sprint premise is stale for this worktree: dycore wind/mass instability precedes the theta-only failure.
- `scripts/run_diagnostic_harness.py` currently needs a compatibility shim for stale `_limit_theta_by_level`; no non-contract source files were edited to repair that.
- No T2 recovery number exists because the 24h pipeline produced no scored forecast.

## Next decision needed

Dispatch a prerequisite sprint to localize the current step-72 dycore wind/mu blow-up, or reset to the exact M17 state where theta first fails at step 141 before retrying M11.2.

Headline: limiter-activity drop `0%`; post-fix T2 RMSE unavailable.
