# Worker Report — M6 Acceptance Attempt 2

Summary: M6 acceptance remains blocked. I made the single allowed code change, removing the 75-step cap in `scripts/m6_guard_disabled_debug.py`. The required 360-step probe for `20260521_18z_l3_24h_20260522T072630Z` reached a theta 10x-envelope breach at step 339 (`theta=7147.17138671875 K`, ratio `10.210244838169643`, cell `[25,19,97]`, first operator localized as `acoustic`). The two other guard-disabled probe invocations failed before running because `m6_guard_disabled_debug.py` is pinned to the 20260521 run ID. Tier-4 RMSE passed on all three ICs, but the acceptance summary is `M6-BLOCKED-ACCEPTANCE-GATE` because bounds/parity aggregate status is `FAIL`; the nested 20260509 10-step compare failed with `max_abs_delta=1e+300` and `largest_bad_field=mu`. No `.agent/decisions/MILESTONE-M6-CLOSEOUT.md` was written.

## Files Changed

- `scripts/m6_guard_disabled_debug.py`
- `.agent/sprints/2026-05-26-m6-acceptance-attempt-2/worker-report.md`
- `.agent/sprints/2026-05-26-m6-acceptance-attempt-2/probe_20260521/*`
- `.agent/sprints/2026-05-26-m6-acceptance-attempt-2/tier4/*`
- `.agent/sprints/2026-05-26-m6-acceptance-attempt-2/logs/*`

## Commands Run + Output

- `taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id "$IC" --n-steps 360 --output ...` for the three ICs. Output captured in `logs/stage1_2_probes.log`. Key output: 20260521 reported `first_explosive_step.step=339`, `field=theta`, `value=7147.17138671875`, `ratio_to_envelope=10.210244838169643`; 20260509 and 20260429 raised `ValueError: this diagnostic is pinned to 20260521_18z_l3_24h_20260522T072630Z`.
- `taskset -c 0-3 python scripts/m6_acceptance_tier4_all3.py --output .../tier4/`. Output captured in `logs/stage3_tier4.log`. Key output: `status=M6-BLOCKED-ACCEPTANCE-GATE`, `stage1_bounds_parity=FAIL`, `stage2_tier4_rmse=PASS`, `closeout_written=false`.
- `taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all`. Output captured in `logs/stage4_m6b6_coupled_step_compare.log`. Key output: `passed=true`, `diverging_field_count=0`, `decision=PROCEED_TO_M6_PERF_DESIGN`.
- `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10`. Output captured in `logs/stage4_m6b_real_ic_operational_compare_steps10.log`. Key output: `status=PASS`, `final_max_abs_delta=0.0`.
- `taskset -c 0-3 pytest tests/test_m6_guard_disabled_debug.py -v`. Output captured in `logs/stage4_pytest_m6_guard_disabled_debug.log`. Key output: `12 passed in 1.03s`.

## Proof Objects Produced

- `.agent/sprints/2026-05-26-m6-acceptance-attempt-2/probe_20260521/proof_first_explosive_step.json`
- `.agent/sprints/2026-05-26-m6-acceptance-attempt-2/probe_20260521/proof_first_explosive_operator.json`
- `.agent/sprints/2026-05-26-m6-acceptance-attempt-2/tier4/proof_acceptance_summary.json`
- `.agent/sprints/2026-05-26-m6-acceptance-attempt-2/tier4/proof_bounds_parity.json`
- `.agent/sprints/2026-05-26-m6-acceptance-attempt-2/tier4/proof_tier4_rmse_all3.json`
- Per-IC Tier-4 bounds/RMSE and parity command logs under `.agent/sprints/2026-05-26-m6-acceptance-attempt-2/tier4/`.

## Risks

- Acceptance is red despite Tier-4 RMSE passing; the guard-disabled probe still breaches the 10x theta envelope at step 339.
- The guard-disabled debug script cannot run the requested 20260509 and 20260429 probes without changing its run-ID pin, which was outside file ownership.
- Nested Tier-4 parity exposed a 20260509 10-step compare failure even though the standalone default 20260521 compare passes.
- Remote push was not performed because the sprint contract explicitly lists `NO remote push`.

## Handoff

Objective: run the M6 acceptance gate exactly as contracted after removing the diagnostic cap.

Unresolved risks: step-339 acoustic theta breach, run-ID pin preventing two guard-disabled probes, and 20260509 nested multi-step parity failure.

Next decision needed: manager should decide whether to dispatch a diagnostic fix sprint for the step-339 acoustic breach / 20260509 parity failure or revise the guard-disabled probe contract to support all three ICs.
