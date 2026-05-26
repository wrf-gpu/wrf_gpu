Summary: M6 acceptance gate is BLOCKED, not closed. I added `scripts/m6_acceptance_tier4_all3.py`, ran the three V3 ICs through the +1h Tier-4 comparator, and wrote proof JSONs under this sprint folder. The gate failed on Stage 1 bounds and Stage 2 Tier-4 RMSE, so `.agent/decisions/MILESTONE-M6-CLOSEOUT.md` was not written.

## Files changed

- `scripts/m6_acceptance_tier4_all3.py`
- `.agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/worker-report.md`
- `.agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/proof_*.json`
- `.agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/validation_logs/*`
- `.agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/v3_521/*`
- `.agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/v3_509/*`

## Commands run and output

- `taskset -c 0-3 python scripts/m6_acceptance_tier4_all3.py --output .agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/`
  - stdout: `{"status": "M6-BLOCKED-ACCEPTANCE-GATE", "stage1_bounds_parity": "FAIL", "stage2_tier4_rmse": "FAIL", "closeout_written": false}`
  - stderr: none observed
- `taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all`
  - exit: 0
  - stdout/stderr: `validation_logs/m6b6_coupled_step_compare.{stdout,stderr}.txt`
  - key output: coupled-step parity passed, max_abs_delta 0.0 in the emitted comparisons.
- `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10`
  - exit: 0
  - stdout/stderr: `validation_logs/m6b_real_ic_operational_compare_steps10.{stdout,stderr}.txt`
  - key output: step 10 `max_abs_delta` 0.0, largest_bad_field null.
- `taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/v3_521/`
  - exit: 0
  - stdout: status `OK`, verdict `NAMED-FIX:boundary_application`
  - stderr: empty
- `taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z --output .agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/v3_509/`
  - exit: 0
  - stdout: status `IC-SPECIFIC`
  - stderr: one NumPy overflow warning in diagnostic L2 calculation, captured in `validation_logs/m6b_v3_localize_509.stderr.txt`

## Proof objects produced

- `proof_acceptance_summary.json`: final status `M6-BLOCKED-ACCEPTANCE-GATE`.
- `proof_bounds_parity.json`: Stage 1 `FAIL`.
  - 20260429 first bounds failure: step 36, U abs max 103.086 m/s.
  - 20260509 first bounds failure: step 12, W abs max 66.032 m/s.
  - 20260521 first bounds failure: step 75, W abs max 1180.878 m/s.
- `proof_tier4_rmse_all3.json`: Stage 2 `FAIL`.
  - T2 RMSE: 20260429 `7.360880204405183e85` K, 20260509 `2.0749024415139776` K, 20260521 `2.8059015813198628e84` K; aggregate mean `2.5471567875123902e85` K, threshold 3 K.
  - U10/V10 RMSE passed both per-IC and aggregate thresholds.
- Per-IC bounds/RMSE JSONs: `proof_bounds_<run_id>.json`, `proof_tier4_rmse_<run_id>.json`.
- Localization proof directories: `v3_521/`, `v3_509/`.

## Risks

- Acceptance is blocked by real forecast instability at +1h. No bug fix was attempted per contract.
- The acceptance driver skipped embedded per-IC parity subprocesses after bounds/RMSE had already blocked the gate; the standalone required parity commands were run and exited 0.
- No M6 closeout memo was written because the contract allows it only for a green acceptance gate.

## Handoff

Objective: run M6 Tier-4 RMSE acceptance on all three V3 ICs and close M6 only if green.
Files changed: listed above.
Commands run: listed above with captured outputs.
Proof objects produced: listed above.
Unresolved risks: Stage 1 wind bounds and Stage 2 T2 RMSE are red; M6 remains blocked.
Next decision needed: choose the next diagnostic/fix sprint for the T2/wind blowups before retrying M6 acceptance.
