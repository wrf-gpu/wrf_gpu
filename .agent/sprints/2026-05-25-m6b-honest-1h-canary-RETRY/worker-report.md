# Worker Report - M6b Honest 1h Canary RETRY

## objective

Re-run the M6b honest Canary d02 operational forecast on the 3 pinned Gen2 IDs after the operational carry expansion, with sanitizer off, CPU cores 0-3, B6 acoustic cadence, corrected per-level theta bounds, Tier-4 RMSE gates, spatial-divergence audit, and an M6 close recommendation.

## verdict

M6 close recommendation: `BLOCKER`.

Primary blocker: `THETA_BOUNDS`. All 3 pinned runs remain finite and wind-bounded through the audited steps, but each violates the corrected theta gate before reaching 1h. Per the sprint kill gate, Tier-4 RMSE and spatial-divergence gates were not honestly run or claimed.

Secondary evidence gap: the warmed D2H sister-sprint memo required by this sprint is not present in this checkout, so transfer-cleanliness inheritance also cannot be closed here. This did not determine the recommendation because the physical theta blocker fires first.

## per-run gate table

| run_id | completed before stop | finite | theta lower-30 first-bad range K | theta upper-14 first-bad range K | max \|u\|/\|v\|/\|w\| m/s at first bad | result |
|---|---:|---|---:|---:|---:|---|
| `20260509_18z_l3_24h_20260511T190519Z` | 10 steps / 100 s | yes | 272.820 .. 423.464 | 331.764 .. 501.850 | 53.738 / 23.031 / 5.778 | `BLOCKER: THETA_BOUNDS` |
| `20260521_18z_l3_24h_20260522T072630Z` | 35 steps / 350 s | yes | 289.084 .. 404.244 | 259.399 .. 492.527 | 25.658 / 24.408 / 1.016 | `BLOCKER: THETA_BOUNDS` |
| `20260523_18z_l3_24h_20260524T004313Z` | 62 steps / 620 s | yes | 272.935 .. 1343.547 | 352.427 .. 492.065 | 64.634 / 13.968 / 2.618 | `BLOCKER: THETA_BOUNDS` |

Bounds policy used:
- lower 30 levels: 200-400 K
- upper 14 levels: 250-700 K
- full-column finite required
- wind limits: `|u|`, `|v|` <= 100 m/s; `|w|` <= 50 m/s

## acceptance gates

| Gate | Status | Evidence |
|---|---|---|
| Operational mode, not validation | PASS | `proof_operational_mode_audit_v2.json`; no forbidden validation-helper/sanitizer/host-transfer tokens found in `src/gpuwrf/runtime/operational_mode.py` |
| Sanitizer off | PASS | `proof_1h_runs.json`; source audit reports `not_present_in_operational_path` |
| 3 pinned Gen2 run IDs | PASS | `proof_1h_runs.json` |
| Per-step corrected theta/wind/finite bounds | FAIL | `proof_bounds_v2.json`; all 3 runs fail theta lower-30 before 1h |
| Tier-4 RMSE T2/U10/V10 | NOT RUN | blocked before valid 1h forecast; `proof_tier4_rmse_v2.json` |
| Spatial-divergence audit | NOT RUN | blocked before valid 1h forecast; `proof_spatial_divergence_v2.json` |
| Wall-clock comparison | INFO only | no valid full 1h run after kill gate; `proof_performance_v2.json` |
| D2H warmed inheritance | MISSING | `proof_d2h_inheritance.json`; referenced `d2h_warmed_memo.md` absent in this checkout |
| Exact no-regression command | FAIL-TO-COLLECT | `proof_no_regression.txt`; `tests/test_m6b_d2h_warmed_*.py` path absent |
| Available no-regression subset | PASS | `proof_no_regression_available.txt`; 142 passed in 460.72 s |

## comparison to prior reports

- Original M6b BLOCKER failed on the first 10 s step for all three runs with extreme theta and vertical-wind growth.
- Carry-fix promoted `t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, and `_save`, and passed the 10 s probe using lower-30 theta bounds.
- RETRY improves the first-step behavior but does not survive the 1h gate: failures move to 100 s, 350 s, and 620 s, still under the corrected theta bounds.

## files changed

- `scripts/m6b_canary_1h_honest_v2.py`
- `tests/test_m6b_honest_v2_acceptance.py`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_1h_runs.json`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_1h_runs.txt`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_bounds_v2.json`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_tier4_rmse_v2.json`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_spatial_divergence_v2.json`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_performance_v2.json`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_d2h_inheritance.json`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_operational_mode_audit_v2.json`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_no_regression.txt`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_no_regression_available.txt`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/worker-report.md`

## commands run

- `python -m py_compile scripts/m6b_canary_1h_honest_v2.py tests/test_m6b_honest_v2_acceptance.py`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6b_canary_1h_honest_v2.py --runs 3 --hours 1 2>&1 | tee .agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_1h_runs.txt`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_carry_expansion_*.py tests/test_m6b_d2h_warmed_*.py tests/test_m6b_honest_v2_*.py -v 2>&1 | tee .agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_no_regression.txt` -> failed before collection because `tests/test_m6b_d2h_warmed_*.py` does not exist.
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_carry_expansion_*.py tests/test_m6b_honest_v2_*.py -v 2>&1 | tee .agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_no_regression_available.txt` -> 142 passed in 460.72 s.

## proof objects produced

- `proof_1h_runs.json`
- `proof_1h_runs.txt`
- `proof_bounds_v2.json`
- `proof_tier4_rmse_v2.json`
- `proof_spatial_divergence_v2.json`
- `proof_performance_v2.json`
- `proof_d2h_inheritance.json`
- `proof_operational_mode_audit_v2.json`
- `proof_no_regression.txt`
- `proof_no_regression_available.txt`

## unresolved risks

- The carry expansion is necessary but insufficient for the 1h operational forecast; corrected theta bounds still fail under real Gen2 d02 states.
- The exact regression command cannot pass in this checkout until the warmed D2H test/proof files are present.
- RMSE and spatial gates remain unevaluated, not passed, because the physical bounds kill gate stops the run before a valid 1h output exists.

## next decision needed

Dispatch a follow-up operational-stability fix sprint focused on theta lower-30 growth after carry expansion. Suggested first targets are the transition from promoted scratch carry to prognostic theta/mu composition and the late-step growth pattern visible at 100 s / 350 s / 620 s across the three pinned cases.
