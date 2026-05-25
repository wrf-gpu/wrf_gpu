# Worker Report - M6b Honest 1h Canary V3

## objective

Run the actual M6b acceptance gate on three pinned Gen2 d02 IDs using `run_forecast_operational` with sanitizer off and CPU affinity `0-3`: 1h operational forecast, per-step per-level bounds, Tier-4 RMSE envelope, spatial-divergence audit, and M6 close recommendation.

## verdict

`BLOCKER`. Do not close M6.

The operational source audit passed, the run used the three requested pinned Gen2 IDs, and no sanitizer path was present. The bounds gate failed before a valid 1h forecast existed, so Tier-4 RMSE and spatial-divergence claims were not run.

## bounds table

| run ID | status | first bad step | lead | blocker | lower-30 theta K | upper-14 theta K | max \|u\| | max \|v\| | max \|w\| |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| `20260521_18z_l3_24h_20260522T072630Z` | FAIL | 46 | 460 s | WIND_BOUNDS | 288.808 to 350.940 | 351.766 to 492.527 | 25.658 | 103.720 | 0.981 |
| `20260521_18z_l3_24h_20260522T133443Z` | FAIL | 46 | 460 s | WIND_BOUNDS | 288.808 to 350.940 | 351.766 to 492.527 | 25.658 | 103.720 | 0.981 |
| `20260509_18z_l3_24h_20260511T190519Z` | FAIL | 11 | 110 s | THETA_BOUNDS | 290.336 to 2.604313608192e12 | 354.467 to 2.9304287232e10 | 53.738 | 28.895 | 19.240 |

Policy checked every 10 s step until fail-fast: lower 30 levels 200-400 K, upper 14 levels 250-700 K, all leaves finite, `|u|/|v| <= 100 m/s`, `|w| <= 50 m/s`.

## Tier-4 RMSE and spatial audit

Not run. Reason: `blocked_before_valid_1h_rmse:WIND_BOUNDS` and `blocked_before_spatial_audit:WIND_BOUNDS`.

It would be misleading to compute T2/U10/V10 RMSE or interior/boundary divergence from an invalid forecast state.

## wall-clock

No valid full-hour wall-clock exists. The only recorded wall time is from the fail-fast stepwise bounds audit:

| run ID | steps checked | stepwise audit wall time |
| --- | ---: | ---: |
| `20260521_18z_l3_24h_20260522T072630Z` | 46 | 149.311 s |
| `20260521_18z_l3_24h_20260522T133443Z` | 46 | 125.286 s |
| `20260509_18z_l3_24h_20260511T190519Z` | 11 | 149.473 s |

The recovered CPU WRF denominator remains informational only and was not used for an M6 close claim.

## files changed

- `scripts/m6b_canary_1h_honest_v3.py`
- `tests/test_m6b_honest_v3_acceptance.py`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/proof_1h_runs.json`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/proof_bounds.json`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/proof_tier4_rmse.json`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/proof_spatial_divergence.json`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/proof_performance.json`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/proof_operational_mode_audit.json`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/proof_no_regression.txt`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/worker-report.md`

Local-only note: `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/proof_warmed.nsys-rep` was copied from `/tmp/wrf_gpu2_standcomp` so the existing ignored-artifact regression test could run locally. It is ignored by git and is not part of this commit.

## commands run

- `python -m py_compile scripts/m6b_canary_1h_honest_v3.py tests/test_m6b_honest_v3_acceptance.py`
- `taskset -c 0-3 python scripts/m6b_canary_1h_honest_v3.py --runs 3 --hours 1`
- `pytest tests/test_m6b_honest_v3_acceptance.py -v`
- `pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_*.py -v`

Regression result after restoring the local ignored Nsight artifact: `176 passed in 422.15s`.

## proof objects produced

- `proof_1h_runs.json`
- `proof_bounds.json`
- `proof_tier4_rmse.json`
- `proof_spatial_divergence.json`
- `proof_performance.json`
- `proof_operational_mode_audit.json`
- `proof_no_regression.txt`

## unresolved risks

- The V3 acceptance gate failed on physical bounds before 1h; no Tier-4 RMSE envelope or spatial-divergence evidence exists.
- The two 20260521 pinned IDs fail identically at step 46 via `|v| = 103.720 m/s`; this needs localization because it may indicate deterministic state reuse, shared forcing behavior, or a common operational-core instability.
- The 20260509 pinned ID fails much earlier at step 11 with explosive theta growth while remaining finite, indicating the 10 s standalone PASS did not generalize to the 1h acceptance path.
- The requested `.agent/sprints/2026-05-25-m6b-reframe-shared-core/worker-report.md` is absent in this checkout; I used the reframe sprint proofs plus the reframe critic report documenting Amendment #1 supersession.

## next decision needed

Dispatch a named-cause blocker sprint before any M6 close attempt. Recommended start: localize the 20260509 step-11 theta explosion and the 20260521 step-46 `v` wind breach against the shared-core/comparator path.
