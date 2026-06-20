# v0.11.0 live nested 24 h validation

- Status: `PASS`
- Verdict: `LIVE_NESTED_24H_FINITE_RMSE_RECORDED`
- Run: `<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z`
- Git: `3db9ec6efb6a81e041fb373e3b8c382914dddf84`
- Live nesting: one-way d01->d02->d03, parent boundary packages generated from live parent state, child subcycling enabled.
- Feedback: `off`.

## 24 h RMSE vs CPU-WRF

| domain | T2 RMSE K | U10 RMSE m/s | V10 RMSE m/s | hourly finite | missing leads |
|---|---:|---:|---:|---|---|
| d01 | 0.784767 | 2.77263 | 3.05261 | True | [] |
| d02 | 1.03028 | 3.5426 | 3.01358 | True | [] |
| d03 | 1.09939 | 3.43763 | 4.00351 | True | [] |

## Proof Boundary

- Proven: a 24 h live one-way nested d01->d02->d03 run completed, emitted paired hourly scores, and recorded finite state checks at the output cadence.
- Not proven: two-way feedback, TOST/ensemble equivalence, profiler/transfer claims, longer horizons, or the separate KI-1 d03 1 km gate.
- Replay-nest baseline was not rerun in this proof; CPU-WRF wrfout files are the paired truth source.
