# v0.22 Operational-Relaxed Small-Grid Gate

- gate_mode: `cpu-wrf-backlog`
- lever_id: `v022_eval_real_wrfout_selfcheck`
- case_id: `CANARY-L2-D02-REAL-WRFOUT-SELFCHECK`
- verdict: `PASS`

| domain | lead h | hard guards | validation band | tier-o band | verdict |
|---|---:|---|---|---|---|
| d02 | 24 | FAIL/UNAVAILABLE | PASS | FAIL/UNAVAILABLE | TIER_O_REJECTED |
| d02 | 72 | FAIL/UNAVAILABLE | PASS | FAIL/UNAVAILABLE | TIER_O_REJECTED |

## Notes

- The metric path is CPU-only and reads existing wrfout directories.
- AEMET/obs scoring is recorded as unavailable in this first wiring pass.
- Full on-device conservation and clamp/transfer audits should be supplied by the run via `--candidate-guards`.
