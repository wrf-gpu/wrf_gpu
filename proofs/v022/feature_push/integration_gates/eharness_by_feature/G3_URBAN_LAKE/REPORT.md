# v0.22 Operational-Relaxed Small-Grid Gate

- gate_mode: `operational-relaxed`
- lever_id: `G3_URBAN_LAKE`
- case_id: `CANARY-L2-D02-SMALLEST-TILE-SYNTHETIC`
- verdict: `PASS`

| domain | lead h | hard guards | validation band | tier-o band | verdict |
|---|---:|---|---|---|---|
| d02 | 24 | PASS | PASS | PASS | TIER_O_ACCEPTED |
| d02 | 72 | PASS | PASS | PASS | TIER_O_ACCEPTED |
| d02 | 120 | PASS | PASS | PASS | TIER_O_ACCEPTED |

## Notes

- The metric path is CPU-only and reads existing wrfout directories.
- AEMET/obs scoring is recorded as unavailable in this first wiring pass.
- Full on-device conservation and clamp/transfer audits should be supplied by the run via `--candidate-guards`.
