# v0.11.0 RRTMG finite recheck

- status: DIAGNOSED_KNOWN_QKE_EDGE
- verdict: KEEP_RRTMG_FEATURE_DO_NOT_MASK_QKE_KI2
- all-state finite with RRTMG on: False
- theta/u/v finite with RRTMG on: True
- RRTMG-on nonfinite signature: {'qke': 2024}
- topo-off same signature: True
- RRTMG-off same signature: True

## Interpretation

The proper segmented cadence does not reproduce the RRTMG lane's cold one-step theta/u/v nonfinite: theta/u/v are finite with zero nonfinite values. The only nonfinite field is qke (2024 cells), and the exact qke signature persists with topo_shading=0/slope_rad=0 and with RRTMG suppressed entirely. This matches the documented KI-2 20260521 d02 production-path qke edge, so the RRTMG slope/shading feature is not the offending term. No masking was applied.

## Mode Results

| mode | topo | slope | cadence | pipeline | nonfinite signature |
|---|---:|---:|---:|---|---|
| on | 1 | 1 | 180 | PIPELINE_BLOCKED | {'qke': 2024} |
| topo_off | 0 | 0 | 180 | PIPELINE_BLOCKED | {'qke': 2024} |
| rrtmg_off | 0 | 0 | 10000000 | PIPELINE_BLOCKED | {'qke': 2024} |

## Commands

- `proofs/v0110/rrtmg_finite_recheck.py --mode on --hours 1 --segment-steps 180 --radiation-cadence-steps 180 --diagnostic-exit-zero`
- `proofs/v0110/rrtmg_finite_recheck.py --mode topo_off --hours 1 --segment-steps 180 --radiation-cadence-steps 180 --out-json proofs/v0110/rrtmg_finite_recheck_topo_off.json --out-md /tmp/v0110_rrtmg_finite_recheck_topo_off.md --diagnostic-exit-zero`
- `proofs/v0110/rrtmg_finite_recheck.py --mode rrtmg_off --hours 1 --segment-steps 180 --radiation-cadence-steps 180 --out-json proofs/v0110/rrtmg_finite_recheck_rrtmg_off.json --out-md /tmp/v0110_rrtmg_finite_recheck_rrtmg_off.md --diagnostic-exit-zero`
- `python proofs/v0110/rrtmg_finite_recheck_aggregate.py`
