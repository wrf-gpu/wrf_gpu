# V0.14 Wrfout Grid Comparison Smoke

Generated UTC: `2026-06-12T05:35:08.777567+00:00`

## Verdict

- verdict: `REPORT_ONLY_NO_TOLERANCE_MANIFEST`
- tolerance manifest: `False`
- CPU-only: `True`; GPU used: `False`
- next: Coverage is the current blocker; resolve missing/incompatible fields before interpreting physics parity.

## Coverage

- domain `d01`; paired files `1`; leads `1`-`1` h
- CPU files `73`; GPU files `1`; unmatched CPU/GPU `72`/`0`
- variables CPU/GPU/common `362`/`107`/`103`
- compared numeric `2`; dynamic `0`; static/time-invariant `2`; time metadata `0`

## Top 10 Field Differences

| Field | Class | RMSE | Bias | p99 abs | Max abs | Worst lead |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `DZS` | static | 0 | 0 | 0 | 0 | 1 |
| `ZS` | static | 0 | 0 | 0 | 0 | 1 |

## Top 5 Drift Signals

| Field | RMSE slope/h | Bias slope/h | Sign consistency | Worst lead RMSE |
| --- | ---: | ---: | ---: | ---: |

## Top 5 Coverage Issues

- `gpu_only` count `4`: `QNCCN`, `QNCLOUD`, `QNGRAUPEL`, `QNSNOW`
- `cpu_only` count `259`: `ACCANHS`, `ACDEWC`, `ACDRIPR`, `ACDRIPS`, `ACECAN`, `ACEDIR`, `ACEFLXB`, `ACETLSM`

## Next Debug Recommendation

Coverage is the current blocker; resolve missing/incompatible fields before interpreting physics parity.
