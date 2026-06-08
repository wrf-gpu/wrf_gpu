# V0.14 Wrfout Grid Comparison Smoke

Generated UTC: `2026-06-08T22:57:01.308911+00:00`

## Verdict

- verdict: `REPORT_ONLY_NO_TOLERANCE_MANIFEST`
- tolerance manifest: `False`
- CPU-only: `True`; GPU used: `False`
- next: Fix or explicitly root-cause static/grid/base-state mismatches first, then rerun this comparator before dycore, radiation, FP32, TOST, or Switzerland equivalence work.

## Coverage

- domain `d02`; paired files `24`; leads `1`-`24` h
- CPU files `73`; GPU files `24`; unmatched CPU/GPU `49`/`0`
- variables CPU/GPU/common `375`/`104`/`100`
- compared numeric `99`; dynamic `37`; static/time-invariant `61`; time metadata `2`

## Top 10 Field Differences

| Field | Class | RMSE | Bias | p99 abs | Max abs | Worst lead |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `C2F` | static | 6.990e+04 | -3.071e+04 | 9.500e+04 | 9.500e+04 | 1 |
| `C2H` | static | 7.006e+04 | -3.045e+04 | 9.500e+04 | 9.500e+04 | 1 |
| `C4F` | static | 1.394e+04 | -1.037e+04 | 2.678e+04 | 2.678e+04 | 1 |
| `C4H` | static | 1.408e+04 | -1.061e+04 | 2.674e+04 | 2.674e+04 | 1 |
| `QNRAIN` | dynamic | 25.011 | 0.283 | 0 | 6995.268 | 24 |
| `PHB` | static | 45.353 | -0.641 | 73.584 | 2237.942 | 24 |
| `PSFC` | dynamic | 525.288 | -504.513 | 756.195 | 1892.891 | 7 |
| `MU` | dynamic | 273.821 | -242.799 | 486.872 | 1500.875 | 7 |
| `P` | dynamic | 228.122 | -147.639 | 633.364 | 1286.163 | 7 |
| `MUB` | static | 58.769 | 3.190 | 268.094 | 1115.211 | 11 |

## Top 5 Drift Signals

| Field | RMSE slope/h | Bias slope/h | Sign consistency | Worst lead RMSE |
| --- | ---: | ---: | ---: | ---: |
| `PBLH` | 6.231 | -14.164 | 0.625 | 235.039 |
| `SWDNB` | 10.605 | -4.251 | 0.643 | 224.091 |
| `SWDOWN` | 10.605 | -4.251 | 0.643 | 224.091 |
| `SWNORM` | 10.603 | -4.243 | 0.643 | 224.043 |
| `PH` | 9.056 | -4.625 | 1.000 | 405.704 |

## Top 5 Coverage Issues

- `gpu_only` count `4`: `QNCCN`, `QNCLOUD`, `QNGRAUPEL`, `QNSNOW`
- `cpu_only` count `275`: `ACCANHS`, `ACDEWC`, `ACDRIPR`, `ACDRIPS`, `ACECAN`, `ACEDIR`, `ACEFLXB`, `ACETLSM`

## Next Debug Recommendation

Fix or explicitly root-cause static/grid/base-state mismatches first, then rerun this comparator before dycore, radiation, FP32, TOST, or Switzerland equivalence work.
