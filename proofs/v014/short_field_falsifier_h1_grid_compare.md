# V0.14 Wrfout Grid Comparison Smoke

Generated UTC: `2026-06-10T12:31:32.053903+00:00`

## Verdict

- verdict: `REPORT_ONLY_NO_TOLERANCE_MANIFEST`
- tolerance manifest: `False`
- CPU-only: `True`; GPU used: `False`
- next: Fix or explicitly root-cause static/grid/base-state mismatches first, then rerun this comparator before dycore, radiation, FP32, TOST, or Switzerland equivalence work.

## Coverage

- domain `d02`; paired files `1`; leads `1`-`1` h
- CPU files `73`; GPU files `1`; unmatched CPU/GPU `72`/`0`
- variables CPU/GPU/common `375`/`104`/`100`
- compared numeric `99`; dynamic `47`; static/time-invariant `51`; time metadata `2`

## Top 10 Field Differences

| Field | Class | RMSE | Bias | p99 abs | Max abs | Worst lead |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `PBLH` | dynamic | 78.950 | 23.978 | 262.671 | 497.104 | 1 |
| `PSFC` | dynamic | 323.115 | -313.780 | 414.673 | 425.680 | 1 |
| `P` | dynamic | 129.754 | -85.662 | 361.913 | 408.390 | 1 |
| `HFX` | dynamic | 38.186 | -5.867 | 250.965 | 377.693 | 1 |
| `LH` | dynamic | 53.896 | 11.997 | 144.995 | 323.310 | 1 |
| `MUB` | static | 9.276 | -0.119 | 18.194 | 250.664 | 1 |
| `PB` | static | 4.521 | -0.036 | 0.105 | 249.875 | 1 |
| `PH` | dynamic | 85.157 | -53.332 | 183.379 | 221.592 | 1 |
| `MU` | dynamic | 121.961 | -85.142 | 204.011 | 213.099 | 1 |
| `SWNORM` | dynamic | 57.363 | -55.561 | 110.865 | 203.471 | 1 |

## Top 5 Drift Signals

| Field | RMSE slope/h | Bias slope/h | Sign consistency | Worst lead RMSE |
| --- | ---: | ---: | ---: | ---: |
| `CLDFRA` | NA | NA | 1.000 | 0.002 |
| `COSZEN` | NA | NA | 1.000 | 0.055 |
| `GLW` | NA | NA | 1.000 | 8.486 |
| `GRAUPELNC` | NA | NA | NA | 0 |
| `HFX` | NA | NA | 1.000 | 38.186 |

## Top 5 Coverage Issues

- `gpu_only` count `4`: `QNCCN`, `QNCLOUD`, `QNGRAUPEL`, `QNSNOW`
- `cpu_only` count `275`: `ACCANHS`, `ACDEWC`, `ACDRIPR`, `ACDRIPS`, `ACECAN`, `ACEDIR`, `ACEFLXB`, `ACETLSM`

## Next Debug Recommendation

Fix or explicitly root-cause static/grid/base-state mismatches first, then rerun this comparator before dycore, radiation, FP32, TOST, or Switzerland equivalence work.
