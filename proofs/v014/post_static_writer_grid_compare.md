# V0.14 Wrfout Grid Comparison Smoke

Generated UTC: `2026-06-08T23:05:12.856910+00:00`

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
| `PHB` | static | 45.353 | -0.641 | 73.584 | 2237.942 | 1 |
| `PSFC` | dynamic | 175.956 | -159.307 | 275.516 | 1310.617 | 1 |
| `MUB` | static | 58.769 | 3.190 | 267.483 | 1115.211 | 1 |
| `PB` | static | 28.642 | 0.960 | 40.471 | 1111.711 | 1 |
| `MU` | dynamic | 96.176 | 64.985 | 205.173 | 922.508 | 1 |
| `P` | dynamic | 60.596 | -39.922 | 196.890 | 669.304 | 1 |
| `HFX` | dynamic | 43.318 | 10.536 | 278.145 | 497.027 | 1 |
| `PBLH` | dynamic | 38.343 | 18.544 | 151.079 | 278.046 | 1 |
| `PH` | dynamic | 104.678 | -87.770 | 186.166 | 249.806 | 1 |
| `HGT` | static | 8.463 | -0.182 | 35.137 | 228.129 | 1 |

## Top 5 Drift Signals

| Field | RMSE slope/h | Bias slope/h | Sign consistency | Worst lead RMSE |
| --- | ---: | ---: | ---: | ---: |
| `CLDFRA` | NA | NA | 1.000 | 0.002 |
| `COSZEN` | NA | NA | 1.000 | 0.055 |
| `GLW` | NA | NA | 1.000 | 1.292 |
| `GRAUPELNC` | NA | NA | NA | 0 |
| `HFX` | NA | NA | 1.000 | 43.318 |

## Top 5 Coverage Issues

- `gpu_only` count `4`: `QNCCN`, `QNCLOUD`, `QNGRAUPEL`, `QNSNOW`
- `cpu_only` count `275`: `ACCANHS`, `ACDEWC`, `ACDRIPR`, `ACDRIPS`, `ACECAN`, `ACEDIR`, `ACEFLXB`, `ACETLSM`

## Next Debug Recommendation

Fix or explicitly root-cause static/grid/base-state mismatches first, then rerun this comparator before dycore, radiation, FP32, TOST, or Switzerland equivalence work.
