# V0.14 Wrfout Grid Comparison Smoke

Generated UTC: `2026-06-13T10:45:07.922326+00:00`

## Verdict

- verdict: `FAIL`
- tolerance manifest: `True`
- CPU-only: `True`; GPU used: `False`
- next: With static fields quiet, localize first dynamic divergence in QNRAIN, QNICE, PBLH, SWNORM, SWDNB using same-state/tendency probes.

## Coverage

- domain `d01`; paired files `72`; leads `1`-`72` h
- CPU files `73`; GPU files `72`; unmatched CPU/GPU `1`/`0`
- variables CPU/GPU/common `362`/`107`/`103`
- compared numeric `102`; dynamic `45`; static/time-invariant `56`; time metadata `2`

## Top 10 Field Differences

| Field | Class | RMSE | Bias | p99 abs | Max abs | Worst lead |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `QNRAIN` | dynamic | 2.130e+06 | 2.402e+05 | 9.141e+06 | 8.860e+07 | 20 |
| `QNICE` | dynamic | 6.033e+04 | 5564.472 | 2.794e+05 | 2.081e+06 | 4 |
| `PBLH` | dynamic | 212.465 | -91.847 | 698.420 | 1451.925 | 38 |
| `SWNORM` | dynamic | 169.945 | 91.047 | 422.672 | 473.006 | 60 |
| `SWDNB` | dynamic | 86.153 | 35.288 | 326.560 | 407.714 | 11 |
| `SWDOWN` | dynamic | 86.153 | 35.288 | 326.560 | 407.714 | 11 |
| `PH` | dynamic | 25.220 | 4.156 | 83.762 | 403.668 | 10 |
| `HFX` | dynamic | 27.689 | 3.361 | 94.149 | 379.371 | 60 |
| `SWUPT` | dynamic | 83.648 | -39.291 | 276.251 | 346.312 | 11 |
| `SWUPB` | dynamic | 30.696 | -3.890 | 160.334 | 279.419 | 59 |

## Top 5 Drift Signals

| Field | RMSE slope/h | Bias slope/h | Sign consistency | Worst lead RMSE |
| --- | ---: | ---: | ---: | ---: |
| `QNRAIN` | -1570.805 | -1387.994 | 0.931 | 4.266e+06 |
| `QNICE` | -970.449 | 117.173 | 0.681 | 1.111e+05 |
| `PBLH` | 1.378 | -0.105 | 0.944 | 330.448 |
| `SWDNB` | -0.495 | -0.527 | 0.933 | 256.476 |
| `SWDOWN` | -0.495 | -0.527 | 0.933 | 256.476 |

## Top 5 Coverage Issues

- `gpu_only` count `4`: `QNCCN`, `QNCLOUD`, `QNGRAUPEL`, `QNSNOW`
- `cpu_only` count `259`: `ACCANHS`, `ACDEWC`, `ACDRIPR`, `ACDRIPS`, `ACECAN`, `ACEDIR`, `ACEFLXB`, `ACETLSM`

## Next Debug Recommendation

With static fields quiet, localize first dynamic divergence in QNRAIN, QNICE, PBLH, SWNORM, SWDNB using same-state/tendency probes.
