# V0.14 Wrfout Grid Comparison Smoke

Generated UTC: `2026-06-13T13:04:27.134094+00:00`

## Verdict

- verdict: `FAIL`
- tolerance manifest: `True`
- CPU-only: `True`; GPU used: `False`
- next: With static fields quiet, localize first dynamic divergence in QNICE, QNRAIN, LH, HFX, SWDNB using same-state/tendency probes.

## Coverage

- domain `d02`; paired files `72`; leads `1`-`72` h
- CPU files `73`; GPU files `72`; unmatched CPU/GPU `1`/`0`
- variables CPU/GPU/common `375`/`107`/`103`
- compared numeric `102`; dynamic `41`; static/time-invariant `60`; time metadata `2`

## Top 10 Field Differences

| Field | Class | RMSE | Bias | p99 abs | Max abs | Worst lead |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `QNICE` | dynamic | 540.042 | -5.348 | 0 | 5.298e+05 | 31 |
| `QNRAIN` | dynamic | 8.419 | -0.088 | 0 | 3122.397 | 11 |
| `LH` | dynamic | 85.074 | -15.288 | 467.931 | 1785.424 | 68 |
| `HFX` | dynamic | 51.817 | 12.607 | 287.932 | 996.897 | 20 |
| `SWDNB` | dynamic | 41.447 | 2.771 | 70.214 | 951.435 | 21 |
| `SWNORM` | dynamic | 42.316 | 2.890 | 95.883 | 951.435 | 21 |
| `SWDOWN` | dynamic | 27.135 | 2.342 | 31.281 | 947.279 | 21 |
| `PBLH` | dynamic | 105.522 | -25.976 | 324.526 | 818.368 | 12 |
| `SWUPT` | dynamic | 18.999 | -1.725 | 29.219 | 600.697 | 21 |
| `PH` | dynamic | 58.685 | -1.527 | 225.733 | 463.467 | 60 |

## Top 5 Drift Signals

| Field | RMSE slope/h | Bias slope/h | Sign consistency | Worst lead RMSE |
| --- | ---: | ---: | ---: | ---: |
| `PBLH` | -0.516 | 1.788 | 0.681 | 181.730 |
| `QNICE` | -1.650 | 0.071 | 1.000 | 3386.093 |
| `MU` | 0.463 | 0.719 | 0.875 | 123.983 |
| `LH` | 0.856 | -0.011 | 0.792 | 184.203 |
| `PH` | 0.336 | 0.316 | 0.514 | 124.427 |

## Top 5 Coverage Issues

- `gpu_only` count `4`: `QNCCN`, `QNCLOUD`, `QNGRAUPEL`, `QNSNOW`
- `cpu_only` count `272`: `ACCANHS`, `ACDEWC`, `ACDRIPR`, `ACDRIPS`, `ACECAN`, `ACEDIR`, `ACEFLXB`, `ACETLSM`

## Next Debug Recommendation

With static fields quiet, localize first dynamic divergence in QNICE, QNRAIN, LH, HFX, SWDNB using same-state/tendency probes.
