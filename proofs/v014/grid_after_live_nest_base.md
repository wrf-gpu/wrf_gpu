# V0.14 Grid After Live-Nest Base Fix

Generated UTC: `2026-06-09T09:28:37.727541+00:00`

## Verdict

- verdict: `GRID_SYMPTOM_NOT_CLOSED`
- GPU run: `L2_D02_GREEN` on `cuda:0`, total wall `1192.299` s, forecast-only `1186.443` s
- VRAM: not recorded by wall_clock_l2_d02.json
- output: `<DATA_ROOT>/wrf_gpu2/v014_grid_after_live_nest_base/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
- log: `proofs/v014/grid_after_live_nest_base/gpu_h12/gpu_h12_run.log`

The live-nest base source fix materially improves base/static payloads, but it does not close the grid-cell dynamic symptom.
Not closed: h1-h12 V10 RMSE remains 2.55 m/s, worst h11 is 4.28 m/s, and PSFC/P/MU/PH retain large dynamic RMSE with h7-h8 worst pressure/mass/geopotential leads.

## Required Core Fields

| Field | RMSE | Bias | MAE | p95 abs | p99 abs | Max abs | Worst lead | Worst lead RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V10` | 2.550 | 0.585 | 1.881 | 5.510 | 8.497 | 16.182 | 11 | 4.277 |
| `U10` | 1.711 | -0.465 | 1.324 | 3.232 | 4.901 | 11.683 | 6 | 2.469 |
| `PSFC` | 517.191 | -486.582 | 486.582 | 723.091 | 777.011 | 929.820 | 7 | 710.255 |
| `P` | 230.307 | -157.827 | 158.054 | 559.263 | 664.574 | 832.872 | 7 | 323.145 |
| `MU` | 266.525 | -214.346 | 227.241 | 446.782 | 505.880 | 637.073 | 7 | 424.640 |
| `PH` | 292.387 | -226.184 | 230.075 | 564.272 | 633.044 | 751.412 | 8 | 384.817 |
| `T` | 1.169 | 0.120 | 0.758 | 2.363 | 4.431 | 9.071 | 12 | 2.444 |

## Before/After: Post-Static h1 Smoke

Scope: old `post_static_writer_grid_compare` h1 vs this fresh h12 run's h1.

| Field | Old RMSE | New RMSE | RMSE change | Old max abs | New max abs | Max change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `V10` | 1.288 | 1.288 | -0.0% | 4.820 | 3.017 | -37.4% |
| `U10` | 0.454 | 0.336 | -26.0% | 9.378 | 2.608 | -72.2% |
| `PSFC` | 175.956 | 170.900 | -2.9% | 1310.617 | 328.859 | -74.9% |
| `P` | 60.596 | 60.891 | 0.5% | 669.304 | 254.558 | -62.0% |
| `MU` | 96.176 | 90.755 | -5.6% | 922.508 | 213.873 | -76.8% |
| `PH` | 104.678 | 106.499 | 1.7% | 249.806 | 223.600 | -10.5% |
| `T` | 0.225 | 0.218 | -3.3% | 4.650 | 2.123 | -54.3% |
| `PB` | 28.642 | 4.521 | -84.2% | 1111.711 | 249.883 | -77.5% |
| `PHB` | 45.353 | 0.025 | -99.9% | 2237.942 | 0.109 | -100.0% |
| `MUB` | 58.769 | 9.276 | -84.2% | 1115.211 | 250.664 | -77.5% |
| `HGT` | 8.463 | 1.233e-06 | -100.0% | 228.129 | 3.052e-05 | -100.0% |
| `XLAT` | 0.010 | 0 | -100.0% | 0.027 | 0 | -100.0% |
| `XLONG` | 0.007 | 0 | -100.0% | 0.027 | 0 | -100.0% |

## Before/After: V10 Diagnostics h1-h12

Scope: old stored `v10_grid_diagnostics` lead rows h1-h12 vs this fresh h1-h12 comparator. T2 is included here because that older artifact did not track perturbation `T` in the same summary.

| Field | Old RMSE | New RMSE | RMSE change | Old max abs | New max abs | Max change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `V10` | 2.545 | 2.550 | 0.2% | 16.034 | 16.182 | 0.9% |
| `U10` | 1.732 | 1.711 | -1.2% | 11.969 | 11.683 | -2.4% |
| `PSFC` | 517.616 | 517.191 | -0.1% | 1892.891 | 929.820 | -50.9% |
| `T2` | 1.220 | 1.234 | 1.1% | 10.373 | 10.354 | -0.2% |

## Before/After: Grid-Cell Envelope

Scope differs: old `grid_cell_envelope` is h1-h24, this proof is h1-h12. Use this as directional context, not a strict identical-window metric.

| Field | Old RMSE | New RMSE | RMSE change | Old max abs | New max abs | Max change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `V10` | 2.524 | 2.550 | 1.1% | 16.034 | 16.182 | 0.9% |
| `U10` | 2.068 | 1.711 | -17.3% | 11.969 | 11.683 | -2.4% |
| `PSFC` | 525.288 | 517.191 | -1.5% | 1892.891 | 929.820 | -50.9% |
| `P` | 228.122 | 230.307 | 1.0% | 1286.163 | 832.872 | -35.2% |
| `MU` | 273.821 | 266.525 | -2.7% | 1500.875 | 637.073 | -57.6% |
| `PH` | 336.208 | 292.387 | -13.0% | 926.573 | 751.412 | -18.9% |
| `T` | 2.268 | 1.169 | -48.5% | 10.471 | 9.071 | -13.4% |
| `PB` | 28.642 | 4.521 | -84.2% | 1111.719 | 249.883 | -77.5% |
| `PHB` | 45.353 | 0.025 | -99.9% | 2237.942 | 0.109 | -100.0% |
| `MUB` | 58.769 | 9.276 | -84.2% | 1115.211 | 250.672 | -77.5% |

## Static/Base Split

- old grid-cell-envelope static mismatch count: `31`
- new static exact/nonzero counts: `47` / `4`
- base-source fix classification: `LIVE_NEST_BASE_SOURCE_PARTIAL_NO_GRID_SYMPTOM_PROOF`
- interpretation: C/DN/RDN/MAPFAC, XLAT/XLONG, and HGT are exact or near-exact in this fresh artifact; PB/MUB/PHB improve strongly versus the h1 post-static smoke and grid-cell envelope but PB/MUB are not exact.

| Field | RMSE | Bias | p99 abs | Max abs |
| --- | ---: | ---: | ---: | ---: |
| `C1H` | 0 | 0 | 0 | 0 |
| `C2H` | 0 | 0 | 0 | 0 |
| `C4H` | 0 | 0 | 0 | 0 |
| `DN` | 0 | 0 | 0 | 0 |
| `RDN` | 0 | 0 | 0 | 0 |
| `MAPFAC_M` | 0 | 0 | 0 | 0 |
| `PB` | 4.521 | -0.036 | 0.105 | 249.883 |
| `PHB` | 0.025 | -0.017 | 0.062 | 0.109 |
| `MUB` | 9.276 | -0.120 | 18.480 | 250.672 |
| `HGT` | 1.233e-06 | -5.429e-08 | 3.815e-06 | 3.052e-05 |
| `XLAT` | 0 | 0 | 0 | 0 |
| `XLONG` | 0 | 0 | 0 | 0 |

## Acceptance

- GPU forecast exited 0: `true`
- Comparator exited 0: `true`
- JSON validates: `true`
- TOST resumed: `false`
- Production `src/` edits: `false`
- V10/grid closure claimed: `false`

## Next Target

Use the existing h10-h12 dynamic window for CPU-WRF same-state term savepoints: pressure-gradient/mass-wind coupling around PSFC, MU, P, PH, U/V, and V10.
