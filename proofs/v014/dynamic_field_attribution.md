# V0.14 Dynamic Field Attribution

Generated UTC: `2026-06-08T23:12:27.518908+00:00`

CPU-only wrfout attribution for retained Case 3. This is not an equivalence claim and does not run the model.

## Verdict

- first materially bad lead: `h1` under report-only thresholds
- selected same-state localization lead: `h10` (`2026-05-02T04:00:00+00:00`)
- worst h10-h14 primary lead: `h10` score `2.565`
- selected cells: `24` mass-grid cells with U/V/W/PH native-stagger context
- next target: CPU-WRF term savepoints for selected lead/cells, then JAX same-state term comparison

## Top Dynamic Fields

| Field | score | overall RMSE | bias | worst lead | worst RMSE | first bad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `PSFC` | 4.732 | 525.288 | -504.513 | 7 | 709.850 | 1 |
| `V` | 3.835 | 5.830 | 3.703 | 13 | 7.670 | 2 |
| `P` | 3.234 | 228.122 | -147.639 | 7 | 323.369 | 2 |
| `T` | 3.111 | 2.268 | 0.758 | 19 | 3.111 | 7 |
| `U` | 3.098 | 4.612 | 2.719 | 12 | 6.196 | 4 |
| `V10` | 2.858 | 2.524 | 1.036 | 11 | 4.287 | 3 |
| `MU` | 2.845 | 273.821 | -242.799 | 7 | 426.783 | 5 |
| `PH` | 2.705 | 336.208 | -244.367 | 18 | 405.704 | 2 |

## Selected Cells

Lead `h10`; full selected-cell details and staggered face indices are in JSON.

| Rank | y | x | lat | lon | score | hits | top components |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 9 | 13 | 27.6951 | -18.1388 | 50.618 | 11 | `dW_k0, dV10, dV_k0, dP_k0` |
| 2 | 25 | 39 | 28.1348 | -17.3494 | 50.089 | 9 | `dW_k0, dP_k0, dV10, dU10` |
| 3 | 41 | 14 | 28.5596 | -18.1219 | 48.524 | 8 | `dV10, dV_k0, dQVAPOR_k0, dP_k0` |
| 4 | 49 | 17 | 28.7767 | -18.0328 | 47.201 | 10 | `dV10, dP_k0, dV_k0, dQVAPOR_k0` |
| 5 | 32 | 53 | 28.3261 | -16.9216 | 47.178 | 10 | `dW_k0, dV10, dP_k0, dQVAPOR_k0` |
| 6 | 27 | 143 | 28.1742 | -14.1633 | 46.763 | 11 | `dPBLH, dP_k0, dV10, dPSFC` |
| 7 | 44 | 15 | 28.6410 | -18.0923 | 45.716 | 8 | `dV10, dV_k0, dP_k0, dQVAPOR_k0` |
| 8 | 38 | 14 | 28.4786 | -18.1206 | 44.053 | 8 | `dV10, dP_k0, dQVAPOR_k0, dV_k0` |
| 9 | 39 | 11 | 28.5044 | -18.2132 | 42.651 | 8 | `dV10, dP_k0, dQVAPOR_k0, dV_k0` |
| 10 | 36 | 73 | 28.4350 | -16.3079 | 42.527 | 11 | `dW_k0, dP_k0, dU10, dU_k0` |
| 11 | 22 | 37 | 28.0533 | -17.4099 | 41.537 | 9 | `dV10, dP_k0, dV_k0, dPSFC` |
| 12 | 30 | 50 | 28.2717 | -17.0133 | 39.741 | 8 | `dV10, dP_k0, dV_k0, dPSFC` |

## Correlation Snapshot

| Pair | selected-lead r |
| --- | ---: |
| `dU10__dU_k0` | 0.996 |
| `dV10__dV_k0` | 0.996 |
| `dPSFC__dMU` | 0.631 |
| `dPSFC__dP_k0` | 0.716 |
| `dPSFC__dPH_k0` | -0.075 |
| `dV10__dP_k0` | -0.001 |
| `dV10__dPH_k0` | 0.028 |

## Region Signal

- `W` in `elevation/land_0_300m`: RMSE `0.616`, severity `12.322`
- `W` in `land_ocean/land`: RMSE `0.428`, severity `8.554`
- `W` in `boundary/frame_5cells`: RMSE `0.270`, severity `5.395`
- `W` in `quadrant/NE`: RMSE `0.245`, severity `4.894`
- `PSFC` in `elevation/land_0_300m`: RMSE `560.642`, severity `3.738`
- `PSFC` in `quadrant/NW`: RMSE `548.062`, severity `3.654`

## Vertical Targets

- `U`: k25 RMSE 8.589, k24 RMSE 8.537, k26 RMSE 8.332
- `V`: k31 RMSE 9.942, k30 RMSE 9.595, k32 RMSE 9.426
- `P`: k0 RMSE 506.738, k1 RMSE 497.772, k2 RMSE 486.402
- `PH`: k29 RMSE 553.615, k28 RMSE 551.894, k30 RMSE 549.651
- `W`: k17 RMSE 0.241, k18 RMSE 0.239, k16 RMSE 0.238

## Limits

- Static/time-invariant writer and base-state fields are excluded from dynamic ranking.
- Wrfout-only evidence cannot identify the first failing tendency term.
- Detailed per-lead, per-level, per-region, colocation, and cell tables are in JSON.
