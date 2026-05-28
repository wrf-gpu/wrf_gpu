# M9.C Comparator Audit

Verdict: `M9C_COMPLETE`.

Headline: the comparator now handles GPU `T` as either absolute theta or WRF-style perturbation theta. On the audited Canary 20260521 GPU wrfout files, all 24 GPU `T` fields were detected as perturbation-from-300 K, matching WRF wrfout. Therefore the theta convention fix produced `0.0%` reduction in current apparent theta divergence: theta RMSE remains `77.3589529727313 K`.

## Sources Audited

- `scripts/operational_trace_compare.py`
- `scripts/m6b6_coupled_step_compare_1000.py`
- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/io/wrfout_writer.py`
- WRF registry: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/Registry/Registry.EM_COMMON`
- WRF reference file: `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z/wrfout_d02_2026-05-21_19:00:00`
- GPU file: `/tmp/m9a_gpu_20260521/wrfout_d02_2026-05-21_19:00:00`

## Confirmed Fix

`scripts/operational_trace_compare.py` now canonicalizes theta side-specifically:

- WRF side: `theta = T + 300.0`
- GPU side: `theta = T` if raw GPU `T` is detected as absolute theta, else `theta = T + 300.0`

Implemented function: `_theta_to_absolute(values, side=...)`.

## Convention Matrix

| Field | GPU side | WRF side | Verdict | Fix |
|---|---|---|---|---|
| theta | `State.theta` is absolute K in `src/gpuwrf/contracts/state.py`; `src/gpuwrf/io/wrfout_writer.py` writes NetCDF `T = theta - 300`. Vertical: `bottom_top` mass levels. | WRF `T`, K, perturbation potential temperature `theta-t0`; `t0 = 300 K`. Vertical: `bottom_top` mass levels. | `CONVERTIBLE_WITH_FIX` | `wrf: values + 300.0`; `gpu: values if median(raw T) > 150 else values + 300.0`. |
| T | GPU NetCDF `T`, K, perturbation theta in current files. Vertical: `bottom_top` mass levels. | WRF NetCDF `T`, K, perturbation theta. Vertical: `bottom_top` mass levels. | `BITWISE_MATCH` for convention, not values | None. Raw `T` is not separately in the M9 16-field trace. |
| T2 | GPU `T2`, K, surface mass grid; writer uses explicit `T2` or defaults to lowest model theta. | WRF `T2`, K, surface mass grid. | `REAL_BUG` | None. Units/reference/stagger match; values diverge. |
| U | `State.u`, m s-1, C-grid X-staggered, `bottom_top/south_north/west_east_stag`. | WRF `U`, m s-1, X-staggered, same dimension convention. | `REAL_BUG` | None. Native stagger comparison is correct. |
| V | `State.v`, m s-1, C-grid Y-staggered, `bottom_top/south_north_stag/west_east`. | WRF `V`, m s-1, Y-staggered, same dimension convention. | `REAL_BUG` | None. Native stagger comparison is correct. |
| U10 | GPU `U10`, m s-1, surface mass grid. | WRF `U10`, m s-1, surface mass grid. | `REAL_BUG` | None. |
| V10 | GPU `V10`, m s-1, surface mass grid. | WRF `V10`, m s-1, surface mass grid. | `REAL_BUG` | None. |
| W | `State.w`, m s-1, Z-staggered, `bottom_top_stag/south_north/west_east`. | WRF `W`, m s-1, Z-staggered, same dimension convention. | `REAL_BUG` | None. Native stagger comparison is correct. |
| PSFC | GPU `PSFC`, Pa, surface mass grid; writer uses explicit `PSFC` or `P + PB` at level 0. | WRF `PSFC`, Pa, surface mass grid. | `REAL_BUG` | None. |
| P | GPU writer outputs `P` from `State.p_perturbation`, Pa, mass levels. | WRF `P`, Pa, perturbation pressure, mass levels. | `BITWISE_MATCH` for convention, not compared in current trace | None. |
| PB | GPU writer outputs `PB` base pressure, Pa, mass levels. | WRF `PB`, Pa, base-state pressure, mass levels. | `BITWISE_MATCH` for convention, not compared in current trace | None. |
| PH | GPU writer outputs `PH` from `State.ph_perturbation`, m2 s-2, Z-staggered. | WRF `PH`, m2 s-2, perturbation geopotential, Z-staggered. | `BITWISE_MATCH` for convention, not compared in current trace | None. |
| QVAPOR | `State.qv`, kg kg-1, mass levels. | WRF `QVAPOR`, kg kg-1, mass levels. | `REAL_BUG` | None. Divergence is small but nonzero. |
| SWDOWN | GPU `SWDOWN`, W m-2, surface mass grid. | WRF `SWDOWN`, W m-2, surface mass grid. | `REAL_BUG` | None. Unit metadata matches; no kW/W conversion found. |
| GLW | GPU `GLW`, W m-2, surface mass grid. | WRF `GLW`, W m-2, surface mass grid. | `REAL_BUG` | None. Unit metadata matches. |
| HFX | GPU `HFX`, W m-2 upward surface heat flux; writer fallback is `theta_flux * rhosfc * cp`. | WRF `HFX`, W m-2 upward surface heat flux. | `REAL_BUG` | None. Sign/unit metadata matches. |
| LH | GPU `LH`, W m-2 latent heat flux; writer fallback is `qv_flux * rhosfc * Lv`. | WRF `LH`, W m-2 latent heat flux. | `REAL_BUG` | None. Unit metadata matches. |
| PBLH | GPU `PBLH`, m, surface mass grid. | WRF `PBLH`, m, surface mass grid. | `REAL_BUG` | None. |
| TSK | GPU `TSK`, K, surface mass grid, replayed from input. | WRF `TSK`, K, surface mass grid. | `BITWISE_MATCH` | None. RMSE and max difference are 0.0. |
| LU_INDEX | GPU `LU_INDEX`, category field, surface mass grid; writer default derives from landmask if no explicit value. | WRF `LU_INDEX`, category field, surface mass grid. | `REAL_BUG` | None in comparator. Static data mismatch remains. |

## Cross-Cutting Findings

- Reference state: only theta needed comparator-side conversion. Current GPU wrfout `T` is perturbation, but the comparator now also supports future GPU files that write absolute theta.
- Units: audited NetCDF metadata matches for all compared fields. No confirmed unit conversion was applied.
- Vertical staggering and indexing: U, V, W, PH use WRF-native staggered dimensions on both sides; theta/T/P/PB/QVAPOR use mass levels. The comparator correctly compares native wrfout shapes without destaggering.
- NaN and missing data: audited fields had zero nonfinite counts in the v2 trace. `_stats` still records nonfinite counts and compares only finite pairs.
- C-grid vs A-grid winds: U/V/W are compared on native C-grid locations; U10/V10 are 2D mass-grid diagnostics. No destaggering fix is appropriate for this wrfout-level comparison.
- `scripts/m6b6_coupled_step_compare_1000.py`: no shared wrfout `T` conversion path was found. It compares savepoint arrays through `m6b5_dycore_step_compare.py`, so no M9.C patch was applied there.

## Proofs

- Fixed trace: `proofs/m9/operational_trace_hourly_v2.json`
- Updated divergence map: `proofs/m9/divergence_map_v2.json`
- Theta convention detection: GPU `T` reference states were `{"perturbation_from_300K_base": 24}`; WRF `T` reference states were `{"perturbation_from_300K_base": 24}`.
