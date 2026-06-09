# V0.14 Same-State Momentum/Mass

Verdict: `JAX_MISMATCH_U_post_after_all_rk_steps_pre_halo`.

## Target

- Surface: `post_after_all_rk_steps_pre_halo`.
- Boundary: `post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges`.
- Domain/step: `d02`, step `6000`, h10 `2026-05-02T04:00:00+00:00`.
- Selected mass cell: `{'south_north': 9, 'west_east': 13}`.
- Native staggering preserved for U/V/W/PH/PHB.

## Comparison

- CPU-only JAX wrapper run: `RAN`.
- First failing field in sprint order: `U` max_abs `6.292358893898424` rmse `2.032497018496295`.
- Worst native key: `[4, 13]`; JAX `-4.735481996086533` vs WRF `1.55687689781189`.

| Field | Truth | Count | Max abs | RMSE | Worst native key |
| --- | --- | ---: | ---: | ---: | --- |
| U | wrf_text_surface | 306 | 6.292358893898424 | 2.032497018496295 | [4, 13] |
| V | wrf_text_surface | 306 | 11.594389190354377 | 4.454165526807795 | [9, 13] |
| W | wrf_text_surface | 578 | 1.7341964464592572 | 0.2385078386197167 | [1, 9, 16] |
| T | wrf_text_surface | 289 | 3.3545763228707983 | 1.0296598586362888 | [12, 17] |
| P | wrf_text_surface | 289 | 590.0020891006836 | 526.3359359284732 | [13, 17] |
| PB | wrf_text_surface | 289 | 1047.015625 | 223.43483550580925 | [7, 16] |
| PH | wrf_text_surface | 578 | 5.097192302780417 | 2.336601078950881 | [1, 8, 19] |
| PHB | green_wrf_h10_wrfout_static_field | 578 | 878.0291748046875 | 188.2021628878216 | [0, 7, 16] |
| MU | wrf_text_surface | 289 | 267.1935248522832 | 195.97477055691238 | [13, 18] |
| MUB | wrf_text_surface | 289 | 1050.3046875 | 224.13660680282618 | [7, 16] |

## Source Hypothesis

- The nearest named surface already fails at post-RK/pre-halo momentum state, before later writer or RK halo cadence can explain it.
- Next source localization should move one layer earlier inside the final RK step: large-step U/V tendency assembly, acoustic U/V update, mass coupling, and theta/history source feeding pressure refresh.
- Base-source priority: The base-source proof is a partial correctness fix and was produced after this h10 carry. It should trigger a fresh h10 carry before attributing PB/MUB/PHB residuals, but it does not lower the priority of the dynamic momentum/mass hypothesis because this same-surface comparison first fails `U` in sprint field order.
