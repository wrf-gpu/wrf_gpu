# V0.14 WRF Post-RK Refresh Localization

Verdict: `REFRESH_LAYER_GREEN_post_after_all_rk_steps_pre_halo`.

CPU-only WRF emitted two refresh surfaces at `d02`, step `6000`, h10
`2026-05-02_04:00:00`: post final `calc_p_rho_phi`, and immediately after
`after_all_rk_steps` before RK halo exchanges. The green post marker from
Herschel was emitted in the same run for the same native patch.

## Target

- selected mass cell: zero-based `(y=9, x=13)`
- mass patch: `{'halo_radius_cells': 8, 'south_north_start': 1, 'south_north_stop_exclusive': 18, 'west_east_start': 5, 'west_east_stop_exclusive': 22}`
- WRF time before step: `2026-05-02_03:59:54`
- WRF step: `6000`
- native coordinates: mass/U/V/W-PH staggering preserved

## Refresh Surface Vs Marker

Post final `calc_p_rho_phi` vs green post marker:

| Field | Count | Max abs | RMSE |
| --- | ---: | ---: | ---: |
| T_HIST_SRC_vs_marker_T | 289 | 0.0 | 0.0 |
| T_THM_vs_marker_T | 289 | 5.702972412109375 | 5.4497435060612585 |
| P_vs_marker_P | 289 | 0.0 | 0.0 |
| PB_vs_marker_PB | 289 | 0.0 | 0.0 |
| U_vs_marker_U | 306 | 0.0 | 0.0 |
| V_vs_marker_V | 306 | 57.9480562210083 | 8.125716617719473 |
| W_vs_marker_W | 578 | 0.00010097026824995581 | 6.654755737086828e-06 |
| PH_vs_marker_PH | 578 | 0.0 | 0.0 |

Post `after_all_rk_steps` pre-halo vs green post marker:

| Field | Count | Max abs | RMSE |
| --- | ---: | ---: | ---: |
| T_HIST_SRC_vs_marker_T | 289 | 0.0 | 0.0 |
| T_THM_vs_marker_T | 289 | 5.702972412109375 | 5.4497435060612585 |
| P_vs_marker_P | 289 | 0.0 | 0.0 |
| PB_vs_marker_PB | 289 | 0.0 | 0.0 |
| U_vs_marker_U | 306 | 0.0 | 0.0 |
| V_vs_marker_V | 306 | 0.0 | 0.0 |
| W_vs_marker_W | 578 | 0.0 | 0.0 |
| PH_vs_marker_PH | 578 | 0.0 | 0.0 |

## Candidate Vs WRFout

Post `after_all_rk_steps` pre-halo vs scratch h10 wrfout:

| Field | Count | Max abs | RMSE |
| --- | ---: | ---: | ---: |
| T | 289 | 0.0 | 0.0 |
| P | 289 | 0.0 | 0.0 |
| PB | 289 | 0.0 | 0.0 |
| MU | 289 | 4.547473508864641e-13 | 3.4568416978372316e-13 |
| MUB | 289 | 0.0 | 0.0 |
| U | 306 | 8.881784197001252e-16 | 2.7027201268900864e-16 |
| V | 306 | 8.881784197001252e-16 | 2.6017692407882053e-16 |
| W | 578 | 4.440892098500626e-16 | 4.0846014518837006e-17 |
| PH | 578 | 5.329070518200751e-15 | 2.0806684742968092e-15 |

Retained GPU/JAX h10 wrfout vs the same WRF candidate:

| Field | Count | Max abs | RMSE |
| --- | ---: | ---: | ---: |
| T | 289 | 3.357818603515625 | 1.0306179245297085 |
| P | 289 | 590.0020751953125 | 526.3440071888446 |
| PB | 289 | 1047.015625 | 223.43483393562332 |
| MU | 289 | 267.19348144531295 | 195.97551106410862 |
| MUB | 289 | 1050.3046875 | 224.13660572852498 |
| U | 306 | 6.29096531867981 | 2.03052660423855 |
| V | 306 | 11.590928077697754 | 4.454532390676284 |
| W | 578 | 1.7341964393854141 | 0.23850793231592415 |
| PH | 578 | 5.097192764282227 | 2.336601070572846 |

## Cadence

The final `calc_p_rho_phi` boundary closes the large `P` gap from Ptolemy's
post-`small_step_finish` layer. The post-`after_all_rk_steps` pre-halo surface
is the named candidate for a JAX CPU wrapper if the table above is green.

Next JAX CPU wrapper target: `post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges`.
