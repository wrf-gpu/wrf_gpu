# V0.14 WRF Dynamic Term Localization

Verdict: `TERM_LAYER_EMITTED_final_stage_small_step_finish`.

This sprint emitted the first compact source-derived dynamic layer from CPU WRF:
`final_stage_pre_small_step_finish` and `final_stage_post_small_step_finish`
inside `dyn_em/solve_em.F::solve_em`, at `d02` step 6000, valid h10
`2026-05-02_04:00:00`. No model source under repo `src/` was edited and no GPU
was used by this sprint.

## Target

- selected mass cell: zero-based `(y=9, x=13)`
- selected patch: `{'halo_radius_cells': 8, 'south_north_start': 1, 'south_north_stop_exclusive': 18, 'west_east_start': 5, 'west_east_stop_exclusive': 22}`
- marker time before step: `2026-05-02_03:59:54`
- WRF step: `6000`
- RK stage: `rk_step=3/rk_order=3`

## Emitted Layer

- pre files: `2`
- post files: `2`
- unique post counts: `{'MASS_K1': 289, 'U_K1': 306, 'V_K1': 306, 'WPH_KSTAG01': 578}`
- duplicate post overlap: `17`, max delta `96731.453125`
- emitted named terms: `RU/RV/RW/T/PH/MU_TEND` and `RU/RV/RW/T/PH/MU_TENDF`

## Boundary Result

The tile-local `post_small_step_finish` layer is a useful source-derived layer,
but it is not yet the history-aligned h10 surface for `P/V/W` or THM-side `T`.
The later post-RK marker remains the green history anchor.

Post `small_step_finish` surface vs post-RK marker:

| Field | Count | Max abs | RMSE |
| --- | ---: | ---: | ---: |
| T_HIST_SRC_vs_marker_T | 289 | 0.003509521484375 | 0.0003686447840350194 |
| T_THM_vs_marker_T | 289 | 5.702972412109375 | 5.449877336887554 |
| P_vs_marker_P | 289 | 1981.2389628887177 | 1937.0772588531227 |
| PB_vs_marker_PB | 289 | 0.0 | 0.0 |
| U_NEW_vs_marker_U | 306 | 0.0 | 0.0 |
| V_NEW_vs_marker_V | 306 | 57.9480562210083 | 8.125716617719473 |
| W_NEW_vs_marker_W | 578 | 0.00010097026824995581 | 6.654755737086828e-06 |
| PH_NEW_vs_marker_PH | 578 | 0.0 | 0.0 |

## Marker Alignment

Post marker vs scratch h10 wrfout:

| Field | Count | Max abs | RMSE |
| --- | ---: | ---: | ---: |
| T | 289 | 0.0 | 0.0 |
| P | 289 | 0.0 | 0.0 |
| PB | 289 | 0.0 | 0.0 |
| U | 306 | 8.881784197001252e-16 | 2.7027201268900864e-16 |
| V | 306 | 8.881784197001252e-16 | 2.6017692407882053e-16 |
| W | 578 | 4.440892098500626e-16 | 4.0846014518837006e-17 |
| PH | 578 | 5.329070518200751e-15 | 2.0806684742968092e-15 |

Post marker vs provided CPU h10 wrfout:

| Field | Count | Max abs | RMSE |
| --- | ---: | ---: | ---: |
| T | 289 | 0.0 | 0.0 |
| P | 289 | 0.0 | 0.0 |
| PB | 289 | 0.0 | 0.0 |
| U | 306 | 4.76837158203125e-07 | 1.3106758362804406e-07 |
| V | 306 | 9.5367431640625e-07 | 3.1466681451378603e-07 |
| W | 578 | 1.1920928977282585e-07 | 9.606584126922052e-09 |
| PH | 578 | 1.9073486328125e-06 | 4.3090067180366245e-07 |

Retained GPU/JAX h10 wrfout minus the same WRF post marker still has the target
patch divergence:

| Field | Count | Max abs | RMSE |
| --- | ---: | ---: | ---: |
| T | 289 | 3.357818603515625 | 1.0306179245297085 |
| P | 289 | 590.0020751953125 | 526.3440071888446 |
| PB | 289 | 1047.015625 | 223.43483393562332 |
| U | 306 | 6.29096531867981 | 2.03052660423855 |
| V | 306 | 11.590928077697754 | 4.454532390676284 |
| W | 578 | 1.7341964393854141 | 0.23850793231592415 |
| PH | 578 | 5.097192764282227 | 2.336601070572846 |

## T Source

- `T_HIST_SRC` (`grid%th_phy_m_t0`) vs post marker T: count=289 max_abs=0.003509521484375 rmse=0.0003686447840350194
- `T_THM` (`grid%t_2`) vs post marker T: count=289 max_abs=5.702972412109375 rmse=5.449877336887554

This preserves the green marker lesson: history `T` must come from
`grid%th_phy_m_t0`, not `grid%t_1/grid%t_2`.

## Next

Next exact layer: instrument the pressure/rho/post-RK refresh path between
`small_step_finish` and the accepted marker after `after_all_rk_steps`, or
compare JAX only against the already-green post-RK marker state. Do not claim a
root cause from this WRF-only layer.
