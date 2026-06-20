# V0.14 T History Source Attribution

Verdict: `T_EVOLUTION_MISMATCH_CONFIRMED`.

## Target

- WRF target: `post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges`.
- WRF history `T`: `MASS_K1.T_HIST_SRC` (`grid%th_phy_m_t0`).
- WRF THM-side candidate: `MASS_K1.T_THM`.
- Tolerance: `2e-06` max_abs, unchanged from the h10 proof.

## Artifact Identity

- Checkpoint: `<DATA_ROOT>/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`.
- Same as producer record: `True`.
- Same as canonical h10 compared artifact: `True`.

## Best Matches To WRF History T

| Candidate | Max abs | RMSE |
| --- | ---: | ---: |
| `captured_pre_halo_state.theta_minus_300` | `3.3545763228707983` | `1.0296598586362888` |
| `captured_post_halo_carry.state.theta_minus_300` | `3.3545763228707983` | `1.0296598586362888` |
| `captured_final_carry.t_save_minus_300` | `3.356384688581045` | `1.0300584843288976` |
| `checkpoint_prestep_carry.t_save_minus_300` | `3.3567233047927516` | `1.0299191049611263` |
| `checkpoint_prestep_carry.state.theta_minus_300` | `3.3581935716514977` | `1.03048156240767` |

## Best Matches To WRF THM

| Candidate | Max abs | RMSE |
| --- | ---: | ---: |
| `captured_final_carry.t_2ave_minus_300` | `3.677881697025043` | `2.6121909969239896` |
| `checkpoint_prestep_carry.t_2ave_minus_300` | `3.677882024133112` | `2.6121910497910905` |
| `physics_state.theta_minus_300` | `6.2164342971682345` | `4.638106603411917` |
| `checkpoint_prestep_carry.state.theta_minus_300` | `6.219254650376172` | `4.638767406133128` |
| `checkpoint_prestep_carry.t_save_minus_300` | `6.220292179567309` | `4.639179299294447` |

## Context

- Pre-halo P/PB/MU/MUB max_abs: P `590.0020891006836`, PB `1047.015625`, MU `267.1935248522832`, MUB `1050.3046875`.
- WRF `T_THM - T_HIST_SRC` max_abs: `5.702972412109375`.

## Next Decision

Open a theta-evolution localization sprint; do not spend the next sprint on JAX-vs-WRF history source remapping for `T`.
