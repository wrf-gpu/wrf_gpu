# V0.14 Step-1 First-RK Part1 P-State Split

Verdict: `STEP1_FIRST_RK_PART1_P_STATE_LOCALIZED_PRE_PART1_RAW_CHILD_STATE`.

## Result

- CPU backend: `cpu`; GPU used: `False`.
- Required ancestor `ebedb3c1` present: `True`.
- Fastest rigorous method: `YES_FASTEST_RIGOROUS_WALL_CLOCK_REUSED_CPU_WRF_SAVEPOINTS_PLUS_CURRENT_CPU_JAX_STAGE_CAPTURE`.
- Predecessor final top residuals preserved: `P` max_abs `974.9820434775493`, `PH` max_abs `67.3623167023926`, `MU` max_abs `14.125275642998986`, `W` max_abs `2.640715693903735`, `U` max_abs `0.7835467705023085`.
- WRF `before_first_rk_step_part1_call` -> `after_first_rk_step_part1` is exact for `P_STATE/MU_STATE/W_STATE/PH_STATE`.
- WRF part1 entry -> `after_phy_prep`: `P_STATE` max_abs `0.0`, `MU_STATE` max_abs `0.0`.
- JAX `live_child_state` vs WRF pre-call: `P_STATE` max_abs `69.96875`, `MU_STATE` `13.256103515625`, `W_STATE` `0.7605466246604919`.
- JAX `haloed_step_entry_state` carries the same residuals: `P_STATE` max_abs `69.96875`, `MU_STATE` `13.256103515625`, `W_STATE` `0.7605466246604919`.
- No production source fix was applied.

## Boundary Table

| Boundary/check | P_STATE | MU_STATE | W_STATE | PH_STATE |
|---|---:|---:|---:|---:|
| WRF pre-call -> after_first_rk_step_part1 | 0.0 | 0.0 | 0.0 | 0.0 |
| JAX raw_child_state vs WRF pre-call | 69.96875 | 13.256103515625 | 0.7605466246604919 | 0.00048828125 |
| JAX live_child_state vs WRF pre-call | 69.96875 | 13.256103515625 | 0.7605466246604919 | 0.00048828125 |
| JAX haloed_step_entry_state vs WRF pre-call | 69.96875 | 13.256103515625 | 0.7605466246604919 | 0.00048828125 |

## Interpretation

- The current `P/MU/W` state mismatch is not introduced by WRF `first_rk_step_part1` or `phy_prep`; it exists before the call.
- JAX live-nest base/theta/QV correction does not update perturbation `P_STATE/MU_STATE/W_STATE`; those leaves are unchanged through boundary package, carry construction, halo, and `_physics_step_forcing.carry.state`.
- The exact missing contract is `raw_child_state -> live_child_state` perturbation-state initialization for `P_STATE/MU_STATE/W_STATE` against WRF `before_first_rk_step_part1_call`.

Detailed metrics are in `proofs/v014/step1_first_rk_part1_p_state_split.json`.
