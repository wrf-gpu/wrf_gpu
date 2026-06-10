# V0.14 Step-1 Thermodynamic Column Inputs

Verdict: `THERMO_COLUMN_INPUTS_FIXED_NEXT_BLOCKER_SURFACE_LAYER_OUTPUTS`.

## Root Cause

- `_surface_column_view` was feeding `State.theta` as WRF `th_phy`; this live-nest state is theta_m, while WRF `phy_prep` passes dry theta: `(theta_m)/(1+R_v/R_d*qv)`.
- `_surface_column_view` was feeding nonhydrostatic `state.p`; WRF `surface_driver` passes `P_PHY=grid%p_hyd` for this call.
- `dz8w` used standard gravity `9.80665`; WRF `phy_prep` uses physics `g=9.81`.
- WRF `t_phy` is the split exception: it is computed from dry theta and nonhydrostatic `p+pb`, then passed beside hydrostatic `P_PHY`.

## Boundary Result

- Legacy theta_m vs WRF `th_phy(kts)`: max_abs `5.490148027499686` K.
- Fixed dry `th_phy(kts)`: max_abs `6.71089752017906e-05` K, RMSE `1.3430183262692343e-05`.
- Fixed `t_phy(kts)`: max_abs `0.013577942721781255` K, RMSE `0.0010959870065792568`.
- Fixed hydrostatic `p_phy(kts)`: max_abs `0.015625` Pa, RMSE `0.0013253267749381015`.
- Fixed `dz8w(kts)`: max_abs `0.00018988715282830526` m, RMSE `7.774106387517223e-06`.
- Fixed `psfc`: max_abs `0.015625` Pa.

## Next Blocker

The thermodynamic input boundary is fixed/bounded. Strict Step-1 is still red, so the remaining WRF-anchored blocker is later: MYNN surface-layer output algebra after the fixed input tuple.

- `UST` max_abs `0.0007252174862408534`, RMSE `1.53999402707944e-05`.
- `HFX` max_abs `0.2643125302157898`, RMSE `0.022548398654638105`.
- `QFX` max_abs `6.468560998136325e-08`, RMSE `3.002727253934746e-08`.
- `BR` max_abs `0.01166976922050278`, RMSE `0.0003583716190119449`.

## Strict Step-1

- after-conv `T_TENDF` max_abs `847.1446969755725`, RMSE `9.627208432391289`.

## Fastest Next Command

`Add a narrow WRF internal hook inside module_sf_mynn.F/SFCLAY1D_mynn for thx/thgb/br/zol/psim/psih/ust/hfx/qfx, then compare against surface_layer_with_diagnostics on the fixed input tuple.`

## Files

- JSON proof: `/home/enric/src/wrf_gpu2/proofs/v014/step1_thermo_column_inputs.json`
- Review: `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-10-v014-step1-thermo-column-inputs.md`
- WRF hook changes this sprint: `none`.
