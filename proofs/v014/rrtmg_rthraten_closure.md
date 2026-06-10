# V0.14 RRTMG RTHRATEN/GLW Closure

Verdict: `RRTMG_RTHRATEN_GLW_MOIST_THETA_INPUT_FIXED_REMAINING_RESIDUAL_SPLIT_BOUNDED`.

## Fix

- Owner: `src/gpuwrf/coupling/physics_couplers.py::_rrtmg_column_inputs`.
- Exact pre-fix WRF boundary: `RRTMG_LWRAD:T3D=t`.
- Change: metric-backed RRTMG input now decouples stored `theta_m` to dry theta before temperature conversion, matching WRF `phy_prep`.

## WRF Oracle Anchor

- Oracle root: `/tmp/wrf_gpu2_step1_tsk_znt_sourcing_fix/wrf_truth_surface/radiation`.
- Dimensions `ni,nk,nj`: `[159, 44, 66]`.
- WRF oracle GLW vs public surface GLW max_abs: `5.000003966415534e-09` W/m2.
- WRF oracle `(RTHRATENLW+RTHRATENSW)*MASS_H` vs public part2 `RTHRATEN` max_abs: `3.249855922149436e-06`.

## Before/After

- `T3D=t` max_abs: `5.521345498302992` K -> `0.08944393302414255` K.
- GLW RMSE: `17.520282676793663` -> `0.35152062180132065` W/m2 (factor `49.84140784404996`).
- GLW max_abs: `22.521139406985185` -> `1.2638192831770994` W/m2.
- Mass-coupled RTHRATEN RMSE: `2.4884141898276413` -> `0.3645729657536835` (factor `6.825558731935404`).
- Mass-coupled RTHRATEN max_abs: `19.425283200182427` -> `2.798351397503893`.

## Remaining Bound

- Production LW split max_abs: `3.0125375954695457`, RMSE `0.1643178432813847`.
- Production SW split max_abs: `0.9634340145625964`, RMSE `0.2378140906941712`.
- Fastest next command: `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/rrtmg_rthraten_closure.py`.
