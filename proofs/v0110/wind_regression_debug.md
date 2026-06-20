# v0.11.0 d02 Wind Regression Debug

Generated: 2026-06-06

## Objective

Localize, root-cause, and fix the v0.11.0 d02 surface-wind regression for
`20260507_18z_l2_72h_20260513T124307Z` on the fp64 segmented operational d02 path.

Requested run root `<DATA_ROOT>/canairy_meteo/runs/wrf_l2` did not contain d01/d02
`wrfout` history for the replay scorer. The harness recorded the fallback to the
same-run complete fixture under `/tmp/vburst_runs`, which is also the reference
fixture used by the v0.9/v0.11 proof objects.

## Bisect Results

| Variant | U10 mean RMSE | V10 mean RMSE | U10 wins/24 | V10 wins/24 | Readout |
| --- | ---: | ---: | ---: | ---: | --- |
| v0.9.0 shipped | 4.405737 | 3.568950 | 23 | 23 | target wind skill |
| v0.11.0 pre-fix fp64 | 5.539653 | 4.521301 | 0 | 0 | release-blocking regression |
| no MYNN momentum MF | 5.536116 | 4.518409 | 0 | 0 | direct MYNN-EDMF momentum MF not culprit |
| no MYNN-EDMF | 5.503782 | 4.470682 | 0 | 0 | MYNN-EDMF not primary |
| no dry physics tendencies | 5.124667 | 3.744659 | 16 | 8 | broad recovery signal in dry tendency bridge |
| no dry momentum tendencies | 5.539247 | 4.521359 | 0 | 0 | dry U/V/W tendencies not culprit |
| no dry theta tendency | 5.124598 | 3.744624 | 16 | 8 | recovery matches broad dry ablation |
| RRTMG topo/slope off | 5.539653 | 4.521301 | 0 | 0 | topo/slope radiation not culprit |
| v0.9 dry-physics cadence | 4.427962 | 3.592362 | 23 | 23 | recovery to shipped wind skill |
| fixed production baseline | 4.427962 | 3.592362 | 23 | 23 | fix confirmed |

No separate 24h GWD/KF controls were needed on d02: the replay namelist has
`GWD_OPT=0` and `CU_PHYSICS=0` for d02, and there is no GPU GWD implementation
being applied on this path. Disabling them is a no-op for this case.

## Culprit

The regression is in the v0.11 dry-physics-to-RK bridge, specifically the aggregate
theta/heating delta being injected through `rk_addtend_dry` as an RK-fixed dry
tendency.

It is not caused by direct momentum drag:

- Dropping dry U/V/W tendencies left U10/V10 at the bad v0.11 values.
- Dropping only dry theta reproduced the broad dry-tendency recovery.
- Restoring the shipped v0.9 post-dycore dry-physics cadence recovered U10/V10 to
  v0.9-level values and restored 23/24 persistence wins.

## Root Cause

v0.11 introduced `_dry_physics_tendencies_from_state_delta`, which converted the
post-physics `State` delta into `DryPhysicsTendencies`, including:

- `ru_tendf`, `rv_tendf`, `rw_tendf` from aggregate wind deltas
- `h_diabatic` from aggregate theta delta

That bridge treated already-integrated `State -> State` physics adapter output as
if it were raw WRF `R*TEN` source tendencies ready for `calculate_phy_tend` /
`rk_addtend_dry`. WRF's `rk_addtend_dry` expects scheme/source tendencies with the
correct mass/map coupling and RK cadence, not an aggregate full-step state delta.
Feeding the aggregate theta delta into the RK merge changed the thermal forcing
cadence and degraded surface winds from h+1 onward.

## Fix

Production fix in `src/gpuwrf/runtime/operational_mode.py`:

- Keep `_dry_physics_tendencies_from_state_delta` empty until a scheme exposes true
  WRF `*_tendf` leaves.
- Apply the already-integrated dry physics state deltas (`u`, `v`, `w`, `theta`)
  after the dycore through the existing post-dynamics physics update path.

This matches the shipped v0.9 dry-physics cadence for the current adapter contract
and avoids fabricating RK tendencies from aggregate state deltas.

## Proof Objects

- Bisect harness: `proofs/v0110/wind_regression_bisect.py`
- Bisect JSONs: `proofs/v0110/wind_regression/*/{variant_summary,d02_coupled_skill}.json`
- Recovery JSONs: `proofs/v0110/wind_regression_recovery/baseline/{variant_summary,d02_coupled_skill}.json`

Key commands:

```bash
PYTHONPATH=src python -m py_compile src/gpuwrf/runtime/operational_mode.py proofs/v0110/wind_regression_bisect.py

/tmp/wrf_gpu_run.sh bash -lc 'set -euo pipefail
for variant in no_mynn_momentum_mf no_mynn_edmf no_dry_physics_tendencies rrtmg_topo_slope_off; do
  taskset -c 0-27 env PYTHONPATH=src python proofs/v0110/wind_regression_bisect.py --variant "$variant" --hours 24 --output-root /tmp/v0110_wind_regression --proof-root proofs/v0110/wind_regression
done'

/tmp/wrf_gpu_run.sh bash -lc 'set -euo pipefail
for variant in no_dry_momentum_tendencies no_dry_theta_tendency dry_physics_post_dynamics_v090; do
  taskset -c 0-27 env PYTHONPATH=src python proofs/v0110/wind_regression_bisect.py --variant "$variant" --hours 24 --output-root /tmp/v0110_wind_regression --proof-root proofs/v0110/wind_regression
done'

/tmp/wrf_gpu_run.sh taskset -c 0-27 env PYTHONPATH=src python proofs/v0110/wind_regression_bisect.py --variant baseline --hours 24 --output-root /tmp/v0110_wind_regression_recovery --proof-root proofs/v0110/wind_regression_recovery
```

## Residual Risk

This fix is scoped to the current State-integrating adapter contract. A future
WRF-faithful raw-tendency adapter can populate `DryPhysicsTendencies`, but it must
provide true per-source WRF tendency leaves and prove the mass/map/RK coupling.
