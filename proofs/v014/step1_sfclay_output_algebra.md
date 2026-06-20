# V0.14 Step-1 SFCLAY Output Algebra

Verdict: `SFCLAY_OUTPUT_ALGEBRA_BOUNDED_NEXT_BLOCKER_MYNN_SOURCE_COUPLING`.

## Fixes

- Ported WRF first-step MYNN `BR` clamp: `[-2,2]` for `itimestep==1`, `[-4,4]` only for warm steps.
- Ported WRF `QVSH=QV1D/(1+QV1D)` in virtual-theta terms.
- Threaded WRF `phy_prep` density `rho=(1+qv)/alt` into the surface column view.

## Surface Boundary

- `UST` max_abs `0.0007252174862408534`, RMSE `1.53999402707944e-05`.
- `HFX` max_abs `0.2643125302157898`, RMSE `0.022548398654638105`.
- `QFX` max_abs `6.468560998136325e-08`, RMSE `3.002727253934746e-08`.
- `BR` max_abs `0.01166976922050278`, RMSE `0.0003583716190119449`.
- `ZOL/PSIM/PSIH` max_abs `0.15103377367224624` / `0.18334270054241664` / `0.15301694678136535`.
- `rho` max_abs `0.00018143653869628906`, RMSE `7.786468426065368e-06`.

## Strict Step-1

- after-conv `T_TENDF` remains red: max_abs `438.5379097262689`, RMSE `5.4654420375782955`.
- Worst cell remains `{'i': 74, 'j': 39, 'k': 1}`; surface-layer output algebra no longer explains this order-847 residual.

## Narrower Blocker

The remaining blocker is later than `sfclay_mynn` output algebra: MYNN/PBL source coupling after the fixed surface outputs. The next proof needs exact WRF MYNNEDMF input fluxes and raw post-driver `dth1/dqv1` before `module_em` mass scaling.

## Fastest Next Command

`Add/rerun a WRF module_pbl_driver/module_bl_mynnedmf raw-source hook after the fixed surface outputs, emitting exact MYNNEDMF input fluxes plus raw post-driver dth1/dqv1 before module_em mass scaling; then compare against mynn_adapter_with_source_leaves.`

## Files

- JSON proof: `<USER_HOME>/src/wrf_gpu2/proofs/v014/step1_sfclay_output_algebra.json`
- WRF hook patch archive: `<USER_HOME>/src/wrf_gpu2/proofs/v014/step1_sfclay_output_algebra_wrf_patch.diff`
- Review: `<USER_HOME>/src/wrf_gpu2/.agent/reviews/2026-06-10-v014-step1-sfclay-output-algebra.md`
