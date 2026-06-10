# V0.14 NoahMP Step-1 Closure

Verdict: `NOAHMP_STEP1_WIRED_STRICT_RED_NARROWED_TO_NOAHMP_LAND_TILE_ENERGY`.

## Configuration (was the contract blocker)

- `use_noahmp=True`, `sf_surface_physics=4`, `inputs_have_noahmp_land=True` (previously False/None/False).
- WRF clock: julian `120.75` (0-based fractional), yearlen `365.0`.
- `topo_shading=1`, `slope_rad=1`, radiation_static loaded `True`.

## Truth provenance

- Strict target re-emitted from ONE run of the rmol-PINNED WRF binary; byte-identical across re-runs and across two pinned builds (`{"mynn_vs_surfacehandoff_run": true, "mynn_vs_rmolpin_build": true, "surface_vs_surfacehandoff_run": true}`).

## Strict gate

- after-conv `T_TENDF` vs JAX dry: max_abs `1489.5135568470864`, RMSE `13.2001844004901` (pass: max_abs <= 0.001, RMSE <= 1e-05).

## WRF-anchored boundary measurements

- Noah-MP forcing seed (lead=radt/2): SOLDN vs WRF SWDOWN all rmse `2.759245466496041` W/m2; LWDN vs WRF GLW bias `17.441338534376907` W/m2.
- lead=0 contrast SWDOWN rmse `56.43169584231224` W/m2 (falsifies the lead-0 seed convention).
- mass-coupled RTHRATEN vs WRF part2 (interior): max_abs `19.425283200182427`, rmse `2.4884141898276413` (WRF field max `41.89738082885742`).
- post-overlay MYNN boundary theta_flux land: max_abs `0.251691908645883`, bias `-0.0418993964983867` (K m/s).
- raw RTHBLTEN vs WRF: max_abs `0.016072531466352954`, rmse `0.00011917325735225103`, strong-median `0.41187007310784046`, corr `0.8561204423861145`.

## Causal split (radiation-swap) + land input parity

- land theta_flux residual with the JAX seed radiation: `{"count": 768, "max_abs": 0.2516927769985068, "rmse": 0.061655405831022324, "bias": -0.04189539728540544, "ref_max_abs": 0.3119690397296882}`.
- land theta_flux residual with WRF's EXACT hook SWDOWN/GLW: `{"count": 768, "max_abs": 0.25215159752150773, "rmse": 0.0643496272333122, "bias": -0.04662137066786276, "ref_max_abs": 0.3119690397296882}` -> the deficit does NOT collapse: it is not the radiation forcing.
- land inputs (tslb1/smois1/sh2o1/tsk/vegfra/snow) match WRF PRE_NOAHMP to hook print precision (max_abs <= 5e-9); the diagnostic-level albedo/znt carry rows flag the two-stream albedo chain for the energy hook.

## Ranked hypotheses

1. JAX Noah-MP land-tile step-1 surface energy balance (land theta_flux max_abs 0.252 K m/s; land inputs match WRF to hook precision; WRF-truth-radiation swap does NOT collapse the residual; sfclay first-call fixed; MYNN kernel exonerated) (evidence: rad_swap_causal_split + land_input_parity + post_overlay_mynn_boundary + kernel_matrix)
2. RRTMG step-1 surface/atmosphere radiation forcing parity (GLW bias 17.44 W/m2, SWDOWN rmse 2.76 W/m2, mass-coupled RTHRATEN residual 19.4) (evidence: rad_seed_vs_wrf_hook + rthraten_vs_wrf_part2)

## Fastest next command

`Emit a per-column WRF noahmplsm ENERGY in/out hook on the pinned tree (FVEG/LAI/SAI, CM/CH in+out, two-stream SAV/SAG/FSR/FSA albedo chain, SH/EV/GH/TRAD/T2MV/T2MB, plus EFLXB terms) at step 1, column-diff it against the JAX physics.noahmp energy solve on the worst land cells (start at the strict worst cell i=66 j=37), fix the diverging chain, then rerun JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/noahmp_step1_closure.py (secondary lane: RRTMG GLW +17 W/m2 uniform clear-sky bias + RTHRATEN parity via an RRTMG forcing hook)`
