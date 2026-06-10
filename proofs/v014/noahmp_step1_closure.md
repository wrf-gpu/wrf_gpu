# V0.14 NoahMP Step-1 Closure

Verdict: `NOAHMP_STEP1_STRICT_RED_SURFACE_WATERPATH_CLOSED_NARROWED_TO_MYNN_EDMF_RTHBLTEN`.

## Configuration (was the contract blocker)

- `use_noahmp=True`, `sf_surface_physics=4`, `inputs_have_noahmp_land=True` (previously False/None/False).
- WRF clock: julian `120.75` (0-based fractional), yearlen `365.0`.
- `topo_shading=1`, `slope_rad=1`, radiation_static loaded `True`.

## Truth provenance

- Strict target re-emitted from ONE run of the rmol-PINNED WRF binary; byte-identical across re-runs and across two pinned builds (`{"mynn_vs_surfacehandoff_run": true, "mynn_vs_rmolpin_build": true, "surface_vs_surfacehandoff_run": true}`).

## Strict gate

- after-conv `T_TENDF` vs JAX dry: max_abs `53.52301833555157`, RMSE `2.5444971494115354` (pass: max_abs <= 0.001, RMSE <= 1e-05).
- worst cell Fortran `{'i': 20, 'j': 7, 'k': 2}`: WRF `-1278.747436523438` vs JAX `-1225.2244181878864`.

## Surface-layer water-path closure (this sprint)

- The Noah-MP/sfclay column view (`coupling.noahmp_surface_hook._build_column_view`) now supplies the WRF `phy_prep` dry `t_air`, true `psfc`, and density (mirroring `physics_couplers._surface_column_view`), threaded via `noahmp_surface_step(grid=...)`. Previously it fed the surface layer raw moist `theta_m` with a naive Exner (~+4 K warm) and the air-pressure/ideal-gas fallback, corrupting the WATER-column sfclay flux that MYNN consumes.
- Effect (proofs/v014/surface_layer_theta_decoupling.*): water HFX rmse 11.87->0.012 W/m2, ust ~exact; strict max_abs 1489.5->53.52301833555157, rmse 12.15->2.5444971494115354. The remaining residual is MYNN-EDMF RTHBLTEN (land+water), not the surface coupling.

## WRF-anchored boundary measurements

- Noah-MP forcing seed (lead=radt/2): SOLDN vs WRF SWDOWN all rmse `2.759245466496041` W/m2; LWDN vs WRF GLW bias `17.441338534376907` W/m2.
- lead=0 contrast SWDOWN rmse `56.43169584231224` W/m2 (falsifies the lead-0 seed convention).
- mass-coupled RTHRATEN vs WRF part2 (interior): max_abs `19.425283200182427`, rmse `2.4884141898276413` (WRF field max `41.89738082885742`).
- post-overlay MYNN boundary theta_flux land: max_abs `0.017792882212240096`, bias `0.005563593586436484` (K m/s).
- raw RTHBLTEN vs WRF: max_abs `0.0026920951794989`, rmse `1.074050795420995e-05`, strong-median `1.0049841082567772`, corr `0.9987784604909195`.

## Causal split (radiation-swap) + land input parity

- land theta_flux residual with the JAX seed radiation: `{"count": 768, "max_abs": 0.017793968832694823, "rmse": 0.006422627622543693, "bias": 0.00556777310000654, "ref_max_abs": 0.3119690397296882}`.
- land theta_flux residual with WRF's EXACT hook SWDOWN/GLW: `{"count": 768, "max_abs": 0.00044824676661980867, "rmse": 0.00012291627662459686, "bias": 9.126523704482347e-05, "ref_max_abs": 0.3119690397296882}` -> AFTER the moist-theta->dry-T decoupling fix this COLLAPSES (the remaining land residual IS the RRTMG radiation forcing). See `proofs/v014/noahmp_land_tile_energy_closure.*`.
- The prior 'NoahMP land-tile energy' narrowing is REFUTED: the energy solve is exact to ~1e-3 W/m2 with WRF-exact inputs; the residual was a +4 K-warm air temperature (state.theta is moist theta_m, converted with a naive Exner) -- FIXED in noahmp_coupler.assemble_noahmp_forcing this sprint.

## Ranked hypotheses

1. MYNN-EDMF RTHBLTEN PBL theta-tendency kernel residual (DOMINANT). With the surface-layer water-path fix landed (sfclay/Noah-MP now receive the WRF phy_prep dry t_air + true psfc + density; water HFX rmse 11.87->0.012 W/m2, ust ~exact -- see proofs/v014/surface_layer_theta_decoupling.*), the strict residual collapsed 1489.5->53.5 max_abs / 12.15->2.54 rmse. The remaining worst cells are RTHBLTEN-dominated on BOTH land and water (worst WRF -1278.7 vs JAX -1225.2; ~4-7% of the local RTHBLTEN where it is large), with RTHRATEN <=~19.4. This is inside module_bl_mynnedmf (mixing length / EDMF mass-flux / cold-start qke), NOT the surface coupling (now WRF-faithful) and NOT radiation. MYNN kernel is outside this sprint's file ownership. (evidence: land/water strict decomposition (RTHBLTEN-dominated, RTHRATEN<=~19.4) + surface_layer_theta_decoupling + post_overlay_mynn_boundary)
2. RRTMG step-1 radiation forcing parity (SECONDARY): GLW bias 17.44 W/m2, SWDOWN rmse 2.76 W/m2, mass-coupled RTHRATEN residual 19.4. The Noah-MP LAND theta_flux still collapses under the WRF-exact radiation swap (rmse 0.00642); RRTMG remains localized to a clear-sky derived optical/gas/top-buffer profile (proofs/v014/rrtmg_step1_forcing_parity.*) but is no longer the dominant strict lane (RTHRATEN max ~19.4 << strict max 53.5). (evidence: rad_seed_vs_wrf_hook + rthraten_vs_wrf_part2 + rrtmg_step1_forcing_parity)

## Fastest next command

`Surface-layer water-path is CLOSED: the Noah-MP/sfclay column view now supplies the WRF phy_prep dry t_air + true psfc + density (noahmp_surface_hook._build_column_view, mirroring physics_couplers._surface_column_view; see proofs/v014/surface_layer_theta_decoupling.*). Strict 1489.5->53.5 max_abs / 12.15->2.54 rmse. Remaining strict lanes: (1) DOMINANT = MYNN-EDMF RTHBLTEN kernel residual (~4-7% where RTHBLTEN is large, land+water, NOT radiation; module_bl_mynnedmf mixing-length/EDMF/cold-start qke) -- needs a MYNN-kernel sprint; (2) SECONDARY = RRTMG step-1 GLW/RTHRATEN forcing (max ~19.4, proofs/v014/rrtmg_step1_forcing_parity.*). Then rerun JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/noahmp_step1_closure.py.`
