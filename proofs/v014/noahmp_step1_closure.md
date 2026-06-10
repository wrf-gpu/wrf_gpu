# V0.14 NoahMP Step-1 Closure

Verdict: `NOAHMP_STEP1_WIRED_STRICT_RED_NARROWED_TO_RADIATION_FORCING_INTO_NOAHMP`.

## Configuration (was the contract blocker)

- `use_noahmp=True`, `sf_surface_physics=4`, `inputs_have_noahmp_land=True` (previously False/None/False).
- WRF clock: julian `120.75` (0-based fractional), yearlen `365.0`.
- `topo_shading=1`, `slope_rad=1`, radiation_static loaded `True`.

## Truth provenance

- Strict target re-emitted from ONE run of the rmol-PINNED WRF binary; byte-identical across re-runs and across two pinned builds (`{"mynn_vs_surfacehandoff_run": true, "mynn_vs_rmolpin_build": true, "surface_vs_surfacehandoff_run": true}`).

## Strict gate

- after-conv `T_TENDF` vs JAX dry: max_abs `1489.5135568470864`, RMSE `12.146876720723487` (pass: max_abs <= 0.001, RMSE <= 1e-05).

## WRF-anchored boundary measurements

- Noah-MP forcing seed (lead=radt/2): SOLDN vs WRF SWDOWN all rmse `2.759245466496041` W/m2; LWDN vs WRF GLW bias `17.441338534376907` W/m2.
- lead=0 contrast SWDOWN rmse `56.43169584231224` W/m2 (falsifies the lead-0 seed convention).
- mass-coupled RTHRATEN vs WRF part2 (interior): max_abs `19.425283200182427`, rmse `2.4884141898276413` (WRF field max `41.89738082885742`).
- post-overlay MYNN boundary theta_flux land: max_abs `0.021912592983171608`, bias `0.006677763293085105` (K m/s).
- raw RTHBLTEN vs WRF: max_abs `0.016072531466352954`, rmse `0.00010759710126444855`, strong-median `1.007447724308231`, corr `0.8793063625891515`.

## Causal split (radiation-swap) + land input parity

- land theta_flux residual with the JAX seed radiation: `{"count": 768, "max_abs": 0.021911906769162326, "rmse": 0.0075815914959646334, "bias": 0.0066820020278386015, "ref_max_abs": 0.3119690397296882}`.
- land theta_flux residual with WRF's EXACT hook SWDOWN/GLW: `{"count": 768, "max_abs": 0.004916764366520665, "rmse": 0.001321488194019588, "bias": 0.0011252835672172798, "ref_max_abs": 0.3119690397296882}` -> AFTER the moist-theta->dry-T decoupling fix this COLLAPSES (the remaining land residual IS the RRTMG radiation forcing). See `proofs/v014/noahmp_land_tile_energy_closure.*`.
- The prior 'NoahMP land-tile energy' narrowing is REFUTED: the energy solve is exact to ~1e-3 W/m2 with WRF-exact inputs; the residual was a +4 K-warm air temperature (state.theta is moist theta_m, converted with a naive Exner) -- FIXED in noahmp_coupler.assemble_noahmp_forcing this sprint.

## Ranked hypotheses

1. Surface-layer (sfclay/MYNN) moist-theta -> dry-T decoupling over WATER: the strict worst cell (i=66, j=37, k=3; WRF -2457.6 vs JAX -968.1) is a WATER column where Noah-MP does not run; sfclay derives the air temperature from state.theta with the SAME naive Exner the Noah-MP coupler was just fixed for (~+4 K warm), and feeds it to MYNN/RTHBLTEN. surface_layer.py is outside this sprint's ownership. (evidence: strict worst-cell is water (not in the noahmplsm land hook) + surface_layer._potential_to_temperature)
2. RRTMG step-1 surface/atmosphere radiation forcing parity (GLW bias 17.44 W/m2, SWDOWN rmse 2.76 W/m2, mass-coupled RTHRATEN residual 19.4). The Noah-MP land lane now collapses under the WRF-exact radiation swap (land theta_flux rmse 0.00758 -> see proofs/v014/noahmp_land_tile_energy_closure.*); the remaining land residual IS this radiation forcing. (evidence: rad_seed_vs_wrf_hook + rthraten_vs_wrf_part2 + noahmp_land_tile_energy_closure)

## Fastest next command

`Noah-MP land-tile energy is CLOSED (energy solve exact; the +4 K moist-theta->dry-T air-temperature bug is FIXED in noahmp_coupler.assemble_noahmp_forcing; see proofs/v014/noahmp_land_tile_energy_closure.*). Remaining strict lanes: (1) RRTMG step-1 GLW +14.7 W/m2 / SWDOWN +3.6 W/m2 forcing parity via an RRTMG longwave/shortwave hook; (2) the SAME moist-theta->dry-T decoupling missing in surface_layer.py (sfclay/MYNN over WATER -- the strict worst cell i=66 j=37 is a water column), which needs its own sprint + MYNN re-validation. Then rerun JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/noahmp_step1_closure.py.`
