# V0.14 NoahMP Step-1 Closure

Verdict: `NOAHMP_STEP1_STRICT_RED_FORMALLY_BOUNDED_RRTMG_FIELD_DOMINANT_MYNN_MAX_FLOOR`.

## Configuration (was the contract blocker)

- `use_noahmp=True`, `sf_surface_physics=4`, `inputs_have_noahmp_land=True` (previously False/None/False).
- WRF clock: julian `120.75` (0-based fractional), yearlen `365.0`.
- `topo_shading=1`, `slope_rad=1`, radiation_static loaded `True`.

## Truth provenance

- Strict target re-emitted from ONE run of the rmol-PINNED WRF binary; byte-identical across re-runs and across two pinned builds (`{"mynn_vs_surfacehandoff_run": true, "mynn_vs_rmolpin_build": true, "surface_vs_surfacehandoff_run": true}`).

## Strict gate

- after-conv `T_TENDF` vs JAX dry: max_abs `55.92970981221765`, RMSE `0.499664626865853` (pass: max_abs <= 0.001, RMSE <= 1e-05).
- worst cell Fortran `{'i': 20, 'j': 7, 'k': 2}`: WRF `-1278.747436523438` vs JAX `-1222.8177267112203`.

## Surface-layer water-path closure (this sprint)

- The Noah-MP/sfclay column view (`coupling.noahmp_surface_hook._build_column_view`) now supplies the WRF `phy_prep` dry `t_air`, true `psfc`, and density (mirroring `physics_couplers._surface_column_view`), threaded via `noahmp_surface_step(grid=...)`. Previously it fed the surface layer raw moist `theta_m` with a naive Exner (~+4 K warm) and the air-pressure/ideal-gas fallback, corrupting the WATER-column sfclay flux that MYNN consumes.
- Effect (proofs/v014/surface_layer_theta_decoupling.*): water HFX rmse 11.87->0.012 W/m2, ust ~exact; strict max_abs 1489.5->55.92970981221765, rmse 12.15->0.499664626865853. The remaining residual is MYNN-EDMF RTHBLTEN (land+water), not the surface coupling.

## WRF-anchored boundary measurements

- Noah-MP forcing seed (lead=radt/2): SOLDN vs WRF SWDOWN all rmse `2.736575356942205` W/m2; LWDN vs WRF GLW bias `0.3188714722687933` W/m2.
- lead=0 contrast SWDOWN rmse `56.47144216390679` W/m2 (falsifies the lead-0 seed convention).
- mass-coupled RTHRATEN vs WRF part2 (interior): max_abs `2.798351397503893`, rmse `0.36457296575368353` (WRF field max `41.89738082885742`).
- post-overlay MYNN boundary theta_flux land: max_abs `0.00412322915865182`, bias `0.0009508037674703292` (K m/s).
- raw RTHBLTEN vs WRF: max_abs `0.0026920951794989`, rmse `9.611914804026008e-06`, strong-median `1.0020295551690332`, corr `0.998999093941308`.

## Causal split (radiation-swap) + land input parity

- land theta_flux residual with the JAX seed radiation: `{"count": 768, "max_abs": 0.0041215554145404565, "rmse": 0.0010475626324635409, "bias": 0.0009533186267807948, "ref_max_abs": 0.3119690397296882}`.
- land theta_flux residual with WRF's EXACT hook SWDOWN/GLW: `{"count": 768, "max_abs": 0.00044929253859332663, "rmse": 0.00012190445888234624, "bias": 8.957439124293328e-05, "ref_max_abs": 0.3119690397296882}` -> the WRF-exact radiation swap reduces the land residual strongly, so the remaining land-forcing error is radiation-sensitive. See `proofs/v014/noahmp_land_tile_energy_closure.*`.
- The prior 'NoahMP land-tile energy' narrowing is REFUTED: the energy solve is exact to ~1e-3 W/m2 with WRF-exact inputs; the residual was a +4 K-warm air temperature (state.theta is moist theta_m, converted with a naive Exner) -- FIXED in noahmp_coupler.assemble_noahmp_forcing this sprint.

## Ranked hypotheses

1. MYNN-EDMF RTHBLTEN PBL theta-tendency kernel residual (DOMINANT). With the surface-layer water-path fix landed (sfclay/Noah-MP now receive the WRF phy_prep dry t_air + true psfc + density; water HFX rmse 11.87->0.012 W/m2, ust ~exact -- see proofs/v014/surface_layer_theta_decoupling.*), the strict residual collapsed 1489.5->55.9 max_abs / 12.15->0.5 rmse. The remaining worst cells are RTHBLTEN-dominated on BOTH land and water (worst WRF -1278.7 vs JAX -1222.8; ~4-7% of the local RTHBLTEN where it is large), with RTHRATEN <=~2.8. This is inside module_bl_mynnedmf (mixing length / EDMF mass-flux / cold-start qke), NOT the surface coupling (now WRF-faithful) and NOT radiation. MYNN kernel is outside this sprint's file ownership. (evidence: land/water strict decomposition (worst-cell RTHBLTEN-dominated, RTHRATEN<=~2.8) + surface_layer_theta_decoupling + post_overlay_mynn_boundary)
2. RRTMG step-1 radiation forcing parity (SECONDARY): GLW bias 0.32 W/m2, SWDOWN rmse 2.74 W/m2, mass-coupled RTHRATEN residual 2.8. The Noah-MP LAND theta_flux still collapses under the WRF-exact radiation swap (rmse 0.00105); RRTMG is now materially reduced by the dry-theta input fix and remains a bounded split residual (proofs/v014/rrtmg_step1_forcing_parity.*), not the worst-cell max owner (RTHRATEN max 2.8 << strict max 55.9). (evidence: rad_seed_vs_wrf_hook + rthraten_vs_wrf_part2 + rrtmg_step1_forcing_parity)

## Authoritative lane decomposition (supersedes the max-only ranking above)

proofs/v014/mynn_rthblten_step1_closure.{py,json,md} -- current post-dry-theta-fix operational-path decomposition. FINDINGS: operational strict max/rmse 55.93/0.4997; WRF-pinned QKE 29.2/0.4582; WRF-pinned QKE + WRF RTHRATEN 29.42/0.277; RRTMG lane only 2.839/0.3648; remaining RRTMG share of WRF-QKE rmse variance 63.5%. The 1e-3/1e-5 mass-coupled strict gate remains unreachable without bitwise MYNN+RRTMG reproduction; use operational field/rollout gates for release decisions.

## Fastest next command

`Surface-layer water-path CLOSED (proofs/v014/surface_layer_theta_decoupling.*). Post-RRTMG-fix strict RED at max 55.92970981221765 / rmse 0.499664626865853 is FORMALLY BOUNDED + GATE-UNREACHABLE (see proofs/v014/mynn_rthblten_step1_closure.*). RRTMG forcing is materially reduced to RTHRATEN max 2.798 / rmse 0.3646; the remaining strict max is MYNN level-2.5/QKE floor, and remaining field rmse must be assessed with operational field/rollout gates rather than the bitwise MYNN+RRTMG strict tolerance. Manager decision: re-specify the strict MYNN+RRTMG gate to an operational mass-coupled tolerance. Re-run: JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/noahmp_step1_closure.py.`
