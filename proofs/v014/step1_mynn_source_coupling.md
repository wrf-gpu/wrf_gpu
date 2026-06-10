# V0.14 Step-1 MYNN Source Coupling

Verdict: `STEP1_MYNN_SOURCE_COUPLING_NARROWED_TO_SURFACE_LAND_FLUX_HANDOFF`.

## Result

- Strict after-conv `T_TENDF` is still red: max_abs `438.5379097262689`, RMSE `5.4654420375782955`.
- WRF inputs + WRF initialized QKE exonerate the MYNN kernel/source units: raw `RTHBLTEN` max_abs `0.00026206000797283305`, RMSE `2.5971191677632803e-06`, corr `0.9999580118448544`.
- Current JAX source leaves remain divergent: raw `RTHBLTEN` max_abs `0.007112169856768911`, RMSE `9.786127476127153e-05`, strong median `0.6971366587911589`.
- Even with WRF QKE injected, current inputs retain a source tail: max_abs `0.006557968236318549`, RMSE `6.227350537104264e-05`.

## Narrower Blocker

The leading broad MYNN source-coupling hypothesis is narrowed upstream of `module_bl_mynnedmf`: WRF changes heat/moisture fluxes between `SFCLAY1D_mynn` output and the MYNN driver input.

- WRF MYNN-driver `UST` vs WRF `SFCLAY1D_mynn` `UST`: max_abs `4.998779168374767e-12`.
- WRF MYNN-driver `HFX` vs WRF `SFCLAY1D_mynn` `HFX`: max_abs `277.80298614281253`, RMSE `23.78077473308822`.
- WRF MYNN-driver `QFX` vs WRF `SFCLAY1D_mynn` `QFX`: max_abs `1.4684322196e-05`, RMSE `1.0634310887382864e-06`.

## Production Fixes

- MYNN grid-backed columns now use WRF `phy_prep` dry theta, hydrostatic pressure, rho, and physics-g dz.
- MYNN dry-theta output is converted back to live theta_m state; raw source leaves stay dry theta.
- First-step MYNN QKE initialization is ordered after surface fluxes in the operational MYNN slot.

## Fastest Next Command

`Add a WRF hook immediately before/after module_surface_driver's sf_surface_physics=4 land-surface flux update (HFX/QFX/LH/TSK/GRDFLX/diagnostic CH where available), then compare it to the JAX Step-1 path and wire the Noah-MP/land flux overlay into the MYNN bottom-boundary handles before rerunning this proof.`

## Files

- JSON proof: `/home/enric/src/wrf_gpu2/proofs/v014/step1_mynn_source_coupling.json`
- WRF patch archive: `/home/enric/src/wrf_gpu2/proofs/v014/step1_mynn_source_coupling_wrf_patch.diff`
- Review: `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-10-v014-step1-mynn-source-coupling.md`
