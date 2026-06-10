# V0.14 Step-1 MYNN Source Coupling

Verdict: `STEP1_STRICT_NOT_CLOSED_AFTER_NOAHMP_ENABLEMENT_SEE_NOAHMP_STEP1_CLOSURE`.

## Result

- Strict after-conv `T_TENDF` is still red: max_abs `1489.5135568470864`, RMSE `13.2001844004901`.
- WRF inputs + WRF initialized QKE exonerate the MYNN kernel/source units: raw `RTHBLTEN` max_abs `0.00026978377168347277`, RMSE `2.5913062928007185e-06`, corr `0.9999269219802973`.
- Current JAX source leaves remain divergent: raw `RTHBLTEN` max_abs `0.016072531466352954`, RMSE `0.00011917325735225103`, strong median `0.41187007310784046`.
- Even with WRF QKE injected, current inputs retain a source tail: max_abs `0.016064435310936355`, RMSE `0.00011906803622303894`.

## WRF Surface/Land Flux Handoff (anchor facts)

WRF changes heat/moisture fluxes between `SFCLAY1D_mynn` output and the MYNN driver input via the Noah-MP land overlay; the JAX Step-1 path now mirrors this (`use_noahmp=True`, `sf_surface_physics=4`).

- WRF MYNN-driver `UST` vs WRF `SFCLAY1D_mynn` `UST`: max_abs `4.998779168374767e-12`.
- WRF MYNN-driver `HFX` vs WRF `SFCLAY1D_mynn` `HFX`: max_abs `277.80298614281253`, RMSE `23.78077473308822`.
- WRF MYNN-driver `QFX` vs WRF `SFCLAY1D_mynn` `QFX`: max_abs `1.4684322196e-05`, RMSE `1.0634310887382864e-06`.

## Production Fixes

- MYNN grid-backed columns now use WRF `phy_prep` dry theta, hydrostatic pressure, rho, and physics-g dz.
- MYNN dry-theta output is converted back to live theta_m state; raw source leaves stay dry theta.
- First-step MYNN QKE initialization is ordered after surface fluxes in the operational MYNN slot.
- Step-1 builder now activates Noah-MP (`sf_surface_physics=4`) with WRF-derived land/static state, WRF clock, topo_shading/slope_rad, and WRF-faithful held radiation seeds (see `proofs/v014/noahmp_step1_closure.py`).

## Fastest Next Command

`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/noahmp_step1_closure.py`

## Files

- JSON proof: `/home/enric/src/wrf_gpu2/proofs/v014/step1_mynn_source_coupling.json`
- WRF patch archive: `/home/enric/src/wrf_gpu2/proofs/v014/step1_mynn_source_coupling_wrf_patch.diff`
- Review: `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-10-v014-step1-mynn-source-coupling.md`
