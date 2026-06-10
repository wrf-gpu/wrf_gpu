# V0.14 Step-1 Surface/Land Flux Handoff

- status: `PROOF_EXECUTED`
- verdict: `STEP1_SURFACE_LAND_FLUX_HANDOFF_NARROWED_TO_JAX_NOAHMP_DISABLED_CONFIGURATION`
- WRF handoff: `SFCLAY1D_mynn output == PRE_NOAHMP`; `POST_NOAHMP == MYNN driver input`.
- WRF NoahMP overlay is the exact HFX/QFX change point; post-surface finalization does not further change HFX/QFX in this fixture.
- JAX Step-1 config: `use_noahmp=False`, `sf_surface_physics=None`, `inputs_have_noahmp_land=False`.

## Key Metrics

- SFCLAY -> PRE_NOAHMP HFX max_abs: `5.000003966415534e-09`
- SFCLAY -> PRE_NOAHMP QFX max_abs: `4.999459505238002e-16`
- PRE_NOAHMP -> POST_NOAHMP HFX max_abs: `277.80298614000003`
- PRE_NOAHMP -> POST_NOAHMP QFX max_abs: `1.4684322196e-05`
- POST_NOAHMP -> MYNN HFX max_abs: `0.0`
- POST_NOAHMP -> MYNN QFX max_abs: `0.0`
- POST_NOAHMP -> MYNN UST max_abs: `0.0`
- prior strict after-conv `T_TENDF` max_abs: `438.5379097262689`, RMSE: `5.4654420375782955`

## Blocker

The blocker is now narrower than the surface/land flux handoff: the WRF handoff itself is closed to the MYNN-driver input, but the JAX Step-1 path is built with NoahMP disabled/missing land state.

Fastest next command after wiring NoahMP land/static into the Step-1 builder:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_mynn_source_coupling.py
```
