# V0.14 Step-1 Surface/Land Flux Handoff

- status: `PROOF_EXECUTED`
- verdict: `STEP1_SURFACE_LAND_FLUX_HANDOFF_CLOSED_JAX_NOAHMP_ENABLED`
- WRF handoff: `SFCLAY1D_mynn output == PRE_NOAHMP`; `POST_NOAHMP == MYNN driver input`.
- WRF NoahMP overlay is the exact HFX/QFX change point; post-surface finalization does not further change HFX/QFX in this fixture.
- JAX Step-1 config: `use_noahmp=True`, `sf_surface_physics=4`, `inputs_have_noahmp_land=True`.

## Key Metrics

- SFCLAY -> PRE_NOAHMP HFX max_abs: `5.000003966415534e-09`
- SFCLAY -> PRE_NOAHMP QFX max_abs: `4.999459505238002e-16`
- PRE_NOAHMP -> POST_NOAHMP HFX max_abs: `277.80298614000003`
- PRE_NOAHMP -> POST_NOAHMP QFX max_abs: `1.4684322196e-05`
- POST_NOAHMP -> MYNN HFX max_abs: `0.0`
- POST_NOAHMP -> MYNN QFX max_abs: `0.0`
- POST_NOAHMP -> MYNN UST max_abs: `0.0`
- prior strict after-conv `T_TENDF` max_abs: `1489.5135568470864`, RMSE: `13.2001844004901`

## Blocker

CLOSED: the JAX Step-1 builder now carries WRF-derived NoahMP land/static state with sf_surface_physics=4 + use_noahmp=True; the remaining gate is the strict Step-1 metric in step1_mynn_source_coupling / noahmp_step1_closure.

Fastest next command:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_mynn_source_coupling.py
```
