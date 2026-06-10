# V0.14 Step-1 surface/land flux handoff review

- verdict: `STEP1_SURFACE_LAND_FLUX_HANDOFF_NARROWED_TO_JAX_NOAHMP_DISABLED_CONFIGURATION`
- production code changes: none
- WRF hook patch archived: `proofs/v014/step1_surface_land_flux_handoff_wrf_patch.diff`
- proof script: `proofs/v014/step1_surface_land_flux_handoff.py`

Evidence:
- SFCLAY -> PRE_NOAHMP HFX max_abs `5.000003966415534e-09`.
- PRE_NOAHMP -> POST_NOAHMP HFX max_abs `277.80298614000003`.
- POST_NOAHMP -> MYNN HFX max_abs `0.0`.

Unresolved risk:
- Strict Step-1 `T_TENDF` remains red until the JAX Step-1 builder/source capture carries WRF-derived NoahMP land/static state and enables `sf_surface_physics=4`.
