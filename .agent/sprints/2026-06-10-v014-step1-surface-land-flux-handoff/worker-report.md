Summary: GPT-5.5 xhigh completed the Step-1 surface/land flux handoff sprint as
a strict narrowing sprint, not a production-fix sprint.

Objective:
Find where WRF changes heat/moisture fluxes between `SFCLAY1D_mynn` output and
the MYNN driver input, compare that boundary to the JAX Step-1 path, and fix the
JAX path if the bug was local and safe.

Files changed:
- `proofs/v014/step1_surface_land_flux_handoff.py`
- `proofs/v014/step1_surface_land_flux_handoff.json`
- `proofs/v014/step1_surface_land_flux_handoff.md`
- `proofs/v014/step1_surface_land_flux_handoff_wrf_patch.diff`
- `.agent/reviews/2026-06-10-v014-step1-surface-land-flux-handoff.md`
- refreshed metadata in `proofs/v014/step1_mynn_source_coupling.json`

Result:
The sprint proved WRF's flux handoff itself is closed after the NoahMP surface
overlay. `SFCLAY1D_mynn` output equals `PRE_NOAHMP`, `PRE_NOAHMP` to
`POST_NOAHMP` is the exact HFX/QFX change point, and `POST_NOAHMP` equals MYNN
driver input for HFX/QFX/UST. The JAX Step-1 source-capture path currently
reports `use_noahmp=False`, `sf_surface_physics=None`, and no NoahMP land/static
state, so the remaining blocker is now the JAX Step-1 NoahMP configuration and
land-state wiring.

Key evidence:
- `SFCLAY -> PRE_NOAHMP` HFX max_abs `5.000003966415534e-09`.
- `PRE_NOAHMP -> POST_NOAHMP` HFX max_abs `277.80298614000003`.
- `PRE_NOAHMP -> POST_NOAHMP` QFX max_abs `1.4684322196e-05`.
- `POST_NOAHMP -> MYNN` HFX/QFX/UST max_abs `0.0`.
- Prior strict after-conv `T_TENDF` remains red at max_abs
  `438.5379097262689`, RMSE `5.4654420375782955`.

Next:
Escalate the whole remaining NoahMP/land-state Step-1 closure to Fable/Mythos as
one endpoint-defined sprint: strict Step-1 green if possible, or exact narrower
blocker with proof.
