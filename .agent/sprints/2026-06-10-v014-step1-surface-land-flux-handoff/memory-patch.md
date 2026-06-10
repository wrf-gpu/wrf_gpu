Reviewer Status: ACCEPTED ROADMAP/MEMORY PATCH.

Patch intent:
Update persistent project memory and the active release checklist to record that
the surface/land flux handoff sprint is closed as a strict narrowing. The active
grid-parity blocker is now JAX Step-1 NoahMP disabled/missing land/static state,
not an unknown WRF handoff between surface layer and MYNN.

Memory facts to preserve:
- Verdict:
  `STEP1_SURFACE_LAND_FLUX_HANDOFF_NARROWED_TO_JAX_NOAHMP_DISABLED_CONFIGURATION`.
- WRF `SFCLAY1D_mynn` output equals `PRE_NOAHMP` for HFX/QFX within roundoff.
- WRF NoahMP overlay is the exact HFX/QFX change point:
  HFX max_abs `277.80298614000003`, QFX max_abs `1.4684322196e-05`.
- WRF `POST_NOAHMP` equals MYNN driver input for HFX/QFX/UST, max_abs `0.0`.
- JAX Step-1 config reports `use_noahmp=False`, `sf_surface_physics=None`, and
  no NoahMP land/static state.
- Strict after-conv `T_TENDF` remains red at max_abs `438.5379097262689`,
  RMSE `5.4654420375782955`.

Next-manager instruction:
Do not open another GPT micro-sprint for this same issue. Send Fable/Mythos one
complete endpoint-defined task after `/compact`: wire/fix the WRF-derived NoahMP
land/static/radiation state in Step-1 production/proof paths and prove strict
Step-1, or produce an exact narrower blocker. Fable is scarce; require a proof
object and manager-rerunnable gates before merge.
