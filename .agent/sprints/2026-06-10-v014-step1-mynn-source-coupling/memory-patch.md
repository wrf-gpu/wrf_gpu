Reviewer Status: ACCEPTED ROADMAP/MEMORY PATCH.

Patch intent:
Update persistent project memory and the active release checklist to record that MYNN source coupling is no longer the current broad blocker. The new active blocker is the surface/land heat-moisture flux handoff into MYNN.

Memory facts to preserve:
- Verdict: `STEP1_MYNN_SOURCE_COUPLING_NARROWED_TO_SURFACE_LAND_FLUX_HANDOFF`.
- Strict after-conv `T_TENDF` remains red: max_abs `438.5379097262689`, RMSE `5.4654420375782955`.
- WRF MYNN inputs plus WRF initialized QKE exonerate MYNN raw source units: raw `RTHBLTEN` max_abs `0.00026206000797283305`, RMSE `2.5971191677632803e-06`, corr `0.9999580118448544`.
- WRF SFCLAY1D_mynn -> MYNN-driver handoff: UST max_abs `4.998779168374767e-12`, HFX max_abs `277.80298614281253`, QFX max_abs `1.4684322196e-05`.

Next-manager instruction:
Do not run TOST, Switzerland, broad FP32, or long GPU validation yet. Open the surface/land flux handoff sprint first. Fable/Mythos is not needed unless two focused GPT/debug attempts fail to locate or fix the flux handoff.
