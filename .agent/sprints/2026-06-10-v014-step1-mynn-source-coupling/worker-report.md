Summary: GPT-5.5 xhigh completed the Step-1 MYNN source-coupling sprint. The sprint did not close strict Step-1, but it narrowed the blocker with WRF-anchored raw-source evidence and landed scoped adapter fixes.

Objective:
Close, or strictly narrow, the remaining Step-1 dry source divergence after the `SFCLAY1D_mynn` output algebra sprint. The leading hypothesis was MYNN/PBL source coupling after fixed surface outputs.

Files changed:
`src/gpuwrf/coupling/physics_couplers.py`, `src/gpuwrf/runtime/operational_mode.py`, `tests/test_v014_dry_source_leaf_wiring.py`, `proofs/v014/step1_mynn_source_coupling.*`, plus refreshed v0.14 proof JSON/MD/review files that were rerun under the new adapter semantics.

Result:
Verdict is `STEP1_MYNN_SOURCE_COUPLING_NARROWED_TO_SURFACE_LAND_FLUX_HANDOFF`. Strict after-conv `T_TENDF` remains red at max_abs `438.5379097262689`, RMSE `5.4654420375782955`. WRF inputs plus WRF initialized QKE exonerate MYNN source units with raw `RTHBLTEN` max_abs `0.00026206000797283305`, RMSE `2.5971191677632803e-06`, corr `0.9999580118448544`.

Narrower blocker:
WRF changes heat/moisture fluxes between `SFCLAY1D_mynn` output and the MYNN driver input: UST is closed at max_abs `4.998779168374767e-12`, but HFX has max_abs `277.80298614281253` and QFX has max_abs `1.4684322196e-05`. The next sprint should hook the surface/land flux update immediately before/after `module_surface_driver` `sf_surface_physics=4` and compare that handoff to the JAX Step-1 path.
