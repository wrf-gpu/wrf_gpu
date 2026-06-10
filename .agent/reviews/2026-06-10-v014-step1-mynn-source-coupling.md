# Review: V0.14 Step-1 MYNN Source Coupling

Verdict: `STEP1_MYNN_SOURCE_COUPLING_NARROWED_TO_SURFACE_LAND_FLUX_HANDOFF`.

The production adapter fixes are scoped and tested, but strict Step-1 is not closed.
Current after-conv `T_TENDF`: max_abs `438.5379097262689`, RMSE `5.4654420375782955`.

MYNN kernel/source units are not the primary blocker: WRF inputs + WRF QKE raw `RTHBLTEN` max_abs `0.00026206000797283305`, RMSE `2.5971191677632803e-06`.
The narrower blocker is the WRF surface/land flux handoff into MYNN: driver-vs-SFCLAY HFX max_abs `277.80298614281253`.

Next: Add a WRF hook immediately before/after module_surface_driver's sf_surface_physics=4 land-surface flux update (HFX/QFX/LH/TSK/GRDFLX/diagnostic CH where available), then compare it to the JAX Step-1 path and wire the Noah-MP/land flux overlay into the MYNN bottom-boundary handles before rerunning this proof.
