# Worker Report

Summary: Fable/Mythos closed the first contracted lane, the NoahMP/sfclay water
path moist-theta boundary, with a production fix and WRF-anchored proof. The
strict Step-1 gate did not become green, but the blocker is now much narrower:
MYNN-EDMF `RTHBLTEN` dominates and RRTMG is secondary.

Files changed:

- `src/gpuwrf/coupling/noahmp_surface_hook.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `tests/test_v014_noahmp_surface_hook_decoupling.py`
- `proofs/v014/surface_layer_theta_decoupling.{py,json,md}`
- `proofs/v014/noahmp_step1_closure.{py,json,md}`
- `.agent/reviews/2026-06-10-v014-fable-strict-step1-closure.md`

Proof objects:

- `proofs/v014/surface_layer_theta_decoupling.json`: water HFX RMSE
  `11.869 -> 1.375 -> 0.0118 W/m2`, water `ust` near exact, theta_flux RMSE
  `0.00981 -> 8.23e-06 K m/s`.
- `proofs/v014/noahmp_step1_closure.json`: strict Step-1 improves to max_abs
  `53.52301833555157`, RMSE `2.5444971494115354`, verdict
  `NOAHMP_STEP1_STRICT_RED_SURFACE_WATERPATH_CLOSED_NARROWED_TO_MYNN_EDMF_RTHBLTEN`.

Unresolved risk: strict Step-1 remains red; no TOST or Switzerland-GPU should run
until MYNN `RTHBLTEN` is fixed or formally bounded.
