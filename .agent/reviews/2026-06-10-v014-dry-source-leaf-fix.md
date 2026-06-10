# Review: V0.14 Dry Source-Leaf Fix

Verdict: `DRY_SOURCE_LEAF_PLUMBING_ACTIVE_BUT_STEP1_T_TENDF_NOT_CLOSED`.

The production plumbing is narrow and covered by `tests/test_v014_dry_source_leaf_wiring.py`: MYNN exposes scheme-local `RTHBLTEN/RQVBLTEN`, and source mode mass-couples `RTHRATEN + RTHBLTEN` then applies WRF `conv_t_tendf_to_moist` into `DryPhysicsTendencies.t_tendf` without double-applying MYNN theta.

The Step-1 proof does not close: after-conv `T_TENDF` residual remains max_abs `2457.578397008898`, rmse `21.364579991779515`.

Next decision: Next source boundary: split MYNN PBL adapter/kernel inputs and outputs against WRF `RTHBLTEN`/`RQVBLTEN`; held `RTHRATEN` and `conv_t_tendf_to_moist` are ranked secondary by the current proof.
