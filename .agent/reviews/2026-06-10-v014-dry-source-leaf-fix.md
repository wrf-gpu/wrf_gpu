# Review: V0.14 Dry Source-Leaf Fix

Verdict: `DRY_SOURCE_LEAF_PLUMBING_ACTIVE_BUT_STEP1_T_TENDF_NOT_CLOSED`.

The production plumbing is narrow and covered by `tests/test_v014_dry_source_leaf_wiring.py`: MYNN exposes a scheme-local `RTHBLTEN`, and source mode mass-couples `RTHRATEN + RTHBLTEN` into `DryPhysicsTendencies.t_tendf` without double-applying MYNN theta.

The Step-1 proof does not close: after-conv `T_TENDF` residual remains max_abs `2457.575215120763`, rmse `21.445918959761645`.

Next decision: Next source boundary: split MYNN PBL adapter/kernel inputs and outputs against WRF `RTHBLTEN`/`RQVBLTEN`, seed or refresh held `RTHRATEN` at the same Step-1 boundary, then implement WRF `conv_t_tendf_to_moist` before feeding `DryPhysicsTendencies.t_tendf`.
