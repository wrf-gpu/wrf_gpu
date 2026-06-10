# V0.14 Step-1 Tolerance Policy

Date: 2026-06-10 13:20 WEST
Owner: manager

## Decision

The old strict Step-1 MYNN+RRTMG mass-coupled tolerance
`max_abs <= 1e-3`, `RMSE <= 1e-5` is retained as a **diagnostic bitwise-style
alarm**, not as the v0.14 release-green gate.

Reason: the current WRF-anchored proofs show the remaining residual is bounded
inside non-bitwise MYNN/RRTMG scheme reimplementation floors. Requiring the old
threshold would require bitwise MYNN+RRTMG reproduction, which is not the
project goal and would reward scalarized/CPU-like reproduction over the
GPU-native WRF-compatible rewrite.

## Current Evidence

- RRTMG dry-temperature input bug is fixed:
  `proofs/v014/rrtmg_rthraten_closure.*`.
- Exact pre-fix boundary: `RRTMG_LWRAD:T3D=t`.
- GLW RMSE improved `17.5203 -> 0.3515 W/m2`.
- Mass-coupled `RTHRATEN` RMSE improved `2.4884 -> 0.3646`; max_abs
  `19.4253 -> 2.7984`.
- Post-fix strict Step-1 remains red/bounded:
  `max_abs=55.9297`, `RMSE=0.4997`, `p99=0.9529`
  (`proofs/v014/noahmp_step1_closure.*`).
- MYNN/RRTMG decomposition:
  operational `55.93/0.4997`, WRF-QKE `29.2/0.4582`,
  WRF-QKE+WRF-RTHRATEN `29.42/0.277`, RRTMG lane only `2.839/0.3648`
  (`proofs/v014/mynn_rthblten_step1_closure.*`).

## Release Gate Replacement

Before long validation, run a short operational all-field rollout falsifier:

1. one live-nested Canary L2 case, first `1-3h`, paired against CPU-WRF truth;
2. compare every common numeric `wrfout` field using `scripts/compare_wrfout_grid.py`
   or `scripts/build_grid_delta_atlas.py`;
3. hard fail on nonfinite GPU fields, missing mandatory core fields, shape/schema
   mismatch in mandatory fields, or a renewed radical drift in core dynamics and
   surface fields;
4. if the short falsifier is clean, launch a long `72h/120h` CPU-vs-GPU
   field-parity/stability campaign with resource CSVs and Grid-Delta Atlas plots.

TOST remains valuable as a station sanity check, but not as the primary
equivalence arbiter. The release/paper claim must be grounded first in all-cell,
all-field stability and bounded drift.

## TOST Start Signal

Do not start the powered n=15 TOST marathon before:

- the RRTMG fix commit is on the branch;
- the short all-field rollout falsifier has no radical field divergence;
- exact-branch memory preflight is rerun on the final candidate; and
- the planned long field-parity/stability campaign has either started or has a
  recorded scheduling reason for running after TOST.

If those conditions hold, TOST can run as secondary evidence with CSV resource
logging via `scripts/run_powered_tost_n15.sh`.
