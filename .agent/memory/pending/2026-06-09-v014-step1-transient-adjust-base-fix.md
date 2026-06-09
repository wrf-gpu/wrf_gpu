# V0.14 Step-1 Transient Adjust-Base Fix

Date: 2026-06-09 17:48 WEST

Sprint:
`.agent/sprints/2026-06-09-v014-step1-transient-adjust-base-fix`.

Proof:
`proofs/v014/step1_transient_adjust_base_fix.*`.

Verdict:
`STEP1_TRANSIENT_ADJUST_BASE_FIX_THETA_CLOSED`.

Source change:

- `src/gpuwrf/integration/d02_replay.py` adds
  `_wrf_live_nest_transient_adjust_mub`.
- The helper computes WRF's transient post-`blend_terrain`/pre-`start_domain`
  current `MUB` for `adjust_tempqv`.
- `_apply_live_nest_base_init` and final BaseState semantics are unchanged.

Important proof facts:

- transient adjust-base `MUB` vs WRF adjust hook delta:
  `4.521e-04 Pa`
- final BaseState `MUB` vs WRF pre-part1 final target delta:
  `4.648e-03 Pa`
- corrected theta max_abs:
  `5.788684885033035e-05 K`, below the `0.001 K` gate
- prior theta max_abs:
  `0.00541785382188209 K`
- corrected QVAPOR max_abs:
  `5.970267497393267e-08`

Manager conclusion:

The helper/proof closes the theta candidate, but production Step-1 is not fully
closed until WRF theta_m conversion plus `adjust_tempqv` is wired into the
live-nest init consumer using this transient adjust-base helper. Keep TOST,
Switzerland, FP32 source landing, and memory source work paused.
