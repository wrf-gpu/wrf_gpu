# Memory Patch: V0.14 Step-1 Transient Adjust-Base Fix

Date: 2026-06-09

Reviewer Status: APPROVED_FOR_PENDING_MEMORY

Record this sprint as closed with verdict
`STEP1_TRANSIENT_ADJUST_BASE_FIX_THETA_CLOSED`.

Source change:

- `src/gpuwrf/integration/d02_replay.py` adds
  `_wrf_live_nest_transient_adjust_mub`.
- It exposes WRF's transient post-`blend_terrain`/pre-`start_domain` current
  `MUB` for `adjust_tempqv`.
- Final post-`start_domain` BaseState semantics are unchanged.

Proof results:

- transient adjust-base `MUB` vs WRF adjust hook delta:
  `4.521e-04 Pa`
- final BaseState `MUB` vs WRF pre-part1 final target delta:
  `4.648e-03 Pa`
- corrected theta max_abs:
  `5.788684885033035e-05 K`
- prior theta max_abs:
  `0.00541785382188209 K`
- corrected QVAPOR max_abs:
  `5.970267497393267e-08`

Next:

Open a wiring sprint for WRF theta_m conversion plus `adjust_tempqv` in the
production live-nest init consumer, using the new transient adjust-base helper.
