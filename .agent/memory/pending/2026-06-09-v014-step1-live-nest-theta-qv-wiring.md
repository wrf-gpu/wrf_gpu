# V0.14 Step-1 Live-Nest Theta/QV Wiring

Date: 2026-06-09 18:17 WEST

Sprint:
`.agent/sprints/2026-06-09-v014-step1-live-nest-theta-qv-wiring`.

Proof:
`proofs/v014/step1_live_nest_theta_qv_wiring.*`.

Verdict:
`STEP1_LIVE_NEST_THETA_QV_WIRING_INIT_CLOSED_NEXT_FIELD`.

Source change:

- `src/gpuwrf/integration/d02_replay.py` now wires WRF live-nest
  `USE_THETA_M=1` theta conversion plus `adjust_tempqv` into
  `build_replay_case(..., live_nest_parent=...)`.
- The adjustment uses WRF's transient post-`blend_terrain`/pre-`start_domain`
  current `MUB` for `adjust_tempqv`.
- Final post-`start_domain` `PB/PHB/MUB` BaseState is unchanged.
- The change is initialization-only and adds no timestep-loop host/device
  transfer.

Important proof facts:

- corrected theta vs same-boundary WRF pre-call truth:
  `5.788684885033035e-05 K`
- corrected QVAPOR vs WRF pre-call truth:
  `5.970267497393267e-08`
- helper-vs-proof-harness theta and QVAPOR deltas:
  `0.0`
- final BaseState `MUB` full-domain max_abs vs WRF pre-part1:
  `0.05002361937658861 Pa`
- Step-1 is still red: first divergent schema field `T`, largest residual
  `P` max_abs `974.9820434775493`, worst Fortran `i=1,j=30,k=1`, boundary band
  true.

Manager conclusion:

The live-nest theta/QV initialization defect is closed in production. Continue
the v0.14 grid-parity ladder at Step-1 `P/PH/MU` boundary/operator
localization; keep TOST, Switzerland validation, FP32 source work, and memory
follow-ups paused until the remaining grid divergence is explained and reduced.
