# Memory Patch: V0.14 Step-1 Live-Nest Theta/QV Wiring

Date: 2026-06-09

Reviewer Status: APPROVED_FOR_PENDING_MEMORY.

Scope:

- Pending memory only. Do not edit stable memory from this sprint.

Reason:

- Production live-nest init is now wired to WRF `USE_THETA_M=1` theta conversion
  plus `adjust_tempqv`.
- The sprint closes the theta/QV initialization defect against same-boundary WRF
  pre-call truth, but does not close the full Step-1 comparison.

Evidence:

- `proofs/v014/step1_live_nest_theta_qv_wiring.json`
- `proofs/v014/step1_live_nest_theta_qv_wiring.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-qv-wiring.md`

Proposed Destination:

- `.agent/memory/pending/2026-06-09-v014-step1-live-nest-theta-qv-wiring.md`

Approved pending memory:

- `src/gpuwrf/integration/d02_replay.py` now applies WRF live-nest theta/QV
  adjustment before `State` construction in the `live_nest_parent` branch.
- Corrected theta max_abs is `5.788684885033035e-05 K`; corrected QVAPOR max_abs
  is `5.970267497393267e-08`.
- Final BaseState `PB/PHB/MUB` remains the post-`start_domain` runtime base.
- Full Step-1 remains red: first divergent schema field `T`, largest residual
  `P` max_abs `974.9820434775493`; next sprint is Step-1 `P/PH/MU`
  boundary/operator localization.
