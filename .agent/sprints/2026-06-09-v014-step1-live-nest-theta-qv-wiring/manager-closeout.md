# Manager Closeout: V0.14 Step-1 Live-Nest Theta/QV Wiring

Date: 2026-06-09 18:17 WEST

## Outcome

The sprint is closed with verdict
`STEP1_LIVE_NEST_THETA_QV_WIRING_INIT_CLOSED_NEXT_FIELD`.

The production live-nest child initialization now applies WRF `USE_THETA_M=1`
theta conversion plus `adjust_tempqv` using the transient post-`blend_terrain`
`MUB` surface proved in the prior sprint. This closes the theta/QV
initialization defect without changing the final post-`start_domain` BaseState.

## Proof Objects

- `src/gpuwrf/integration/d02_replay.py`
- `proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `proofs/v014/step1_live_nest_theta_qv_wiring.json`
- `proofs/v014/step1_live_nest_theta_qv_wiring.md`
- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_init_rerun.json`
- `proofs/v014/step1_live_nest_init_rerun.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-qv-wiring.md`

Key proof metrics:

- corrected theta max_abs vs WRF pre-call truth:
  `5.788684885033035e-05 K`
- corrected QVAPOR max_abs:
  `5.970267497393267e-08`
- transient adjust-base `MUB` vs WRF hook delta:
  `4.521e-04 Pa`
- final BaseState `MUB` full-domain max_abs vs WRF pre-part1:
  `0.05002361937658861 Pa`

## Merge Decision:

Commit and push the production initialization fix and proof artifacts. Do not
claim full Step-1 closure, grid parity, TOST readiness, or validation readiness
from this sprint.

## Validation

Manager reran:

- `python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/step1_live_nest_theta_qv_wiring.py proofs/v014/step1_live_nest_init_rerun.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `python -m json.tool proofs/v014/step1_live_nest_theta_qv_wiring.json >/tmp/step1_live_nest_theta_qv_wiring.manager.validated.json`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_init_rerun.py`
- `python -m json.tool proofs/v014/step1_live_nest_init_rerun.json >/tmp/step1_live_nest_init_rerun.manager.validated.json`
- CPU-forced replay tests: `4 passed, 2 skipped`

## Scope Changes

No GPU, TOST, Switzerland, FP32 source work, memory source work, or Hermes was
used. The only production source file touched is
`src/gpuwrf/integration/d02_replay.py`.

## Next Sprint

Open a Step-1 `P/PH/MU` boundary/operator localization sprint. The next proof
must start from the now-corrected live-nest theta/QV initialization, target the
largest `P` residual (`974.9820434775493 Pa`, worst Fortran `i=1,j=30,k=1`,
boundary band true), and explain how that relates to the first divergent schema
field `T`.
