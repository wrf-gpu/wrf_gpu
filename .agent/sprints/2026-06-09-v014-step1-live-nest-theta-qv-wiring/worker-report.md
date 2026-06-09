# Worker Report: V0.14 Step-1 Live-Nest Theta/QV Wiring

Summary:

Opus wired WRF live-nest `USE_THETA_M=1` theta conversion plus
`adjust_tempqv` into the production live-nest child initialization path. The
patch keeps the final post-`start_domain` BaseState unchanged and only adjusts
the live-nest child `State.theta` and `State.qv` before `OperationalCarry`
construction.

Files Changed:

- `src/gpuwrf/integration/d02_replay.py`
- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_init_rerun.json`
- `proofs/v014/step1_live_nest_init_rerun.md`
- `proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `proofs/v014/step1_live_nest_theta_qv_wiring.json`
- `proofs/v014/step1_live_nest_theta_qv_wiring.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-qv-wiring.md`

Commands Run:

- `python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `python -m json.tool proofs/v014/step1_live_nest_theta_qv_wiring.json >/tmp/step1_live_nest_theta_qv_wiring.validated.json`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_init_rerun.py`
- `python -m json.tool proofs/v014/step1_live_nest_init_rerun.json >/tmp/step1_live_nest_init_rerun.manager.validated.json`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_m7_l2_d02_replay.py tests/test_m6x_d02_boundary_replay.py tests/test_m6x_d02_replay_hang_debug.py`

Proof Objects:

- `proofs/v014/step1_live_nest_theta_qv_wiring.*`
- `proofs/v014/step1_live_nest_init_rerun.*`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-qv-wiring.md`

Result:

Verdict is `STEP1_LIVE_NEST_THETA_QV_WIRING_INIT_CLOSED_NEXT_FIELD`.
Production wiring is statically present in `build_replay_case`, and the proof
mirrors the production helpers exactly (`theta/qv` helper-vs-harness max_abs
`0.0`).

Key metrics:

- corrected theta vs same-boundary WRF pre-call truth: max_abs
  `5.788684885033035e-05 K`
- corrected QVAPOR vs WRF pre-call truth: max_abs
  `5.970267497393267e-08`
- transient adjust-base `MUB` vs WRF adjust hook delta: `4.521e-04 Pa`
- final BaseState `MUB` domain max_abs vs WRF pre-part1: `0.05002361937658861 Pa`

Handoff:

Initialization is closed for theta/QV. The Step-1 16-field comparison remains
red: first divergent schema field `T`; largest residual field `P` max_abs
`974.9820434775493`, RMSE `135.98147360593399`, worst Fortran cell
`i=1,j=30,k=1`, boundary band true. The next sprint should localize the
remaining Step-1 `P/PH/MU` boundary/operator residual while accounting for the
small remaining `T` residual.
