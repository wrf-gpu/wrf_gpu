# Tester Report: V0.14 Step-1 Live-Nest Theta/QV Wiring

## Tests Added Or Run

The worker and manager ran CPU-only proof validation and narrow replay tests.
No GPU, TOST, Switzerland, FP32, or memory source work was run.

Commands:

- `python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/step1_live_nest_theta_qv_wiring.py proofs/v014/step1_live_nest_init_rerun.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `python -m json.tool proofs/v014/step1_live_nest_theta_qv_wiring.json >/tmp/step1_live_nest_theta_qv_wiring.manager.validated.json`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_init_rerun.py`
- `python -m json.tool proofs/v014/step1_live_nest_init_rerun.json >/tmp/step1_live_nest_init_rerun.manager.validated.json`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_m7_l2_d02_replay.py tests/test_m6x_d02_boundary_replay.py tests/test_m6x_d02_replay_hang_debug.py`

## Results

The authoritative proof verdict is
`STEP1_LIVE_NEST_THETA_QV_WIRING_INIT_CLOSED_NEXT_FIELD`.

Validation facts:

- JSON validation succeeded for both `step1_live_nest_theta_qv_wiring.json` and
  `step1_live_nest_init_rerun.json`.
- CPU replay tests passed: `4 passed, 2 skipped`.
- Proof recorded CPU-only execution and `gpu_used=false`.
- Production source diff is limited to
  `src/gpuwrf/integration/d02_replay.py`.

Closed gates:

- production live-nest theta/QV helper output matches the proof-local harness
  exactly (`0.0` max_abs for both fields);
- corrected theta closes below the `0.001 K` gate;
- QVAPOR matches same-boundary WRF pre-call truth to `5.970267497393267e-08`;
- final BaseState `MUB` guard remains within the established tolerance.

## Gaps

The full Step-1 16-field comparison is not green. The next remaining evidence
target is the boundary-band `P/PH/MU` residual, with first divergent field `T`
and largest residual `P`.

Decision:

Accept the sprint as a production initialization fix plus proof. Do not resume
TOST or validation campaigns from this artifact; continue the Step-1
operator-localization chain.
