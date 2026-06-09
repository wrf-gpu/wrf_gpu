# V0.14 Step-1 Live-Nest Theta/QV Production Wiring

Verdict: `STEP1_LIVE_NEST_THETA_QV_WIRING_INIT_CLOSED_NEXT_FIELD`.

## Result

- CPU-only proof; GPU used: `False`.
- Required ancestor `a8f5c485` present: `True`.
- Production source diff is the single allowed file: `True` (+173/-1).
- `build_replay_case` live-nest branch calls transient-MUB helper: `True`, adjust_tempqv helper: `True`, resolves use_theta_m: `True`, uses corrected qv: `True`.
- Resolved `use_theta_m` for d02: `1`.
- Harness mirror vs direct helper output: theta max_abs `0.0`, qv max_abs `0.0`.

## MUB Surfaces (target cell Fortran 18,10)

| Surface | MUB | WRF target | Delta | Within tol |
|---|---:|---:|---:|:--:|
| Transient adjust-base (adjust_tempqv) | 86812.250452110 | 86812.250000000 | 4.521e-04 | True |
| Final BaseState (post start_domain) | 86794.574960129 | 86794.570312500 | 4.648e-03 | True |
- Final BaseState MUB full-domain vs WRF pre-call: max_abs `0.05002361937658861`, rmse `0.008025019829604947` (domain within tol `True`).

## Production theta/QV init vs same-boundary WRF pre-call truth

- Corrected theta (production helper): max_abs `5.788684885033035e-05`, rmse `1.5530104194219152e-05`, p99 `3.699252157858269e-05`, p99.9 `4.657611257400175e-05`.
- Raw (un-adjusted) theta: max_abs `5.490173101425171`, rmse `1.9175184863907806`.
- Closure ratio raw/corrected max_abs: `94843.18477276794`.
- Corrected QVAPOR vs WRF pre-call: max_abs `5.970267497393267e-08`, rmse `4.714587793401452e-09`.
- Theta init closes to 0.001 K gate: `True`.

## Target Cell (Fortran 18,10,2)

- WRF T_STATE `-0.24365234375`; corrected `-0.24366182404384062` (delta `-9.480e-06`); raw `-3.635589599609375` (delta `-3.392e+00`).
- WRF QVAPOR `0.007874930277466774`; corrected `0.007874929777307155` (delta `-5.002e-10`).

## Step-1 same-input 16-field comparison (next grid-parity step)

- Comparison status: `COMPARISON_EXECUTED`.
- First divergent field (schema order): `T`.
- Largest residual field: `P`.
- Largest residual `P`: max_abs `974.9820434775493`, rmse `135.98147360593399`, worst Fortran `{'i': 1, 'j': 30, 'k': 1}`, boundary band `True`.
- Ranked top-5 residuals:
  - `P` max_abs `974.9820434775493` rmse `135.98147360593399`
  - `PH` max_abs `67.3623167023926` rmse `17.41457632317551`
  - `MU` max_abs `14.125275642998986` rmse `0.9041845253003058`
  - `W` max_abs `2.640715693903735` rmse `0.4497356728466974`
  - `U` max_abs `0.7835467705023085` rmse `0.02056418486197857`

## Handoff

objective: wire WRF theta_m + adjust_tempqv into production live-nest child init and run the next Step-1 grid-parity comparison.

files changed:
- `src/gpuwrf/integration/d02_replay.py` (added `_wrf_use_theta_m`, `_wrf_live_nest_adjust_tempqv`; wired both helpers into `build_replay_case` live-nest branch)
- `proofs/v014/step1_live_nest_init_rerun.py` (proof-local mirror now applies the production theta/qv adjustment)
- `proofs/v014/step1_live_nest_theta_qv_wiring.{py,json,md}`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-qv-wiring.md`

commands run:
- `python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `python -m json.tool proofs/v014/step1_live_nest_theta_qv_wiring.json >/tmp/step1_live_nest_theta_qv_wiring.validated.json`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_m7_l2_d02_replay.py tests/test_m6x_d02_boundary_replay.py tests/test_m6x_d02_replay_hang_debug.py`
- `git diff --stat`

unresolved risks:
- Production live-nest theta_m + adjust_tempqv init closes vs WRF pre-call truth (theta max_abs 5.788684885033035e-05 K, qv max_abs 5.970267497393267e-08).
- Step-1 16-field comparison still divergent; first divergent (schema order) = T; largest residual = P max_abs 974.9820434775493.
- build_replay_case calls State.zeros (GPU-only), so the CPU proof exercises the exact production helpers it consumes plus a static wiring check, not the full GPU build_replay_case object.
- The Step-1 16-field comparison is post-RK/pre-halo; residuals after init closure name a field-level symptom (next operator), not yet the exact dycore/physics operator.

next decision needed: Run the next operator-localization sprint at Step-1 field P (worst cell {'i': 1, 'j': 30, 'k': 1}, boundary band True).
