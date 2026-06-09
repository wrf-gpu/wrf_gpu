# V0.14 Step-1 Transient Adjust-Base MUB Fix

Verdict: `STEP1_TRANSIENT_ADJUST_BASE_FIX_THETA_CLOSED`.

## Result

- CPU-only proof; GPU used: `False`.
- Required ancestor `43173cb2` present: `True`.
- Source diff (src/gpuwrf) additive only: `True` (+89/-0).
- New source helper: `gpuwrf.integration.d02_replay._wrf_live_nest_transient_adjust_mub`.
- Target zero `{'k': 1, 'y': 9, 'x': 17}`, Fortran `{'i': 18, 'j': 10, 'k': 2}`.

## MUB Surfaces (target cell)

| Surface | MUB | WRF target | Delta | Within tol |
|---|---:|---:|---:|:--:|
| Transient adjust-base (adjust_tempqv) | 86812.250452110 | 86812.250000000 | 4.521e-04 | True |
| Final BaseState (post start_domain) | 86794.574960129 | 86794.570312500 | 4.648e-03 | True |
- Transient minus final-base MUB: `17.675492` Pa (two distinct legitimate WRF base surfaces).
- Final BaseState MUB full-domain vs WRF pre-call: max_abs `0.05002361937658861`, rmse `0.008025019829604947`.

## Corrected theta/QV vs same-boundary WRF pre-call truth

- Corrected theta (transient MUB): max_abs `5.788684885033035e-05`, rmse `1.5530104194219152e-05`, p99 `3.699252157858269e-05`, p99.9 `4.657611257400175e-05`.
- Prior theta (final-base MUB): max_abs `0.00541785382188209`, rmse `5.068868142015466e-05`.
- Closure ratio prior/corrected max_abs: `93.593863364203`.
- Corrected boundary band (`<= 5`): max_abs `5.788684885033035e-05`, rmse `1.561905054270297e-05`.
- Corrected interior (`> 5`): max_abs `5.762945258425134e-05`, rmse `1.5501355247143662e-05`.
- Corrected QVAPOR vs WRF pre-call: max_abs `5.970267497393267e-08`, rmse `4.714587793408254e-09`.
- Material gate: `0.001` K.

## Target Cell (Fortran 18,10,2)

- WRF T_STATE `-0.24365234375`; corrected `-0.24366182404384062` (delta `-9.480e-06`); prior `-0.2382344899281179` (delta `5.418e-03`).
- Corrected p_new `92686.185658` vs prior p_new `92668.693493`.
- WRF QVAPOR `0.007874930277466774`; corrected `0.007874929777307155` (delta `-5.002e-10`).

## Corrected worst cell

- Zero index `{'k': 42, 'y': 61, 'x': 48}`, Fortran `{'i': 49, 'j': 62, 'k': 43}`; boundary band `True`.
- WRF `180.8074798583984`, corrected `180.80742197154956`, delta `-5.788684885033035e-05` (prior delta `-5.788684885033035e-05`).

## Handoff

objective: add the smallest production-source path exposing the WRF transient post-blend adjust-base MUB and rerun the Step-1 theta/QV candidate with it.

files changed:
- `src/gpuwrf/integration/d02_replay.py` (added `_wrf_live_nest_transient_adjust_mub`)
- `proofs/v014/step1_transient_adjust_base_fix.py`
- `proofs/v014/step1_transient_adjust_base_fix.json`
- `proofs/v014/step1_transient_adjust_base_fix.md`
- `.agent/reviews/2026-06-09-v014-step1-transient-adjust-base-fix.md`

commands run:
- `python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/step1_transient_adjust_base_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_transient_adjust_base_fix.py`
- `python -m json.tool proofs/v014/step1_transient_adjust_base_fix.json >/tmp/step1_transient_adjust_base_fix.validated.json`
- `git diff --stat`

unresolved risks:
- The corrected theta/QV candidate is validated in this CPU proof; wiring theta_m+adjust_tempqv into the production live-nest init consumer is a separate, larger grid-parity step.

next decision needed: Wire WRF theta_m conversion + adjust_tempqv (with the transient adjust-base MUB) into the production live-nest init consumer of _apply_live_nest_base_init, then run the next larger grid-parity step: the full step-1 same-input d02 comparison (step1_live_nest_init_rerun) across all 16 fields.
