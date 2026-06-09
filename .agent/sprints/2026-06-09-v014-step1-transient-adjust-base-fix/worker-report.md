# Worker Report: V0.14 Step-1 Transient Adjust-Base Fix

Date: 2026-06-09

Summary: Added the init-only helper
`_wrf_live_nest_transient_adjust_mub` in `src/gpuwrf/integration/d02_replay.py`
and proved that using WRF's transient post-`blend_terrain`/pre-`start_domain`
current `MUB` closes the Step-1 theta/QV residual against same-boundary WRF
truth.

## Files Changed

- `src/gpuwrf/integration/d02_replay.py`
- `proofs/v014/step1_transient_adjust_base_fix.py`
- `proofs/v014/step1_transient_adjust_base_fix.json`
- `proofs/v014/step1_transient_adjust_base_fix.md`
- `.agent/reviews/2026-06-09-v014-step1-transient-adjust-base-fix.md`

## Commands Run

- `python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/step1_transient_adjust_base_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_transient_adjust_base_fix.py`
- `python -m json.tool proofs/v014/step1_transient_adjust_base_fix.json >/tmp/step1_transient_adjust_base_fix.validated.json`
- `git diff --stat`

## Proof Result

Verdict: `STEP1_TRANSIENT_ADJUST_BASE_FIX_THETA_CLOSED`.

Key metrics:

- transient adjust-base `MUB` vs WRF adjust hook: delta
  `4.521e-04 Pa`
- final BaseState `MUB` vs WRF pre-part1 final target: delta
  `4.648e-03 Pa`
- corrected theta max_abs vs same-boundary WRF pre-call truth:
  `5.788684885033035e-05 K`
- prior theta max_abs: `0.00541785382188209 K`
- corrected QVAPOR max_abs: `5.970267497393267e-08`

## Handoff

The helper is production-callable but not yet wired into the live-nest init
consumer. The next sprint must wire WRF theta_m conversion plus `adjust_tempqv`
using this transient adjust-base `MUB`, then rerun the full Step-1 same-input
d02 comparison.
