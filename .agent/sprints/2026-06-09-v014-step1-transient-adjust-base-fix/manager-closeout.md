# Manager Closeout: V0.14 Step-1 Transient Adjust-Base Fix

Date: 2026-06-09 17:48 WEST

## Outcome

The sprint is closed with verdict
`STEP1_TRANSIENT_ADJUST_BASE_FIX_THETA_CLOSED`.

The source change is a narrow additive helper in `d02_replay.py` that exposes
the WRF transient post-`blend_terrain`/pre-`start_domain` current `MUB` surface
for live-nest `adjust_tempqv`.

## Proof Objects

- `src/gpuwrf/integration/d02_replay.py`
- `proofs/v014/step1_transient_adjust_base_fix.py`
- `proofs/v014/step1_transient_adjust_base_fix.json`
- `proofs/v014/step1_transient_adjust_base_fix.md`
- `.agent/reviews/2026-06-09-v014-step1-transient-adjust-base-fix.md`

Key proof metrics:

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

## Merge Decision:

Commit and push the helper plus proof artifacts. Do not claim full production
Step-1 closure yet: the helper is not wired into the production live-nest
theta/QV init consumer.

## Validation

Manager reran:

- `python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/step1_transient_adjust_base_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_transient_adjust_base_fix.py`
- `python -m json.tool proofs/v014/step1_transient_adjust_base_fix.json >/tmp/step1_transient_adjust_base_fix.manager.validated.json`
- CPU-forced narrow replay tests: `4 passed, 2 skipped`

## Scope Changes

No GPU, TOST, Switzerland, FP32 source work, memory source work, or Hermes was
used. The only production source file touched is
`src/gpuwrf/integration/d02_replay.py`.

## Next Sprint

Open `v014-step1-live-nest-theta-qv-wiring`: wire WRF theta_m conversion plus
`adjust_tempqv` using `_wrf_live_nest_transient_adjust_mub` into the production
live-nest init consumer, then run the full Step-1 same-input d02 comparison.
