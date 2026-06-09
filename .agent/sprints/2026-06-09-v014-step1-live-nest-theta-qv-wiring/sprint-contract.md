# Sprint Contract: V0.14 Step-1 Live-Nest Theta/QV Wiring

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Wire WRF live-nest theta_m conversion plus `adjust_tempqv` into the production
live-nest initialization consumer, using
`gpuwrf.integration.d02_replay._wrf_live_nest_transient_adjust_mub`.

The previous sprint proved the corrected theta/QV candidate closes the Step-1
theta residual, but the helper is not yet consumed by production live-nest init.
This sprint must make production initialization produce the corrected state and
then run the next Step-1 grid-parity proof.

## Method Rule

Use the smallest source patch and the strongest cheap CPU proof. Do not run a
long forecast, TOST, Switzerland, GPU validation, FP32 source work, or memory
source work.

## Non-Goals

- No dycore/acoustic/physics rewrites.
- No radiation/memory cleanup.
- No FP32 source work.
- No GPU.
- No TOST or Switzerland.
- No Hermes or Telegram.
- No release/tag work.

## Inputs

- `src/gpuwrf/integration/d02_replay.py`
- `proofs/v014/step1_transient_adjust_base_fix.{py,json,md}`
- `proofs/v014/step1_qvapor_precall_savepoint.{py,json,md}`
- `proofs/v014/step1_live_nest_init_rerun.{py,json,md}`
- `proofs/v014/step1_same_input_truth.{py,json,md}`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

## Write Scope

Allowed production source:

- `src/gpuwrf/integration/d02_replay.py`

Allowed proof artifacts:

- `proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `proofs/v014/step1_live_nest_theta_qv_wiring.json`
- `proofs/v014/step1_live_nest_theta_qv_wiring.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-qv-wiring.md`

Optional proof updates only if needed:

- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_init_rerun.json`
- `proofs/v014/step1_live_nest_init_rerun.md`

Do not edit unrelated production files.

## Required Work

1. Verify branch/head and that `a8f5c485` is an ancestor.
2. Identify the production live-nest child initialization path in
   `build_replay_case(..., live_nest_parent=...)`.
3. Apply WRF semantics only for live-nested child init:
   - convert raw dry `T` to moist theta when `USE_THETA_M=1`;
   - call/implement WRF `adjust_tempqv` using:
     - saved pre-blend child `MUB`,
     - transient post-blend current `MUB` from
       `_wrf_live_nest_transient_adjust_mub`,
     - unchanged perturbation pressure `P`,
     - existing QVAPOR.
4. Keep final post-`start_domain` BaseState (`PB/PHB/MUB`) unchanged.
5. Prove production initialization, not only proof-local candidate, matches:
   - transient `MUB` vs WRF adjust hook;
   - final BaseState `MUB` vs WRF pre-part1 target;
   - theta and QVAPOR vs same-boundary WRF pre-call truth.
6. Run the next larger Step-1 same-input d02 comparison or a strict equivalent
   16-field initialization comparison. If it is still red, name the first
   remaining divergent field and boundary.

## Verdicts

Emit exactly one final verdict:

- `STEP1_LIVE_NEST_THETA_QV_WIRING_FULL_STEP1_CLOSED`
- `STEP1_LIVE_NEST_THETA_QV_WIRING_INIT_CLOSED_NEXT_FIELD`
- `STEP1_LIVE_NEST_THETA_QV_WIRING_NO_EFFECT`
- `STEP1_LIVE_NEST_THETA_QV_WIRING_BLOCKED_<specific_reason>`

Use `FULL_STEP1_CLOSED` only if the Step-1 same-input d02 comparison meets the
accepted 16-field gate. Use `INIT_CLOSED_NEXT_FIELD` if theta/QV initialization
closes but another Step-1 field remains divergent.

## Commands / Validation

At minimum:

```bash
python -m py_compile src/gpuwrf/integration/d02_replay.py \
  proofs/v014/step1_live_nest_theta_qv_wiring.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_live_nest_theta_qv_wiring.py
python -m json.tool proofs/v014/step1_live_nest_theta_qv_wiring.json \
  >/tmp/step1_live_nest_theta_qv_wiring.validated.json
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q \
  tests/test_m7_l2_d02_replay.py \
  tests/test_m6x_d02_boundary_replay.py \
  tests/test_m6x_d02_replay_hang_debug.py
git diff --stat
```

If `step1_live_nest_init_rerun.py` is updated, rerun and validate that proof
too.

## Acceptance Criteria

- Production source diff is limited to `src/gpuwrf/integration/d02_replay.py`.
- Proof records `gpu_used=false`.
- Production live-nest init theta/QV matches same-boundary WRF pre-call truth
  under the established gate.
- Final BaseState guard remains unchanged.
- The report names whether Step-1 is closed or the next exact field/boundary.
