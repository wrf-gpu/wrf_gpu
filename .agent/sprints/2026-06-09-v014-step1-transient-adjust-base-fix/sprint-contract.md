# Sprint Contract: V0.14 Step-1 Transient Adjust-Base Fix

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Implement the smallest production-source fix for the Step-1 live-nest theta/QV
initialization mismatch identified by
`proofs/v014/step1_current_mub_base_input_split.json`.

WRF live-nest initialization has two legitimate base surfaces:

1. transient post-`blend_terrain` / pre-`start_domain` current `MUB`, consumed
   by `adjust_tempqv`;
2. final post-`start_domain` BaseState, used by step-entry/pre-part1 state.

The current proof path uses the final BaseState `MUB` for `adjust_tempqv`, which
is wrong for WRF parity. This sprint must add a transient adjust-base path for
theta/QV adjustment only, while keeping final BaseState unchanged.

Target evidence from the prior sprint:

- WRF adjust hook current `MUB`: `86812.25`
- proof-side direct WRF blend `MUB`: `86812.250452109511`
- JAX final base `MUB`: `86794.574960128695`
- WRF pre-part1 final `MUB`: `86794.5703125`

## Method Rule

Use the fastest rigorous wall-clock method: small source patch, CPU-only proof,
field-level guard. Do not run a long forecast. Do not use GPU.

## Non-Goals

- No dycore/acoustic/physics rewrites.
- No TOST.
- No Switzerland validation.
- No FP32 source work.
- No memory source work.
- No GPU.
- No Hermes or Telegram.
- No release/tag work.

## Inputs

- `proofs/v014/step1_current_mub_base_input_split.{py,json,md}`
- `proofs/v014/step1_theta_same_qvapor.{py,json,md}`
- `proofs/v014/step1_live_nest_theta_semantics.{py,json,md}`
- `proofs/v014/step1_qvapor_precall_savepoint.{py,json,md}`
- `src/gpuwrf/integration/d02_replay.py`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

## Write Scope

Allowed production source:

- `src/gpuwrf/integration/d02_replay.py`

Allowed proof/test artifacts:

- `proofs/v014/step1_transient_adjust_base_fix.py`
- `proofs/v014/step1_transient_adjust_base_fix.json`
- `proofs/v014/step1_transient_adjust_base_fix.md`
- `.agent/reviews/2026-06-09-v014-step1-transient-adjust-base-fix.md`

Optional focused tests if useful:

- `tests/**` only if narrowly tied to the new helper and fast on CPU.

Do not edit unrelated production files.

## Required Work

1. Verify branch/head and that `43173cb2` is an ancestor.
2. Inspect `src/gpuwrf/integration/d02_replay.py` live-nest base-init helper and
   the proof scripts that call it.
3. Add the narrowest source helper/API needed to expose or use transient
   post-`blend_terrain` current `MUB` for `adjust_tempqv`.
   - Final BaseState after `start_domain` semantics must remain unchanged.
   - Existing public callers must remain backward-compatible unless the proof
     updates them explicitly.
4. Rerun the Step-1 theta/QV candidate proof path with the corrected transient
   adjust base.
5. Emit a field-level guard comparing:
   - transient adjust-base `MUB` vs WRF adjust hook target;
   - final BaseState `MUB` vs WRF pre-part1 final target;
   - corrected theta/QV candidate vs same-boundary WRF pre-call truth.
6. If the theta residual closes below the accepted gate, name the next larger
   grid-parity validation step. If it does not close, classify the new residual
   and name the next exact boundary.

## Verdicts

Emit exactly one final verdict:

- `STEP1_TRANSIENT_ADJUST_BASE_FIX_THETA_CLOSED`
- `STEP1_TRANSIENT_ADJUST_BASE_FIX_THETA_IMPROVED_NEXT_BOUNDARY`
- `STEP1_TRANSIENT_ADJUST_BASE_FIX_NO_EFFECT`
- `STEP1_TRANSIENT_ADJUST_BASE_FIX_BLOCKED_<specific_reason>`

Use `THETA_CLOSED` only if the corrected Step-1 theta/QV field meets the
predeclared gate from the previous theta proof or a stricter field-level gate.
Use `THETA_IMPROVED_NEXT_BOUNDARY` if it materially reduces the residual but
leaves a named residual surface.

## Commands / Validation

At minimum:

```bash
python -m py_compile src/gpuwrf/integration/d02_replay.py \
  proofs/v014/step1_transient_adjust_base_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_transient_adjust_base_fix.py
python -m json.tool proofs/v014/step1_transient_adjust_base_fix.json \
  >/tmp/step1_transient_adjust_base_fix.validated.json
git diff --stat
```

If source is changed, also run the narrowest existing CPU tests/proofs that
cover `d02_replay.py` live-nest initialization.

## Acceptance Criteria

- The source diff is limited to the allowed source file unless a manager
  explicitly opens a new contract.
- Existing final BaseState semantics are proven unchanged.
- The transient adjust-base value matches WRF adjust hook at the target cell.
- The corrected theta/QV candidate is compared against same-boundary WRF truth.
- The proof records `gpu_used=false`.
- The report names the next manager decision and any remaining residual.
