# Sprint Contract: V0.14 Base-State Split Fix

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Fix the native live-nested d02 base-state split mismatch identified by
`proofs/v014/earlier_source_bisect.json` verdict
`BASE_STATE_SPLIT_DEFINITION_MISMATCH`.

The fix target is `src/gpuwrf/integration/d02_replay.py::build_replay_case`.
The initial d02 child carry currently matches `wrfinput_d02` `PB/MUB`, but
CPU-WRF h0/h1/h10 and h10 pre-RK truth use a stable different `PB/MUB` split.
The sprint must either reproduce WRF's post-initialization `PB/MUB` split in
the native path or emit a blocked verdict naming the exact WRF routine/formula
or hook needed.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No FP32 or mixed-precision source work.
- No broad dycore, acoustic, radiation, surface-layer, or memory cleanup.
- No tolerance widening.
- No normal production dependency on CPU-WRF `wrfout` history unless explicitly
  fail-closed or bounded as validation-only/oracle-only.

## Inputs

- `proofs/v014/earlier_source_bisect.json`
- `proofs/v014/earlier_source_bisect.md`
- `proofs/v014/pre_rk_input_boundary.json`
- `proofs/v014/base_state_writer_attribution.json`
- `proofs/v014/static_metric_base_parity.json`
- `proofs/v014/previous_step_handoff_bisect.json`
- `src/gpuwrf/integration/d02_replay.py`
- CPU-WRF h0/h1 backfill outputs under
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z/`

## Write Scope

Allowed production source:

- `src/gpuwrf/integration/d02_replay.py`

Allowed proof/review outputs:

- `proofs/v014/base_state_split_fix.py`
- `proofs/v014/base_state_split_fix.json`
- `proofs/v014/base_state_split_fix.md`
- `.agent/reviews/2026-06-09-v014-base-state-split-fix.md`

Allowed tests only if needed:

- focused tests under `tests/` that exercise the corrected base split without
  broad unrelated fixture churn.

Default rule: keep the source patch narrow. If the fix requires a broader state
schema, restart writer contract, or WRF source instrumentation, stop and emit
`BASE_STATE_SPLIT_FIX_BLOCKED_<reason>` with the exact next hook/file.

Do not use Hermes, Telegram, `ask-hermes`, or any human-notification bridge.

## Required Work

1. Identify where WRF's stable h0 `PB/MUB` split differs from `wrfinput_d02` on
   the target patch.
2. Patch the native d02 child initial-state construction only if the WRF
   transformation is clear and local.
3. Prove the corrected initial `OperationalCarry` `PB/MUB` match CPU-WRF h0 and
   h10 pre-RK truth on the existing patch.
4. Prove `P/T/MU` are not made worse at the same surfaces; if their same-step
   truth is unavailable, report that honestly.
5. Run the existing earlier-source bisection or a force-replay equivalent after
   the patch, not just a JAX-vs-JAX self-compare.
6. Record whether the fix affects standalone native init, live-nested child
   init, wrfbdy boundary leaves, `BaseState`, restart output, or writer
   reconstruction.

## Validation Commands

At minimum:

```bash
python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/base_state_split_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/base_state_split_fix.py
python -m json.tool proofs/v014/base_state_split_fix.json \
  >/tmp/base_state_split_fix.validated.json
```

If a targeted GPU replay is required, use the same low-memory settings as the
earlier-source proof, record the exact command, backend, allocator, peak VRAM,
and why CPU replay was not practical.

## Acceptance Criteria

- JSON validates and top-level Markdown is compact.
- Source diff is limited to the declared scope or the sprint emits a blocked
  verdict.
- Corrected initial d02 `PB/MUB` match CPU-WRF h0/pre-RK truth on the target
  patch within frozen exact/tight tolerance declared before comparison.
- The proof explains whether `wrfinput_d02` remains different and why that is
  acceptable or unacceptable for native standalone operation.
- No TOST, Switzerland, FP32, or broad memory work is run.

## Closeout

Close with verdict, files changed, commands run, proof objects, unresolved
risks, GPU use if any, and next decision.
