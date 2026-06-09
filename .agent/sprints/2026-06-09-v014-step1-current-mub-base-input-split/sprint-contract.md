# Sprint Contract: V0.14 Step-1 Current-MUB/Base-Input Split

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Explain the `17.5 Pa` current pressure/base-input mismatch found by
`proofs/v014/step1_adjust_tempqv_intermediate.json` at the Step-1 live-nest
theta residual path.

Target cell:

- zero index `{k:1,y:9,x:17}`
- Fortran index `{i:18,j:10,k:2}`
- domain `d02`

The previous proof showed:

- `p`, `mub_save`, `c3h`, `c4h`, and `p_top` match WRF.
- current `mub` differs by `17.67503987130476 Pa`.
- `pb_new_equiv` and `p_new` differ by `17.49400702366256 Pa`.
- `t_2_post` differs by `0.00541785382188209 K`.

This sprint must identify whether the mismatch is caused by a wrong JAX
live-nest base-init reconstruction, a WRF hook-boundary misunderstanding, a
missing WRF live-nest terrain/base blend operation, or a still-missing
intermediate.

## Method Rule

Use the fastest rigorous wall-clock method. Prefer a compact target-cell or
small-neighborhood WRF savepoint plus a proof-only JAX reconstruction over
another broad runtime chain. Do not edit production model source.

## Non-Goals

- No production `src/gpuwrf/**` edits.
- No source fix yet.
- No TOST.
- No Switzerland validation.
- No FP32 source work.
- No memory source work.
- No GPU.
- No Hermes or Telegram.

## Inputs

- `proofs/v014/step1_adjust_tempqv_intermediate.{py,json,md}`
- `proofs/v014/step1_theta_same_qvapor.{py,json,md}`
- `proofs/v014/step1_jax_loader_tstate.{py,json,md}`
- `proofs/v014/step1_live_nest_theta_semantics.{py,json,md}`
- Disposable WRF tree:
  `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF`
- Prior WRF hook output:
  `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth/adjust_tempqv_d2_i18_j10_k2.txt`

New scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_current_mub_base_input_split/**`

## Write Scope

Required repo files:

- `proofs/v014/step1_current_mub_base_input_split.py`
- `proofs/v014/step1_current_mub_base_input_split.json`
- `proofs/v014/step1_current_mub_base_input_split.md`
- `proofs/v014/step1_current_mub_base_input_split_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-current-mub-base-input-split.md`

Allowed scratch writes:

- `/mnt/data/wrf_gpu2/v014_step1_current_mub_base_input_split/**`
- disposable WRF source edits under
  `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF/**`
  only when env-gated and only for debug emission.

## Required Work

1. Verify branch/head and that `9a7016d9` is an ancestor.
2. Read the WRF call path around live-nest initialization:
   - `share/mediation_integrate.F` around `mub_save`, `blend_terrain`, and
     the `adjust_tempqv` call;
   - `dyn_em/nest_init_utils.F::blend_terrain`;
   - `dyn_em/nest_init_utils.F::adjust_tempqv`.
3. Freeze the formula relation for the target cell:
   `p_new = p + c3h * (mub + p_top) + c4h`.
4. Emit or recover WRF current `mub`, `mub_save`, `pb_new`/equivalent,
   terrain/base inputs, and any pre/post-`blend_terrain` values for the target
   cell or a small neighborhood.
5. Recompute the same quantities on the JAX/proof side using the exact
   live-nest base-init path used by `step1_theta_same_qvapor`.
6. Compare WRF and JAX values, name the first divergence surface, and state the
   smallest source-changing sprint that would be justified next.

If OpenMPI/PMIx is blocked in the Codex sandbox, fail closed with the exact
manager rerun command and log path. Do not fake a WRF truth surface.

## Verdicts

Emit exactly one final verdict:

- `STEP1_CURRENT_MUB_BASE_SPLIT_JAX_BASE_INIT_BUG`
- `STEP1_CURRENT_MUB_BASE_SPLIT_WRF_HOOK_BOUNDARY_MISMATCH`
- `STEP1_CURRENT_MUB_BASE_SPLIT_WRF_BLEND_UNIMPLEMENTED_OR_MISMATCHED`
- `STEP1_CURRENT_MUB_BASE_SPLIT_ROUNDING_BOUNDED`
- `STEP1_CURRENT_MUB_BASE_SPLIT_NEEDS_BROADER_FIELD_SAVEPOINT`
- `STEP1_CURRENT_MUB_BASE_SPLIT_BLOCKED_<specific_reason>`

Use `JAX_BASE_INIT_BUG` only if WRF truth clearly identifies a JAX-side
reconstruction error that can be patched without changing WRF truth. Use
`WRF_BLEND_UNIMPLEMENTED_OR_MISMATCHED` if the missing operation is WRF's
current live-nest terrain/base blend or equivalent current-`mub` update. Use
`WRF_HOOK_BOUNDARY_MISMATCH` if the previous proof compared values from
different call boundaries.

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_current_mub_base_input_split.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_current_mub_base_input_split.py
python -m json.tool proofs/v014/step1_current_mub_base_input_split.json \
  >/tmp/step1_current_mub_base_input_split.validated.json
git diff -- src/gpuwrf
```

## Acceptance Criteria

- CPU-only proof records `gpu_used=false`.
- The target residual cell is covered.
- WRF and JAX current `mub`/`pb_new`/`p_new` source values are compared with
  numeric deltas.
- The report names the first divergence surface and the next justified
  source-changing sprint, or one exact blocker.
- No production model source is edited.
