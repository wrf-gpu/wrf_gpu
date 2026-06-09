# Sprint Contract: V0.14 Step-1 Adjust-TempQV Intermediate Truth

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Emit or recover CPU-WRF's exact `adjust_tempqv` intermediate values for the
Step-1 live-nest theta residual path, then compare them against the JAX proof
transcription. The immediate target is the interior worst cell from
`proofs/v014/step1_theta_same_qvapor.json`:

- zero index `{k:1,y:9,x:17}`
- Fortran index `{i:18,j:10,k:2}`
- horizontal boundary distance `9`
- residual `0.00541785382188209 K`

This sprint must explain whether the residual is caused by formula
transcription, pressure/base inputs, WRF source-order/rounding, or a still
missing WRF intermediate.

## Method Rule

Use the fastest rigorous wall-clock method: instrument the disposable CPU-WRF
tree only enough to emit exact WRF internals for the residual path. Prefer a
compact one-cell or small-neighborhood savepoint over another huge full-domain
dump if it contains all needed values. Do not edit production model source and
do not use GPU.

## Non-Goals

- No `src/gpuwrf/**` edits.
- No production theta/`adjust_tempqv` patch.
- No TOST.
- No Switzerland validation.
- No FP32 or memory source work.
- No GPU.
- No Hermes or Telegram.
- No broad WRF or dycore rewrite.

## Inputs

- `proofs/v014/step1_theta_same_qvapor.{py,json,md}`
- `proofs/v014/step1_qvapor_precall_savepoint.{py,json,md}`
- `.agent/reviews/2026-06-09-v014-theta-qvapor-opus-critic.md`
- Disposable instrumented WRF tree:
  `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF`
- Disposable run directory:
  `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/run`
- Existing wrf-build environment under:
  `/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build`

New scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/**`

## Write Scope

Required repo files:

- `proofs/v014/step1_adjust_tempqv_intermediate.py`
- `proofs/v014/step1_adjust_tempqv_intermediate.json`
- `proofs/v014/step1_adjust_tempqv_intermediate.md`
- `proofs/v014/step1_adjust_tempqv_intermediate_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-adjust-tempqv-intermediate.md`

Allowed scratch writes:

- `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/**`
- disposable WRF source edits under
  `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF/**`
  only when env-gated and only for debug emission.

Do not touch `src/gpuwrf/**`, TOST outputs, Switzerland outputs, FP32 work,
memory source work, or unrelated untracked artifacts.

## Required Work

1. Verify branch/head and that `c3620d09` is an ancestor.
2. Inspect the disposable WRF call path for live-nest `adjust_tempqv`.
   Expected source evidence includes:
   - `share/mediation_integrate.F` live-nest branch around the
     `adjust_tempqv` call;
   - `dyn_em/nest_init_utils.F::adjust_tempqv`.
3. Add an env-gated disposable WRF hook that emits for d02 and the target
   residual path either:
   - the exact target cell plus a small neighborhood; or
   - compact full fields if easier and still cheap.
4. The hook must emit enough to compare WRF against JAX:
   - pre-adjust `t_2` / theta value;
   - post-adjust `t_2` / theta value;
   - pre-adjust QVAPOR and post-adjust QVAPOR;
   - `p_old`, `p_new`, `tc`, `rh`, `dth1`, `dth`, and any equivalent WRF
     intermediates available inside `adjust_tempqv`;
   - `p`, `pb`, `mub`, `mub_save`, `c3h`, `c4h`, `p_top`;
   - cell indices and domain/timestep context.
5. Save the WRF patch diff to
   `proofs/v014/step1_adjust_tempqv_intermediate_wrf_patch.diff`.
6. Rebuild the disposable WRF tree or prove the edited executable contains the
   hook.
7. Run the shortest CPU-WRF truth capture needed. If the Codex sandbox blocks
   OpenMPI/PMIx sockets, fail closed with the exact command/log path and state
   whether the manager must rerun unsandboxed.
8. Assemble a JSON/Markdown report that compares WRF emitted intermediates to
   the JAX proof values already recorded in
   `proofs/v014/step1_theta_same_qvapor.json`.

## Verdicts

Emit exactly one final verdict:

- `STEP1_ADJUST_TEMPQV_INTERMEDIATE_TRANSCRIPTION_BUG_FOUND`
- `STEP1_ADJUST_TEMPQV_INTERMEDIATE_PRESSURE_INPUT_MISMATCH`
- `STEP1_ADJUST_TEMPQV_INTERMEDIATE_ROUNDING_TAIL_BOUNDED`
- `STEP1_ADJUST_TEMPQV_INTERMEDIATE_NEEDS_BROADER_WRF_SAVEPOINT`
- `STEP1_ADJUST_TEMPQV_INTERMEDIATE_BLOCKED_<specific_reason>`

Use `ROUNDING_TAIL_BOUNDED` only if WRF intermediate values match the JAX
transcription closely enough that the remaining `0.0054 K` is explained by
recorded precision/source-order limits. Use `PRESSURE_INPUT_MISMATCH` if WRF
and JAX differ materially in `p_old`, `p_new`, `mub`, `mub_save`, `p`, or base
inputs. Use `TRANSCRIPTION_BUG_FOUND` only when the formula/intermediate
sequence is clearly wrong in the JAX proof transcription or port.

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_adjust_tempqv_intermediate.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_adjust_tempqv_intermediate.py
python -m json.tool proofs/v014/step1_adjust_tempqv_intermediate.json \
  >/tmp/step1_adjust_tempqv_intermediate.validated.json
git diff -- src/gpuwrf
```

## Acceptance Criteria

- CPU-only proof records `gpu_used=false`.
- The report names the exact WRF source hook and command/log paths.
- The target residual cell is covered.
- The report compares WRF intermediates against JAX proof values with numeric
  deltas.
- The verdict maps directly to the next manager decision.
- No production model source is edited.
