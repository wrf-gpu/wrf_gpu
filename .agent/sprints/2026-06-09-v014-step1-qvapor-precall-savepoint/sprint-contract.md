# Sprint Contract: V0.14 Step-1 QVAPOR Pre-Call Savepoint

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Create the missing same-boundary WRF `QVAPOR` truth at
`before_first_rk_step_part1_call`, then validate that it is the same boundary
as the accepted Step-1 pre-call truth.

Trigger evidence:

- `proofs/v014/step1_live_nest_theta_semantics.json` verdict:
  `STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_PARTIAL_NEXT_TSTATE_MILLIKELVIN_RESIDUAL`.
- `proofs/v014/step1_qvapor_precall_truth_schema.json` verdict:
  `STEP1_QVAPOR_PRECALL_TRUTH_MISSING_SAVEPOINT_SPEC_READY`.
- Existing QVAPOR truth is post-RK/pre-halo or different-boundary and must not
  be reused for the live-nest theta pre-call proof.

## Method Rule

Use the fastest rigorous wall-clock method: extend the existing disposable
CPU-WRF pre-call hook by one mass-grid field, rerun only the short Step-1 truth
capture, and prove old fields are unchanged versus the accepted dump. Do not
touch production JAX source and do not use GPU.

## Non-Goals

- No `src/gpuwrf/**` edits.
- No production WRF source edit.
- No TOST.
- No Switzerland validation.
- No FP32 or memory source work.
- No GPU.
- No Hermes or Telegram.
- No broad dycore/physics rewrite.

## Inputs

- `proofs/v014/step1_qvapor_precall_truth_schema.*`
- `proofs/v014/step1_pre_part1_handoff.*`
- `.agent/sprints/2026-06-09-v014-step1-qvapor-precall-truth-schema/artifacts/proposed_wrf_savepoint.md`
- Disposable instrumented WRF tree:
  `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF`
- Disposable run directory:
  `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/run`
- Accepted pre-call truth root:
  `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`

New scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/**`

## Write Scope

Required repo files:

- `proofs/v014/step1_qvapor_precall_savepoint.py`
- `proofs/v014/step1_qvapor_precall_savepoint.json`
- `proofs/v014/step1_qvapor_precall_savepoint.md`
- `proofs/v014/step1_qvapor_precall_savepoint_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-qvapor-precall-savepoint.md`

Allowed scratch writes:

- `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/**`
- disposable edits under
  `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF/dyn_em/solve_em.F`
  only to extend the existing debug hook.

Do not touch `src/gpuwrf/**`, TOST outputs, Switzerland outputs, FP32 work,
memory source work, or old unrelated untracked artifacts.

## Required Work

1. Verify branch/head and that `86c5448b` is an ancestor.
2. Backup the current disposable WRF `solve_em.F` before editing.
3. Extend `wrfgpu2_dump_pre_part1_surface` in the disposable WRF tree:
   - keep the existing `MASS_PREPART` fields
     `T_STATE/P_STATE/PB/MU_STATE/MUB/MUT`;
   - add `QVAPOR` from `REAL(moist(i,k,j,P_QV),KIND=8)`;
   - update the `record_schema MASS_PREPART` line and numeric write format.
4. Save the WRF patch diff to
   `proofs/v014/step1_qvapor_precall_savepoint_wrf_patch.diff`.
5. Recompile the disposable WRF tree or otherwise prove the edited `wrf.exe`
   contains the new hook. Record command/log path.
6. Run the short CPU-WRF truth capture with:
   - `WRFGPU2_STEP1_PRE_PART1_HANDOFF=1`;
   - `WRFGPU2_STEP1_PRE_PART1_HANDOFF_ROOT=/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/wrf_truth`;
   - grid/domain `2`, step `1`, and the existing run directory.
7. Assemble/validate the new truth:
   - all files are `before_first_rk_step_part1_call`, domain `2`, step `1`,
     rk `1`;
   - `QVAPOR` assembles to shape `[44,66,159]`, finite;
   - old fields `T_STATE/P_STATE/PB/MU_STATE/MUB/MUT/W_STATE/PH_STATE/PHB`
     match the accepted pre-call dump exactly or within documented parse
     roundoff;
   - no post-RK/pre-halo QVAPOR is used.
8. Produce a compact JSON/Markdown report with the machine-readable path to the
   new same-boundary truth root.

## Verdicts

Emit exactly one final verdict:

- `STEP1_QVAPOR_PRECALL_SAVEPOINT_READY`
- `STEP1_QVAPOR_PRECALL_SAVEPOINT_BLOCKED_<specific_reason>`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_qvapor_precall_savepoint.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_qvapor_precall_savepoint.py
python -m json.tool proofs/v014/step1_qvapor_precall_savepoint.json \
  >/tmp/step1_qvapor_precall_savepoint.validated.json
git diff -- src/gpuwrf
```

If the WRF run cannot be completed, the proof must fail closed with the exact
command, log path, and blocker.

## Acceptance Criteria

- CPU-only proof records `gpu_used=false`.
- Same-boundary `QVAPOR` truth exists or a precise blocker is named.
- Existing pre-call fields are not silently changed by the schema extension.
- The report gives the exact truth root path for the next theta proof rerun.
- No production model source is edited.
