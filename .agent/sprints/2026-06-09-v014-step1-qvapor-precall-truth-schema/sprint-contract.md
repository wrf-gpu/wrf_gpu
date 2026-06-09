# Sprint Contract: V0.14 Step-1 QVAPOR Pre-Call Truth Schema

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Establish the authoritative WRF truth boundary for live-nest `QVAPOR` and
theta semantics at the same pre-call point used by the Step-1 `T_STATE` proofs.

Trigger evidence:

- Active sprint `v014-step1-live-nest-theta-semantics` found that the accepted
  pre-call text schema used for `T_STATE` does not include `QVAPOR`.
- A proof-local `adjust_tempqv` transcription using raw `MUB/P/T/QVAPOR`, live
  recomputed `MUB`, real `C3H/C4H/P_TOP`, and `USE_THETA_M=1` did not close the
  `T_STATE` residual; max residual remained about `5.49 K`.
- Existing adjacent QVAPOR artifacts appear to be post-RK or from a different
  boundary, so they must not be silently substituted for pre-call truth.

This sprint must decide whether an authoritative same-boundary WRF `QVAPOR`
truth already exists and, if not, specify the exact minimal WRF savepoint needed
for the next proof.

## Method Rule

Use the fastest rigorous wall-clock method: schema inventory and boundary proof
before another formula or source patch. Do not run long WRF/JAX validation, do
not use GPU, and do not modify production JAX source.

## Non-Goals

- No production `src/gpuwrf/**` edits.
- No TOST.
- No Switzerland validation.
- No FP32 or memory source work.
- No GPU.
- No Hermes or Telegram.
- No broad dycore/physics rewrite.

## Inputs

- `proofs/v014/step1_jax_loader_tstate.*`
- `proofs/v014/step1_live_nest_init_rerun.*`
- `proofs/v014/step1_pre_part1_handoff.*`
- `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth/**`
- `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/**`
- Active sprint folder
  `.agent/sprints/2026-06-09-v014-step1-live-nest-theta-semantics/`
- WRF source:
  - `/home/enric/src/wrf_pristine/WRF/share/mediation_integrate.F`
  - `/home/enric/src/wrf_pristine/WRF/dyn_em/nest_init_utils.F`

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_truth_schema/**`

## Write Scope

Required repo files:

- `proofs/v014/step1_qvapor_precall_truth_schema.py`
- `proofs/v014/step1_qvapor_precall_truth_schema.json`
- `proofs/v014/step1_qvapor_precall_truth_schema.md`
- `.agent/reviews/2026-06-09-v014-step1-qvapor-precall-truth-schema.md`

Optional non-source artifact:

- `.agent/sprints/2026-06-09-v014-step1-qvapor-precall-truth-schema/artifacts/proposed_wrf_savepoint.md`
  if a missing WRF savepoint is the verdict.

Do not touch `src/gpuwrf/**`, TOST outputs, Switzerland outputs, FP32 work, or
old untracked artifacts.

## Required Work

1. Verify branch/head and that `5b1f6b10` is an ancestor.
2. Inventory every existing Step-1 WRF truth artifact that contains `QVAPOR`,
   with boundary name, timestamp/step/rk if available, shape, and whether it is
   same-boundary as `before_first_rk_step_part1_call`.
3. Inventory the accepted pre-call text schema fields and explicitly prove
   whether `QVAPOR` is absent from that boundary.
4. Inspect WRF call order around live-nest input, `adjust_tempqv`, and first
   RK/physics handoff. Record exact source lines that determine where `QVAPOR`
   must be captured.
5. Decide the theta contract needed by the next proof:
   - whether WRF `T`/`t_2` at this boundary is dry theta perturbation or moist
     theta perturbation when `USE_THETA_M=1`;
   - which JAX state field should be compared to it.
6. If same-boundary `QVAPOR` truth exists, produce a machine-readable manifest
   pointing to it and a compact validator script.
7. If same-boundary `QVAPOR` truth does not exist, produce the exact minimal WRF
   savepoint spec: file/function, before/after call location, fields, shapes,
   and acceptance checks.

## Verdicts

Emit exactly one final verdict:

- `STEP1_QVAPOR_PRECALL_TRUTH_EXISTS`
- `STEP1_QVAPOR_PRECALL_TRUTH_MISSING_SAVEPOINT_SPEC_READY`
- `STEP1_QVAPOR_PRECALL_TRUTH_BLOCKED_<specific_reason>`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_qvapor_precall_truth_schema.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_qvapor_precall_truth_schema.py
python -m json.tool proofs/v014/step1_qvapor_precall_truth_schema.json \
  >/tmp/step1_qvapor_precall_truth_schema.validated.json
git diff -- src/gpuwrf
```

## Acceptance Criteria

- CPU-only proof records `gpu_used=false`.
- The report distinguishes same-boundary pre-call truth from post-RK or
  different-boundary QVAPOR artifacts.
- WRF source line evidence is included for the required savepoint boundary.
- The final output is short enough for manager context: one verdict, one truth
  inventory table, one next-action paragraph.
