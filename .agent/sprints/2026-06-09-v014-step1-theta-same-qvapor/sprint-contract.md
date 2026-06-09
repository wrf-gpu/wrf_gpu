# Sprint Contract: V0.14 Step-1 Theta Same-Boundary QVAPOR Rerun

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Rerun the Step-1 live-nest theta semantics proof using the newly validated
same-boundary WRF pre-call `QVAPOR` truth, then classify the remaining worst
`T_STATE` residual as boundary-band or interior.

Trigger evidence:

- `proofs/v014/step1_live_nest_theta_semantics.json` reduced `T_STATE`
  residual from `5.490173101425171 K` to `0.00541785382188209 K`, but used
  report-only `wrfout_d02` H0 QVAPOR.
- `proofs/v014/step1_qvapor_precall_savepoint.json` now provides accepted
  same-boundary pre-call QVAPOR at
  `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`.
- Opus critic says the remaining max_abs may be a single boundary outlier and
  must be decomposed before any source patch.

## Method Rule

Use the fastest rigorous wall-clock method: adapt the existing theta proof to
read the filtered same-boundary QVAPOR root, compute the same candidate
sequence, and add worst-cell/boundary decomposition. Do not edit production
source and do not use GPU.

## Non-Goals

- No `src/gpuwrf/**` edits.
- No production theta or `adjust_tempqv` patch.
- No WRF source edit or WRF rerun.
- No TOST, Switzerland, FP32, or memory source work.
- No GPU.
- No Hermes or Telegram.

## Inputs

- `proofs/v014/step1_live_nest_theta_semantics.py`
- `proofs/v014/step1_live_nest_theta_semantics.json`
- `proofs/v014/step1_qvapor_precall_savepoint.json`
- `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`
- `.agent/reviews/2026-06-09-v014-theta-qvapor-opus-critic.md`

## Write Scope

Required repo files:

- `proofs/v014/step1_theta_same_qvapor.py`
- `proofs/v014/step1_theta_same_qvapor.json`
- `proofs/v014/step1_theta_same_qvapor.md`
- `.agent/reviews/2026-06-09-v014-step1-theta-same-qvapor.md`

Sprint closeout files under this sprint folder.

Do not edit `src/gpuwrf/**`, old proof files, TOST outputs, Switzerland
outputs, FP32 work, memory source work, or unrelated untracked artifacts.

## Required Work

1. Verify branch/head and that `912b7371` is an ancestor.
2. Read and reuse the existing theta proof logic; copy only what is needed into
   a new proof script rather than overwriting the old proof.
3. Load same-boundary QVAPOR only from
   `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`.
   Fail closed if the root is missing, wrong-boundary, wrong-shape, or
   nonfinite.
4. Recompute the candidate sequence from the previous proof:
   - raw/current live dry `T_STATE` versus WRF pre-call;
   - dry-to-moist theta conversion;
   - WRF `theta_m` conversion plus `adjust_tempqv`;
   - same candidate under fp32 arithmetic if cheap.
5. Report full-grid metrics and boundary decomposition:
   - all-cell max_abs/RMSE/p95/p99/p99.9;
   - boundary band (`distance_to_edge <= 5`) versus interior;
   - worst-cell zero and Fortran indices, boundary distance, WRF value,
     candidate value, delta, QVAPOR, pressure/base inputs available in the
     proof, and whether the cell is in the boundary band.
6. Emit a compact verdict that decides the next manager action.

## Verdicts

Emit exactly one final verdict:

- `STEP1_THETA_SAME_QVAPOR_PATCH_READY`
- `STEP1_THETA_SAME_QVAPOR_BOUNDARY_TAIL_BOUNDED_NEXT_BASE`
- `STEP1_THETA_SAME_QVAPOR_INTERIOR_RESIDUAL_NEEDS_WRF_INTERMEDIATE`
- `STEP1_THETA_SAME_QVAPOR_BLOCKED_<specific_reason>`

Use `PATCH_READY` only if the same-boundary QVAPOR candidate brings max_abs
under the predeclared `1e-3 K` material threshold or gives a source patch with
equivalent evidence. Use `BOUNDARY_TAIL_BOUNDED_NEXT_BASE` only if the remaining
failure is boundary-local and interior metrics are below threshold.

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_theta_same_qvapor.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_theta_same_qvapor.py
python -m json.tool proofs/v014/step1_theta_same_qvapor.json \
  >/tmp/step1_theta_same_qvapor.validated.json
git diff -- src/gpuwrf
```

## Acceptance Criteria

- CPU-only proof records `gpu_used=false`.
- Same-boundary QVAPOR root is used and named.
- Worst-cell boundary/interior classification is present.
- Verdict maps directly to the next sprint decision.
- No production model source is edited.
