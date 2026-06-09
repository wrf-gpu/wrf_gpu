# Sprint Contract: V0.14 Parallel Memory/FP32 Manager

Date: 2026-06-09
Manager: GPT-5.5 xhigh side manager
Branch: `worker/gpt/v014-memory-fp32-manager`
Base commit: `131b27cd`

## Objective

Run a parallel manager lane for v0.14 memory and FP32 work while the primary
manager continues fp64 grid-parity debugging.

End goal for this lane:

- close every non-conflicting memory optimization that can be implemented with
  stable, bit-identical or predeclared-tolerance proof;
- advance FP32 acoustic/mixed precision as far as possible without colliding
  with active fp64 grid-parity debug work;
- prove if a requested FP32 path is infeasible, or identify exact blockers and
  gates rather than hand-waving;
- hand back a reviewed branch/report for the primary manager to inspect and
  merge only after validation.

## Current Primary-Manager State

The primary manager is debugging fp64 grid drift. Current closed proof:

- `proofs/v014/step1_first_rk_part1_p_state_split.json`
- verdict: `STEP1_FIRST_RK_PART1_P_STATE_LOCALIZED_PRE_PART1_RAW_CHILD_STATE`
- next fp64 debug target: live-nest `raw_child_state -> live_child_state`
  perturbation-state initialization for `P_STATE/MU_STATE/W_STATE`.

Do not invalidate that proof chain.

## Hard Collision Limits

Until the primary manager explicitly merges or releases these locks, this lane
must not edit production files in these areas:

- `src/gpuwrf/dynamics/**`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/nesting/**`
- `src/gpuwrf/boundary*`
- `src/gpuwrf/contracts/state.py`
- any live-nest/base-state/init/restart/wrfout compatibility file touched by the
  grid-parity chain

Allowed production source edits, if proven file-disjoint and narrow:

- WDM6 `slmsk` shape-only cleanup in the specific microphysics/coupling adapter
  files after exact-output proof;
- active moisture transport velocity reuse only if it does not touch locked
  runtime/dycore files; otherwise produce an implementation plan and wait;
- profiler/preflight scripts, proof scripts, docs, and roadmap artifacts.

If a memory fix needs a locked file, record the proof and proposal only. Do not
patch the locked file.

## FP32 Rules

FP32 acoustic/mixed precision is high-value v0.14 work, but it touches the same
fault surface as current grid-parity debug. Therefore:

- no production dycore/runtime/source edits for FP32 in this lane;
- allowed: ADR refinement, static audit, CPU-only scalar/one-column probes,
  proof scripts, feasibility/infeasibility report, and a detailed R0-R8 plan;
- allowed: isolated throwaway prototype under `proofs/v014/` if it does not
  import or edit production dycore state in a way that changes the branch;
- not allowed: mixed-mode GPU forecast, dtype demotion in production, source
  changes to acoustic/prep/finish/carry/boundary/restart/init.

The lane may conclude "not currently implementable without colliding with
grid-parity" for source work, but must still maximize useful proofs and plans.

## Required Inputs

- `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
- `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`
- `proofs/v014/empirical_memory_map.{json,md}`
- `proofs/v014/exact_branch_memory_preflight.{json,md}`
- `proofs/v014/fp32_acoustic_probes.{json,py}`
- `.agent/reviews/2026-06-08-gpt-v014-fp32-status-freeze.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`

## Required Work

1. Verify branch/head and confirm base `131b27cd` is an ancestor.
2. Build a collision map: memory/FP32 source files versus active grid-parity
   locked files.
3. Re-run or refresh the exact-branch memory preflight if it is CPU-only or
   short-GPU-safe. For any GPU use, first acquire the repo GPU lock via
   `scripts/run_gpu_lowprio.sh`; do not run concurrently with the primary
   manager's GPU jobs.
4. Implement at most one non-conflicting memory source fix first, preferably
   WDM6 `slmsk` shape-only cleanup, with exact-output proof. If no fix is truly
   file-disjoint, produce a no-source proof/report.
5. For FP32, produce a refreshed feasibility/infeasibility proof bundle:
   cancellation probe, one-column/analytic probe if cheap, static collision
   audit, and a concrete implementation gate list.
6. Update this branch's roadmap/proof files only after evidence exists.
7. Commit and push the branch if and only if validation passes.
8. Report to the primary manager with concise top-level status and paths.

## GPU Lock Protocol

- Default: CPU-only.
- If GPU is needed, use `scripts/run_gpu_lowprio.sh`.
- Before a GPU job, check for existing GPU jobs and leave a short report in the
  sprint review.
- If primary manager starts TOST/Schweiz/Grid-Delta or another GPU run, this
  lane yields.

## Validation

Minimum validation for any source edit:

```bash
python -m py_compile <changed python files>
python -m pytest -q <focused tests>
python -m json.tool <proof json> >/tmp/<proof>.validated.json
git diff --check
git diff -- src/gpuwrf
```

For a no-source analysis sprint:

```bash
python -m py_compile <new proof scripts>
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python <proof script>
python -m json.tool <proof json> >/tmp/<proof>.validated.json
git diff --check
```

## Deliverables

- `.agent/reviews/2026-06-09-v014-parallel-memory-fp32-manager.md`
- `proofs/v014/parallel_memory_fp32_manager.json`
- `proofs/v014/parallel_memory_fp32_manager.md`
- optional source/proof commits on branch `worker/gpt/v014-memory-fp32-manager`
- clear merge recommendation: `MERGE_NOW`, `REVIEW_ONLY`, or `DO_NOT_MERGE`

## Completion Signal

Notify the primary manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT PARALLEL_MEMORY_FP32_MANAGER DONE - see proofs/v014/parallel_memory_fp32_manager.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
