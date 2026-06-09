# Memory Manager Contract 2026-06-09

## Objective

You are a secondary manager for `wrf_gpu2` v0.14 memory and FP32 work.
Your job is to advance memory optimization and FP32 acoustic de-risking without
invalidating or colliding with the primary manager's active grid-parity debug.

Primary project goal: a WRF-faithful-enough, GPU-optimized, near compute- and
memory-optimal, scalable GPU rewrite. Do not optimize memory by weakening
correctness evidence.

## Read First

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
5. `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
6. `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`
7. `proofs/v014/parallel_memory_fp32_manager.md`
8. `.agent/reviews/2026-06-09-v014-parallel-memory-fp32-manager.md`

## Current State

The previous parallel memory/FP32 manager was merged as `ee6cbbe1`.

Closed:
- WDM6 `slmsk` shape-only cleanup in `src/gpuwrf/coupling/scan_adapters.py`.
- CPU exact-output proof.
- 85 WDM6 savepoint parity tests.
- WDM6 operational smoke.
- Approximate saving: `76.9 MiB` at `641x321x50`.

Still open:
- Exact-branch GPU memory preflight after grid-parity stabilizes.
- Moisture transport velocity reuse.
- Non-radiation physics column-tiling pilot.
- Moisture limiter workspace reduction.
- Acoustic carry split / acoustic memory work.
- State alias reduction.
- FP32 acoustic R0-R8 implementation roadmap.

## Hard Locks

Until the primary manager explicitly releases the grid-parity lock, do not edit:

- `src/gpuwrf/dynamics/**`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/nesting/**`
- `src/gpuwrf/contracts/state.py`
- boundary, carry, restart, init, wrfout, or live-nest/base-state files
- any file currently touched by an active grid-parity sprint

Allowed without primary-manager approval:

- proof scripts under `proofs/v014/`
- reviews under `.agent/reviews/`
- sprint contracts and closeouts under `.agent/sprints/`
- roadmap updates under `.agent/decisions/`
- CPU-only static audits and collision maps
- short GPU measurement only through the repo GPU lock protocol, and only if no
  primary validation/debug GPU job is active

Production source edits are allowed only if all are true:

- the file is outside the hard locks;
- the edit is bit-identical or has a predeclared physical tolerance;
- the proof gate is run before handoff;
- the changed file is not needed by the active grid-parity debug path.

## Priority Order

1. Build and keep current a collision map: memory/FP32 candidates versus locked
   grid-parity files.
2. Prepare exact-branch memory preflight commands for the post-grid-parity
   branch, but do not run a long GPU validation campaign.
3. Implement only non-conflicting, bit-identical memory fixes if any remain.
4. For FP32, do R0/R1 de-risking only: ADR, static audit, scalar/one-column
   probes, and source-collision plan. No production mixed-precision dycore edit
   until the active grid bug is closed or the primary manager releases the lock.
5. If a candidate collides, write the exact blocker and a ready-to-run sprint
   plan instead of editing source.

## Required Deliverables

Write or update:

- `proofs/v014/memory_manager_260609.json`
- `proofs/v014/memory_manager_260609.md`
- `.agent/reviews/2026-06-09-v014-memory-manager-260609.md`

The report must include:

- recommendation: `MERGE_NOW`, `REVIEW_ONLY`, or `DO_NOT_MERGE`
- files changed
- commands run
- proof objects produced
- exact source locks respected
- exact source locks blocking each deferred item
- GPU usage: yes/no, peak VRAM if used, and lock protocol evidence
- top 5 next memory/FP32 tasks after grid-parity closes

## Validation

For source edits:

```bash
python -m py_compile <changed python files>
python -m pytest -q <focused tests>
python -m json.tool proofs/v014/memory_manager_260609.json \
  >/tmp/memory_manager_260609.validated.json
git diff --check
git diff -- src/gpuwrf
```

For no-source analysis:

```bash
python -m py_compile <new proof scripts>
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python <proof script>
python -m json.tool proofs/v014/memory_manager_260609.json \
  >/tmp/memory_manager_260609.validated.json
git diff --check
```

## Completion Signal

If running in tmux, notify the primary manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT MEMORY_MANAGER_260609 DONE - see proofs/v014/memory_manager_260609.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```

No Hermes or Telegram updates.
