# Memory Patch: V0.14 Fable/Mythos Heavy Lane And Memory Closure

Reviewer Status: accepted after primary manager review.

## Process Memory Scope

Update durable manager operating memory and the local sprint-management skill for
the principal's 2026-06-09 directive:

- Extremely hard v0.14 problems go to Fable/Mythos in tmux `0:1` as whole
  endpoint-defined assignments.
- Fable/Mythos tokens are scarce: do not use Fable for routine polling, simple
  proof grooming, normal GPT-debug collection, or tasks likely solvable in one
  focused GPT 5.5 sprint.
- For validation failures, first send GPT 5.5 workers to collect, localize,
  and attempt direct fixes when the issue is tractable. Escalate only the
  unresolved hard core to Fable/Mythos.
- Before each new Fable/Mythos sprint after completion or context-risk point,
  send `/compact`, wait about two minutes, then send the full assignment.
- The manager remains responsible for contracts, locks, proof gates, diff
  review, merge/reject decisions, and final v0.14 closure.

## Memory/FP32 Closure Facts

v0.14 memory/FP32 lane closure facts:

- MYNN BouLac dense `(B, nz, nz)` matrices were the largest measured
  non-radiation transient and are now column-tiled using the RRTMG pattern.
  GPU tile-vs-untiled is bit-identical, including the ragged production-tile
  case; CPU tridiag differences are bounded at XLA:CPU codegen ulp scale.
- Moisture transport-velocity reuse is source-level hygiene, but measured
  non-material for VRAM because XLA already CSEs the duplicate expression.
- Static duplicate-expression estimates are not VRAM gains until compiled-memory
  measurement confirms XLA did not already CSE them.
- FP32 acoustic R0 contract is landed default-inert. R1+ resumes only after the
  one-RK-step fp64 dynamics frontier closes.

## Evidence

- Principal directive in-chat on 2026-06-09 for Fable/Mythos token conservation.
- `proofs/v014/mythos_kernel_fix_260609.md`
- `proofs/v014/mythos_memory_fixes_260609.json`
- `proofs/v014/mythos_memory_gpu_suite_260609.json`
- `proofs/v014/exact_branch_memory_preflight*.json`
- `proofs/v014/fp32_acoustic_static_audit.json`
- `proofs/v013/moisture_advection_wiring.json`

## Proposed Destinations

- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/skills/managing-sprints/CHANGELOG.md`
- `.agent/memory/stable/mythos-heavy-lane.md`
- `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
- `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`

## Patch

- Record Fable/Mythos as a scarce hard-debug lane only.
- Record MYNN tiling default ON, tile width 16384, env-gated with
  `GPUWRF_MYNN_COLUMN_TILING` / `GPUWRF_MYNN_COLUMN_TILE_COLS`; whole-batch
  reference path retained for proofs.
- Record FP32 R1 resume gate as one-RK-step fp64 dynamics closure.
