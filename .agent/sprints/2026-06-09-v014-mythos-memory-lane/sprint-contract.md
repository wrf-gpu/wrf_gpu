# Sprint Contract: V0.14 Mythos Memory/FP32 Lane

Date: 2026-06-09 22:40 WEST
Manager: GPT-5.5 xhigh
Assignee: Mythos in tmux `0:1`
Base commit: `a32efce3` (`v014 close live-nest start-domain init`)
Branch/worktree required:
`/home/enric/src/wrf_gpu2/.codex/worktrees/mythos-memory-v014` on
`worker/mythos/v014-memory-fp32`

## Objective

Close the complete v0.14 memory improvement/fix lane, including FP32/mixed
precision de-risking, without weakening the WRF-faithful GPU-native project
goal.

Endpoint:

- All known memory issues in `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`,
  `proofs/v014/empirical_memory_map.md`, and
  `proofs/v014/memory_manager_260609.md` are fixed where technically safe.
- Any additional material memory issue discovered during the sprint is fixed or
  exactly proven/deferred with a quantified reason.
- FP32/acoustic memory work is solved into a default-off or production-safe mode
  if feasible; otherwise prove precisely why it is not currently safe/possible
  and provide the minimal remaining roadmap.
- Every correctness, memory, VRAM, speed, or FP32 claim has a proof object.

Do not treat this as a report-only task. A report-only result is acceptable only
for a row that is proven non-material, technically unsafe until another
correctness gate closes, or impossible without violating the project
constitution.

## Manager/Assignee Boundary

The manager remains responsible for review, merge/reject, final validation, and
0.14 closure. Mythos should work in an isolated worktree and commit on its own
branch. Do not edit the main worktree.

The manager will later inspect the branch, rerun gates, and merge only accepted
changes.

## Required Setup

From `/home/enric/src/wrf_gpu2`:

```bash
git worktree add .codex/worktrees/mythos-memory-v014 -b worker/mythos/v014-memory-fp32 a32efce3
cd .codex/worktrees/mythos-memory-v014
git log -1 --oneline
git status --short
```

If the branch/worktree already exists, reuse it only if it is based on
`a32efce3` and has no unrelated dirty changes. Otherwise create a fresh
purpose-named branch/worktree. Never hard-reset the main worktree.

## Read First

Read only what is needed, in this order:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
5. `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
6. `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`
7. `proofs/v014/empirical_memory_map.md`
8. `proofs/v014/memory_manager_260609.md`
9. `proofs/v014/parallel_memory_fp32_manager.md`
10. `proofs/v014/mythos_kernel_fix_260609.md`

Use JSON proof files for exact numbers when needed.

## Current Known State

- RRTMG leading-column tiling, optics/taumol chunking, SW g-point chunking,
  nested allocator/segmentation, two-way feedback cleanup, and WDM6 `slmsk`
  cleanup are already fixed in the v0.13/v0.14 lineage.
- Commit `a32efce3` closes the live-nest/start-domain `P/MU/W` init family.
- The strict Step-1 one-RK comparison still has real post-init dynamics
  divergence (`P/PH/MU/W` lane). Memory work must not hide or loosen that.
- Long TOST/Switzerland validation remains paused.

## Must Address

Treat these as mandatory work items. Close by implementation/proof, or close by
exact evidence-backed non-material/defer/impossible verdict:

1. Exact-branch memory preflight for the stabilized `a32efce3` lineage.
2. Moisture advection duplicate transport velocity reuse.
3. Non-radiation physics column tiling for any measured material offender.
4. Post-physics non-dry sparse/donated merge if material.
5. Moisture limiter/species workspace reduction if material.
6. PBL/surface bottom-only prep and duplicate diagnostics reuse if material and
   correctness-safe.
7. Acoustic scan carry split / evolving-only carry if compatible with the
   remaining dynamics frontier, otherwise exact defer.
8. Small dycore mask/pad helper cleanup if it naturally belongs to acoustic
   work and is worth the risk.
9. State total/perturbation/base alias reduction only with ADR-quality proof; do
   not silently break state, boundary, restart, wrfout, or validation ABI.
10. FP32 acoustic / mixed perturbation-authoritative mode:
    solve with default-off fp64 bit identity and proof, or prove the exact
    remaining blocker and write the minimal implementation roadmap.
11. Any newly discovered memory issue with material expected VRAM effect.

## Correctness Rules

- Default fp64 production behavior must stay bit-identical unless a change is
  explicitly a semantic or precision-mode change with its own gates.
- No clamps, masks, or tolerance widening to hide physics/grid divergence.
- No CPU-WRF runtime dependency in production.
- No host/device transfer inside timestep loops unless explicitly proven,
  justified, and documented.
- GPU-native, scalable design remains mandatory.
- Separate layout-only fixes from semantic/mixed-precision changes in commits
  and proof reports.

## GPU Rules

- Use one GPU job at a time.
- Prefer CPU/static/HLO proof first; use GPU only for short VRAM/preflight gates
  and only through `scripts/run_gpu_lowprio.sh`.
- Do not start TOST or long Switzerland validation.
- Record allocator mode, peak VRAM, output count, and finiteness when using GPU.

## Required Deliverables

Write these in the Mythos worktree:

- `proofs/v014/mythos_memory_fixes_260609.py`
- `proofs/v014/mythos_memory_fixes_260609.json`
- `proofs/v014/mythos_memory_fixes_260609.md`
- `.agent/reviews/2026-06-09-v014-mythos-memory-fixes.md`
- Updated `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
- Updated `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md` if FP32 status changes
- Sprint closeout files under
  `.agent/sprints/2026-06-09-v014-mythos-memory-lane/`

If source changes are made, commit them on `worker/mythos/v014-memory-fp32` with
clear separated commits:

- layout/bit-identical memory fixes;
- semantic/dycore/FP32 work;
- docs/proofs.

Do not push unless instructed by the manager.

## Required Report Shape

The top-level Markdown must be short enough for manager context:

- One verdict line.
- One table with every known item: status, files changed, expected/measured
  memory gain, correctness gate, merge recommendation.
- One table for GPU proof runs: command, peak VRAM, allocator mode, pass/fail.
- One table for tests/proofs: command, result.
- A ranked list of any remaining deferred/impossible items with exact reason.
- Final merge recommendation: `MERGE_NOW`, `MERGE_PARTIAL`, `REVIEW_ONLY`, or
  `REJECT`.

Detailed evidence goes in JSON.

## Minimum Validation

At minimum:

```bash
python -m py_compile proofs/v014/mythos_memory_fixes_260609.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/mythos_memory_fixes_260609.py
python -m json.tool proofs/v014/mythos_memory_fixes_260609.json \
  >/tmp/mythos_memory_fixes_260609.validated.json
git diff --check
```

Also run focused tests/proofs for each changed module. Examples:

- RRTMG touched: RRTMG bit-identity/chunking proof and short GPU VRAM proof.
- `scan_adapters.py` or physics coupling touched: relevant savepoint tests and
  operational smoke for affected schemes.
- `operational_mode.py`/moisture touched: default-off/default-config bit
  identity, active moisture conservation/positivity, no-transfer audit.
- acoustic/dycore touched: fp64 default bit identity, acoustic savepoint parity,
  warm-bubble/Straka/terrain-rest or an exact blocker.
- state ABI touched: ADR, restart roundtrip, wrfout compatibility, boundary
  package tests.

## Completion Signal

When finished, print and send to the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'MYTHOS MEMORY DONE - see proofs/v014/mythos_memory_fixes_260609.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```

If genuinely blocked, write the same required proof/report files with verdict
`BLOCKED_<exact_reason>` and send:

```bash
tmux send-keys -t 0:2 'MYTHOS MEMORY BLOCKED - see proofs/v014/mythos_memory_fixes_260609.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
