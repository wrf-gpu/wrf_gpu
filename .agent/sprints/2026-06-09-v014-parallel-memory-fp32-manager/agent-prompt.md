You are GPT-5.5 xhigh, a SECONDARY manager for wrf_gpu2 v0.14 memory/FP32.

Worktree:

- `/home/enric/src/wrf_gpu2/.codex/worktrees/v014-memory-fp32-manager`
- branch `worker/gpt/v014-memory-fp32-manager`

Read and obey:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-parallel-memory-fp32-manager/sprint-contract.md`
4. `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
5. `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`
6. `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

Goal:

Advance v0.14 memory and FP32 work in parallel without colliding with the
primary manager's fp64 grid-parity debug. Implement only non-conflicting
bit-identical memory fixes. For FP32, maximize proof/analysis and prove
infeasible blockers if source work would collide.

Important current primary-debug lock:

- Do NOT edit `src/gpuwrf/dynamics/**`, `src/gpuwrf/runtime/operational_mode.py`,
  `src/gpuwrf/integration/d02_replay.py`, `src/gpuwrf/nesting/**`,
  state/restart/live-nest/base-state init files, or boundary/carry files.
- Current fp64 debug target is live-nest raw child -> live child
  perturbation-state initialization for `P/MU/W`.

Work style:

- Start by building a collision map.
- Use CPU-only by default.
- Use GPU only through `scripts/run_gpu_lowprio.sh` and only for short
  validation/preflight, yielding to primary-manager GPU campaigns.
- Commit and push your branch only after validation.
- Report top-level, context-sparing results.

Deliver:

- `proofs/v014/parallel_memory_fp32_manager.py` if useful
- `proofs/v014/parallel_memory_fp32_manager.json`
- `proofs/v014/parallel_memory_fp32_manager.md`
- `.agent/reviews/2026-06-09-v014-parallel-memory-fp32-manager.md`
- optional source/proof commits on `worker/gpt/v014-memory-fp32-manager`

Your final recommendation must be exactly one of:

- `MERGE_NOW`
- `REVIEW_ONLY`
- `DO_NOT_MERGE`

At completion, print a handoff and attempt:

```bash
tmux send-keys -t 0:2 'GPT PARALLEL_MEMORY_FP32_MANAGER DONE - see proofs/v014/parallel_memory_fp32_manager.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
