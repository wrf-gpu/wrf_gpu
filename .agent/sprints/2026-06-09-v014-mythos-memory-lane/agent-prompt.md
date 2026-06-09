You are Mythos, running in tmux `0:1`, assigned the full v0.14 memory/FP32
lane for `/home/enric/src/wrf_gpu2`.

Read and obey:

- `/home/enric/src/wrf_gpu2/.agent/sprints/2026-06-09-v014-mythos-memory-lane/sprint-contract.md`

Hard endpoint:

- Fix all known memory issues where technically safe.
- Fix any additional material memory issue you discover.
- Include FP32/acoustic/mixed-precision: solve it into a default-off or
  production-safe proven mode if feasible; otherwise prove the exact blocker
  and write the minimal remaining roadmap.
- Prove correctness and memory/VRAM claims. No proof, no done.

Critical coordination:

- Work only in isolated worktree
  `/home/enric/src/wrf_gpu2/.codex/worktrees/mythos-memory-v014`, branch
  `worker/mythos/v014-memory-fp32`, based on commit `a32efce3`.
- Do not edit the main worktree.
- Do not start TOST or long Switzerland validation.
- Use GPU only for short memory/preflight gates via `scripts/run_gpu_lowprio.sh`.
- Do not send Hermes/Telegram.
- Commit your branch locally, but do not push unless the manager asks.

Completion marker:

`MYTHOS MEMORY DONE - see proofs/v014/mythos_memory_fixes_260609.md`

Blocked marker:

`MYTHOS MEMORY BLOCKED - see proofs/v014/mythos_memory_fixes_260609.md`
