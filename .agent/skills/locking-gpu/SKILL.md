---
name: locking-gpu
description: The single mandatory protocol for serializing the one shared GPU across all parallel agents — lock before every GPU use, auto-free after.
---

## When to use

ALWAYS, for EVERY command that touches the GPU — any JAX/CUDA/model run: gates,
benchmarks, forecasts, short coupled smokes, savepoint emits on GPU, nsys/CUPTI
captures, VRAM probes, identity gates. The workstation has exactly ONE GPU. Two
agents running GPU work at the same time collide, OOM, or crash the box. This
protocol guarantees mutual exclusion. It is non-optional and applies to every
agent (manager, worker, reviewer) in every worktree.

## The protocol — one canonical wrapper

Wrap every GPU command with the shared lock script. It blocks until the lock is
free, runs the command while holding the lock, and **frees the lock
automatically the instant the command exits** (even on crash/timeout — the lock
lives on an open fd that closes on exit). There is no separate "unlock" step to
forget.

```
scripts/with_gpu_lock.sh [--timeout SECONDS] [--label NAME] -- <gpu-command> [args...]
```

- `--label NAME` — your agent/sprint label (shows in the holder file so siblings
  see who holds the GPU). Default `<user>:<pid>`.
- `--timeout SECONDS` — max wait to acquire (default 7200 = 2 h). On timeout the
  wrapper exits 124; do ONE retry in a fresh process, then report GPU-blocked.
- The lock file is `/tmp/wrf_gpu2_gpu.lock`; the current holder is in
  `/tmp/wrf_gpu2_gpu.lock.holder` (label, pid, since, cmd). Read it to see who
  holds the GPU; never delete the lock file.

Standard GPU env (compose with the wrapper):
`taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src JAX_ENABLE_X64=true
XLA_PYTHON_CLIENT_PREALLOCATE=false GPUWRF_CANAIRY_ROOT=/mnt/data/canairy_meteo`.

## Lock-per-command vs lock-per-sequence

- **One GPU step** → wrap that one command. The lock frees on exit, letting a
  waiting sibling take the GPU next. This is the default and the politest.
- **A sequence that must not be interrupted** (e.g. a multi-run A/B where an
  interleaved sibling would perturb timing, or a warmup+measure pair) → wrap the
  WHOLE sequence in ONE `with_gpu_lock.sh -- bash -c '...'` invocation so the
  lock is held throughout. Do NOT hold the lock longer than the GPU work needs.

## Hard rules

- **NEVER launch a GPU process outside the wrapper.** A single un-wrapped GPU
  command breaks the guarantee for everyone. This is a protocol violation.
- **CPU-only work does NOT take the lock** — source reading, building oracles,
  writing code, analyzing on-disk artifacts. Holding the lock for CPU work
  starves GPU siblings. Do all non-GPU work first; acquire the lock only for the
  actual GPU run.
- **Kill orphan GPU procs before launching** and after a crash; verify GPU
  sanity (`nvidia-smi`) when you acquire after another agent's long run.
- **Hibernation kills the CUDA context.** The box can suspend; a CUDA context
  does NOT survive it. Kill and rerun any GPU command whose wall-clock spanned a
  suspend (huge gap, context errors).
- **GPU hand-off caveat:** an agent that arms a "GPU-free monitor" can grab the
  lock the instant it frees — which may be a GAP inside a sibling's multi-run
  sequence, not its true end. If your sequence must be atomic, use
  lock-per-sequence (above), not lock-per-command.

## Manager responsibility

Every GPU-worker dispatch MUST (a) name this skill, (b) give the exact
`with_gpu_lock.sh --label <name> -- <cmd>` invocation, and (c) state that the GPU
is shared and serialized so the worker does CPU work first and the flock will
queue it behind a sibling. The manager never runs two GPU workers that could
bypass the lock, and verifies GPU sanity when a holding worker's completion
notification arrives.

## Validation

Mutual exclusion is provided by `flock(2)` on the shared lock file — correct as
long as every GPU command goes through the wrapper. Confirm by reading the
holder file while a sibling runs; a second `with_gpu_lock.sh` call prints
"waiting for GPU lock; current holder: ..." until the first releases.
