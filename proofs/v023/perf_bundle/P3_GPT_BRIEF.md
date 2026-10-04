# P3 — 2-domain root fusion (GPT, CPU-ONLY build + validate)

## Scope (v0.23 roadmap Table-B/A, P3)
Fuse the **2-domain** (parent + one nest) root integration into **one compiled XLA program**
per step (fewer kernel launches / less host-side orchestration overhead), **bit-identical**
to the current 2-domain path. The *launch-count / wall* win is confirmed later on GPU; your
job is the CPU-provable fusion + identity.

## CPU-ONLY build + validation (do NOT use the GPU — it is 0:2's lane)
Run everything `taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=` (`-j<=4`). Prove CPU-only:
- The fused 2-dom program **compiles** (CPU) and the **HLO shows the fusion** (fewer separate
  programs / fewer top-level launches than the unfused 2-dom path — quantify the launch/program count delta).
- A **small 2-dom CPU run before-vs-after → byte-identical output** (the fusion must not change results).
- The **GPU warm perf confirm** (launch-count → wall reduction) is **DEFERRED** to the validation phase — state the expected win + the measurement, do NOT block on it.

## Rules
- Bit-identical to the current 2-dom path; opt-in (env-gated) if it cannot be default-byte-identical. No masking / shortcuts.
- Commit on branch `worker/gpt/p3-root-fusion` (off `origin/main`).

## Deliverable
`proofs/v023/perf_bundle/P3_REPORT.md` (the change + CPU bit-identity + HLO-fusion/launch-count proof + the deferred-GPU-perf note) + `touch proofs/v023/perf_bundle/P3_DONE`. Report to `0:1`. If it cannot progress without the GPU → `P3_QUESTION.md` + ping `0:1` (do NOT fake it).
