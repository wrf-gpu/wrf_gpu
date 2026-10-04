# P0 — M9 RRTMG radiation bit-identical reduction (GPT, CPU-ONLY build + validate)

## Scope (v0.23 roadmap Table-A, P0)
v0.22.2 already added a default-on **bit-identical** M9 RRTMG radiation column-tile cap
(512 cols, output re-solve only). P0 = a **further bit-identical reduction of redundant M9
radiation work** on the default path (e.g. eliminate a redundant re-solve, or narrow the
output re-solve to exactly the fields the wrfout needs), **default-on and byte-identical to
v0.22.2**. The goal is less device work / VRAM transient at the M9 output boundary — the
*speed/VRAM* win is confirmed later on GPU; your job is the CPU-provable correct reduction.

## CPU-ONLY build + validation (do NOT use the GPU — it is 0:2's lane)
Run everything `taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=` (`-j<=4`). Prove bit-identity CPU-only:
- **HLO/StableHLO before-vs-after** (compile-only): the reduction must not change output numerics (provably output-equal), and should show the removed/narrowed work.
- **Small CPU forecast (or the M9-radiation output path) before-vs-after → byte-identical output fields.**
- The **GPU warm perf/VRAM confirm is DEFERRED** to the validation phase — state the expected win + the exact measurement to run, do NOT block on it.

## Rules
- Default-on but **byte-identical to v0.22.2** (a reduction of redundant work, not a numerics change). If it genuinely cannot be byte-identical, make it opt-in + document why.
- No masking / nan_to_num / shortcuts. Commit on branch `worker/gpt/p0-m9-reduction` (off `origin/main`).

## Deliverable
`proofs/v023/perf_bundle/P0_REPORT.md` (the change + CPU bit-identity proof [HLO + CPU field-compare] + the deferred-GPU-perf measurement note) + `touch proofs/v023/perf_bundle/P0_DONE`. Report to `0:1` (`scripts/tmux_submit.sh 0:1 '<msg>'`). If it cannot progress without the GPU → `P0_QUESTION.md` + ping `0:1` (do NOT fake it).
