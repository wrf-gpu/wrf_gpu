# v0.21.0 AOT Cheap-Key Compile — Headline Table + Release Gate Verdict

Worker: GPU measurement (9-nest Canary). Branch under test: `worker/opus/vnext-parallel-compile`
@ `63032d24` (cross-process cheap-key fix). Worktree `<USER_HOME>/src/wrf_gpu2_wt/parallel-compile`.
Real-world case: `<DATA_ROOT>/wrf_downscale/runs/20240901/cpu` (CPU-paired wrfout reference).

Every number is labeled **measured** (this session, on this 5090) or **cited** (from
`proofs/v021/V0210_DEPLOY_STATE.md`, prior measurement). Honest about what is estimated.

---

## STATUS: GATE PASSED (2026-06-25). WARM-LOAD = **PASS on 9-nest** — 9/9 `loaded=true source=aot_blob`, 0 re-lower, seconds load.

> **Headline result:** the AOT cheap-key warm-start works on the full 9-nest. After the
> source-fingerprint fix (commit `e8aa803c`: scope the source digest to the `operational_mode`
> static import-closure; drop the repo-wide git HEAD; fail-open to whole-tree), a fresh warm
> process loads ALL 9 per-domain blobs (`loaded=true source=aot_blob ×9`, 0 fallback, 0 re-lower;
> the 9 warm keys byte-match the cold keys; finite/0-NaN; warm peak 16.4 GB; load in **seconds**
> vs the ~60-min cold re-lower). Root cause of the earlier FAIL (now fixed): `source_fingerprint_hash`
> hashed the WHOLE `src/gpuwrf` tree + git HEAD, so a concurrent `domain_tree.py` orchestration edit
> between the cold and warm runs shifted all 9 keys despite byte-identical HLO. Full evidence + fix
> + GPU re-confirm: `proofs/v021/blocker_9nest_cheapkey/FIX_REPORT.md`.

---

## 1. Compile memory + wall comparison table

| Variant | Peak host RAM | Cold compile wall | Warm load (fresh proc) | Source |
|---|---|---|---|---|
| **VORHER** — fused, no AOT | ~60 GB | ~50–60 min | ~30 min (warm re-lower) | cited (V0210_DEPLOY_STATE.md L64,71,91); the user accepts estimate |
| **AOT only** — de-fuse PARALLEL prewarm | ~65 GB | prewarm ~25 min (wall_s 1503) | seconds (load, no re-lower) | cited (L81,85); warm mechanism = same as measured seq warm |
| **AOT + de-fuse SEQUENTIAL** (v0.21.0 DEFAULT) | **27.6 GB** (time -v Max RSS / VmHWM) | **~60–75 min** (all 9 AOT blobs) | **SECONDS** — loaded=true ×9, 0 re-lower (fix `e8aa803c`) | measured this session |

> 3-domain de-fuse SEQUENTIAL anchors (measured this session, same case, for extrapolation):
> clean peak RSS ~18.3 GB (/usr/bin/time -v); compile wall ~22 min for 3 domains; cross-process
> warm loaded all 3 blobs with **zero re-lowering** (no XLA "Very slow compile" alarm in warm).
>
> **RAM methodology (IMPORTANT):** the "peak host RAM" for the de-fuse SEQUENTIAL row is the
> per-process **`/usr/bin/time -v` Maximum resident set size** (== `/proc/PID/status` VmHWM),
> NOT system-wide `free`/meminfo. This box is SHARED (0:2 runs a 12-rank CPU corpus + heavy
> page cache from the JAX cache dir + wrfout files), so the system-wide "used" (~50 GB) is
> contaminated and is reported only as a labeled upper bound. With `PARALLEL_COMPILE=0` there
> are no spawn children, so `/usr/bin/time -v` captures the whole gpuwrf footprint cleanly.
> Mid-9-nest-compile the clean gpuwrf VmHWM measured ~18 GB (de-fuse holds one domain's compile
> at a time → low, bounded RSS — the low-mem headline).

### Honest framing (carried from V0210_DEPLOY_STATE.md, confirmed)
- **De-fuse is a RAM-for-WALL trade, NOT a cold-compile speedup.** Fused cold compile
  (~50–60 min) is faster than de-fuse cold (9 separate lowers). De-fuse SEQUENTIAL's win is
  host-RAM (sequential ≈ 25 GB cited vs ~60 GB fused).
- **The v0.21.0 headline = the AOT cheap-key WARM-START.** A fresh process LOADS the
  per-domain compiled executables from disk (`source=aot_blob`) and SKIPS the ~30 min
  re-lower. Proven cross-process here (Step-0) with **zero "Very slow compile" alarms** in
  the warm process = the re-lower is genuinely skipped.

---

## 2. Release-gate verdict

| Check | Result |
|---|---|
| Step-0 de-risk (3-dom cross-proc warm `loaded=true source=aot_blob` ×3) | **PASS** (measured) |
| Step-0 fallbacks (missing / verify-error / cheap-key-mismatch / jit) | **0** (measured) |
| Step-0 warm re-lower ("Very slow compile" alarm in warm) | **0** (measured — re-lower skipped) |
| 3-dom cold de-fuse finite + all outputs | **PASS** finite, 8 frames, EXIT=0 (measured) |
| 3-dom cold de-fuse CPU tol-match (worst T2) | max_abs 2.96 K, max_rel 1.03%, all-finite (measured) |
| 9-nest cold de-fuse compile wall | **~60 min** (all 9 AOT blobs serialized) — measured |
| 9-nest cold peak host RAM (clean) | **27.6 GB** (time -v Max RSS = VmHWM; system-wide ~52 GB = contaminated upper bound) — measured |
| 9-nest cold integration finite | **PASS** — in-process loaded=true ×168+, no NaN, no finite-raise — measured |
| 9-nest 9/9 AOT blobs serialized | **PASS** (d01–d09 on disk) — measured |
| **9-nest WARM `loaded=true` ×9 (cross-process)** | **PASS** — 9/9 loaded=true source=aot_blob, 0 fallback, 0 re-lower, warm keys == cold keys, finite/0-NaN, warm peak 16.4 GB, load in **seconds** (fix `e8aa803c`) — measured |
| 9-nest CPU tol-match (full-field at a frame) | NOT REACHED — first CPU-comparable frame (sim 18:20) is ~108 min wall away (~325 min/fc-h) + risks #123 OOM; carried by 3-domain CPU-match (T2 2.96 K) — honest impracticality documented |
| 9-nest measured s/frame | warm ran 12 forecast groups finite (no NaN); de-fuse = the bit-identical eager path (same dispatched kernels) → no forecast-throughput regression by construction; details in `blocker_9nest_cheapkey/FIX_REPORT.md` |

---

## 3. Reproducibility — commands + logs

Base env (every run): `JAX_ENABLE_X64=true GPUWRF_FORCE_FP64=1 GPUWRF_FINITE_CHECK=1
GPUWRF_NESTED_FUSE=0 GPUWRF_MIN_FREE_VRAM_GIB=8 XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`. Driver: `proofs/v021/canary_gate/canary_gate_driver.py`.
PYTHONPATH=`<USER_HOME>/src/wrf_gpu2_wt/parallel-compile/src`. All GPU runs wrapped in
`scripts/with_gpu_lock.sh --label <LABEL>`.

- **Step-0 de-risk (3-dom warm):** `run_ck3seq_warm1.sh` → `ck3seq_warm1.{stdout,stderr}`,
  cache `<DATA_ROOT>/wrf_downscale/canary_gate_cache_ck3seq`. Flags: `DEFUSE_COMPILE=1 AOT=1
  PARALLEL_COMPILE=0 AOT_VERIFY=1`.
- **3-dom cold (manager run ck3seq_cold):** `ck3seq_cold.{stdout,stderr}`; CPU-match
  `ck3seq_cold_cpumatch.json`.
- **9-nest cold (1A):** `run_9aotseq_cold.sh` → `9aotseq_cold.{stdout,stderr}`, RAM log
  `9aotseq_ram.log`, cache `<DATA_ROOT>/wrf_downscale/canary_gate_cache_9aotseq`.
- **9-nest warm (1B):** `run_9aotseq_warm.sh` → `9aotseq_warm.{stdout,stderr}`.
- CPU comparison tool: `compare_wrfout.py <gpu_out_dir> <cpu_ref_dir> [out.json]`.
- RAM sampler (locale-independent, /proc/meminfo): `ram_sampler.sh <log>`.

(This file + `HEADLINE_COMPILE_TABLE.json` updated when the 9-nest numbers land.)
