# v0.21.0 B200 rerun — canonical params + expected values (manager source-doc for 0:3's the user playbook)

Author: 0:1 (manager), 2026-06-26. Sourced from deploy_pods.md, B200_WALLCLOCK_STRATEGY.md, xla_autotune.py,
compile_cache.py, aot_precompile.py, and the validated 2026-06-23 B200 run env. Honest MEASURED vs PROJECTED labels.
**0:3: fold this into your tabular Pod-ID playbook for the user. Two RED corrections are flagged ⚠ — do not miss them.**

## 1. Canonical runtime env/flags (B200 = Blackwell sm_100, pod cu128 / CUDA 12.8)
| Var | Value | Why / source |
|---|---|---|
| jaxlib | `jax[cuda12]==0.10.0` | matches release local jaxlib, supports Blackwell. deploy_pods.md:96. Reuse the validated 2026-06-23 venv build — do NOT bump. |
| `JAX_ENABLE_X64` | `true` | fp64 dynamics. env.txt 2026-06-23 |
| `GPUWRF_FORCE_FP64` | `1` | this rerun is fp64 (paper ladder). env.txt 2026-06-23 |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | pod runs; avoids grabbing all HBM up front. deploy_pods.md:114 |
| `GPUWRF_ALLOCATOR` | `cuda_async` | stream-ordered pool, BFC-frag workaround (default). cli.py:339 |
| `GPUWRF_WRF_ROOT` | `<staged WRF pristine>` | .TBL/.F lookups (#115 carried). env.txt 2026-06-23 |
| `OMP_NUM_THREADS` | `4` | env.txt 2026-06-23 |
| ⚠ `JAX_COMPILATION_CACHE_DIR` | **`<S3-VOLUME>/b200_jaxcache_sm100_v0210`** | **MUST be on the persistent S3 network volume, NOT pod-local `/tmp`** (the 2026-06-23 run used `/tmp/b200_jaxcache` = ephemeral → no cross-pod/region reuse). This is THE change that makes the cache survive + warm-reuse. compile_cache.py:93-120 |
| `GPUWRF_JAX_CACHE_DIR` | same as above | belt-and-suspenders (same resolver). compile_cache.py |
| `JAX_LOG_COMPILES` | `1` | makes cold-vs-warm visible in logs (the monitoring signal). |
| `GPUWRF_XLA_PARALLEL_COMPILE` | `1` (opt-in) | faster cold compile, `min(cpu,8)` threads, probe-validated. xla_autotune.py:495. Safe, no numeric change. |
| `GPUWRF_XLA_AUTOTUNE_CACHE` + `..._DIR=<S3-VOL>/autotune_sm100_v0210` | opt-in | persist GPU autotune across pods (extra warm win). xla_autotune.py:364 |
| `GPUWRF_NESTED_AOT` | `1` ONLY for the nested path | the v0.21 cheap-key AOT blob warm-start is for the NESTED pipeline. For a SINGLE-domain 1024² it is the **JIT persistent cache** (the dir above) that delivers warm reuse — AOT not required. aot_precompile.py |

## 2. Expected values
| Quantity | Value | MEASURED? |
|---|---|---|
| B200 1024² single-domain fp64 runtime | **475.97 s/fc-h, 34.90M cell-steps/s, peak VRAM 58.25 GB** | ✅ MEASURED 2026-06-23 (Swiss geometry; Alps 1024² ≈ same cost). fp64_scaling.json |
| VRAM headroom | 58 GB of 192 GB HBM3e ≈ 30% → **no OOM risk at 1024² single-domain** | ✅ derived from measured |
| Cold compile, 1024² single domain, B200 | **PROJECTED ~10–40 min** (smaller HLO than the 9-nest; 9-nest fused cold ≈ 50–60 min/~60 GB on 5090). MEASURE on first run. | ⚠ NOT MEASURED standalone |
| Warm-start Region 2 (SAME 1024² shape) | **cache hit → near-instant compile (seconds–low min), then pure compute 475 s/fc-h.** 9-nest AOT warm-load was SECONDS; JIT mini-run grew 470→1744 cache files in a 1h warm reuse. | ✅ analog measured; exact 1024² number = measure |
| Warm-start Region 2 if shape DIFFERS | second COLD compile (HLO keyed by shape). Keep Region 1 & 2 grid dims IDENTICAL for reuse. | mechanism, compile_cache.py |
| B200 vs 5090 | ~3.49× sustained 1024² (conservative 2.7× in budget) — 5090 warm steady is a placeholder, not measured | ⚠ ratio rests on a 5090 placeholder |

## 3. Cache-warm proof signals (what to monitor)
- **Cold compile happened:** log lines `Finished XLA compilation of jit(...) in N sec` (from `JAX_LOG_COMPILES=1`).
- **Warm cache HIT:** log lines `Persistent compilation cache hit for ...`. Codebase canonical detector = `cache_entry_count()` delta: warm ⇔ `after==before and before>0` (compile_cache.py:254).
- **Cache physically on S3 volume:** `JAX_COMPILATION_CACHE_DIR` must resolve under the network-volume mount (check `df`/`mount` on that path, NOT `/tmp`); file count + bytes grow on cold (sm_100 1024² prior run → 3501 files / 219 MB), then STABLE on warm. Snapshot `ls $CACHE | wc -l` + `du -sh $CACHE` before/after each region.
- **Reuse proven:** restore the S3 cache before Region 2 → expect cache-hit lines + near-zero compile + stable file count. cache_delta single-domain expected **0** on warm. (For the 9-nest path, bounded cache_delta=2 is EXPECTED — K=2 re-lower — not a failure.)

## 4. STOP criteria (paid-pod cost protectors)
| Trip | Threshold | Action |
|---|---|---|
| ⚠ **Stability (Alps terrain)** | the 2026-06-23 **1024² Alps REGIONAL nest FAILED — steep-terrain d02 divergence** (carried Mont-Blanc / v0.21.1 dycore). Any non-finite / divergence. | STOP immediately; fall back to the validated **Swiss 1024²** (stable). Do NOT burn pod hours re-trying the Alps nest blind. NEXT_PORT_ACTIVATION_RUNBOOK.md gates this. |
| Recompile when warm expected | `Finished XLA compilation` lines on Region 2 for the same shape | STOP — cache not restored / wrong path (`/tmp` vs volume) / sm mismatch. Fix before spending compute. |
| cache_delta unexpected | single-domain warm shows delta>0 (or 9-nest >2) | STOP — key mismatch (version/shape/device). Diagnose, don't proceed. |
| VRAM / OOM | >150 GB used (of 192) or any `cudaErrorMemoryAllocation` | STOP — shape/leak bug (1024² should sit at ~58 GB). |
| Tempo regression | s/fc-h >20% over the 475.97 anchor (~>571) | STOP — allocator/contention/fp64-island regression; investigate. |

## 5. Teardown (so the cache is actually reusable)
Sync `$JAX_COMPILATION_CACHE_DIR` (and the autotune dir) back to S3 **before** pod terminate; verify object count on S3 ≈ local file count. deploy_pods.md §termination. Next pod restores it → Region 2 warm. Save the run bundle (logs + fp64 timing json) before `podTerminate`.
