# GPT full-working-set fp32 adversarial cross-check

**Worker:** GPT-5.5 xhigh on `worker/gpt/v016-fp32-fullws-refute`
**Date:** 2026-06-14
**Scope:** independent reproduction + adversarial attempt to break the claim that valid fp32 cannot reach the ~4.3x all-fp32 cost-proxy ceiling.

## Verdict

**VERDICT: IMPOSSIBILITY CONFIRMED (valid fp32 ceiling is ~1.1x on this implementation, with 0% VRAM peak reduction from precision alone).**

I could not find a valid-numerics lever toward >2x, let alone ~4x. The only measured ~4.29x path is the invalid all-fp32/x64-off cost proxy. The valid path keeps base/total absolutes, dry-mass cancellation anchors, and qke/MYNN stability-critical work effectively fp64; the remaining fp32 work gives a real but small ~1.1x speedup and does not reduce the transient-dominated peak.

## Reproduction

Command, under GPU lock:

```bash
bash scripts/with_gpu_lock.sh --label gpt-fullws-xcheck -- \
  taskset -c 0-3 env PYTHONPATH=src:.:proofs/perf/v015/km_bench \
  JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  TF_GPU_ALLOCATOR=cuda_malloc_async GPUWRF_FP32_FULLWS=1 \
  GPUWRF_MIXED_FP32_COMPACT=1 \
  python proofs/perf/v016/fullws_fp32_km_bench.py \
  --steps 24 --tiles 1x1,2x2 --precisions fp64,fullws \
  --out proofs/perf/v016/gpt_fullws_reproduce.json
```

The lock wrapper released normally.

| ncol | fp64 ms/step | fullws ms/step | speedup | fp64 peak | fullws peak | VRAM ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 | 70.658 | 63.950 | 1.105x | 3.73078 GiB | 3.73078 GiB | 1.000 |
| 65,536 | 255.257 | 229.852 | 1.111x | 11.42603 GiB | 11.42603 GiB | 1.000 |

This confirms the Opus headline. The persistent State did shrink substantially:
16k fp64 State bytes went from 352.5 MiB fp64 + 13.8 MiB fp32 to 113.2 MiB fp64 + 133.4 MiB fp32; 65k went from 951.4 MiB fp64 + 55.0 MiB fp32 to 251.9 MiB fp64 + 404.8 MiB fp32. Peak VRAM did not move.

## Peak Attribution

Direct GPU XLA `memory_analysis()` probe, also under GPU lock, released normally:

```bash
bash scripts/with_gpu_lock.sh --label gpt-fullws-xcheck -- \
  taskset -c 0-3 env PYTHONPATH=src:.:proofs/perf/v015/km_bench \
  JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  TF_GPU_ALLOCATOR=cuda_malloc_async GPUWRF_FP32_FULLWS=1 \
  GPUWRF_MIXED_FP32_COMPACT=1 \
  python proofs/perf/v016/fullws_transient_memory_probe.py \
  --steps 4 --tiles 1x1 --out proofs/perf/v016/gpt_fullws_transient_memory_probe.json
```

| lane | temp arena | args | outputs | HLO f64 MiB | HLO f32 MiB | f64 float fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp64 | 1974.98 MiB | 366.30 MiB | 366.30 MiB | 307274 | 1037 | 0.9966 |
| broad fullws | 2001.23 MiB | 246.66 MiB | 246.66 MiB | 240048 | 26764 | 0.8997 |

The evidence supports "peak is transient, not persistent State": broad fullws saves about 120 MiB in arguments/outputs at 16k, but the temp arena is about 2.0 GiB and actually increases by ~26 MiB. That matches the unchanged real peak.

I also ran a CPU-lowered structural variant probe. It is not a GPU timing claim, but it localizes the arena:

| CPU-lowered variant | temp arena | HLO f64 MiB | HLO f32 MiB | interpretation |
| --- | ---: | ---: | ---: | --- |
| fp64 | 5305 MiB | 838564 | 3140 | baseline |
| fullws | 5379 MiB | 767543 | 148563 | precision demotion does not reduce temp |
| fullws_norad | 5379 MiB | 767543 | 148563 | radiation off does not move this graph |
| fullws_no_pbl | 1273 MiB | 405504 | 142439 | PBL/MYNN is a large arena contributor |
| fullws_no_physics | 885 MiB | 257864 | 135619 | physics removal collapses arena, but is not a valid forecast |

## Refutation Attempts

**1. Physics working set fp32.** I did not find a sound physics-fp32 path. Radiation is not the peak in either the prior GPU no-radiation control or my CPU-lowered variant probe. PBL/MYNN is a large temp contributor, but the valid model cannot simply demote it: the qke family is pinned by the prior 1 km finiteness failure where qke was the sole nonfinite field. Removing PBL/physics reduces the arena, but that is not a valid precision lever. The real arena lever is algorithmic MYNN BouLac O(nz), not fp32 storage/compute.

**2. Dycore transient fp32.** The compact acoustic islands already demote the safe transient work and keep cancellation in explicit fp64 islands. My reproduction shows this and broad State demotion still produce only ~1.11x and 0% VRAM. The direct GPU memory analysis shows why: HLO f64 declarations drop, but the temp arena does not.

The base-absolute oracle remains decisive: storing `p_total`/`ph_total` fp32 corrupts the geopotential/PGF differences 26.85x and 126.75x beyond the perturbation fp32 floor. The fp64 island cannot recover bits lost at storage. I also found the `safe` mode alias trap: `State.replace(p=fp32, ph=fp32)` syncs those aliases into `p_total`/`ph_total`. HEAD now excludes `p`/`ph` from `FULLWS_FP32_FIELDS_SAFE`, leaving only `p_perturbation`, `ph_perturbation`, and `w`.

**3. Double-single.** The CPU cancellation microprobe recovers the fp64 cancellation bits, but it uses two fp32 words per scalar, so storage is 8 bytes/scalar, and the microprobe was ~16.0x slower than fp64 subtraction on CPU. This can preserve accuracy, but it is not a path to the 2x storage or ~4x speed ceiling.

**4. Arena reduction.** The meaningful reductions I measured are not precision reductions. Radiation-off is unchanged; no-PBL/no-physics collapses temp but removes required physics. Donation/remat/XLA scheduling may reduce both lanes in future, but the present precision experiment shows no fp32-specific arena reduction: fullws temp is slightly larger than fp64 in direct GPU `memory_analysis()`.

**5. Cost proxy validity.** `true_fp32_cost_proxy.json` is a valid upper-bound cost proxy for invalid all-fp32 compute: 16k true-fp32 is 16.436 ms/step versus 70.492 ms/step fp64, a 4.289x speedup. It is not a valid step-speed claim for WRF-faithful numerics because it turns off the exact fp64 compute/storage pins that make the forecast finite and conservation-safe.

## Proof Objects

- `proofs/perf/v016/gpt_fullws_reproduce.json` - independent GPU reproduction of 16k/65k timing and peak VRAM.
- `proofs/perf/v016/gpt_fullws_transient_memory_probe.json` - direct GPU XLA temp/arg/output memory analysis.
- `proofs/perf/v016/gpt_fullws_transient_memory_probe_cpu.json` - CPU-lowered structural temp check, secondary only.
- `proofs/perf/v016/gpt_fullws_variant_memory_cpu.json` - CPU-lowered no-radiation/no-PBL/no-physics arena localization.
- `proofs/perf/v016/gpt_safe_alias_probe.json` - alias-sync proof explaining why safe mode must not demote `p`/`ph`.
- `proofs/perf/v016/gpt_double_single_probe.json` - double-single cancellation/storage/cost probe.

## Handoff

- objective: independently reproduce fullws fp32 performance/VRAM and try to refute the impossibility verdict.
- files changed: this review; `gpt_fullws_transient_memory_probe.json`; completed `gpt_fullws_variant_memory_cpu.json` records. Earlier related artifacts on the branch include the safe-alias and double-single probes.
- commands run: fullws GPU bench; GPU transient-memory probe; CPU-lowered transient probe; CPU-lowered arena variant probe; safe-alias probe; double-single probe; JSON/stat/status inspections.
- proof objects produced: listed above.
- unresolved risks: I did not run a production-valid long-horizon safe-mode forecast in this cross-check. That is a numerics acceptance gate, not a path to the missing >2x speed/VRAM lever.
- next decision needed: ship/report the ~1.1x valid fp32 lane plus algorithmic MYNN BouLac O(nz) as separate work; stop treating the invalid ~4.3x all-fp32 proxy as a reachable valid-numerics target without a new conservation/physics algorithm.

Co-Authored-By: GPT-5.5 <noreply@openai.com>
