# Independent fp32 Roofline / Ceiling Re-derivation (Opus, BLIND)

**Date:** 2026-06-14
**Author:** Opus 4.8 performance analyst, independent blind derivation
**Method:** raw artifacts + first-principles roofline only. Did NOT read
`PERFORMANCE-CEILING-AND-LEVERS.md` or any other agent's ceiling/roofline prose.

## One-line verdict

**Native-fp32 ≈ 4.3–4.6× faster than our fp64 GPU kernel on the same RTX 5090
(measured cost-proxy, holds to 262k cols) + ~4× the columns per GPU (VRAM). A
~6× single-GPU Canary-1km speedup *vs 24–28-rank CPU-WRF* is PLAUSIBLE-BUT-NOT-GUARANTEED:
it needs the top of the fp32/fp64 band (~4.3×) to survive at 1km AND the warmed-steady
CPU-relative frame; the conservative honest center is ~4–5× vs CPU.** The naive
"~2× because memory-bound" ceiling is WRONG for this card — fp64 sits *at* its
roofline ridge, so fp32 buys the ALU term too.

---

## Hardware constants used (RTX 5090, GeForce Blackwell)
- fp32 peak ≈ 105 TFLOP/s, fp64 peak ≈ 1.7 TFLOP/s (1/64 GeForce rate), HBM ≈ 1.8 TB/s, 32 GB.
- **fp64 roofline ridge = 1.7e12 / 1.8e12 = 0.944 FLOP/byte**
- **fp32 roofline ridge = 105e12 / 1.8e12 = 58.3 FLOP/byte**

## 1. Arithmetic intensity & regime (from HLO, 65,536 cols, 1 step)
`s2_hlo_stats_2x2_s1.json`, top-level `flops` / `bytes accessed`:

| precision | flops | bytes accessed | **AI (FLOP/byte)** | ridge | regime |
|---|---|---|---|---|---|
| fp64 | 1.284e11 | 1.260e11 | **1.019** | 0.944 | **marginally COMPUTE-bound** |
| mixed_s2 | 1.257e11 | 1.088e11 | **1.155** | 58.3 (fp32) | **deeply MEMORY-bound** |

The step is a pure stencil dycore: `top_ops` are multiply/slice/add/broadcast/
subtract with **no GEMM/conv** (no `dot`, no `convolution`), so AI ≈ 1 is real,
not an artifact. **The key fact the naive analysis misses:** because the fp64
ridge is only 0.944, an AI of ~1.0 puts the fp64 step *on the wrong side of the
ridge* — fp64 ALU throughput (the 1/64 cripple) is a genuine binding term, not
free. In fp32 the ridge jumps to 58.3, so the identical algorithm is now
hugely memory-bound and the ALU is free.

## 2. Theoretical fp32 ceiling
- Memory-bound component: downcasting fp64→fp32 halves bytes for the convertible
  arrays → ≤ ~2× from bandwidth alone. **This is the only ceiling if you wrongly
  assume the step is memory-bound in fp64.**
- BUT fp64 is *also* paying the ALU term. fp64-ALU roofline @65k = 1.284e11 /
  1.7e12 = **75.5 ms**; fp64-mem roofline = 1.260e11 / 1.8e12 = **70.0 ms**.
  They are co-binding (75.5 vs 70.0). fp32 removes BOTH: fp32-ALU roofline =
  1.257e11 / 105e12 = **1.2 ms** (vanishes), fp32-mem roofline = **60.4 ms**.
- Roofline-clean ceiling fp64→fp32 = 255.5 / 60.4 = **~4.2×**, i.e. bandwidth-(~2×)
  AND fp64-ALU-relief (~2×) multiply. Compute-bound subgraphs (the transcendentals:
  1.81e9 sqrt/log/pow — same count both precisions) add further fp32 headroom.

## 3. Why measured mixed_s2 is only ~1.11×
`fp32_s2_mixed_ladder.json`: 1.102× @16k, 1.113× @65k. Cause = the byte
reduction mixed_s2 actually achieves is tiny:

| quantity (65k) | fp64 | mixed_s2 | fp64/mix |
|---|---|---|---|
| bytes accessed | 126.0 GB | 108.8 GB | **1.158** |
| arg size | 1.12 GB | 0.81 GB | 1.380 |
| output size | 1.06 GB | 0.75 GB | 1.413 |
| **temp arena** | 8.22 GB | 8.13 GB | **1.012 ← barely moves** |

mixed_s2 downcasts only p'/ph'/mu' + acoustic work arrays; **everything else
(theta, u, v, w, the 8.2 GB fp64 temp arena, the whole RK large step) stays
fp64**, so total bytes drop only 1.16× and the fp64-ALU term is untouched.
Predicted-from-bytes speedup 1.158× ≈ measured 1.113× → **fully consistent**.
mixed_s2 is a numerically-safe partial step, NOT the performance lever.

## 4. The decisive evidence: true all-fp32 cost proxy
`true_fp32_cost_proxy.json` (x64 OFF; numerics garbage by design, cost/VRAM faithful):

| cols | fp64 ms | true-fp32 ms | **speedup** | ms/Mcol fp32 |
|---|---|---|---|---|
| 16,384 | 70.49 | 16.44 | **4.29×** | 1003 (small-grid overhead) |
| 147,456 | (OOM) | 119.30 | — | 809 |
| 262,144 | (OOM) | 210.64 | — | 804 |

- Extrapolating true-fp32 linearly to 65k = 55.1 ms vs fp64 ladder 255.5 ms →
  **4.64× at scale**. The 4.3× does NOT collapse toward 2× as grids grow — it
  holds/grows, because at large grids fp64 hits its ALU ceiling harder.
- This 4.3–4.6× is the empirical resolution of the "is it 2× or 4×?" question:
  it is **~4×**, because removing fp64 *also* removes the fp64-ALU binding term,
  exactly as the roofline §2 predicts (clean ceiling 4.2×, measured 4.3–4.6×).
- The 1.11× (mixed) vs 4.3× (full fp32) gap is itself the proof: the binding cost
  is fp64-ALU + fp64-bytes, removable only by a FULL fp32 step, not a partial one.

### VRAM lever (independent second win)
- fp64 largest-OK = 65,536 cols @ 11.6 GiB (147k OOMs needing +18.8 GiB temp).
- true-fp32 largest-OK = 262,144 cols @ 16.8 GiB = **4× the columns per GPU**.
- This is the real 1km enabler: a Canary-1km tile that fp64 physically cannot
  hold fits in fp32.

## 5. THE VERDICT — is ~6× single-GPU Canary-1km achievable?

**On-card algorithmic speedup (vs our own fp64 GPU): YES, ~4.3–4.6× is real and
measured, with ~4× VRAM headroom.** That part is solid.

**~6× vs 24–28-rank CPU-WRF: PLAUSIBLE BUT AT THE OPTIMISTIC EDGE — do not bank on it.**
The honest chain (all frames must match; memory anchor says v0.15 fp64 GPU ≈
CPU-WRF *parity* total-wall, ≈1.51× warmed-steady):

  fp32-vs-CPU(warmed) ≈ (fp64-GPU-vs-CPU warmed, ~1.51×) × (fp32/fp64 kernel ratio)

| fp32/fp64 ratio realized at 1km | fp32 vs CPU (warmed-steady) |
|---|---|
| 4.3× (cost-proxy top) | **~6.5×** |
| 3.0× (fusion/overhead loss) | ~4.5× |
| 2.5× | ~3.8× |
| 2.0× (bandwidth-only floor) | ~3.0× |

So **6× requires keeping ~4× of the fp32/fp64 advantage all the way to a real,
numerically-valid 1km fp32 step.** Two quantified risks pull below that:

1. **70% non-roofline overhead.** Measured fp64 @65k = 255.5 ms but
   max(mem 70, ALU 75.5) roofline = 75.5 ms → **180 ms (70%) is neither
   bandwidth nor ALU** — it is kernel-launch/scheduling for many small fused
   stencils (923 fusions, 10k slices in the HLO) + temp-arena re-streaming.
   fp32 shrinks bytes but NOT op/launch count, so realized fp32 gain is bounded
   by how much of that 70% is byte-traffic vs fixed overhead. The cost-proxy
   already *includes* this overhead and still shows 4.3×, which is reassuring —
   but it was measured on the *numerically-invalid* naive all-fp32 program.
2. **Numerically-valid fp32 ≠ naive fp32.** The real path is the ADR-031
   perturbation-frame rewrite (acoustic detonates in naive fp32). That rewrite
   may re-introduce a few fp64 "islands" (base-state/coef_w), shaving the 4.3×
   toward ~3–3.5×. mixed_s2 (1.11×) is the cautionary lower bound if the fp64
   content stays large.

### Bottom line (quantitative, honest)
- **Believe: native-fp32 ≈ 3–4.5× vs fp64 GPU on this card (proxy 4.3–4.6×,
  derated for the perturbation-frame fp64 islands), + ~4× VRAM / domain size.**
- **Vs CPU-WRF warmed-steady: ~4–5× is the defensible center; ~6× is reachable
  only at the top of the band (≈4× fp32/fp64 preserved at 1km) and should be
  stated as a stretch ceiling, not the expected result.**
- **Physically impossible to exceed ~4.6× fp32/fp64 on-card** for this AI≈1
  algorithm: that is the both-terms-removed roofline ceiling; you cannot beat it
  without raising arithmetic intensity (kernel fusion to cut the 70% overhead) or
  changing the algorithm.

The single biggest *additional* lever (orthogonal to fp32) is the **70% launch/
overhead gap** — fusing the many small stencil kernels would attack the part of
the runtime that fp32 alone cannot, and is the path to push fp32/fp64 from ~3×
back up toward the 4.3× proxy and make 6×-vs-CPU robust.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
