# SYNTHESIS — Max GPU utilization for a Tenerife mini-grid (GPT + Opus, 2026-06-30)

Manager 0:1 synthesis of two independent analyses: `ANALYSIS_gpt.md` (GPT-5.5 xhigh) + `ANALYSIS_opus.md` (Opus 4.8). They **converge** on the winner; they **diverge only** on the exact VRAM/max-B (the one thing to measure first).

## Verdict: both models independently pick the SAME winner
**Homogeneous independent-case batching via an OUTER `jax.vmap`** — run B independent Tenerife 2-nest cases (same geometry, different init-days) in lockstep through ONE compiled program. (= the user's idea #1.)

| Approach | GPT rank | Opus rank | Verdict |
|---|---|---|---|
| **Batched `vmap` over independent cases** (idea #1) | **#1** | **#1** | **WINNER** — zero dycore invasion, directly fills the starved SMs |
| Spatial stitching, cut borders (idea #2) | #4, "do not pursue" | rejected | **REJECT** — ~27–32 % area waste @ w=10, and the guard isn't a fixed halo (influence propagates every step) → seam contamination |
| CUDA-streams / MPS multi-process | #2 fallback | fallback | **Fallback only** — no kernel coalescing, duplicate contexts, and the RRTMG transient duplicates per process → OOMs at N≈2–3 |
| Host/output pipeline tightening | #3 (pair) | pair | Complementary, not a main lever (already largely done in v0.22.2) |
| Persistent-kernel / Pallas/Triton rewrite | #6 out-of-scope | out-of-scope | Violates "minimal kernel invasion" |

## Why it wins (both agree)
- **Kernel-invasion: NONE on the dycore.** The fp64 stencils stay byte-identical. The change is an `vmap` shell in the *orchestration* layer + a batched loader + a de-batched output loop.
- **Critical design rule (both stress):** `vmap` AROUND the existing unbatched functions — do NOT teach the dycore that every leaf is rank+1. `build_child_boundary_package` / `interp._gather` assume rank-2/3; an outer `vmap` keeps them seeing normal ranks while XLA adds the batch axis to the callee.
- **Batch WITHIN a domain, NEVER across domains** d01/d02 — crossing domains re-triggers the v0.19 ragged-shape trap.

## v0.19 "vmap = 2.5× slower" dead-end — reconciled (both, independently)
That regression was batching the **7 heterogeneous, different-shaped nested leaves** d03–d09 → padding to a bounding box → occupancy collapse + wasted compute. Homogeneous independent-case batching is the **structural opposite**: identical geometry, **zero padding, zero ragging, no inter-case coupling**. The mechanism does **not** recur. Treated as top candidate, **not proven** → gated by a B=2-vs-2-standalone bit-identity test.

## Complexity: **M (medium sprint)**
Numerics trivial (a `vmap`); the real work is plumbing: batched carries (`tree.map(stack)`), a B-way output writer, fixed-B AOT keys, a special batched **root** `d01→d02` cascade (the generic fused path skips root parents), the bit-identity gate, and one VRAM/compile A/B.

## Throughput / "% of large-grid 5090 best-case"
Both agree: **batching cannot exceed the GPU roofline** (~1.6–2.7× vs CPU for large grids, ~2× center); it makes the mini-grid **behave like a large grid**. Measured as cases/GPU-hour vs running them serially:

| B | GPT estimate | Opus estimate |
|---:|---|---|
| 2 | ~1.5–1.9× | (state-bound) |
| 4 | ~2.8–3.5× | within ~6–12× band |
| 8–16 | ~4–6× (if VRAM ok) | ~6–12× → **~55–75 % of large-grid best-case** |

## The ONE divergence = per-case VRAM → max B (measure first)
- **Opus optimistic:** ~1 GB/case (state-bound; the dominant ~8–9 GiB RRTMG radiation transient is column-tile-capped at 1024 and **does not multiply** by B) → **B≈8–16 fits 32 GB**.
- **GPT conservative:** ~3–9 GiB/case dynamic peak → **B≈2–6**.
- Resolved by the first GPU gate (peak VRAM at B=1/2/4). **Plan conservatively for B=4 (~3× cases/hour, both agree); upside B=8+ if the transient doesn't multiply under `vmap`.**

## Honest ceilings / failure modes
1. **Bit-identity** — XLA may change per-element fusion under batching → MANDATORY B=2-vs-2-standalone byte-compare before any training data is generated. (#1 correctness risk: a latent reduction/broadcast over the batch axis, or `*_save` WRF scratch aliasing.)
2. **VRAM caps B** below the useful regime (the divergence above).
3. **Per-case output D2H** can become the new host bottleneck for short mini cases (async writer helps).
4. **Compile-RAM/time** of the batched HLO (fix B to 1–2 production values).

## Recommendation (manager)
**Approve a MEDIUM implementation sprint** for batched-`vmap` 2-nest homogeneous-case orchestration. It's the right lever: **zero dycore risk, directly attacks the mini-grid underutilization, perfectly matched to "generate many independent Tenerife training-days fast."** Gate ladder (needs GPU when 0:2 frees it): **B=2 bit-identity → B=1/2/4 VRAM A/B → cases/hour benchmark → pick fixed production B.** Keep MPS/streams as the conservative fallback only if the bit-identity gate fails.
