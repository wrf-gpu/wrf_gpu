# Critical Review: `FP32_GRADIENT_PROBLEM_OPEN_CHALLENGE.md`

Date: 2026-06-14

Reviewed input: `<USER_HOME>/src/wrf_gpu2/FP32_GRADIENT_PROBLEM_OPEN_CHALLENGE.md`

Independent proof produced:

- `proofs/perf/v016/fp32_challenge_independent_audit.py`
- `proofs/perf/v016/fp32_challenge_independent_audit.json`

## Verdict

The challenge document is useful, but its final strategic conclusion is too
pessimistic about speed and too optimistic about double-single arithmetic.

What I can verify:

- The numerical cancellation problem is real.
- Perturbation storage fixes most of the toy pressure-gradient cancellation.
- Double-single can recover near-fp64 accuracy in the toy total-field gradient.
- Kahan/compensated accumulation strongly reduces synthetic long-horizon drift.
- The current runnable mixed-fp32/S2 lane is a performance dead end by itself:
  about 1.1x and no VRAM relief.

What I do **not** accept:

- "`FP64:FP32 on the real step is ~1x, therefore precision cannot buy wall-clock."
  That only describes the current partial mixed graph, where the hot path remains
  fp64-dominated. It does not falsify the true-fp32 cost proxy.
- "Double-single is the solution." Broad double-single is a bad default on this
  RTX 5090: it costs about 25-32x fp32 and about 15-16x native fp64 in my probe,
  while storing `(hi, lo)` costs 8 bytes again. Used broadly, it kills both speed
  and VRAM.

Recommended path:

1. Pursue a **source-level perturbation-native acoustic/operator rewrite**.
2. Add **targeted compensated accumulation** only for true long-lived drift paths.
3. Avoid broad double-single; keep it as an oracle or last-resort local island.
4. Only after the HLO/profiler shows the hot path is genuinely fp32, spend effort
   on Pallas/Triton fused acoustic/vertical kernels.

Realistic warm full-step result on the local GPU:

- Current S2 dtype path: **1.1x**, no VRAM win.
- Correct perturbation-native fp32 hot path: **1.8x conservative**, **2.1-2.9x
  realistic**, **~4x stretch** only if non-core overhead also shrinks heavily or
  fused/custom kernels land.

## What I Reproduced Independently

My independent audit uses different seeds and a separate script. Key results:

### Accuracy

Pressure-gradient toy, total pressure around 100000 Pa:

| method | RMSE vs fp64 |
|---|---:|
| naive total stored fp32 | 3.190e-3 |
| perturbation fp32 | 1.585e-7 |
| double-single total | 1.144e-10 |

This confirms the core numerical story: storing totals in fp32 is bad;
perturbation storage is already a huge improvement; double-single can recover
more bits in a toy total-difference.

Accumulation toy, 100000 steps:

| method | abs error |
|---|---:|
| naive fp32 | 4.858e-1 |
| Kahan fp32 | 2.911e-3 |

This confirms Kahan/compensated accumulation is a real tool for drift. It does
not prove it is sufficient for full WRF conservation; it only proves the toy
mechanism is valid.

### Throughput

My independent FMA-chain probe shows the fp64/fp32 ratio is not a stable hardware
constant in JAX/XLA generated code:

| shape | fp64/fp32 |
|---|---:|
| N=4M, K=512 | 1.79x |
| N=4M, K=2048 | 1.77x |
| N=8M, K=512 | 2.08x |
| N=8M, K=2048 | 2.16x |
| N=16M, K=512 | 7.26x |
| N=16M, K=2048 | 7.20x |

So the challenge's measured ~4.3x is plausible for one kernel shape, but not a
general law. The only safe conclusion is weaker: native fp64 is not 64x slower
inside these XLA kernels, but it can still be materially slower.

Double-single cost in my probe:

| K | DD/fp32 | DD/fp64 |
|---|---:|---:|
| 256 | 24.9x | 15.0x |
| 1024 | 28.8x | 16.2x |

This falsifies broad DD as a performance path on the local GPU.

## Recomputed Artifact Facts

From v016 S2 artifacts:

| metric | value |
|---|---:|
| fp64 warm step | 254.305 ms |
| mixed S2 warm step | 229.948 ms |
| speedup | 1.106x |
| VRAM ratio | 1.0x |
| kernel-time speedup from Nsight CSV | 1.106x |
| temp live-set ratio | 1.012x |
| XLA bytes-accessed ratio | 1.158x |
| XLA flops ratio | 1.022x |
| mixed convert-op increase | 5.28x |

This verifies that S2 did not change the real hot graph enough. It changed field
dtypes, but not the dominant temp arena, top kernels, or launch topology.

From v015 true-fp32 cost proxy:

| metric | value |
|---|---:|
| fp64 core at 16k columns | 70.492 ms |
| true-fp32 proxy core at 16k columns | 16.436 ms |
| core speedup | 4.289x |
| true-fp32 large-grid slope | 0.790 ms / 1000 columns |

This is numerically invalid as a forecast, but it is still the best available
cost upper-bound for "what if the hot path really became fp32". The challenge
document underweights this evidence.

## Amdahl Bound For Wall Clock

Using the measured v015 full step:

```text
current full step        = 119.8 ms
fp64 proxy core          = 70.49 ms
non-core remainder       = 49.31 ms
true-fp32 proxy core     = 16.44 ms
```

If only the core reaches the true-fp32 proxy:

```text
new step = 16.44 + 49.31 = 65.74 ms
speedup  = 1.82x
```

If the non-core remainder also improves:

| remainder speedup | projected full step | projected speedup |
|---:|---:|---:|
| 1.00x | 65.74 ms | 1.82x |
| 1.25x | 55.88 ms | 2.14x |
| 1.50x | 49.31 ms | 2.43x |
| 2.00x | 41.09 ms | 2.92x |
| 3.00x | 32.87 ms | 3.64x |
| 4.00x | 28.76 ms | 4.17x |

This is why the honest target band is 2-3x, not "only 10%" and not "guaranteed
4x".

## Critical Claim Review

| Challenge claim | Review |
|---|---|
| Naive fp32 totals lose pressure-gradient bits. | Confirmed. |
| Perturbation storage fixes much of the gradient problem. | Confirmed in toy and consistent with S2 correctness. |
| Double-single recovers near-fp64 toy-gradient accuracy. | Confirmed. |
| Kahan greatly reduces toy accumulation drift. | Confirmed. |
| Real step is fp64/fp32 ~1x, so fp32 is mostly VRAM not speed. | Misleading. Current runnable mixed path is ~1x because the hot path is not truly fp32. True-fp32 proxy shows a 4.29x core cost change. |
| Double-single should be part of the solution. | Only as a tiny targeted island. Broad DD is worse than fp64 on this GPU and uses 8 bytes per value. |
| Intervention layer is JAX algorithm + XLA barriers/custom lowering, not SASS/silicon. | Mostly correct. Source-level algebra is required first; Pallas/Triton may be needed for speed, but not for the numerical identity. |
| Stochastic rounding is a candidate. | Unproven and likely a poor fit for WRF identity/reproducibility unless a deterministic, validated path exists. |

## Path Estimates

### Path A: Continue Current S2/Dtype Edits

Expected result: **1.05-1.2x**, no meaningful VRAM reduction.

Reason: already measured 1.106x, temp live set changed only 1.15%, and convert
ops increased 5.28x. More scattered dtype edits are likely to keep hitting the
same liveness/convert wall.

Recommendation: **stop as a speed path**.

### Path B: Perturbation-Native Rewrite, No Broad DD

Expected result: **1.8x conservative**, **2.1-2.9x realistic**.

Reason: the measured true-fp32 proxy gives 4.29x on the core, but Amdahl limits
the full production step unless non-core overhead also shrinks. This path also
has the best chance of halving the relevant live set because it removes full-grid
`p_total/ph_total/mu_total` from the timestep loop instead of carrying them as
fp64 aliases.

Recommendation: **main path**.

Required algebra:

```text
X = X_base + x
grad(X) = precomputed_grad(X_base) + grad(x)
1 / (M_base + mu) = (1 / M_base) / (1 + mu / M_base)
alpha' = (dD - alpha_base * dM) / (M_base + dM)
p' = PB * expm1(s)
```

### Path C: Path B + Targeted Kahan/Compensated Accumulation

Expected result: **same speed band minus small overhead**, likely still
**1.7-2.7x** if compensation is targeted.

Reason: Kahan adds arithmetic, but only matters if applied to long-lived sums or
state updates that actually drift. My toy test reduced error 167x. It is
valuable as a correctness stabilizer, not as a speed lever.

Recommendation: **use selectively** for mass/pressure/geopotential accumulated
updates and conservation reductions after an oracle shows naive perturbation-fp32
drifts.

Do not add a compensation field for every prognostic variable by default; that
turns 4-byte storage back toward 8-byte storage.

### Path D: Broad Double-Single State Or Hot Loops

Expected result: **slowdown**, not speedup. Likely catastrophic if used broadly.

Reason: DD is 25-32x fp32 and 15-16x native fp64 in my probe. `(hi, lo)` also
uses 8 bytes, so broad DD throws away the VRAM goal. A simple Amdahl check:
if DD replaces only 5% of current fp64 work and costs 15x that work, whole-step
time multiplies by roughly `(0.95 + 0.05*15) = 1.70`.

Recommendation: **do not use broadly**. Keep DD for:

- independent oracle checks;
- one or two tiny cancellation islands only if they remove a larger fp64 island;
- possible custom-kernel experiments after profiling proves the island is small.

### Path E: Compiler Rematerialization/Liveness Only

Expected result: **VRAM may improve, speed neutral or worse**, no complete
numerical fix.

Reason: rematerialization can reduce live temporaries, but if the algebra still
forms fp64 totals and subtracts them, it only schedules the same bad graph.
Recomputing can also increase wall time.

Recommendation: use only after Path B defines the perturbation-native graph.

### Path F: Pallas/Triton Megakernels On Current fp64 Graph

Expected result: **0-15% for launch-only cleanup**, possibly **1.3-2x** if it
also fuses memory traffic and the vertical/acoustic solve.

Reason: whole-step CUDA graph/capture style launch reduction was already
wall-neutral, so launch count alone is not enough. But the current graph has
about 12k kernels/step and large repeated full-grid traffic, so a real fused
acoustic/vertical kernel could still matter.

Recommendation: not first. Build it after Path B, so custom kernels implement the
right perturbation algebra rather than optimized fp64-total algebra.

### Path G: Stochastic Rounding

Expected result: **unproven correctness, likely negative performance** in
software.

Reason: no local proof, nondeterminism conflicts with the project's identity
discipline, and software stochastic rounding adds RNG and extra ops.

Recommendation: defer unless all deterministic compensation paths fail.

## Next Sprint Gate

Do not start with a 72 h forecast. First prove the graph crossed the structural
threshold on the 256x256x44 S2 ladder:

- correctness: savepoint/oracle for rewritten `calc_p_rho`, pressure-gradient,
  hypsometric alpha, and `advance_w`;
- static audit: no fp32 `total - perturbation` in hot loops;
- state contract: `p_total`, `ph_total`, `mu_total` not live timestep-loop state;
- HLO: temp live set materially below 7.57 GiB; convert ops far below 697;
- profiler: top kernels must change time/dtype; not the same top kernels at the
  same wall time;
- performance stop/go: `<130 ms/step` at 256x256 means credible 2x candidate;
  `<90 ms/step` means credible 3x candidate.

## Final Recommendation

Use the challenge document as a source of numerical tools, not as the strategic
answer.

The best path is:

1. Implement perturbation-native algebra in JAX source.
2. Carry fp64 static/base arrays and coefficients only at explicit boundaries.
3. Add targeted Kahan where drift is proven.
4. Avoid broad double-single.
5. Once HLO/profiler proves a real fp32 hot path, fuse the acoustic/vertical
   solve with Pallas/Triton if the graph is still fragmented.

My expected local-GPU outcome if this is done well: **2x is credible, 3x is a
good outcome, 4x is a stretch target requiring fused kernels or broad overhead
reduction.**
