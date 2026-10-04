# FP32 Perturbation Identity: Would It Actually Move Wall Clock?

Date: 2026-06-14

## Verdict

The perturbation-authoritative fp32 rewrite is **not proven as a 2-4x wall-clock
win yet**, but it is also **not the same dead end as the current runnable fp32
lane**.

Current evidence says:

- The fp32 lane that merely stores some fields in fp32 is a dead end for
  performance: v015 measured 1.01-1.02x, and v016 S2 measured 1.106x.
- A true all-fp32 cost proxy, where the acoustic/pressure hot path actually runs
  fp32, measured 4.29x on the core step and about 0.79 ms / 1000 columns at large
  grid. That is the hardware upside, but its numerics are intentionally invalid.
- The perturbation identity is the mathematically plausible route to make that
  cost proxy valid. It is necessary. It is not sufficient unless the generated
  HLO/Profiler artifacts show the real hot kernels and temporaries have moved
  out of fp64.

Expected wall-clock band if the rewrite is completed correctly:

- **Conservative full-production warm step:** about **1.8x** over current fp64
  GPU if only the measured core becomes true-fp32 and all remaining overheads
  stay unchanged.
- **Reasonable target:** about **2.1-2.9x** if some physics/coupler overhead and
  convert traffic also shrink.
- **4x full-production wall:** only credible if most of the non-core overhead
  also becomes fp32/fused, or if Pallas/Triton kernel consolidation lands too.

So the right promise is not "identity alone guarantees 4x". The right promise is:
**identity unlocks the only measured path from 10% to multi-x; without it, the
fp32 lane stays capped near 10%.**

## Why Current S2 Is Only 10%

v016 S2 on the 256x256x44 tiled Switzerland operational step:

| metric | fp64 | mixed S2 | ratio / change |
|---|---:|---:|---:|
| warm step | 254.305 ms | 229.948 ms | 1.106x |
| peak VRAM | 11.646 GiB | 11.646 GiB | no gain |
| temp size | 7.658 GiB | 7.570 GiB | -1.15% |
| XLA bytes accessed | 117.35 GiB | 101.34 GiB | -13.6% |
| XLA flops | 128.43 G | 125.67 G | -2.15% |
| HLO f64 tokens | 76067 | 38245 | -49.7% |
| HLO convert ops | 132 | 697 | +428% |
| Nsight kernel time | 2.999 s / 12 steps | 2.710 s / 12 steps | 1.106x |

Interpretation: S2 changes many leaves to fp32, but the real temporary arena and
the top kernels remain largely the same. This is not yet a perturbation-native
operator graph; it is a mixed graph with many fp64 islands and conversion tax.

The profiler confirms that the largest kernels barely move:

- `loop_multiply_fusion_1`: 592.8 ms fp64 vs 590.5 ms mixed S2.
- `input_reduce_fusion_19`: 383.5 ms fp64 vs 382.9 ms mixed S2.
- the largest 12-instance fused block: 155.0 ms fp64 vs 154.8 ms mixed S2.

That is exactly the signature of an incomplete precision rewrite.

## Why The Identity Could Be Different

The v015 true-fp32 cost proxy is numerically invalid but performance-informative:

| grid | fp64 ref | true-fp32 proxy | speedup |
|---|---:|---:|---:|
| 128x128, 16k columns | 70.49 ms | 16.44 ms | 4.29x |
| 384x384, 147k columns | fp64 OOM in harness | 119.30 ms | fits |
| 512x512, 262k columns | not run fp64 | 210.64 ms | fits |

Linear fit through the true-fp32 proxy:

```text
ms_per_step ~= 3.27 + 0.790 ms_per_1000_columns
```

The current fp64 large-grid core fit is about:

```text
ms_per_step ~= 19.58 + 4.937 ms_per_1000_columns
```

This means the cost proxy shows about **4-6x less marginal per-column cost** when
the actual hot path is fp32. That is too large to dismiss as a cosmetic dtype
change. It is the performance target the perturbation identity must reproduce
with valid numerics.

## Amdahl Check Against Full Production

The relevant caution is that the 4.29x number is a core-step proxy, not the full
production wall. Using the v015 128x128 numbers:

```text
current full step ~= 119.8 ms
measured fp64 proxy core ~= 70.49 ms
non-core remainder ~= 49.31 ms
true-fp32 proxy core ~= 16.44 ms
```

If only the core gets the 4.29x win:

```text
new full step ~= 16.44 + 49.31 = 65.75 ms
speedup ~= 119.8 / 65.75 = 1.82x
```

If the non-core remainder also improves:

| non-core remainder speedup | projected full step | projected full-step speedup |
|---:|---:|---:|
| 1.00x | 65.74 ms | 1.82x |
| 1.25x | 55.88 ms | 2.14x |
| 1.50x | 49.31 ms | 2.43x |
| 2.00x | 41.09 ms | 2.92x |
| 3.00x | 32.87 ms | 3.64x |
| 4.00x | 28.76 ms | 4.17x |

This is the key risk assessment:

- **2x is credible** if the perturbation rewrite reaches the true-fp32 core
  target.
- **3x is credible** if it also removes a meaningful fraction of conversion,
  reconstruction, and physics/coupler fp64 overhead.
- **4x is not credible from algebra alone**; it needs broad fp32 propagation and
  likely kernel consolidation.

## What Would Make It A Dead End?

The rewrite is likely a performance dead end if after implementation any of
these are still true:

- warm-step speedup remains below 1.3x on the 256x256 S2 ladder;
- temp size remains near 7.6 GiB instead of dropping materially;
- top Nsight kernels remain time-identical to fp64;
- HLO `convert` count stays several hundred;
- `p_total`, `ph_total`, and `mu_total` are still live timestep-loop state
  rather than boundary/output reconstructions;
- `base = total - perturbation` still occurs inside hot acoustic or physics
  loops;
- the pressure/geopotential diagnostic still computes a total-field expression
  and subtracts a base expression instead of evaluating the perturbation form
  directly.

Those conditions describe the current S2 partial state and explain the 10% result.

## Required Gate For The Next Sprint

A credible speedup sprint should not start with a 72 h forecast. It should first
prove that the HLO and profiler have crossed the structural threshold.

Minimum gate on the 256x256x44 S2 ladder:

- correctness: savepoint/oracle gate for the rewritten pressure/geopotential
  perturbation operators before any speed claim;
- warm step: target below **130 ms/step** for "real 2x candidate", below
  **85 ms/step** for "real 3x candidate";
- temp size: materially below current 7.57 GiB, ideally near half for
  perturbation-heavy state;
- HLO: f64 tokens confined to static base/metrics/islands, convert ops sharply
  below current 697;
- Nsight: top three kernels must change dtype/cost, not remain numerically
  identical to fp64 timings;
- static audit: no fp32 `total - perturbation` in timestep-loop operators.

Only after that should the project spend expensive 24-72 h wall-clock gates.

## Implementation Level

This should be attacked at the **JAX source / discrete-operator level first**.

Compiler flags, register-level tricks, or assembler cannot recover mantissa bits
lost by feeding the compiler cancellation-prone fp32 total-field algebra. The
source graph must expose perturbation-scale variables and identities:

```text
X = X_base + x
grad(X) = grad(X_base) + grad(x)
1 / (M_base + mu) = (1 / M_base) / (1 + mu / M_base)
alpha' = (dD - alpha_base * dM) / (M_base + dM)
p' = PB * expm1(s)
```

Pallas/Triton can matter after that, because the current graph has many small
kernels and while/scan/fusion overhead. But Pallas before the perturbation
rewrite risks hand-optimizing the wrong fp64 algebra.

## Bottom Line

The perturbation identity is not decorative math. It targets the exact reason the
runnable fp32 lane failed: the hot path never became true fp32. The measured cost
proxy says a multi-x core win exists on this GPU. The current S2 proof says the
partial implementation has not reached it.

Decision-quality expectation:

- **Do pursue** a perturbation-authoritative mixed-fp32 rewrite.
- **Do not claim** 2-4x until the HLO/profiler threshold above passes.
- **Use 2x warm full-step as the first serious success bar.**
- Treat **3x** as the good outcome.
- Treat **4x** as stretch, requiring broad fp32 propagation and probably Pallas
  or equivalent kernel consolidation.
