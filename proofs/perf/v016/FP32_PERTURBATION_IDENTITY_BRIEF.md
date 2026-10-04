# FP32 perturbation identity brief

Date: 2026-06-14
Author: GPT-5.5 audit
Scope: README + v0.15/v0.16 performance proofs + current kernel code audit

## One-sentence conclusion

The likely path to using the RTX 5090's FP32 capacity is not a compiler flag or
register trick. It is a source-level, discrete WRF-operator rewrite that makes
the perturbation state authoritative and computes every pressure/geopotential
diagnostic as a base-plus-perturbation identity, never as `large total - large
base` or `large total - small perturbation` inside the timestep loop.

## Simple explanation

The GPU is not weak. The current calculation often asks FP32 to represent a
large atmospheric number and then recover a small physically important
difference from it. At pressure around 100000 Pa, FP32 spacing is already about
0.0078 Pa. At geopotential magnitudes of hundreds of thousands of m2/s2, FP32
spacing is larger again. If the solver stores the big total and later subtracts
a nearly equal big base, the small pressure-gradient signal can be rounded away.
The unused exponent range in FP32 cannot be reassigned to mantissa precision by
the compiler; the practical way to use those bits is to store offset/scaled
perturbation variables whose exponent is near the perturbation scale.

The way around this is to stop storing and subtracting the big totals in the hot
loop. Store:

- the static base state in FP64: `PB`, `PHB`, `MUB`, base theta / base alpha,
- the evolving perturbations in FP32 where validated: `p'`, `ph'`, `mu'`,
  `theta'`, winds and tracers,
- only reconstruct full totals at controlled boundaries: output, restart,
  diagnostics, boundary exchange, or physics adapters that truly require totals.

This is a mathematical identity, not a physics change. It is the same equation
written in coordinates that match the hardware's precision.

## Evidence from current proofs

- README v0.15 says no multi-x speedup is claimed. The measured final 72 h
  gates are roughly parity total-wall: Switzerland 0.99x, Canary 1.04x, with
  forecast-only about 1.05x to 1.20x.
- `proofs/perf/v015/kernel_characterization.md` refutes the earlier 6-10x
  large-grid claim for the current fp64 coupled step. The honest large-grid
  asymptote is about 1.6x to 2x unless structural FP32/Pallas work lands.
- `proofs/perf/v015/viability/true_fp32_cost_proxy.json` shows the hardware
  upside: a deliberately numerically invalid all-FP32 proxy is 4.29x faster on
  the 128x128 cost harness.
- `proofs/perf/v015/viability/fp32_fp64_ladder.json` shows why naive mixed
  precision did not work: the runnable mixed path gave only 1.01x to 1.02x,
  because the acoustic path remained effectively FP64-pinned.
- `proofs/perf/v016/fp32_s2_mixed_ladder_final_2x2.json` improves this to
  254.3 -> 229.9 ms/step on the 256x256 tiled case, about 1.11x, but VRAM peak
  remains identical at 11.65 GiB. The mixed state still carries FP64 totals:
  `p_total`, `ph_total`, `mu_total`.
- `proofs/perf/v016/s2_hlo_stats_2x2_s1_final.json` confirms the partial nature
  of S2: mixed mode cuts HLO f64 tokens from 76067 to 38245 and f32 tokens rise
  to 38403, but temporary memory barely moves, 7.66 -> 7.57 GiB, and f64
  conversions rise sharply. The graph is still not a clean FP32 perturbation
  graph.
- `proofs/perf/v016/s2_tiered_identity_s20_final.json` shows short-run mixed
  S2 is numerically plausible: PASS, worst rmse/manifest limit 1.14e-4.

## Code audit findings

The code already contains the right architectural seed:

- `src/gpuwrf/runtime/operational_state.py` adds `OperationalCarry.base`, a
  carried `BaseState` intended to stop reconstructing large base fields as
  `total - perturbation`.
- `src/gpuwrf/dynamics/core/small_step_prep.py` requires `BaseState` for
  `MIXED_PERTURB_FP32` and builds many work arrays in FP32.
- `src/gpuwrf/dynamics/core/small_step_finish.py` reconstructs totals in mixed
  mode as `base + perturbation`, but still writes `p_total`, `ph_total`,
  `mu_total` back into the hot `State`.
- `src/gpuwrf/contracts/state.py` still makes `p_total`, `ph_total`, and
  `mu_total` first-class State leaves and has helper functions that recover
  base fields as `total - perturbation`.
- `src/gpuwrf/runtime/operational_mode.py` still has compatibility paths that
  compute `state.p_total - state.p_perturbation`, `state.ph_total -
  state.ph_perturbation`, and `template.*_total - template.*_perturbation`.

So the statement "FP32 loses the small gradient because we start from large
absolute numbers" is true for the remaining total-minus-perturbation and
total-minus-base paths. It is not a law of physics. It is a representation
choice that the code has only partially removed.

## The identities to use

### 1. Affine base-plus-perturbation state

For any WRF field with a large static base:

```text
X = Xb + x
```

where `Xb` is static or slow base, and `x` is the evolving perturbation. Then:

```text
grad(X) = grad(Xb) + grad(x)
```

If the discrete WRF operator contains a base hydrostatic cancellation, combine
the base terms analytically before evaluation and evolve only the residual.

### 2. Difference identity

Do not compute:

```text
(Xb_i + x_i) - (Xb_j + x_j)
```

as two FP32 totals. Compute:

```text
(Xb_i - Xb_j) + (x_i - x_j)
```

with `Xb_i - Xb_j` precomputed or kept in FP64, and `x_i - x_j` in FP32 if the
validation gate allows it.

### 3. Ratio identity

For denominators such as dry mass:

```text
1 / (Mb + mu) = (1 / Mb) / (1 + mu / Mb)
```

This keeps the small variable as a nondimensional ratio. Use `mu/Mb` and
`log1p`/`expm1` style forms where the expression is nonlinear.

### 4. Linear hypsometric alpha perturbation

For a linear form:

```text
alpha = D / M
alpha_b = Db / Mb
D = Db + dD
M = Mb + dM
```

The perturbation is exactly:

```text
alpha' = alpha - alpha_b
       = (dD - alpha_b * dM) / (Mb + dM)
```

This avoids computing two nearly equal `alpha` values and subtracting them.

### 5. Log-hypsometric alpha perturbation

For WRF `hypsometric_opt=2`, write:

```text
L(M) = phm(M) * log(pfd(M) / pfu(M))
alpha(M, D) = D / L(M)
alpha_b = Db / L(Mb)
```

Then:

```text
alpha' = alpha(Mb + dM, Db + dD) - alpha_b
       = (dD - alpha_b * (L(Mb + dM) - L(Mb))) / L(Mb + dM)
```

Compute `L(Mb + dM) - L(Mb)` with `log1p` on face-pressure perturbation ratios.
This preserves the WRF discrete operator while avoiding large nearly equal
subtractions.

### 6. EOS pressure perturbation with `expm1`

For the WRF EOS shape:

```text
p_total = P0 * (R * theta / (P0 * alpha)) ** gamma
PB      = authoritative WRF base pressure
```

The perturbation can be computed as:

```text
s = log(p_total / PB)
p' = PB * expm1(s)
```

This is the same real-valued equation as `p_total - p_b`, but it does not form
and subtract two O(100000 Pa) numbers. This identity is probably the most
important one to implement for FP32 acoustic viability.

When the loaded WRF base also satisfies the same EOS to proof tolerance, compute
`s` without large logs as:

```text
s = gamma * (log1p(theta'/theta_b) - log1p(alpha'/alpha_b))
```

When `PB` is the only authoritative base, precompute the static base log term in
FP64 and evaluate:

```text
s = gamma * log(R * theta / (P0 * alpha)) - log(PB / P0)
```

using `log1p` around the perturbation ratios wherever possible. The acceptance
gate must decide which base relation is WRF-identical for the fixture.

## Required implementation level

This is primarily a JAX/source-level discrete algebra rewrite.

Do this first:

- Make `BaseState` mandatory for the operational hot loop, not optional.
- Remove `p_total`, `ph_total`, `mu_total` from the high-frequency scan carry in
  mixed mode, or carry them only in a cold/diagnostic side channel.
- Rewrite `diagnose_pressure_al_alt`, `small_step_prep_wrf`,
  `small_step_finish_wrf`, `rk_addtend_dry`, boundary apply/feedback, and writer
  interfaces around explicit `(base, perturbation)` inputs.
- Replace EOS total-minus-base with the `expm1(log1p(...))` pressure perturbation
  identity.
- Replace hypsometric `al = total_alpha - base_alpha` with the perturbation
  identities above.
- Add a static audit that fails any in-loop `*_total - *_perturbation` in mixed
  mode.
- Gate with WRF savepoints, deterministic short-run identity, 72 h bounded
  divergence, and Nsight/HLO temp-memory proof.

Do not start here:

- GPU assembly or register-level tricks. They cannot recover precision after
  source code has already asked for cancellation-prone FP32 totals.
- A compiler pass. XLA cannot infer WRF hydrostatic/base-state identities safely
  from arbitrary JAX code, especially with legacy aliases and `State.replace`
  synchronization.
- A global FP32 dtype flip. v015 already proved a true all-FP32 proxy is fast but
  numerically invalid.

Possible later levels:

- A JAX custom primitive or small affine-state helper can enforce the
  `(base, perturbation)` representation and prevent accidental total recovery.
- Pallas/Triton/CUDA kernels may be useful after the algebra is correct, mainly
  to fuse the perturbation operators and keep base coefficients resident. They
  are performance tools, not the numerical fix.
- A custom number format is a last resort. What we need is not more exponent
  range; it is more relative precision for small residuals. Offset/scaled
  perturbation variables already provide that with ordinary FP32.

## Sprint-sized acceptance gates

1. Static mixed-mode audit: zero hot-loop occurrences of
   `*_total - *_perturbation`, except explicitly whitelisted output/restart code.
2. Micro oracle: EOS `p'` via `expm1/log1p` matches fp64 total-minus-base to a
   tight tolerance over real WRF ranges, but remains stable when FP32 totals
   would lose low bits.
3. Savepoint oracle: `calc_p_rho_phi` / `advance_w` WRF fixture parity with the
   perturbation formulas.
4. Short deterministic forecast: reproduce the v016 20-step TieredIdentity PASS
   or better.
5. Performance proof: HLO f64 tokens, convert-to-f64 count, temp memory, and
   Nsight ms/step all improve together. A valid win should reduce temp memory,
   not just flip some leaf dtypes.
6. Long-run proof: repeat the v016 bounded-divergence classification over the
   two v0.15 carry fields, then run the normal 72 h field gate.

## Bottom line for external reviewers

There is a real mathematical route. The current mixed FP32 lane is partial and
therefore only yields about 1.1x. The next attempt must be an operator-level
perturbation formulation, especially EOS and hypsometric alpha, not a broader
dtype policy change. If that rewrite still leaves f64 temp memory dominant, then
the next bottleneck is implementation structure/fusion, not the mathematical
representation.
