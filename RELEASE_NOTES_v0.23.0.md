# Release Notes — v0.23.0

A performance + capability release on top of the v0.22.2 nested-grid wall-clock line.
The default (no-env) forecast is **numerically equivalent to v0.22.2** — see "Default
numerics" below for the precise, honest statement.

## Headline — batched-ensemble GPU saturation on small grids (F1)

A single small operational nest under-uses the GPU: it is launch/occupancy-bound, so the
card sits mostly idle. v0.23 adds an **opt-in batched ensemble** path
(`GPUWRF_BATCH_ENSEMBLE=B`): **B independent, same-geometry forecasts run concurrently
under one `jax.vmap`** wrapped *around* the unbatched orchestration. The dynamical core is
untouched, and the default path (`GPUWRF_BATCH_ENSEMBLE` unset / `=1`) is byte-identical to
the unbatched run.

**Measured — Tenerife 3 km→1 km 2-nest, one RTX 5090, warm (total cells-in-flight/s):**

| B (concurrent cases) | total throughput | GPU util | % of large-grid ceiling | peak VRAM |
|---:|---:|---:|---:|---:|
| 1 | 4.51 M cells/s | 28.5 % | 78 % | 9.4 GiB |
| 2 | 4.62 M cells/s | 37 %   | 79 % | 10.8 GiB |
| 3 | 5.24 M cells/s | 45 %   | 90 % | 15.4 GiB |
| **4** | **5.63 M cells/s** | 53 % | **97 %** | 18.3 GiB |
| 5 | 5.62 M cells/s (plateau) | 56 % | 97 % | ~26 GiB (cap) |

The large-grid **saturation ceiling = 5.81 M cells/s**, measured on a compute-bound
**Alps 433²@1 km** mountain case on the same card. Batching the tiny Tenerife grid climbs
from 4.51 M to 5.63 M cells/s and reaches **≈97 % of that ceiling** — it **fills the
otherwise-idle GPU on the small operational grid**, recovering the launch-bound penalty
with no dycore change or precision compromise. **B_max = 5 is VRAM-capped at ~26 GiB on the
5090; a larger card (e.g. B200) batches further.**

Use case: many-case throughput — ensembles, parameter sweeps, parallel-in-time multi-day
chunks — on small operational domains where one case leaves the GPU underused.

## Performance bundle (default path)

- **P0 — M9 radiation flux-slice reduction (default-on).** The training-subset radiation
  re-solve now computes only the surface/TOA flux slices the wrfout writer consumes,
  dropping the heating-rate / clear-sky work it never outputs. **Value-preserving**
  (CPU bit-identical, 17/17 fields; forecast radiation untouched). It changes the compiled
  program (fewer ops), so on GPU the output matches v0.22.2 within the autotune floor
  (see "Default numerics").
- **P3 — flat 2-domain root fusion (default-on, fail-closed).** A flat d01→d02 (1-way)
  parent+child cascade compiles as one entry program instead of three (174→169 launch
  markers). Fail-closed for every other topology (deeper roots, leaves, feedback, bad
  ratios). Bit-identical (CPU sha256 fused==eager; the 3-domain canary is unaffected —
  the gate fires only for flat 2-domain).
- **P1/P2/P4/P6/P7b — launch/compile-count reductions.** Single-scan knob (default-off),
  gated safe-floors + a single finite-guard reduction, a vectorized `calc_coef_w` inner
  loop, dead-code/idiom cleanup, and an opt-in low-effort cold-compile knob. Default-inert.

*Canary benchmark (mandatory gate):* 3-domain `canary_all7`, 1 forecast-hour, warm —
v0.23 **407 s** / 11619 MiB vs v0.22.2 **411 s** / 11566 MiB (≈1 % faster wall, VRAM
+0.5 %; no regression).

## Extended physics + nesting — reference-only / opt-in (default untouched)

Every scheme below is **namelist-gated**: it is either opt-in or **reference-only and
fail-closed** (selecting it operationally raises a named error — it can never silently run
a broken or no-op kernel). The default scheme selection is unchanged.

- **F2** — New-Tiedtke cumulus (`cu_physics=16`) and Morrison-aerosol microphysics
  (`mp_physics=40`) ported and validated to machine precision against pristine-WRF
  single-column oracles; RUC-LSM integrated + validated; **NSSL 2-moment (`mp_physics=18`)
  is reference-only with a verified oracle** (faithful port = its own milestone).
- **F3** — CAM-UW moist PBL (`bl_pbl_physics=9`) is **reference-only/fail-closed**: a
  standalone WRF-Fortran CAM-UW column oracle proved the prior JAX scaffold RED, so the
  scaffold is walled off and the faithful port is a separate milestone. Default PBL (MYNN)
  is untouched.
- **G2** — real operational moving-nest driver (per-move force/feedback weight rebuild,
  vortex tracker, adaptive-Δt), **opt-in**; the static-nest path is byte-identical.
- **G3** — urban BEP/BEM (`sf_urban_physics=2/3`) and the WRF lake model (`sf_lake_physics=1`)
  are **reference-only/fail-closed** (source/Registry inventory staged; numerical oracle +
  faithful port = their own milestone). Default `sf_urban_physics/sf_lake_physics=0`.

## fp32-operational foundation (M1) — separate milestone

The historical fp32 "compile pathology" that appeared to block operational fp32 (and the
BouLac O(nz) reformulation) for three versions was shown on the current stack to be a
**measurement artifact** (a warm-cache baseline compared against a cold variant, plus
timeout-budget and cold-compile-alarm misreads), **not a real XLA fusion pathology** — with
an independent adversarial cross-exam returning zero refutations and a re-runnable
compile-budget regression guard. A 24 h real-case Swiss proof shows mixed-perturbation fp32
tracks byte-identical-IC CPU-WRF truth to ≤ 1.003× across all fields with the masking guard
disabled, and −14…−19 % peak VRAM. **fp32 is NOT enabled in the v0.23 default path** — the
full operational rollout is its own milestone (ADR-031). The fp64 default is byte-identical.

## Default numerics — the honest statement

`STABILITY > IDENTITY > SPEED > MEMORY` is unchanged, and **no default numerical result
changes**. Evidence, strongest first:

1. **The dynamical core is untouched** — the net dynamics diff v0.22.2→v0.23 is **0 lines**
   (P1/P2/P4 were kept as reverts / default-off to preserve identity).
2. The only default-on changes that touch the compiled program are **P0** (M9 radiation
   flux-slice reduction) and **P3** (flat-2-dom root fusion), both proven **CPU
   bit-identical** (P0: 17/17 M9 fields `sha256(before)==sha256(after)`; P3: fused==eager
   carry sha256 identical — and P3 is inert on the 3-domain canary, which stays on the
   eager path).
3. With XLA autotune held fixed (**same-cache control**), the default path is
   **bit-identical** to v0.22.2 (`max_abs=0.0`, 0/178 leaves).
4. Across two independent cold compiles (fresh autotune), the default matches v0.22.2
   **within the autotune floor** — a self-control confirms v0.23-vs-v0.22.2 field
   differences over the 1 h canary are statistically indistinguishable from (in aggregate
   slightly smaller than) v0.22.2 differing from its own second cold compile.

In short: the only difference is the same run-to-run GPU autotune nondeterminism v0.22.2
already exhibits against itself. No masking, clamps, or `nan_to_num`. An independent
adversarial pre-release gap-critic reviewed the integration and returned **RELEASE-SAFE,
no blockers**.

## Validation

- Default byte-equivalence (A1, self-control-anchored) + canary benchmark: GREEN.
- Full CPU test suite: no new regressions vs the pre-merge baseline.
- Per-scheme reference-only fail-closed + default-inertness verified; pre-release gap-critic.
- Deferred post-tag credibility follow-ups: the 24–120 h skill gate and the GPU perf
  confirmations of the P-bundle wins (the wins are launch/compile-count reductions; the
  default numerics are already proven byte-equivalent).
