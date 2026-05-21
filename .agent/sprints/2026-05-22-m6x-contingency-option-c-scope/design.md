# M6.x Contingency Option (c) — Architecture Scope (Design Analysis)

**Sprint**: `2026-05-22-m6x-contingency-option-c-scope`
**Worker**: Claude Opus 4.7 xhigh
**Date**: 2026-05-22
**Status**: SCOPING (insurance — invoked only if M6.x fails Tier-2 lifted-cap + sanitize <5% + 24h finite + ≥4× speedup)

## 1. Context

M6-S5 (ADR-007 4× verdict) closed FAIL-with-9.70×-throughput. The GPU pipeline clears the constitutional 4× throughput target end-to-end, but the **M4 reduced dycore is not stability-grade at WRF-canonical 3km coupled timesteps**. Root cause per Opus reviewer §Probe 2:

- `dynamics/acoustic.py:69-70` uses proxy constants `c² = 1.0` and `pressure_coupling = 1.0e-3` (physical c² ≈ 1.15e5 m²/s²).
- M4 canonical `mu`-continuity was deferred (M4-CLOSEOUT §3); `state.mu` is carried but never integrated through a dry-mass tendency.
- The acoustic substep treats vertical pressure-gradient explicitly (`acoustic.py:73-76`), giving a strict vertical CFL on tall narrow columns.

M6.x is the binding next sprint: replace those proxies with WRF `dyn_em/module_small_step_em.F` canonical formulation (physical sound speed + per-cell CFL + `mu`-continuity from `dyn_em/module_em.F` advance_mu_t). The Opus tiebreak narrowed M6.x scope to **option (a)-narrowed** (canonical-physics completion, not a full WRF dycore port).

**Reading B risk** (Opus §4): M6-S5 reviewer raised the possibility that "canonical dycore in JAX is 3-6 months, not 2-4 weeks." If that reading turns out correct, M6.x lands RED and the project pivots to **option (c) re-architecture**. This document pre-scopes (c) so a pivot dispatch costs minutes instead of hours.

Three candidate architectures to scope:

- **(c1)** Klemp-Skamarock vertical-implicit acoustic — clean-room implementation from Klemp et al. 2007 paper (not a port of small_step_em.F).
- **(c2)** Semi-implicit integration — treat both acoustic and gravity-wave terms implicitly; eliminate substepping; one Helmholtz-like solve per large step.
- **(c3)** ML-emulator hybrid — train an NN on Gen2 wrfout (or a residual NN around a cheap explicit dycore step) and run inside the coupled loop.

The framing constraint that ties all three together: **preserve M4 architectural invariants** (zero post-init H↔D transfer; SoA JAX pytree of fp64 leaves per ADR-002; XLA-resident state; debug-stripped HLO equivalence). Any option that breaks those invariants pays for it in regression of the M5-S7 9.70× headline.

## 2. Option (c1) — Klemp-Skamarock vertical-implicit acoustic damping (clean-room)

### 2.1 What changes from M6.x?

M6.x ports WRF `module_small_step_em.F` (2089 LoC) faithfully. The complexity of small_step_em.F is real:
- hybrid sigma-pressure coordinate with c1h/c2h/c1f/c2f/c3h/c4h/c3f/c4f vertical metric coefficients (`module_small_step_em.F:572-599`);
- off-centering parameter `epssm` interleaved through advance_uv, advance_mu_t, advance_w (`module_small_step_em.F:624, 1107, 1341-1342, 1444-1448, 1497-1498, 1583`);
- map-scale-factor (msfx/msfy/msft) plumbing in every flux;
- top-lid boundary distinction (`top_lid` argument, `module_small_step_em.F:626, 1396`);
- sumflux time-averaging for ru/rv/ww that couples small step → large step (`module_small_step_em.F:1601-1761`).

If M6.x fails because porting this surface area faithfully is taking weeks of debugging, (c1) drops the WRF-specific scaffolding and re-implements the **canonical Klemp et al. 2007 §3a-c formulation** from the paper. The mathematical heart is unchanged:

1. **Horizontal explicit, vertical implicit acoustic**: small-step update of (u, v, w, p', φ', μ').
   - Horizontal momentum: forward Euler at small dt with physical c² ∂p'/∂x, c² ∂p'/∂y pressure gradient.
   - Vertical w + φ' coupled via tridiagonal Thomas solve per column per substep, off-centered by ε (Klemp et al. 2007 Eq. 19-22). This is **exactly the structure of WRF `advance_w` (`module_small_step_em.F:1186-1597`) and `calc_coef_w` (`module_small_step_em.F:570-652`), without hybrid-coord and msf complications**.
   - Pressure update from divergence with physical c² = γRT̄ at reference state (γ=1.4, R=287, T̄≈260K → c² ≈ 1.04e5 m²/s²).
2. **μ-continuity**: ∂μ/∂t = -∇·(μV) from sumflux of small-step mass fluxes (mimics WRF `advance_mu_t` `module_small_step_em.F:969-1175` but on pure pressure-sigma, no hybrid).
3. **Per-cell CFL diagnostic**: c·dt_sub/dx ≤ 1 and c·dt_sub/dz ≤ 1 enforced per substep.

### 2.2 File ownership

- **REWRITE** `src/gpuwrf/dynamics/acoustic.py` — `acoustic_once` becomes:
  1. compute mass-point divergence (existing fn);
  2. update p' from divergence with physical c² (1-line constant change);
  3. update u, v with horizontal pressure-gradient (explicit, like today but real c²);
  4. **NEW**: per-column tridiagonal Thomas solve for (w_new, φ'_new) — replaces `_grad_z_to_w` + `state.ph + dt_sub*w_next` (`acoustic.py:60-77`).
- **NEW** `src/gpuwrf/dynamics/tridiag.py` — vmapped Thomas solve (see §2.4).
- **MODIFY** `src/gpuwrf/dynamics/step.py` and `rk3.py` — sumflux accumulator carried through `lax.scan` of acoustic substeps and consumed by μ-continuity update at end of large RK stage.
- **MODIFY** `src/gpuwrf/contracts/state.py` — add `mu_tendency` diagnostic field, preserve SoA pytree (ADR-002 frozen).
- **FROZEN**: physics modules, IO, coupling/physics_couplers, validation engines (per M6.x file boundary).

### 2.3 Preserves M4 SoA pytree

Yes. Tridiagonal coefficients (a, b, c, α, γ from `calc_coef_w`) are derived per-substep from existing State leaves (`w`, `ph`, `mu`, `p`). No new persistent storage on the State pytree beyond the optional `mu_tendency` diagnostic. ADR-002 invariants intact.

### 2.4 JAX tridiagonal: `lax.linalg.tridiagonal_solve` vs hand-rolled vmapped Thomas

Two paths, both viable:

- **(c1.A) `jax.lax.linalg.tridiagonal_solve`**: JAX provides a native batched tridiagonal solve since jax 0.4.20. Signature: `tridiagonal_solve(dl, d, du, b) -> x` where dl/d/du are sub/main/super diagonals and b is RHS. Batched over leading axes via vmap. **Pro**: zero-line solver, XLA cuSPARSE-backed on GPU; should match WRF advance_w speed. **Con**: XLA cuSPARSE batch performance is not documented well for the (ny*nx, nz=45) batch shape; may have per-call overhead that hurts for substep loop where it fires `n_acoustic` × 3 stages = 12-18 times per dt_large.
- **(c1.B) Hand-rolled Thomas vmapped over (j, i)**: pre-compute α, γ once per large step (they depend only on slowly-varying μ and c²a, like WRF's `calc_coef_w`); inside acoustic substep just do RHS build + back-substitute, both linear-in-k. Vmapped via `jax.vmap` over horizontal axes. **Pro**: full XLA fusion, no opaque solver call; α/γ amortized across acoustic substeps. **Con**: more code; XLA may serialize the recurrence across k, hurting parallelism on the inner-most axis.

**Recommendation**: prototype both, benchmark on (1, 160, 67, 45) batch. Default (c1.A) for simplicity; fall to (c1.B) if XLA shows >5% wall-time penalty from cuSPARSE call overhead at this batch shape.

**Surprising-XLA risk**: vmapped tridiagonal can spill registers if XLA fully unrolls the k-recurrence at compile time (45-iteration unroll on 8 register-heavy LHS variables). Mitigate by carrying the recurrence inside `lax.scan` (over k) inside vmap (over j, i) — XLA respects scan as a fused loop. Documented risk; not a blocker.

### 2.5 Estimated wall

**5-9 wall days** (3-5 days clean-room implementation following Klemp 2007 §3a-c equations directly; 2-3 days CFL diagnostic + per-stage μ-continuity wiring; 1-2 days Tier-2/Tier-3 re-validation + 24h run).

Faster than M6.x (~16-32h estimate) was optimistic for WRF-canonical fidelity. (c1) trades fidelity for ~2× faster implementation by not porting hybrid-coord, off-centering, msf, and sumflux. Honest: it's still hard. The acoustic tridiagonal + μ-continuity coupling is the hard part of dycore work, regardless of provenance.

### 2.6 Risk

- **Known unknowns**:
  - Per-column tridiagonal in JAX/XLA may have unexpected fp64 numerical noise that triggers sanitize false-positives even on a correct algorithm. Mitigation: validate against a NumPy reference Thomas first; bound max relative residual at 1e-12.
  - Without WRF's off-centering ε, the dispersion relation will differ subtly; gravity-wave phase speeds may be 1-3% off (Klemp 2007 §4 quantifies this). Operational impact: tolerable for Tier-4 RMSE on U10/V10/T2 per the [[feedback_validation_philosophy]] memo; not tolerable if M7 starts gravity-wave-resolved physics work.
- **Cited precedent**: Klemp, Skamarock, Dudhia (2007), "Conservation Split-Explicit Time Integration for the Compressible Nonhydrostatic Equations," *Mon. Wea. Rev.* 135, 2897-2913. Equations 19-22 are the vertical-implicit Thomas solve; §3c is the μ-continuity coupling. WRF dyn_em is the operational descendant — `module_small_step_em.F:817, 890, 1128` cite this paper by name in comments.

## 3. Option (c2) — Semi-implicit integration

### 3.1 Conceptual difference from c1

(c1) is split-explicit with vertical-implicit acoustic — small dt_sub ≈ dt_large / 4-6. Substepping is the cost. (c2) is **fully semi-implicit**: all fast waves (acoustic + gravity) treated implicitly via a single elliptic solve per large step. No substepping. dt_large grows to the advective CFL limit (~4-6× current dt = 40-60s coupled).

### 3.2 Algorithm sketch

Per Robert (1981) and Côté et al. (1998) (CMC GEM model) and ECMWF IFS semi-Lagrangian semi-implicit:
1. Linearize the equations around a reference state (T̄, p̄).
2. Express the prognostic update as `∂X/∂t = L X + N(X)` where L is the linear acoustic-gravity operator (sparse, vertical-direction-dominated) and N is the nonlinear advection/Coriolis remainder.
3. Crank-Nicolson on L (implicit) + explicit on N: `(I - dt/2 L) X^{n+1} = (I + dt/2 L) X^n + dt N(X^n)`.
4. The implicit step reduces to a **3D Helmholtz problem** for one elliptic scalar (typically the geopotential or pressure perturbation) per timestep. With horizontally-constant reference state, the Helmholtz operator separates as `(Lz + k_h² I) φ = RHS` after horizontal FFT.

### 3.3 JAX implementation feasibility

- **Helmholtz via 2D FFT + per-mode tridiagonal**: feasible. `jax.numpy.fft.fft2` along horizontal axes; per-mode k_h²; tridiagonal in vertical for each (k_x, k_y). This is exactly the structure used in pseudo-spectral atmospheric models. JAX supports it efficiently.
- **Difficulty**: doubly-periodic BC assumption is broken at a limited-area regional grid (160×67 d02). Need a **Robin/Dirichlet treatment at lateral boundaries**, which breaks pure FFT and forces an iterative solver (BiCGStab or multigrid).
- **Multigrid in JAX**: doable but nontrivial. JAX has no built-in geometric multigrid. A V-cycle multigrid with 3 levels on (160×67×45) would be ~200 lines of careful indexing. Convergence ~5-10 iterations per timestep.

### 3.4 Net speedup at 3km Canary scale

Back-of-envelope:
- Today: dt_large = 10s, substep ratio 4 → 4 acoustic substeps per RK stage × 3 RK stages = 12 vertical-implicit tridiagonals per dt_large.
- (c2) with dt_large = 40s and one Helmholtz per step: dt_large/4 fewer steps × (1 Helmholtz / 12 tridiagonals) per step ≈ 3× fewer "tridiag-equivalents" overall. **If** Helmholtz costs 5-10× a single tridiagonal, net wall ≈ 0.5-1× current. Maybe 1.5-2× faster, maybe not faster at all.
- ECMWF IFS gets a 6× wall reduction from semi-implicit over fully explicit, but IFS is a global spectral model where the Helmholtz solve is essentially free (per-spectral-coefficient).

### 3.5 Precedent — does any GPU NWP code do semi-implicit?

- **HOMMEXX (E3SM dycore C++/Kokkos port, ORNL)**: semi-implicit for tracer transport (vertical remap) but the dynamics (HOMME) is explicit hyperviscosity Runge-Kutta. Bertagna et al. (2019), "HOMMEXX 1.0: a performance-portable atmospheric dynamical core for the Energy Exascale Earth System Model," *Geosci. Model Dev.* 12, 1423-1441. **Not a semi-implicit dycore precedent.**
- **SCREAM (E3SM atmosphere GPU port)**: uses HOMMEXX dycore — explicit, same as above.
- **Pace (NOAA/Vulcan FV3 in GT4Py)**: FV3 is **explicit** finite-volume with vertically Lagrangian remap; sub-cycled acoustic; not semi-implicit. Dahm et al. (2023), "Pace: A Python-based Performance-Portable Atmospheric Model," *Computing in Science & Engineering*.
- **NEMO (NEMO ocean GPU port via PSyclone)**: not atmospheric.
- **GEM (CMC, Canada)**: semi-Lagrangian semi-implicit, the canonical example. CPU-only. No published GPU port.
- **ICON (DWD/MPI-M)**: split-explicit with vertical-implicit fast waves (similar to c1, not c2 fully implicit). GPU port via OpenACC. Giorgetta et al. (2018), "ICON-A, the Atmosphere Component of the ICON Earth System Model," *J. Adv. Model. Earth Syst.* 10, 1638-1662.

**No published precedent for fully semi-implicit on GPU at limited-area regional scale.** This is significant: it suggests either nobody's tried it (opportunity) or everybody who tried found the elliptic solve too painful on GPU (warning). The latter is more likely given the lateral-boundary multigrid complexity.

### 3.6 Estimated wall

**10-20 wall days** (4-6 days linearization + Helmholtz operator derivation; 4-6 days multigrid/BiCGStab solver; 3-5 days lateral BC treatment; 3-5 days validation). Higher than c1, with significantly higher architectural risk.

### 3.7 Risk

- **Known unknowns**:
  - Multigrid coarsening on a 160×67 grid hits the coarse-grid bottleneck within 3 levels; convergence may be 10-20 iterations (not 5). Performance unclear.
  - Lateral-boundary semi-implicit treatment is a known open research problem. ECMWF avoids it with global spectral grid; limited-area semi-implicit codes (e.g., COSMO) have decades of accumulated boundary trickery.
  - dt_large extension to 40-60s assumes the advective CFL is the binding constraint. At 3km horizontal resolution with 50 m/s jet-stream wind, advective CFL of 0.5 limits dt to ~30s — not 60s. Net step-count reduction is closer to 3×, not 6×.
- **Cited precedent**: Robert (1981), "A stable numerical integration scheme for the primitive meteorological equations," *Atmos. Ocean* 19, 35-46. Côté et al. (1998), "The operational CMC–MRB global environmental multiscale (GEM) model," *Mon. Wea. Rev.* 126, 1373-1418. Wood et al. (2014), "An inherently mass-conserving semi-implicit semi-Lagrangian discretization of the deep-atmosphere global non-hydrostatic equations" (Met Office ENDGame), *Q. J. R. Meteorol. Soc.* 140, 1505-1520.

## 4. Option (c3) — ML-emulator hybrid

### 4.1 What does the NN replace?

Three sub-variants, ranked by ambition:

- **(c3.A) Full coupled-system emulator (Pangu/GraphCast style)**: NN learns `state(t+Δt) = f(state(t), forcing(t))` end-to-end. Replaces the entire coupled loop (dycore + physics) inside the GPU. Pure inference.
- **(c3.B) Dycore-only emulator**: NN learns `state_after_dycore = g(state_before_dycore)`. Physics (Thompson/MYNN/RRTMG/sfclay/Noah-MP) stays as JAX kernels; only the dycore step is NN-emulated. Tighter scope, less generalization burden.
- **(c3.C) Residual / dycore correction**: NN learns `correction = h(state, cheap_explicit_dycore_output)`. A cheap explicit dycore (the broken M4 reduced version with bigger dt) gets stable post-hoc by adding the learned correction. **Hybrid**: physics is explicit and physically sound; NN only corrects the residual that the cheap dycore would otherwise accumulate.

(c3.C) is the most defensible for operational use because it (a) lets physics stay physical, (b) lets the NN learn only the small unstable mode amplification, and (c) degrades gracefully if the NN is out-of-distribution (cheap dycore is still physical, just biased).

### 4.2 Training data — Gen2 corpus

Per [[project_canairy_meteo_baseline]]: Canairy Gen2 has live CPU WRF baseline with **~1 month of d02 1km/3km solutions** on disk under `/mnt/data/canairy_meteo/artifacts/`. NVHPC-built WRF wrfout files.

**Data adequacy assessment**:
- 1 month × 24 forecasts/day × 24h × 6 outputs/h = ~104k snapshot states. Per state, (160×67×45×8 fields × 8 bytes fp64) ≈ 31 MB. Total ~3.2 TB raw — feasible to store; needs decimation for training.
- **GraphCast** (Lam et al. 2023, *Science* 382, 1416-1421) trained on ERA5 1979-2017 (39 years) at 0.25° = ~38 years × 8 timesteps/day = ~110k states global. **Our corpus is per-grid-point comparable but globally tiny** (1 region vs world). Severe out-of-distribution risk on any non-Canary input.
- **Pangu-Weather** (Bi et al. 2023, *Nature* 619, 533-538) trained on ERA5 1979-2017, similar scale. Limited-area regional precedent thinner.
- **NeuralGCM** (Kochkov et al. 2024, *Nature* 632, 1060-1066) is a **hybrid ML+physics** model — closest precedent to (c3.C). It trains an NN correction on top of a dycore. Trained on ERA5 1979-2017.

**Verdict**: with 1 month of Canary-only data, (c3.A) full-emulator is **not viable** without transfer learning from a global pretrained model (e.g., fine-tune GraphCast/FourCastNet on Gen2). (c3.B) dycore-only is **marginally viable** if we train on per-step state pairs (3-7× more samples than per-forecast). (c3.C) residual correction is **most defensible** because it leverages a physical prior; the NN learns a much smaller signal.

### 4.3 Architecture choices

- **FNO (Fourier Neural Operator)** — Li et al. 2021. Resolution-invariant, learns global ops via FFT. Strong precedent in PDE-solver replacement (FourCastNet, Pathak et al. 2022). Fits limited-area if we pad to 2^n size. **Good fit for (c3.B/C)**.
- **3D UNet / Pangu-style 3D Swin transformer** — Bi et al. 2023. Heaviest, best long-range skill, expensive to train. Overkill for 1-month corpus.
- **Graph Neural Network (GraphCast/MeshGraphNet)** — Lam et al. 2023. Mesh-invariant. Best skill on global. Limited-area precedent (LAM-GraphCast, ECMWF AIFS-LAM) is < 1 year old. **Active research area, high uncertainty.**
- **ConvLSTM / ResNet-3D** — older, less data-hungry, easier to debug. Probably the right scaling for 1 month of regional data, even though it's not the headline-grabbing choice.

**Recommendation**: prototype FNO (c3.B residual) on a 256×128×48 padded grid as the v0 attempt. If FNO fails to learn, fall back to ResNet-3D.

### 4.4 Operational risk

- **Generalization**: NN learns Gen2 biases. If Canary's actual operational conditions drift (different ECMWF AIFS initial conditions, different SSTs, novel synoptic regime), the NN extrapolates outside its training distribution. Pangu-style models degrade gracefully; bespoke regional models often degrade catastrophically.
- **Mass / energy conservation**: NNs are notoriously non-conservative. (c3.C) inherits conservation from the explicit dycore underneath if the residual is small. (c3.A) and (c3.B) require explicit conservation post-corrections (CorrDiff, Diffusion-Modulation, etc.).
- **Calibration / uncertainty**: no straightforward way to quantify NN forecast uncertainty without ensemble training. Per [[feedback_validation_philosophy]], the operational gate is Tier-4 RMSE; an NN trained on RMSE will look fine on RMSE but may have catastrophic L_∞ failures (single-cell qv spikes, etc.).

### 4.5 Estimated wall

**4-8 weeks** (1-2 weeks data extraction + featurization from Gen2 wrfout; 1-2 weeks model architecture + training pipeline; 1-2 weeks training runs on the workstation GPU [RTX 5090, 32 GB VRAM — possibly insufficient, may need cloud]; 1 week integration into coupled loop + JAX inference path; 1 week validation). This is roughly an order of magnitude longer than c1 and 2-3× longer than c2.

**Wall-clock-honesty: 4-8 weeks is the optimistic engineering wall. The validation wall before this could be claimed as a forecast capability is longer — likely 3-6 months of operational shadow-mode running against Gen2 ground truth before we'd trust it for daily forecasts.**

### 4.6 Cited precedent

- **Pangu-Weather**: Bi, Xie, Zhang, Wang, Tian (2023), *Nature* 619, 533-538.
- **GraphCast**: Lam, Sanchez-Gonzalez, et al. (2023), *Science* 382, 1416-1421.
- **NeuralGCM** (hybrid, closest to c3.C): Kochkov, Yuval, Langmore, et al. (2024), *Nature* 632, 1060-1066.
- **FourCastNet (FNO precedent)**: Pathak, Subramanian, Harrington, et al. (2022), arXiv:2202.11214.
- **AIFS-LAM (ECMWF limited-area AI forecast)**: ECMWF technical memo, 2024 (limited-area regional precedent — still active research).

## 5. AC4 Decision Matrix

| Criterion | (c1) Klemp-Skamarock clean-room | (c2) Semi-implicit | (c3) ML-emulator (c3.C residual) |
|---|:-:|:-:|:-:|
| **Wall-to-PASS** (smaller = better) | 5-9 days | 10-20 days | 4-8 weeks + validation tail |
| **Architectural risk** (smaller = better) | LOW — well-established algorithm; published reference equations; preserves M4 invariants | MEDIUM-HIGH — no GPU LAM precedent; lateral-BC multigrid is research-grade | HIGH — under-data; conservation concerns; operational-shadow tail |
| **Operational viability for Canary 3km daily** (higher = better) | HIGH — same operational class as WRF dyn_em; will pass Tier-4 RMSE | MEDIUM — if Helmholtz converges, viable; but if dt_large only gets to 30s, marginal speedup | UNKNOWN — depends on NN generalization to live AIFS inputs |
| **End-state code complexity** (smaller = better) | MEDIUM — adds tridiag.py + sumflux accumulator (~400 LoC delta) | HIGH — adds multigrid/BiCGStab + FFT path + Helmholtz operator (~1000 LoC delta) | HIGH — adds training pipeline + checkpoint storage + inference path; permanent dependency on Gen2 corpus integrity |
| **Composite rank** | **1 (best)** | **2** | **3 (last-resort)** |

## 6. Invocation rule (proposed)

If M6.x fails (Tier-2 lifted-cap invariants FAIL OR sanitize firing rate ≥5% OR 24h forecast NaN-explodes OR speedup <4×):

1. **First**: dispatch (c1) Klemp-Skamarock clean-room. Lowest risk, fastest, highest operational viability. If (c1) also fails Tier-2 within 9 wall-days, escalate.
2. **Second**: dispatch (c2) semi-implicit. Higher architectural risk but bounded scope (10-20 days). If (c2) also fails or runs into multigrid pathology, escalate to user.
3. **Last-resort**: dispatch (c3.C) residual ML correction on top of either the broken M4 reduced dycore or the (c1) clean-room result. This is a months-long path; treat as a parallel R&D track, not a primary forecast pipeline replacement.

ADR-017 (next section) codifies this. Status stays SCOPING; transitions to ACCEPTED only if M6.x lands RED and an option is invoked.

## 7. Cross-cutting concerns

- **M5-S7 9.70× speedup retention**: (c1) adds 1 tridiag per substep, ~5-10% wall overhead expected. (c2) replaces substepping with a heavier Helmholtz; net speedup unclear pre-implementation. (c3) inference cost depends on NN size; FNO at this grid is ~50ms/step on RTX 5090 — comparable to today's whole dycore. All three preserve the constitutional ≥4× target with margin; only (c2) carries real risk of regression.
- **Zero-transfer constitutional gate**: (c1) and (c2) trivially preserve. (c3) requires NN weights as a frozen JAX pytree leaf at init; inference is XLA-resident. The training pipeline necessarily runs offline, so it doesn't pollute the operational transfer budget.
- **Debug-stripped HLO equivalence**: all three preserve if they follow the M4 pattern of `debug: bool` static-arg and dead-branch elimination. No new structural issues.
- **fp64 default per ADR-002**: (c1) and (c2) trivially fp64. (c3) NN training typically uses bf16/fp32 for throughput; inference can be promoted to fp64 before re-entry to the dycore pytree (or kept fp32 with a per-field downcast mini-ADR per [[project_backend_decision]] precedent).

## 8. Honest limitations of this analysis

- I have not measured tridiag-solve XLA overhead at the actual batch shape (160×67, nz=45). The c1 wall estimate assumes XLA does what cuSPARSE does in CUDA; if there's a hidden constant cost, c1 could regress to 7-12 days.
- The 1-month Gen2 corpus size is reported in [[project_canairy_meteo_baseline]] but I have not directly counted snapshots on disk. If the corpus is smaller (e.g., 1 week, or stored at 1h output cadence not 10min), c3 viability degrades sharply.
- No published precedent for fully semi-implicit GPU LAM means (c2) wall estimate is speculative; the 10-20 day band could easily be 20-40 days if the lateral-BC multigrid path goes badly.
- The decision matrix weights are subjective. A different weighting (e.g., heavy weight on "operational viability for Canary 3km daily today" with low weight on "wall-to-PASS") would still rank c1 first, but might invert c2 vs c3.

## 9. What's NOT in this scope

- WRF NMM port (the original c1 framing): NMM source not present at `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_nmm/`. NMM is also operationally deprecated (NCEP moved to FV3). Re-framed c1 as clean-room Klemp-Skamarock based on the original journal paper.
- Multi-GPU dycore decomposition: orthogonal to the c1/c2/c3 choice; ADR-002 halo `apply_halo` placeholder still applies.
- ADR-007 mixed-precision policy: unchanged by any of c1/c2/c3.
- M7 operational deployment work: any of c1/c2/c3 must land first.

---

**Deliverables**: this file + `c1-klemp-skamarock-contract.md` + `c2-semi-implicit-contract.md` + `c3-ml-emulator-contract.md` + `.agent/decisions/ADR-017-m6x-contingency-options-scope.md`. Worker pushes to `worker/opus/m6x-contingency-option-c-scope` and exits. If M6.x lands GREEN, this directory becomes archived design rationale.
