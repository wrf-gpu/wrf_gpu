# v0.20.0 — fp32/mixed-precision kernel: SYNTHESIZED sprint plan (manager, from Opus ∥ GPT)

**Author:** manager (Opus), synthesizing the two independent analyses
(`OPUS_FP32_PLAN.md` + `GPT_FP32_PLAN.md`, 2026-06-21). **Status:** plan; to be
GPT-critiqued then finalized into the v0.20.0 roadmap. Sources agree to a remarkable degree;
where they differ it is emphasis (Opus: where speed physically lives; GPT: the per-phase
mechanics + falsifiability gates) and both are folded in.

## 0. Headline verdict (both analysts, independent, CONVERGENT)

**"Turn on fp32" and "conservative mixed precision" are both proven traps; the prior
~1.1× is NOT a tuning failure, it has an exact provable cause — TWO barriers:**
1. **NUMERICAL** (tiny-difference cancellation in PGF/hydrostatic/acoustic) — REAL but
   defeatable: perturbation form + compensated sums + in-register/in-bracket fp64.
2. **STRUCTURAL** (launch-count + HBM-bandwidth + occupancy bound, NOT fp64-ALU bound on the
   grids we run) — this is what *caps* the speedup, and **precision does not touch it.**
   Smoking gun: the one rigorous valid mixed build (GPT FP32-S2) measured **1.106×** with
   launch topology unchanged and the two dominant kernels at *identical* fp32/fp64 wall.

**Honest reframe (this is the answer to the user's "or find the provable reason"):**
- On the **nested all-7 max_dom=9 benchmark, fp32 alone is provably capped (~1.1–1.4×)** —
  working set is L2-resident (96 MiB L2 on the 5090) so fp32's bandwidth edge is unused, and
  kernels are tiny/launch-bound so fp32's ALU is irrelevant. **The nest's role is the
  stability + 24–120 h wind/temp/cloud skill gate, NOT the speed demonstrator.** It stays
  fully functional at every step (the mandatory benchmark).
- The **massive win lives at DRAM-bound scale + the 1 km capability unlock**: a single large
  domain / 1 km Canary (working set > L2) where the GPU already projects ~4.35× core vs CPU
  in fp64 but is **VRAM-blocked (OOM)**. fp32's job at scale: (a) **fit the grid fp64 OOMs on
  [the capability prize]**, (b) halve HBM traffic (~1.5–2× on DRAM-bound kernels), (c) up to
  ~4.3× ALU on compute-dense kernels → realistic **2–3.5× wall + the 1 km unlock**.
- **the user's relaxed-tolerance policy is the enabler, not cosmetic:** every prior negative was
  measured under tight frozen-manifest tolerance, which is *exactly what forced the big
  arrays + cancellation islands to stay fp64* (so the transient/VRAM/HBM never shrank).
  Relaxing to "no blow-up + wind/temp/cloud 24–120 h skill" is the one lever never tested,
  and the prerequisite that makes p′/ph′/mu′/w validly fp32.

## 1. Acceptance policy (the user 2026-06-21, baked in) — per-variable gates
- **Hard floor:** NO variable blows up / goes non-finite — EXCEPT documented cumulative vars
  (rain/snow/graupel/ice acc) and QVAPOR-class (already-known issues). The frozen grid-delta
  manifest becomes a **diagnostic comparator, not the primary veto** once the fp32 skill gate
  passes (GPT Gate D).
- **Binding acceptance = 24–120 h forecast skill of WIND, TEMP, CLOUD** at a defensible margin
  that does not degrade the 24–120 h forecast. Per-variable tolerances: tightest on wind
  (U/V/W layers) + cloud water/ice + T; looser on pressure/geopotential perturbations and
  moisture totals; carve-out (drift allowed, but use compensated sums) for cumulative vars.
  (Full per-variable table: GPT §7; Opus §2.)

## 2. MEASURE-FIRST decision gate (do this BEFORE funding the rewrite) — Opus STEP 0 = GPT Gate A+B
**G0 probe (~1 GPU session, when GPU free):** on the largest DRAM-bound grid that fp64 fits
(~130–147 k cols), warm fp64-vs-fp32 A/B of an **aggressive** variant that downcasts
p′/ph′/mu′/w storage AND drops the fp64 PCR solve to fp32 (relaxed tolerance now permits).
Measure: (a) warm ms/step ratio; (b) peak VRAM incl. transient; (c) transient split
fp32-able vs fp64-pinned; (d) did the dominant kernels' wall drop?
- **GO (fund the rewrite):** transient majority fp32-able AND VRAM nets lower AND ms/step on
  a >2× trajectory at scale.
- **KILL (report honest negative; redirect to multi-GPU/tiling + structural-only):** transient
  still fp64-island/qke-dominated AND ms/step ≤ ~1.2× even with p′/ph′/mu′/w fp32.
This tests the one thing prior negatives could not, at the grid where speed can exist, for one
session instead of dozens of sprints. **Gate B corollary:** if at any point the large-grid
acoustic/dycore component is not ≥2.5× warm (or the profiler doesn't show a *different*
dominant component), STOP and write the exact-cause proof — do not burn 120 h validating a
broad-island path.

## 3. Step-by-step sprint (incremental, each validated + reversible; fp64_default stays bit-identical)
- **S1 Free wins + harness (precision-invariant, no risk, do regardless):** remove the
  hot-carry total aliases `p_total/ph_total/mu_total` (reconstruct only at I/O/restart/
  savepoint) — pure liveness reduction; + convert-counter / per-field-dtype audit tooling +
  CPU dtype-stability unit test (catches silent f32→f64 promotion).
- **S2 Lock fp64 islands INTRINSIC:** EOS / PGF cancellation brackets / implicit-w·φ solve
  force-upcast inputs to fp64 *inside the operator* (caller-independent safety net) so later
  storage downcasts can't contaminate a genuine-fp64 bracket. Gate: fp64_default bit-identical.
- **S3 Aggressive non-acoustic fp32 (high-confidence bulk):** u,v,θ,qv,hydrometeors,numbers
  fp32 storage + fp32 non-acoustic arithmetic end-to-end; verify HLO has fp32 fusions (no
  silent promotion). Gate: skill ladder to 24 h on all-7 + bandwidth win starts at scale.
- **S4 Perturbation-authoritative p′,ph′,mu′ (THE rewrite core; the new policy unlocks it):**
  carry p′/ph′/mu′ as first-class fp32; static PB/PHB/MUB + base gradients fp64 (1-D, once per
  RK stage); compute `grad(base)_fp64 + grad(pert)_fp32` — NEVER difference fp32 totals. Gate:
  reproduce the v0.17 cancellation challenge as a *failing baseline* + show this path does NOT
  reproduce the 27×–127× corruption; peak VRAM nets below fp64 at the OOM grid; skill ladder.
- **S5 w storage fp32 (solve fp64-in-bracket), K-STAB gated:** paired 24 h, inspect w-column
  spectra (sea/lee/ridge/peak) for spurious 2Δx noise; revert if dirty.
- **S6 Compensated accumulators (cumulative carve-out):** fp32 storage + Kahan/Neumaier for
  precip + long acoustic accumulators (~4 fp32 flops, ~200× less drift).
- **S7 STRUCTURAL lever (the real multiplier; [S2]'s "next step — structural not dtype"):**
  (7a) fuse the acoustic substep loop + vertical solve into fewer/larger kernels (XLA fusion
  first, then a **Pallas megakernel** if XLA can't) so the ≤5 fp64 cancellation brackets live
  **in registers** (zero HBM/VRAM cost) and the launch count collapses; (7b) replace the
  double PCR tridiagonal (precompute stiff coeffs fp64, solve/apply fp32 or double-single).
  Gate: launches/step drop materially AND the dominant kernels' wall finally drops in fp32 (the
  [S2] failure signature gone). Gated behind G0-GO + S4-success — never built on an unproven base.
- **S8 24–120 h validation + capability proof:** full skill ladder to 120 h on all-7 +
  1 km Canary/bigswiss capability run (does fp32 fit what fp64 OOMs on?) + warm speedup at
  scale. Ship iff skill+stability pass AND speedup/capability demonstrated; else honest
  negative with the exact provable cause.

## 4. Validation / probes / tools (built around skill+stability, not bit-tightness)
- **Skill ladder (cheap→expensive, pass each rung before climbing, NEVER widen mid-run):**
  1-substep → 1-step → 1 h → 6 h → 24 h → 120 h on the all-7 nest, checking wind/temp/cloud
  skill + no-blow-up at each rung.
- **Early-divergence localization:** savepoint/oracle diffs per operator; the fp32-divergence-
  growth metric (`tests/test_fp32_divergence_growth_metric.py`); per-field A/B to localize the
  first operator that breaks a gate.
- **Speed/roofline probe:** warm fp64-vs-fp32 ms/step + launch-count + HBM-bytes + occupancy at
  a DRAM-bound grid (confirm the needle moves for the *named* reason).
- **HLO/liveness/transfer audits:** `fusion_transfer_audit.py` + convert counters + temp-memory
  + no in-loop host transfer; HLO shows no full-grid total aliases in the hot carry.
- **Final-claim artifact set (GPT Gate E — mandatory for any v0.20 perf claim):** nest
  wall/fc-h, fp64 GPU baseline, CPU denominator, warm (compile-excluded) timings, profiler
  artifacts, transfer audit, HLO dtype/liveness, component timing table, + every remaining fp64
  island named with the reason it stays.

## 5. Roadblocks needing math/computational creativity — ranked (solves the tiny-diff trap)
- **R1 Perturbation-authoritative storage (the enabler):** store O(1–100) perturbations not
  O(10⁵) totals; base static 1-D fp64; gradients = base_fp64 + pert_fp32. The ONLY thing that
  makes the big arrays validly fp32 → the only path to real transient/HBM reduction.
- **R2 Fused acoustic+vertical megakernel with in-register fp64 brackets (the multiplier):**
  the ≤5 cancellation brackets + tridiagonal in registers in one kernel → removes the
  precision-invariant launch overhead AND keeps the fp64 island off HBM.
- **R3 Compensated (Kahan/Neumaier) sums** for accumulators+acoustic (kills drift, ~free).
- **R4 ECMWF-style fp64 stage-constant precompute** (bound fp64 cost to O(stages)).
- **R5 Double-single (DD) ONLY at irreducible brackets** (guard with optimization_barrier;
  DD is ~29× fp32 arith here — a speed loss if it leaks; prefer in-register fp64).
- **R6 Rescaling/non-dimensionalization** (1/(M_b+mu′) identities, log1p/expm1) — each needs a
  savepoint oracle. **R7 per-column/regional precision** (steep terrain fp64) — fallback.
  **R8 stochastic rounding** — research fallback only.

## 6. Honest speedup budget + how we avoid re-deriving 1.1×
The 1.1× trap = "move dtype boundaries in pure XLA while islands stay fp64 and launch topology
is unchanged." This plan breaks all three: (a) relaxed tolerance lets the big arrays go fp32
(transient+HBM shrink, R1); (b) the structural step (R2/S7) collapses launches + keeps islands
in registers; (c) speed is measured at DRAM-bound scale where fp32 has leverage, not on the
L2-resident nest where it provably cannot. **Expected:** nest ~1.1–1.4× (stability/skill gate,
not the demo); DRAM-bound scale **2–3.5× + the 1 km capability unlock** IF G0 clears. If it
does not, the deliverable is the exact provable negative (Gate B/E), not a vague "fp32 didn't
help."

## 7. Nested all-7 functionality (the benchmark) — preserved throughout
Every step is a per-field flip (the matrix + force_fp64/mode plumbing is already field-granular)
→ a failing field reverts to fp64-in-bracket without unwinding the others; fp64_default stays
bit-identical; the all-7 skill+stability ladder is the gate on every step. The nest is never
the speed demo (provably capped) but is the always-green correctness/skill anchor.

## 8. Decision gates (GPT A–E, binding)
A compactness-before-speed (no total aliases in hot carry; fp64 live+temp drop; convert local)
· B component-speed-before-long-forecasts (≥2.5× warm large-grid acoustic/dycore or STOP+prove)
· C stability ladder (pass each rung, no mid-run gate widening) · D skill acceptance
(wind/temp/cloud 24–120 h + per-variable gates; manifest = diagnostic) · E final-claim artifact
set (full timing/profiler/HLO/island accounting).

## Manager note for the roadmap
This is point 1 of v0.20.0. The **structural lever S7 overlaps point 3** (on-device
boundary-builds + fused kernel) — they are the same launch/HBM attack and should be planned as
one structural workstream. Points 3 & 4 analyses (in progress) attach next; then GPT critiques
the assembled roadmap and the manager finalizes.
