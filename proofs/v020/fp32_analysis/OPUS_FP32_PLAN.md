# OPUS — v0.20.0 fp32 / mixed-precision kernel rewrite: strategic analysis + implementation & validation plan

**Author:** Opus 4.8 (analyst #1 of 2; the other is a GPT). Independent analysis.
**Date:** 2026-06-21. **Status:** PLANNING ONLY — no GPU runs, no source edits.
**Inputs read:** KERNEL-OPTIMIZATION-FINDINGS-FINAL.md; V0140-FP32-ACOUSTIC-ROADMAP.md;
ADR-003 / ADR-007 / ADR-031; 2026-06-04 v0100 kernel superplan; PRECISION_POLICY.md;
FP32_GRADIENT_PROBLEM_OPEN_CHALLENGE.md; proofs/perf/fp32_downcast_{plan,spec}.md;
proofs/v018/k1_fp32_stability_scoping.md + k1_critic_opus.md; proofs/thompson_perf/*;
proofs/v018/fused_restore/* + maxdom9_speedup/profile_opus.md; the 2026-06-08/13 gpt-fp32
dispatches/reviews; .agent/decisions/v0192-efficiency-roadmap.md; current source
(`contracts/precision.py`, `runtime/operational_mode.py`, `runtime/domain_tree.py`).
**Incorporates the user's 2026-06-21 precision-acceptance policy** (skill-based gates, not
bit-tightness; see §0 and §3).

---

## 0. TL;DR — the headline verdict, and what changed

**There are TWO barriers to a massively-faster fp32 kernel, not one. Every prior attempt
fixed (or tried to fix) the wrong one, and that is the *exact, provable* reason they stalled
at ~1.1×.**

1. **The NUMERICAL barrier** — tiny-difference cancellation (PGF, hydrostatic, acoustic
   accumulation) corrupts naive fp32. This is the barrier the project obsessed over, and it
   is **real but already solved in principle** (perturbation form + compensated summation +
   in-bracket fp64). It is the reason the fp64 *islands* exist.

2. **The STRUCTURAL barrier** — the step is **launch-count + HBM-bandwidth + occupancy
   bound, NOT fp64-ALU bound**, on the grids we actually run. This is the barrier that
   *actually caps the speedup*, and **precision changes do not touch it.** This is why the
   one rigorous valid mixed-precision build (GPT FP32-S2, 2026-06-14) measured **1.106×**
   with the kernel launch topology essentially unchanged (12,136 → 11,945 launches/step) and
   the two dominant kernels at **identical fp32-vs-fp64 wall time**
   (`.agent/reviews/2026-06-13-gpt-fp32-s2-result.md`, hereafter **[S2]**).

**The "~4× proven compact-explicit-fp64-island" framing is stale and must be retired.** The
4.29× was the *invalid* global-fp32 proxy (acoustic detonates); the *valid* compact-island
path has been measured at **1.04–1.11× three times** by two independent agents
(`k1_critic_opus.md` E1/E2/E5). Repeating "shrink the fp64 islands in pure XLA" reproduces
1.1× a fourth time. I will not propose that.

**What the user's 2026-06-21 policy changes — and this is the genuinely new degree of freedom:**
The prior negatives were all measured under the **frozen-manifest tight-tolerance
constraint**, which is *exactly what forced the fp64 islands to stay fp64*. The K1 critic's
decisive finding (E3) was that "the transient VRAM peak is precision-**insensitive** — it is
dominated by the fp64 cancellation islands (EOS/PGF/advance_w) + qke." **That was a
consequence of keeping those islands fp64 — which the tight tolerance mandated.** Relaxing
the acceptance gate to *24–120 h wind/temp/cloud skill + no-blow-up* removes that mandate for
most of the islands. That is the one thing the prior 1.1× experiments **never tested**, and
it is the only lever that can make the transient peak (hence VRAM and HBM traffic) actually
shrink. So the relaxation is not cosmetic — it is the prerequisite that makes the bandwidth
and VRAM wins physically reachable.

**The honest bottom line, stated up front so the roadmap is built on it:**

- **On the nested all-7 max_dom=9 benchmark, fp32 will NOT deliver a large speedup, and the
  reason is provable hardware physics, not numerics:** every domain (d02 ≈ 18 k cols; leaves
  d03–d09 ≈ 1.5–8.6 k cols, 39×39) has an 11–74 MB working set that is **resident in the
  5090's 96 MiB L2 cache**, so the DRAM-bandwidth edge fp32 buys is **unused**; and the
  kernels are tiny (~1.5 µs, no saturating kernel) so the wall is **launch/occupancy bound**,
  where fp32's faster ALU is irrelevant (`k1_critic_opus.md` E6/E7;
  `proofs/v018/maxdom9_speedup/profile_opus.md`). **Expected all-7 speedup from aggressive
  fp32 ≈ 1.1–1.4×** (better than the compact-island 1.0× only because aggressive downcasting
  *removes* the island+convert overhead and the fp64 PCR solve, not because occupancy
  improves). The all-7 nest's role is the **correctness / stability / 24–120 h skill gate**,
  not the speed demonstrator — and it stays fully functional at every step.

- **The massive speedup lives at DRAM-bound scale** (a single large domain / 1 km Canary /
  bigswiss, working set > 96 MiB L2). There the GPU *already* projects **~4.35× core vs CPU
  in fp64** but is **VRAM-blocked** (OOM at ~167 k cols on an 18.8 GiB fp64 transient;
  `k1_critic_opus.md` E8/E9). fp32's job at scale is **(a) fit the grid fp64 OOMs on
  [VRAM — the capability prize], (b) halve HBM traffic [bandwidth, ~1.5–2× on the
  DRAM-bound kernels], (c) up to ~4.3× ALU on the compute-dense kernels.** Realistic combined
  at scale: **2–3.5× wall + the 1 km unlock**, IF the perturbation rewrite makes the
  transient actually shrink (the G0 probe, §4, settles this in ~1 GPU session before any
  rewrite is funded).

So the plan is: **(I)** retire the wrong target, **(II)** use the relaxed tolerance to
downcast aggressively in perturbation form so the *big arrays* (p′, ph′, mu′, w, acoustic
scratch) become fp32 and the transient finally shrinks, **(III)** demonstrate the speedup
where it is physically realizable (DRAM-bound scale), **(IV)** keep the all-7 nest as the
stability/skill gate, and **(V)** attack the structural barrier (launch/HBM) with the fused
acoustic+vertical-solve kernel that [S2] explicitly identified as "the next step — structural,
not another dtype tweak."

---

## 1. STRATEGIC ANALYSIS (brief deliverable a)

### 1.1 Two barriers — and the precise decomposition of "why 1.1×"

The project repeatedly framed the obstruction as **numerical** ("fp32 is impossible because
of cancellation/drift"). The most rigorous measurements show that framing is **half right**
and that it has been hiding the binding constraint. Decompose the per-step cost on the grids
we run:

| Cost component | fp64→fp32 effect | Binding on our grids? | Evidence |
|---|---|---|---|
| Kernel-launch latency (~12 k tiny dependent kernels/step) | **none** (precision-invariant) | **YES** on tiny nests | [S2]: 12,136→11,945 launches/step; `KERNEL-OPTIMIZATION-FINDINGS-FINAL.md` §1 (~10–18 k kernels/step, ~10 µs, latency-bound) |
| DRAM bandwidth (HBM traffic) | ~2× *if* arrays are fp32 *and* working set > L2 | only on **large** grids | `k1_critic_opus.md` E7 (L2-resident < 96 MiB → BW edge unused); [S2] modeled bytes −13.6% only |
| fp64 ALU throughput (1/64 peak rate) | up to ~4.3× *if* compute-bound | rarely — only compute-dense large kernels | `FP32_GRADIENT_PROBLEM…` §3 (~4.3× compute-bound, ~1.0× real step) |
| XLA temp liveness / VRAM transient | shrinks *only if islands also fp32* | gates **capability** (1 km fit) | `k1_critic_opus.md` E3 (temp 5305→5379 MiB *unchanged* with islands fp64) |

**The provable reason naive/compact fp32 yields ~1.1×:** on the all-7 nest the binding cost
is launch latency + occupancy (top row), which is precision-invariant; on mid grids the
working set is L2-resident so the bandwidth row doesn't bind; and the compact-island approach
kept the big arrays + cancellation temps fp64, so neither the bandwidth row nor the VRAM row
moved. [S2] is the smoking gun: it **halved the HLO fp64 token count** (76,067 → 38,245) and
the oracles passed — i.e. it *did* shrink the islands — yet temp memory fell 1.1 %, modeled
bytes 13.6 %, modeled FLOPs 2.1 %, and wall 1.106×, because **the two dominant kernels
(`loop_multiply_fusion_1` 592 ms × 32,256 instances; `input_reduce_fusion_19` 383 ms) are not
fp64-ALU-bound — they have identical fp32 and fp64 wall time** and the fp64 PCR tridiagonal
solve survived. **Shrinking islands is necessary; it is provably not sufficient.**

### 1.2 What the numerical barrier really is (and why it is defeatable)

The meaningful atmospheric quantities are tiny differences of huge numbers: a ~1 Pa pressure
gradient on a ~10⁵ Pa background; a buoyancy residual between two ~10⁵ m²/s² geopotential
terms (`FP32_GRADIENT_PROBLEM…` §1b). fp32 has 24 mantissa bits (~7 digits); after a
near-cancellation ~2 digits survive, and per-substep increments swamp under accumulation.
The four binding operations (`FP32_GRADIENT_PROBLEM…` §2.2):

1. **Horizontal PGF** — base difference `p̄_R − p̄_L` is a difference of O(10⁵) numbers;
   stored as an fp32 *total* it is quantized to ULP(10⁵) ≈ 8e-3 Pa **at storage**, before any
   subtraction (this is why "fp32 totals + fp64 island" failed at 27×–127× — the loss is at
   storage, not at the difference: `k1_critic_opus.md` E4).
2. **Hydrostatic / vertical PGF + buoyancy** (`advance_w`) — near-exact cancellation against
   gravity.
3. **Acoustic accumulation** — `X_{n+1}=X_n+Δt·tend` iterated 10⁵–10⁷ times; swamping →
   secular drift.
4. **EOS / mass-flux divergence / conservation telescoping sums.**

**The defeat is an exact algebraic identity, not a tolerance compromise** (this is the key —
it lets us downcast WITHOUT loss where it matters): store the **perturbation** `X′ = X − X̄(k)`
(reference state `X̄` is a static 1-D column, negligible memory) so the *stored* number is
O(1–100), which fp32's 24 bits resolve cleanly; carry the long-horizon accumulation with
**compensated (Kahan/Neumaier) summation** (~4 fp32 flops, recovers ~fp64-grade sums); and at
the ≤5 residual cancellation brackets that still cancel even in perturbation form, upcast to
fp64 **in-register** (or double-single). Measured on the target GPU
(`FP32_GRADIENT_PROBLEM…` §6): PGF cancellation 3.2e-3 (naive) → 1.6e-7 (perturbation) →
1.1e-10 (double-single); accumulation drift ~200× better with compensation. **So the
numerical barrier is solved by storage form + a handful of register-level fp64/EFT brackets,
not by keeping whole 3-D arrays fp64.**

### 1.3 What the user's 2026-06-21 policy changes (and what it does NOT)

> *"We accept less-tight margins vs WRFv4… keep fp64 ONLY where genuinely required for
> stability or for wind/cloud/temp 24–120 h skill… prioritize the massive speedup."*

**It LIFTS the constraint that created the islands.** Every fp64 lock in
`precision.py:147–230` traces to a *tolerance* argument ("FP32 leaves ~2 sig digits —
operationally insufficient for momentum balance", ADR-007 §pressure-gradient) gated against
the **frozen v0.14 grid-delta manifest** (near-noise field tolerances). Under the new policy
the binding question for each field becomes **empirical and forecast-relevant**: *does fp32
storage of this field (in perturbation form) blow up, or measurably degrade 24–120 h
wind/temp/cloud skill?* If not → it goes fp32. This re-opens **p′, ph′, mu′, w** — the big
arrays whose fp32 storage is the *only* path to a real VRAM + bandwidth win. The prior
negatives never tested this because the manifest forbade it.

**It does NOT change the structural barrier.** Launch latency, occupancy, and L2-residency
are hardware physics; they are precision- and tolerance-invariant. Relaxing tolerance lets us
put *more* in fp32, but on a grid that is occupancy-bound, more fp32 still doesn't speed up.
So the relaxation is necessary for the *bandwidth/VRAM* wins (at scale) but cannot, by itself,
make the all-7 nest fast. **Both facts must be in the roadmap.**

### 1.4 The all-7 nest: provably ~1.0–1.4×, and why that is fine

`profile_opus.md` + `k1_critic_opus.md` E6/E7 establish: the all-7 leaves are tiny, their
working sets fit L2, the kernels are ~1.5 µs with no saturating kernel, and the GPU edge
(20× DRAM BW, parallelism) is therefore **unexposed**. fp32 cannot accelerate a kernel that is
waiting on launch latency or starved for parallelism. **This is the EXACT PROVABLE REASON
fp32 ≈ 1.0× on the all-7 nest, independent of tolerance.** Aggressive downcasting recovers a
little (remove island+convert overhead, drop the fp64 PCR solve, fewer full-grid fp64
temps → less L2 pressure), so I project **1.1–1.4×** there — to be measured, not promised.

**Implication for the benchmark's role:** the all-7 nest is the **functional / stability /
skill gate** ("does the whole 9-domain cascade stay finite and skill-preserving in fp32?"),
which is exactly the user's hard constraint. It is **not** the speed demonstrator. The speed
demonstrator must be a DRAM-bound fixture. Both are reported; conflating them is the category
error the brief's "bottleneck = fp64 dycore" premise makes.

### 1.5 Where the ~4× actually lives — DRAM-bound scale

`k1_critic_opus.md` E8/E9 + `bigswiss_gpu_benchmark`: at ≥147 k cols the working set exceeds
L2 → the GPU enters its **DRAM-bound regime** where its 20× bandwidth edge and parallelism
finally pay. There the **fp64 core already projects ~4.35× vs CPU** — but it OOMs at ~167 k
cols on an 18.8 GiB fp64 transient. So at scale the win is *already real in fp64* and the
binding constraint is **VRAM**. fp32's value at scale, in priority order:

1. **VRAM / capability (most certain):** perturbation-form fp32 storage roughly halves the
   resident + transient working set → 1 km Canary and bigswiss grids that fp64 OOMs on now
   **fit on one card**. This is the deliverable thesis (capability, not single-card wall;
   `feedback_gpu_scalable_positioning…`). **Caveat from E5: the compact-island build made
   VRAM *worse* at 147 k (islands ADD buffers). The rewrite must NET fewer bytes — that is a
   hard gate, not an assumption (§4 G0).**
2. **Bandwidth (~1.5–2× on DRAM-bound kernels):** halved HBM traffic on the now-fp32 big
   arrays directly speeds the bandwidth-bound kernels.
3. **ALU (up to ~4.3× on the compute-dense fraction):** EOS, PGF, coefficient builds,
   acoustic math in fp32.

Realistic combined at scale: **2–3.5× wall + the 1 km unlock.** The full 4.3× is the
ALU-only ceiling and is not reachable while any kernel stays bandwidth/launch bound.

### 1.6 The fp64-island boundary, DEFINED PRECISELY under the new policy

> **Island rule (binding):** a field/op stays fp64 **iff** fp32 (in perturbation form, with
> in-bracket fp64 where cheap) either **(K-STAB)** causes blow-up / non-finite / loss of the
> implicit-solve conditioning, **or** **(K-SKILL)** measurably degrades 24–120 h wind / temp /
> cloud skill beyond §3's bands. **Tiny pointwise differences vs WRFv4 are NOT a reason to
> stay fp64** (explicit the user directive). Everything else is fp32.

Applying this to the current locks (`precision.py:147–230`), the islands collapse to a
**short list** (detailed in §2). The static base state `X̄` stays fp64 but is free (1-D). The
only genuinely-fp64 *3-D* survivors are the ones with a STABILITY mechanism: the
implicit-w/φ tridiagonal **coefficients + solve** (conditioning), `qke`/`qsq` (measured 1 km
non-finite in fp32 — `precision.py:175–198`), and possibly the surface-layer Monin–Obukhov
stability functions (ill-conditioned near neutral). Even several of these can be
**fp32-storage / fp64-in-register-bracket** rather than fp64-resident.

---

## 2. PER-VARIABLE fp32 ACCEPTANCE GATES (the user's explicit ask)

Classification under the new policy. **Carve-outs (the user):** cumulative variables and
QVAPOR-class are allowed to drift / are already-known-imperfect, so they get *stability-only*
gates (no blow-up), not skill-tightness gates. **Skill-priority (the user):** wind (U, V, U10,
V10), cloud (qc, qi, qr, qs, qg, cloud fraction), temp (T, theta, T2) get the binding 24–120 h
skill gate.

| Field(s) | Current | v0.20 target | Gate class | Concrete acceptance gate |
|---|---|---|---|---|
| **u, v** (+ u10/v10 diag) | fp32 | **fp32** (storage + non-acoustic arith) | **K-SKILL (wind, top priority)** | 24/48/72/120 h U/V & U10/V10 vs CPU-WRF/fp64-GPU: skill (ACC / RMSE-vs-obs-noise) **not degraded** beyond the fp64-GPU-vs-CPU gap; per-step finite |
| **theta, T (T2 diag)** | fp32 | **fp32** | **K-SKILL (temp)** | T2/θ 24–120 h skill preserved; the fp64 mass-conserving θ-limiter (`operational_mode.py:1194` `after_mass`) stays fp64-in-bracket |
| **qc, qr, qi, qs, qg, Ni…Ng, Nc, Nn** | fp32 | **fp32** (already) | **K-SKILL (cloud) + K-STAB** | cloud-field 24–120 h skill preserved; non-negative; no blow-up. Thompson internal solve stays fp64-in-column (adapter upcasts) |
| **p′, ph′, mu′** (perturbations) | **fp64** | **fp32 (perturbation-authoritative)** | **K-STAB + K-SKILL** | the high-value reopen. fp32 *perturbation* storage; base `p̄/ph̄/mū` fp64 (1-D, free); gradients = `grad(base)_fp64 + grad(pert)_fp32`; cancellation brackets fp64-in-register. Gate: no blow-up + wind/temp skill preserved + mass drift bounded/non-escalating |
| **w** | **fp64** | **fp32 storage; fp64 in implicit solve** | **K-STAB** | fp32 resident; upcast at `advance_w` coefficient build + Thomas solve (conditioning). Gate: w-column spectra at sea/lee/ridge/peak show no spurious 2Δx growth; no blow-up |
| **p_total, ph_total, mu_total** (aliases) | fp64 | **eliminate from hot carry** | — | reconstruct totals only at I/O / restart / WRF-savepoint / diagnostics. NEVER fp32 total−base in-loop (ADR-031 kill-gate) |
| **implicit-w/φ coefficients + Thomas/PCR solve** | fp64 | **fp64-in-register (or double-single)** | **K-STAB (genuine)** | tridiagonal conditioning is a real stability requirement; keep fp64 *inside the fused bracket* (registers), not as a resident fp64 array. Demote to fp32/DD only if the one-column oracle + 24 h ladder pass |
| **qke, qsq** | fp64 | **fp64** (or fp32+fp64-budget) | **K-STAB (proven)** | measured 1 km d03 non-finite in fp32 after hour 1 (`precision.py:175`). Keep fp64; optional later test: fp32 qke with fp64 tridiagonal/variance budget. Do NOT lead with this |
| **surface flux handles** (ustar, *_flux, tau_*, rhosfc, fltv, t_skin, soil_moisture, roughness_m) | fp64 | **fp64-in-bracket** | **K-STAB** | Monin–Obukhov iteration ill-conditioned near neutral; keep the *stability functions* fp64 (in-column), allow fp32 state in/out |
| **rain_acc, snow_acc, graupel_acc, ice_acc, rainc_acc** | fp64 | **fp32 storage + fp64/Kahan accumulate** | **CARVE-OUT (cumulative)** | the user carve-out: drift allowed, no blow-up. Use compensated add (fp32 store + fp32 compensation) → ~fp64-grade without fp64 storage. Gate: precip totals within a *forecast-relevant* band (not bitwise), monotone, finite |
| **qv (QVAPOR)** | fp32 | **fp32** | **CARVE-OUT (QVAPOR-class) + K-SKILL(cloud-coupling)** | the user carve-out: known-imperfect, drift allowed; but it feeds cloud → keep water-budget non-escalating + no blow-up |
| **base state p̄/ph̄/mū (PB/PHB/MUB), pgeop, static metrics** | fp64 | **fp64 (free)** | — | static 1-D / map factors; negligible memory; fp64 keeps the perturbation identities exact |
| **fp64 boundary leaves** (w_bdy, p_bdy, pb_bdy, ph_bdy, phb_bdy, mu_bdy, mub_bdy) | fp64 | **follow their field** (fp32 where the field went fp32) | K-STAB | once p′/ph′/mu′/w go fp32, their boundary leaves follow in perturbation form; base boundary stays fp64 |

**Net island after v0.20 (the "tiny fp64 islands" the user wants):** static base state (free) +
the implicit-w/φ conditioning bracket (in-register) + qke/qsq turbulence budget + surface
stability functions (in-column). Everything else — including the big p′/ph′/mu′/w 3-D arrays
— is **fp32 resident**. That is the configuration the prior 1.1× experiments were forbidden
from testing, and it is the one that can finally shrink the transient and the HBM traffic.

---

## 3. VALIDATION DESIGN — built around 24–120 h skill + stability, NOT bit-tightness

Per the user: the binding gate is **forecast skill of wind/temp/cloud over 24–120 h + no
blow-up**, with cumulative/QVAPOR carve-outs. This **replaces** the frozen-manifest tiered
identity as the *production* gate (the manifest is retained only as a *secondary diagnostic*,
to localize regressions, never as pass/fail). Concretely:

### 3.1 The skill gate (binding, production)
- **References (ladder, no JAX-vs-JAX as sole oracle — B6 lesson):** (1) CPU-WRF v4 for the
  Canary config where available; (2) the fp64-GPU baseline at the same commit; (3)
  observations (AIFS/AEMET) for true skill where available.
- **Metric:** for wind (U/V, U10/V10), temp (T/theta, T2), cloud (qc/qi/qr/qs/qg, cloud
  fraction): anomaly-correlation + RMSE at **24/48/72/96/120 h** leads. **Pass = the fp32
  build's skill is not worse than the fp64-GPU build by more than the fp64-GPU-vs-CPU-WRF
  gap itself** (i.e. fp32 adds no skill loss beyond the port's existing gap). This is a
  forecast-impact band, deliberately looser than ULP, exactly as the user directs.
- **Why this is defensible (my tolerance call):** the operational signal floor (CPU-vs-obs)
  is ~1–3 K / 2–4 m/s; the fp64-GPU-vs-CPU gap is already T2≈1.35 K. An fp32 perturbation that
  stays an order of magnitude below the *existing* gap (e.g. ΔRMSE < 0.1 K / 0.1 m/s at each
  lead, the ADR-007 §pt5 band) is forecast-invisible. I adopt **ΔRMSE ≤ 0.1 K (temp),
  ≤ 0.1 m/s (wind)** at every lead as the binding skill band, tightening to "no ACC drop at
  120 h" for the longest lead. Cloud: cloud-fraction Brier/equitable-threat not degraded.

### 3.2 The stability gate (binding, hard floor)
- **No blow-up:** every prognostic finite at every step EXCEPT the carve-outs (cumulative +
  QVAPOR-class). Non-finite in wind/temp/cloud/p/ph/mu/w = **hard fail**, revert the offending
  field to fp64-in-bracket.
- **No hidden guards:** no new clamp/mask/finite-guard beyond WRF's own physical limiters
  (project rule; ADR-031 kill-gate). A change that only "passes" by clamping is a fail.
- **Bounded, non-escalating divergence:** the **fp32-divergence-growth metric**
  (`tests/test_fp32_divergence_growth_metric.py`) must show error vs the fp64 oracle that is
  **bounded/saturating, not super-linear**. CRITICAL nuance from the critic (Q3.2): the fused
  cascade already shows P diverging 1.3→20 over 2 h *chaotically while staying skill-PASS* —
  so **vs-fp64-GPU RMSE is a decorrelation measure, not an error**, and must be a *diagnostic*,
  not a pass/fail (a chaotic system decorrelates by construction). Pass/fail divergence is
  measured **only** against the absolute CPU-WRF / climatological envelope.

### 3.3 The early-divergence localization ladder (run before any 24 h campaign — cheap)
Adopt the K1 ladder (`k1_fp32_stability_scoping.md` §"Early-divergence detector"):
1. Static/HLO audit (convert counts, fp64 token counts, live total-aliases, in-loop host
   transfers) — `proofs/perf/fusion_transfer_audit.py` extended with the f32↔f64 convert
   counters (spec'd in `fp32_downcast_spec.md` §Gate P-2).
2. One acoustic substep savepoint (PGF, calc_p_rho, advance_w) vs fp64.
3. One full RK step → 20 steps → 0.1 h → 1 h → 3 h → 24 h (only if all green) → 120 h.
The detector writes a **localization object** naming the first field/step/RK-substage/kernel
to exceed budget. **Abort triggers:** any NaN/Inf in a non-carve-out field; pressure/geo/mass
residual > 25 % of budget by step 20; any wind/temp/cloud field > 50 % of its 24 h band by
1 h; super-linear monotone growth across 20-step/0.1 h/1 h; d03 qke non-finite signature.

### 3.4 The speed/roofline probe (binding — confirm the needle actually moves)
Each step must show its speedup **at a DRAM-bound grid**, not only the all-7 nest:
- **Warm ms/step A/B** fp64 vs fp32 at the largest grid both fit (~130–147 k cols) AND at a
  grid only fp32 fits (1 km Canary / bigswiss ≥ 200 k cols).
- **nsys** kernel/launch census: launches/step, top-kernel share, D2D copies, and (the
  key sentinel) **did the dominant kernels' wall time drop?** ([S2] showed they didn't — that
  is the failure signature to watch for.)
- **`jax` HLO `memory_analysis`** transient-peak decomposition into fp32-able vs fp64-pinned
  bytes, and **peak VRAM** (must NET *lower* than fp64 — the E5 "fp32 made it worse" trap is a
  hard fail).
- **Profiler caveat (honest):** `ncu`/`nsys --gpu-metrics` return `ERR_NVGPUCTRPERM` on this
  box (no SM-occupancy/DRAM-BW counters; `profile_opus.md` §1). Use CUDA-timeline durations +
  grid dims + the L2-residency model (working-set vs 96 MiB) as the occupancy/BW proxy.

### 3.5 Per-step gates (the binding numbers)
| Gate | Threshold | Action if failed |
|---|---|---|
| Stability (all non-carve-out fields finite, 24 h, all-7) | hard | revert offending field to fp64-in-bracket |
| Wind/temp/cloud skill ΔRMSE @24/48/72/96/120 h | ≤ 0.1 K / 0.1 m/s; no 120 h ACC drop | revert offending field; re-gate |
| Divergence vs CPU-WRF envelope | bounded/non-escalating | revert; investigate cancellation site |
| Peak VRAM at the OOM grid | **< fp64 peak** (net byte reduction) | the rewrite is not netting bytes → fix liveness before claiming capability |
| Warm speedup at DRAM-bound grid | **> 2× trajectory** (the keep-bar) | if ≤ ~1.2× and transient still fp64/launch-dominated → STOP, report honest negative |

---

## 4. STEP-BY-STEP IMPLEMENTATION PLAN (incremental, each validated, reversible)

**Ordering principle:** measure-then-commit (the critic's central correction). Cheapest
decisive experiments first; the expensive perturbation rewrite is funded only after a probe
clears a >2×-trajectory bar. Every step keeps `fp64_default` bit-identical (ADR-031
kill-gate) and keeps the all-7 nest green.

### STEP 0 — G0 decision probe (the gate before any rewrite; ~1 GPU session)
**Goal:** settle, cheaply, whether the relaxed-tolerance aggressive-fp32 path can net the
VRAM/bandwidth win that the compact-island path could not — BEFORE funding a multi-sprint
rewrite. This is the critic's G0, upgraded for the new policy.
- Using the existing `worker/opus/v016-fp32-compact` / `worker/perf/v017-fp32-dycore`
  branches **plus** an aggressive variant that downcasts p′/ph′/mu′/w storage (relaxed
  tolerance now permits it) and **drops the fp64 PCR solve to fp32**: warm fp64-vs-fp32 A/B +
  HLO transient decomposition at the largest DRAM-bound grid both fit (~130–147 k cols).
- **Measure:** (a) warm ms/step ratio; (b) peak VRAM incl. transient (must drop below fp64);
  (c) transient split fp32-able vs fp64-pinned; (d) did the dominant kernels' wall drop?
- **GO (fund the rewrite):** transient majority fp32-able AND VRAM nets lower AND ms/step on
  a >2× trajectory at scale. **KILL (report honest negative + redirect to multi-GPU/tiling):**
  transient still fp64-island/qke-dominated AND ms/step ≤ ~1.2× even with p′/ph′/mu′/w fp32.
- **Why this is the right first gate:** it directly tests the *one thing the prior negatives
  could not* (aggressive downcast under relaxed tolerance), at the grid size where speed can
  exist, for ~1 session instead of dozens of sprints.

### STEP 1 — Free wins + harness, precision-invariant (bank immediately, no risk)
- **Eliminate the hot-carry total aliases** `p_total/ph_total/mu_total` (reconstruct only at
  I/O/restart/savepoint) — this is pure liveness reduction, helps every downstream step, and
  is independent of dtype. (`operational_state.py`, `small_step_finish.py`.)
- Extend `fusion_transfer_audit.py` with f32↔f64 convert counters + the per-field dtype
  assertions (`fp32_downcast_spec.md` §Gate P-2, §G-H5). Add the CPU-runnable dtype-stability
  unit test (catches the prior bug classes #1/#2/#4 with no GPU).
- **Gate:** bit-identical `fp64_default`; all-7 unchanged.

### STEP 2 — Lock the fp64 islands as INTRINSIC (force-upcast at boundaries) — correctness scaffold
Make the implicit-w/φ solve, EOS, PGF cancellation brackets force-upcast their inputs to
fp64 **inside the operator** so the lock is intrinsic, not caller-dependent (the ~6
`.astype(jnp.float64)` edits in `fp32_downcast_spec.md` §H-2/H-3/H-7/G-L2/G-L3/G-L4:
`acoustic_wrf.py` calc_coef_w already does this at the historical `:636`; harden `advance_w`,
`calc_p_rho`, `rhs_ph` call sites). This is the safety net that lets later steps downcast
storage without contaminating a genuine-fp64 bracket.
- **Gate:** `fp64_default` bit-identical; HLO shows the islands fp64 and convert counts at
  exactly the named boundaries.

### STEP 3 — Aggressive non-acoustic fp32 (the easy, high-confidence bulk) — already mostly fp32
Confirm u, v, theta, qv + all hydrometeors + numbers run fp32 storage + fp32 non-acoustic
arithmetic end-to-end (flip `daily_pipeline.py` real-case `force_fp64=False`; the matrix +
`_enforce_operational_precision` else-branch at `operational_mode.py:965` already routes
this). Verify the HLO has fp32 fusions (no silent f32→f64 promotion — bug class #1).
- **Gate:** STEP-0/§3 skill ladder to 24 h on all-7 (wind/temp/cloud skill preserved, no
  blow-up); warm A/B at a DRAM-bound grid (expect the bandwidth win on these leaves to start
  showing). This is the proven-safe floor (`fp32_downcast_plan.md` Phase 1, ~1.3–1.8× *was*
  the projection but only at scale).

### STEP 4 — Perturbation-authoritative p′, ph′, mu′ (the high-value reopen) — the rewrite core
This is the step the new policy unlocks and where the VRAM/bandwidth win is won or lost.
- Carry `p′, ph′, mu′` as **first-class fp32 hot fields**; keep static `PB/PHB/MUB` + derived
  **base gradients** fp64 (1-D / precomputed once per RK stage — ECMWF-style stage constants,
  [S2] recommendation #5).
- Compute gradients as **`grad(base)_fp64 + grad(perturbation)_fp32`** — never difference
  fp32 totals (ADR-031 kill-gate; K1 Rank-1). Reconstruct totals only at interfaces.
- Sites: `small_step_prep.py` (perturbation-frame inputs, mandatory `BaseState`),
  `small_step_finish.py` (emit perturbation carries, totals only at I/O), `acoustic.py`
  `advance_uv_wrf` (base-gradient + perturbation-gradient PGF), `rk_addtend_dry.py`
  `large_step_horizontal_pgf` (same decomposition).
- **Gate (binding):** the §3 ladder 1-substep → 24 h on all-7 (no blow-up; wind/temp skill
  preserved); reproduce the v0.17 cancellation challenge as a *failing* baseline and show this
  path does NOT reproduce the 27×–127× corruption; **peak VRAM nets below fp64** at the OOM
  grid (the E5 trap); warm speedup on a >2× trajectory at scale. **Revert per-field on fail.**

### STEP 5 — w storage fp32 (solve stays fp64-in-bracket) — K-STAB gated
fp32 resident `w`; upcast at the `advance_w` coefficient build + Thomas/PCR solve (genuine
conditioning). Follow the ADR-007:70 empirical sub-gate: paired 24 h fp64-w vs fp32-w, inspect
w-column spectra at sea/lee/ridge/peak for spurious 2Δx noise, water-budget non-escalating.
- **Gate:** no blow-up; no spurious vertical noise; wind/cloud skill preserved. Revert to fp64
  resident if spectra dirty.

### STEP 6 — Compensated accumulators (cumulative carve-out) — cheap, removes drift
Replace fp64 storage of `rain_acc/snow_acc/graupel_acc/ice_acc/rainc_acc` + the long acoustic
accumulators with **fp32 storage + compensated (Kahan/Neumaier) summation** (I5;
~4 fp32 flops, ~200× less drift measured `FP32_GRADIENT_PROBLEM…` §6). the user's carve-out
allows drift, but compensation gets ~fp64-grade for near-free and removes a swamping risk.
- **Gate:** precip totals within a forecast-relevant band, monotone, finite over 120 h.

### STEP 7 — The STRUCTURAL lever (the real multiplier): fused acoustic + vertical-solve kernel
This is the step that attacks the *binding* barrier and is what [S2] explicitly identified as
"the next step — structural, not a dtype tweak." Two coordinated moves:
- **(7a) Collapse launches:** fuse the acoustic substep loop (`operational_mode.py:2111`
  `jax.lax.scan(..., unroll=_acoustic_unroll())`) + the vertical solve into **fewer, larger
  kernels** — via XLA fusion tuning first, then a **Pallas/custom megakernel** if XLA can't
  (the v0.15 "S2 deferred Pallas" lever, reopened because we now have a concrete speed target
  and the fused-cascade infra). In a megakernel the **fp64 cancellation bracket lives in
  registers** (never materialized to HBM) → the island has ~zero VRAM/bandwidth cost, the
  launch count collapses, and the fp32 storage win is fully realized.
- **(7b) Replace the double PCR tridiagonal** ([S2] showed `pcrGtsvBatchSharedMemKernel<double>`
  survives mixed mode): precompute stiff coefficients fp64, solve/apply in fp32 (or
  double-single) with the `advance_w` oracle + 24 h ladder re-run.
- **Gate:** launches/step drop materially; the dominant kernels' wall **finally** drops in
  fp32 (the [S2] failure signature is gone); all-7 stays green; speedup at scale.
- **Note:** 7a/7b are the highest-effort, highest-payoff steps. They are gated behind STEP-0
  GO and STEP-4 success so we never build them on an unproven base.

### STEP 8 — 24–120 h validation + capability proof
Full §3 skill ladder to 120 h on all-7 (stability/skill) + the 1 km Canary / bigswiss
capability run (does fp32 fit what fp64 OOMs on?) + warm speedup at scale. Ship if the skill
gate + stability gate pass and the speedup/capability is demonstrated; otherwise report the
honest negative with the exact provable reason from §6.

**Reversibility:** every step is a per-field flip (the matrix + `force_fp64`/mode plumbing is
already field-granular), so a failing field reverts to fp64-in-bracket without unwinding the
others. `fp64_default` stays bit-identical throughout.

---

## 5. ROADBLOCKS NEEDING MATH/COMPUTATIONAL CREATIVITY — ranked ideas (brief deliverable d)

Ranked by expected payoff × probability, under the new (relaxed-tolerance) policy.

**R1 — Perturbation-authoritative storage of p′/ph′/mu′/w (the enabler). PAYOFF: high
(unlocks VRAM+bandwidth); RISK: medium (architectural, but the technique is proven).**
Store perturbations (O(1–100)) not totals (O(10⁵)); base is static 1-D fp64. Gradients =
`grad(base)_fp64 + grad(pert)_fp32`. This is the *only* idea that makes the big arrays validly
fp32, which is the *only* path to a real transient/HBM reduction. It is necessary for every
downstream win. The relaxed tolerance is what finally permits it for p/ph/mu/w.

**R2 — Fused acoustic+vertical megakernel with in-register fp64 brackets (the multiplier).
PAYOFF: high (the structural barrier); RISK: high (Pallas effort; v0.15 deferred it).**
Put the ≤5 cancellation brackets + tridiagonal solve in registers inside one fused kernel.
This is the [S2]-identified real next step and the only thing that removes the
precision-invariant launch overhead AND keeps the fp64 island off HBM. Without it, fp32 caps
at the bandwidth fraction.

**R3 — Compensated / Kahan-Neumaier summation for accumulators + acoustic. PAYOFF: medium
(kills drift cheaply); RISK: low.** ~4 fp32 flops/step, ~200× less drift (measured). Directly
serves the user's cumulative carve-out. Watch: adds registers inside fused kernels (R2 tension).

**R4 — ECMWF-style fp64 stage-constant precompute. PAYOFF: medium; RISK: low.** Precompute
stiff/cancellation-prone coefficients fp64 once per RK stage; emit fp32 products in the
substep loop ([S2] #5). Bounds the fp64 cost to O(stages), not O(substeps×steps).

**R5 — Double-single (DD) at the irreducible brackets only. PAYOFF: medium (validity from
fp32 storage); RISK: medium (DD is ~29× fp32 arith on this GPU — a *speed loss* if it leaks
out of the bracket; `FP32_GRADIENT_PROBLEM…` §3).** Use DD ONLY at the ≤5 points where even
perturbation form cancels and fp64 storage is undesirable, guarded by
`jax.lax.optimization_barrier` (XLA's algebraic simplifier zeroes the error term otherwise —
confirmed, §5 of the open challenge). Prefer in-register fp64 (R2) over DD where possible.

**R6 — Rescaling / non-dimensionalization for conditioning. PAYOFF: medium; RISK: medium.**
Identities like `1/(M_b+mu′) = (1/M_b)/(1+mu′/M_b)`, `log1p/expm1` for small relative
perturbations (K1 Rank-3). Reduces cancellation without fp64; each identity needs a savepoint
oracle (discrete formula placement matters, not just continuous algebra).

**R7 — Per-column / regional precision. PAYOFF: low–medium; RISK: medium.** Steep-terrain
columns (where qke went non-finite) keep fp64; flat/oceanic columns go fp32. Complex control
flow; only if a uniform policy fails the stability gate locally.

**R8 — Stochastic rounding. PAYOFF: low; RISK: medium (unproven here).** Unbiased accumulation
as a fallback where DD is too costly (I7). Weaker than compensation (bounds bias not variance);
keep as a research fallback, not a primary lever.

**The tiny-difference problem specifically (the user's emphasis):** R1 (perturbation form) makes
the *stored* number small so fp32 resolves it; R3/R4 handle accumulation; R5/R2 handle the
residual brackets in registers. Together these keep the fp64 footprint to the static base
(free) + register brackets — i.e. the "tiny islands" the user wants — **without** keeping whole
3-D arrays fp64.

---

## 6. THE MASSIVE-SPEEDUP "GUARANTEE" — honest budget + how we avoid the 1.1× trap

**How this plan avoids re-deriving 1.1×:** the 1.1× trap is "move dtype boundaries in pure
XLA while the islands stay fp64 and the launch topology is unchanged." This plan breaks all
three: (a) the relaxed tolerance lets the *big arrays* go fp32 (transient + HBM actually
shrink — R1); (b) the structural step (R2/7) collapses launches and keeps islands in registers
(the precision-invariant cost finally drops); (c) the speed is *measured at DRAM-bound scale*
where fp32 has leverage, not on the L2-resident nest where it provably cannot.

**Honest speedup budget (to be confirmed by G0 + per-step A/B, NOT promised):**

| Regime | fp32 mechanism | Honest range | Certainty |
|---|---|---|---|
| **all-7 nest** (L2-resident, occupancy-bound) | island/convert removal + drop fp64 PCR | **1.1–1.4×** | high it's *small*; reason provable (§1.4) |
| **DRAM-bound large grid — bandwidth** | halved HBM on fp32 big arrays | **1.5–2×** | medium (needs R1 to net bytes) |
| **DRAM-bound — + ALU on compute-dense** | fp32 EOS/PGF/coeff/acoustic | **2–3.5× combined** | medium (some kernels stay BW-bound) |
| **+ structural fusion (R2/7)** | launch collapse + in-register islands | **toward 3–4×** | lower (Pallas effort/risk) |
| **VRAM / capability** | ~half working set | **1 km grids fit (fp64 OOMs)** | **highest-value, most certain prize** |

**The few mandatory-fp64 ops and their bounded cost:** static base state (1-D, ~0 memory);
implicit-w/φ conditioning bracket + tridiagonal solve (in-register, O(stages) not
O(substeps×steps)); qke/qsq turbulence budget (one small 3-D field — measured not to affect
the speedup, `precision.py:196`); surface stability functions (in-column). **Total mandatory
fp64 ≈ a few register brackets + 2 small 3-D fields — provably tiny.** If even with this tiny
island the speedup is small, §1.4 names the exact provable reason (occupancy/L2-residency),
and G0 surfaces it in 1 session.

---

## 7. RISKS + MITIGATIONS + how the nested all-7 stays functional (brief deliverable f)

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Repeat the 1.1× trap** (move boundaries, islands+launches unchanged) | high if plan ignored | G0 first (STEP 0); measure transient + dominant-kernel wall, not just HLO tokens; structural step (R2/7) is the real multiplier |
| **fp32 p′/ph′/mu′ blows up the acoustic/PGF** (the warm-bubble detonation history) | medium | perturbation form (R1) + in-bracket fp64 (R4/R5); §3.3 ladder catches it at 1-substep; revert per-field |
| **VRAM nets WORSE** (E5: islands add buffers) | medium | hard gate: peak VRAM < fp64 at the OOM grid (STEP-4 gate); eliminate total aliases (STEP 1); in-register islands (R2) |
| **w / steep-terrain instability** (d03 qke non-finite history) | medium | keep qke/qsq fp64; w solve fp64-in-bracket; d03 1 km is the early detector; revert |
| **XLA destroys EFTs / DD** (algebraic simplifier zeroes error terms) | high if DD used | `jax.lax.optimization_barrier` around each rounded intermediate (proven bit-exact, open-challenge §5); prefer in-register fp64 over DD |
| **Pallas megakernel is a dead end / too costly** | medium | it is gated behind G0 GO + STEP-4 success; the bandwidth+VRAM wins (R1) land WITHOUT it; R2 is upside, not load-bearing |
| **Chaotic decorrelation false-fails a physical mode** | medium | divergence pass/fail vs CPU-WRF *absolute envelope* only; vs-fp64-GPU RMSE is diagnostic (critic Q3.2) |

**How every change keeps the nested all-7 max_dom=9 fully functional (the user's hard
constraint):**
- The all-7 nest is the **stability + skill gate on every step** (§3.3 ladder runs on it;
  STEP-3/4/5/8 gates require it green). No step ships if any of the 9 domains goes non-finite
  (outside carve-outs) or loses wind/temp/cloud skill.
- All changes are **per-field reversible** (matrix + mode plumbing is field-granular), and
  `fp64_default` stays **bit-identical** (ADR-031 kill-gate) so the nest can always fall back.
- The fused-cascade orchestration (`domain_tree.py` `_build_fused_cascade_program` /
  `_operational_fused_cascade_factory`) is **precision-orthogonal** — it fuses *domains*, not
  dtypes — so fp32 storage rides inside it unchanged; the v0.19.1 VRAM-leak fix and the
  cold-megacompile management (`v0192-efficiency-roadmap.md` items 1–4) are unaffected, and
  fp32's halved working set *eases* the cold-megacompile RAM (a secondary benefit).
- The nest's two-way feedback + ph′ boundary forcing are flagged high-risk (ADR-031): keep
  their base/reference leaves fp64 initially; demote only after the feedback-path savepoint
  passes.

---

## 8. RECOMMENDATION (for the manager's synthesis)

1. **Retire the "~4× proven compact-island" framing** — it is the invalid global-fp32 number;
   the valid compact-island path is a 3×-confirmed ~1.1×. State the two-barrier model instead.
2. **Run G0 first** (STEP 0, ~1 session): aggressive fp32 (incl. p′/ph′/mu′/w + fp32 PCR) at a
   DRAM-bound grid, measuring transient/VRAM/dominant-kernel-wall — the one experiment the
   prior negatives could not run, now permitted by the user's tolerance relaxation.
3. **If G0 GO:** fund the perturbation rewrite (STEPS 1–6) for the **VRAM + bandwidth** win
   (the certain, high-value prize: 1 km capability + 1.5–2× at scale), then the structural
   fusion (STEP 7) for the multiplier toward 3–4×.
4. **If G0 KILL:** report the honest negative with the exact provable reason
   (occupancy/L2-residency on the nest; transient fp64-island/launch-dominated at scale) and
   **redirect to multi-GPU / transient column-tiling** — which the critic shows reaches a
   defensible >2× *without* fp32, by attacking the actual (VRAM + scaling) bottleneck.
5. **Validation is skill-based** (24–120 h wind/temp/cloud + no-blow-up, with cumulative/QVAPOR
   carve-outs), per the user — the frozen manifest is demoted to a localization diagnostic.
6. **The all-7 nest is the functional/stability/skill benchmark, not the speed demonstrator** —
   demonstrate speed at DRAM-bound scale, prove correctness on the nest.

**My honest expectation:** the **VRAM/capability win is real and high-probability** (R1 nets
fewer bytes once islands can be fp32 — the thing E3/E5 were forbidden from testing); the
**bandwidth win at scale is likely** (1.5–2×); the **3–4× requires the structural fusion
(R2/7)** and is the genuine reach. The all-7 nest itself stays ~1.1–1.4× by hardware law, and
that is the correct, defensible story — not a failure, because the nest's job is fidelity, and
the speedup's job is done where the hardware can actually deliver it.
```

---

*Appendix — key file:line anchors (current source, verified 2026-06-21):*
*`contracts/precision.py:147–230` (PRECISION_MATRIX locks); `runtime/operational_mode.py:965`
(`_enforce_operational_precision`), `:2111` (acoustic substep `jax.lax.scan`, `unroll=`),
`:1194` (fp64 θ-limiter `after_mass`), `:4275/:4302/:4586` (advance-chunk scans);
`runtime/domain_tree.py` (`_build_fused_cascade_program` / `_operational_fused_cascade_factory`,
precision-orthogonal); `dynamics/acoustic_wrf.py` (calc_coef_w / diagnose_pressure_al_alt
fp64 brackets); `dynamics/core/{advance_w,calc_p_rho,small_step_prep,small_step_finish}.py`
(island sites); `daily_pipeline.py` real-case `force_fp64`. Probe/test anchors:
`proofs/perf/fusion_transfer_audit.py` (HLO convert audit), `scripts/precision_bench.py`,
`tests/test_fp32_divergence_growth_metric.py`, `tests/test_m6_precision_matrix.py`,
`scripts/compare_wrfout_grid.py` + `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`
(demoted to diagnostic).*
