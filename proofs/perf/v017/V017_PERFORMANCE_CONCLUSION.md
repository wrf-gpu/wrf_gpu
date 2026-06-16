# v0.17 PERFORMANCE CONCLUSION — Can one RTX 5090 beat 28-rank CPU-WRF by ~6× on the 1 km Canary? And if not, WHY?

**Author:** Opus 4.8 (1M ctx), 2026-06-14. **Branch:** `worker/perf/v017-perf-conclusion`
(worktree `.wt-perfreport`, detached at **v0.16.0** — `git diff v0.16.0 -- src/` is empty;
this is a **documentation + plotting** synthesis, no GPU, no benchmarks, no model-code change).
**Mandate:** the principal's "explain why + full documentation" deliverable. This is the single
authoritative answer; it synthesizes the already-committed MEASURED v0.17 perf evidence into one
proof-quality conclusion. Every number is labeled **MEASURED** or **PROJECTED**, and every
measured number cites the committed proof file it came from.

> **Method doctrine inherited from every source report:** *device-time (CUDA stream) is the truth
> metric; host-wall is async-amortized and lies.* This was proven the hard way by the megakernel
> and R2 experiments below and is the load-bearing methodological fact of the whole investigation.

---

## 0. TL;DR VERDICT

> **~6× on a single RTX 5090 is NOT achievable with valid numerics. It was never physically on
> the table.** The proven, validated single-card ceiling is **~1.6× MEASURED** (the shipped-able
> fp32-physics lever), with a **PROJECTED path to ~1.8–2×** if the high-risk fp32-dycore
> (perturbation rewrite, ADR-031) is earned — *plus* the genuinely valuable **1 km-on-one-card
> capability** (the BouLac→O(nz) VRAM lever). **Anything beyond ~2–3× on this problem requires
> multi-GPU weak-scaling.**

| Question | Answer | Basis |
|---|---|---|
| ~6× single-card? | **NO** — exceeds the RTX 5090 roofline and requires numerically-invalid global fp32 | v0.16 fp32 make-or-break + roofline (§3, §5) |
| Best **measured** single-card lever? | **fp32-physics ~1.6×** warm @ realistic radiation cadence, +VRAM | `fp32_physics_VALIDATION.md` (§4) |
| Why can't kernel **fusion** close the gap? | The step is **device-WORK-bound**, not launch-bound — proven by collapsing launches −83% for **0%** wall change (R2) | `r2_operator_fusion_poc.md` (§2) |
| Path to ~1.8–2×? | **fp32-dycore** (memory-bandwidth lever) — PROJECTED, HIGH risk, needs the perturbation rewrite | `fp32_verdict/`, roofline (§5) |
| Path to **>3×**? | **multi-GPU** only | §6 |
| Real product value? | **1 km capability** (fits one card) + **cluster weak-scaling**, NOT single-card peak speed | §6, §7 |

**The one-line ceiling verdict:**
**Single-card valid ceiling ≈ 1.6× MEASURED (fp32-physics) → ~1.8–2× PROJECTED with fp32-dycore; ~6× is roofline-impossible; >3× needs multi-GPU.**

---

## 1. The starting point — the MEASURED fp64 baseline (the denominator)

Everything below is measured against the **shipped v0.16 all-fp64 operational step** on the
RTX 5090 (GB202). Source: `proofs/perf/v017/SPRINT0_baseline.md` (`.wt-perf-base`),
`fp64_baseline.json`, `real_dycore_fp64_penalty.json`.

| case | ncol | warm ms/step (MEAS) | peak VRAM | vs 28-rank CPU-WRF |
|---|---:|---:|---:|---|
| Canary d02 3km | 10,494 | **50.65** | 10.77 GiB | **2.73×** (MEAS/MEAS, paired CPU) |
| Workstation 128² | 16,384 | **72.63** | 11.38 GiB | **2.76×** (MEAS/MEAS, paired CPU) |
| Canary d01 9km | 5,487 | 39.50 | 9.34 GiB | 1.12× (CPU extrapolated) |
| Canary d03 1km | 6,975 | 38.64 | 11.07 GiB | 1.46× (CPU extrapolated) |
| 1km single | 147,456 | 569.64 | **26.08 GiB** | 2.09× (CPU extrapolated) |

Two facts from the baseline frame the entire question:

1. **The fp64 core step is *already* 2.7–2.8× faster than 28-rank CPU-WRF on the two
   paired-CPU grids** — this is the *core* step (boundary/GWD/NoahMP OFF; honesty flag carried
   from S0). So the "GPU is at parity" framing from v0.15 is the *full production* step; the
   lighter core step is already ahead. The ~6× question is whether that 2.7× can be pushed to ~6×.
2. **The MEASURED dycore/physics split** (`real_dycore_fp64_penalty.json`): the dynamical core is
   only **25–38%** of the step; **physics (radiation+PBL+MP+surface) is 62–75%** and *grows* at
   scale. This MEASURED split (it replaces an earlier numerically-invalid cost-proxy) is the
   decisive Amdahl fact: **the larger, lower-risk slice to attack is physics, not the dycore.**

---

## 2. THE WHY (the core finding): the step is DEVICE-WORK-BOUND, not launch-bound

This is the heart of the deliverable. The investigation went through a **correct, documented
reversal**, and the reversal is the answer to "why won't it go faster."

### 2.1 Phase R's first diagnosis: "launch-bound, 68% idle" (MEASURED, but misread)

Phase R (`PHASE_R_root_cause.md`, `.wt-phaseR`) profiled the warm fp64 step with nsys
(device-time, capture-range warm window) and found, on Canary d02 3km (MEASURED):

- **5,174 kernel launches/step** — a sea of sub-µs XLA `loop_*_fusion` kernels.
- device-**BUSY only 16.5 ms** of a **52.0 ms** step → **busy fraction 31.8%**, i.e. the GPU
  appeared **idle 68% of every step** in inter-kernel gaps (35.4 ms/step, mean gap 6.87 µs ≈ one
  CUDA launch latency).
- This held **across a 6.25× grid range** (busy fraction pinned 26–32%, d02 10k → 256² 65k cols),
  and `device-busy + gaps = 100% of host-wall` (no hidden time) — the textbook launch-bound
  signature.
- **Precision was eliminated as the bottleneck**: convert/cast ≈ 0%, transpose 0.03%, Thomas
  tridiagonal solve only 5.1%. The fp64 ridge is **1.19 FLOP/byte**, so the stencils are
  memory-bound even in fp64 — the 1/64 fp64 ALU throttle is *not* binding.

Phase R's projection from this: removing the launch gap → ~17 ms step → **~3× headroom from
structure alone**. **This projection was wrong, and the next two experiments proved it.**

### 2.2 The refutation: collapsing the launches bought 0% wall (MEASURED — the pivotal result)

The Thomas-only megakernel spike (`megakernel_spike.md`) fused the hottest vertical solve →
**1.006× full-step**. That alone was Amdahl-explainable (Thomas is 5.1% of busy). The decisive
experiment was **R2** (`r2_operator_fusion_poc.md`, `.wt-r2`): fuse the **entire `advance_w`
op-chain** — the largest tractable column-local operator — into one Pallas/Triton column
megakernel and measure the **full-step wall**.

The R2 result is the single most important number in the whole investigation
(`r2_fullstep.json`, `r2_launchcount.json`, all MEASURED, fp64, bit-exact):

| metric (per step) | XLA baseline | R2 megakernel | change |
|---|---:|---:|---|
| individual kernel launches | **5,281** | **0** (one CUDA-graph replay) | — |
| effective kernel count | **5,281** | **885** | **−83%** |
| D2D scan-carry copies | **2,639** | **0** | **−100%** |
| **full-step warm wall** (canary_d01_128) | **21.651 ms** | **21.530 ms** | **1.006×** |
| full-step geomean (4 grids) | — | — | **1.010×** (range 0.997–1.030×) |
| bit-exactness vs XLA | — | **max_abs 0.0, every field** | byte-identical |

**R2 did *exactly* what Phase R prescribed — it collapsed the launch sea 83% and eliminated every
D2D copy, bit-exact — and the wall did NOT move.** The fused kernel's own device-time was neutral
(geomean ~1.03×, range 0.97–1.14×): same FLOPs on same bytes in fp64.

### 2.3 The corrected mechanism: the "68% idle" is async-hidden, not recoverable

The reconciliation is clean and is the report's core explanation:

- Phase R measured device-busy **against host-wall**. Under nsys's serializing measurement the
  inter-kernel gaps appear as idle. **But under normal async execution** (no profiler), XLA and
  the CUDA driver **run the host ahead and queue kernels**, so that launch latency is
  **overlapped / hidden** behind device work. It was never on the wall-clock critical path.
- Therefore the 35 ms/step of "launch-gap idle" is a **profiler-attribution artifact**, not a
  recoverable cost. R2 removed the launches (proving capture works) and the wall was unchanged —
  the **only** explanation consistent with the data.
- This independently explains the **R4 refutation** (`r4_cudagraph.md`, `.wt-r4`): wrapping the
  same thunks in XLA command buffers / CUDA graphs was MEASURED **+5% to +14.5% SLOWER** (and
  bit-exact). Graphs change the launch *mechanism*, not the kernel *count/size*; they trade
  per-launch latency (which was already hidden) for graph-instantiate/boundary overhead → net
  slower. R4's own device-level nsys confirmed: command buffers captured (activities 5,174→3,633,
  −30%) yet the inter-kernel gap **rose** 35→50 ms (+42%) and stream-wall rose +5%.

> **Conclusion of §2 (the WHY):** the warm fp64 step is bound by the **device compute/memory
> WORK itself**, not by launch dispatch. Three independent structural-fusion experiments —
> Thomas-only (1.006×), full-operator R2 (1.010×), CUDA-graphs/R4 (0.87–1.00×) — **all measured
> ~1.0× or slower.** Fewer/bigger kernels cannot help. **The only lever that moves the wall is
> reducing the device WORK** — i.e. fewer bytes/FLOPs via **fp32** — and that is bounded by the
> roofline (§3) and by the dycore's cancellation-pins (§5).

See **`plots/fig1_device_busy_vs_gap.png`** (the busy-vs-gap breakdown + the flat R2 wall) and
**`plots/fig4_r2_launchcount.png`** (the −83% launch collapse with the unmoved wall).

---

## 3. The roofline — why fp32 is bounded to ~2×, and ~6× is impossible

Source: `PHASE_R_root_cause.md` §3 (RTX 5090 GB202 peaks). See **`plots/fig3_roofline.png`**.

| peak (RTX 5090 GB202) | value |
|---|---|
| fp32 compute | 136.4 TFLOP/s |
| fp64 compute (1/64) | 2.13 TFLOP/s |
| HBM bandwidth | 1,792 GB/s |
| **fp64 ridge point** | **1.19 FLOP/byte** |
| fp32 ridge point | 76.1 FLOP/byte |

The dycore stencils have arithmetic intensity ~1–3 FLOP/byte — **at or above the fp64 ridge**,
so they are **memory-bound, not fp64-ALU-bound**. The consequences are exact:

- **fp32 halves the bytes moved → at most ~2× on a memory-bound stencil.** This is the hard
  physical bound on the dycore lever. It is *not* a 4× or 6× lever, because the kernels are
  bandwidth-limited, not ALU-limited (so the 64× fp32-vs-fp64 ALU ratio is irrelevant here).
- **~6× is above the roofline** for this workload at these grid sizes, full stop. The historical
  "4.3× cost-proxy" was a **numerically-invalid global-fp32 artifact** (x64 off, conservation/
  cancellation corrupted — `fp32_verdict/`), and "6×" always exceeded the achievable roofline.

---

## 4. The MEASURED win — fp32-physics ~1.6× (validated)

This is the one lever that **moves the wall, is measured, and is shippable**.
Sources: `fp32_physics_bench.md` (`.wt-perf-physics`, frontrunner) + the adversarial
`fp32_physics_VALIDATION.md` (`.wt-fp32phys-val`, independent re-measure). **VERDICT: ACCEPT.**

| radiation cadence | fp64 ms/step | fp32-phys ms/step | **speedup (MEAS)** | peak VRAM fp64→fp32 |
|---|---:|---:|---:|---:|
| radt=5 min (cadence 30) | 82.84 | 51.14 | **1.62×** | 7.39 → 4.69 GiB |
| **radt=10 min (this case's namelist)** | 77.61 | 48.56 | **1.598×** | 6.12 → 3.74 GiB |
| radt=0 (cadence 1, upper bound, *not* operational) | 386.08 | 201.63 | 1.915× | 12.77 → 7.92 GiB |

- **Honest operational headline: ≈ 1.6× warm, −≈39% physics-scratch VRAM** (WS-128², force_fp64
  dynamics, boundary/GWD/NoahMP off — the physics-isolated A/B).
- **The 1.915× GPT headline was ~0.3× cadence-inflated** (it ran radiation every step); the
  validation corrected it to the realistic ~1.6×.
- **Crucially, the win is MYNN-PBL-driven, not radiation-driven** — halving the radiation cadence
  barely moved the speedup (1.598→1.62×), and MYNN runs *every* step. So the ~1.6× does **not**
  collapse when radiation amortizes. This is why it is robust.
- **Validated, not greenwashed:** all 3 oracles (RRTMG SW/LW, MYNN) PASS in fp32 within
  *unmoved* fixture tolerances (the fp32 increment is small and dominated by the pre-existing
  JAX-vs-WRF port gap); flag-OFF is byte-identical to the v0.16 program; a 120-step coupled run
  stays finite and tracks fp64 to ≲1e-4 relative on every dynamical field. Implementation is the
  compact fp64-island shape (cast to fp32 at the scheme seam, couple back the small increment).

---

## 5. The remaining candidate — fp32-dycore (~1.8–2× PROJECTED, HIGH risk)

This is the lever that *could* take the single card from ~1.6× toward ~1.8–2×, but it is
**not earned** and is the hard/risky one.

- **It is a memory-bandwidth lever, not a speed-of-light lever.** The dycore is memory-bound
  (§3), so fp32 halves its bytes → **at most ~2×** on the dycore's 25–38% slice. By Amdahl
  (MEASURED fractions) even a *perfect* fp32 dycore on top of fp32-physics lands the full step
  around **~1.8–2×**, not higher.
- **It is numerically pinned.** The v0.16 fp32 make-or-break (`fp32_verdict/README.md`,
  double-confirmed Opus+GPT) proved fp32-dycore-compute is **cancellation-pinned**: the base
  absolutes `p_total`/`ph_total` (~1e5) **cannot be stored fp32** without corrupting the
  geopotential/PGF gradients **27× / 127×** beyond the gated-fp32 budget (bits are lost at
  *storage*, so an in-loop fp64 island cannot rescue it). qke goes non-finite in fp32 at 1 km.
  **Valid-numerics naive fp32 ≈ 1.1×.** To get the real win it needs the **ADR-031 perturbation
  rewrite** (store perturbations, keep the large base-state fp64) — a multi-sprint, high-chaos-
  risk effort.
- **Its real value is VRAM, not small-grid speed.** The dycore is ~87% of the 1 km transient
  peak; fp32 would roughly halve the 17.85 GiB transient. But the **orthogonal BouLac→O(nz)
  lever already unlocks 1 km** (next section) without touching precision, so fp32-dycore is the
  *speed* candidate, gated behind the perturbation rewrite.

---

## 6. The 1 km capability lever + the only >3× path

**BouLac → O(nz) (the 1 km-on-one-card unlock).** Source: `boulac_onz_VALIDATION.md`
(`.wt-boulac-val`). **VERDICT: ACCEPT.** MEASURED: the dense MYNN BouLac path **OOMs at an
18.80 GiB single allocation** at 147k cols; the O(nz)/chunked path **FITS at 18.58 GiB**,
**bit-identical** to dense (`max_abs = 0.0`, mathematically guaranteed — it slices only the
source axis), default-OFF byte-identical to v0.16. It is **not a speed lever** (~+2.5% warm on
small grids) — its value is that **a 1 km Canary now fits one RTX 5090**.

**Multi-GPU weak-scaling — the only >3× path.** Every single-card lever is bounded by §2–§5 to
~1.8–2×. The genuine large-factor speedup is **N cards ≈ N× on the 1 km grid** (the grid large
enough that the GPU finally fills — §2.1 showed the busy fraction only flips at ≳147k cols, which
is exactly the 1 km case). This is **PROJECTED** (no multi-GPU benchmark in this evidence set) and
is the strategic direction for real value.

---

## 7. The lever-by-lever table (the decision matrix)

See **`plots/fig2_lever_bars.png`**. Every row labeled MEASURED or PROJECTED.

| Lever | Magnitude | Status | Mechanism / why |
|---|---:|---|---|
| **fp32-physics** (Fork C) | **~1.6×** | **MEASURED — DONE** | MYNN-driven, halves physics bytes/compute; +39% physics VRAM; oracle-validated. The shippable single-card win. |
| **BouLac → O(nz)** | VRAM (1 km) | **MEASURED — DONE** | Bit-identical; not a speed lever; **unlocks 1 km on one card** (dense OOMs at 18.8 GiB). |
| Thomas-only megakernel | **1.006×** | **MEASURED — DEAD** | Thomas is 5.1% of device-busy → Amdahl-capped. |
| **R2 column megakernel** | **1.010×** | **MEASURED — DEAD** | Collapsed launches −83%, D2D −100%, **bit-exact**, yet wall unchanged → step is device-WORK-bound, not launch-bound. The pivotal refutation. |
| R4 CUDA-graphs / command-buffer | **0.87–1.00×** | **MEASURED — DEAD** | Captured (−30% activities) but graph-mgmt overhead > launch saving → measured SLOWER. |
| **fp32-dycore** (Fork A) | **~1.8–2×** | **PROJECTED — HARD/RISKY** | Memory-bandwidth (fp32 halves bytes); cancellation-pinned → needs ADR-031 perturbation rewrite; HIGH chaos risk. The remaining candidate. |
| **multi-GPU weak-scaling** | **>3× (N×)** | **PROJECTED** | The ONLY path beyond ~2–3×; N cards on the 1 km grid where the GPU finally fills. |

**Reading the table:** the three structural-fusion levers (Thomas/R2/R4) are **measured dead** at
~1.0× — this is the empirical proof that the launch-bound hypothesis is false and that fusion
cannot help. The only single-card lever that moved the wall is **fp32-physics (~1.6×, measured)**;
the only candidate beyond it is **fp32-dycore (~1.8–2×, projected, gated on a hard rewrite)**;
beyond ~2–3× is **multi-GPU only**.

---

## 8. Measured-vs-projected ledger (full honesty)

| Quantity | Status |
|---|---|
| fp64 baseline ms/step, VRAM, dycore/physics split | **MEASURED** (S0, `fp64_baseline.json`, `real_dycore_fp64_penalty.json`) |
| 2.73× / 2.76× core-step vs paired CPU-WRF | **MEASURED / MEASURED** |
| Phase-R per-step launches / device-busy / gaps (5,174 / 16.5 / 35.4 ms) | **MEASURED** (nsys device-time) |
| R2: launches 5,281→0/885, D2D 2,639→0, full-step 1.010×, bit-exact | **MEASURED** |
| R4: command buffers 0.87–1.00× (slower), bit-exact | **MEASURED** |
| Thomas-only megakernel 1.006× full-step | **MEASURED** |
| fp32-physics ~1.6× warm + −39% physics VRAM + oracle PASS | **MEASURED** (independently validated) |
| BouLac→O(nz): dense OOM @18.8 GiB, O(nz) fits @18.58 GiB, bit-identical | **MEASURED** |
| fp32 make-or-break: valid-numerics ≈1.1×, 0% VRAM-peak, base-absolute pin 27×/127× | **MEASURED** (double-confirmed Opus+GPT) |
| RTX 5090 roofline peaks + ridge points | **MEASURED** (nvidia-smi clocks + GDDR7 BW) |
| fp32-dycore ~1.8–2× | **PROJECTED** — bounded by roofline + Amdahl; needs ADR-031 |
| multi-GPU >3× (N×) | **PROJECTED** — no multi-GPU benchmark in this evidence set |

**Honesty flags carried forward:** (1) the 2.7× baseline and the ~1.6× fp32 figure are on the
**core step** (boundary/GWD/NoahMP OFF); the full production step is heavier, so the end-to-end
production number is lower. (2) fp32-physics is measured on **WS-128² only** (Canary fixtures lack
a `namelist.input`); production fractions shift with NoahMP/boundary ON. (3) All speedups use the
shape-driven cost-proxy (the project's standing assumption — cost depends on array shapes).

---

## 9. FINAL VERDICT

**Can a single RTX 5090 beat 28-rank CPU-WRF by ~6× on 1 km Canary? — NO.**

- **~6× is roofline-impossible** with valid numerics; it was an artifact of numerically-invalid
  global fp32 (`fp32_verdict/`) and exceeds the achievable RTX 5090 roofline (§3, §5).
- **The proven, validated single-card ceiling is ~1.6× MEASURED** (fp32-physics, MYNN-driven,
  oracle-validated, +39% physics VRAM) (§4).
- **A path to ~1.8–2× is PROJECTED** via fp32-dycore (a memory-bandwidth lever bounded by the
  memory-bound roofline + Amdahl), but it is **HIGH risk** and requires the ADR-031 perturbation
  rewrite to be numerically valid (§5).
- **The WHY behind the ceiling:** the step is **device-WORK-bound, not launch-bound** — proven by
  R2 collapsing the launches −83% (and D2D −100%, bit-exact) for **0% wall change**, corroborated
  by the Thomas-only (1.006×) and R4/CUDA-graph (0.87–1.00×) measured-dead results. The "68%
  idle" Phase R first measured is **async-hidden**, not recoverable. So structural fusion cannot
  help; only reducing device WORK (fp32) can, and that is roofline-bounded to ~2× (§2, §3).
- **Beyond ~2–3× requires multi-GPU weak-scaling** (PROJECTED) (§6).
- **The real product value is NOT single-card peak speed** — it is the **1 km-on-one-card
  capability** (BouLac→O(nz), MEASURED) + **multi-GPU cluster weak-scaling** (PROJECTED). On the
  *core* step the single card is already 2.7× ahead of 28-rank CPU; the levers add ~1.6× on top of
  that for physics-heavy steps, plus the capability unlock.

**One-line ceiling verdict:**
**Single-card valid ceiling ≈ 1.6× MEASURED (fp32-physics) → ~1.8–2× PROJECTED with fp32-dycore; ~6× is roofline-impossible; >3× needs multi-GPU. The step is device-WORK-bound (R2: −83% launches, 0% wall), so only precision (fp32), not fusion, moves it — and fp32 is roofline-capped at ~2×.**

**[R2 cross-check double-confirmation: ✅ CONFIRMED — stamped 2026-06-14 by manager]**
The GPT adversarial cross-check (`worker/perf/v017-r2-xcheck @ffad9c98`, `r2_xcheck.md` + `r2_xcheck_fullstep_clean.json`) **CONFIRMS the device-work-bound reversal — and strengthens it:**
- **Decisive new proof (the discriminator):** a **no-op kernel flood at the same launch count is far too fast to explain the real step wall** → the step is provably **device-WORK-bound, NOT launch-bound** (if it were launch-bound, the no-op flood would be slow too). This is the cleanest available proof.
- The cross-check independently reproduced the **launch collapse (−83%)** + **D2D elimination** + graph-capturability.
- **Honest divergence on the exact PoC numbers (conclusion unchanged, strengthened):** the cross-check's fresh clean full-step runs do NOT reproduce the PoC's `1.006× + bit-exact`; instead it measured the fused step **SLOWER — geomean 0.834× (range 0.763–0.920×)** with slight fp64 drift (`max_abs ~2.6e-11`, not bit-exact), across all grids, cache-on/off, and COLBLK tuning (0.61–0.92×; no config won). So structural fusion is not merely *neutral* — it *hurts*. The strategic verdict (do NOT scale fusion; device-work-bound; ~6× not achievable by fusion) is **double-confirmed and stronger.**
- *Note:* the deeper hardware "WHY the work is so much" (fp64-throttle vs bandwidth-underfill vs algorithm-recurrence) is the separate in-flight investigation `worker/perf/v017-deep-why` (ncu roofline + CPU decomposition + reformulation question) — that report extends this conclusion at the hardware level.

---

## 10. Plots

All under `proofs/perf/v017/plots/` (matplotlib, CPU-rendered from the committed MEASURED JSONs;
regenerate with `python3 proofs/perf/v017/make_conclusion_plots.py`):

| file | shows |
|---|---|
| `fig1_device_busy_vs_gap.png` | The device-busy-vs-launch-gap breakdown (busy 18–32% flat across a 6.25× grid range) **+** the R2 reversal (launches −83%, D2D −100%, wall unchanged). The core "WHY". |
| `fig2_lever_bars.png` | The lever bar chart — measured wins (fp32-physics ~1.6×), the dead structural levers pinned at ~1.0×, and the projected fp32-dycore (~1.8–2×) / multi-GPU (>3×) paths. |
| `fig3_roofline.png` | RTX 5090 roofline with the dycore stencils placed at/above the fp64 ridge → memory-bound → fp32 ≤ ~2×, ~6× above the roof. |
| `fig4_r2_launchcount.png` | Launch-count −83% (and D2D −100%) under R2 with the **flat full-step wall** across 4 grids — fusing the launches away buys no wall time. |

## 11. Source proof files (all committed, in sibling worktrees)

- `proofs/perf/v017/SPRINT0_baseline.md` + `fp64_baseline.json`, `real_dycore_fp64_penalty.json` (`.wt-perf-base`)
- `proofs/perf/v017/PHASE_R_root_cause.md` + `nsys/phaseR_*_attribution.json` (`.wt-phaseR`)
- `proofs/perf/v017/r2_operator_fusion_poc.md` + `r2_fullstep.json`, `r2_launchcount.json`, `r2_advancew_device_time.json` (`.wt-r2`)
- `proofs/perf/v017/r4_cudagraph.md` (`.wt-r4`)
- `proofs/perf/v017/megakernel_spike.md` + `megakernel_spike.json`, `megakernel_devtime.json` (`.wt-perf-megakernel`)
- `proofs/perf/v017/fp32_physics_bench.md` (`.wt-perf-physics`) + `fp32_physics_VALIDATION.md`, `fp32_physics_validation.json` (`.wt-fp32phys-val`)
- `proofs/perf/v017/boulac_onz_VALIDATION.md` (`.wt-boulac-val`)
- `proofs/v016/fp32_verdict/README.md` + `opus-fullws-fp32-verdict.md`, `gpt-fullws-fp32-crosscheck.md`

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
