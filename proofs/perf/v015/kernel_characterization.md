# v0.15 Kernel-Performance Characterization (the "final kernel" evidence pack)

**Author:** opus-v015-kernel-review (2026-06-13). **Branch:** `worker/opus/v015-kernel-review`.
**Purpose:** independent adversarial completeness review of the v0.15 kernel work + the
publication artifacts (plots/tables) that justify the v0.15 perf headlines for the README.
Machine-readable: `proofs/perf/v015/kernel_characterization.json`. Plots: `docs/assets/v015/kernel/`.

Every number traces to a measured artifact under `proofs/perf/v015/`. Identity-relevant
deltas are scored against the FROZEN v0.14 manifest
`proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`.

Case (128² anchor): Switzerland d01 reinit-h36, 128×128×44, dt=18 s, ns=10, `force_fp64`,
RTX 5090 (GB202). CPU denominator: 24-rank gfortran dmpar = **200.5 ms/step**
(`run_h36/cpu_timing.json`, 40.11 s/fc-hr).

---

## 0. Headline numbers (all measured)

| state | 128² ms/step | vs v0.14 | vs 24-rank CPU | source |
|---|---|---|---|---|
| v0.14 baseline | 173.9 | 1.00× | **1.15×** | `ab_s1_base` / `s1_bisect_walls.json` |
| **niter=16 (SHIPPED DEFAULT)** | **119.8** | **1.45×** | **1.67×** | `s1_bisect_walls.json` row D; `niter16_revalidation.json` 1.43× clean re-check |
| + fp32-BouLac (OPT-IN) | 104.0–108.2 | 1.61–1.67× | **1.85–1.93×** | `s1_bisect_walls.json`; `ab_s1_cond16_fp32boulac.json` |
| 1 km deploy 560×280 (now) | — | — | **1.6–2.59×** | `km_feasibility_verdict.json` |
| 1 km deploy 560×280 (+S2 Pallas) | — | — | 2.08–4.14× | `km_feasibility_verdict.json` |
| asymptotic large-grid full-step floor | — | — | **1.63×** | `grid_scaling.json` fit vs CPU-28 marginal |

---

## 1. Per-phase steady-step breakdown @128² (`phase_breakdown_128.png`)

From `nvtx_phase_attribution.json` (nsys NVTX projection, 150 steps, niter=50 baseline):

| phase | device ms/step | share of device-busy |
|---|---|---|
| MYNN PBL (EDMF) | 88.6 | 87.2 % |
| dycore inner scans (acoustic Thomas, Thompson sed) | 9.9 | 9.8 % |
| Thompson microphysics | 2.0 | 2.0 % |
| step body / other | 1.0 | 1.0 % |
| **total device-busy** | **101.6** | (full wall 177.9) |

The shipped **niter=16** cut collapses the MYNN-EDMF condensation loop
(88.6 → ~2.7 ms device, `nvtx_phase_attribution_post.json`), moving the steady **wall
173.9 → 119.8 ms/step**. After the cut, the **dycore inner scans (~10 ms) become the
largest device family** — but they are only ~8 % of the post-cut wall, so there is no
single dominant remaining device phase to attack short of structural (Pallas) work.

## 2. Scaling with grid size (`grid_scaling.png`)

From `km_bench/grid_scaling.json` (spatial-tile COST proxies; core = dycore+rad+MYNN with
boundary/GWD/NoahMP OFF uniformly so the law is apples-to-apples):

| ncol | ms/step | per-cell µs/col | peak VRAM (GiB) | speedup vs CPU-28 |
|---|---|---|---|---|
| 10 494 | 62.1 | 5.92 | 3.96 | 1.36× |
| 20 988 | 146.3 | 6.97 | 3.96 | 1.16× |
| 41 976 | 232.1 | 5.53 | 5.50 | 1.46× |
| 62 964 | 318.4 | 5.06 | 9.15 | 1.59× |
| 94 446 | 473.3 | 5.01 | 14.10 | 1.61× |
| 125 928 | 631.4 | 5.01 | 19.84 | 1.61× |
| 167 904 | 863.9 | 5.15 | 26.25 | 1.57× |

(The 20 988 dip is the only outlier point — a known per-cell anomaly at that single tile
geometry; the trend from 42k cols up is steady ~1.5–1.6×, asymptote 1.63×.)

**Fit:** `ms/step = 19.6 + 4.94·(ncol/1000)`, R²=0.997. Peak VRAM linear at
0.165 GiB/1000 col. The largest grid that fit ~30 GiB = 167 904 cols @26.25 GiB;
209 880 cols OOMs on a single 28.44 GiB op.

**The "6–10× per-cell at large grids" claim is REFUTED by this measured data.** Per-cell
cost *slightly worsens* at the top (5.0 → 5.15 µs/col), throughput plateaus at ~195–200
cols/ms. The mild overall sublinearity (×16 cols → ×13.9 ms) is *entirely* the fixed
~20 ms per-step intercept amortizing, NOT genuine GPU saturation headroom. The 87 → 730
GB/s figure cited in prior docs is a TRIVIAL 1R1W copy-kernel microbench
(`micro_kernels.json`); it does not represent the coupled step.

## 3. Init/compile overhead vs run length (`compile_amortization.png`)

`baseline_3h.json` + probe §4: cold compile **~448 s/graph × 2 graphs** (fp32-mixed
hour-1 + all-fp64 hours-2+); warm persistent cache **~32 s/graph deserialize**. Steady
throughput (niter=16) 119.8 ms/step vs CPU 200.5 ms/step → steady asymptote **1.67×**.
Warm-cache effective speedup: 0.84× @1 h → 1.36× @24 h → 1.54× @72 h → asymptote 1.67×.
Cold-compile effective speedup stays <1× until ~24 h; the **fp64 operational-state ADR is
the structural remover** of the double-compile.

## 4. Asymptotic large-grid speedup limit (`asymptotic_largegrid.png`)

Marginal per-column cost (identical units, µs/col/step): **GPU 4.94** vs **CPU-28 8.06**
→ **asymptotic full-step speedup floor = 1.63×**. This is the honest large-grid asymptote
for the *coupled* step, and it is fully consistent with the km-feasibility deployment band
(1.6–2.7×, centered ~2×, the extra coming from intercept amortization). It is NOT the
6–10× a bandwidth roofline would suggest, because the full step never reaches the
copy-kernel BW ladder.

**fp64 hardware truth:** the 5090 (GeForce) runs fp64 at 1/64 fp32 = ~1.7 TFLOP/s ≈
**0.77× the CPU's AVX-512 ALU peak**. The GPU's only edges are DRAM BW (20×) and
parallelism, and the latency-bound coupled step exposes them only partially even at large
grids — which is exactly why the asymptote is ~1.6× and not ~20×.

## 5. Profile-bound resolution (a CORRECTION to the prior record)

Two prior artifacts disagree and **the disagreement is material**:

- `perf_fix_findings.json` + the kernel-probe: "**LAUNCH-BOUND**, ~159 ms launch API, GPU
  compute busy only ~13–16 ms/step, GPU ~90 % idle, 16 ms device floor."
- `s1_bisect_walls.json` + s1-host-removal: "**DEVICE-BOUND** — clean nsys (cache-hit
  warmups) shows 4.2 % GPU idle, device busy 168.5 ms/step; the 16 ms floor was a CUPTI
  dropped-events undercount; whole-step CUDA-graph capture collapsed launches 13 958 → 196
  but was **WALL-NEUTRAL** in 4 A/B pairs."

**The later, cleaner s1-host-removal measurement is authoritative: the step is
DEVICE-BOUND.** The km_bench scaling independently confirms it (a launch-bound,
GPU-idle step would show ~constant ms/step with ncol and rising throughput; neither is
observed — ms/step is linear and throughput plateaus). Consequence: **all host-side levers
(graph capture, denominator hoist, scan-unrolls) are exhausted or wall-neutral**, and the
only remaining LARGE levers are device-structural (Pallas, BouLac O(nz²)→O(nz), fp32).

> **NOTE for the FINDINGS-FINAL doc:** §1 correctly adopts DEVICE-BOUND, but §6's ceiling
> table and the probe's §2c/§5 still quote the refuted "15.4–16.3 ms device floor." That
> floor is a CUPTI artifact; the real cond16 device-busy floor is **~112–116 ms/step**
> (`s1_bisect` row D nsys). The "12–22× / 25–45× at 128²" ceiling rows depend on that
> refuted floor and should be read as Pallas+fp32 *aspirations*, not measured device floors.

## 6. Identity regime

All speedup states above are gated against the FROZEN v0.14 manifest. niter=16:
worst rmse/limit = **6.4e-4** (`niter16_revalidation.json`, PASS, hard_gate_fails []).
fp32-BouLac combined: worst rmse/limit = 6.3e-4 (`tiered_gate_combined.json`, PASS).
Tier-P (33/56 leaf-hash differ) by construction — fusion-boundary changes cannot be
strict-bitwise; this is XLA FMA contraction, not numerical sloppiness.

---

# PART A — Ranked completeness ledger (adversarial + creative)

Status legend: **SHIPPED** | **CLOSED** (closed-with-evidence) | **DEFERRED** |
**NOT-CHECKED**. "Payoff" is at 128² unless a large-grid figure is given. "Sound?" is my
independent judgment of whether the cited evidence holds.

| # | angle | status | payoff @128² | payoff @large/deploy | risk | sound? |
|---|---|---|---|---|---|---|
| 1 | **niter cut 50→16** (WRF's own convergence bound) | **SHIPPED default** | **1.45× / 1.67× CPU** | same ratio (MYNN ~linear) | low (tiered gate PASS, oracle) | YES — re-validated clean 1.43×, gate 6.4e-4 |
| 2 | **fp32-BouLac** (+cond16) | **DEFERRED / opt-in** | +1.10–1.15× → 1.85–1.93× CPU | same | med (compile pathology) | YES the win (104–108 ms measured); the *blocker* (>4 min XLA "very slow compile" in full pipeline) is the real open item |
| 3 | **Pallas column megakernels** | **DEFERRED / EXCLUDED** (principal) | 2.5–3.5× (est, unbuilt) | the only 512²/H200 lever; 1km +S2 2.08–4.14× | high | PARTLY — the *direction* is sound; the *magnitude* is an estimate, NOT measured |
| 4 | **fp32 operational state / mixed-perturb-fp32 acoustic** | **DEFERRED S3** | ÷1.8–2.4 dycore / ÷9 pow (of remainder) | same ratios | high | YES as a sequenced lever; needs S2 first (latency-bound); dycore is only ~10 ms so absolute win is modest |
| 5 | **Whole-step CUDA-graph capture** | **CLOSED** wall-neutral | 0 (launches 13958→196, wall unchanged) | smaller share | — | YES, decisively — 4 A/B pairs; step is DEVICE-bound |
| 6 | **Denominator / stage-constant hoist** | **CLOSED** anti-opt | −5.5 % (LOSS) | — | — | YES — inline FMA recompute fuses free |
| 7 | **Scan-unrolls (THOMAS/SED/ACOUSTIC) in-program** | **CLOSED** ~0 | ~0 in-program (3.3× solo on a tiny op) | ~0 | — | YES — Thomas solve is ~0.5 ms total |
| 8 | **Thomas reverse-scan** | **CLOSED** | wall-neutral solo | — | — | YES |
| 9 | **Command-buffer global flag** | **CLOSED** off | −15..−21 % coupled | — | — | YES |
| 10 | **Implicit sedimentation** | **REJECTED** fidelity | — | — | — | YES (correct to reject) |
| 11 | **BouLac dense search O(nz²)→O(nz)** | **DEFERRED** clean win | −7–14.5 ms (the 23M fp64 reduce ×2/step) | larger absolute at scale | med | YES and UNDER-RATED — see "named lever" below |
| 12 | **Large-grid amortization** (deploy choice) | **SHIPPED** (MP tiling enables it) | n/a | **1.5–2.7× (centered ~2×), asymptote 1.63×** | low | the **6–10× per-cell claim is REFUTED** by km_bench; honest number is ~2× |
| — | **CREATIVE EXTENSIONS below** | | | | | |
| 13 | **Alt GPU tridiag solver (PCR/cuSPARSE)** | **CLOSED — already shipped where it matters** | n/a | n/a | — | cuSPARSE `pcrGtsvBatch` ALREADY runs the MYNN/BouLac solve (9/step, ~2.9 ms, `solve_tridiagonal_xla`); acoustic uses scan-Thomas (ADR-023) but that op is ~0.5 ms so a swap saves <0.5 ms. Not a large lever. |
| 14 | **Multi-stream / async overlap** | **CLOSED by device-bound** | ~0 | ~0 | — | step is 96 % GPU-busy; nothing to overlap host-side. Independent-physics overlap needs Pallas-level restructure. |
| 15 | **Persistent kernels** | folds into **#3 Pallas** | — | — | high | not separable from Pallas |
| 16 | **Buffer donation / cut the ~3.3k D2D/step** | **DEFERRED — modest** | D2D = 23.8 ms/step API but it overlaps device work (device-bound); freeing it ≈ small | small | med | the 3.3k D2D are scan-carry slice copies; only Pallas removes them structurally |
| 17 | **Cross-substep fusion** | folds into **#3/#5** | — | — | high | XLA already fuses within the scan body; cross-substep = capture (closed) or Pallas |
| 18 | **Eliminate 23M fp64 reduce (BouLac)** | = **#11** | −7–14.5 ms | — | med | this is the single largest NAMED non-Pallas device item left |
| 19 | **XLA flags not yet A/B'd** | **CLOSED for capture set**; mostly explored | ~0 | ~0 | low | command-buffer/update-mode/profiling flags A/B'd; autotune/allocator tried in fp32-BouLac probe |
| 20 | **Layout / `wrapped_transpose` elimination** | **NOT-CHECKED quantitatively** | small (transposes 4.5–5 µs each, few/step) | small | low-med | the only genuinely under-measured item; bounded SMALL by kernel counts (not a large lever) |
| 21 | **Batch nests/ensemble into one program** | **DEFERRED — throughput, not latency** | n/a | amortizes intercept (helps small grids) | med | sound as a throughput play; doesn't change per-step law |
| 22 | **Recompute-vs-store** | **CLOSED** (= #6) | recompute wins | — | — | proven by the hoist anti-opt |
| 23 | **Reduced precision beyond BouLac** | folds into **#4** | — | — | high | gated by S3 ADR |
| 24 | **Can the device floor drop further w/o Pallas?** | **CLOSED — NO** | the cond16 device floor is ~112–116 ms; non-Pallas levers (#11 + small ones) take it to ~100–105 | — | — | the honest answer: NO large non-Pallas device win remains |

## Independent soundness audit of the prior record (what I CHALLENGE)

1. **The "15.4–16.3 ms device floor" (probe §2c/§5, FINDINGS-FINAL §6) is REFUTED** by the
   s1-host-removal clean nsys + the s1_bisect row-D device-busy of ~112–116 ms/step. The
   probe itself flagged its CUPTI drops. FINDINGS-FINAL §1 adopts DEVICE-BOUND but §6's
   ceiling rows ("12–22×, 25–45× at 128²") still ride the refuted floor → those rows are
   **Pallas+fp32 aspirations, not measured headroom.** Manager should annotate FINDINGS-FINAL §6.
2. **The "512²-class per-cell 6–10× better" / "much-better-speed lever" headline
   (FINDINGS-FINAL §3 row + §2) is REFUTED** by the project's own `km_bench`:
   `km_feasibility_verdict.json` states "saturation_hypothesis REFUTED … per-cell cost
   slightly WORSENS … NOT the 6–10× the saturation hypothesis predicted … ~1.5–2.7×
   (centered ~2×)." The 6–10× came from a trivial copy-kernel microbench, not the coupled
   step. **This is the single most important honesty correction for the README:** do NOT
   headline 6–10× at large grids; the honest deployment number is **~2× vs 28-rank CPU
   (1.6–2.7× band), ~2–4× with S2 Pallas.**
3. Everything else in the prior record holds up against the artifacts: niter cut, fp32-BouLac
   win+blocker, capture wall-neutrality, the hoist/unroll/reverse-scan closures, the fp64
   hardware truth, and the tiered identity gates all reproduce.

## NAMED missed low-risk lever worth implementing in v0.15

**BouLac dense search O(nz²)→O(nz)** (ledger #11/#18). The MYNN BouLac mixing-length
builds dense (B, nz, nz) matrices and does a **23M-element fp64 reduce ×2/step ≈ 14.5 ms
fp64 (~7 ms fp32)** — that is **~12 % of the 119.8 ms post-niter wall**, the largest named
non-Pallas device item left. WRF's own algorithm is incremental O(nz). This is:
device-side, algorithmic (no precision change), tiered-gateable against the frozen manifest,
and independent of the Pallas exclusion. **Expected payoff ~1.1× on its own (≈109 ms/step),
stacking toward the fp32-BouLac number without the compile pathology.** It is the one
LARGE-ish, reasonable-risk lever that is DEFERRED rather than closed. Recommend the manager
scope it as a small v0.15 device sprint (or explicitly accept it as a documented carry).

The fp32-BouLac default (ledger #2) is the *other* unrealized standard-grid win, but its
blocker is a compile pathology, not a missing implementation — it is correctly opt-in until
the fp64-operational-state ADR removes the double-compile.

## VERDICT

- **Is the kernel near-optimal given JAX/XLA + the megakernel exclusion?** YES, with one
  named exception. The step is DEVICE-BOUND (4.2 % idle); every host-side lever is shipped
  or proven wall-neutral; the alternative-tridiag/multi-stream/overlap/flag angles are
  closed or already shipped (cuSPARSE PCR is live for the PBL solve). The remaining
  non-Pallas device headroom is bounded SMALL — except **BouLac O(nz²)→O(nz)** (~1.1×) and
  the opt-in fp32-BouLac (~1.1×), which together would take 119.8 → ~100 ms (≈2.0× CPU).
- **Is Pallas the only remaining LARGE lever, and its deployment payoff?** YES — Pallas
  column megakernels are the only path to >2.5× at 128² and the only 512²/H200 lever. But
  its DEPLOYMENT-grid payoff is **modest and measured-bounded: ~2.08–4.14× vs 28-rank CPU
  at the 1 km 560×280 target** (vs ~1.6–2.59× without it), NOT 6–10×. So excluding Pallas
  from v0.15 does NOT leave a 6–10× win on the table — it leaves roughly a further ~1.3–1.6×
  on top of the ~2× the current kernel already delivers at deployment scale.
- **v0.15 kernel can/cannot be honestly called final:** **CAN** — provided the README uses
  the honest numbers (1.45×/1.67× @128² default; ~1.6–2.7× centered ~2× at 1 km; NOT 6–10×),
  documents Pallas+fp32 as the named deferred >2.5× path, and the manager either lands or
  explicitly carries the one named low-risk device lever (BouLac O(nz²)→O(nz), ~1.1×). The
  kernel is near-optimal within JAX/XLA without megakernels; the remaining large wins are
  exactly the two structurally-excluded items (Pallas, fp32-operational), which are
  legitimately a *future-version* scope, not a v0.15 gap.
