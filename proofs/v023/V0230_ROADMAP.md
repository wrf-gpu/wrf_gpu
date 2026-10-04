# v0.23.0 ROADMAP — synthesized (GLM audit + GPT opinion + v0.22.x state + carry-over)

**Built 2026-06-28 by manager 0:1.** Sources: `Performance_review_GLM_0.22.md` (GLM kernel audit),
`proofs/v023/GPT_PERF_OPINION.md` (GPT second opinion), the v0.20–v0.22 release line, the v0.22.2
nested-perf work, and the carry-over backlog.

## Reconciliation (load-bearing)
GLM + GPT read a STALE checkout (`49cced41`/v0.20.2). On the real v0.22 line: **GLM 4.1 FUSED+AOT
warm-start is SHIPPED**; **the nested output-boundary work (GLM/GPT's other top picks) = v0.22.2
(Phase 1 host-work cuts default-on + async pipeline/radiation-from-carry/M9-radcap opt-in)**. So the
two highest GLM/GPT levers are already done. Kernel levers (dycore unchanged v0.20→v0.22) still valid
but need a 1-pass re-verify vs v0.22.1 src before execution.

## Honest framing (GLM, settled, do not re-litigate)
The **fp64 dycore is near-optimal at fp64** (1.45×/1.67× CPU @128²); the tiny-nest is launch/latency-
bound by hardware law (fp64 = 1/64 fp32 on the GeForce 5090), NOT an architecture failure. So there is
**no big dycore-kernel quick win**. Remaining wins = compile/cache + small bit-identical launch-count
cuts + the fp32 structural milestone (separate major) + **manufacturing scale from independent mini-grid
cases (the new lead feature F1 below)**.

---
## ⭐ LEAD FEATURE — F1: Batched-Ensemble Mini-Grid GPU Saturation (the user 2026-06-30 — FIRST item of v0.23.0)

**Added to the FRONT of the v0.23.0 roadmap by the user 2026-06-30 — the new first feature item. NOT started
(the user: "noch nicht starten").** Source = the convergent GPT-5.5 + Opus-4.8 analysis
`proofs/v023/gpu_minigrid_util/SYNTHESIS_AND_RECOMMENDATION.md` (both models *independently* ranked it #1).

**What it brings:** one 5090 is badly underutilized by a mini-grid (Tenerife 2-nest 3 km/1 km — launch/occupancy-
bound, util ~1–50 %). F1 runs **B independent same-geometry cases (different init-days) through ONE compiled
program via an outer `jax.vmap`**, filling the starved SMs + amortizing launch/host overhead → **~3× (B=4,
conservative) up to ~6–12× (B=8–16 if VRAM fits) more Tenerife training-days per GPU-hour** → a fast feasibility
answer for the downscaler (which needs years of local data per island).

**Approach:** outer `vmap` *around* the existing unbatched orchestration — **dycore stencils UNTOUCHED, byte-
identical fp64** — + batched loader + de-batched output + a batched root `d01→d02` cascade + fixed-B AOT keys.
**Batch WITHIN a domain, never across domains** (the v0.19 ragged-leaf trap, both analyses reconcile it as a
shape-padding artifact that does NOT recur for homogeneous cases). Files: `runtime/domain_tree.py` (vmap
wrappers), `integration/nested_pipeline.py` (batch entry + output). **NO dycore kernel surgery.**

**Complexity:** **M** (medium sprint) — numerics trivial (a vmap); the work is plumbing + the gates.
**Risk:** MED — the one correctness risk is a latent cross-case coupling op (a batch-axis reduction/broadcast or
`*_save` scratch aliasing) → caught by G1.

**VERIFIABLE ENDPOINT — all three gates must PASS; proof object `proofs/v023/batched_minigrid/REPORT.md`:**

- **G1 — Correctness (contamination-free + tight-tolerance; prerequisite, BEFORE any training data).**
  *[REVISED 2026-07-02 after the real-dycore CPU gate — the original "byte-identical `max_abs_diff=0.0` vs
  standalone" is UNACHIEVABLE under `vmap`:* XLA reorders the dycore's vertical-integral/accumulation fusion under
  the batched shape (ph_tend ~5.8e-11 even at **B=1** with no second lane; scales with grid size) — the SAME class
  as the already-shipped fused-vs-eager tolerance path, NOT a bug.* So G1 = **(a) CONTAMINATION-FREE, proven: a
  singleton `vmap` (B=1) reproduces the same diff (no second lane exists) AND identical-IC B=2 lanes are
  byte-identical to each other; (b) each lane matches standalone WITHIN the accepted fused-vs-eager tolerance band**
  (quantified per-field). No masking / clamp / `nan_to_num`. Contamination proof done on CPU (real dycore); GPU G1
  confirms on-backend. **Consequence: F1 ships opt-in, contamination-free, tolerance-not-bitwise — G2 (identity-
  stats vs CPU-WRF) is the real fidelity gate. Manager-resolved via the fused-mode precedent (report to the user).**
- **G2 — CPU-identity statistics (the validation the user specified):** take **N≈4 independent Tenerife init-days
  that already have CPU-WRF results**, run them as ONE **batched B=N program for a few forecast-hours (≈3–6 h)**,
  and compute per-field identity statistics (RMSE, correlation, bias) of **GPU-batched vs CPU-WRF, per day**.
  PASS = stats are **within the established single-case GPU-vs-CPU identity bands** — i.e. a batched lane agrees
  with CPU-WRF *as well as* a standalone GPU run does (batching does not degrade fidelity).
- **G3 — Benchmark / throughput (the win):** at a fixed production B (≥4 if VRAM permits) measure **cases/GPU-hour
  vs serial standalone**, **peak VRAM (must fit 32 GB with margin)**, **GPU-util** (Nsight / util-summary), and
  compile cost. PASS = **throughput ≥ ~2.5× at B=4** (the conservative both-models floor) with VRAM fitting;
  report the resulting **% of large-grid 5090 best-case** (target band ~55–75 %).

**Gate ladder:** the CPU-side vmap-plumbing + the B=2 bit-gate harness build **WITHOUT the GPU** → then G1
(bit-identity) → G2 (CPU-identity stats) → G3 (benchmark) → pick the fixed production B. The 5090 is needed only
for the G1/G3 *measurement*, when 0:2 frees it. **Fallback if G1 fails:** CUDA-MPS multi-process (weaker, escape
hatch only — its radiation transient duplicates per process and OOMs at N≈2–3).

---
## ⭐ LEAD FEATURE (cont.) — F1b: Parallel-in-Time FAST Multi-Day Forecast (the user 2026-07-01, sub-feature of F1)

**Added by the user 2026-07-01 as a sub-feature of F1. NOT started (roadmap only).** The near-term highest-value
application of the F1 batch engine: **compute a Tenerife-only 5–10-day 1 km forecast as fast as technically
possible.**

**What:** split the N-day forecast into **B uniform temporal chunks**, **re-initialize each chunk from the driver
at its start time** (t0 + n·U), feed each chunk the driver's lateral BCs for its window, **run all B in parallel via
the F1 `vmap` batch**, then **stitch the usable windows in time with a seam-blend**. This is re-initialized
("parallel-in-time") downscaling — a DIFFERENT product from a continuous run (driver-anchored, NOT bit-identical),
which is explicitly fine here.

**Driver = AIFS (project decision, the user 2026-07-01):** AIFS won the Tenerife base-model bakeoff vs GFS + WN2,
scored on several-hundred weather stations → **AIFS = the daily-forecast driver; GFS = historical-only** (AIFS
history is limited). So F1b targets **AIFS cadence ≈ 6-hourly** → chunk starts + U are 6 h-aligned (e.g. U=12 h,
spin-up S=6 h, W=18 h, B=10 → 120 h / 5 days).

**Bit-identity NOT required (the user):** seam-blend is expected + OK; beyond ~2-day lead time the forecast is
inherently uncertain, so the re-init/seam error barely moves *real* skill. The gate is **real forecast skill +
wall-clock**, not byte-identity to a continuous run.

**HARD requirement (the user): NO hidden losses + wall-clock-maximized.** The WHOLE pipeline (loading/IO **and** the
kernel) must run hard-optimized — no hidden host-bound stall, no silent skill degradation. Also evaluate the
**coarser-dt lever** (ties to P4 / K2 `time_step`/`n_sound`): if a larger dt cuts wall-clock **without** raising the
divergence/NaN rate (skill stays ≈ fine-dt), fold it in.

**Reuses:** the F1 `vmap` batch engine (gpuwrf side ~free) + a met chunk-generator (B chunk-ICs from ONE AIFS
forecast) + a temporal stitcher/blender. Same-geometry, same-W chunks → clean homogeneous batch (never ragged).

**TARGET: a Tenerife-only 5-day (120 h) 1 km forecast in ≤ ~5 h warm wall-clock** (the user's estimate — ~comparable
compute to the 9-nest 7-island 24 h fine-dt run that landed ~5 h).

**VERIFIABLE ENDPOINT (proof `proofs/v023/batched_minigrid/F1B_REPORT.md`):**
- **H0 (inherited F1-G1 — batch-engine correctness):** each chunk lane **byte-identical (`max_abs_diff = 0.0`)** to
  that chunk run standalone → no cross-lane contamination. (Distinct from "not bit-identical to a continuous run",
  which is the *method*, not the *engine*.)
- **H1 (completes):** a full **Tenerife 5-day (120 h) parallel-in-time** run goes through — **all B chunks finite +
  spun-up**, seam-blended stitch produced.
- **H2 (skill — NO hidden skill loss):** the stitched 5-day product's skill (vs the AIFS-driven reference and/or the
  station network) is **not materially worse** than a serial continuous reference; any coarser-dt used does **not**
  raise the divergence/NaN rate (skill ≈ fine-dt). Seam-blend + re-init explicitly allowed.
- **H3 (THE metric — NO hidden perf loss):** **warm end-to-end wall-clock** (full pipeline: load + compute + output)
  for the Tenerife 5-day 1 km forecast at the **cells/sec-optimal B** (sweep B). **TARGET ≤ ~5 h.** Report warm
  cells/sec at each B, the pipeline breakdown (load / kernel / output), peak VRAM, and confirm no host-bound stall.

**Status:** roadmap only — NOT started. Executed with/after F1 (shares the batch engine); needs the 5090 for
H0/H1/H3 when 0:2 frees it.

---
## TABLE A — v0.23.0 PERFORMANCE bundle (recommended scope: bit-identical/tiered, one release)
| ID | Item (simple) | What it brings | Complexity | Risk | Notes |
|---|---|---|---|---|---|
| P1 | **Single-scan forecast as default** (collapse the per-radiation-interval scans into ONE `lax.cond`-gated scan) | Cold compile **stops growing with forecast horizon** — one compiled program for 1 h and 72 h (today 3 h ≈ 30 min compile) | MED | LOW-MED | fix EXISTS (`run_forecast_operational_single_scan`); needs the short-horizon **tiered** equivalence gate (cond reorders reductions) |
| P2 | **`_safe_floors` gating completion** + **finite-guard 46→1 stacked reduction** | ~**100+ fewer micro-kernels/step** + 46→1 launches at the output boundary; bit-identical (gates proven 0/168 fire) | LOW | LOW | partly started (v0.22.2 Q3 batched the finite guard); finish the ungated safe-floors |
| P3 | **2-domain root fusion** (fuse root+single-leaf-child when root has one fusable child) | Halves host dispatch for the **common 2-domain dev/validation** case (proportionally the most launch-bound) | MED | MED | = the deferred S4; workload-specific (small win for 9-nest, bigger for 2-dom); needs CPU determinant-matrix gate |
| P4 | **`calc_coef_w` Python-for-loop → vectorized** (`jnp.arange`+`where`) | Cold-compile-time cut (≈258 scatters → vectorized HLO); no runtime change | LOW-MED | LOW-MED | bit-identical vs savepoint oracle (WRF source uses a sequential recurrence — verify) |
| P5 | **Autotune-cache default-ON** (after a canary A/B) | Cold-compile cut for the fp64 acoustic cluster; resolves a doc/code mismatch | LOW | LOW | verify it isn't already on in v0.22; measure RAM + warm s/step don't regress |
| P6 | **Dead-code + idiom cleanups** (4.13–4.21: vestigial constants, `w_solve_core`/`thomas` import, duplicate decouple fns, `jnp.pad`→`concatenate`, RRTMG `+zeros_like` idioms, bounded event-tail default) | Smaller HLO graph + removes mis-transcription traps; tiny per-item, free | LOW | NONE-LOW | all bit-identical; good hygiene-sprint filler |
| P7 | **Parallel cold-compile of fused-subtree modules** + **lower XLA opt/fitting effort on fused cold compile** (ship-the-blob fallback) | Cold-compile wall/RAM cut on big trees | LOW-MED | LOW-MED | runtime-neutral (captured in exec_key); reject any lever that regresses warm s/step |

## TABLE B — KERNEL win (tiered, the one named structural kernel lever)
| ID | Item | What it brings | Complexity | Risk | Notes |
|---|---|---|---|---|---|
| K1 | **BouLac O(nz²)→O(nz)** (replace dense `(B,nz,nz)` PE matrices) | ~**1.30× on the BouLac step + ~387 MB VRAM**; the named low-risk kernel win | MED-HIGH | MED | BLOCKED by the fp32 mixed-precision compile pathology; **spike first:** does a `lax.scan`-based O(nz) form dodge the "very slow compile" WITHOUT the fp32 rewrite? If yes → tiered-gated ship; if no → waits for fp32 |

## TABLE C — THE STRUCTURAL MILESTONE (separate major, NOT a point release)
| ID | Item | What it brings | Complexity | Risk | Notes |
|---|---|---|---|---|---|
| M1 | **fp32 operational state — ADR-031 acoustic-perturbation-form rewrite** | **÷1.8–2.4 dycore, ÷9 pow-heavy physics, ~2× less VRAM, fits 1 km grids fp64 OOMs on**; measured 4.3× core-step in the spike. THE multiplier + unblocks K1 + fp32-BouLac | VERY HIGH (multi-sprint, joint Opus+GPT) | HIGH | pervasive precision change; needs full tiered re-gate + the **24–120 h skill gate** (future work). Its own milestone (v0.24/v1.0-track), NOT v0.23.0 |

## TABLE D — OPERATIONAL / COMPILE-CACHE (carry-over, gate the next paid campaign)
| ID | Item | What it brings | Complexity | Risk | Notes |
|---|---|---|---|---|---|
| O1 | **#145 cross-region nested-AOT prewarm gate** (per-geometry prewarm on B200, persisted) | Multi-region B200 campaigns warm-start (no per-region cold compile mid-run) | MED | LOW-MED | AOT blobs are arch+geometry-specific (sm120≠sm100); manager-owned, pre-paid-campaign |
| O2 | **#111 compile-cache "just works" OOTB** (RAM-aware auto-defuse + a `prewarm` command) | A cold fused 9-nest on a RAM-tight box no longer thrashes (the issue 0:2 hit) | MED | LOW | the missing auto-protection; pairs with O1 |
| O3 | **#146 B200 runbook tooling** (plot-watcher venv, deadline-wrapper, pod-lifecycle, colon-free drains) | Smoother/cheaper paid pods | LOW | LOW | from the v0.21.1 run |

## TABLE E — FEATURES (carry-over v0.22.0 scaffolds → faithful physics)
| ID | Item | What it brings | Complexity | Risk | Notes |
|---|---|---|---|---|---|
| F-a | **F2: New-Tiedtke cumulus + NSSL 2-moment MP + aerosol-Morrison + RUC LSM** | Scheme coverage (more configs runnable) | HIGH (per-scheme oracle) | MED | scaffolds from v0.22.0; one focused GPT per scheme, WRF-Fortran ref |
| F-b | **F3: CAM-UW PBL** | PBL scheme coverage | MED-HIGH | MED | scaffold |
| F-c | **G2: moving nests + adaptive Δt + global nests** | Capability (vortex-following, etc.) | HIGH | MED-HIGH | scaffold; 375-var output part already landed |
| F-d | **G3: urban BEP/BEM + lake** | Urban/lake capability | MED-HIGH | MED | scaffold |
| F-e | *(multi-GPU NVLink sharding)* | Multi-card throughput | — | — | implement-only — **cannot test (no NVL hardware)** |

## TABLE F — VALIDATION (the honest debt)
| ID | Item | What it brings | Complexity | Risk | Notes |
|---|---|---|---|---|---|
| V1 | **E skill-band harness runs** for the landed opt-in features (G0 two-way nest, F1 3-D TKE, G1 DA, K2) on small grids | Confirms the v0.22.0 opt-in features actually skill, not just run | MED (small grids) | LOW | the validation backlog the user flagged |
| V2 | **24–120 h skill gate** (vs CPU-WRF, the operational-relaxed bands) | Gates fp32 (M1) + any tiered change's FULL value | HIGH | — | the gate the whole fp32 milestone hinges on |
| V3 | **v0.22.2 mountain-2-nest validation** (0:2, in progress) | Confirms the host-bound wall-clock win on real terrain | — | — | already running; feeds back |

## Recommended v0.23.0 scope (manager view)
**v0.23.0 = F1 (lead feature, batched mini-grid saturation) + F1b (parallel-in-time fast multi-day forecast) + TABLE A (P1–P7) + the K1 spike** — F1/F1b first
(the user 2026-06-30/07-01), then a coherent **bit-identical/tiered compile+launch-count
performance release** (the real remaining wallclock wins now that 4.1 + the output boundary shipped).
**M1 (fp32) = its own major milestone** (the biggest lever but HIGH risk + needs V2). Features (E) +
O1/O2/O3 + V1 run as parallel tracks the user prioritizes. Re-verify GLM's kernel levers vs v0.22.1 src
in a 1-pass spike before executing P-items.

## ADDENDUM (2026-06-28, empirical from 0:2's v0.22.2 384 A/B)
Per-phase NEST_PERF_TIMERS (mean/boundary): **M9-diag 23,219 ms DOMINANT** >> writer 869 > prepare 760 >
finite_guard 71. The M9 per-output RRTMG radiation RE-SOLVE is the single biggest remaining output-boundary
cost (wall-clock AND VRAM). v0.22.2 attacks it 3 ways (radcap VRAM default; pipeline overlap opt-in;
radiation-from-carry skip opt-in/lossy). **NEW v0.23 perf lever P0 (data-driven, highest output-boundary
impact): a BIT-IDENTICAL reduction of the M9 re-solve** — investigate reusing the forecast scan's own
radiation diagnostics instead of a full re-solve (GPT: M9 recomputes full RRTMG; the carry overlaps part of
it but at a different sample time — the bit-identical-reuse question is whether the forecast radiation at the
output step can be threaded out without a re-solve). If bit-identical → removes ~23 s/boundary by default.
High value, MED-HIGH complexity, MED risk (radiation timing/identity). Pairs with P3 (pipeline default-on).
Also confirmed: Phase-1 host-work-reduction = **9% steady win, byte-identical, default-on** (789 vs 865 s/fc-h).

## P-ITEM RE-VERIFY vs v0.22.2 src (Opus, 2026-07-02) — detail `proofs/v023/perf_bundle/PITEM_SCOPING.md`
Ground truth = `origin/main` v0.22.2 (`53bd20bf`), verified against the byte-clean worktree (not the stale v0.22.1 checkout).
- **ALREADY SHIPPED → HARVEST OUT of the active bundle:** **P5** (autotune-cache default-ON, `compile_cache.py:479`), **P7a** (parallel cold-compile), **P7c** (fused-AOT serialize/load "ship-the-blob"). GLM 4.1 FUSED+AOT confirmed shipped.
- **CPU-SIDE, FRONTRUNNER-READY NOW (bit-identical / CPU-oracle-gated, no GPU):** **P2** (`_safe_floors`: 7 still-ungated sites + finite-guard real fix = ONE ravel+concatenate reduction, NOT `jnp.stack` — shape trap), **P4-Loop1 ONLY** (`acoustic_wrf.py:830-833` vectorize; **Loop-2 :836-846 is a Thomas recurrence — do NOT vectorize**, breaks bit-identity), **P6** (dead-code/idiom hygiene, tiered; 4.21 donate_argnums kept separate MED-risk), **P7b** (lower-XLA-effort wrapper; CPU cold-compile A/B), **P1 gate-run** (CPU equiv gate exists; `segscan_equiv.json` Jul-2 already proves segmented==single-scan BITWISE → likely pass).
- **GPU-GATED (wait for the 5090):** P1 default-flip (warm s/step confirm), P3 2-domain root fusion (= deferred S4), P7b final warm-s/step no-regress.
- **K1 BIG CORRECTION — do NOT re-run the stale spike.** The roadmap's "does `lax.scan` O(nz) dodge the slow compile?" is **already answered NO**: `k1_boulac_onz_verdict.json` = O(nz) still 727s vs dense 51.5s; pathology reproduces for BOTH lax.scan + Python-unrolled (`mynn_pbl.py:149-155`). Reframe K1 as a hard fusion-boundary reformulation (shares root cause with fp32/ADR-031) OR **park for M1**. Correctness already proven (1.30×/−387MB in isolation).
- **P0** (M9 re-solve bit-identical reduction) = output-boundary/radiation-timing research in the **F1-owned pipeline** → separate scoping task, sequence AFTER F1 plumbing.

## STATUS: LOCKED by the user 2026-06-28, EXTENDED 2026-06-30
v0.23.0 = **F1 (LEAD FEATURE — batched mini-grid GPU saturation, the user 2026-06-30, G1/G2/G3 endpoint above) + F1b (parallel-in-time FAST multi-day forecast — Tenerife 5-day 1km in ≤~5h warm wall-clock, AIFS-driven, the user 2026-07-01, H0/H1/H2/H3 endpoint above)** + P0 (M9 bit-identical reduction) + P1-P7 + K1 spike. M1 (fp32) = own milestone. Features(E)/Operational(O1-3)/Validation(V1-2) = parallel tracks. **NOT started — roadmap only (the user 2026-06-30: "noch nicht starten").** When unparked, F1 leads; its CPU-side plumbing + B=2 bit-gate harness build without the GPU, GPU only needed for the G1/G3 measurement. Then the P-items (CPU-side/bit-identical) while the 5090 stays 0:2-locked.
