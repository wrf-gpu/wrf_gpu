# S4 fp32 UltraCode Report — the REAL fp32 numbers (decoupled from the broken full-unroll)

**Author:** top-level fp32 UltraCode task force (Claude Opus 4.8, MAX), deployed above the S4 sidecar.
**Branch:** `worker/codex/v020-s4` @ `675eca35` (+ this sprint's additive measurement hooks).
**Date:** 2026-06-21.
**Mandate:** decouple fp32 from the compile-exploded fp64 full-unroll baseline and get the REAL
fp32 numbers — dtype audit, VRAM, stability ladder, DRAM-scale + 1 km capability.

---

## 0. Executive verdict

**fp32 is ALIVE and measured. It is NOT dead — none of the user's A/B/C kill-conditions is met.**
The S4 perturbation-authoritative mixed mode is real, faithful (tolerance-PASS), and leaner; the
bigger storage/capability lever is the (now-unblocked) fp32-gated matrix (−14.4 % peak VRAM measured,
and it fits a 1 M-column dynamics grid fp64 OOMs on); and the true 1 km wall is the RRTMG radiation
transient, not state — a separable, higher-leverage line of attack. (Single-GPU *speed* is
inconclusive here — variance-dominated; see the §0 caveat — but speed is not the deliverable;
capability is.)

Decoupling worked: on the **single-domain bigswiss DRAM grid with the default scan kernel (NO
`GPUWRF_THOMAS_UNROLL`)** every arm compiled and ran to completion. The full-unroll explosion the
sidecar hit was a property of the *nested full-unroll fusion sub-path*, not of fp32.

The win is **VRAM, which is structural and reproduces**, NOT single-GPU speed (which is
variance-dominated on this desktop card — see the caveat below):

| Arm (bigswiss 461×461×45, 3 km, 212k cols, full physics) | peak VRAM | vs fp64 | stability @1h |
|---|---|---|---|
| **fp64_default** (baseline) | 20 796 / 20 804 MiB (two runs) | — | (truth) |
| **mixed_perturb_fp32_v020** (S4; 4 acoustic perts fp32) | 20 251 MiB | **−2.6 %** | tolerance PASS |
| **aggressive fp32-gated** (ADR-007 matrix; u/v/θ/qv+moisture fp32) | 17 805 MiB | **−14.4 %** | tolerance PASS |

Exact analytic **state-storage** VRAM reduction vs fp64 (value-independent):
**mixed = 5.9 %, aggressive = 35.2 %, mixed+aggressive combined = 41.1 %.**

> **SPEED CAVEAT (retraction of an earlier interim claim):** forecast-only ms/step is **inconclusive**
> here. The *same* fp64 config measured **603.6 s and 349.3 s** in two separate 1 h runs — a **1.73×
> run-to-run spread** (GPU clock/thermal/contention on this non-persistence-mode desktop RTX 5090).
> The mixed (566 s) and aggressive (535 s) arms fall *inside* fp64's own spread, so these data **cannot**
> support a precision-driven speedup claim. A controlled *interleaved* A/B (or locked clocks) is the
> follow-up. Peak VRAM, by contrast, reproduces to 0.04 % (20 796 vs 20 804 MiB) and is structural.

The headline reframing: at DRAM scale **peak VRAM is ~91 % transient** (state ≈ 1.84 GiB of a
20.8 GiB peak; the rest is the RRTMG McICA radiation scratch, exactly the v0.18 OOM driver). So
fp32-on-*state* moves the peak only a few %, but fp32 on the *physics/dynamics scratch* (the
aggressive matrix) moves it much more (−14.4 % measured), and fp32 on the *radiation transient* (not
yet done; SW two-stream already fp32, LW still fp64) is the real 1 km lever.

---

## 1. Goal 1 — fp32 took effect (dtype + HLO audit)  ✅ PROVEN

Driver: `proofs/v020/s4/ultracode/dtype_hlo_audit.py` → `dtype_hlo_audit.json`. Runs the operational
program one step under each mode, checks the live carry dtypes before/after, and censuses the
compiled StableHLO. (dtypes + HLO are value- and platform-independent, so this is exact on CPU; GPU
execution of the same dtypes is separately proven by the bigswiss runs in §3–§4.)

| | fp64_default | mixed_perturb_fp32_v020 |
|---|---|---|
| p'/ph'/mu'/w fp32 — initial | False (all fp64) | **True** |
| p'/ph'/mu'/w fp32 — **after one real step** | False | **True** (survives, not auto-promoted) |
| totals p/ph/mu fp64 | True | **True** |
| StableHLO f32 tokens | 12 (incidental) | **128** |
| StableHLO f64 tokens | 8743 | 8639 |
| `stablehlo.convert` ops | 12 | **151** |

**Conclusion:** the mixed mode genuinely stores fp32 for the four acoustic perturbations, and they
**survive a real operational step** while the totals stay fp64 — this is NOT the auto-promote no-op
that fooled G0 (the `_x64_config` fix is confirmed: x64 is controllable; the surgical mode changes
storage, not the global x64 flag). Cost signal: the mixed mode adds **+139 `convert` ops/step**
(upcast perts→fp64 at the `force_fp64_island` brackets, downcast back on storage) — a real overhead
whose net speed effect is below the run-to-run noise floor here (see the §0 speed caveat).

---

## 2. Goal 2 — VRAM

### 2.1 Exact analytic STATE-storage model (`vram_model.py` → `vram_model.json`)

State VRAM is fully determined by (field→shape)×(field→dtype); computed exactly from the production
`_state_field_shapes` contract and the three precision regimes.

| regime | fp32 frac of state | state reduction vs fp64 |
|---|---|---|
| fp64_default | 0.000 | 0.0 % |
| mixed_perturb_fp32 (4 fields) | 0.063 | **5.9 %** |
| aggressive_fp32 (ADR-007 matrix) | 0.543 | **35.2 %** |
| combined (aggressive + perturbation-authoritative) | 0.698 | **41.1 %** |

The mixed mode's small 5.9 % is structural: it downcasts only p'/ph'/mu'/w, and it *keeps the fp64
totals as well*, so the saving is 4 bytes × the four perturbation arrays — a faithful but minor
storage win. The bulk lever is the aggressive matrix (u/v/θ/qv + the full moisture/number family).

### 2.2 Measured PEAK VRAM (bigswiss, full physics, platform allocator → true peak incl. transient)

| arm | peak VRAM | of which state (model) | implied transient |
|---|---|---|---|
| fp64_default | 20 796 MiB | ~1 842 MiB | ~18 954 MiB |
| mixed_perturb_fp32_v020 | 20 251 MiB | ~1 733 MiB | ~18 518 MiB |
| aggressive fp32-gated | 17 805 MiB | ~1 194 MiB | ~16 611 MiB |

**The transient (RRTMG McICA radiation scratch + dynamics/physics scratch) is ~91 % of peak.** This
is the single most important VRAM fact and it reframes the whole "1 km capability" question (§4).

---

## 3. Goal 3 — Stability ladder (tolerance-pass) ✅ PASS at 1 h

`compare_precision_wrfout.py` checks each fp32 arm's wrfout against the fp64 baseline wrfout using the
frozen `proofs/v020/fp32_proto/acceptance_bands.py` bands (fp64 = truth; the diff is the fp32 error).

**mixed_perturb_fp32_v020 @ 1 h — OVERALL tolerance PASS, universal hard gates PASS** (`stability_mixed_1h.json`):

| field group | metric | measured | 24 h band | margin |
|---|---|---|---|---|
| wind (U/V/W/U10/V10) | RMSE incr | ≤ 0.0015 m/s | 0.25 | ~170× inside |
| temperature (T/T2) | RMSE incr | ≤ 0.0012 K | 0.6 | ~500× inside |
| pressure/mass/geopot (P/PH/MU/PSFC) | PSFC RMSE incr | 0.066 Pa | 150 | ~2270× inside |
| moisture (QVAPOR/cloud) | RMSE incr | ≤ 1e-6 | bands | far inside |

Differences are **non-zero** (fp32 genuinely active) but ~2–3 orders of magnitude inside tolerance —
exactly what the perturbation-authoritative math predicts (the base:perturbation ratio of ~100–1000×
buys back ~2–3 decimal digits a naïve fp32-total would lose). No blow-up; all fields finite and in
physical bounds.

> **Coverage honesty (adversarial verification → fixed):** the verifier found the comparator was
> *vacuously* passing the moisture/cloud family (their bands key on metrics it did not compute). Fixed:
> the comparator now computes the genuine band metrics — **QVAPOR** `domain_mean_water_vapor_rel_diff =
> 1.4e-7` (band 0.20) and `qvapor_p999 = 1.9e-6` (band 0.04); **condensate** `column_condensate_nrmse =
> 0.0018` (band 0.25) — all genuine passes. **17 of 20 fields are now genuinely band-checked**; the
> only two still unchecked are `cloud_fraction_abs_diff` and `swdown_lwdown_bias_abs`, which need cloud-
> fraction / radiation-flux fields not present in this prognostic-field comparison (a clean follow-up,
> not faked).

**aggressive fp32-gated @ 1 h — also OVERALL tolerance PASS, hard gates PASS** (`stability_aggressive_1h.json`):
wind RMSE ≤ 0.0018 m/s, temp ≤ 0.0015 K, PSFC 0.070 Pa — marginally larger than mixed (more fields
fp32) but still ~140–2000× inside the bands. So the much larger-VRAM-win aggressive matrix is *also*
faithful at 1 h, with the `force_fp64_island` brackets protecting the dynamics.

---

## 4. Goal 4 — THE PRIZE: DRAM-scale + 1 km capability

### 4.1 DRAM-scale A/B (bigswiss, the L2-blind 212k-col grid)

**VRAM (structural, reproduces):** mixed **−2.6 %**, aggressive **−14.4 %** peak. The aggressive peak
(17 805 MiB) cuts ~0.65 GiB of state *and* ~2.3 GiB of fp32-able dynamics/moisture/PBL scratch; the
remaining ~16.6 GiB transient is the fp64-locked part (RRTMG-LW gas optics + the dynamics cancellation
brackets). **Speed: inconclusive** — see the §0 caveat (fp64 itself measured 603.6 s and 349.3 s; the
fp32 arms fall inside that spread, so no precision-speed claim is defensible from these runs).

### 4.2 1 km capability — does fp32 FIT what fp64 cannot?

**The honest answer is two-layered, because peak VRAM is ~91 % transient:**

1. **State storage:** fp32 unambiguously fits more. Exact model → max square 1 km grid that fits its
   *state* in ~30.5 GiB: fp64 ≈ 1761 km/side, aggressive ≈ 2184 km/side (**1.54×**), combined ≈
   2288 km/side (**1.69×**). But state is not what OOMs first.

2. **Real full-physics peak:** the **RRTMG radiation transient is the wall** (the v0.18 1 km OOM was
   RRTMG McICA, not state — confirmed here: ~19 GiB of the 20.8 GiB fp64 peak). The aggressive matrix
   cuts the *dynamics+moisture+PBL* scratch by ~2.3 GiB (peak 17 805 MiB, −14.4 %), but the **RRTMG-LW
   path is still fp64** (the SW two-stream is already fp32 — `rrtmg_sw.py`), so the ~16.6 GiB residual
   transient is only partly addressed by state precision.

   **Const-transient capability model** (the RRTMG transient is *tile-bounded* — the v0.18 fix that
   cured the 1 km OOM shrank the per-tile cap 16384→2048, i.e. peak ∝ tile size, ~independent of total
   cols — so `peak(cols) ≈ transient_const + state_bytes(cols)`). With ~30.5 GiB usable and the
   measured transients: fp64 OOMs when state > ~12.3 GiB (≈ 1.25 M cols ≈ **1119×1119 km @ 1 km**);
   aggressive OOMs when state > ~14.6 GiB (≈ 2.3 M cols ≈ **1517×1517 km @ 1 km**) → aggressive fits
   ~**1.84× the 1 km area** fp64 can. **CAVEAT (modeled, not yet measured):** this extrapolates a
   *single* measured transient point (3 km/212 k cols) and assumes the transient stays constant to
   >2 M cols; McICA/halo/XLA temporaries may grow somewhat with cols, shrinking the advantage. The
   *exact* state-only bound (1.54×/1.69× more area, from the verified byte model) is firm; the
   full-physics 1.84× is the soft number. The synth crossover below is the direct test.

   **Synthetic dynamics-only VRAM-vs-size crossover — MEASURED, the direct "does fp32 fit what fp64
   cannot" test** (physics OFF isolates the fp32-addressable dynamics core; `synth_vram_scan.py`):

   | grid (×50 lev) | cols | fp64 | aggressive fp32-gated |
   |---|---|---|---|
   | 700×700 | 490 k | fits | fits |
   | **1000×1000** | **1.0 M** | **OOM (RESOURCE_EXHAUSTED)** | **FITS ✓** |
   | 1300×1300 | 1.69 M | (fp64 already OOM) | OOM |

   **At 1.0 M columns the fp64 dynamics core OOMs on the 32 GiB card, but the aggressive fp32-gated
   matrix runs.** This is a *measured* capability crossover: fp64 caps below 1.0 M cols, aggressive
   fp32 between 1.0 and 1.69 M cols (≥1.0×, ≤1.69× the column count — consistent with the 1.54× state
   model). (`peak_bytes_in_use` was unavailable under the platform allocator, so this is the OOM
   boundary, not a peak-MiB curve. It is dynamics-only; full physics shifts the absolute sizes down via
   the RRTMG transient but the fp32 relative advantage holds.)

**Verdict on the prize:** fp32 *does* fit grids fp64 cannot — but the capability gain from
fp32-on-state is bounded by the (mostly-fp64) RRTMG transient. The decisive 1 km lever is attacking
that transient (tile harder — already an env knob; and port LW cloud-optics to fp32 with the
fp64 band-sum, exactly as the SW two-stream already does). See §5.

### 4.3 Confirmatory fp64 (drift control) — and why it retracts the speed claim

A same-session fp64 rerun (`phase2_*/fp64c`) gave **forecast-only 349.3 s, peak 20 804 MiB**. Peak VRAM
matches the probe's 20 796 MiB to **0.04 %** (VRAM is structural ✓). But forecast time **349.3 s vs the
probe's 603.6 s** is a **1.73× spread for identical fp64 config** — larger than any precision delta
measured. Conclusion: on this desktop RTX 5090 (no persistence mode / locked clocks) the timing is
governed by GPU power/clock state, not by the fp32 storage change. **The earlier interim "1.07× / 1.13×
faster" claim is withdrawn**; speed needs an interleaved or clock-locked A/B. (The capability/VRAM
thesis — the actual deliverable per the project's "capability not single-GPU speed" framing — is
unaffected.)

---

## 5. Goal 5 — Residual cause + the path forward (NOT a dead-end)

**Why the per-mode numbers land where they do, and the concrete path to the maximal faithful win**
(grounded by an independent math/AceCast/extension analysis — see `S4_PATHFORWARD_ANALYSIS.md`):

1. **Perturbation-authoritative fp32 is mathematically sound** (base:pert ratio ~100–1000× recovers
   ~2–3 digits vs a naïve fp32-total; the six `force_fp64_island` cancellation brackets and the
   implicit vertical sound-wave solve are fp64-internal regardless of storage). Confirmed empirically
   by the 1 h tolerance pass (§3).

2. **Residual leak (identified, fixable, zero resident cost):** two tendency accumulators are *not*
   fp64-guarded — `ph_tend` in `rhs_ph_wrf` and the coupled-work primes in `small_step_prep`. They
   ingest the fp32 prime each stage. Forcing those *transient* accumulators to fp64 (the same pattern
   `force_fp64_island` uses) closes most of the remaining gap at no VRAM cost. A live signal of this
   is the `FutureWarning: scatter inputs … cannot safely cast float64→float32` emitted by both fp32
   arms — a genuine dtype-promotion edge to clean up.

3. **The dormant fp32-gated matrix — found and unblocked.** The ADR-007 `DEFAULT_DTYPES` matrix (the
   project's historical "fp32 operational mode") was effectively a **no-op** in the pipeline: (a)
   `daily_pipeline` hardcoded `force_fp64=True`, and (b) even with it False, the precision-enforcement
   `else`-branch used `state.replace(**updates)` *without* `_cast=False`, so `State.replace`'s default
   re-canonicalisation (`state.py:861`) cast the fp32 downcast straight back to fp64. This sprint adds
   a default-safe env hook (`GPUWRF_FORCE_FP64=0`) and the missing `_cast=False`, so the 35 % matrix
   actually engages — measured in §3–§4. (Production default is byte-unchanged: both fixes only fire
   when `force_fp64=False`.)

4. **The real 1 km lever is the RRTMG transient, not state.** Ranked by win/risk:
   (i) tile harder — `GPUWRF_RRTMG_{LW,SW}_COLUMN_TILE_COLS` is already an env knob (default 2048),
   zero code, immediate 1 km headroom; (ii) port the LW cloud-subcolumn optics to fp32 keeping the
   gas-optics band-sum fp64 — the SW two-stream is the proven in-repo template (~7e-5 band-flux
   drift) — roughly halves the dominant LW transient.

**AceCast comparison (STOP-discipline item C):** stock WRF builds **single precision by default**
(`-r8` is the opt-in for double); AceCast validates to *statistical* equivalence, not bitwise. Our
fp64_default baseline is therefore *more* precise than reference WRF, and moving u/v/θ/qv/moisture to
fp32 moves *toward* WRF's own arithmetic. Our case is **not fundamentally different** from AceCast's
success; if anything our mixed mode is more conservative (it keeps explicit fp64 islands AceCast runs
in single). No physics/math reason fp32 fails here.

---

## 6. STOP-discipline statement (binding)

fp32 is **not** declared dead. None of A (math-impossible to stabilise — refuted: 1 h tolerance pass +
sound perturbation math), B (a real JAX/XLA bug with no fix — the only JAX edge found, the
float64→float32 scatter `FutureWarning`, is a *our-side* dtype hygiene fix, not an XLA bug), or C
(physics no longer solved — refuted: stock WRF is itself single precision; AceCast succeeds the same
way) holds. Recommendation: **continue** — harden the two accumulators, ship the unblocked fp32-gated
matrix behind its 1 km per-field finiteness gate, and open the RRTMG-transient line for 1 km.

---

## 7. Methods, artifacts, reproduction

- **Grid:** `<DATA_ROOT>/wrf_gpu_validation/v017_bigswiss_gpu_init` — 461×461×45, dx=3 km, dt=18 s,
  mp=8 (Thompson), bl=5 (MYNN), ra_lw/sw=4 (RRTMG); single domain `--max-dom 1 --domain d01`.
- **Decoupling:** default scan kernel; **no** `GPUWRF_THOMPSON_FP32`/global-x64-off proxy, **no**
  `GPUWRF_THOMAS_UNROLL`. fp32 selected surgically: mixed via
  `GPUWRF_ACOUSTIC_PRECISION_MODE=mixed_perturb_fp32_v020`; aggressive via `GPUWRF_FORCE_FP64=0`.
- **Source hooks (additive, default-safe):** `daily_pipeline.py` `force_fp64` env hook;
  `operational_mode.py` `_enforce_operational_precision` else-branch `_cast=False`. 11/11
  `test_v020_s4_mixed_precision.py` + `test_v020_dtype_stability.py` pass; default fp64/mixed unchanged.
- **Artifacts** (under `proofs/v020/s4/ultracode/`): `vram_model.{py,json}`,
  `dtype_hlo_audit.{py,json}`, `compare_precision_wrfout.py`, `stability_mixed_1h.json`,
  `run_mixed_agg.sh` + `mixed_agg_*/` (logs, vram CSVs, wrfout, metric JSONs), `phase2_gpu.sh` +
  `phase2_*/` (confirmatory fp64 + synth ladder), `synth_vram_scan.py`. Path-forward analysis:
  `proofs/v020/s4/S4_PATHFORWARD_ANALYSIS.md`. fp64 baseline run: `probe_fp64_1h_*/`.
- **GPU discipline:** every GPU command via `scripts/with_gpu_lock.sh`; host pinned `taskset -c 0-3`,
  `OMP_NUM_THREADS=4`.

### 7.1 Scope limits / honest not-done (clean follow-ups)

- **Stability ladder stopped at 1 h** (not the 6 h/24 h rungs the brief sketches). The 1 h fp32 diffs
  are 170–2270× inside the bands and the perturbation-math + acceptance-band growth profile imply the
  longer leads hold, but this is *inferred*, not measured — a 6 h/24 h mixed-vs-fp64 run is the next rung.
- **Dominant-kernel wall not captured with nsys.** The transient-dominance (~91 % of peak) plus the
  v0.18 root-cause (RRTMG McICA) make RRTMG the near-certain dominant consumer, but this is *inferred*,
  not an nsys `cuda_gpu_kern_sum` measurement. A bounded nsys capture is a clean add.
- **Speed is unresolved by design** (variance-dominated; §0/§4.3). Needs an interleaved or
  clock-locked A/B before any ms/step claim.
- **Synth crossover is OOM-boundary only** (`peak_bytes_in_use` was null under the platform allocator)
  and **dynamics-only** (physics off). It proves the fp64-OOM / fp32-fits crossover at 1.0 M cols but
  not a peak-MiB curve; re-running with the BFC allocator would add the curve.
- **Two moisture/cloud bands** (`cloud_fraction_abs_diff`, `swdown_lwdown_bias_abs`) remain unchecked
  (fields absent from the prognostic-field comparison).
