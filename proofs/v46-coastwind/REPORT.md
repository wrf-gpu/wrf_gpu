# v46-coastwind — coastal 10 m wind excess: port defect or chaos? (Opus 5.5, 2026-10-31, rev2)

rev2 answers cr-v05pre FIX INFERENCE_SCOPE_ONLY (2026-10-31T01:36:03Z) on 97f413c4ef. The critic recomputed the numbers
and found them EXACT; only the conclusions changed. No measurement was added or altered.

**Verdict (scope-limited):** these checks found **no additional coastal 10 m wind port defect**. They do **not prove that
none exists**. On WN3 0227 d03, GPU−CPU-WRF coastal U10/V10 differences are **consistent with realization spread at the
stated scope** (one case, domain-wide CPU floor, two scientifically different GPU versions). The contract's specific
incidents (0115 stations, IC-R V10 h42–44, night stable coastal context) stay **OPEN**: a different-case aggregate cannot
close them. The only known systematic port difference on this path is still the root ring-0 boundary clock (o5-coast A4,
walk fix opt-in, HOLD). Recommendation: **no ad-hoc friction/roughness/U10 change.** No evidence calls for one, and such a
change would leave WRF. This is a recommendation, not a proof of absence. Labels [M] measured, [I] inferred.

## Inputs (read-only; no ALISIOS truth or CPU-pair reads)
| tag | what | sha256 |
|---|---|---|
| CPU | original CPU-WRF 4.7.1 twin 0227 (w43-twin0227, 4×1 decomposition), d03, 73 frames (o5-coast `twin0227_d03.npz`) | 4f5f285c… |
| G042 | GPU v0.4.2, released clock (w43-teacher `te_gpu/20260227_gpu/d03_sfc.npz`) | 57731754… |
| G040 | GPU v0.4.0 F20, 25 frames (o5-coast `f20gpu0227_d03.npz`) | 58269662… |
| C | CPU-WRF vs CPU-WRF ±1 full-θ f32 ULP twin, d03 **domain-wide** RMSE per lead (`docs/release/evidence/v033/w2/W2bR/metrics.json` pair_rmse) | d3577c22… |

Masks: coast = 5×5 dilation of the land/sea edge (same as w43-coastwind/o5-coast A1): coast_sea 617, coast_land 572,
open_sea 7672, interior_land 1462 cells. Tools: `tools/cw01_spread.py` (ecc353f7…), `tools/cw02_sensitivity.py`.
Results: `results/CW01_spread_0227_d03.json` (8e8b1894…), `results/CW02_sensitivity.json`.

## What is shown [M] (0227, d03)
**R1. Domain-wide, GPU−CPU is at or below the CPU-WRF ±1 ULP self-spread C from τ7 on** (pooled RMSE m/s):

| block | U10 GPU042−CPU | U10 C | ratio | V10 GPU042−CPU | V10 C | ratio |
|---|---|---|---|---|---|---|
| τ1–6 | 0.030 | 0.023 | 1.28 | 0.035 | 0.027 | 1.30 |
| τ7–24 | 0.103 | 0.108 | 0.96 | 0.125 | 0.135 | 0.93 |
| τ25–47 | 0.204 | 0.262 | 0.78 | 0.233 | 0.303 | 0.77 |
| τ48–72 | 0.639 | 0.856 | 0.75 | 0.744 | 0.902 | 0.82 |

Scope: C is domain-wide. **There is no coastal-mask CPU floor**, so R1 does not bound the coastal subset. τ1–6 ×1.3: the
GPU differs at round-off in every operation from step 1, while C seeds one 1-ULP θ perturbation at t0. The released root
clock lead (A4) also contributes [I].

**R2. Version comparison (regression evidence, not a chaos yardstick).** GPU042−GPU040 (GG) is about the same size as
GPU−CPU (GC) on every coast mask (WS10 pooled RMSE m/s, τ ≤ 24 h):

| mask | τ1–6 GC042 / GC040 / GG | τ7–24 GC042 / GC040 / GG |
|---|---|---|
| coast_sea | .046 / .048 / .043 | .173 / .141 / .166 |
| coast_land | .051 / .051 / .045 | .156 / .130 / .143 |
| open_sea | .017 / .016 / .013 | .098 / .082 / .096 |
| interior_land | .049 / .048 / .036 | .116 / .107 / .105 |

At τ1–6 the WS10 GC/GG ratio is 1.07 / 1.10 (GC042 / GC040) on coast_sea, 1.13 / 1.12 on coast_land, 1.39 / 1.24 on open
sea and 1.36 / 1.33 on interior land. Component coast_sea ratios: U10 1.15 / 1.16, V10 1.06 / 1.38.

Limits (critic F1): v0.4.0 and v0.4.2 differ scientifically, not only by a controlled seed, and they can share errors.
GC ≈ GG therefore **cannot identify chaos and cannot exclude a defect shared by both versions or tied to a weather
regime**. It shows only that, relative to a GPU-version change, the GPU−CPU difference is not concentrated at the coast.

**R3. Coastal amplification appears in every pair.** At τ1–6, the U10 coast_sea/open_sea RMSE ratio is GC042 2.96, GC040
3.16 and GG 3.64; V10 is 2.65 / 3.91 / 3.21. The coast amplifies differences between GPU versions as much as GPU−CPU
differences. This is consistent with flow sensitivity [I], but not a proof (same F1 limit).

**R4. Signed coastal WS10 mean GPU−CPU ≈ 0.** It is −0.001…+0.009 m/s on the coast masks for τ ≤ 47, with mixed sign and
similar to the GG biases. At τ48–72 it is −0.040 (coast_sea) and +0.005 (coast_land). Limits: a near-zero domain-masked
mean does not rule out local, directional, conditional (e.g. stable night, onshore flow) or RMSE-only effects. A real bias
could also be cancelled by a realization error of opposite sign.

**R5. CW02 is a pipeline sensitivity control, not a bound.** Injecting a coast-only +1 % WS10 into GPU042 moves the coastal
mean by +0.056…+0.092 m/s per block, so the pipeline responds to a uniform signal of that size. It is **not** a detection
limit or an upper bound on a systematic excess. rev1's "≲ 0.1 %" claim is **withdrawn**: a bound would need a null/noise
model, an uncertainty estimate and independent paired realizations (critic F2).

**R6. Version sensitivity at τ1–24 (not an uncertainty bound on the HOLD).** GPU-vs-CPU coastal RMSE on the same case
differs by up to 20 % between port versions (coast_sea WS .152 v0.4.2 vs .124 v0.4.0, A1). That does **not** bound the
WALK72 HOLD misses (+5.5…+10.5 % at τ25–47): different leads, and no realization uncertainty model. The HOLD and the
coastal "wind τ > 40 HELD" item stay under their original frozen mask/lead rules; this packet proposes **no**
lifting or scope change.

## Prior evidence reused (accepted elsewhere, not re-reviewed)
- BP46 pristine MYNN-SL R8 oracle with WRF constants on all PROD d01 8400 + d02 31,239 columns at **13Z (day only)**
  (Canary coasts included): U10/V10/UST/ZNT/HFX 0 fails (B39/B39b). It supports those operators in that context.
- o5-coast A3 relation test (τ1–24, class × Rib, GPU v0.4.0 vs CPU twin): WS10/WS_L0, drag and CH agree to ≤ 0.1 % near
  neutral and ≤ 0.3 % in weakly stable bins. The o5-coast code audit found MYNN-SL Charnock/ZNT, Noah-MP z0, the masks and
  the U10 formula identical on the release WN3 path.
- o5-coast A4: root ring-0 boundary clock one step ahead (systematic, GPU-specific), fixed opt-in (walk), HOLD.

## ALISIOS evidence (C9, SCIENCE_STATE.md gap table; checked BEFORE any change)
The station-network wind "excess" of WRF vs WN3 (pooled bias +2.9 vs +2.0 kn) splits into WU +1.8 kn and AEMET −0.9 kn
(WRF better there). 88 % of station hours are WU; 0 of 76 stations have a documented anemometer height. ALISIOS attributes
it to the WU measurement contract, not too little friction. That is a CPU-WRF-shared configuration matter and fits R1–R4
[I]. → No friction/roughness change in the port. Coastal T2 cold −1.2 °C (333 m vs WN3) and the night land T2 cold bias
(D29) belong to v46-nightt2.

## Contract items — status
| item | status |
|---|---|
| 0115 stations +1.8 % WS10 RMSE (v0.3.3, ALISIOS R32-67) | **OPEN**. Not reproduced: needs 0115 CPU/station data under admission and ≥ 2 GPU realizations per arm (released clock and walk). 0227 aggregates cannot close it. |
| V10 h42–44 event under IC R | **OPEN**. Not reproduced: needs the ALISIOS IC-R data. The 0227 τ25–47 domain ratio says nothing about that event. |
| Mechanism candidates (roughness, MYNN-SL coastal mixing, Noah-MP masking, U10 diagnostic) | No defect found by the checks above (day-column oracle, relation test, code audit, 0227 masked statistics). Not excluded for night stable or conditional regimes. |
| Coastal 72 h floor / wind τ > 40 HELD | Unchanged, HELD under the frozen rule. A per-mask floor needs a CPU pair: the partner of our 4×1 twin is the original 2×2 CPU run under ALISIOS paths (no admission), or a new CPU twin (72 h 3-nest ≈ 7–8 h on 4 physical cores, not available tonight). Alternative: GPU self-twin G (SEED, o5-coast kit, GPU window). Separate admission either way. |
| Night stable coastal MYNN-SL column oracle | **OPEN**, left to v46-nightt2's stable-BL work to avoid duplicate work. |

## Model change
None. Performance/memory impact: none.
