# v46-nightt2 — night-time land cold T2 bias (FINDINGS D29): mechanism + current aggregate check

Lane v46-nightt2 (Opus 5.5, CPU only; no GPU, no model run, no source change). 2026-10-31T00:16Z;
rev 2 2026-10-31T01:40Z after cr-v05pre FIX REPORT_SCOPE_AND_BOUNDS (F1 scope, F2 bounds).

## Verdict

**The D29 pattern is attributed to B53 and is not evident in the measured current aggregates.** D29 was
real on LW9 (v0.3.x). In the 0227 single-change A/B (FC05) it is strongly reduced by B53. The MYNN SGS
cloud fields (`qsq`, `qc_bl`, `qi_bl`, `cldfra_bl`) were dropped at the RK non-dry replace seam, so
RRTMG saw CF = 0 for boundary-layer clouds after step 1. The land surface then lost 10–48 W/m² of
downwelling LW at night and cooled radiatively; by day it received more SW and ran warm. The fix is
a95f2d7576 (`fid-cloud: retain four MYNN cloud fields across RK`), on main since 2026-10-03, still
present in `operational_mode._PHYSICS_NON_DRY_REPLACE_FIELDS:598`, and guarded by deletion-sensitive
tests (`tests/v025/fid_cloud/test_sgs_carry.py`, 5/5 PASS [M CPU] on f3cc329faa, 319 s).

Scope of the claims [M]:
- Controlled attribution covers **case 0227 only** (both arms present). 0614 has no paired baseline
  (base frames deleted), and 0502 has no B53-only intervention here.
- On v0.4.0 (3 WN3 cases) and v0.4.5 (0227 d03 τ60–72) the GPU−CPU **spatial land/coast/interior
  means** show no night cold pattern (bounds in §2). These are area means, not station residuals:
  they do not exclude spatially compensating errors at individual cells.
- No ad-hoc surface change is proposed.

This upgrades the v0.3.3 docs statement "D29 FIXED by lineage, attribution [I]" to a **measured
single-change attribution on 0227 [M]**.

## Evidence

### 1. Mechanism: single-change A/B (fid-cloud FC05, re-read here, no new run)
- Plan `evidence/fc05_plan.json` (sha256 73eb095d…). The baseline is `wn3/W6/bench24_p3`, the same LW9
  release-benchmark runs that D29/RD02 measured. The candidate is snapshot e296edcde, and its only
  source change is the 4 SGS names in the non-dry RK replace list. Same flags (flags_lw9) and the same cache.
- Signed land-mean GPU−CPU bias, case 0227 (`evidence/fc05_surface_0227.json`, sha256 e143ab42…). The
  base arm reproduces D29: d02 +6 h −0.381 K (D29 −0.42) and d03 +24 h −0.579 K (D29 −0.58, exact).

| dom | lead | T2 base | T2 +B53 | TSK base | TSK +B53 | GLW base W/m² | GLW +B53 |
|---|---|---|---|---|---|---|---|
| d02 | 3 | −0.317 | −0.012 | −0.755 | +0.012 | −17.6 | +0.6 |
| d02 | 6 | −0.381 | −0.006 | −0.767 | +0.060 | −14.4 | +1.4 |
| d02 | 14 (day) | +0.182 | −0.059 | +0.764 | −0.063 | −10.0 | +2.7 |
| d02 | 21 | −0.367 | −0.034 | −0.674 | +0.011 | −19.7 | +2.3 |
| d02 | 24 | −0.293 | −0.050 | −0.541 | −0.020 | −14.0 | +1.3 |
| d03 | 6 | −0.285 | −0.071 | −0.330 | +0.063 | −10.5 | +0.9 |
| d03 | 17 (day) | +0.588 | −0.009 | +2.529 | −0.053 | −44.8 | +1.3 |
| d03 | 22 | −0.624 | +0.019 | −1.403 | +0.208 | −39.2 | +4.8 |
| d03 | 24 | −0.579 | +0.049 | −1.382 | +0.224 | −33.0 | +5.3 |

- TSK moves about twice as far as T2, and the GLW deficit holds at all hours. The night cold bias and
  the day warm bias have the same radiative cause: missing BL cloud gives less GLW at night and more
  SWDOWN by day (FC05 land SWDOWN MAE 94.5 → 41.9 W/m²). This is the sign pattern D29 described
  ("night cold, ~0 by day": the day warm half was partly hidden by the sampled 12Z lead).
- Night land spatial T2 MAE 0.281 → 0.127 K (FC05): a clear improvement, not zero pointwise error.
- The 0614 base raw frames were deleted by an earlier shared cleanup (FC05 note), so 0614 has the
  candidate arm only (no attribution): night (CPU land SWDOWN < 1) land-mean T2 bias with +B53 is
  −0.1082…+0.0681 K on d02 (10 frames; +24 h −0.0918 vs D29 −0.96) and −0.0635…+0.0070 K on d03.

### 2. Non-defect on current code (original CPU-WRF vs GPU, land-mean, LANDMASK)
- **v0.4.0 F20** (`archive/v040/v-prep/F20_20261016`, GPU source e7e37390cc, 3 WN3 cases × d02/d03, 24 h,
  originals `<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_*`). Night frames are those with CPU land SWDOWN < 1.
  Land T2 mean of hourly means: 0227 d02 +0.0004 K; 0614 d02 +0.0077 K. For all 3 cases and both domains,
  every hourly coast and interior mean, night and day, has magnitude ≤ 0.0397 K (`evidence/coast_bias.txt`,
  `evidence/f20_bias.txt`). Hourly land RMSE is 0.04–0.16 K.
- Bounds above are **spatial means per frame** (or multi-frame means where stated), not per-cell or
  per-station bounds; they are disclosures, not new fidelity thresholds, and say nothing against D6.
- **v0.4.5 release bundle** (RW3 N72 run 20261029, bundle 0.4.5 fused_d02 exe 2836ab48…). The d03 frames
  for τ60–72 are paired with the original CPU-WRF 72 h run (`v45-rc/cpu72_254c48237452` → `wg_20260227_18z_a1`):
  night frames (CPU land SWDOWN < 1: 20, 21, 22, 23, 00 UTC) land mean per frame +0.045, +0.015,
  −0.057, −0.000, −0.001 K, i.e. **−0.057…+0.045 K hourly**, five-frame mean +0.000117 K. The 19 UTC
  frame (−0.021 K) has CPU land SWDOWN 45 W/m² and is not night by this rule (`evidence/v045_rcn72_d03_t2.txt`).
  Coast hourly −0.1105…+0.0540 K (5-frame mean −0.0142), interior −0.0515…+0.0543 K (mean +0.0017)
  (`evidence/coast_bias.txt`).
- Coast definition: a land cell with a sea cell within one grid cell (`tools/coast_stats.py`).

### 3. ALISIOS C9 cross-check (SCIENCE_STATE gap table, read before any change, as the manager asked)
- ALISIOS measures a coastal T2 bias of −1.2 °C (333 m and WN3 runs vs stations) and finds R/icloud3
  colder at 09 UTC. These are **model-vs-observation, station-sampled** numbers.
- This lane has **no station-weighted, date/configuration-matched CPU/GPU/observation residuals**, so it
  cannot attribute that −1.2 °C to CPU-WRF or to the port: **untested**. What was measured is narrower:
  the GPU−CPU *spatial coastal mean* is within ±0.0397 K hourly (v0.4.0, 24 h) and −0.1105…+0.0540 K
  hourly (v0.4.5, τ60–72 night). An area mean near zero cannot exclude compensating cell errors that a
  station happens to sample. A matched check needs the ALISIOS station list + dates + configuration,
  paired CPU and GPU frames at those cells.
- The 333 m nest was not paired here [U].
- Decision unchanged: no ad-hoc surface change.

## Performance / memory
No model change, so there is no performance or memory consequence.

## Proposed FINDINGS edit (manager)
D29 → "✓ attributed to B53 (a95f2d757) [M single-change A/B fid-cloud FC05, case 0227: d02 +6 h
−0.381 → −0.006 K, d03 +24 h −0.579 → +0.049 K, GLW −14…−48 → +1…+7 W/m²; 0614/0502 not controlled].
Current aggregates show no night cold pattern: v0.4.0 F20 3 cases hourly coast/interior spatial means
|≤ 0.0397| K; v0.4.5 RW3 0227 d03 τ60–72 night land hourly −0.057…+0.045 K (5-frame mean +0.0001),
coast −0.1105…+0.0540 K. ALISIOS coastal −1.2 °C vs stations: CPU/port attribution untested (no matched
station residuals)." Proof dir `proofs/v46_nightt2/`.

## Reproduce (CPU, about 10 min on 2 cores)
`JAX_PLATFORMS=cpu python3 proofs/v46_nightt2/tools/coast_stats.py <cpu_dir> <gpu_dir> d03` (also
`t2_stats.py`, `bias_hours.py`). FC05 tables: read `evidence/fc05_surface_*.json` rows[region=land].fields.*.{base,candidate}.bias.
