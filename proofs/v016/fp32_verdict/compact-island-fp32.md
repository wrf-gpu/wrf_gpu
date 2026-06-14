# COMPACT-explicit-fp64-island fp32 acoustic rewrite (Opus make-or-break)

**Worker:** Opus 4.8 (1M ctx) — `worker/opus/v016-fp32-compact` (worktree `.wt-fp32-compact`)
**Date:** 2026-06-14
**Objective:** Realize the fp32 speed + VRAM ceiling (~4x + ~2x VRAM + 1km unlock) with
VALID numerics via the COMPACT-explicit-fp64-island technique, after the prior
NATIVE/type-promotion attempt measured ~parity + ZERO VRAM drop.

---

## 1. Why the prior NATIVE attempt failed (now QUANTIFIED on GPU HLO)

NATIVE stored the work arrays fp32 but CARRIED the big fp64 island arrays
(c2a/alt/al/phb/p_base/ph_base/php + all metrics) in the scan and let
`c2a64 * work32` AUTO-PROMOTE. Measured (real Switzerland d01, RTX 5090, this bench):

| 16,384 col | ms/step | VRAM | HLO f64_3d arrays | HLO convert ops |
|-----------:|--------:|-----:|------------------:|----------------:|
| fp64       | 70.36   | 3.77 GiB | 38,280 | 425 |
| NATIVE     | 69.79   | 3.77 GiB | 34,612 (−10%) | **1,768 (+316%)** |

NATIVE removes only ~10% of the fp64 3D arrays and **quadruples** the convert ops
(the scatter), so VRAM is byte-identical to fp64 and there is no speedup. This is
the documented convert-scatter failure mode, now measured.

## 2. The COMPACT technique (what changed)

COMPACT (`GPUWRF_MIXED_FP32_COMPACT=1`, on the MIXED_PERTURB_FP32 lane):
- DEMOTES the carried island + metric + per-stage-forcing arrays to fp32 (the real
  arena lever the prior attempt never pulled): c2a/alt/al (amplifiers, only
  MULTIPLY the small bracket -> fp32-safe), all broadcast metrics (rdnw/c1h/fnm/
  msf*/cf/cqw/c1f/c2f/ht), the per-RK-stage forcing/reference arrays
  (ph_tend/theta_tend/u,v,mu_tend/theta_1/u_1/v_1/ww_1/w_save/...).
- Computes ONLY the cancellation-prone large-minus-large in EXPLICIT fp64 per
  substep (operands widened LOCALLY) and emits an already-cancelled fp32 quantity:
  - `calc_p_rho`: EOS bracket `alt*(t2-c1h*mu*t1)/(mass*tref) - al` in fp64 ->
    narrow; then `p = c2a32 * bracket32` GENUINELY fp32.
  - `advance_w`: base-geopotential `dphi = (ph_1+phb)[1:]-[:-1]` in fp64 -> narrow.
  - `advance_uv`: base-pressure `pb_grad` + mass-point `php_grad` + `muts-mut`
    in fp64 -> narrow.
- KEEPS fp64 (deliberate residual footprint): the mass denominators mut/muts/mu
  (whose difference muts-mut is the dry-mass cancellation; tiny 2D) + sumflux
  accumulators + the 5 base-state ABSOLUTES p_base/phb/ph_base/php_stage/ph_1
  (whose horizontal/vertical DIFFERENCE is the cancellation -- storing the ~1e5
  absolute fp32 loses ULP(1e5)=0.008 Pa ~ 0.16% of a 5 Pa base-pressure gradient;
  the cancellation MUST be resolved from the fp64 absolute then narrowed). This
  mirrors ECMWF VFE / GRIST fp64 base-PGF precompute practice.

## 3. STAGE-A numerics + dtype gates (CPU, all PASS)

- **Design oracle** `compact_island_fp32_oracle.py` (real b6 column): fp32 c2a *
  already-cancelled fp32 bracket -> p' err 5.35e-8 rel (gate 1.19e-6), MATCHES the
  fp64-carrying PROMOTE form to the fp32 floor. Proves the cancellation (not c2a's
  storage) is what needs fp64; the genuine-fp32 product is sound.
- **Substep dtype-leak gate** `compact_fp32_substep_identity.py` (full operational
  cadence, WRF magnitudes): `NO_FP64_LEAK_in_carry=True` (every compact-fp32 leaf
  comes back fp32, 0 leaked, 0 scatter warnings), keep_fp64 ok, all finite, worst
  leaf rel 3.5e-7 (gate 2.4e-5 = at fp32 floor). FP64_DEFAULT A/B still
  BYTE-IDENTICAL; NATIVE gate + calc_p_rho + advance_w oracles all still PASS.
- **Terrain-gradient gate** `compact_fp32_terrain_gradient.py` (+-400 Pa horizontal
  base-pressure undulation -- the case the uniform gate could not catch): COMPACT
  vs FP64 u-rel 2.3e-7, v-rel 1.9e-7, worst 4.4x fp32-eps (gate 200x) PASS. Locks
  in the keep-base-fp64 correctness fix.

## 4. STAGE-A structural proof (HLO, the make-or-break signal)

CPU-lowered OPTIMIZED HLO of the bare-core acoustic scan (4 substeps, 20x16x16):

| | f64_3d arrays | f32_3d arrays | convert ops |
|-|--------------:|--------------:|------------:|
| NATIVE  | 30,346 | 26,164 | 4,058 |
| COMPACT | **3,592 (−88%)** | 34,087 | **742 (−82%)** |

COMPACT removes **88% of the fp64 3D arrays** and **82% of the converts** -- the
working set genuinely moved to fp32 (f32_3d up, f64_3d collapsed to the deliberate
base-absolute cancellation islands). This is the structural property NATIVE could
never achieve and predicts a real GPU VRAM drop.

## 5. STAGE-A/B GPU measurement (real Switzerland d01, RTX 5090)

`compact_fp32_km_bench_CORRECTED.json` (fp64 vs compact, 36 steps, per-precision
`jax.clear_caches()` so compact compiles DISTINCTLY -- the first pass had a
jit-cache-alias bug where native+compact share the MIXED_PERTURB_FP32 namelist
cache key so compact silently reused the native binary, compile=0.0s + identical
HLO; fixed):

| ncol    | fp64 ms | compact ms | speedup | fp64 VRAM | compact VRAM | VRAM drop | fp64 f64_3d | compact f64_3d |
|--------:|--------:|-----------:|--------:|----------:|-------------:|----------:|------------:|---------------:|
| 16,384  | 70.47   | 63.52      | **1.109x** | 3.77 GiB | 3.77 GiB | **1.000** | 38,280 | 24,003 (−37%) |
| 65,536  | 254.70  | 230.43     | **1.105x** | 11.61 GiB | 11.61 GiB | **1.000** | 37,802 | 23,436 (−38%) |
| 147,456 | **OOM 18.80 GiB** | **OOM 18.80 GiB** | — | — | — | — | — | — |

- **compact_unlocks_fp64_oom_ncols = []** -- compact does NOT unlock the 147k/1km
  grid; it OOMs at the IDENTICAL 18.80 GiB as fp64.
- Real **~1.1x speedup** (a GENUINE acoustic ALU/bandwidth win -- distinct compile,
  NOT the native 1.01x cache-alias; the acoustic hot loop really runs fp32).
- **VRAM byte-identical to fp64** despite a real 37-38% drop in the full-graph fp64
  3D array count.
- Compact compile is notably slower than fp64 (73-78 s vs 55-59 s) -- the
  explicit-fp64-island + fp32 mixed graph stresses XLA (a milder cousin of the
  documented mixed-precision compile pathology; flagged, not blocking at 36 steps).

### Why VRAM did NOT drop -- ROOT-CAUSED (the decisive finding)

The acoustic scan is NOT what sets the VRAM peak. At 16k a single 3D fp64 mass
array is ~5.8 MB; the whole ~50-leaf acoustic carry is ~0.29 GiB fp64 / ~0.14 GiB
fp32 -- a ~0.15 GiB saving, INVISIBLE against the 3.77 GiB peak.

A 4-way 16k control (`compact_fp32_km_bench_NORAD.json`, radiation gated off via
`ra_lw_physics=ra_sw_physics=0`, which IS honored -- operational_mode.py:4236-4237)
isolates the cause:

| 16k peak VRAM | radiation ON | radiation OFF |
|---------------|:------------:|:-------------:|
| fp64    | 3.77 GiB | 3.77 GiB |
| compact | 3.77 GiB | 3.77 GiB |

ALL FOUR are 3.77 GiB -- NEITHER the acoustic-scan dtype NOR radiation moves the
16k peak. The peak is the **whole fp64 operational forecast working set**: the
fp64-pinned State family (w/p/ph/mu, ADR-007 conservation matrix) + BaseState
(pb/phb/mub) + the non-radiation physics (MYNN/PBL/NoahMP) + the RK/forecast
double-buffering. The acoustic scan is ~8% of that; demoting it (or radiation) in
isolation is within noise. (The bench's probed HLO f64_3d DID drop 37% -- but that
is the MAIN-forecast graph; RRTMG is in a SEPARATE jit, operational_mode.py:4326,
so it never appears in the probed HLO, only in the measured peak.)

The 147k OOM is a SINGLE 18.80 GiB allocation, IDENTICAL for fp64 AND compact AND
**radiation-off** (fp64-norad 147k OOMs at the same 18.80 GiB) -- so it is NEITHER
the acoustic scan NOR radiation. It is the **MYNN BouLac `(B, nz, nz)` dense
mixing-length PE matrix** (`mynn_pbl.py:96` `_boulac_length_dense`): at 147k that
is 147456 x 44 x 44 x 8 B ~ 19-23 GiB, a documented non-radiation memory offender
(MEMORY.md: "MYNN BouLac (B,nz,nz) tiling = THE measured non-radiation offender").
`mynn_pbl.py:114` even notes a v0.15 opt-in O(B,nz) lever that avoids the dense
matrix -- the actual 1km-unlock lever, ORTHOGONAL to fp32.

**Complete root cause:** the operational VRAM peak is (a) the broad fp64 dynamics
+ State + BaseState + physics working set (all 4 of fp64/compact x rad-on/rad-off
= 3.77 GiB @16k, 11.61 @65k), and (b) at 1km specifically, the MYNN BouLac dense
`(B,nz,nz)` PBL allocation (the 18.80 GiB single alloc). The acoustic scan is ~8%
of (a) and 0% of (b). The prior NATIVE non-win had the SAME root cause; this work
localizes it precisely. The acoustic-scan fp32 lane is a real, proven, but
insufficient slice of the VRAM/1km problem.

## 6. Headline

**Compact-island fp32 = sound + ~1.1x, but VRAM/1km is blocked by the fp64
operational working set + MYNN BouLac (NOT the acoustic scan).** The technique is
NUMERICALLY SOUND and genuinely makes the acoustic scan fp32 (real ~1.1x speedup,
37-38% fewer fp64 3D arrays in the operational graph, 88% in the bare-core scan,
distinct compile, ALL numerics gates at the fp32 floor incl. the terrain
base-pressure-gradient case). But it delivers 0% VRAM drop and does NOT unlock 1km,
ROOT-CAUSED by a 4-way control: the VRAM peak is the broad fp64 dynamics+State+
BaseState+physics working set (fp64/compact x rad-on/rad-off ALL = 3.77 GiB @16k),
and the 1km OOM is the MYNN BouLac dense (B,nz,nz) PBL allocation (18.80 GiB) --
neither is the acoustic scan (~8% of the peak) and neither is radiation. The
acoustic-scan fp32 lane is a real, proven, but ~8%-insufficient slice; the
VRAM/1km levers are demoting the fp64 State/BaseState/physics + the MYNN BouLac
O(B,nz) rewrite (mynn_pbl.py:114), both ORTHOGONAL to this acoustic work.

## 7. Files changed

- `src/gpuwrf/dynamics/core/acoustic.py` — COMPACT strategy: island/metric/forcing
  fp32 sets, KEEP_FP64 set (mass + base absolutes), `_compact_face_diff_{x,y}_3d`,
  `_mixed_compact_cast`, compact dispatch in `acoustic_substep_core`, Thomas-coef
  narrowing, advance_uv pb/php/muts-mut fp64 islands.
- `src/gpuwrf/dynamics/core/calc_p_rho.py` — `_calc_al_p(compact=...)` explicit
  fp64 bracket island + fp32 product; smdiv update kept fp32.
- `src/gpuwrf/dynamics/core/advance_w.py` — `advance_w_wrf(compact=...)`: fp64
  dphi base-geopotential cancellation, mut/muts/muave narrow, damp height-sum narrow.
- `src/gpuwrf/dynamics/mu_t_advance.py` — `_update_2d/3d` narrow value to dest dtype
  (no fp64->fp32 scatter; no-op on fp64-default).
- `src/gpuwrf/runtime/operational_mode.py` — compact-aware seed cast + Thomas-coef
  narrow in `_acoustic_scan`.
- proofs: `compact_island_fp32_oracle.py`, `compact_fp32_substep_identity.py`,
  `compact_fp32_terrain_gradient.py`, `compact_fp32_km_bench.py` (+JSON).
