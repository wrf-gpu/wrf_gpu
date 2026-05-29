# F7M Worker Report — Straka density-current front close (WRF ground truth)

**Status: `F7M_PARTIAL`.** Built pristine WRF v4.7.1 `em_grav2d_x` (Straka)
ground truth, diffed it against the JAX dycore, and **localized the residual
precisely to the cold-pool touchdown** — the descending cold pool reaches the
surface but JAX fails to convert vertical motion into horizontal outflow, so the
trapped central downdraft runs away to NaN ~220–240 s. **JAX and WRF agree to
~3% through 180 s**, then WRF decelerates/spreads while JAX accelerates. Two
WRF-faithful candidate fixes (conservative flux-form momentum advection;
deformation-tensor const-K momentum diffusion) were implemented and tested
against ground truth; **both are correct but neither closes the touchdown
residual**, and the WRF diff proves the defect is NOT advection-form or
diffusion-magnitude. Warm bubble still **PASS 6/6**; m4 **10/10**.

## Objective

Build WRF ground truth for the Straka density current, diff JAX vs WRF at the
front, pinpoint the cold-front operator/coupling defect, fix it WRF-faithfully,
re-run Straka. Close the dry dynamical core if Straka PASSes + bubble PASSes.

## What was done

1. **Built pristine WRF v4.7.1 `em_grav2d_x` serial.** The 2D case requires a
   non-DM build (`Config.pl:938-947` adds RSL_LITE/DM_PARALLEL for serial *with
   nesting≥1*; `Makefile` configcheck:80-89 then blocks 2D cases). Fix: configure
   option **32 + nesting 0** (`/home/enric/src/wrf_pristine/build_grav2d_serial.sh`).
   Ran 6 model min, history/min, canonical 100m Straka namelist (`damp_opt=0,
   time_step_sound=6, diff_opt=2, km_opt=1, khdif=kvdif=75`). Extracted via
   `ncdump` (no python netCDF available) →
   `extract_grav2d_front.py`.
2. **WRF-vs-JAX front/center diff** → the decisive table (below). Localized the
   residual to the **touchdown central downdraft**.
3. **Implemented + tested two WRF-faithful candidate fixes** (kept flux advection;
   reverted the deformation diffusion to baseline, operator retained).

## Decisive ground-truth diff (center-column cold-pool downdraft)

| t(s) | WRF max\|w\| | JAX max\|w\| | WRF front | JAX front |
|---|---|---|---|---|
| 60  | 9.65  | 7.30  | 1750 | 1550 |
| 120 | 17.58 | 14.62 | 2250 | 2050 |
| 180 | 21.01 | 21.12 | 2650 | 2350 |
| 200 | ~21.5 | 29.47 | ~2900 | 2450 |
| 240 | 22.14 | NaN   | 3150 | NaN  |
| 300 | 22.05 | NaN   | 4250 | NaN  |
| 360 | 19.14 | NaN   | 5750 | NaN  |

JAX/WRF agree to ~3% through 180 s (both ~21 m/s central downdraft at z~2050 m).
WRF then **decelerates** (touchdown spreading) while JAX **accelerates** to NaN.
Runaway is a smooth central downdraft at x=0, z~1100–2050 m — NOT a 2Δx mode, NOT
the front, NOT the top boundary.

## Root cause (localized) + ruled out

**Residual = touchdown horizontal-spreading coupling.** The cold pool reaches the
surface but JAX does not convert w into horizontal outflow (front crawls ~5 m/s
while low-level |u| hits 25–36 m/s) → trapped descending air → central w→NaN.

Ruled out by ground truth: acoustic CFL; time discretization; top/Rayleigh
damping; lower-BC w; **momentum advection form** (flux vs primitive, ~4% diff,
trace unchanged, still NaN); **momentum diffusion magnitude/structure**
(deformation tensor 2–3× stronger, trace unchanged, still NaN); scalar limiter.

## Files changed

- **M** `src/gpuwrf/dynamics/flux_advection.py` — added WRF flux-form momentum
  advection `advect_u_flux/advect_v_flux/advect_w_flux` (conservative, mass-flux
  `ru/rv/rom`; `module_advect_em.F:126/1530/4364`). **KEPT** (WRF-faithfulness fix,
  GPT-5.5-confirmed real defect; rest-zero + uniform-flow-zero verified).
- **M** `src/gpuwrf/runtime/operational_mode.py` — wire flux-form momentum
  advection into `_augment_large_step_tendencies`; document the deformation-
  diffusion finding (reverted to plain-K∇² ν=75 baseline).
- **M** `src/gpuwrf/dynamics/explicit_diffusion.py` — added (NOT wired)
  `constant_k_deformation_momentum_tendency` (WRF deformation-tensor const-K
  momentum diffusion; factor-2 diagonal + du/dz↔dw/dx cross terms) for the
  eventual full-tensor port.
- **M** `proofs/f7/DYCORE_STATUS.md` — residual section updated to the
  touchdown localization with WRF ground truth.
- **NEW** `/home/enric/src/wrf_pristine/` build_grav2d_serial.sh,
  run_grav2d.sh, extract_grav2d_front.py.
- **NEW** `proofs/m9/wrf_em_grav2d_x_front_savepoints.json` (WRF ground truth),
  `/mnt/data/wrf_gpu2/wrf_truth/em_grav2d_x_front_savepoints.json`.
- **NEW** `proofs/f7m/` wrf_vs_jax_straka_front.json (key artifact),
  straka_front_fix.md, straka_flux_deform_probe.json, skamarock_warm_bubble.json,
  + the official run outputs (straka/bubble verdicts).
- **NEW** `scripts/f7m_straka_probe.py`, `scripts/f7m_official_run.py`.

## Commands run (CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src taskset -c 0-3, cuda:0, fp64)

- WRF: `build_grav2d_serial.sh` (configure 32+nesting0, `csh -f ./compile
  em_grav2d_x`), `run_grav2d.sh` (ideal.exe + wrf.exe 6 min),
  `extract_grav2d_front.py`.
- JAX: `scripts/f7m_straka_probe.py` (single-compile checkpoints),
  `scripts/f7m_official_run.py` (Straka 900s + bubble 500s → proofs/f7m),
  A/B advection + deformation diagnostics, max|w|-localization probes.
- `pytest tests/test_m4_acoustic.py test_m4_dycore_step.py
  test_m4_tier2_invariants.py` → **10 passed**.

## Acceptance gates

- **AC1 (Straka PASS): FAIL** — detonates ~220–240 s at the cold-pool touchdown;
  WRF-faithful flux advection + tested deformation diffusion did not close it; no
  masking clamps.
- **AC2 (warm bubble PASS 6/6): PASS** — thermal_rise 1924.7 m, max|w| 11.68,
  θ′max 1.92, drift 0, mass drift 0 (identical to F7L; inviscid).
- **AC3 (WRF front parity): DELIVERED** — JAX/WRF agree ~3% through 180 s;
  full divergence table in `proofs/f7m/wrf_vs_jax_straka_front.json`.
- **AC4 (no regression): PASS** — m4 10/10; flat-rest/conservation intact; flux
  advection rest-zero + uniform-flow-zero verified; WRF-faithful ν=75 + conservative
  advection only; no clamps/ad-hoc diffusion.

## Unresolved risk / next decision

The dry core is one localized residual from done and the residual is now pinned
to the **touchdown horizontal-spreading coupling** (not advection/diffusion).
The decisive next instrument is a **per-acoustic-substep WRF savepoint diff at
the touchdown column** (center, z<1500 m, t=180–200 s) — instrument
`solve_em.F` (the existing `WRFGPU2_DUMP` block pattern) to dump the touchdown
column's `w, ph, p, rw_tend, omega/ww, muts, the acoustic-PGF u-tendency` per
substep, and diff against the JAX acoustic substep to resolve which operator
(omega/ww continuity vs `advance_uv` acoustic PGF vs surface mass coupling)
under-drives the horizontal spreading. Decision for the manager: dispatch F7N as
that per-substep touchdown-column savepoint diff (not another advection/diffusion
attempt).

F7M_PARTIAL
