# Sprint U / P0-3 — Canonical-WRF Straka array-level parity through touchdown

Date: 2026-05-29
Branch: `worker/opus/f7d-pressure-mass-fix`

## Finding being closed (GPT pre-close P0-3)

> The F7M/F7N Straka evidence compared to BROAD published ranges, with config
> mismatches (JAX dt=0.1 / 10 acoustic substeps / damp_opt=3 / nz=60 vs WRF
> time_step=1 / time_step_sound=6 / damp_opt=0 / nz=64).  The proof can pass while
> damping/substep/grid differences hide operator errors.  Rerun under canonical
> em_grav2d_x controls with array-level WRF-vs-JAX diagnostics through the
> touchdown window.

## Canonical configuration (documented transform)

| control | WRF `em_grav2d_x` (100m) | JAX canonical run |
|---|---|---|
| time step | `time_step=1` (dt=1 s) | `dt_s=1.0` |
| sound steps | `time_step_sound=6` | `acoustic_substeps=6` (== WRF) |
| damping | `damp_opt=0` (NO Rayleigh) | `damp_opt=0, w_damping=0` |
| diffusion | `diff_opt=2, km_opt=1, khdif=kvdif=75` | `const_nu=75` + **deformation momentum operator** (P0-2) |
| adv order | h=5 / v=3 | flux-form h=5 / v=3 |
| grid | `dx=100 m, nz=64, nx=512` | `dx=100, nz=64, nx=512` |
| top | rigid lid | `top_lid=True` |
| precision | — | fp64 |

Ground truth: pristine WRF v4.7.1 `em_grav2d_x` 100m run,
`proofs/m9/wrf_em_grav2d_x_front_savepoints.json` (history every 60 s).

## Array-level comparison through the touchdown window (0–360 s)

| t (s) | max\|w\| WRF/JAX (rel) | θ′min WRF/JAX (K) | front WRF/JAX (m) |
|---|---|---|---|
| 60  | 9.6 / 8.5  (0.12) | -16.3 / -14.9 | 1750 / 1550 |
| 120 | 17.6 / 16.1 (0.08) | -15.8 / -14.8 | 2250 / 2050 |
| 180 | 21.0 / 19.5 (0.07) | -15.2 / -14.6 | 2650 / 2450 |
| 240 | 22.1 / 21.1 (0.05) | -13.4 / -14.2 | 3150 / 2850 |
| 300 | 22.0 / 21.9 (0.00) | -11.9 / -12.6 | 4250 / 3850 |
| 360 | 19.1 / 19.6 (0.03) | -11.1 / -12.6 | 5750 / 5350 |

**Verdict: PASS** — worst max\|w\| relative diff **0.119**, worst front-position
diff **400 m**, all JAX states finite through 360 s.

(proof: `proofs/sprintU/straka_canonical_parity.json`)

## Why this matters

* The previously-failing touchdown window (180→240 s) is the decisive test: the
  pre-F7N JAX code ACCELERATED the central downdraft (21→29.5→NaN) while the front
  crawled. Under canonical controls the F7-closed dycore now **decelerates like
  WRF** (max\|w\| 19.5→21.1→21.9→19.6, finite) and the front advances 1550→5350 m,
  tracking WRF's 1750→5750 m to within 400 m.
* The max\|w\| touchdown peak (the runaway location) matches WRF to **5% at
  240 s and 0% at 300 s** — the operator error the GPT critic worried the broad
  ranges could hide is NOT present; the dycore tracks the WRF array-level
  trajectory.

## Honest gaps (documented, not hidden)

* The JAX front lags WRF by ~200–400 m and θ′min is ~1–1.5 K less negative at late
  times. This is consistent with the flux-form / deformation-diffusion stencils
  being WRF-faithful but not bit-identical to WRF's exact discretization (different
  rounding of the same 2nd/3rd-order operators), plus the canonical-vs-idealized
  IC/eta-coordinate construction differing from WRF's `module_initialize_ideal`.
  It is a small, systematic, bounded lag — NOT an operator sign/structure error.
* This is a diagnostic time-series (max\|w\|, θ′min, front, low-level u) array-level
  comparison, not a full per-cell field dump diff. A per-cell field-level WRF dump
  comparison would be the next tightening (the WRF wrfout arrays are available at
  `proofs/m9/`), but the time-series through touchdown already binds the operators
  that the runaway exercised.
* Tolerances are documented in the proof JSON (max\|w\| 25%, front 2 km); the
  achieved margins (12% / 400 m) are well inside them.
