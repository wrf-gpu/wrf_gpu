# CPU WRF Baseline (Canairy Gen2) — Reference

Authoritative pointer to the existing CPU WRF infrastructure that this project's GPU port replaces and validates against. Recorded 2026-05-20 by manager on user instruction so future workers/testers/reviewers reading repo md files have direct access — do not duplicate this into other md files; cite this file instead.

## What exists

- **Live 24/7 daily AIFS-driven CPU WRF forecast at the Canary domain** running out of `~/src/canairy_meteo/Gen2/`.
- **Backfill on disk** as of 2026-05-20:
  - `/mnt/data/canairy_meteo/runs/wrf_l3/` — **23 daily 24-h forecast runs** (campaign label `l3`; multi-domain WRF run; each run has 5 nested domains d01..d05). Date span: `20260501_18z_l3_24h_…` → `20260519_18z_l3_24h_…`.
  - `/mnt/data/canairy_meteo/runs/wrf_l2/` — **18 daily 72-h forecast runs** (campaign label `l2`).
  - Each run directory contains: `wrfout_d0{1..N}_<UTC>:00:00`, `wrfinput_d01..d05`, `wrfbdy_d01`, `namelist.input` (run-specific), `namelist.output`.
- **Raw AIFS IC/BC source files** at `/mnt/data/canairy_meteo/data/aifs_single/` (monthly `aifs_single_YYYYMM.nc`).
- **WRF source + NVHPC-built objects**: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/` (compiled with `nvfortran`; this is why M5-S1 Thompson harness uses `nvfortran` instead of `gfortran` — see `scripts/wrf_thompson_harness_build.sh`).
- **WRF env setup**: `source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh`.
- **WRF binary**: `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` (version 4.7.1) — already pinned in `PROJECT_PLAN.md:164`.

## How this is used per milestone

**M1 (closed)**: One Tier-1 Canary fixture was sliced from `wrf_l3/20260517_18z_l3_24h_20260518T004341Z/wrfout_d01_2026-05-18_13:00:00` (see `MILESTONE-M1-CLOSEOUT.md:33`). No new CPU WRF jobs were triggered for fixtures.

**M5-S1 (current)**: The compiled Thompson microphysics objects under `wrf_gpu_src/WRF/phys/` are linked into a Fortran harness that produces the JAX kernel's fixture — see `ADR-006-thompson-jax-implementation.md:23-25`. This is the structural anti-tautology mechanism. No backfill wrfouts are used by M5-S1 (which is column physics, not 3D forecast).

**M6 (coupled short forecast — pending)**: Use the backfill as Tier-3 / Tier-4 validation:
1. Pick days where the d01 (3 km) and d04/d05 (1 km nest) solutions are mutually consistent — these are "well-converged" reference days suitable for first-pass GPU validation.
2. Confirm corresponding AIFS month files in `aifs_single/` are present (months 04..05 are confirmed on disk).
3. Use those days as the Tier-3 envelope baseline (`tier 3 short-run timestep convergence` per `VALIDATION_STRATEGY.md`).
4. **Skip days where 1 km and 3 km diverge significantly** — those are hard reference cases not appropriate for first-pass validation; defer to M7.

**M7 (Canary operational v0 — pending)**: Use the same backfill as the production-equivalence comparison source. The wall-clock evidence required by M7's exit gate (per `MILESTONES.md` M7) is measured against the Gen2 CPU runtime for the same input AIFS file. The "WRF baseline" comparator in the M7 exit gate is the Gen2 daily run, not an external WRF installation.

## Operating rules

- **Never trigger a fresh CPU WRF job for fixture work** when an existing Gen2 run covers the time window — this was the §11.6 decision and it held in M1. Slice existing wrfouts instead.
- **Never write into `/mnt/data/canairy_meteo/`** — that is Gen2's data domain. This project reads it; mutations live in `wrf_gpu2/` and `data/`.
- **Treat the backfill dates as a rolling window**: new daily runs arrive nightly; M6/M7 reproducibility commitments should pin a specific subset of run-IDs into the relevant sprint contract, not wildcard the directory.
- **AIFS coverage**: as of 2026-05-20, monthly files for 2024-04 .. 2024-08 are visible; before binding M6 fixtures to a specific date, confirm the matching `aifs_single_YYYYMM.nc` is present.

## Cross-links

- `PROJECT_PLAN.md:164` — WRF v4.7.1 binary pinning.
- `PROJECT_PLAN.md:174` — AIFS adoption + Gen2 reuse policy.
- `PROJECT_PLAN.md:176` — Gen2 operational stack (Thompson + MYNN-EDMF + Noah-MP + RRTMG).
- `RISK_REGISTER.md:14` — IC/BC dataset availability risk (already references Gen2-shared fixture).
- `MILESTONE-M1-CLOSEOUT.md:33,64` — proves the §11.6 reuse policy worked for M1.
- `ADR-005-first-physics-suite.md` — picks Thompson, citing Gen2 stack as project-specific evidence.
- `ADR-006-thompson-jax-implementation.md:23-25` — Fortran harness links Gen2-built WRF objects.
- `scripts/wrf_thompson_harness_build.sh` — concrete linker invocation against Gen2 WRF object tree.
