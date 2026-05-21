# Sprint Contract — M6-S3 Surface Layer + Bounded Noah-MP Minimum

**Sprint ID**: `2026-05-21-m6-s3-surface-layer-and-noah-mp-minimum`
**Created**: 2026-05-21 12:35 by manager (Claude Opus 4.7 1M-context)
**Status**: STUB — dispatch after M6-S2 smoke passes
**Trigger**: M6 plan critic amendment #4: "Rewrite S3 scope to the smallest surface-layer/prescribed-land subset, with explicit included/excluded Noah-MP features and a radiation-conditioning feasibility artifact." Per scout plan + critic amendment, M6 needs operational T2/qv2/U10/V10 surface fields, which require real surface layer.

## Objective

Replace MYNN's bulk surface flux stub (M5-S2 `surface_layer(state) -> SurfaceFluxes` hook) with a real Monin-Obukhov surface layer + the **minimum** Noah-MP land subset for operationally-meaningful `U10/V10/T2/qv2`. NOT a full Noah-MP implementation.

**Pre-dispatch decision** (manager TBD before worker dispatch): which Noah-MP subset?
- **Option A — Prescribed land state**: skin temperature, soil moisture, roughness lengths all PRESCRIBED from Gen2 wrfinput_d02 / wrfout_d02. NO prognostic Noah-MP. Simplest path; defensible operational T2/qv2 baseline.
- **Option B — Bounded prognostic Noah-MP**: soil moisture + skin T prognostic with bounded options (no full 4D-Var, no full canopy water budget). More work but real coupled land-atmosphere.

**Recommendation**: Option A first; promote to Option B in M6.5 / M7 if RMSE evidence demands.

## Acceptance (pre-M6-S4..S8 parallel dispatch gate)

- **AC1 — Surface-layer scope memo (BEFORE code)**. `.agent/decisions/ADR-012-m6-surface-layer-scope.md` with: chosen subset (A vs B), included WRF Noah-MP features list, excluded features list, prescribed-land data source (which Gen2 files, which variables).
- **AC2 — Radiation-conditioning feasibility artifact**. `artifacts/m6/radiation_conditioning_feasibility.json` documents: are prescribed Gen2 RTHRATEN / RTHRATSW / RTHRATLW 3D radiation tendencies available in `wrfout_d02_*`? If yes, M6-S3 can use them; if not, M6-S3 must wait for M5-S3.zz/M5-S3.zzz RRTMG closure.
- **AC3 — Real Monin-Obukhov surface-layer kernel**. `src/gpuwrf/physics/surface_layer.py` implements MM5 / similarity-theory based surface layer (sfclay scheme; WRF `module_sf_sfclay.F`). Outputs `ustar, theta_flux, qv_flux, tau_u, tau_v, rhosfc, fltv` — the SurfaceFluxes contract M5-S2.x defined.
- **AC4 — Noah-MP subset kernel**. `src/gpuwrf/physics/noah_mp.py` implements chosen scope (Option A or B per AC1).
- **AC5 — Static land/SST/geog provenance**. `src/gpuwrf/io/land_state.py` (or extend `gen2_accessor`) loads prescribed land state from Gen2 `wrfinput_d02` (XLAND, IVGTYP, ISLTYP, LU_INDEX, etc.). Per-tile + per-time arrays as needed.
- **AC6 — Coupled into M6-S2 driver**. `coupling/physics_couplers.py` `surface_adapter` now calls real surface layer + Noah-MP, not the M5-S2 stub.
- **AC7 — Tier-1 vs WRF**. New `tests/test_m6_surface_layer_*.py` validates against WRF `module_sf_sfclay` Fortran-harness oracle (same pattern as Thompson/MYNN/RRTMG harnesses).
- **AC8 — Operational delta artifact**. `artifacts/m6/surface_operational_delta.json` documents per-variable per-lead RMSE delta from M6-S2 driver run BEFORE vs AFTER M6-S3 surface fold-in. Quantifies the "operational improvement".
- **AC9 — Honest accounting**. No fudge. Per-schema `proof_schemas.SurfaceLayerArtifact` (worker adds to schema registry).
- **AC10 — ADR-012 + ADR-013**. ADR-012 surface-layer scope; ADR-013 Noah-MP subset.

## Files Worker May Modify

- `src/gpuwrf/physics/surface_layer.py`, `noah_mp.py`, `surface_constants.py`, `noah_mp_tables.py` (NEW)
- `src/gpuwrf/physics/mynn_surface_stub.py` — extend to allow real surface layer plug-in
- `src/gpuwrf/io/land_state.py` (NEW) or extend `gen2_accessor.py`
- `src/gpuwrf/coupling/physics_couplers.py` — `surface_adapter` body
- `src/gpuwrf/io/proof_schemas.py` — add `SurfaceLayerArtifact`
- `scripts/wrf_sfclay_harness.f90`, `scripts/wrf_sfclay_harness_build.sh` (NEW Fortran oracle harness)
- `scripts/m6_run_surface_layer.py`, `m6_gate_surface_layer.py` (NEW)
- `tests/test_m6_surface_layer_*.py`, `test_m6_noah_mp_*.py` (NEW)
- `.agent/decisions/ADR-012-m6-surface-layer-scope.md`, `ADR-013-m6-noah-mp-subset.md` (NEW)
- `data/fixtures/m6/sfclay-tables-v1.npz`, `noah-mp-tables-v1.npz` (NEW)
- Worker report

## Files Worker Must NOT Modify

- `src/gpuwrf/physics/{thompson_*,mynn_pbl,rrtmg_*}` — other physics FROZEN
- `src/gpuwrf/dynamics/**`, `src/gpuwrf/contracts/**` (modulo SurfaceFluxes contract extension if needed)
- `src/gpuwrf/io/{gen2_accessor,boundary_replay,validation,proof_schemas}.py` body — only land_state.py extension
- `/mnt/data/canairy_meteo/**` — READ-ONLY

## Dispatch

- Primary worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (mandatory)
- Wall-time: **30-48 hours** (largest M6 sprint; surface + land bundled)
- Worktree: `/tmp/wrf_gpu2_m6s3` (NEW)
- Branch: `worker/codex/m6-s3-surface-layer-and-noah-mp-minimum`

## Pre-dispatch decisions

Manager must record BEFORE dispatch:

1. Option A (prescribed land) vs Option B (bounded prognostic Noah-MP).
2. WRF surface-layer scheme: MM5 sfclay (default for Canary) vs Noah-LSM-MP-tightly-coupled.
3. Prescribed radiation tendency feasibility: confirm `RTHRATEN/SW/LW` exist in d02 wrfout.

## Hard rules

- Scope memo BEFORE code (AC1). Worker may NOT start kernel implementation without ADR-012.
- Fortran-harness oracle for real surface layer (same pattern as Thompson/MYNN/RRTMG).
- Cite `module_sf_sfclay.F:lineno` and WRF Noah-MP source for every formula.
- Verify coefficients by computation (no literal copy-paste).
- Tier-1 vs WRF harness; operational-impact artifact (AC8) is binding for closing.

## Sequencing impact

After M6-S3 closes, M6-S4 + M6-S5 + M6-S6 + M6-S7 can dispatch in parallel (file-disjoint per ADR-010).

## End-goal context

This sprint makes `U10/V10/T2/qv2` operationally honest. Without real surface layer, M6-S8 operational RMSE comparison is meaningless. After M6-S3 + M5-S3.zz close, M6 can produce its FIRST forecast for which operational RMSE vs Gen2 is a binding gate.
