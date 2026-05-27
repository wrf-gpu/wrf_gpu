# Tester Report — M7 L2 Nest Scout

**Sprint**: `2026-05-27-m7-l2-nest-scout`
**Role**: opus research scout (acting as sonnet-test-engineer per the role prompt; the contract dispatches this sprint to opus xhigh as research-only)
**Branch**: `tester/opus/m7-l2-nest-scout`
**Date**: 2026-05-27
**Predecessor**: `MILESTONE-M7-CLOSEOUT.md` (PIPELINE_GREEN on Canary 3 km single-domain)

## What this sprint did

Pure scouting, no code changes. Surveyed the L2 (9 km d01 + 3 km d02 nest) backfill on disk, audited the GPU port for hidden 3 km assumptions, traced the AIFS → wrfbdy → GPU path, and produced an end-to-end nested-grid backfill design plus a top-5 risk register. All proof objects landed under `.agent/sprints/2026-05-27-m7-l2-nest-scout/`.

## Proof objects produced

| File | AC | What it proves |
|---|---|---|
| `l2_day_inventory.json` | AC1 | 28 L2 day directories at `/mnt/data/canairy_meteo/runs/wrf_l2/`; 26/28 successful WRF runs; all 28 carry `wrfinput_d01`, `wrfinput_d02`, `wrfbdy_d01`, namelist; 4 days carry raw `wrfout_d01/d02_*`; the other 24 carry only `thin_gridded_d0{1,2}_v1.nc` (surface, 21 vars). Header-reads confirm L2 d01 = 94×60×45 @ 9 km, L2 d02 = 160×67×45 @ 3 km, same Lambert projection as L3. |
| `9km_feasibility_audit.md` | AC2 | The GPU dycore is dx-clean: `Projection.dx_m` is plumbed end-to-end through `OperationalNamelist.grid.projection`, `MAPFAC_*` and η coefficients are wrfinput-sourced per domain. Three small risks surfaced (silent 3 km default in `physics/surface_layer.py:162`; 12 s dt cap in `coupling/driver.py:37`; the d02-replay-only ingestion path that ignores `wrfbdy_d01`). None are dycore-stopping. |
| `d01_boundary_forcing_audit.md` | AC3 | Gen2's L2 wrfbdy_d01 schema fully documented (12 × 6 h time slabs, 20 variables, bdy_width=5, base+tendency layout). GPU side: `decode_wrfbdy` already exists in `gpuwrf.io.boundary_replay`; `apply_lateral_boundaries` already exists in `gpuwrf.coupling.boundary_apply` and supports `spec_bdy_width=5, spec_zone=1, relax_zone=4` exactly as the L2 namelist demands. The single missing piece is the gluing function (~3 small helpers, ~½-day sprint). |
| `nest_backfill_design.md` | AC4 | End-to-end pipeline: Stage A (CPU pre-process — kept) → Stage B (GPU d01, NEW) → Stage C (GPU d02, existing) → Stage D (writer + verify, existing). Wall-clock projection: ~3-5 min for 24 h d01 GPU forecast, ~5.4 min for 24 h d02 (measured), ~14 min per day-24 h end-to-end, ~6 h for the full 27-day backfill at 24 h horizon. Sprint sequencing L2.1..L2.6 listed with ~1.5 days of worker time + one overnight GPU window. |
| `risk_assessment.md` | AC5 | Top 5 risks (R1 ingest slip, R2 d01 9 km dynamics blow-up, R3 zero hydrometeor inflow, R4 publishing over-promise, R5 24/28 days have stripped d01 wrfouts) with severity/likelihood/owner/mitigation. Plus go/no-go gates (G1-G3 for backfill, P1-P2 for publishing). |
| `tester-report.md` | AC6 | This file. |

## Tests added or run

This was a research-only sprint with **no model code changes**, so no new pytest cases were authored. The role-prompt template inherits from the worker+tester pattern; this contract reshapes it into "scout + report" per the sprint header (`Status: READY — RESEARCH ONLY (no model code changes)`). The verification I did is documented in the artefacts above:

- `l2_day_inventory.json`: built by walking `/mnt/data/canairy_meteo/runs/wrf_l2/` and header-reading every `wrfinput_d01`, `wrfinput_d02`, `wrfbdy_d01` via `netCDF4` directly (no GPU, no JAX). All 28 dirs survived the read. Header attrs (`DX/DY/MAP_PROJ/CEN_LAT/CEN_LON/TRUELAT1/TRUELAT2/MOAD_CEN_LAT/STAND_LON/GRID_ID/PARENT_ID/I_PARENT_START/J_PARENT_START/PARENT_GRID_RATIO`) all read non-null and consistent with the L2 namelist.
- `9km_feasibility_audit.md`: substantiated by grep/read of `src/gpuwrf/{contracts,dynamics,runtime,coupling,physics,io,integration}/*.py`. The `DEFAULT_DX_M = 3000.0` silent fallback in `physics/surface_layer.py:162` is documented file:line; the `MAX_LIFTED_DYCORE_DT_S = 12.0` cap is documented at `coupling/driver.py:37`; the wrfbdy-vs-wrfout-history gap is documented at `integration/d02_replay.py:274-277`.
- `d01_boundary_forcing_audit.md`: substantiated by reading `wrfbdy_d01` schema directly (verified 12-time, 5-bdy_width, 20-variable layout matches the L2 namelist) and tracing the existing `decode_wrfbdy` / `wrfbdy_boundary_oracle_probe` / `_pack_wrfbdy_outer_leaf` / `apply_lateral_boundaries` code path. The integration design in the audit relies only on functions that already exist.
- `nest_backfill_design.md` wall-clock projection uses the M7-CLOSEOUT measured numbers (5.71 s warm 1 h on d02, 324.78 s 24 h end-to-end). Projection arithmetic shown in the document; not a measurement.

If the sprint sequence proceeds, L2.1's contract should include a pytest covering the new `pack_wrfbdy_all_times` helper against synthetic linearly-varying wrfbdy data. That test belongs in the worker sprint, not this tester scout.

## Fixtures used

- `/mnt/data/canairy_meteo/runs/wrf_l2/*` — all 28 day directories (read-only).
- `/mnt/data/canairy_meteo/runs/wrf_l3/20260428_18z_l3_24h_*` — namelist comparison only (read-only).
- `/mnt/data/canairy_meteo/runs/wrf_l2/20260429_18z_l2_72h_20260524T204451Z/wrfbdy_d01` — schema reference (read-only).

No binary fixture data was added to git. The inventory JSON is text and is committed to the sprint folder.

## Gaps and unresolved issues

1. **L2.1 ingest sprint not yet dispatched.** This scout sprint does not create it; the manager dispatches.
2. **Tier-4 d01 RMSE on the 4 surviving-wrfouts days has not been computed.** First measurement would be inside the L2.3 sprint after L2.2 produces a GPU d01 wrfout.
3. **Hydrometeor `*_bdy` schema extension question (Q1 in AC3) is unresolved.** Recommendation: ship v0 with qv_bdy only, accept the gap, follow up.
4. **dx-aware surface vsgd term (F3 in AC2) is unresolved.** Recommendation: ship v0 with the 3 km silent default at 9 km, accept the small drag underestimate, follow up.
5. **dt cap relaxation (F4 in AC2) is unresolved.** Performance-only; v0 ships with the 12 s cap.
6. **Sprint contract claims "27 days verified"; AC1 inventory finds 28 dirs (one is a rerun: `20260521_18z_l2rerun_l2_72h_*`).** Trivial discrepancy; the contract's "27" stands as the count of distinct calendar days.

## Decision: BACKFILL_NEEDS_NEW_CODE

The L2 9 km d01 parent cannot be backfilled today without one small (`~½-day`) worker sprint that wires `gpuwrf.io.boundary_replay.decode_wrfbdy` into a new `gpuwrf.integration.d01_replay.build_d01_replay_case_from_wrfbdy(run_dir)` — analogous to the existing `build_replay_case` but consuming `wrfinput_d01` + `wrfbdy_d01` instead of `wrfout_d02` history. All other building blocks (boundary apply, dycore, physics, halo, RK, wrfout writer, AEMET verify) already exist and are domain-agnostic.

**Concrete next-sprint recommendation**: dispatch **L2.1 — d01 ingest** (worker, ~½ day) with the schema and helper signatures pre-specified in `d01_boundary_forcing_audit.md`. Acceptance: builds a `ReplayCase` for L2 day `20260524_18z_l2_72h_20260525T225640Z` (success=1, raw d01+d02 wrfouts on disk for Tier-4 comparison), integrates one 1 h GPU step, produces finite output. Then L2.2 chains d01 ↔ d02, L2.3 Tier-4 RMSE on the same day. Total critical path: ~1.5 days of focused worker time + one overnight GPU window for the 27-day batch.

**Publishing the M7 result is independent and can proceed today** with the phrasing recommended in R4 of `risk_assessment.md` ("Canary 3 km single-domain forecast validated at 5.4 min/24 h on RTX 5090, 156.82× CPU. Multi-day historical L2 nested backfill in progress."). Backfill rollout completes inside this week without any change to that publishing claim.
