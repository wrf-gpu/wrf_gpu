# M6-S3 Manager Closeout — Surface Layer + Bounded Noah-MP

**Sprint**: M6-S3 surface layer + bounded Noah-MP minimum
**Status**: **CLOSED — Opus ACCEPT-WITH-MINOR-FOLLOWUPS; M6-S4..S8 UNBLOCKED**
**Date**: 2026-05-21 ~16:00

## What landed

Codex worker (~18min wall, much faster than 30-48h estimate due to bounded scope):
- ADR-012 (surface-layer scope) + ADR-013 (Noah-MP subset) — written BEFORE code
- MM5 sfclay kernel (`src/gpuwrf/physics/surface_layer.py`) with WRF source citations, FP64 SurfaceFluxes contract
- Bounded Noah-MP subset (Option A prescribed land) per `noah_mp.py`
- Land state manifest with SHA-256
- WRF sfclay Fortran-harness oracle + nm linkage verified
- Coupled into M6-S2 driver via `surface_adapter` 
- 28/28 M6 tests pass
- Lead-0 deltas: U10 -0.026 m/s, V10 -0.11 m/s, T2 -0.094 K IMPROVED; Q2 +0.0004 slightly degraded (root-caused as physical effect of sat-flux introduction, not bug)

## Opus reviewer findings

**ACCEPT-WITH-MINOR-FOLLOWUPS**. M6-S4..S8 UNBLOCKED with 3 binding M6-S4 prereqs:

- **F-S4-1**: Extend State pytree with `xland, lakemask, mavail, roughness_m` (and optionally `pblh`) as static FP32 surface-2D leaves. Requires **ADR-014** (or ADR-010 amendment) authorizing the contracts/state.py modification. Currently coupled surface_adapter runs against State defaults (effectively all-land) because M6-S3 contract barred state.py modification.

- **F-S4-2**: Re-pin Gen2 run for full 1h/6h/12h/24h surface RMSE. Move M6-S4 (and M6-S5/S8) onto **`20260520_18z_l3_24h_20260521T045821Z`** (25 hourly d02 history files + wrfinput_d02). This recovers F-S3-2 mu_bdy waiver and unlocks full operational-delta sweep. **Manager pre-decision: APPROVED**. Update M6-S2a + downstream sprint contracts to use new Gen2 pin.

- **F-S4-3**: Measure Tier-2 conservation **PRE-`sanitize_state`** (M6-S2 R-17 binding). Instrument pre-sanitize tap OR run sanitize-OFF parallel forecast. Also closes F-S3-1 sanitize-OFF attribution.

## Manager actions

1. **ADR-014 authoring** (manager opus task, in-progress this turn): authorize State extension for prescribed land leaves + re-pin Gen2 reference + sanitize-OFF measurement contract.
2. **M6-S4 contract update** with F-S4-1/2/3 binding + new Gen2 pin.
3. **M6-S5 inherited prereqs** (no new): lift 1s dycore cap + end-to-end wall + denominator selection.
4. **M6-S8 inherited prereqs**: same Gen2 re-pin (closes F-S3-2 waiver substantively).

## Non-blocking follow-ups

- F-min-1: When F-S4-1 lands, tighten `surface_layer._roughness_from_state` cm-guard parity with `noah_mp.roughness_from_prescribed_fields`
- F-min-2: Remove dead-code pair in `surface_layer.py:237-239`
- F-min-3: Broaden Tier-1 harness coverage from 2 rows to (stable/damped/unstable/water-warm/water-cold/snow-cover) regimes — M6-S4 owner
- F-min-4: Flip ADR-012/013 status from PROPOSED to ACCEPTED (this commit)

## M6 progress update

| Sprint | Status |
|---|---|
| M6-S1, S2a, S2, S3 | ✓ all CLOSED |
| **M6-S4** | dispatching now (codex) with F-S4-1/2/3 bundled |
| M6-S5..S7 | ready to dispatch after M6-S4 ADR-014 lands |
| M6-S8 | serial final after S4-S7 |

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 16:00
