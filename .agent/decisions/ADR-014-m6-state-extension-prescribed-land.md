# ADR-014 — State Extension for Prescribed Land Leaves (M6-S4 prereq)

**Date**: 2026-05-21
**Status**: ACCEPTED (manager authorization for M6-S4)
**Author**: Manager (Claude Opus 4.7 1M-context)
**Trigger**: M6-S3 Opus reviewer §M6-S4 Binding F-S4-1 requires State extension for prescribed land surface fields; M6-S3 contract had barred contracts/state.py modification.

## Context

M6-S1 ADR-010 froze the State pytree shape after the M6-S2 boundary leaves landed. M6-S3 added MM5 sfclay surface-layer kernel + bounded Noah-MP Option A prescribed land. But because M6-S3 contract barred state.py modification, the coupled `surface_adapter` falls back to State-pytree defaults (effectively all-land assumption) for `xland, lakemask, mavail, roughness_m`. This means the coupled-driver U10/V10/T2/Q2 NPZ outputs are NOT operationally meaningful diagnostics yet.

M6-S2 R-17 + M6-S3 R-11 also flag that `sanitize_state` silently rebalances clipped values; M6-S4 Tier-2 conservation must instrument a pre-sanitize tap.

## Decision

Authorize M6-S4 to extend `src/gpuwrf/contracts/state.py` and `precision.py` with the following prescribed land leaves:

| Leaf | Shape | dtype | Source |
|---|---|---|---|
| `xland` | `(ny, nx)` | FP32 (gated) | Gen2 wrfinput_d02 (land mask) |
| `lakemask` | `(ny, nx)` | FP32 (gated) | Gen2 wrfinput_d02 |
| `mavail` | `(ny, nx)` | FP32 (gated) | Gen2 wrfinput_d02 (moisture availability) |
| `roughness_m` | `(ny, nx)` | FP64 (locked) | Gen2 wrfinput_d02 ZNT OR derived per `noah_mp.roughness_from_prescribed_fields` |
| `pblh` | `(ny, nx)` | FP64 (locked) | Gen2 wrfinput_d02 OR diagnostic from MYNN | (optional in M6-S4)

Loaded once at IC build time from prescribed Gen2 state; static throughout forecast (Option A scope).

## File-ownership amendment

Amends ADR-010 §File Ownership Freeze: `src/gpuwrf/contracts/state.py` and `precision.py` are MODIFIABLE by M6-S4 for this specific extension only. Subsequent M6-S5..S8 sprints inherit the new shape; no further state.py modifications without an ADR amendment.

## Gen2 re-pin decision (binding for M6-S4/S5/S8)

Manager pre-decision per M6-S3 Opus §Adversarial Probe 5: **re-pin Gen2 reference run** to one of the 5 alternative Gen2 runs that have hourly `wrfout_d02_*` files. Recommended pin: `/mnt/data/canairy_meteo/runs/wrf_l3/20260520_18z_l3_24h_20260521T045821Z/` (25 hourly d02 history per M7 plan scout inventory). M6-S4 worker confirms exact filename and pins SHA in `artifacts/m6/gen2_manifest_v2.json`.

This recovers:
- F-S3-2 mu_bdy waiver → substantive close (real wrfout history available)
- F-S4-2 full 1h/6h/12h/24h surface RMSE → unblocked
- M6-S8 operational comparison can use full multi-lead data, not interior-only fallback

## Sanitize-OFF measurement (F-S4-3 binding)

M6-S4 must instrument Tier-2 conservation BEFORE `sanitize_state` runs. Two approaches acceptable:

- (a) Pre-sanitize tap in `coupling/driver.py.run_forecast_segment` that records State pytree at every step before `sanitize_state` clip; Tier-2 conservation computed on pre-sanitize state
- (b) Sanitize-OFF parallel forecast (separate scan body without finite-state guard); blow-up acceptable; Tier-2 conservation computed during the safe-window leads (1h pre-blowup)

Worker chooses based on implementation cost; both are acceptable.

## Consequences

**Positive**:
- Coupled surface diagnostics become operationally meaningful for M6-S8 RMSE
- Tier-2 conservation no longer measures the guard
- Full 1h-24h operational-delta sweep achievable

**Negative**:
- State pytree grows by 4-5 leaves (~few MB at d02 = 160×67); negligible vs 130 MB temp/step
- M6-S4 wall-time estimate increases ~4-8h for the state extension + re-pin + sanitize-OFF instrumentation
- M6-S5/S6/S7/S8 inherit re-pin (same Gen2 fixture path; small adapter change to their contracts)

## Cross-references

- ADR-002 (state layout): SoA pytree pattern preserved
- ADR-007 (precision policy): FP32-gated fields per Authorization Matrix
- ADR-010 (M6 coupled state extension): this is the second authorized amendment
- ADR-011 (shared I/O): Gen2 re-pin uses the same accessor; no I/O change needed
- ADR-012 (surface layer scope) + ADR-013 (Noah-MP subset): M6-S3 deliverables this ADR enables
- M6-S3 reviewer report §M6-S4 Binding (F-S4-1/2/3) is the binding spec

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 16:00
