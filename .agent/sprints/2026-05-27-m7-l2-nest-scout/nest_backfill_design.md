# L2 one-way-nest backfill pipeline design

**Sprint**: `2026-05-27-m7-l2-nest-scout` AC4 (the main deliverable)
**Scope**: end-to-end daily backfill flow for the L2 configuration (9 km d01 parent + 3 km d02 island nest) on top of the GPU port, with a wall-clock estimate.

## TL;DR pipeline shape

```
                          ┌─────────────────────────────────────────────┐
                          │   STAGE A — CPU pre-processing (existing)    │
 AIFS NetCDF (day D)  →   │   prepare_aifs_pure_forcing.py               │
                          │   ungrib → metgrid                           │
                          │   real.exe   →   wrfinput_d01 (9 km IC)      │
                          │                  wrfinput_d02 (3 km IC)      │
                          │                  wrfbdy_d01 (6 h cadence)    │
                          └─────────────────────────────────────────────┘
                                              │
                          ┌─────────────────── ▼ ─────────────────────┐
                          │   STAGE B — GPU d01 forecast (NEW)         │
                          │   build_d01_replay_case_from_wrfbdy(...)   │
                          │   execute_daily_pipeline(..., domain=d01,  │
                          │                          hours=24 or 72)   │
                          │   Output: hourly wrfout_d01_* (94×60×45)   │
                          └────────────────────────────────────────────┘
                                              │
                          ┌─────────────────── ▼ ─────────────────────┐
                          │   STAGE C — GPU d02 forecast (existing,    │
                          │   re-pointed to L2 dirs)                   │
                          │   build_l2_d02_replay_case(...)  uses      │
                          │     L2-d02 wrfinput as IC                  │
                          │     STAGE-B d01 hourly wrfouts as boundary │
                          │   execute_daily_pipeline(..., domain=d02)  │
                          │   Output: hourly wrfout_d02_* (160×67×45)  │
                          └────────────────────────────────────────────┘
                                              │
                          ┌─────────────────── ▼ ─────────────────────┐
                          │   STAGE D — output post + scoring          │
                          │   wrfout_writer (existing) ×2 domains      │
                          │   forecast_vs_obs vs AEMET (existing)      │
                          │   archive thin_gridded_d01_v1.nc /         │
                          │   thin_gridded_d02_v1.nc                   │
                          └────────────────────────────────────────────┘
```

The only **new code** is Stage B's d01 ingestor (~1 sprint, see `d01_boundary_forcing_audit.md`). Stages A, C, D are already operational.

## Stage A — AIFS → WPS → real.exe (CPU, kept as-is)

**No GPU work here.** Gen2's existing scripts handle this nightly via `~/src/canairy_meteo/Gen2/scripts/run_pipeline_l2.sh`. For a backfill we just point at a historical day.

| Step | Wall-clock estimate | Notes |
|---|---|---|
| AIFS fetch | seconds, network-bound | Already cached for the 28 audited L2 days |
| ungrib | seconds per 6 h slice | CPU-trivial |
| geogrid + metgrid | ~1 minute total | Static geog once + 13 metgrid files per day |
| real.exe | ~30 s for 72 h forcing | Single-threaded; CPU cores 4-31 available |
| **Stage A total** | **≈ 2 minutes per day** | Already running nightly |

All 28 surveyed L2 days have valid `wrfinput_d01`, `wrfinput_d02`, and `wrfbdy_d01` on disk (AC1). For these days Stage A is **already done**; new days require re-running the pipeline.

## Stage B — GPU d01 9 km forecast (NEW, ~1 sprint to ship)

### What ships

A new `gpuwrf.integration.d01_replay.build_d01_replay_case_from_wrfbdy(run_dir, domain="d01")` that returns the same `ReplayCase` shape as the existing `build_replay_case`. The function is a near-copy of `d02_replay.build_replay_case` with two differences:

- IC source: `wrfinput_d01` instead of `wrfout_d02` time-0 snapshot.
- BC source: `wrfbdy_d01` decoded by `gpuwrf.io.boundary_replay.decode_wrfbdy` and packed into `(N_times=13, 4_sides, z, max_side)` leaves with `update_cadence_s=21600.0`.

See `d01_boundary_forcing_audit.md` for the exact helper signatures, schema mapping, and the moisture-species-boundary question.

### Run config for d01

| Parameter | Value | Source |
|---|---|---|
| Grid mass shape (z, y, x) | (44, 59, 93) | wrfinput_d01 header |
| dx_m, dy_m | 9000, 9000 | wrfinput_d01 attrs |
| dt_s (outer) | 12.0 (capped) | `MAX_LIFTED_DYCORE_DT_S` — F4 in 9km audit |
| acoustic_substeps | 6 | downscaled from default 10; 9 km supports it |
| BoundaryConfig | spec_bdy_width=5, spec_zone=1, relax_zone=4, update_cadence_s=21600.0 | L2 namelist |
| top_lid | True | matches WRF L2 `damp_opt=3, zdamp=5000, dampcoef=0.2` semantics |
| Physics | Thompson MP + MYNN PBL + MM5 sfclay + RRTMG | same as d02; same suite already validated |

### Wall-clock estimate (Stage B)

The L3 d02 reference: 1 h GPU warm = **5.71 s** on (44, 66, 159) at dt=10 s (`pipeline_run_20260521.json` / M7-CLOSEOUT). 24 h = ~324.78 s (5.4 min).

d01 cell count is `44 × 60 × 94 = 248,160` vs d02's `45 × 67 × 160 = 482,400` → d01 is **51.4 %** of d02's cell count. With dt cap at 12 s (vs d02's 10 s) and ~38 % faster math per cell but more steps per simulated hour, the wall-clock balance approximates:

```
d01 / d02 ratio = (d01_cells / d02_cells) × (d02_steps / d01_steps)
                ≈ 0.514 × (360 / 300)
                ≈ 0.617
```

So a 24 h d01 GPU run ≈ **0.617 × 324.78 s ≈ 200 s (3.3 min)**. A 72 h d01 run ≈ **10 min**.

These are projections, not measured. The honest call: **plan for ~3-5 min/24h d01 on RTX 5090, measure on first run, revise**.

## Stage C — GPU d02 3 km forecast (existing, repointed to L2)

This is exactly what the parallel sprint `2026-05-27-m7-l2-d02-replay-validation` is validating today. The L2 d02 grid is bit-identical to L3 d02 (`160×67@3km, 45 levels`) so the existing `build_replay_case` already works — it just needs the run dir pointed at an L2 directory and the boundary source pointed at the Stage-B d01 wrfouts (not L3's d01 = L2's d01 already exists on disk if the parallel sprint chose a day where both d01 AND d02 wrfouts are present).

**Caveat from AC1**: only 4/28 L2 days have raw d01 wrfouts on disk (`20260509, 20260521 (failed), 20260524, 20260525`). For the other 24 days the parallel sprint cannot start at Stage C without first running Stage B. So in the **bootstrap** ordering we ship Stage B first, then Stage C consumes Stage B's output.

### Wall-clock estimate (Stage C)

24 h GPU d02 wall **≈ 5.4 min**, **measured** (M7 closeout). For 72 h: ~16 min linear extrapolation.

## Stage D — output writer + scoring (existing)

`gpuwrf.io.wrfout_writer` already emits the 41-variable wrfout subset documented at M7 closeout. `gpuwrf.validation.forecast_vs_obs` already scores against AEMET. Both are domain-agnostic.

### Wall-clock estimate (Stage D)

NetCDF write is I/O-bound, ~5-10 s per snapshot × 73 snapshots × 2 domains ≈ **8-12 min**. Verification vs AEMET adds ~1 min. Note: most of this overlaps Stage C if writes happen incrementally.

## Total daily backfill wall-clock (one day, 24 h forecast)

| Stage | Mode | Wall-clock |
|---|---|---|
| A | CPU (kept) | 2 min (or 0 if pre-staged) |
| B | GPU d01 9 km, NEW | **3-5 min (projected)** |
| C | GPU d02 3 km, existing | 5.4 min (measured) |
| D | CPU writer + verify | ~5 min (mostly overlapped with C) |
| **Total per day (24 h horizon)** | | **≈ 13-17 min**, with all pre-staged inputs |
| **Total per day (72 h horizon)** | | **≈ 30-40 min** (Stage B and C scale ~linearly) |

For a **27-day historical backfill** (the 27 successful L2 days currently on disk), pre-staged inputs:

- 24 h horizon: 27 × 14 min ≈ **6.3 hours** of GPU+CPU work.
- 72 h horizon (what Gen2 actually runs): 27 × 35 min ≈ **15.8 hours** of GPU+CPU work.

This fits inside one overnight window with margin. Anyone hitting "go" on the morning can have the historical backfill complete by next morning, no parallelization needed.

## Concurrency and resource pinning

- GPU: one forecast at a time (RTX 5090 single device). Stages B and C run sequentially per day; days can be sequenced inside one orchestrator loop.
- CPU: `taskset -c 0-3` on the GPU orchestrator (per project convention); cores 4-31 reserved for CPU WRF runs that may be in flight (Stage A for the *next* day).
- Disk: one L2 day's GPU outputs ≈ 3 × (~12 + 21) MB = ~100 MB after `thin_gridded` post-processing; 27 days ≈ 2.7 GB; trivial against `/mnt/data/wrf_gpu2`'s ~1 TB free.
- Memory: d01 fits trivially on RTX 5090 (much smaller than d02; d02 fits with headroom per M7 1km memory audit).

## Sprint sequencing recommendation

| Sprint | Deliverable | Wall-time | Dependencies |
|---|---|---|---|
| **L2.1 — d01 ingest** | `build_d01_replay_case_from_wrfbdy` + 3 helpers + unit tests | ~½ day | wrfbdy decoder (exists), boundary_apply (exists) |
| **L2.2 — d01 ↔ d02 chain** | New script `scripts/m7_l2_daily_pipeline.py` that runs Stage B → wrfout → Stage C in one go; `daily_pipeline.py` configured for both domains | ~½ day | L2.1 |
| **L2.3 — d01 ↔ d02 24 h Tier-4 RMSE** | Tier-4 RMSE on T2/U10/V10 on a single day vs L2 thin_gridded_d02_v1.nc | ~½ day | L2.2 |
| **L2.4 — surface-vsgd fix (optional)** | Thread `dx_m` from grid into `surface_adapter`; remove silent 3 km fallback | ~1 hour | none |
| **L2.5 — dt cap relaxation (optional)** | Make `MAX_LIFTED_DYCORE_DT_S` per-grid or raise to 18 s for d01 | ~1 hour | benchmark from L2.2 |
| **L2.6 — historical batch** | Backfill 27 L2 days × 72 h, archive thin_gridded outputs, verify vs AEMET | overnight | L2.3 |

**Critical path**: L2.1 → L2.2 → L2.3 → L2.6. Total ~1.5 days of focused worker time + one overnight GPU window.

L2.4 and L2.5 are improvements; L2.6 can ship without them. The numbers above assume v0 lands without them and we accept the F3/F4 caveats.

## What this design does NOT do

- Does not change the dycore, physics, halo, RK, acoustic, or wrfout writer.
- Does not introduce any new ADR — d01 ingestion is a data-loading sprint within the existing operational-mode contract.
- Does not promise bitwise parity with WRF d01 at 9 km. Tier-4 RMSE on T2/U10/V10 is the operational gate, same as M6/M7.
- Does not commit to a two-way nest. One-way (parent forces child, child does not feed back) per the L2 namelist `feedback=0`.
- Does not gate publishing on backfilling 27 days first. Publishing the M7 result is independent — the 27-day backfill is operational rollout, not a prerequisite to publication.

## Open question for the principal

The dx_m fallback (F3) and the dt cap (F4) are both small numerical concerns. For a 9 km parent driving a 3 km island nest, the **errors stay at the parent edge**: the d02 boundary nudge uses our d01 wrfout, which slightly underestimates wind drag. Whether this matters for AEMET scoring is empirical. **Recommendation**: ship v0 with the caveats, measure RMSE on the first 3 days, then prioritise L2.4/L2.5 if the numbers warrant.
