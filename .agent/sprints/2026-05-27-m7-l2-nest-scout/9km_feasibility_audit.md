# 9 km d01 feasibility audit

**Sprint**: `2026-05-27-m7-l2-nest-scout` AC2
**Branch**: `tester/opus/m7-l2-nest-scout`
**Date**: 2026-05-27
**Scope**: can the GPU port produce a forecast on the L2 d01 9 km parent grid (94×60 mass = 93×59, 45 vertical levels), or are there hidden 3 km assumptions?

## Bottom line

`dx_m` is **mostly** a configurable parameter — the dycore inherits it from `GridSpec.projection.dx_m`, which `Gen2Run.grid()` reads from `wrfinput` attributes / namelist. Map factors (`MAPFAC_UX`/`MAPFAC_VY`/etc.) and hybrid-η coefficients (`C1F`/`C2F`/`FNM`/`FNP`/...) are loaded per-domain from `wrfinput`, so they automatically pick up the 9 km projection.

There are **three real risks** specific to 9 km, none of which are dycore-stopping:

1. **Silent 3 km default in `physics/surface_layer.py`** — drops one diffusion term at 9 km, slightly underestimates wind. (Numerical, not crash.)
2. **`MAX_LIFTED_DYCORE_DT_S = 12.0` cap** — forces dt ≤ 12 s on a grid where WRF uses 18 s. Run is ~1.5× longer than a CFL-tight 9 km run would be. (Performance, not correctness.)
3. **`build_replay_case` requires wrfout history** — its boundary leaves come from hourly wrfout side-packs, not from `wrfbdy_d01`. There is no production code path that ingests `wrfbdy_d01` directly. (Architecture gap — see AC3.)

The grid plumbing itself is clean. We did **not** find any hard-coded `3000.0` literal on the dycore path; all dynamics modules pass `dx_m` from `Projection`.

## Findings (file:line)

### F1 — `Projection.dx_m` is the canonical handle (clean)

- `src/gpuwrf/contracts/grid.py:25` — `Projection.dx_m: float` field; carried through `GridSpec`.
- `src/gpuwrf/io/gen2_accessor.py:377` — `Gen2GridSpec(dx_m=float(self._nml_list_value(domains_nml, "dx", index, attrs.get("DX"))))`. So `dx_m` is read from the `&domains` namelist's `dx` array indexed by domain number (d01 → index 0 → 9000.0).
- `src/gpuwrf/io/gen2_accessor.py:392-395` — `parent_id`, `parent_grid_ratio`, `i_parent_start`, `j_parent_start` all read identically. The accessor already knows about nest topology.
- `src/gpuwrf/runtime/operational_mode.py:254` — `dx_m=namelist.grid.projection.dx_m` plumbed into `_horizontal_pressure_gradient_tendencies`.
- `src/gpuwrf/runtime/operational_mode.py:374, 521` — `AcousticCoreConfig(dx=..., dy=...)` populated from the same projection.
- `src/gpuwrf/dynamics/acoustic.py:18` and `src/gpuwrf/dynamics/advection.py:24` — helper accessors take `GridSpec` and return `grid.projection.dx_m`. Reduced-M4 path is also clean.
- `src/gpuwrf/dynamics/acoustic_wrf.py:64,335,427` — `acoustic_wrf` jits with `dx_m`/`dy_m` as `static_argnames`. Cache-keyed on the value; 9 km will compile its own cache entry, not silently reuse 3 km's.

**Assessment**: no hard-coded 3 km in dycore. ✅

### F2 — `MAPFAC_*` and hybrid-η are wrfinput-sourced per domain (clean)

- `src/gpuwrf/dynamics/metrics.py:41-80` — `load_wrfinput_metrics(path)` reads `MAPFAC_MX/MY/UX/UY/VX/VY`, `C1F/C2F/C3F/C4F`, `CF1/CF2/CF3`, `FNM/FNP`, `ZNU/ZNW`, `HGT` from whatever `wrfinput` path you pass.
- `src/gpuwrf/integration/d02_replay.py:347` — `metrics = load_wrfinput_metrics(run.wrfinput_file(domain))`. Pass `domain="d01"` and you get the 9 km metrics. The function does not encode the domain anywhere; it just opens the file.

The d01 `wrfinput_d01` files audited in AC1 do contain all required variables (329 vars, full WRF schema). So the metric load is a drop-in for d01.

**Assessment**: clean. The d02-replay metric loader is domain-agnostic by construction. ✅

### F3 — `physics/surface_layer.py:162` falls back to 3 km **silently** ⚠️

```python
# src/gpuwrf/physics/surface_layer.py:162
dx_m = jnp.maximum(_as_surface(_field(state, "dx_m", DEFAULT_DX_M), shape), 1.0)
# DEFAULT_DX_M = 3000.0 (src/gpuwrf/physics/surface_constants.py:20)
```

The surface layer reads `dx_m` as an **optional** state field. `gpuwrf.contracts.state.State` does **not** declare a `dx_m` field (verified by grep — no occurrences in `state.py` / `precision.py` / `operational_state.py`), and the column-view wrapper `_SurfaceColumnState` in `coupling/physics_couplers.py:288-302` does not set it either.

Effect at 9 km: `vsgd = 0.32 * jnp.maximum(dx_m / 5000.0 - 1.0, 0.0) ** (1/3)` (line 181). With `dx_m = 3000.0` (the silent default) `vsgd = 0`. With true 9 km, `vsgd ≈ 0.297` m/s, an additional subgrid wind contribution that gets folded into `wspd` and changes drag, fluxes, etc.

**Severity**: low–medium. Single physics term; affects coastal/island wind drag at 9 km only; no crash. Easy fix: thread `grid.projection.dx_m` into `surface_adapter` and stash on the column-state, **or** at minimum lift `DEFAULT_DX_M` into a runtime constant set by namelist. Either is small.

**Note**: this audit is read-only. The fix belongs in a follow-up sprint.

### F4 — `MAX_LIFTED_DYCORE_DT_S = 12.0` global cap ⚠️

- `src/gpuwrf/coupling/driver.py:37` — `MAX_LIFTED_DYCORE_DT_S = 12.0`.
- `src/gpuwrf/coupling/driver.py:271` — `validate_lifted_coupled_dt` raises if `dt_s > 12.0`.

WRF's L2 namelist uses `time_step = 18` for d01 (9 km). Our GPU port refuses any `dt_s > 12 s`. For a 9 km d01 24 h run, that is 24×3600/12 = 7200 steps vs WRF's 4800 — 50 % more work than necessary.

**Severity**: performance only, not correctness. The cap was tuned for the 3 km path. Two options:

- Make `MAX_LIFTED_DYCORE_DT_S` a per-grid scalar tied to acoustic CFL: `dt_max = K · dx / c_s` with `K ≈ 0.6` and `c_s ≈ 340 m/s`. At 9 km this gives ~16 s, safely > 18 s if we relax K slightly.
- Or simply raise it to 18 s after a 9 km bench shows stability.

Either is a small follow-up; backfills can start at dt=12 s and we get a 9 km forecast in proportionally more wall-clock.

### F5 — `build_replay_case` is wrfout-history-driven, not wrfbdy-driven ⚠️⚠️

- `src/gpuwrf/integration/d02_replay.py:341` — `grid = run.grid(domain).as_grid_spec()` requires `run.history_files(domain)[0]` to exist (i.e., at least one wrfout file).
- `src/gpuwrf/integration/d02_replay.py:274-277` — `history_count < 2 → FileNotFoundError("fewer than two wrfout_{domain} history files")`.
- `src/gpuwrf/integration/d02_replay.py:288-319` — `load_history_boundary_leaves` reads `U/V/T/QVAPOR/PH/PHB/MU/MUB` from hourly wrfouts and packs them into side-history boundary leaves.

There is **no** code path that ingests `wrfbdy_d01` (the lateral-tendency NetCDF that WRF normally consumes). All L3-pipeline boundary forcing has been a "borrow wrfouts from the parent CPU WRF run and slice the edges". That is fine for d02 because Gen2 always produced d01 wrfouts. For L2 backfills, Gen2 stripped d01 wrfouts on 24/28 days (see AC1 inventory) — meaning today, on most L2 days, neither L3-style d02 replay nor a hypothetical d01 standalone run can use the existing build_replay_case.

For L2 d01 the situation is worse: there is **no** d01 parent — `wrfbdy_d01` is the only boundary source, and we cannot ingest it.

**Severity**: high for any production "AIFS-driven d01" path. See AC3 for full discussion.

### F6 — `physics/surface_constants.py:20` and `io/wrfout_writer.py:697` ⚠️ (informational only)

- `surface_constants.py:20` — `DEFAULT_DX_M = 3000.0`. Only used by F3 above.
- `wrfout_writer.py:697` — `dx_m = float(_lookup(projection, "dx_m", _lookup(namelist, "dx", 3000.0)))`. Final fallback if projection AND namelist both miss; in practice projection will always supply it. Harmless.
- `contracts/grid.py:469` — `canary_3km_template` test helper; not on operational path.

### F7 — Vertical level count and η profile are per-domain

- L2 d01 wrfinput has `bottom_top=44, bottom_top_stag=45` (45 η levels — same as d02; same as L3). `ZNU/ZNW/C1F/C2F/...` come from `wrfinput_d01` and will be identical-or-near-identical to d02 because both share `e_vert=45, p_top_requested=5000` in the L2 namelist.

**Assessment**: no vertical-level surprises. ✅

## Grid sanity table — L2 d01 vs L2 d02 (header-read from wrfinput)

| Quantity | L2 d01 | L2 d02 | Notes |
|---|---|---|---|
| Mass shape (z, y, x) | (44, 59, 93) | (44, 66, 159) | Both fit comfortably on RTX 5090 |
| Staggered shape | (45, 60, 94) | (45, 67, 160) | C-grid |
| DX, DY (m) | 9000, 9000 | 3000, 3000 | Read from attrs + namelist |
| MAP_PROJ | 1 (Lambert) | 1 (Lambert) | Same projection family |
| TRUELAT1/TRUELAT2 | 25, 30 | 25, 30 | Identical projection |
| MOAD_CEN_LAT, STAND_LON | 28.3, -16.4 | 28.3, -16.4 | Same |
| PARENT_GRID_RATIO | 1 | 3 | d02 sits inside d01 by ×3 refinement |
| I/J_PARENT_START | 1, 1 | 24, 20 | d02 anchor in d01 |
| WRF time_step (s) | 18 | 6 | Our GPU cap forces 12 / 6 |

## Recommendation

**9 km d01 is feasible from a dycore-physics standpoint with one small known artifact** (F3 surface-vsgd term) **and two small performance-ish patches** (F3 + F4). The blocking concern is **F5**: the existing GPU pipeline has no AIFS / wrfbdy ingestion path. That gap is addressed in AC3.

For the user's "publish + start backfills today" timeline, the only L2 d01 path that works **without new code** is to feed the GPU d01 from a CPU-WRF d01 wrfout history (treating CPU-WRF d01 as the boundary source for the GPU d01 run) — which is a regression to "GPU replays a CPU WRF result" rather than "GPU drives the operational forecast". The honest call is `BACKFILL_NEEDS_NEW_CODE` for the 9 km parent. See `nest_backfill_design.md` for the staged plan.
