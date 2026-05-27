# Pinned-Grid Analysis — M7 Gen2 Corpus Scout

**Sprint**: `2026-05-27-m7-gen2-corpus-scout` (AC2)
**Scope**: classify every Gen2 d02 run under `/mnt/data/canairy_meteo/runs/wrf_l[23]/` against the M7 Tier-4 pinned-grid requirement; identify the modal d02 shape; confirm where the pinning is enforced in the codebase.
**Source artifact**: `full_gen2_inventory.json` (this sprint).

## Pinned reference (from `artifacts/m6/gen2_manifest_v2.json`)

| Domain | mass `(ny, nx)` | staggered `(e_sn, e_we)` | source wrfout |
|---|---|---|---|
| d01 | `(59, 93)`  | `(60, 94)`  | `20260520_18z_l3_24h_20260521T045847Z` |
| d02 | `(66, 159)` | `(67, 160)` | `20260520_18z_l3_24h_20260521T045847Z` |
| d03 | `(75, 93)`  | `(76, 94)`  | `20260520_18z_l3_24h_20260521T045847Z` |
| d04 | `(60, 69)`  | `(61, 70)`  | `20260520_18z_l3_24h_20260521T045847Z` |
| d05 | `(57, 69)`  | `(58, 70)`  | `20260520_18z_l3_24h_20260521T045847Z` |

The reference manifest itself is sourced from the held-out cycle `20260520_18z` (cycle dir cleaned out in current snapshot — see AC1; only `wrfbdy_d01` and metadata may survive). The grid is what M6.5-D1 pinned and what `compute_rmse_against_gen2` will refuse if shapes diverge (`src/gpuwrf/validation/data_quality.py:374`).

## Where "pinning" is enforced

The codebase enforces the pinning in **three implicit places**, not via an explicit shape check inside `iter_complete_runs`:

1. **Selection by directory pattern** — `src/gpuwrf/validation/tier4_probtest.py:34`
   `RUN_DIR_RE = re.compile(r"^(?P<cycle>\d{8}_\d{2}z)_l3_24h_(?P<created>\d{8}T\d{6}Z)$")`
   → silently filters out all `wrf_l2` and any rerun-suffixed dirs (e.g. `..._l2rerun_l3_24h_...`).

2. **Selection window** — `src/gpuwrf/validation/tier4_probtest.py:27-28`
   `DEFAULT_ENDING_CYCLE = "20260520_18z"`; `DEFAULT_HELDOUT_CYCLE = "20260519_18z"` are baked in. Members with `cycle > 20260520_18z` are dropped (`select_historical_members:128`).

3. **Required-leads completeness** — `has_required_history_files(...)` (`tier4_probtest.py:147`) demands a `wrfout_d02_{LEAD}` file for each of `(0, 6, 12, 24)` h.

4. **Shape enforcement** — only at compare time:
   - `tier4_probtest.derive_probtest_tolerances` reduces with a mask matching `data.shape[1:]` (`:355`), so any off-grid member would surface as a numpy broadcast error before tolerance freeze.
   - `compute_rmse_against_gen2` raises `ValueError(f"{field} shape mismatch: forecast {predicted.shape} vs Gen2 {truth.shape}")` (`data_quality.py:375`). Adapter version is bound to M6.5-D1.
   - `data_inventory.inventory_run(...)` records `metadata.dimensions` but does **not** filter by shape (`:188` `complete` ignores grid). This explains how the old (66, 120) cycle still flags `complete=True` in raw inventory output until the harness rejects it downstream.

## d02 classification (every run with any d02 file)

| Run ID | Path tree | d02 files | mass `(ny, nx)` | Status | Tier-4 eligible window? |
|---|---|---|---|---|---|
| `20260428_18z_l3_24h_20260525T221139Z` | wrf_l3 | 3 | `(66, 159)` | `MISSING_TIMES` (only 18-20Z, +0..+2 h of 24) | yes-cycle, no-completeness |
| `20260509_18z_l3_24h_20260511T190519Z` | wrf_l3 | 25 | **`(66, 120)`** | `WRONG_GRID` (old d02 layout pre-pinning) | yes-cycle, no-grid |
| `20260509_18z_l3_24h_20260512T154354Z` | wrf_l3 | 3 | `(66, 159)` | `MISSING_TIMES` (superseded by 0511T190519Z; only 18-20Z retained) | yes-cycle, no-completeness |
| `20260521_18z_l3_24h_20260522T072630Z` | wrf_l3 | 9 | `(66, 159)` | `MISSING_TIMES` (18-20Z plus 00-02Z, 18-23Z; gap 03-17Z) | no-cycle (> end) |
| `20260521_18z_l3_24h_20260522T133443Z` | wrf_l3 | 25 | `(66, 159)` | `PINNED_GRID_COMPLETE` | **no-cycle** (cycle 20260521 > 20260520) |
| `20260524_18z_l3_24h_20260525T225640Z` | wrf_l3 | 25 | `(66, 159)` | `PINNED_GRID_COMPLETE` | **no-cycle** (cycle 20260524 > 20260520) |
| `20260525_18z_l3_24h_20260526T221207Z` | wrf_l3 | 8 | `(66, 159)` | `MISSING_TIMES` (still spinning up at scout time; the +1 hour beyond inventory snapshot) | no-cycle |
| `20260509_18z_l2_72h_20260511T190519Z` | wrf_l2 | 73 | `(66, 120)` | `WRONG_PATTERN` + `WRONG_GRID` (l2 72-h run; old grid) | filtered by `RUN_DIR_RE_L3` |
| `20260521_18z_l2_72h_20260522T133443Z` | wrf_l2 | 20 | `(66, 159)` | `WRONG_PATTERN` + `MISSING_TIMES` | filtered |
| `20260524_18z_l2_72h_20260525T225640Z` | wrf_l2 | 73 | `(66, 159)` | `WRONG_PATTERN` (l2 72-h run; pinned grid; full 73 h) | filtered |

**All 51 other run directories under `wrf_l2/` + `wrf_l3/` have zero retained d02 wrfouts** — Canairy Gen2's existing retention policy strips `wrfout_d02_*` after run completion, leaving only `wrfbdy_d01`, `namelist.*`, support tables, and `metadata.json`. Confirmed by counting `wrfout_d02_` matches across all 61 directories (10 dirs have ≥1 d02 file, the other 51 are stripped).

## Modal d02 shape

Across the 10 d02-bearing runs:
- `(66, 159)` mass = the pinned new-grid layout → 8 runs (3 complete, 5 partial)
- `(66, 120)` mass = the pre-pinning legacy layout → 2 runs (both complete, both grid-incompatible)

**Modal pinned shape = `(66, 159)`**, matching `gen2_manifest_v2.json`. Two of the three complete `(66, 159)` runs are L3 24h members (the M7 target shape); the third is the L2 72h `20260524` run, which the harness pattern filter excludes.

## What blocks the M7-S0 corpus gate

| Requirement | Reality |
|---|---|
| 10 members | 2 pinned-grid-complete L3 24h members exist anywhere on disk |
| matching `RUN_DIR_RE_L3` | yes for both surviving complete L3 members |
| cycle ≤ `20260520_18z` | **NO — both are cycle 20260521 / 20260524, *after* the frozen ending cycle** |
| heldout `20260519_18z` excluded | irrelevant — that cycle dir is wrfout-empty |
| required leads `(0, 6, 12, 24)` h present | yes for both complete L3 24h members |
| grid `(66, 159)` | yes for both |

→ `tier4_eligible_pinned_complete_runs = 0` (see `aggregate` in `full_gen2_inventory.json`).

The corpus is therefore "BLOCKED_CORPUS" against the pinned-default Tier-4 harness, and would still be `BLOCKED_CORPUS` even if the harness's hard-coded `DEFAULT_ENDING_CYCLE` were lifted (only 2 pinned-grid-complete L3 24h members vs. the requested 10).

## Cross-reference with adapter

`compute_rmse_against_gen2(gpu_forecast_state, gen2_wrfout_path, valid_time, fields=("U10","V10","T2"))` (`src/gpuwrf/validation/data_quality.py:350`) is shape-strict: any `(66, 120)` member would raise on the first compare. It does no surrogate fallback, which matches the M7-S0 contract's "no surrogate grids" hard rule. The adapter version is whatever M6.5-D1 froze; no rework needed for this sprint.

## Conclusion

The pinning logic is correct and well-defined; the corpus is the binding constraint. Two distinct fixes are required:

1. **Cycle window**: the `DEFAULT_ENDING_CYCLE` literal must advance to at least `20260524_18z` once the corpus reaches gate size, otherwise the surviving complete members aren't even visible.
2. **Member count**: even after that bump, only 2 of the 10 required pinned-grid-complete L3 24h members exist. Recovery options listed in `recovery_candidates.md` / `recommendation.md`.
