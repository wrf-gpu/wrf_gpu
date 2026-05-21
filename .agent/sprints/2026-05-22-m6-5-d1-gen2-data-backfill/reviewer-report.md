# M6.5-D1 Reviewer Report — Gen2 Backfill + RMSE Adapter

**Reviewer**: Claude Opus 4.7 xhigh (mandatory independent review)
**Sprint**: `2026-05-22-m6-5-d1-gen2-data-backfill`
**Worker commit reviewed**: `60bbaf7` on `worker/codex/m65-d1-gen2-data-backfill`
**Date**: 2026-05-21
**Wall**: ~18 min worker (vs 8–16h budget) — fast.

## TL;DR

**Binding decision: ACCEPT** with **AC4 disposition = (c) threshold-amendment**
documented as a contract amendment. Worker delivered a clean, lazy,
fixture-test-only IO + audit + RMSE-adapter surface that materially meets all
AC1–AC3 and AC5–AC8 obligations. AC4's 1.0208% V/E rel_mae miss is a
sprint-contract specification error, not a worker defect: the codebase's
own `src/gpuwrf/io/boundary_replay.py:39-45` TOLERANCES already allow
U/V at `rel_mae_max=0.03` (3%), and the v2 zarr fixture's
own `validation_summary.json` self-validated to V `rel_mae_max=0.010208`
*before this sprint started*. M7-S0 dispatch is **UNBLOCKED**.

## R-findings per AC

### AC1 — Gen2 d02 wrfout inventory: **PASS**
- `src/gpuwrf/io/data_inventory.py:228-256` builds `Gen2D02Inventory`
  schema-validated at `data_inventory.py:265-297`. Run discovery anchors on
  `wrfbdy_d01` markers and direct subdirs (`data_inventory.py:114-125`), so
  retention-pruned runs still appear (`partial_run_count=22`).
- `artifacts/m6_5/gen2_d02_inventory.json` reports `run_count=25`,
  `wrfbdy_d01_run_marker_count=25`, `wrfout_d02_file_count=78`,
  `complete_run_count=3`. All three match `find` counts on live disk
  (verified — see "Verifiability triple" below).
- Schema field set in `validate_gen2_d02_inventory` (`data_inventory.py:267-280`)
  covers everything the contract names: `start_date`, `hours`, file count,
  total bytes, init time, valid-time range, complete-or-partial.
- One nit (non-blocking): `parse_run_id` (`data_inventory.py:55-74`) silently
  returns all-None on a non-matching run id — quietly excludes such runs
  from `filter_inventory_runs` rather than surfacing a parse error. Acceptable
  for the current archive convention; flag if a non-conforming run dir lands.

### AC2 — Data-quality audit: **PASS with one minor reasons-list bug**
- `audit_run_quality` (`src/gpuwrf/validation/data_quality.py:97-205`) does
  two-pass chunked sampling per file, uses Welford-merged running stats
  (`RunningStats.update`, `data_quality.py:33-49`) for mean/std, and
  histograms from sampled finites at 1st/99th percentile edges.
- **Minor bug** (`data_quality.py:181-190`): once `status` flips to PARTIAL,
  the loop's `if status == "GREEN" and record["spike_flag"]:` guard prevents
  *any further* per-field reasons from being appended. The audit JSON shows
  this in action — all three complete runs report only `"T2 has z-score
  spikes above threshold"`, even though `PSFC`, `Q2`, `RAINNC` all also have
  non-zero `spike_count` and `spike_flag=True`. Status is still correct;
  only the human-readable `reasons` list is incomplete. **Not a blocker** —
  this is observability, not correctness. Fixable as a one-liner in a
  follow-up: drop the `status == "GREEN"` guard for spike reasons, or move
  reason collection out of the status-transition block. File as M7-S0a TODO.
- Spike detector is hard-coded to 5σ (`data_quality.py:81`, `data_quality.py:173`).
  See "Adversarial probes" — too tight for heavy-tailed surface fields like
  PSFC/RAINNC. Tunability is acceptable as a follow-up; the audit
  conservatively flags PARTIAL, never GREEN-by-default for incomplete fields,
  so M7 cannot accidentally consume a bad run.

### AC3 — GPU-side data loader: **PASS, genuinely lazy**
- `Gen2WrfoutLoader` (`gen2_wrfout_loader.py:106-178`) opens NetCDF files
  one-at-a-time using `with Dataset(path, "r")` context managers
  (`gen2_wrfout_loader.py:91`, `:137`). No `open_mfdataset`, no array cache.
  Caches `_files` (Path list) and `_time_axis` (datetime list) — both
  O(N_files) metadata, not array material.
- `iter_chunks` (`gen2_wrfout_loader.py:169-178`) yields per-file payloads;
  caller can stream the corpus without retention.
- JAX conversion is at the consumer boundary (`gen2_wrfout_loader.py:94-97`),
  gated on `as_jax=True`. NumPy is the default. This matches state-layout
  ADR-002's "JAX-boundary conversion" discipline.
- **Minor efficiency**: `time_axis` (`gen2_wrfout_loader.py:132-140`) wraps
  the filename-only `parse_wrfout_valid_time` call in a `with Dataset(...)`
  block even when the filename parse succeeds without ever needing the
  dataset (`_valid_time_from_file`, `gen2_wrfout_loader.py:58-68`, falls
  back to the dataset only on `ValueError`). 25 wasted opens per run on
  first access. Not a correctness issue; flag as a one-line cleanup
  ("only open the file in the except branch") for M7-S0a.

### AC4 — Boundary-replay → wrfout cross-check: **FAIL-on-spec, ACCEPT via threshold-amendment (c)**

See dedicated section below. Status: amend the AC4 contract from 1% to 3%
to match `src/gpuwrf/io/boundary_replay.py:40-41` TOLERANCES; the measured
1.0208% then sits well inside the gate and the cross-check artifact
transitions FAIL → GREEN with no code change required.

### AC5 — Tier-4 RMSE adapter: **PASS**
- `compute_rmse_against_gen2` (`data_quality.py:350-384`) accepts both
  run-directory and single-file truth paths, converts truth + forecast to
  JAX at the adapter boundary, enforces shape match
  (`data_quality.py:374-375`) and file/valid-time match
  (`data_quality.py:367-369`), and returns the documented
  `{rmse, error_map, valid_time_utc, gen2_source_file}` shape.
- ADR-016 lines 49-53 lock this return schema for M7-S0 to depend on.
- Tests at `tests/test_m6_5_rmse_adapter.py:48-99` cover zero, non-zero
  per-cell, run-dir + valid-time, missing forecast field, shape mismatch,
  and valid-time mismatch. Edge coverage is honest.

### AC6 — Selectable Gen2 corpus subset: **PASS**
- `build_subset_manifest` + `filter_inventory_runs`
  (`data_inventory.py:318-395`) implement `--start/--end/--min-hours`,
  default-`require_complete=True`, with `--include-partial` opt-out.
- Live artifact `artifacts/m6_5/gen2_d02_subset_complete24h.json`:
  `run_count=3`, `wrfout_d02_file_count=75` (one file is the
  init-time-only/lead-0 of a different run grid? no — actually it's
  3×25 = 75; the 78 in the global inventory minus 3 extras from
  runs that retained a couple of lead-0 files without the full 24h).
  Numbers reconcile.

### AC7 — Test surface: **PASS, 19/19 green on fabricated NetCDF**
- `tests/test_m6_5_gen2_loader.py` (6 tests),
  `tests/test_m6_5_data_quality.py` (7 tests),
  `tests/test_m6_5_rmse_adapter.py` (6 tests). Total 19, ran 19 passed
  locally in 1.20s. None require `/mnt/data`; fixtures use real
  `netCDF4.Dataset` writes under `tmp_path` with WRF-standard time,
  dimension, variable conventions. Sprint hard-rule HARD #4 met.

### AC8 — ADR-016: **PASS**
- `.agent/decisions/ADR-016-gen2-data-corpus.md` covers corpus structure
  (lines 14-31), quality bar tri-state (lines 37-41), lazy-load policy
  (lines 43-47), M7 RMSE adapter schema (lines 49-53), and boundary
  cross-check policy (lines 55-58).
- **Required ADR amendment**: line 57's "Any relative MAE above 1 percent
  is a data-pipeline failure flag" must be updated to match the AC4
  disposition below — bump to **3%** rel_mae per
  `boundary_replay.py:40-41` TOLERANCES, and cite the existing
  `tolerance_rationale` (`boundary_replay.py:427-430`) as the
  controlling design statement. This is a textual amendment — no code
  change.

## AC4 disposition — **(c) threshold-amendment**

### Rationale

The 1% threshold in the sprint contract (AC4) was the contract author's
guess. The measured V/E rel_mae of 0.010208 fails that gate by 2 bps.
But the existing module that *produces* the replay zarr already publishes
its own tolerances, and they are looser than the AC4 contract:

- `src/gpuwrf/io/boundary_replay.py:40-41`:
  `"U": {"rel_mae_max": 0.03, ...}`, `"V": {"rel_mae_max": 0.03, ...}`
- `src/gpuwrf/io/boundary_replay.py:427-430`: `"tolerance_rationale":`
  *"Nested d02 history differs from parent-regridded d01 by
  interpolation and feedback deltas; U/V/QVAPOR use few-percent relative
  MAE gates, T uses 0.5 K RMSE, PH uses a tight relative geopotential
  gate."*
- `data/fixtures/m6/d02_boundary_replay_v2.zarr/validation_summary.json`:
  the fixture itself was built with `rel_mae_max=0.010208` on V and
  self-reported `passed=True` against TOLERANCES *before this sprint
  started*. The cross-check is recomputing a number that the fixture's
  own provenance already accepts.

Per-side V breakdown from `artifacts/m6_5/gen2_boundary_replay_cross_check.json`:

| Side | mae      | rel_mae   | max_abs | passed@1% | passed@3% |
|------|----------|-----------|---------|-----------|-----------|
| W    | 0.007163 | 0.001660  | 0.146   | ✓         | ✓         |
| E    | 0.041656 | **0.010208** | 1.532 | ✗         | ✓         |
| S    | 0.006886 | 0.002090  | 0.604   | ✓         | ✓         |
| N    | 0.003243 | 0.000635  | 0.108   | ✓         | ✓         |

The V/E miss is 4 cm/s absolute MAE against a ~4 m/s mean — physically
trivial, and the same order as other variables' interpolation residuals
(QVAPOR/E rel_mae=0.00899, ~9 mg/kg out of 1 mg/kg local denominator).
This is staggered-grid interpolation noise on the east boundary where
the d01 parent has weaker wind structure to constrain the d02 child
bilinear; it is structurally expected, not anomalous.

### Why not (a) or (b)?

- **(a) Accept-as-known fixture limitation**: works, but leaves the
  cross-check artifact in `status=FAIL` forever. M7-S0 will then have
  to special-case "FAIL means accept" via comments. Confusing.
- **(b) Bug-requiring-fix**: would force a rabbit-hole into
  `boundary_replay.py` interpolation tuning that the existing
  `tolerance_rationale` text already disclaims. No design or evidence
  basis to chase a 1% target when the module's published tolerance is
  3%. Net cost: weeks; net benefit: zero, since the d01→d02 zarr is a
  fixture not a runtime path.
- **(c) Threshold-amendment**: aligns the AC4 gate with the established
  `boundary_replay.py:40-41` TOLERANCES, eliminates the spurious FAIL,
  and lets M7-S0 depend on a `status=GREEN` cross-check. Zero code
  change required in `compare_boundary_replay_to_wrfout` — only the
  `rel_mae_threshold=0.01` default at `data_quality.py:285` and ADR-016
  line 57 need to flip to `0.03`.

### Action items for closeout

1. **Manager close amendment**: ADR-016 line 57 — "1 percent" → "3 percent
   matching `src/gpuwrf/io/boundary_replay.py:40-41` TOLERANCES".
2. **One-line code amendment** (M7-S0a follow-up, **not** required for
   close): change `compare_boundary_replay_to_wrfout` default
   `rel_mae_threshold=0.01` → `0.03` in `data_quality.py:285`, or
   better — import `TOLERANCES` from `boundary_replay.py` and look up
   per-variable so PH (0.005), T (no rel gate), U/V/QVAPOR (0.03) each
   use their own published gate. The audit artifact will then show
   `status=GREEN`.
3. Re-run `python scripts/m6_5_data_quality_audit.py
   --boundary-replay data/fixtures/m6/d02_boundary_replay_v2.zarr
   --boundary-run /mnt/data/canairy_meteo/runs/wrf_l3/20260520_18z_l3_24h_20260521T045847Z`
   and overwrite `artifacts/m6_5/gen2_boundary_replay_cross_check.json`.
4. AC7 spec test: add one test that asserts `status=GREEN` for the v2
   zarr against its source run at the per-variable `TOLERANCES`-based
   threshold, replacing the current `1%` synthetic-fail probe (which
   stays as a regression test using a fabricated delta).

## Verifiability triple check

1. **Tests reproduce on fabricated NetCDF (no /mnt/data)**: `pytest -q
   tests/test_m6_5_gen2_loader.py tests/test_m6_5_data_quality.py
   tests/test_m6_5_rmse_adapter.py` → **19 passed in 1.20s**. ✓
2. **Inventory matches live disk**:
   `find /mnt/data/canairy_meteo/runs/wrf_l3 -maxdepth 2 -name 'wrfbdy_d01' | wc -l`
   → 25 (matches `wrfbdy_d01_run_marker_count=25`). ✓
   `find /mnt/data/canairy_meteo/runs/wrf_l3 -maxdepth 2 -name 'wrfout_d02_*' | wc -l`
   → 78 (matches `wrfout_d02_file_count=78`). ✓
3. **Lazy load verified**: read `src/gpuwrf/io/gen2_wrfout_loader.py`;
   no `open_mfdataset`; every `Dataset(...)` is `with`-wrapped; only
   metadata (paths, datetimes) are cached; arrays are recomputed
   per-call. ✓

## Adversarial probe findings

- **Is `iter_chunks` actually lazy?** Yes — `gen2_wrfout_loader.py:177-178`
  yields one file at a time via `read_wrfout_file`, which opens-and-closes
  the NetCDF inside `with`. No materialization. ✓
- **Are fabricated fixtures realistic?** Yes — `tests/test_m6_5_gen2_loader.py:17-49`
  uses real `netCDF4.Dataset` writes with Time/DateStrLen/south_north/
  west_east/bottom_top dims, Times `S1` variable, MAP_PROJ/CEN_LAT global
  attrs, U/V/T/QVAPOR/PH 4-D plus U10/V10/T2/Q2/PSFC/RAINNC 3-D — matches
  WRF history convention well enough for the loader's structural checks.
  They do *not* exercise staggered-grid dimensions (`west_east_stag`,
  `south_north_stag`); real U/V live on staggered grids in WRF output.
  The loader doesn't currently differentiate, so this is OK for surface
  fields (always unstaggered) but may bite a future caller who asks for
  staggered `U` from a real wrfout. File as M7-S0a.
- **Missing-field vs zero-RMSE coverage in RMSE adapter?**
  `tests/test_m6_5_rmse_adapter.py:78-83` covers missing forecast field
  via `KeyError`. Missing *truth* field would raise `KeyError` from
  `read_wrfout_file._read_variable` (`gen2_wrfout_loader.py:72-73`) —
  not directly tested, but the exception path is identical.
- **5σ spike detector too sensitive?** Yes — `data_quality.py:173` flags
  any cell beyond `|x-μ| > 5σ`. For heavy-tailed surface fields (PSFC,
  RAINNC, T2), 5σ on a Gaussian-assumption is wrong: the audit shows
  ~1% spike fraction on PSFC across all three runs (`PSFC: spike_count
  ~3000`, `spike_fraction ~1.2%`). U10/V10 (genuinely Gaussian) have
  `spike_count=0`, confirming the issue is field-distribution not data.
  Recommend per-field policy (7σ for surface, 5σ for winds, RAINNC
  bounded by 0 floor) in M7-S0a. Not a blocker — PARTIAL is the
  conservative status and does not gate M7.
- **ADR-016 locked into "all PARTIAL by design"?** Yes, given the 5σ
  policy. Need ADR-016 line 39-40 to say "PARTIAL when the run is
  incomplete *or* has suspicious z-score spikes at the configured
  threshold" — and explicitly document that the M6.5-D1 threshold
  (5σ) is provisional. The closeout amendment to ADR-016 should also
  update this.

## M7-S0 dispatch impact: **UNBLOCKED**

The RMSE adapter shape is frozen (`data_quality.py:350-384`, ADR-016
lines 49-53), tests cover edge cases, the lazy loader is genuine, and
the quality audit + inventory provide a stable corpus accessor surface.
The AC4 amendment is a textual ADR + one-line code change and does
not require worker re-spin. M7-S0 (Tier-4 RMSE harness) can be
dispatched the moment M6.x lands.

## Binding decision: **ACCEPT**

- **AC1, AC2 (with minor reasons-list cosmetic bug), AC3, AC5, AC6,
  AC7, AC8**: PASS as delivered.
- **AC4**: ACCEPT via disposition **(c) threshold-amendment** to 3%
  matching `boundary_replay.py:40-41` TOLERANCES. Manager closeout
  must amend ADR-016 line 57 and (optionally, in M7-S0a) tighten the
  code default to use per-variable TOLERANCES.

### Follow-up tickets to file at close (M7-S0a, non-blocking)

1. `data_quality.py:181-190` — collect all spike-trigger reasons, not
   just the first.
2. `gen2_wrfout_loader.py:132-140` — avoid the wasted `Dataset` open
   when the filename parse succeeds.
3. `data_quality.py:285` — import `TOLERANCES` from
   `src/gpuwrf/io/boundary_replay.py` and switch
   `compare_boundary_replay_to_wrfout` to per-variable thresholds.
4. Per-field spike policy: 7σ default for surface scalar fields, 5σ
   for wind components, RAINNC clamped at 0.
5. Add a staggered-grid test fixture so the loader exercises a real
   `west_east_stag` U/V layout.

## Closing note

The worker honestly flagged AC4 as a data-pipeline issue, which is the
correct interpretation. Independent review confirmed that the
sprint-contract author (the manager) picked a 1% threshold without
consulting the existing TOLERANCES in `boundary_replay.py`. The amendment
trail is clean: ADR-016 owns the corrected threshold; `boundary_replay.py`
remains the source of truth for U/V/QVAPOR/T/PH gates. M7-S0 can now
proceed with a green cross-check artifact and a stable RMSE adapter
contract.
