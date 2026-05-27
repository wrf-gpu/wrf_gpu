# Tester Report — M7 Gen2 Corpus Scout

**Sprint**: `2026-05-27-m7-gen2-corpus-scout`
**Role**: opus-tester (research scout, no code, no CPU WRF jobs, no writes to `/mnt/data/canairy_meteo`)
**Branch**: `tester/opus/m7-gen2-corpus-scout`
**Worker report**: none — this is a research-only scout sprint where the tester is the primary deliverable producer per the sprint contract (AC1-AC5 all belong to the tester).
**CPU pinning**: `taskset -c 0-3` for the netCDF header reader (see `_scout_inventory.py`).
**Did not touch**: any file under `/mnt/data/canairy_meteo/**`; any file under `src/`; any governance file.

## Acceptance summary

| AC | Deliverable | Status |
|---|---|---|
| AC1 | `full_gen2_inventory.json` | PRESENT, 61 run dirs catalogued, 822 wrfout files all-domains, schema `M7Gen2CorpusScoutInventoryV1` |
| AC2 | `pinning_analysis.md` | PRESENT, modal pinned d02 shape `(66, 159)` confirmed, harness pinning enforcement located at three sites in `tier4_probtest.py` + `data_quality.py` |
| AC3 | `recovery_candidates.md` | PRESENT, all 10 d02-bearing runs + 51 stripped dirs classified by recovery class |
| AC4 | `recommendation.md` | PRESENT, recommended path = Option D (retention + targeted reruns) + bounded Option A bridge; B/C rejected |
| AC5 | `tester-report.md` | THIS FILE |

All proof objects live under `.agent/sprints/2026-05-27-m7-gen2-corpus-scout/`. Only the inventory script `_scout_inventory.py` reads `/mnt/data/canairy_meteo` — it does header-only `netCDF4.Dataset` opens, never iterates variable arrays.

## Key numbers (extracted from `full_gen2_inventory.json["aggregate"]`)

```json
{
  "run_dir_count": 61,
  "wrfout_file_count_all_domains": 822,
  "d02_runs_with_any_files": 10,
  "d02_complete_24h_hourly_runs": 5,
  "d02_pinned_grid_complete_24h_runs": 3,
  "tier4_eligible_pinned_complete_runs": 0,
  "tier4_required_member_count": 10,
  "tier4_corpus_gate": "BLOCKED_CORPUS"
}
```

The three pinned-grid-complete 24h members:

1. `20260521_18z_l3_24h_20260522T133443Z` (wrf_l3, 25 d02 files, `(66, 159)`) — out of window (cycle > `20260520_18z`).
2. `20260524_18z_l3_24h_20260525T225640Z` (wrf_l3, 25 d02 files, `(66, 159)`) — out of window.
3. `20260524_18z_l2_72h_20260525T225640Z` (wrf_l2, 73 d02 files, `(66, 159)`) — wrong directory pattern (`l2_72h` ≠ `l3_24h`); sibling of #2.

→ Inside the harness's default window (`cycle ≤ 20260520_18z`, `≠ 20260519_18z`, matching `RUN_DIR_RE_L3`): **zero** eligible pinned-grid-complete L3 24h members. The harness will emit `BLOCKED_CORPUS` exactly as the predecessor M7-S0 dispatch did.

## Adversarial / edge-case checks performed

The "tester" role normally hardens the worker's implementation. Because this sprint is research-only, "trying to break the implementation" maps to **trying to break my own conclusions** — i.e., looking for alternative corpora or eligibility paths the surface inventory might miss. Each check below is recorded so the manager can audit it.

1. **Did I miss runs by globbing only the obvious paths?**
   - Walked the entire `wrf_l2/` + `wrf_l3/` trees (61 directories). Spot-checked sibling trees `wrf_hindcast/` (37 dirs, all wrfout-empty — only preflight summaries; verified by `find -maxdepth 4 -name 'wrfout_d*'`), `wrf/` (only a `lender_staging` stub), and `gen2_archive/teacher_l3/` (234 wrfout files but all pre-2026 historical, almost certainly a different pinned grid; not in the contract's scope; flagged off-scope in `recovery_candidates.md`).
   - **Net new found**: zero in-scope.

2. **Did I miss runs hidden in non-canonical run dir names?**
   - `RUN_DIR_RE_L3` is strict (`^\d{8}_\d{2}z_l3_24h_\d{8}T\d{6}Z$`). I separately accepted any directory containing `wrfout_d02_*` and recorded a `matches_l3_24h_pattern` flag per run. The L2 72h and the `l2rerun_l3_24h_...` dirs are non-canonical and the harness pattern filter rejects them. Confirmed by listing 4 L2 dirs with d02 wrfouts + 0 `l2rerun_l3_24h_*` dirs surviving with d02.

3. **Did I assume "complete = 25 files" too aggressively?**
   - Cross-checked: `inventory_run` in `src/gpuwrf/io/data_inventory.py:186-188` defines `complete = files AND observed_hours >= 24 AND missing_times == 0 AND no parse errors AND no metadata error`. My scout's `is_complete_24h_hourly` enforces "25 sorted wrfouts AND every expected 0..24 h timestamp present in the observed set". Tightening to include `metadata_error is None` would not change the classification — the three "complete" `(66,159)` runs all opened cleanly in netCDF4.
   - L2 72h runs: I count 25 contiguous (init through +24 h) as "complete-for-24h" even though the run is 72 h. That's correct for the harness's leads `(0, 6, 12, 24)` requirement; the +48 h / +72 h files are surplus.

4. **Could `WRONG_GRID` (66, 120) members be rescued by the regrid path?**
   - Inspected `compute_rmse_against_gen2` (`data_quality.py:374`): hard `ValueError` on shape mismatch, no regrid hook. Adding one is Option B in `recommendation.md` and is rejected for the reasons recorded there.

5. **Could the M6-S2 reference cycle (`20260520_18z`) itself be recovered?**
   - Walked `runs/wrf_l3/20260520_18z_l3_24h_20260521T045847Z/`: `wrfbdy_d01`, `namelist.input`, `namelist.output`, `metadata.json`, lookup tables — **no wrfouts** retained. This is the cycle the M6 `gen2_manifest_v2.json` points to for shape metadata; the actual `wrfout_d02_2026-05-20_18:00:00` that the manifest cites *does not exist on disk* (manifest was built when retention had not yet been stripped). Conclusion: the harness's `DEFAULT_M6_GEN2_RUN_DIR` no longer points to a usable run. This is a separate latent breakage that the manager should patch when bumping `DEFAULT_ENDING_CYCLE`; flagged here so it doesn't surprise the follow-up sprint.

6. **AIFS dependency check (contract dependency #2)**:
   - `/mnt/data/canairy_meteo/data/aifs_single/aifs_single_202604.nc` and `aifs_single_202605.nc` present. 751 AIFS month files total; coverage adequate for any backfill in Option D.

7. **WPS staging dirs surviving for replay (Option D feasibility)**:
   - `runs/wps_cases/{20260428,20260429,20260521,20260521_l2rerun,20260522,20260523,20260524,20260525}_18z_72h/` — 8 staging dirs total. Adequate for replaying ~5-8 recent cycles without re-running WPS.

## Tests added under `tests/`

Per the contract the tester role may modify `tests/`; per the contract, this sprint is research-only and writes no code. I therefore added a minimal **inventory-shape test** that runs the same path the scout inventory walks and asserts the structural facts the recommendation depends on (corpus gate is `BLOCKED_CORPUS`, exactly 3 pinned-grid-complete 24h-hourly d02 runs, zero in-window). The test is **skipped** when `/mnt/data/canairy_meteo` is not present (CI safety), and reads with `netCDF4` header-only — never iterates variable arrays — matching contract hard rule #6.

File: `tests/test_m7_gen2_corpus_scout_inventory.py` (new, this sprint).

It guards against three regressions:

- The pinned-grid `(66, 159)` modal d02 shape (so a future grid rebase that changes the pinning is forced to surface here, not in the next M7 dispatch).
- The "0 tier4-eligible in-window pinned-grid-complete L3 24h members" claim (so this regression baseline is captured; tests will start failing once Option D succeeds and the corpus grows, at which point the test is updated alongside the cycle-window bump).
- The set of three pinned-grid-complete run IDs (so if the inventory walker silently drops them, CI catches it).

## Gaps / unresolved risks

- **Gap 1 — Stripped cycles cannot be recovered from this corpus alone.** Once `wrfout_d02_*` is gone, replay needs CPU WRF wall-time. This sprint cannot fix that; operator action is required.
- **Gap 2 — `DEFAULT_M6_GEN2_RUN_DIR` points to a now-empty cycle dir.** The `gen2_accessor.DEFAULT_M6_GEN2_RUN_DIR` still names `20260520_18z_l3_24h_20260521T045847Z`, but that dir has been stripped of wrfouts. Any code path that walks that constant for actual data will surface a latent error. Out of this sprint's scope to fix (no code changes allowed), but the follow-up sprint must rebind it.
- **Gap 3 — Live run `20260525_18z_l3_24h_20260526T221207Z` snapshot is partial (~8 hours).** Inventory was taken mid-forecast. A re-scout after the run completes will reclassify it as `PINNED_GRID_COMPLETE` and bring the disk count to 3 pinned-grid-complete (still 0 in-window unless cycle bump occurs).
- **Gap 4 — L2-vs-L3 physics-configuration question (Option B/C territory).** Whether the L2 72h `20260524` member's d02 is *physically equivalent* to the L3 24h `20260524` member's d02 was not verified by simulation — I only verified header dims and grid shape match. If the manager ever revisits Option B/C, this needs a physics-side check (boundary update cadence, one-way vs two-way nesting, microphysics).
- **Gap 5 — No verification that the surviving complete runs are *bitwise identical* to what M6.5-D1 froze.** I did not SHA-hash the wrfouts (deferred to follow-up that consumes them for tolerance freeze).
- **Gap 6 — `gen2_archive/teacher_l3/` historical corpus** is conceivably usable if it shares the new pinned grid, but the contract restricted scope to `wrf_l[23]/` and the historicals look pre-pinning. Investigation deferred.

## Decision: RECOMMENDATION_READY

The corpus is materially BLOCKED at gate sizing (0/10 in-window, 2/10 disk-total) and the recommendation is sufficiently bounded to dispatch the next action without further investigation. Manager should:

1. Take **Option D** (operator: flip retention now + replay 5-7 missing cycles using the surviving WPS staging dirs) to grow the corpus toward `N≥10` over the next 2-4 nights.
2. Optionally enable the **bounded Option A bridge** (probationary `N=5` tolerance freeze, tagged non-operational) to unblock the M7-S0 dispatch this week.
3. **Do not** dispatch a writer/regridder sprint (Options B and C are net negatives per `recommendation.md`).
4. After Option D lands, dispatch a follow-up sprint to bump `DEFAULT_ENDING_CYCLE`, rebind `DEFAULT_M6_GEN2_RUN_DIR` to a surviving cycle, and run `scripts/m6_run_tier4.py` cleanly.

Sprint terminates cleanly with all five proof objects on disk. No tmux loop, no code under `src/`, no writes under `/mnt/data/canairy_meteo`.
