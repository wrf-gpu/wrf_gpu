# Recovery Candidates — M7 Gen2 Corpus Scout

**Sprint**: `2026-05-27-m7-gen2-corpus-scout` (AC3)
**Scope**: enumerate every non-`PINNED_GRID_COMPLETE` d02 member, name the specific reason, mark data-side recoverability (no fresh CPU WRF forecast required).

The pinned reference shape is `(66, 159)` mass (`d02`). "Data-side recoverable" means: the wrfout history exists somewhere and only needs relabeling, regridding, or sub-sampling — no new forecast.

## Per-member assessment

### 1. `20260521_18z_l3_24h_20260522T133443Z` (wrf_l3)
- **Current status**: `PINNED_GRID_COMPLETE` (25 d02 wrfouts, `(66, 159)`).
- **Blocker**: cycle `20260521_18z` is *after* `DEFAULT_ENDING_CYCLE = 20260520_18z` in `src/gpuwrf/validation/tier4_probtest.py:27`.
- **Recovery**: **trivial data-side** — bump `DEFAULT_ENDING_CYCLE` (or `--ending-cycle` flag on `m6_run_tier4.py`) to `20260524_18z` and this member becomes eligible. Held-out cycle `20260519_18z` remains valid (it's a separate cycle).

### 2. `20260524_18z_l3_24h_20260525T225640Z` (wrf_l3)
- **Current status**: `PINNED_GRID_COMPLETE` (25 d02 wrfouts, `(66, 159)`).
- **Blocker**: same as #1 — `cycle > DEFAULT_ENDING_CYCLE`.
- **Recovery**: **trivial data-side** — same ending-cycle bump.

### 3. `20260524_18z_l2_72h_20260525T225640Z` (wrf_l2)
- **Current status**: 73 d02 wrfouts, `(66, 159)`, 72-hour forecast.
- **Blockers** (two):
  1. **Pattern filter** — `RUN_DIR_RE_L3 = ^\d{8}_\d{2}z_l3_24h_\d{8}T\d{6}Z$` rejects the `l2_72h` directory name.
  2. **Distinct WRF configuration** — this is the L2 master that *spawned* the L3 1-km nest via two-way nesting. L2 has lower d01 resolution (9 km parent) but the `d02` here is also 3 km on the pinned domain shape, so it's grid-compatible. However the forecast initial conditions and integration step length differ from the L3 24h run for cycle 20260524.
- **Recovery**: **data-side, but with caveats** —
  - Option (a) symlink the L3 24h sister `20260524_18z_l3_24h_20260525T225640Z` already exists ⇒ no relabel needed; just use that one (the L2 path is redundant).
  - Option (b) treat L2 72h members as an independent stratum. **Not advised**: they're a different physics configuration (L2 cold-start with one-way nesting), and mixing them into the same RMSE pool distorts the tolerance distribution. Note the L2 72h covers a 72-hour window so it could provide the +24 h leads of three consecutive cycles, but the +48 h and +72 h leads sample the same model state from a single 18Z init, not three independent inits.

### 4. `20260509_18z_l3_24h_20260511T190519Z` (wrf_l3)
- **Current status**: 25 d02 wrfouts, **`(66, 120)` legacy mass shape**.
- **Blocker**: pre-pinning d02 dimensions; `compute_rmse_against_gen2` will raise on first compare.
- **Recovery**: **NOT data-side recoverable** — the legacy d02 grid was a different geographic footprint (probably narrower W-E extent over the Canaries). Regridding from `(66, 120)` to `(66, 159)` would require a 2-D horizontal interpolation per variable per hour and would no longer be a CPU WRF baseline — it'd be a derived product that defeats the purpose of "operational baseline". **Exclude**.
- **Side note**: the L2 sister `20260509_18z_l2_72h_20260511T190519Z` is also `(66, 120)` for d02 and has the same disqualification.

### 5. `20260521_18z_l3_24h_20260522T072630Z` (wrf_l3)
- **Current status**: 9 d02 wrfouts, `(66, 159)`. Times: 18-20 Z + 00-02 Z + 18-23 Z (non-contiguous).
- **Blocker**: a sibling run `20260521_18z_l3_24h_20260522T133443Z` (the same cycle, **later** `created` stamp) is `PINNED_GRID_COMPLETE`. The selection rule "latest-`created`-per-cycle wins" already supersedes this member.
- **Recovery**: **not needed** — discard; the later-created sibling carries that cycle.

### 6. `20260521_18z_l2_72h_20260522T133443Z` (wrf_l2)
- **Current status**: 20 d02 wrfouts, `(66, 159)`. First 20 hours of a 72-hour run; truncated mid-forecast.
- **Blocker**: pattern filter (wrf_l2) and incompleteness (≥25 needed for 24 h).
- **Recovery**: **not data-side** — would require resuming the CPU WRF run from a restart file. The corresponding L3 24h sibling `20260521_18z_l3_24h_20260522T133443Z` already covers this cycle completely on the pinned grid. **Discard**.

### 7. `20260428_18z_l3_24h_20260525T221139Z` (wrf_l3)
- **Current status**: 3 d02 wrfouts (18-20 Z), `(66, 159)`.
- **Blocker**: 22 hourly wrfouts missing; only the first 3 hours retained.
- **Recovery**: **not data-side** — needs a fresh 24-h CPU WRF run from this cycle. AIFS `aifs_single_202604.nc` is on disk so re-running is feasible; WPS case `runs/wps_cases/20260428_18z_72h/` already exists, so re-issuing a `wrf.exe` from the existing met_em files is the only step. *Operator action, not GPU-project action.*

### 8. `20260509_18z_l3_24h_20260512T154354Z` (wrf_l3)
- **Current status**: 3 d02 wrfouts (18-20 Z), `(66, 159)`. Same cycle as the legacy `(66, 120)` complete run — but this one is on the new grid (re-issued later).
- **Blocker**: 22 hourly wrfouts missing.
- **Recovery**: **not data-side** — same as #7; needs a re-run of the 24-h forecast on the pinned grid. AIFS `aifs_single_202605.nc` available; no WPS case under `runs/wps_cases/20260509_18z_72h/` (would have to be regenerated from met_em).

### 9. `20260525_18z_l3_24h_20260526T221207Z` (wrf_l3)
- **Current status**: 8 d02 wrfouts (18-23 Z + 00-01 Z), `(66, 159)`.
- **Blocker**: appears to be **still running** as of the scout snapshot (last file `2026-05-26_01:00:00`, ~17/24 hours pending). This is the live nightly run.
- **Recovery**: **time only** — will become `PINNED_GRID_COMPLETE` once the live forecast finishes. *Wait.*

### 10. The 51 wrfout-empty cycle directories
- **Current status**: every `*_l[23]_*` cycle directory between roughly `20260429` and `20260523` has zero retained `wrfout_d02_*` files (the d01, d03, d04, d05 are also gone — these are completely stripped of history). Only `wrfbdy_d01`, `namelist.input`, `namelist.output`, lookup tables, and `metadata.json` survive.
- **Blocker**: Canairy Gen2 retention policy strips wrfouts after the run finishes (confirmed pattern in `.agent/sprints/2026-05-22-m7-s0a-ops-data-prologue/gen2_corpus_backfill_plan.md:30-32`).
- **Recovery**: **not data-side** — wrfout was discarded; only fresh CPU WRF re-runs can recover. WPS met_em are also gone for cycles outside `runs/wps_cases/{20260428,20260429,20260521..20260525}_18z_72h/` (only 8 WPS staging dirs survive). For most stripped cycles, even WPS would need re-running before WRF can re-run.

## Supplementary off-scope candidates (read-only mention)

`/mnt/data/canairy_meteo/gen2_archive/teacher_l3/` contains six pre-2026 historical runs (`20231120_00z`, `20240204_00z`, `20240315_00z`, `20240620_00z`, `20240805_12z`, `20250601_00z`) totaling 234 wrfout files across all domains. These predate the current `d02 (66, 159)` pinning by months/years and almost certainly use a different grid; **not usable** without verification by the team that maintains the archive. Out of scope for this sprint by literal contract reading; flagged for the manager.

## Recoverability tally

| Class | Count | Member IDs |
|---|---|---|
| `PINNED_GRID_COMPLETE`, in-window | **0** | — |
| `PINNED_GRID_COMPLETE`, out-of-window (date) — trivial fix | **2** | `20260521_18z_l3_24h_20260522T133443Z`, `20260524_18z_l3_24h_20260525T225640Z` |
| `PINNED_GRID_COMPLETE`, wrong directory pattern (L2) — re-pattern questionable | **1** | `20260524_18z_l2_72h_20260525T225640Z` |
| `WRONG_GRID` legacy | **2** | `20260509_18z` L3+L2 |
| `MISSING_TIMES` recoverable only by re-run | **3** | `20260428_18z_l3_24h_*`, `20260509_18z_l3_24h_20260512T154354Z`, `20260521_18z_l2_72h_20260522T133443Z` |
| `IN_PROGRESS` (live) | **1** | `20260525_18z_l3_24h_20260526T221207Z` |
| `STRIPPED` (no wrfout retained) | **51** | every cycle dir without any `wrfout_d02_*` file |

**Net new data-side wins**: at most 2 → 3 pinned-grid-complete L3 24h members after ending-cycle bump + live `20260525` finishes. Still 7 short of the M7-S0 default gate of 10.
