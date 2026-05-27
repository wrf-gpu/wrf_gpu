# Sprint Contract — M7 NetCDF WRF-Compatible Writer

**Sprint ID**: `2026-05-27-m7-netcdf-writer`
**Created**: 2026-05-27 (autonomous overnight loop, parallel to restart-continuity)
**Status**: READY
**Predecessor**: `.agent/sprints/2026-05-27-m7-wrfout-io-compat/` (COMPAT_MATRIX_READY — major gap surfaced)

## Objective

The wrfout I/O compat audit found that the current `write_wrfout_gpu` produces a `.npz` proof container, not real NetCDF, and is missing 21 downstream-critical WRF variables (XLAT, XLONG, PSFC, RAINC, RAINNC, SWDOWN, GLW, PBLH, HFX, LH, CLDFRA, terrain, P/PB pairs, etc.). Gen2 post-processing and AEMET verification can't consume this output.

This sprint implements a real WRF-compatible NetCDF writer that produces wrfout files Gen2 downstream consumers can read with their existing tools. Schema conformance to the Gen2 reference wrfout (the file used in the compat audit: `/mnt/data/canairy_meteo/runs/wrf_l3/20260525_18z_l3_24h_20260526T221207Z/wrfout_d02_*`).

Scope: a minimal-but-correct NetCDF writer covering the downstream-critical fields. Out-of-scope: fields that aren't consumed by any downstream pipeline (the audit will name 100+ such variables; we don't replicate them all in v0).

## Acceptance

- **AC1 — Minimum-viable variable list**: read `.agent/sprints/2026-05-27-m7-wrfout-io-compat/compat_matrix.md` and `explicit_deviations.md`. Identify the **minimal set** of variables required for Gen2 post-processing + AEMET station verification. Emit `.agent/sprints/2026-05-27-m7-netcdf-writer/minimum_variable_list.md` listing each, with downstream consumer + reasoning. Aim for ~30-50 variables (not all 362).

- **AC2 — NetCDF writer**: implement `src/gpuwrf/io/wrfout_writer.py` with `write_wrfout_netcdf(state, grid, namelist, path, *, valid_time, lead_hours, run_start)`. Uses `netCDF4` (or `xarray` if cleaner) to produce a file with:
  - WRF-standard dimensions: `Time`, `west_east`, `west_east_stag`, `south_north`, `south_north_stag`, `bottom_top`, `bottom_top_stag`, `soil_layers_stag`, `DateStrLen`
  - WRF-standard global attributes: `TITLE`, `START_DATE`, `SIMULATION_START_DATE`, `WEST-EAST_GRID_DIMENSION`, etc.
  - `Times` (DateStrLen × 1) variable with the standard `YYYY-MM-DD_HH:MM:SS` format
  - `XTIME` (Time) variable with minutes since simulation start
  - All AC1 variables, with correct dims, units, descriptions, and stagger attributes

- **AC3 — Base+perturbation pair correctness**: WRF stores `P` as perturbation (with companion `PB` base state), `PH` perturbation (with `PHB` base), `MU` perturbation (with `MUB` base). The GPU code carries total-state in some fields. Emit `.agent/sprints/2026-05-27-m7-netcdf-writer/total_to_perturbation_mapping.md` documenting how each GPU field maps to WRF's perturbation+base pair, and the math used in the writer.

- **AC4 — Round-trip read test**: write a wrfout via the new writer using a synthetic Canary 3km State, then read it back via `netCDF4.Dataset(...)` and confirm: (a) all AC1 variables present, (b) dims correct, (c) attributes correct, (d) Gen2 reference file can be opened with the same xarray code path (i.e., we don't break compat with existing readers). Emit `.agent/sprints/2026-05-27-m7-netcdf-writer/roundtrip_proof.json`.

- **AC5 — Schema comparison vs Gen2 reference**: rerun `scripts/m7_wrfout_io_compat_audit.py` against the new writer output; the resulting compat matrix should show **0 downstream-critical missing fields** and **0 dimension/dtype mismatches** on the AC1 minimum set.

- **AC6 — Tests**: add `tests/test_m7_netcdf_writer.py` with:
  - Round-trip test (write + read via netCDF4)
  - Dim/attr conformance test against a reference wrfout schema
  - At least 5 fields validated for total = base + perturbation

- **AC7 — No GPU runtime in this sprint.** All testing uses synthetic State constructed in Python (CPU). The real GPU integration happens after this sprint, when the writer is wired into the daily-pipeline driver.

- **AC8 — Worker report** with verdict `WRITER_READY / PARTIAL / BLOCKED`.

## Files Worker May Modify

- `src/gpuwrf/io/wrfout_writer.py` (NEW)
- `src/gpuwrf/io/__init__.py` (export)
- `tests/test_m7_netcdf_writer.py` (NEW)
- `scripts/m7_netcdf_writer_smoke.py` (NEW — runs the round-trip + AC5 schema compare)
- `.agent/sprints/2026-05-27-m7-netcdf-writer/**`

## Files Worker Must Not Modify

- `src/gpuwrf/coupling/driver.py` (`write_wrfout_gpu` is the OLD .npz path; leave it in place for now; do not delete or refactor — that's a different sprint's responsibility)
- `src/gpuwrf/contracts/state.py`
- governance files
- `/mnt/data/canairy_meteo/**`

## Hard Rules

1. **CPU-only**: AC7 enforces this. Parallel-safe with the restart-continuity sprint (which uses GPU).
2. **CPU pinning**: `taskset -c 0-3`.
3. **No model code changes.** Writer is additive.
4. **Do not interfere with tmux `0:1`** (nightly WRF).
5. **No remote push.** Local commit on `worker/gpt/m7-netcdf-writer` only.
6. **One reference wrfout for schema comparison** — use the file the iocompat sprint chose: `/mnt/data/canairy_meteo/runs/wrf_l3/20260525_18z_l3_24h_20260526T221207Z/wrfout_d02_*`. Do not bulk-iterate.
7. **Variable subset is OK**: do NOT try to replicate all 362 CPU fields. Focus on the AC1 minimum set.

## Dependencies

- iocompat audit complete (commit `a181d68`)
- `netCDF4` Python package available (likely already in env; verify on AC0 preflight)

## Proof Objects

- `.agent/sprints/2026-05-27-m7-netcdf-writer/minimum_variable_list.md` (AC1)
- `.agent/sprints/2026-05-27-m7-netcdf-writer/total_to_perturbation_mapping.md` (AC3)
- `.agent/sprints/2026-05-27-m7-netcdf-writer/roundtrip_proof.json` (AC4)
- `.agent/sprints/2026-05-27-m7-netcdf-writer/compat_matrix_v2.md` (AC5 — gate)
- `.agent/sprints/2026-05-27-m7-netcdf-writer/worker-report.md` (AC8)
- `tests/test_m7_netcdf_writer.py` (AC6)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 4-8 h (substantial; lots of WRF schema detail)
- Branch: `worker/gpt/m7-netcdf-writer`
- Worktree: `/tmp/wrf_gpu2_ncwriter`
- GPU usage: NONE (parallel-safe with restart-continuity)
