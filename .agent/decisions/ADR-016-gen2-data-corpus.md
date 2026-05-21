# ADR-016: Gen2 d02 Data Corpus

## Status

ACCEPTED — M6.5-D1 Opus reviewer ACCEPT 2026-05-21 (commit `89ab922`);
manager closeout commit `3fb2bae`; AC4 threshold amended commit `d16dfb2`.
Operational follow-ups (5 non-blocking tickets) queued for M7-S0a.

## Context

M7-S0 needs a repeatable way to consume historical Gen2 d02 `wrfout`
truth from `/mnt/data/canairy_meteo/runs/wrf_l3/` without rediscovering
archive shape, quality issues, and valid-time selection in every sprint.
The source tree is read-only. Tests must remain reproducible without
access to `/mnt/data`.

Gen2 L3 run directories follow the observed convention:

`YYYYMMDD_18z_l3_24h_YYYYMMDDTHHMMSSZ/wrfout_d02_YYYY-MM-DD_HH:MM:SS`

Each d02 history file is treated as one valid-time chunk. Complete
24-hour runs require hourly files from lead 0 through lead 24, a
continuous valid-time axis, and readable WRF-standard NetCDF variables.

## Decision

Adopt `artifacts/m6_5/gen2_d02_inventory.json` as the corpus inventory
schema for M7 consumers. Each run record includes `start_date`, `hours`,
`init_time_utc`, valid-time range, `d02_wrfout_file_count`,
`total_bytes`, missing valid times, completeness, and first-file NetCDF
metadata. The inventory count is anchored on Gen2 run directories and
`wrfbdy_d01` markers so runs with zero retained d02 history are still
visible.

Adopt `artifacts/m6_5/gen2_d02_quality_audit.json` as the quality bar.
For complete runs, the audit samples `U10`, `V10`, `T2`, `Q2`, `PSFC`,
and `RAINNC` by opening one file at a time. A run is:

- `GREEN` when required fields are finite, hourly, and have no spike
  flags.
- `PARTIAL` when the run is incomplete or has suspicious z-score spikes
  above 5 standard deviations.
- `FAIL` when required fields are missing or contain NaN/Inf values.

Adopt `src/gpuwrf/io/gen2_wrfout_loader.py` as the lazy-load policy.
The loader opens a single NetCDF file per requested valid time or
iterator chunk. It returns NumPy arrays by default and converts to JAX
arrays only when a validation consumer explicitly requests `as_jax=True`.
No full-corpus materialization is allowed.

Adopt `compute_rmse_against_gen2(gpu_forecast_state,
gen2_wrfout_path, valid_time, fields=("U10","V10","T2"))` as the M7
RMSE adapter schema. It returns:

`{field: {"rmse": float, "error_map": jax.Array, "valid_time_utc": str, "gen2_source_file": str}}`

Boundary replay cross-checks compare lead-0 replay-zarr d02 boundary
strips against d02 wrfout boundary strips for the same valid time.
Thresholds match `src/gpuwrf/io/boundary_replay.py:40-41` TOLERANCES
(U/V/QVAPOR `rel_mae_max=0.03`, T `rmse 0.5K`, PH tight relative gate);
any breach is a data-pipeline failure flag, not a
model-performance result. (Amended 2026-05-21 per M6.5-D1 Opus AC4
disposition c — initial 1% spec was manager error, not matching the
existing source-of-truth TOLERANCES.)

## Alternatives

Materializing whole runs or the whole corpus into memory was rejected
because it violates the sprint hard rule and would hide future scaling
risks.

Using `/mnt/data` fixtures in tests was rejected because tests must be
portable. Tests instead fabricate small WRF-standard NetCDF files with
the same time, dimension, and variable conventions.

Extending `src/gpuwrf/io/boundary_replay.py` was rejected because that
file is read-only for this sprint. The cross-check lives in the new
validation module.

## Consequences

M7 can select complete Gen2 d02 subsets by date and minimum observed
hours without rescanning the archive manually. M7 RMSE code can depend
on a stable adapter return shape while keeping the Gen2 truth IO lazy.

The quality audit is a corpus/data-pipeline gate only. It does not prove
GPU physics accuracy and must not be used as a performance or physics
claim.

## Proof Objects

- `artifacts/m6_5/gen2_d02_inventory.json`
- `artifacts/m6_5/gen2_d02_quality_audit.json`
- `artifacts/m6_5/gen2_d02_subset_complete24h.json`
- `tests/test_m6_5_gen2_loader.py`
- `tests/test_m6_5_data_quality.py`
- `tests/test_m6_5_rmse_adapter.py`

## Review

Requires independent Opus review before manager close because this ADR
and new production IO/validation code are non-exempt under the sprint
lifecycle rule.
