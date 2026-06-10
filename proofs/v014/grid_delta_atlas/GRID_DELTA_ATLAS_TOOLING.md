# V0.14 Grid-Delta Atlas Tooling Proof

Date: 2026-06-10
Worker: GPT-5.5 xhigh
Scope: offline validation tooling only

## Objective

Implemented `scripts/build_grid_delta_atlas.py`, an offline CPU-WRF-vs-GPU
wrfout comparator for the v0.14 Grid-Delta Atlas gate. The tool reads existing
NetCDF `wrfout_*` files and emits stable real-run artifacts at:

- `proofs/v014/grid_delta_atlas/manifest.json`
- `proofs/v014/grid_delta_atlas/grid_delta_summary.json`
- `proofs/v014/grid_delta_atlas/GRID_DELTA_ATLAS.md`
- `docs/assets/v014/grid_delta_atlas/`

No model kernels, runtime source, TOST runner semantics, Switzerland
generation, memory/FP32 code, README release claims, or `KNOWN_ISSUES` were
modified.

## Command Syntax

Single case:

```bash
python scripts/build_grid_delta_atlas.py \
  --cpu-dir /path/to/cpu_wrf_case \
  --gpu-dir /path/to/gpu_case \
  --domain d02 \
  --init 2026-05-01T18:00:00Z
```

Multi-case manifest:

```bash
python scripts/build_grid_delta_atlas.py \
  --case-json /path/to/grid_delta_cases.json
```

The case JSON can be either a list or an object with `cases`, where each case
has `case_id`, `cpu_dir`, `gpu_dir`, optional `domain`/`domains`, and optional
`init_time_utc`.

Useful options:

- `--min-lead` / `--max-lead` restrict compared lead hours.
- `--field FIELD` repeats to limit a smoke run to selected fields.
- `--tolerance-json` reads predeclared field tolerance limits.
- `--no-plots` skips optional matplotlib plots.
- `--no-default-mandatory-fields` disables the built-in v0.14 core-field
  missing-field hard-fail list for synthetic/unit runs.

## Emitted Fields, Metrics, and Records

For every discovered field in the CPU/GPU union, the tool records whether each
paired wrfout has the field. For numeric fields with exact matching shape and
dimension names, it computes per-field, per-lead, per-pair, and pooled metrics:

- `count`, `finite_cpu_count`, `finite_gpu_count`, `finite_pair_count`
- `nonfinite_cpu_count`, `nonfinite_gpu_count`, `finite_pair_fraction`
- `max_abs`, `rmse`, `mae`, `bias`
- `p50_abs`, `p95_abs`, `p99_abs`, `p999_abs`
- safe relative metrics: `mean_abs_rel`, `p95_abs_rel`, `max_abs_rel`
- Pearson `correlation`
- worst index with CPU value, GPU value, signed delta, lead, case, domain, and
  file paths

The summary explicitly records:

- missing fields
- non-numeric fields
- nonfinite CPU/GPU fields
- shape mismatches
- dimension-name mismatches
- missing mandatory v0.14 core fields
- tolerance failures when a tolerance manifest is supplied

Lead-time stability summaries include RMSE and bias slopes per lead hour,
maximum lead-to-lead jumps, late-window minus early-window deltas, and worst
lead records.

## Plots

When matplotlib is available and `--no-plots` is not set, the tool writes small
deterministic PNGs under `docs/assets/v014/grid_delta_atlas/` by default:

- `heatmap_rmse.png`
- `heatmap_bias.png`
- `heatmap_p99_abs.png`
- `heatmap_max_abs.png`
- `core_fields_rmse_timeseries.png`
- `dashboard.png`
- up to `--spatial-plot-limit` core-field worst-case spatial max-abs maps

If matplotlib is missing, JSON and Markdown outputs still emit and the manifest
records `plots.status = skipped_missing_dependency`.

## Synthetic Test Coverage

`tests/test_grid_delta_atlas.py` builds tiny WRF-style NetCDF wrfout fixtures
with:

- paired CPU/GPU `wrfout_d02_*` files at two lead hours
- one unmatched CPU wrfout
- common numeric fields, including 2-D and staggered 3-D shapes
- a char non-numeric field
- CPU-only and GPU-only numeric fields
- a GPU nonfinite value
- a deliberate shape mismatch
- case JSON input and lead filtering
- optional plot generation when matplotlib is installed

The tests verify the manifest, summary, Markdown report, inventory records,
metrics, worst-value record, nonfinite accounting, shape-mismatch accounting,
case-manifest pairing, lead filtering, and dashboard/heatmap plot paths.

## Validation Commands

Passed in this worktree:

```bash
python -m py_compile scripts/build_grid_delta_atlas.py
PYTHONPATH=src pytest -q tests/test_grid_delta_atlas.py
python scripts/build_grid_delta_atlas.py --help
git diff --check
```

Observed pytest result:

```text
3 passed
```

## Real Campaign Inputs Still Required

This sprint does not claim v0.14 equivalence. The real atlas still requires the
post-grid-parity CPU-WRF and GPU wrfout campaign inputs, selected case manifest,
and frozen tolerance manifest or manager decision that fields remain report-only.

The generated `manifest.json`, `grid_delta_summary.json`, and
`GRID_DELTA_ATLAS.md` should be produced from real campaign data only after the
current grid-parity blocker closes and long validation is approved.

## Limitations and Assumptions

- NetCDF `wrfout_*` files are expected in standard WRF filename form:
  `wrfout_dNN_YYYY-MM-DD_HH:MM:SS`, optionally with `.nc`.
- Pairing is exact by case, domain, and valid time; lead hour is inferred from
  explicit init time, case/path `YYYYMMDD_HHz`, or earliest wrfout fallback.
- Shape and dimension-name mismatches are recorded and skipped; there is no
  silent crop or stagger conversion.
- Exact percentiles are computed field-by-field, so memory stays bounded by one
  field across paired files rather than the full campaign.
- Default mandatory core fields are hard-fail inventory checks for real runs;
  synthetic tests disable them explicitly.
