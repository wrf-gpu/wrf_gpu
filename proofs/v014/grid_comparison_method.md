# V0.14 Wrfout Grid Comparison Method

Date: 2026-06-08

## Purpose

`scripts/compare_wrfout_grid.py` is the primary v0.14 CPU-WRF-vs-GPU wrfout grid comparator. It is a validation tool only: it reads existing NetCDF wrfout files, uses CPU `numpy`/`netCDF4`, and does not run model code, JAX, CUDA, or GPU work.

## Pairing

The script accepts `--cpu-dir`, `--gpu-dir`, and `--domain`. It discovers `wrfout_<domain>_<timestamp>` files, pairs by exact domain/timestamp, records unmatched CPU/GPU files, and infers lead hours from `--init`, a `YYYYMMDD_HHz` token in the path, or the earliest wrfout as a fallback.

## Coverage

For paired files it inventories the union of variables across CPU and GPU outputs:

- `cpu_only` and `gpu_only` variables are listed explicitly.
- Common non-numeric variables are audited separately when supported, currently `Times`.
- Common numeric variables are compared when dimensions and native shapes match.
- Shape/dimension/name incompatibilities and missing-on-some-lead cases are recorded in JSON.

The smoke case compared 99 numeric compatible fields and audited `Times`, covering all 100 common variables in the retained Case 3 d02 pair set.

## Streaming And Memory

The comparator is variable-major. It opens paired files for one variable/lead at a time and never loads a full campaign state. For the current variable it retains finite absolute-difference chunks to compute exact p95/p99, plus current/previous CPU/GPU arrays to detect time invariance. This keeps peak memory bounded by the largest single field and split accumulators, not by the number of wrfout files.

Observed smoke behavior on Case 3 d02:

- wall time: 1:27.33
- max RSS: 446044 KB
- CPU/GPU wrfout pairs: 24
- GPU used: no

## Statistics

For every numeric compatible field and lead, JSON includes:

- `n`, `finite_cpu`, `finite_gpu`, `finite_pair`, `finite_pair_fraction`
- `bias`, `rmse`, `mae`, `p95_abs`, `p99_abs`, `max_abs`
- Pearson `r` when both sides have variance
- worst cell with lead, index, CPU value, GPU value, and absolute difference

Per-field pooled stats are accumulated across all paired leads. Drift summaries include RMSE slope per hour, bias slope per hour, worst lead, and lead-wise bias sign consistency.

## Static And Dynamic Separation

Fields in the known WRF grid/metric/static set are classified as `static`. Other fields whose CPU and GPU arrays are both unchanged across all paired leads are classified as `time_invariant`. Only remaining numeric fields are classified as `dynamic`, so static/base-state failures are not mixed into dynamic forecast RMSE summaries.

`XTIME` is numeric metadata and gets normal stats while remaining classified as `time_metadata`. `Times` is string metadata and is audited for equality by lead.

## Spatial Splits

When `HGT`, `LANDMASK`, `XLAT`, and `XLONG` exist in CPU truth, the script emits optional split stats:

- land/ocean
- elevation bins: ocean, land 0-300 m, land 300-1000 m, land >1000 m
- quadrants by median lat/lon
- boundary frame vs interior, default 5 cells

Masks are applied on native mass, U-staggered, or V-staggered horizontal shapes. Fields without a supported horizontal tail are skipped for splits.

## Tolerance Policy

`--tolerance-json` is optional. Supported schemas are direct field maps or nested `fields`, `variables`, or `tolerances` maps. Supported metrics are `rmse`, `mae`, `bias_abs`, `p95_abs`, `p99_abs`, `max_abs`, `element_abs`, `pearson_min`, and `finite_pair_fraction_min`.

No tolerance is derived or tuned by the comparator. Without a supplied manifest the verdict is `REPORT_ONLY_NO_TOLERANCE_MANIFEST`; with a manifest the verdict is `PASS` only if all supplied checks pass.

## Human And Machine Outputs

The markdown report is intentionally short and contains only verdict, coverage, top 10 field differences, top 5 drift signals, top coverage issues, and the next debug recommendation. The detailed per-field/per-lead/split/worst-cell tables live in JSON.

Smoke artifacts:

- `proofs/v014/grid_comparison_framework_smoke.json`
- `proofs/v014/grid_comparison_framework_smoke.md`

## B4 And Switzerland Integration

For v0.14 B4, run this comparator after any static metric/base-state or dycore fix and treat static/grid mismatches as the first gate. B4 should not advance to FP32, TOST, or speed claims until the JSON shows static fields exact or covered by predeclared exceptions and the hard dynamic fields are below frozen tolerances.

For Switzerland, use the same script against the generated GPU wrfout directory and CPU truth with `--domain d01`. Supply a frozen hard-field tolerance manifest for `T2`, `U10`, `V10`, `PSFC`, `RAINNC`, `T`, `U`, `V`, `W`, and `QVAPOR`; require 24 paired hours and require every hard field to be present and compatible. Extra common fields should remain report-only inventory until Switzerland-specific tolerances are frozen before candidate scoring.
