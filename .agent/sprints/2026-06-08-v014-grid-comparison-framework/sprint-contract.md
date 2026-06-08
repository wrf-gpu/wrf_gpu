# Sprint Contract: V0.14 Grid Comparison Framework

Date: 2026-06-08
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Build the v0.14 primary CPU-WRF-vs-GPU-WRF grid-field validation method: a fast,
complete, reproducible wrfout-directory comparator that finds unacceptable cell
divergence, systematic drift, and likely debug directions without relying on
station TOST.

The top-level output must be short enough for manager/agent context windows.
Detailed diagnostics belong in machine artifacts, not in chat/handoffs.

## Priority Context

Current project priority order:

1. Grid-cell parity across all comparable wrfout fields.
2. FP32 acoustic / mixed precision.
3. Remaining memory issues.
4. Powered TOST after grid fields are no longer radically divergent.

This sprint is validation infrastructure. It must not resume TOST, launch long
GPU campaigns, or edit model physics/dycore code.

## Scope

Allowed write scope:

- `scripts/compare_wrfout_grid.py`
- `proofs/v014/grid_comparison_framework_smoke.json`
- `proofs/v014/grid_comparison_framework_smoke.md`
- `proofs/v014/grid_comparison_method.md`
- `.agent/reviews/2026-06-08-v014-grid-comparison-framework.md`
- Optional focused tests under `tests/validation/` if small synthetic NetCDFs
  are generated in a tempdir.

Read-only:

- `src/gpuwrf/io/wrfout_writer.py`
- existing proof scripts under `proofs/v014/`
- CPU/GPU retained wrfout directories

## Required Comparator Behavior

The comparator must:

- accept CPU and GPU wrfout directories plus domain name;
- pair files by timestamp and domain;
- stream one variable/lead at a time, not load the full campaign into memory;
- compare every common variable with compatible dimensions;
- enumerate missing, GPU-only, CPU-only, and incompatible fields;
- classify static/time-invariant fields separately from dynamic fields;
- compute for every comparable field and lead: `n`, finite counts, bias, RMSE,
  MAE, p95_abs, p99_abs, max_abs, Pearson r when meaningful;
- compute compact drift signals: lead-wise RMSE/bias slope, worst lead, and
  sign consistency;
- compute optional spatial splits when fields exist: land/ocean, elevation bins,
  quadrant, boundary frame, and worst-cell/worst-region summaries;
- support a tolerance manifest but never tune tolerances after seeing results;
- emit a concise human summary and a detailed JSON machine artifact.

## Context-Saving Output Contract

Human markdown must fit in roughly 80 lines and include only:

- verdict;
- file/domain coverage;
- top 10 failing fields by severity;
- top 5 systematic drift signals;
- top 5 missing/incompatible coverage issues;
- next debug recommendation.

Detailed per-field/per-lead tables must stay in JSON and optional CSV/NDJSON.

## Smoke Inputs

Use retained Case 3 d02 artifacts:

- CPU:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- GPU:
  `/tmp/v0120_powered_tost_runs/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
- Domain: `d02`

## Commands

CPU-only:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python scripts/compare_wrfout_grid.py \
    --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z \
    --gpu-dir /tmp/v0120_powered_tost_runs/l2_d02_20260501_18z_l2_72h_20260519T173026Z \
    --domain d02 \
    --out-json proofs/v014/grid_comparison_framework_smoke.json \
    --out-md proofs/v014/grid_comparison_framework_smoke.md
```

Validation:

```bash
python -m py_compile scripts/compare_wrfout_grid.py
python -m json.tool proofs/v014/grid_comparison_framework_smoke.json >/dev/null
```

## Acceptance Criteria

- The smoke run completes CPU-only on retained Case 3.
- The report compares all common compatible variables, not only `T2/U10/V10`.
- Static fields are not mixed into dynamic forecast RMSE.
- Missing/incompatible fields are explicit.
- The top-level markdown is concise and does not dump giant tables.
- The machine JSON contains enough per-field/per-lead detail for later debug
  agents to inspect without rerunning the comparator.
- The review report explains how this becomes the v0.14 primary validation
  artifact for Canary and Switzerland.

## Closeout

Close with:

- files changed
- commands run
- proof objects produced
- runtime and memory behavior observed
- limitations
- exact recommendation for plugging this into v0.14 B4 and Switzerland
