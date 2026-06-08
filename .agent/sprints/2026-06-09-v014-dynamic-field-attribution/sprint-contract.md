# Sprint Contract: V0.14 Dynamic Field Attribution

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Use the new wrfout-grid comparator and retained Case 3 wrfouts to identify the
smallest useful dynamic-debug target: first/worst leads, fields, vertical
levels, regions, and co-located cell sets for CPU-WRF-vs-GPU divergence.

This sprint does not fix model code. It produces the field/cell manifest that a
same-state tendency localization sprint should use.

## Inputs

- `proofs/v014/grid_comparison_framework_smoke.json`
- `proofs/v014/grid_comparison_framework_smoke.md`
- `proofs/v014/static_metric_base_parity.json`
- `proofs/v014/wind_mass_divergence_probe.json`
- CPU truth:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- retained GPU:
  `/tmp/v0120_powered_tost_runs/l2_d02_20260501_18z_l2_72h_20260519T173026Z`

## Write Scope

- `proofs/v014/dynamic_field_attribution.py`
- `proofs/v014/dynamic_field_attribution.json`
- `proofs/v014/dynamic_field_attribution.md`
- `.agent/reviews/2026-06-09-v014-dynamic-field-attribution.md`

No `src/` edits. No GPU.

## Required Analysis

The probe must be CPU-only and must:

- ignore/root-cause-known retained static writer fields when ranking dynamic
  forecast errors;
- focus at least on `PSFC`, `MU`, `P`, `PH`, `U`, `V`, `U10`, `V10`, `T`,
  `QVAPOR`, `W`, `PBLH`, and radiation flux fields if present;
- report per-lead RMSE, bias, p99_abs, max_abs, and finite coverage;
- identify first materially bad lead under predeclared report-only thresholds;
- rank vertical levels for 3-D fields by RMSE/max_abs;
- rank regions using land/ocean, elevation, quadrant, and boundary-vs-interior
  masks;
- select 16-32 candidate mass-grid cells for same-state localization, including
  adjacent U/V/W/PH native-stagger context;
- quantify co-location/correlation among `dU10`, `dV10`, lowest-level `dU/dV`,
  `dPSFC`, `dMU`, `dP`, and `dPH`;
- emit a compact Markdown summary and detailed JSON.

## Commands

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 \
  python proofs/v014/dynamic_field_attribution.py
python -m json.tool proofs/v014/dynamic_field_attribution.json >/tmp/dynamic_field_attribution.validated.json
python -m py_compile proofs/v014/dynamic_field_attribution.py
```

## Acceptance Criteria

- Script exits 0 CPU-only.
- JSON validates and contains enough cell/lead/level detail for a same-state
  tendency harness without rerunning the broad comparator.
- Markdown is concise: verdict, top dynamic fields, first/worst leads, selected
  cells, and next same-state term target.
- No pass/fail equivalence claim is made.

## Closeout

Close with proof paths, commands, selected lead/cells, top suspects, and whether
the next sprint should build WRF term savepoints or first run an existing JAX
same-state/operator probe.
