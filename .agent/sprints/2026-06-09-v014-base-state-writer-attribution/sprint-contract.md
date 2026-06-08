# Sprint Contract: V0.14 Base-State Writer Attribution

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Root-cause the remaining h1 CPU-WRF-vs-GPU static/base-state wrfout
differences after the stale `GridSpec.metrics` fix, without touching production
model code.

The target fields are `PHB`, `MUB`, `PB`, `HGT`, `XLAT`, and `XLONG`. The
deliverable must separate:

- CPU `wrfinput` vs CPU `wrfout` convention differences;
- GPU native input vs CPU `wrfinput`;
- GPU emitted wrfout vs GPU/input/runtime payload;
- true h1 forecast-step changes in `PB/MUB/PHB`-derived quantities;
- writer fallback or missing-state issues for lat/lon/terrain.

## Inputs

- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `proofs/v014/static_metric_base_parity.json`
- `proofs/v014/static_metric_base_parity.md`
- `proofs/v014/post_static_writer_grid_compare.json`
- `proofs/v014/post_static_writer_grid_compare.md`
- CPU truth:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- fresh post-fix GPU h1 output:
  `/tmp/v014_post_static_writer_smoke/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
- available run-root inputs under `/tmp/v0120_merged_run_root` and retained
  wrfinput/wrfbdy files discovered from the run metadata.

## Write Scope

- `proofs/v014/base_state_writer_attribution.py`
- `proofs/v014/base_state_writer_attribution.json`
- `proofs/v014/base_state_writer_attribution.md`
- `.agent/reviews/2026-06-09-v014-base-state-writer-attribution.md`

No `src/` edits. No GPU. No WRF source edits.

## Required Analysis

The probe must be CPU-only and must:

- locate the exact CPU/GPU `wrfinput_d02`, CPU `wrfout_d02` h1, and fresh GPU
  `wrfout_d02` h1 files used for comparison;
- compare `PHB`, `MUB`, `PB`, `HGT`, `XLAT`, and `XLONG` across all available
  sources with RMSE, bias, p99_abs, max_abs, finite coverage, and worst cell;
- for `PHB/PB/MUB`, test whether CPU `wrfout` differs from CPU `wrfinput` by a
  known WRF output convention or by forecast-time recomputation;
- for `HGT/XLAT/XLONG`, test whether GPU output is reproducing input, falling
  back to synthetic grid values, or missing State payloads;
- report whether each remaining field is harmless writer/output convention,
  a runtime input mismatch, or a blocker before same-state tendency work;
- keep the Markdown top-level concise and put full tables in JSON.

## Commands

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 \
  python proofs/v014/base_state_writer_attribution.py
python -m json.tool proofs/v014/base_state_writer_attribution.json \
  >/tmp/base_state_writer_attribution.validated.json
python -m py_compile proofs/v014/base_state_writer_attribution.py
```

## Acceptance Criteria

- Script exits 0 CPU-only.
- JSON validates and names exact files compared.
- Each target field has a classification: `exact`, `cpu_output_convention`,
  `writer_fallback`, `runtime_input_mismatch`, `forecast_step_change`, or
  `unresolved_blocker`.
- The report explicitly states whether dynamic same-state localization can
  proceed while excluding these fields, or whether another base-state fix must
  come first.

## Closeout

Close with proof paths, commands, per-field classifications, and the next
manager decision: base-state source fix, writer-only fix, or proceed to
same-state dynamic localization with documented exclusions.
