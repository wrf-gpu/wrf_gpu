# Sprint Contract: V0.14 Direct Grid Symptom After Live-Nest Base Fix

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Run a bounded fresh GPU forecast on the current branch after commit `7d11be42`
and directly compare d02 wrfout grid fields against CPU-WRF truth. This answers
whether the native live-nest base source fix materially improves the V10/grid
symptom, without resuming TOST.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No FP32 or memory source work.
- No production `src/` edits.
- No Hermes or Telegram.

## Write Scope

- `proofs/v014/grid_after_live_nest_base.json`
- `proofs/v014/grid_after_live_nest_base.md`
- optional helper `proofs/v014/grid_after_live_nest_base.py`
- `proofs/v014/grid_after_live_nest_base/`
- `.agent/reviews/2026-06-09-v014-grid-after-live-nest-base.md`

Scratch/output:

- `/mnt/data/wrf_gpu2/v014_grid_after_live_nest_base/**`

## Required Work

1. Verify branch head includes `7d11be42`.
2. Run exactly one bounded live-nested L2 d02 GPU forecast for case
   `20260501_18z_l2_72h_20260519T173026Z`, preferably `--hours 12` unless GPU
   lock/resource state forces a shorter but still useful run.
3. Use the repo GPU wrapper and TOST-fixed live-nested runner:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- \
  python proofs/v0120/powered_tost_n15/run_one_case_v0120.py \
    --run-root /tmp/v0120_merged_run_root \
    --cpu-truth-root /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output \
    --run-id 20260501_18z_l2_72h_20260519T173026Z \
    --hours 12 \
    --output-root /mnt/data/wrf_gpu2/v014_grid_after_live_nest_base \
    --proof-dir proofs/v014/grid_after_live_nest_base/gpu_h12
```

4. Compare d02 wrfouts against CPU-WRF truth:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python scripts/compare_wrfout_grid.py \
    --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z \
    --gpu-dir /mnt/data/wrf_gpu2/v014_grid_after_live_nest_base/l2_d02_20260501_18z_l2_72h_20260519T173026Z \
    --domain d02 \
    --init 2026-05-01T18:00:00+00:00 \
    --min-lead 1 --max-lead 12 \
    --out-json proofs/v014/grid_after_live_nest_base.json \
    --out-md proofs/v014/grid_after_live_nest_base.md \
    --progress 20
```

5. Compare new results against the pre-fix/older artifacts:
   `proofs/v014/post_static_writer_grid_compare.json`,
   `proofs/v014/grid_cell_envelope.json`, and
   `proofs/v014/v10_grid_diagnostics.json`.
6. Emit a concise verdict:
   - `GRID_SYMPTOM_MATERIALLY_IMPROVED`
   - `GRID_SYMPTOM_NOT_CLOSED`
   - `GRID_RUN_BLOCKED_<reason>`

## Acceptance Criteria

- GPU forecast exits 0 or a blocked report records exact rc/log.
- JSON validates.
- Report separates base/static improvements from dynamic V10/grid symptom.
- Report includes V10/U10/PSFC/P/MU/PH/T summary over common leads.
- No TOST is resumed.

## Closeout

Close with commands, runtime/VRAM if available, output path, proof paths,
V10 before/after summary, and next recommended dynamic-debug target.
