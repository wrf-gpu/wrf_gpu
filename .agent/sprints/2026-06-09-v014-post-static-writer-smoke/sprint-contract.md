# Sprint Contract: V0.14 Post-Static Writer GPU Smoke

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Produce a fresh short GPU wrfout artifact after commit `a42865e8` so the grid
comparator no longer reasons from retained pre-fix writer output for vertical
metrics and map factors.

This is a targeted writer/static proof, not a TOST run and not an equivalence
claim.

## Priority Context

Current priority order:

1. Grid-cell parity and root-cause fixes.
2. FP32 acoustic / mixed precision.
3. Remaining memory issues.
4. Powered TOST only after grid fields are no longer radically divergent.

TOST remains paused. This sprint may use the GPU only for a short h1 replay
smoke while CPU-only attribution work runs in parallel.

## Scope

Allowed write scope:

- `proofs/v014/post_static_writer_smoke/`
- `proofs/v014/post_static_writer_grid_compare.json`
- `proofs/v014/post_static_writer_grid_compare.md`
- `.agent/reviews/2026-06-09-v014-post-static-writer-smoke.md`
- Manager handoff docs.

No source edits are allowed in this sprint.

## Case And Command

Use retained Case 3:

- run id: `20260501_18z_l2_72h_20260519T173026Z`
- CPU run root: `/mnt/data/canairy_meteo/runs/wrf_l2`
- expected CPU compare dir:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`

Run through the versioned GPU wrapper and the TOST-fixed live-nested per-case
runner. Do not use `scripts/m7_l2_d02_replay.py` for this L2 case: that old
single-domain d02 replay path demands a non-existent `wrfbdy_d02`, while the
valid TOST path is live-nested d01->d02 with d02 LBC from the parent.

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- \
  python proofs/v0120/powered_tost_n15/run_one_case_v0120.py \
    --run-root /tmp/v0120_merged_run_root \
    --cpu-truth-root /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output \
    --run-id 20260501_18z_l2_72h_20260519T173026Z \
    --hours 1 \
    --output-root /tmp/v014_post_static_writer_smoke \
    --proof-dir proofs/v014/post_static_writer_smoke/live_nested_h1
```

Then compare h1:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python scripts/compare_wrfout_grid.py \
    --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z \
    --gpu-dir /tmp/v014_post_static_writer_smoke/l2_d02_20260501_18z_l2_72h_20260519T173026Z \
    --domain d02 \
    --min-lead 1 --max-lead 1 \
    --out-json proofs/v014/post_static_writer_grid_compare.json \
    --out-md proofs/v014/post_static_writer_grid_compare.md
```

## Acceptance Criteria

- The GPU smoke exits 0 and writes at least one d02 wrfout.
- The comparator h1 JSON validates.
- `C1H/C2H/C3H/C4H/C1F/C2F/C3F/C4F/DN/DNW/RDN/RDNW/MAPFAC_*`
  no longer appear as large retained-writer mismatches.
- Any remaining static/base-state fields are classified as one of:
  writer-only fallback, CPU wrfinput-vs-wrfout convention, or unresolved.
- Dynamic divergence is not declared fixed by this sprint.
- No TOST is resumed.

## Closeout

Close with commands run, GPU runtime/VRAM if available, proof paths, comparator
top fields, and the exact next dynamic-debug action.
