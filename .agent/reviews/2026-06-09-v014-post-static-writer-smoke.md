# V0.14 Post-Static Writer GPU Smoke

Date: 2026-06-09
Owner: manager

## Objective

Produce a fresh h1 GPU wrfout after the static-metric plumbing fix and verify
that retained pre-fix writer payload errors no longer dominate the grid
comparator.

## Verdict

PASS for the targeted writer/static gate. The corrected live-nested h1 run is
`L2_D02_GREEN`, and the post-fix h1 grid comparison shows exact equality for
the formerly-bad vertical and map-factor writer fields:

- `C1H/C2H/C3H/C4H/C1F/C2F/C3F/C4F`: `rmse=max_abs=bias=0`
- `DN/DNW/RDN/RDNW`: `rmse=max_abs=bias=0`
- `MAPFAC_M/U/V/MX/MY/UX/UY/VX/VY`: `rmse=max_abs=bias=0`

This closes the stale `GridSpec.metrics` writer-payload bug on disk. It does
not close dynamic grid parity.

## Commands Run

Wrong-path witness, intentionally not accepted as the smoke:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- \
  python scripts/m7_l2_d02_replay.py \
    --run-root /mnt/data/canairy_meteo/runs/wrf_l2 \
    --run-id 20260501_18z_l2_72h_20260519T173026Z \
    --hours 1 \
    --output-root /tmp/v014_post_static_writer_smoke \
    --proof-dir proofs/v014/post_static_writer_smoke
```

This exited `PIPELINE_BLOCKED`: the old single-domain d02 path requires missing
`wrfbdy_d02`. For L2 nested cases this is the wrong runner, not a GPU/OOM
failure.

Correct live-nested smoke:

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

Comparator:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 \
  python scripts/compare_wrfout_grid.py \
    --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z \
    --gpu-dir /tmp/v014_post_static_writer_smoke/l2_d02_20260501_18z_l2_72h_20260519T173026Z \
    --domain d02 \
    --min-lead 1 --max-lead 1 \
    --out-json proofs/v014/post_static_writer_grid_compare.json \
    --out-md proofs/v014/post_static_writer_grid_compare.md
python -m json.tool proofs/v014/post_static_writer_grid_compare.json >/tmp/post_static_writer_grid_compare.validated.json
python -m json.tool proofs/v014/post_static_writer_smoke/live_nested_h1/l2_d02_validation_summary.json >/tmp/post_static_writer_summary.validated.json
```

## Proof Objects

- `proofs/v014/post_static_writer_smoke/live_nested_h1/l2_d02_validation_summary.json`
- `proofs/v014/post_static_writer_smoke/live_nested_h1/pipeline_run_l2_d02.json`
- `proofs/v014/post_static_writer_smoke/live_nested_h1/bounds_check_l2_d02.json`
- `proofs/v014/post_static_writer_smoke/live_nested_h1/tier4_rmse_l2_d02.json`
- `proofs/v014/post_static_writer_smoke/live_nested_h1/wall_clock_l2_d02.json`
- `proofs/v014/post_static_writer_grid_compare.json`
- `proofs/v014/post_static_writer_grid_compare.md`
- `docs/GPU_RUNBOOK.md` now documents the correct L2 live-nested debug path.

## Key Numbers

- h1 live-nested run: `PIPELINE_GREEN`
- bounds/RMSE/wallclock: `PASS/PASS/PASS`
- wall clock total: `156.37 s`; forecast-only: `151.58 s`
- h1 surface RMSE vs CPU-WRF: `T2=0.447 K`, `U10=0.454 m/s`, `V10=1.288 m/s`
- h1 comparator: 1 paired file, 100 common variables, 99 numeric fields

Remaining top h1 fields:

- static/base: `PHB`, `MUB`, `PB`, `HGT`, `XLAT/XLONG`
- dynamic: `PSFC`, `MU`, `P`, `HFX`, `PBLH`, `PH`, radiation fluxes, `V/V10`

## Limits

This h1 run does not prove 24 h equivalence and does not resume TOST. It only
proves the static metric writer payload is fixed on disk and that the next
debug work should focus on base-state conventions plus dynamic divergence.

## Next

Integrate the CPU-only dynamic field attribution proof, then open the same-state
tendency localization sprint using its selected lead/cells.
