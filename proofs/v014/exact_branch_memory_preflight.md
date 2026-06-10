# v0.14 Exact-Branch Memory Preflight

- Verdict: `PASS_SHORT_GPU_PREFLIGHT`
- Branch: `worker/fable/v014-noahmp-d01-lu16`
- HEAD: `80a693e2ed7af0728ebcdcd2aea9bef92624676b`
- Dirty worktree: `True`

## Static Controls

- RRTMG column tiling present: `True`
- Nested allocator/segmentation controls present: `True`

## GPU Run

- Command: `/home/enric/miniconda3/bin/python -m gpuwrf.cli run --input-dir /mnt/data/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z --output-dir /mnt/data/wrf_gpu_validation/v014_noahmp_l2_preflight_fix_20260610T205333Z/nested_1h_out --scratch-dir /mnt/data/wrf_gpu_validation/v014_noahmp_l2_preflight_fix_20260610T205333Z/scratch --proof-dir /mnt/data/wrf_gpu_validation/v014_noahmp_l2_preflight_fix_20260610T205333Z/proofs --max-dom 2 --hours 1`
- Output path: `/mnt/data/wrf_gpu_validation/v014_noahmp_l2_preflight_fix_20260610T205333Z/nested_1h_out`
- Duration: `988.362` s
- Return code: `0`
- Nested payload verdict: `PIPELINE_GREEN`
- All finite: `True`
- All outputs present: `True`
- Output count: `2`
- Peak total VRAM: `9784` MiB
- Peak compute-app VRAM: `9010` MiB
- Peak increment over baseline: `7765` MiB
- Allocator re-exec line seen: `True`
- OOM markers: `0`

## Caveats

- This is a memory-fit preflight, not TOST, not a long validation, and not a skill/equivalence claim.
- The nested run intentionally uses the platform allocator path; peak VRAM is from nvidia-smi sampling, not JAX memory_stats.
- This is not a full transfer audit. Hourly wrfout preparation necessarily moves output payloads to host; no claim is made that every loop is transfer-free.
- Feedback is recorded as a setting. If the next long validation enables feedback, rerun this preflight with `--feedback`.

## Next

Run V014-MEM-1: empirical memory map on the same exact branch, measuring MYNN BouLac, non-radiation column physics, post-physics merge, and moisture limiter liveness before any new memory rewrite.
