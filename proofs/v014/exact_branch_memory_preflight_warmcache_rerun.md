# v0.14 Exact-Branch Memory Preflight

- Verdict: `PASS_SHORT_GPU_PREFLIGHT`
- Branch: `worker/mythos/v014-memory-fp32`
- HEAD: `a32efce328520e544d28605d81e5d921db06de1e`
- Dirty worktree: `True`

## Static Controls

- RRTMG column tiling present: `True`
- Nested allocator/segmentation controls present: `True`

## GPU Run

- Command: `/home/enric/miniconda3/bin/python -m gpuwrf.cli run --input-dir /mnt/data/canairy_meteo/runs/wrf_l3/20260531_18z_l3_24h_20260601T125256Z --output-dir /mnt/data/wrf_gpu_validation/v014_exact_branch_memory_preflight_20260609T222217Z/nested_1h_out --scratch-dir /mnt/data/wrf_gpu_validation/v014_exact_branch_memory_preflight_20260609T222217Z/scratch --proof-dir /mnt/data/wrf_gpu_validation/v014_exact_branch_memory_preflight_20260609T222217Z/proofs --max-dom 3 --hours 1`
- Output path: `/mnt/data/wrf_gpu_validation/v014_exact_branch_memory_preflight_20260609T222217Z/nested_1h_out`
- Duration: `378.366` s
- Return code: `0`
- Nested payload verdict: `PIPELINE_GREEN`
- All finite: `True`
- All outputs present: `True`
- Output count: `3`
- Peak total VRAM: `9184` MiB
- Peak compute-app VRAM: `8116` MiB
- Peak increment over baseline: `6546` MiB
- Allocator re-exec line seen: `True`
- OOM markers: `0`

## Caveats

- This is a memory-fit preflight, not TOST, not a long validation, and not a skill/equivalence claim.
- The nested run intentionally uses the platform allocator path; peak VRAM is from nvidia-smi sampling, not JAX memory_stats.
- This is not a full transfer audit. Hourly wrfout preparation necessarily moves output payloads to host; no claim is made that every loop is transfer-free.
- Feedback is recorded as a setting. If the next long validation enables feedback, rerun this preflight with `--feedback`.

## Next

Run V014-MEM-1: empirical memory map on the same exact branch, measuring MYNN BouLac, non-radiation column physics, post-physics merge, and moisture limiter liveness before any new memory rewrite.
