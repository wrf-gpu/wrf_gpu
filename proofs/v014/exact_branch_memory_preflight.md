# v0.14 Exact-Branch Memory Preflight

- Verdict: `NO_RUN_PLAN`
- Branch: `worker/gpt/v013-close-manager`
- HEAD: `5865d41112c6f36ebc648834da29a06b7919d8a2`
- Dirty worktree: `True`

## Static Controls

- RRTMG column tiling present: `True`
- Nested allocator/segmentation controls present: `True`

## GPU Run

- Run attempted: `False`
- Reason: `audit-only mode; pass --run-gpu through scripts/run_gpu_lowprio.sh`
- Planned command: `scripts/run_gpu_lowprio.sh --cores 0-23 -- python proofs/v014/exact_branch_memory_preflight.py --run-gpu --nested-input /mnt/data/canairy_meteo/runs/wrf_l3/20260531_18z_l3_24h_20260601T125256Z --max-dom 3 --hours 1 --timeout-s 600.0`

## Observed Timed-Out Attempt

- Command: `scripts/run_gpu_lowprio.sh --cores 0-23 -- python proofs/v014/exact_branch_memory_preflight.py --run-gpu --timeout-s 600.0`
- Run root: `/mnt/data/wrf_gpu_validation/v014_exact_branch_memory_preflight_20260608T223250Z`
- Duration class: `timed_out_or_reached_cap`
- Duration cap: `600.0` s
- Peak total VRAM observed: `3204.0` MiB
- Baseline total VRAM observed: `1539.0` MiB
- Peak increment observed: `1665.0` MiB
- Output count: `0`
- Completed: `False`
- No OOM observed: `True`

## Caveats

- This is a memory-fit preflight, not TOST, not a long validation, and not a skill/equivalence claim.
- The nested run intentionally uses the platform allocator path; peak VRAM is from nvidia-smi sampling, not JAX memory_stats.
- This is not a full transfer audit. Hourly wrfout preparation necessarily moves output payloads to host; no claim is made that every loop is transfer-free.
- Feedback is recorded as a setting. If the next long validation enables feedback, rerun this preflight with `--feedback`.

## Next

Run V014-MEM-1: empirical memory map on the same exact branch, measuring MYNN BouLac, non-radiation column physics, post-physics merge, and moisture limiter liveness before any new memory rewrite.
