# Worker Report

Summary:

GPT ran the bounded direct d02 grid symptom proof after the live-nest base-source
partial fix. The GPU forecast exited green, the CPU-only wrfout grid comparator
ran over h1-h12, and a proof-only synthesis helper compared the result to the
older post-static, grid-envelope, and V10 diagnostic artifacts.

Files Changed:

- `proofs/v014/grid_after_live_nest_base.py`
- `proofs/v014/grid_after_live_nest_base.json`
- `proofs/v014/grid_after_live_nest_base.md`
- `proofs/v014/grid_after_live_nest_base/gpu_h12/`
- `.agent/reviews/2026-06-09-v014-grid-after-live-nest-base.md`

Commands Run:

- `git merge-base --is-ancestor 7d11be42 HEAD`
- `scripts/run_gpu_lowprio.sh --cores 0-23 -- python proofs/v0120/powered_tost_n15/run_one_case_v0120.py --run-root /tmp/v0120_merged_run_root --cpu-truth-root /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output --run-id 20260501_18z_l2_72h_20260519T173026Z --hours 12 --output-root /mnt/data/wrf_gpu2/v014_grid_after_live_nest_base --proof-dir proofs/v014/grid_after_live_nest_base/gpu_h12`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python scripts/compare_wrfout_grid.py --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z --gpu-dir /mnt/data/wrf_gpu2/v014_grid_after_live_nest_base/l2_d02_20260501_18z_l2_72h_20260519T173026Z --domain d02 --init 2026-05-01T18:00:00+00:00 --min-lead 1 --max-lead 12 --out-json proofs/v014/grid_after_live_nest_base.json --out-md proofs/v014/grid_after_live_nest_base.md --progress 20`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/grid_after_live_nest_base.py`
- `python -m json.tool proofs/v014/grid_after_live_nest_base.json >/tmp/grid_after_live_nest_base.json.valid`
- `python -m json.tool proofs/v014/grid_after_live_nest_base/gpu_h12/l2_d02_validation_summary.json >/tmp/grid_after_live_nest_base.summary.valid`
- `python -m py_compile proofs/v014/grid_after_live_nest_base.py`

Proof Objects:

- `proofs/v014/grid_after_live_nest_base.json`
- `proofs/v014/grid_after_live_nest_base.md`
- `proofs/v014/grid_after_live_nest_base/gpu_h12/l2_d02_validation_summary.json`
- `proofs/v014/grid_after_live_nest_base/gpu_h12/wall_clock_l2_d02.json`
- `.agent/reviews/2026-06-09-v014-grid-after-live-nest-base.md`

Result:

Verdict is `GRID_SYMPTOM_NOT_CLOSED`. The GPU forecast itself is green
(`L2_D02_GREEN`) and took total wall `1192.2986149120043` s, forecast-only
`1186.442607951998` s. Peak VRAM was not recorded in the runner artifact.

The h1-h12 core dynamic fields remain far from CPU-WRF:

- `V10` RMSE `2.55039100124724`, worst h11 RMSE `4.277008742661733`
- `U10` RMSE `1.7111033260122948`
- `PSFC` RMSE `517.1905702423264` Pa
- `P` RMSE `230.30713670774634` Pa
- `MU` RMSE `266.52491970646497`
- `PH` RMSE `292.3872984317863`

Static/base fields improved strongly. `C1H/C2H/C4H/DN/RDN/MAPFAC_M/XLAT/XLONG`
are exact, `HGT` is near exact, and `PHB` max abs is `0.109375`. `PB` and `MUB`
still have max abs around `250`, so base/static is improved but not fully exact.

Handoff:

Do not resume TOST from this proof. The next target is dynamic same-state
localization in the h10-h12 window, focused on pressure-gradient/mass-wind
coupling around `PSFC/MU/P/PH/U/V/V10`.
