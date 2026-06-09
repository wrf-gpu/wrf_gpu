# Tester Report

## Tests Added Or Run

The sprint ran one bounded h12 GPU forecast and a CPU-only wrfout grid
comparison over d02 h1-h12. The manager reran JSON validation and helper compile
after worker completion.

Commands:

- `git merge-base --is-ancestor 7d11be42 HEAD`
- `scripts/run_gpu_lowprio.sh --cores 0-23 -- python proofs/v0120/powered_tost_n15/run_one_case_v0120.py ... --hours 12 ...`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python scripts/compare_wrfout_grid.py ... --min-lead 1 --max-lead 12 ...`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/grid_after_live_nest_base.py`
- `python -m json.tool proofs/v014/grid_after_live_nest_base.json >/tmp/grid_after_live_nest_base.manager.validated.json`
- `python -m json.tool proofs/v014/grid_after_live_nest_base/gpu_h12/l2_d02_validation_summary.json >/tmp/grid_after_live_nest_base.summary.manager.validated.json`
- `python -m json.tool proofs/v014/grid_after_live_nest_base/gpu_h12/wall_clock_l2_d02.json >/tmp/grid_after_live_nest_base.wall.manager.validated.json`
- `python -m py_compile proofs/v014/grid_after_live_nest_base.py`

## Results

GPU run status was `L2_D02_GREEN` with `bounds`, `pipeline`, `rmse`, and
`wall_clock` all passing inside `l2_d02_validation_summary.json`.

The grid comparator completed over 12 paired d02 files and emitted valid JSON.
The synthesized verdict is `GRID_SYMPTOM_NOT_CLOSED`.

Required field summary:

- `V10`: RMSE `2.55039100124724`, worst lead h11 RMSE `4.277008742661733`
- `U10`: RMSE `1.7111033260122948`
- `PSFC`: RMSE `517.1905702423264`
- `P`: RMSE `230.30713670774634`
- `MU`: RMSE `266.52491970646497`
- `PH`: RMSE `292.3872984317863`
- `T`: RMSE `1.169125225842937`

## Fixtures Used

- CPU truth:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- GPU output:
  `/mnt/data/wrf_gpu2/v014_grid_after_live_nest_base/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
- Older comparison artifacts:
  `proofs/v014/post_static_writer_grid_compare.json`,
  `proofs/v014/grid_cell_envelope.json`, and
  `proofs/v014/v10_grid_diagnostics.json`

## Gaps

Peak VRAM was observed externally during the run but was not captured by the
runner artifact, so the committed proof records VRAM as not recorded. The
comparison is report-only because no tolerance manifest was supplied. TOST and
Switzerland validation were not run.

Decision:

Accept as a valid direct grid not-closed proof. Reject any release or TOST
closure claim from this artifact.
