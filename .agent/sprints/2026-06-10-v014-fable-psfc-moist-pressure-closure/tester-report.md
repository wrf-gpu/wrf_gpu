# Tester Report

## Tests Added Or Run

Manager reran:

- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/psfc_moist_pressure_state_closure.py`
- `python -m json.tool proofs/v014/psfc_moist_pressure_state_closure.json`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest tests/test_v014_psfc_moist_hydrostatic.py -q`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest tests/test_m7_netcdf_writer.py tests/test_m7_daily_pipeline.py tests/test_async_wrfout_equiv.py tests/test_auxhist_multistream.py tests/test_auxhist_stream.py -q`
- `python -m compileall -q src tests proofs`
- `git diff --check`

## Results

The proof rerun included h24 after the manager h24 compare became available:
post-fix `PSFC` h24 expectation is bias `-58.082 Pa`, RMSE `64.202 Pa`.
CPU formula residual remains sub-Pa. New tests passed `2/2`; focused
writer/pipeline tests passed `29 passed, 1 skipped`; compileall and diff-check
passed.

## Fixtures Used

- Fixed LBC Canary run root:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_lbcfix_20260610T151455Z`
- CPU truth:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- WRF source anchor:
  `/home/enric/src/wrf_pristine/WRF`

## Gaps

No post-merge GPU h1/h4 validation has run yet. The active pre-fix 72h
characterization run was stopped after h24 and is not release evidence.

## Decision

Decision:

CPU_VALIDATION_PASS_GPU_SHORT_GATE_PENDING.
