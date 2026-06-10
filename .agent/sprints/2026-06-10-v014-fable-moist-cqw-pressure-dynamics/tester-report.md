# Tester Report

Decision: PASS.

Manager-rerun commands:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src \
  python proofs/v014/moist_cqw_pressure_dynamics_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/moist_cqw_gpu_h4_validation.py
python -m json.tool proofs/v014/moist_cqw_pressure_dynamics_closure.json
python -m json.tool proofs/v014/moist_cqw_gpu_h4_validation.json
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src \
  python -m pytest tests/test_v014_moist_cqw_pressure_dynamics.py -q
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src \
  python -m pytest tests/test_v013_operational_smoke.py -q
python -m compileall -q src tests proofs
git diff --check
```

Results:

- CPU proof rerun: PASS.
- GPU h1-h4 validation proof: `MOIST_CQW_GPU_H4_ACCEPT`.
- JSON validation: PASS.
- Focused unit test: `4 passed`.
- Default-ON operational smoke: `39 passed, 5 warnings`.
- `compileall`: PASS.
- `git diff --check`: PASS.

GPU gate:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 \
  --resource-log-dir /mnt/data/wrf_gpu_validation/v014_canary_d02_moistcqw_h4_20260610T165255Z/resources \
  --resource-label v014_canary_d02_moistcqw_h4 --resource-interval 5 -- \
  env GPUWRF_MOIST_CQW=1 JAX_ENABLE_COMPILATION_CACHE=false \
  python proofs/v0120/powered_tost_n15/run_one_case_v0120.py \
    --run-root /mnt/data/canairy_meteo/runs/wrf_l2 \
    --cpu-truth-root /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output \
    --run-id 20260501_18z_l2_72h_20260519T173026Z \
    --hours 4 \
    --output-root /mnt/data/wrf_gpu_validation/v014_canary_d02_moistcqw_h4_20260610T165255Z/gpu_output \
    --proof-dir /mnt/data/wrf_gpu_validation/v014_canary_d02_moistcqw_h4_20260610T165255Z/proofs
```

GPU rc `0`; resource CSV peak `16921 MiB`.
