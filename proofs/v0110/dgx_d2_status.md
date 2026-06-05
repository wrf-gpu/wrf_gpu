# v0.11.0 DGX-D2 sharded operational forecast

- verdict: PASS
- operational sharded forecast parity: True
- bit-identical: False
- within tolerance: True
- flag-off graph unchanged: True
- flag-off selection identity: True
- fake/local devices: 3 of 3
- run_boundary in sharded proof: False
- radiation on step 1: False

## What This Proves

- Disabled sharding still selects the exact existing `run_forecast_operational` function.
- The flag-off compiled graph has unchanged op count and zero collective/SPMD tokens.
- A real d02 replay state runs through the operational forecast entrypoint on x-sharded fake/local devices and is compared with the single-device reference.
- Non-dry physics/carry fields are bit-identical in the one-step proof; dry dynamic fields are within the recorded absolute tolerances.

## Max Differences

- `theta`: max_abs=0.00902884994542319 atol=0.01 exact=False
- `u`: max_abs=0.016005361576425958 atol=0.02 exact=False
- `v`: max_abs=7.735405308473275e-05 atol=0.0001 exact=False
- `w`: max_abs=0.003366667443927329 atol=0.004 exact=False
- `mu`: max_abs=1.1188932126242435 atol=1.2 exact=False
- `p`: max_abs=3.25187722199189 atol=3.5 exact=False
- `ph`: max_abs=0.20959438425779808 atol=0.25 exact=False
- `qv`: max_abs=0.0 atol=0.0 exact=True
- `qke`: max_abs=0.0 atol=0.0 exact=True
- `rain_acc`: max_abs=0.0 atol=0.0 exact=True

## Carry

- Real DGX performance, NCCL behavior, and transfer cleanliness still require hardware.
- `run_boundary=True` is intentionally not claimed for sharded execution until specified/nested boundary decomposition is implemented.
- The fake-mesh proof exercises local-device `pmap` and `lax.ppermute`; real-DGX profiler artifacts are still required before any speedup claim.

## Real-DGX Smoke Checklist

1. Confirm 8 H200-class GPUs with `nvidia-smi -L` and topology with `nvidia-smi topo -m`.
2. Run the flag-off graph proof on one GPU and all 8 visible GPUs through `/tmp/wrf_gpu_run.sh`.
3. Run D1 halo/operator fake-mesh tests on real 8-GPU pmap and compare max diffs to committed proofs.
4. Run this D2 operational forecast proof with `--devices 8`, first `run_boundary=False`, then after boundary decomposition with `run_boundary=True`.
5. Capture Nsight Systems; verify collectives are only documented halo exchanges and no host/device transfers occur inside timestep loops.
6. Only after profiler artifacts exist, run 1/2/4/8 GPU weak and strong scaling and report measured speedup.
