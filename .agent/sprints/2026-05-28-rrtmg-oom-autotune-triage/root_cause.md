# Root Cause — RRTMG XLA OOM / Autotune Triage

## Classification

Verdict: **PARTIAL**.

The reproduced RRTMG failure is a **JAX/XLA GPU BFC allocator transient / fragmentation failure on a large RRTMG allocation**, not a proven absolute VRAM shortfall.

Evidence:

- Exact AC4 default pipeline reached XLA GPU compilation for `jit_run_forecast_operational` and logged CUDA allocation failures for 15.52 GiB, 13.97 GiB, 12.57 GiB, 11.31 GiB, and 10.18 GiB while the 200 ms `nvidia-smi` trace peaked at 30,997 MiB used / 1,114 MiB free.
- A direct full-domain `rrtmg_adapter` probe reproduced `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 8.03GiB` from `GPU_0_bfc`.
- This local JAX 0.10 PJRT path reads `XLA_PYTHON_CLIENT_ALLOCATOR`, not `TF_GPU_ALLOCATOR`; the attempted `TF_GPU_ALLOCATOR=cuda_malloc_async` run still used BFC and failed.
- The same direct RRTMG adapter completed with `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, finite theta, and active SW/LW diagnostics.

## Quantification

| Run | Proof | Peak VRAM | Status |
| --- | --- | ---: | --- |
| AC4 default 24h pipeline | `proofs/rrtmg_triage/nvidia_smi_ac4_24h_default.csv` | 30,997 MiB | XLA allocation failures logged, then pipeline blocked on nonfinite state after hour 1 |
| Direct RRTMG default allocator | `proofs/rrtmg_triage/nvidia_smi_rrtmg_adapter_default_probe.csv` | 31,396 MiB | Failed with 8.03 GiB BFC allocation OOM; trace was contended by another worker |
| Direct RRTMG with `TF_GPU_ALLOCATOR` | `proofs/rrtmg_triage/nvidia_smi_rrtmg_adapter_cuda_malloc_async_probe.csv` | 25,376 MiB | Failed; not the allocator variable JAX 0.10 uses |
| Direct RRTMG with `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` | `proofs/rrtmg_triage/nvidia_smi_rrtmg_adapter_xla_cuda_async_probe.csv` | 31,260 MiB | Passed; trace was partly contended, so peak is not clean steady-state evidence |

Exact HLO pass was **not fully proven**. The stderr identifies the phase as XLA GPU compilation of `jit_run_forecast_operational`; the local flag surface shows default `--xla_gpu_autotune_level=4`, but no XLA dump artifact was captured to name the precise pass beyond GPU compile/autotune/allocator behavior.

## Fix / Runbook

For JAX 0.10 in this repo, set before Python starts:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
```

The diagnostic harness now sets both defaults before importing JAX. The M7 pipeline script is outside this sprint's writable list, so use the same variables as a process-start runbook for pipeline runs.

`TF_GPU_ALLOCATOR=cuda_malloc_async` is not sufficient for this installed JAX/PJRT path.

## Remaining Blockers

- AC3 exact harness command still fails before RRTMG with `ImportError: cannot import name '_limit_theta_by_level' from gpuwrf.runtime.operational_mode`.
- AC4 exact pipeline command no longer demonstrates a hard process-ending OOM in the default run, but it does log XLA allocation failures and then blocks after hour 1 on `NONFINITE_STATE` in `qke` and surface flux fields. That stability issue is outside the writable RRTMG/env scope.
- AC6 speedup could not be measured because no 24h pipeline completed.
