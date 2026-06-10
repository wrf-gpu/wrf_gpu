# Switzerland/Gotthard CPU-WRF 72 h Reference Summary

Date: 2026-06-10

Status: **PASS**.

Run root:
`/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z`

CPU truth:
`/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`

## Result

| Item | Value |
|---|---:|
| WRF return code | 0 |
| `wrfout_d01_*` frames | 73 |
| First frame | `wrfout_d01_2023-01-15_00:00:00` |
| Last frame | `wrfout_d01_2023-01-18_00:00:00` |
| Grid | 129 x 129 outer, 128 x 128 x 44 mass grid |
| Forecast horizon | 72 h |
| CPU ranks | 24 dmpar MPI ranks |
| Total wall | 2906.3 s |
| Mainloop sum | 2887.6 s |
| Mainloop per forecast hour | 40.11 s/h |

The final frame finite check passed for `T2`, `U10`, `V10`, `PSFC`, `T`, `U`,
`V`, and `QVAPOR`.

## Resource Logging

CSV artifacts:

- process/RSS:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/resources/switzerland_72h_cpu_process_usage.csv`
- system memory:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/resources/switzerland_72h_cpu_system_memory.csv`

Peaks:

| Metric | Value |
|---|---:|
| Peak sum of 24 `wrf.exe` rank RSS | 12636.176 MiB |
| Peak single `wrf.exe` rank RSS | 580.184 MiB |
| Peak system used memory | 40828.090 MiB |

The WRF-rank RSS peak excludes wrapper and `prterun` process lines.

## Gate Status

The Switzerland CPU side of the v0.14 field-parity gate is complete. The matched
GPU-JAX 72 h Switzerland run should start only after:

- the Canary h1 EOS/theta field-parity blocker is closed;
- exact-branch memory preflight is green on the final candidate branch;
- the GPU run is launched through `scripts/run_gpu_lowprio.sh` with resource CSV
  logging.

Machine-readable summary:
`proofs/v014/switzerland_cpu72_reference_resource_summary.json`

