# Switzerland CPU24 Reference Resource Summary

Status: `PASS`

Date: 2026-06-10 WEST.

Purpose: create a tracked 24-rank CPU-WRF Switzerland/Gotthard reference run so
the later v0.14 GPU-vs-CPU Switzerland validation has a matched CPU timing and
memory baseline.

Run artifacts:

- Runroot: `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_cpu24_20260610T073414Z`
- CPU dir: `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_cpu24_20260610T073414Z/run_cpu`
- Resource dir: `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_cpu24_20260610T073414Z/resources/cpu`
- Summary JSON: `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_cpu24_20260610T073414Z/cpu24_resource_summary.json`
- Summary markdown: `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_cpu24_20260610T073414Z/cpu24_resource_summary.md`

Run configuration:

- WRF build: `dmpar MPI gfortran`
- Ranks: `24`
- CPU cores: `4-27`
- Grid: `128x128` mass points, `45` full levels, `dx=dy=3000 m`
- Time step: `18 s`
- Forecast hours: `24`
- Launch branch/head: `95fb7b40ce8da5beb8ba457906ab886a5ee975ea`
- Resource-monitor code head after HUP-hardening: `66cc65fa88d2420043af0df33988ec34a360c21b`

WRF result:

- `wrf.exe rc=0`
- `SUCCESS COMPLETE WRF`
- `wrfout` count: `25`
- First output: `wrfout_d01_2023-01-15_00:00:00`
- Last output: `wrfout_d01_2023-01-16_00:00:00`
- Last-frame finite check: `PASS` for `T2`, `U10`, `V10`, `PSFC`, `T`, `U`,
  `V`, and `QVAPOR`

Timing:

- Total wall: `1084.6 s`
- Total wall per forecast hour: `45.19 s`
- Mainloop sum: `1078.4 s`
- Mainloop per forecast hour: `44.93 s`
- Mainloop steps: `4800`

Resource summary:

- Process CSV:
  `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_cpu24_20260610T073414Z/resources/cpu/switzerland_cpu24_live_process_usage.csv`
- Host-memory CSV:
  `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_cpu24_20260610T073414Z/resources/cpu/switzerland_cpu24_live_system_memory.csv`
- Initial launch samples:
  `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_cpu24_20260610T073414Z/resources/cpu/initial_switzerland_cpu24_process_usage.csv`,
  `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_cpu24_20260610T073414Z/resources/cpu/initial_switzerland_cpu24_system_memory.csv`
- WRF-rank process rows: `3936`
- System-memory samples: `166`
- Peak single-rank RSS: `579.203 MiB` at `2026-06-10T08:39:39+01:00`
- Peak sampled total WRF-rank RSS: `12563.766 MiB` across `24` ranks at
  `2026-06-10T08:37:00+01:00`
- Minimum sampled host `MemAvailable`: `50505.062 MiB` at
  `2026-06-10T08:36:47+01:00`

Notes:

- The first monitor process captured launch-time samples, then was replaced by
  a detached live monitor after `scripts/monitor_resource_usage.sh` was
  HUP-hardened in `66cc65fa`.
- This is CPU reference evidence only. It is not a GPU-vs-CPU equivalence
  verdict and does not unblock Switzerland-GPU before the Step-1/grid-parity
  frontier is closed or explicitly bounded.

