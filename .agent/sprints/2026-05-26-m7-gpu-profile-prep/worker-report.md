# Worker Report — M7 GPU Profile Prep

Summary: BLOCKED-D2H. The 20260521 warmed full-hour Nsight Systems trace completed, but the D2H audit found 89 inter-kernel D2H transfers inside the profiled kernel window, totaling 102,199,152 bytes. That violates the sprint hard invariant. Warm wall-clock reproducibility for 20260521 passed with CV 0.001606877. NCU spot-checks were attempted for the top three kernels but Nsight Compute returned `ERR_NVGPUCTRPERM`, so hardware-counter metrics are blocked by GPU performance-counter permissions.

## Summary

- Decision: BLOCKED-D2H
- 20260521 warm full-hour wall-clock: 5.706958 s in AC1, 6.351514 s mean across AC5 reruns.
- AC5 reproducibility: PASS, CV 0.1607% <= 5%.
- AC2 Nsight Systems trace: PASS, raw report at `/tmp/m7_profile_artifacts/m7_20260521_warm_360.nsys-rep`.
- AC3 D2H audit: BLOCKED-D2H, 89 inter-kernel D2H copies.
- AC4 NCU spot-check: BLOCKED-PROFILER due `ERR_NVGPUCTRPERM`.
- AC1 incomplete for 20260429: `/mnt/data/canairy_meteo/runs/wrf_l3/20260429_18z_l3_24h_20260524T204451Z` has `wrfinput_d02` but no `wrfout_d02` history files, so `build_replay_case` cannot construct boundary leaves.

## Files Changed

- `scripts/m7_gpu_profile_1h.py`
- `scripts/m7_nsys_stats_extract.py`
- `scripts/m7_d2h_audit.py`
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/wall_clock.json`
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/nsys_capture_call_log.json`
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/nsys_summary.json`
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/d2h_audit.json`
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/ncu_hot_kernels.json`
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/reproducibility.json`
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/command_outputs/*`
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/worker-report.md`

No model code or governance files were modified. Raw profiler artifacts larger than 10 MB were left under `/tmp/m7_profile_artifacts/`.

## Commands Run

Full stdout/stderr is captured under `.agent/sprints/2026-05-26-m7-gpu-profile-prep/command_outputs/`.

- `nsys --version`
  - stdout: `NVIDIA Nsight Systems version 2025.5.2.266-255236693005v0`
  - stderr: empty
- `ncu --version`
  - stdout: `NVIDIA (R) Nsight Compute Command Line Profiler ... Version 2025.4.1.0`
  - stderr: empty
- `python -m py_compile scripts/m7_gpu_profile_1h.py scripts/m7_nsys_stats_extract.py scripts/m7_d2h_audit.py`
  - exit 0; stdout/stderr empty
- `taskset -c 0-3 python scripts/m7_gpu_profile_1h.py wall-clock --hours 1.0`
  - exit 2 because 20260429 data is missing required `wrfout_d02` history files.
  - 20260509: cold 102.583075 s, warm 5.872897 s, 360 RK steps.
  - 20260521: cold 106.175877 s, warm 5.706958 s, 360 RK steps.
- `taskset -c 0-3 python scripts/m7_gpu_profile_1h.py reproducibility --run-key 20260521 --hours 1.0 --runs 3`
  - exit 0; samples `[6.365868970000065, 6.345639855999934, 6.343033267999999]`, mean 6.351514 s, CV 0.001606877.
- `nsys profile --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --output=/tmp/m7_profile_artifacts/m7_20260521_warm_360 ... profile-window --steps 360`
  - exit 0; generated `/tmp/m7_profile_artifacts/m7_20260521_warm_360.nsys-rep`.
  - profiled full 360-step 1h forecast: 6.573110 s. The contract requested a 60-second segment, but the complete warm 1h forecast is shorter than 60 seconds, so I captured the whole 1h path instead of looping artificially.
- `nsys stats --report cudaapisum,gputrace --format json ...`
  - this Nsight version does not provide those older report names; stderr recorded `Report 'cudaapisum' could not be found` and `Report 'gputrace' could not be found`.
- `nsys stats --report cuda_api_sum,cuda_gpu_trace --format json ...`
  - exit 0; generated `/tmp/m7_profile_artifacts/m7_20260521_warm_360_stats_cuda_api_sum.json` and `/tmp/m7_profile_artifacts/m7_20260521_warm_360_stats_cuda_gpu_trace.json`.
- `python scripts/m7_nsys_stats_extract.py /tmp/m7_profile_artifacts/m7_20260521_warm_360.nsys-rep --sqlite /tmp/m7_profile_artifacts/m7_20260521_warm_360.sqlite --output .agent/sprints/2026-05-26-m7-gpu-profile-prep/nsys_summary.json`
  - exit 0; kernel_count 645341, summed kernel time 0.792808 s.
  - top kernels: `pcrGtsvBatchSharedMemKernelLoop<double>`, `loop_add_fusion_4`, `loop_multiply_fusion`.
- `python scripts/m7_d2h_audit.py /tmp/m7_profile_artifacts/m7_20260521_warm_360.nsys-rep --sqlite /tmp/m7_profile_artifacts/m7_20260521_warm_360.sqlite --output .agent/sprints/2026-05-26-m7-gpu-profile-prep/d2h_audit.json --marker-regex m7_profile_window`
  - exit 2; status BLOCKED-D2H.
  - D2H total trace 89; D2H inter-kernel inside window 89; D2H bytes inside window 102,199,152.
- `python scripts/m7_gpu_profile_1h.py ncu-hot-kernels --nsys-summary .agent/sprints/2026-05-26-m7-gpu-profile-prep/nsys_summary.json --output .agent/sprints/2026-05-26-m7-gpu-profile-prep/ncu_hot_kernels.json --run-key 20260521 --steps 100 --count 3`
  - exit 2; status BLOCKED-PROFILER.
  - all three NCU attempts connected to the Python process but failed with `ERR_NVGPUCTRPERM`.
- Data check:
  - `find /mnt/data/canairy_meteo/runs -maxdepth 3 -type f -name 'wrfout_d02*' -print | rg '2026-04-29|20260429' | head -50`
  - output empty.

## Proof Objects

- AC1: `.agent/sprints/2026-05-26-m7-gpu-profile-prep/wall_clock.json`
- AC2: `.agent/sprints/2026-05-26-m7-gpu-profile-prep/nsys_summary.json`
- AC2 raw trace: `/tmp/m7_profile_artifacts/m7_20260521_warm_360.nsys-rep`
- AC2 SQLite export: `/tmp/m7_profile_artifacts/m7_20260521_warm_360.sqlite`
- AC2 nsys stats JSON: `/tmp/m7_profile_artifacts/m7_20260521_warm_360_stats_cuda_api_sum.json`
- AC2 nsys stats JSON: `/tmp/m7_profile_artifacts/m7_20260521_warm_360_stats_cuda_gpu_trace.json`
- AC3: `.agent/sprints/2026-05-26-m7-gpu-profile-prep/d2h_audit.json`
- AC4: `.agent/sprints/2026-05-26-m7-gpu-profile-prep/ncu_hot_kernels.json`
- AC5: `.agent/sprints/2026-05-26-m7-gpu-profile-prep/reproducibility.json`

## Risks

- P0: nonzero D2H inter-kernel transfers violate ADR-027/M7 residency invariant. M7 should not make a GPU-residency or speedup claim until localized and eliminated.
- NCU metrics are unavailable without enabling NVIDIA GPU performance-counter access for this user/session.
- 20260429 AC1 timing is blocked by missing Gen2 `wrfout_d02` history files. Either restore the history files or amend the sprint IC set.
- The Nsight report names in the contract are stale for Nsight Systems 2025.5. I used the current equivalents and recorded the failed old-name command.
- The warm 1h forecast is shorter than 60 s, so the trace covers the whole 1h forecast rather than a 60 s repeated loop.

## Handoff

- objective: prepare M7 GPU profile proof objects for 1h Canary d02 wall-clock, Nsight trace, D2H audit, NCU top-kernel spot-check, and reproducibility.
- files changed: listed above; all edits are inside contract-owned paths.
- commands run: listed above; full stdout/stderr captured in `command_outputs/`.
- proof objects produced: listed above.
- unresolved risks: D2H blocker, NCU permissions, missing 20260429 history files, stale Nsight report names.
- next decision needed: prioritize a D2H localization/fix sprint before any M7 wall-clock ratio claim; separately decide whether to enable NCU counters or accept Nsight Systems-only hotspot evidence for this sprint.
