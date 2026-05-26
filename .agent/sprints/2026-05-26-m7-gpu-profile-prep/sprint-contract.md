# Sprint Contract — M7 GPU Profile Prep (1h Canary wall-time + Nsight + D2H audit)

**Sprint ID**: `2026-05-26-m7-gpu-profile-prep`
**Created**: 2026-05-26 (post M6-CLOSED, autonomous manager loop)
**Status**: READY
**Predecessor**: commit `01b7737` — M6 CLOSED on Tier-4 RMSE; 1h Canary d02 forecast finite and physically bounded on all 3 V3 ICs.

## Objective

Measure the GPU-side performance of the now-passing 1h Canary d02 forecast on RTX 5090 Blackwell and produce the profiler artifacts required by M7 ADR / `PERFORMANCE_TARGETS.md`. This sprint does **not** make a speedup claim — it produces the GPU half of the wall-clock comparison. The CPU half is computed independently by the 28-rank WRF baseline currently running in tmux `0:1`. Once both numbers land, a follow-up M7 sprint computes the ratio.

Constitutional invariants under audit:
- **D2H inter-kernel transfers = 0** (ADR-027). This is a hard invariant, not a soft target. Any nonzero D2H inter-kernel count is a P0 blocker for M7.
- **GPU memory residency**: full timestep state lives on device; only checkpoint I/O may cross host/device.

## Acceptance

- **AC1 — Wall-clock measurement**: run `python -m gpuwrf.runtime.operational_mode` (or the operational driver invoked by `scripts/m6b_canary_1h_honest_v3.py`) on each of the 3 V3 ICs (20260429, 20260509, 20260521) for the full 1h forecast. Record:
  - Total wall-clock (cold-start, JIT-compile inclusive)
  - Total wall-clock (warm, JIT-cached)
  - Per-RK-step median wall-clock (ms/step)
  - Total RK steps in 1h
  Emit `.agent/sprints/2026-05-26-m7-gpu-profile-prep/wall_clock.json`.

- **AC2 — Nsight Systems trace**: capture `nsys profile --trace=cuda,nvtx,osrt --output=...` on a warm 60-second segment (≥100 RK steps) of the 20260521 forecast. Save `.qdrep` + `nsys stats --report cudaapisum,gputrace ...` JSON export. Emit `.agent/sprints/2026-05-26-m7-gpu-profile-prep/nsys_summary.json` with: total GPU time, kernel count, longest 10 kernels (name + duration + occupancy if available), CUDA API call counts.

- **AC3 — D2H transfer audit (HARD)**: from the Nsight trace, count `cudaMemcpy*D2H` calls inside the timestep loop (between RK-step start and RK-step end NVTX markers, or via process-of-elimination from the kernel timeline). The count **must be 0** for the inter-kernel/intra-step path. I/O at step boundaries (checkpoint, output) is allowed. Emit `.agent/sprints/2026-05-26-m7-gpu-profile-prep/d2h_audit.json` with exact counts + classification.

- **AC4 — Nsight Compute spot-check on top 3 kernels**: `ncu --set basic --kernel-regex <hot kernel name>` on each of the 3 longest kernels from AC2. Capture: registers/thread, achieved occupancy, achieved memory bandwidth, local memory bytes, achieved FLOPS. Emit `.agent/sprints/2026-05-26-m7-gpu-profile-prep/ncu_hot_kernels.json`.

- **AC5 — Reproducibility envelope**: rerun the 20260521 1h forecast 3 times (warm); compute coefficient of variation (CV) on total wall-clock. CV ≤ 5% PASS, > 5% emit BLOCKED with diagnosis.

- **AC6 — Worker report** with one of: `PASS` (all artifacts present, D2H=0), `BLOCKED-D2H` (nonzero inter-kernel D2H — top-priority M7 blocker), `BLOCKED-PROFILER` (profiler unavailable or fails on sm_120), `BLOCKED-PERF` (CV > 5%).

## Files Worker May Modify

- `scripts/m7_gpu_profile_1h.py` (NEW — orchestrator)
- `scripts/m7_nsys_stats_extract.py` (NEW — Nsight Systems summary extractor)
- `scripts/m7_d2h_audit.py` (NEW — D2H counter from nsys trace)
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/**`

## Files Worker Must Not Modify

- `src/gpuwrf/**` — measurement-only sprint, no model code change
- governance files, tests of existing dycore
- `/mnt/data/canairy_meteo/**`

## Hard Rules

1. **No model code changes.** This is pure measurement; correctness is already gated by M6 close.
2. **No JIT-cache wipe between AC5 reruns** — measure warm steady-state.
3. **CPU pinning**: `taskset -c 0-3` for the Python driver process. GPU work runs on the device.
4. **No remote push.** Local commit on `worker/gpt/m7-gpu-profile-prep` only.
5. **Profiler artifacts not committed** if larger than 10 MB — instead, write paths to artifact files into the JSON proof objects, and keep the raw `.qdrep`/`.ncu-rep` files in `/tmp/m7_profile_artifacts/`.
6. **Do not interfere with tmux `0:1`** (28-rank WRF baseline) — share only GPU; CPU cores 4-31 are reserved.
7. **Do not modify guards or operational namelist defaults.** Run the operational mode exactly as it produces the Tier-4 PASS.

## Dependencies

- M6-CLOSED (commit `01b7737`)
- RTX 5090 + driver matching `project_target_hardware.md`
- Nsight Systems + Nsight Compute installed on the workstation (verify by running `nsys --version` and `ncu --version` as AC0 preflight; if missing, emit `BLOCKED-PROFILER` cleanly)

## Proof Objects

- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/wall_clock.json` (AC1)
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/nsys_summary.json` (AC2)
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/d2h_audit.json` (AC3)
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/ncu_hot_kernels.json` (AC4)
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/reproducibility.json` (AC5)
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/worker-report.md` (AC6)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: deferred (manager triages worker report)
- Wall-time: 4-10 h (dominated by 1h forecast runs + cold JIT)
- Branch: `worker/gpt/m7-gpu-profile-prep`
- Worktree: `/tmp/wrf_gpu2_m7profile`
