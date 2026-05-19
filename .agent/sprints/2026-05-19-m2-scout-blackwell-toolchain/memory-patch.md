# Memory Patch Proposal

## Scope

One genuine standing rule to add to auto-memory: the **manager-owned readiness ordering pattern** for multi-candidate sprints. Plus one project-fact to capture: the Blackwell driver/CUDA versions on this system.

## Evidence

- M2-S1 produced a readiness matrix that drives the order of S2..S7. This pattern (research-scout-first → ordered implementation sprints) will repeat in M3 (state layout candidates), M4 (dycore numeric choices), M5 (physics scheme choices). Reusable.
- Driver 590.48.01 / CUDA 13.1 / NVHPC 26.3 / Kokkos 4.7.1 will be referenced by every M2 implementation sprint as the build context. Worth capturing once.

## Proposed Destination

A new memory entry of type `project` at `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/project_target_hardware.md`. Index entry in `MEMORY.md`.

## Patch

```markdown
---
name: Target hardware + toolchain baseline (RTX 5090 Blackwell)
description: System the project must hit its 4-8x CPU speedup target on; pinned driver/CUDA/compiler versions verified by M2-S1 scout
type: project
---

The single-node target for v0 is one **NVIDIA GeForce RTX 5090**, compute capability **12.0 (Blackwell)**, 32 GB VRAM. Verified by `nvidia-smi` during M2-S1 (2026-05-19):

- Driver: 590.48.01
- CUDA runtime: 13.1
- NVHPC SDK: 26.3 (provides nvcc / nvfortran / nvc++)
- Kokkos: 4.7.1 with `Kokkos_ARCH_BLACKWELL120=ON` (source build required)
- jaxlib CUDA-13 wheel: 0.10.0
- triton: 3.7.0 with torch 2.12.0 CUDA-13
- cupy-cuda13x: 14.0.1

**Why:** the 4-8x wall-clock-vs-CPU target is measured on THIS machine, not a generic GPU. Toolchain versions are pinned because Blackwell is new enough that older wheels do not work. M2-S1 hello-GPU smoke tests verified all five non-blocked candidates actually compute on the device.

**How to apply:** any sprint that touches the GPU (M2+, M3+ implementation sprints, M4 dycore, M5 physics, M7 ops) should source `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh` for NVHPC paths, then install candidate-specific deps into a sprint-local venv at `data/scratch/<sprint>-venv/`. gt4py is blocked on Python 3.13 + DaCe 0.10.0; use a Python 3.12 venv if attempting gt4py remediation.

CPU baseline for the 4-8x measurement: 28-rank CPU WRF on the same workstation (Ryzen 9 / 96 GB RAM). The previous `../wrf_gpu/` attempt established that as the operational baseline.
```

## Reviewer Status

Reviewer Status: not required — this is a factual capture, not a behavioral rule. The reviewer's Decision (Accept) already validates the hardware/version evidence indirectly.
