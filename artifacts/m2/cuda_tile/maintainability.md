Build complexity: one Makefile (22 lines) and one build wrapper (21 lines). Runtime system deps are CUDA/NVHPC plus zlib; the host driver vendors only a minimal NPZ reader/writer in C++.

Error legibility: a deliberate temporary typo (`phi_nxt`) in `data/scratch/cuda_tile/stencil_bug.cu` produced a direct nvc++ diagnostic pointing at the bad line. Plain `nvcc` currently fails earlier on a CUDA 13.1 + glibc/GCC 15 `rsqrt` prototype conflict, so `build.sh` retries with `nvc++ -cuda -gpu=cc120`.

Debugger story: `cuobjdump --dump-resource-usage` reports sm_120 code, registers, and zero local memory. `ncu` connects but exits with `ERR_NVGPUCTRPERM` under this user; logs are in `data/profiler_artifacts/cuda_tile/`. Compute Sanitizer was not required by the sprint contract.

Agent-iteration friction: five compile/build-test cycles, including the NPZ writer fix, CUDA header/toolchain fallback, occupancy reporting, and the temporary bug check.
