Build complexity: one 12-line CMake file plus a 46-line build wrapper. First run cloned Kokkos 4.7.1 and built/installed CUDA+BLACKWELL120 under `data/scratch/kokkos-{src,build,install}` in about 41 seconds on this workstation; cached reruns rebuild only the bench. Disk use after the first run: source 21 MB, install 9.4 MB, Kokkos build 12 MB, bench build 6.8 MB.

Error legibility: a deliberate typo in `data/scratch/kokkos/stencil_bug.cpp` produced a direct undefined-identifier diagnostic, but it was preceded by the known CUDA 13.1/GCC 15 `rsqrt` header conflict from plain `nvcc_wrapper`; see `data/scratch/kokkos/deliberate_bug_stderr.txt`.

Debugger story: `cuobjdump --dump-sass` shows `sm_120`; `cuobjdump --dump-resource-usage` exposes Kokkos functor names, registers, and zero local memory. `bench config` records Kokkos 4.7.1, CUDA execution space, and runtime compute capability 12.0.

Agent-iteration friction: moderate. Kokkos required source-install orchestration and resource-parser tuning, but the final kernel code stayed compact and correctness passed without numerical special cases. `ncu` still hits `ERR_NVGPUCTRPERM`, so profile metrics use the project-standard fallback.
