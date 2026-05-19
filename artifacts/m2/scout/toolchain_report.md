# M2 Blackwell Toolchain Scout

Generated: 2026-05-19T08:29:27Z.

Target hardware: `nvidia-smi` reports `NVIDIA GeForce RTX 5090`, compute capability `12.0`, 32607 MiB VRAM, driver `590.48.01`, CUDA runtime `13.1`. The Gen2 environment exposes NVHPC `nvc++ 26.3` and `nvcc 13.1.115`.

## jax

Version pinned: `jax[cuda13]==0.10.0`.
Install command: `source data/scratch/m2-scout-venv/bin/activate && pip install 'jax[cuda13]==0.10.0'`.
Upstream context: JAX documents CUDA 13 wheels via `pip install --upgrade "jax[cuda13]"` and says those wheels are built for CUDA 13.0 compatibility (`https://docs.jax.dev/en/latest/installation.html`).
Hello-GPU result: pass in `artifacts/m2/scout/hello_gpu/jax/output.txt`; JAX saw `CudaDevice(id=0)` and returned `[2.0, 4.0, 6.0, 8.0]`.
Gaps: functional only; no profiler, transfer audit, or stencil evidence.
Verdict: `go-with-version-bump`.

## triton

Version pinned: `triton==3.7.0`, `torch==2.12.0`.
Install command: `source data/scratch/m2-scout-venv/bin/activate && pip install 'triton==3.7.0' 'torch==2.12.0'`.
Upstream context: Triton 3.7 release notes mention Blackwell work (`https://github.com/triton-lang/triton/releases`), and PyTorch 2.12 says newer GPUs such as Blackwell should use CUDA 13.0+ wheels (`https://pytorch.org/blog/pytorch-2-12-release-blog/`).
Hello-GPU result: pass in `artifacts/m2/scout/hello_gpu/triton/output.txt`; the Triton JIT kernel ran on the 5090 through PyTorch CUDA and returned `[2.0, 4.0, 6.0, 8.0]`.
Gaps: current Triton driver activation required `torch.cuda`; no performance data.
Verdict: `go-with-version-bump`.

## gt4py

Version attempted: `gt4py==1.1.9`, `dace==0.10.0`.
Install command attempted: `source data/scratch/m2-scout-venv/bin/activate && pip install 'gt4py==1.1.9' 'dace==0.10.0'`.
Upstream context: GT4Py's quick-start documents GPU extras in the `cuda11x` family and DaCe documents CUDA as required for NVIDIA GPU use (`https://gridtools.github.io/gt4py/latest/quickstart.html`, `https://spcldace.readthedocs.io/en/latest/setup/installation.html`).
Hello-GPU result: blocked in `artifacts/m2/scout/hello_gpu/gt4py/output.txt`. In this Python 3.13 sprint venv, DaCe failed before code generation: first on Python 3.13 AST handling, then after the Triton/PyTorch install exposed a hard `sympy` dependency conflict (`dace==0.10.0` pins `sympy==1.5.1`, while `torch==2.12.0` requires modern SymPy).
Gaps: no clean CUDA 13/Blackwell install path established in this sprint.
Verdict: `blocked`.

## kokkos

Version pinned: Kokkos `4.7.1` tag `4.7.01`.
Install command: local source build under `data/scratch` with `Kokkos_ENABLE_CUDA=ON`, `Kokkos_ARCH_BLACKWELL120=ON`, and `nvcc_wrapper`.
Upstream context: Kokkos documents `Kokkos_ARCH_BLACKWELL120`, compute capability `12.0`, as available since Kokkos 4.7 (`https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html`).
Hello-GPU result: pass in `artifacts/m2/scout/hello_gpu/kokkos/output.txt`; execution space was `Cuda` and result was `[2, 4, 6, 8]`.
Gaps: build setup is heavier than Python candidates; no bakeoff kernel yet.
Verdict: `go-with-version-bump`.

## cupy_or_numba

Version pinned: `cupy-cuda13x==14.0.1`; CuPy was selected over Numba for this candidate.
Install command: `source data/scratch/m2-scout-venv/bin/activate && pip install 'cupy-cuda13x==14.0.1'`.
Upstream context: CuPy 14 documents CUDA 13 wheel support and `cupy-cuda13x` (`https://docs.cupy.dev/en/stable/upgrade.html`).
Hello-GPU result: pass in `artifacts/m2/scout/hello_gpu/cupy_or_numba/output.txt`; CuPy ran on `NVIDIA GeForce RTX 5090` and returned `[2.0, 4.0, 6.0, 8.0]`.
Gaps: array expression only; M2 still needs raw-kernel ergonomics.
Verdict: `go`.

## cuda_tile

Version pinned: CUDA Toolkit `13.1.115` from NVHPC `26.3`.
Install command: `source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh && nvcc -std=c++17 -arch=sm_120 -O2 hello.cu -o hello_cuda_tile`.
Upstream context: NVIDIA's CUDA 13.1 compiler docs list `compute_120` and `sm_120` as supported Blackwell targets (`https://docs.nvidia.com/cuda/pdf/CUDA_Compiler_Driver_NVCC.pdf`).
Hello-GPU result: pass in `artifacts/m2/scout/hello_gpu/cuda_tile/output.txt`; the native CUDA kernel returned `[2, 4, 6, 8]`.
Gaps: NVIDIA-only manual path; no maintainability or performance evidence yet.
Verdict: `go`.

## Closing Recommendation

Dispatch M2 implementation sprints in this readiness order: `cuda_tile`, `cupy_or_numba`, `kokkos`, `jax`, `triton`. Keep `gt4py` out of S2-S7 unless a follow-up scout finds a Python-version and DaCe/GT4Py combination with a clean CUDA 13 GPU codegen path. This gives the manager two direct CUDA baselines first, then Kokkos, then the higher-level Python compiler stacks.
