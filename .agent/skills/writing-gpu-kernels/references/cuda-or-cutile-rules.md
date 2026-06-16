# CUDA Or CUDA Tile Rules

- Use explicit CUDA-family kernels only when bakeoff evidence justifies lower-level control.
- Record launch configuration, shared memory use, register count, and portability cost.
- Do not choose NVIDIA-only paths without ADR.
