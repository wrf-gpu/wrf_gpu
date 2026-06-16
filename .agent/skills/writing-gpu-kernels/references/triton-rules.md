# Triton Rules

- Use Triton when explicit tiling or register pressure control is required.
- Keep block sizes tied to profiler evidence.
- Record `num_warps`, block shape, and dtype.
- Compare against fixture before tuning.
