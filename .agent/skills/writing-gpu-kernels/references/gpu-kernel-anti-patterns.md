# GPU Kernel Anti-Patterns

- CPU reads inside timestep loops.
- Tiny launch chains with global-memory round trips.
- Mixed precision without tolerance evidence.
- Per-cell Python loops.
- Hidden `.item()`, `numpy()`, or host conversion calls.
- Performance measured before warmup or correctness.
