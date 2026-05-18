# Performance Targets

## Hard Rules

- No host/device transfer inside timestep loops unless an ADR explicitly approves it.
- No GPU optimization claim without a profiler artifact.
- Benchmark names and results must be machine-readable.
- Single-GPU RTX 5090 comes first. Multi-GPU comes later.

## Required Metrics

- wall time
- memory traffic
- host/device transfer count and bytes
- kernel launch count
- occupancy
- register pressure
- local-memory spill indicators
- achieved memory bandwidth
- compile time and warmup time where relevant

## Benchmark JSON Schema

```json
{
  "benchmark": "m2_stencil_candidate",
  "backend": "jax|triton|gt4py-dace|cupy|numba|cuda-tile|other",
  "hardware": "RTX 5090 32GB",
  "case": "fixture-id",
  "wall_time_s": 0.0,
  "kernel_launches": 0,
  "host_device_transfer_bytes": 0,
  "occupancy_pct": null,
  "registers_per_thread": null,
  "local_memory_bytes": null,
  "artifact_paths": []
}
```
