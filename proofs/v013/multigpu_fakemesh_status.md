# v0.13 Multi-GPU S1 — `shard_map` dycore fake-mesh bit-identity

**Date:** 2026-06-08
**Branch:** `worker/opus/v013-multigpu`
**Scope:** Tier-1 #8 — advance single-node multi-GPU domain decomposition (S1)
via sharded stencils + halo exchange, validated bit-identical on a fake CPU mesh.

## Verdict

`GATE_PASS = True` (`proofs/v013/multigpu_fakemesh.json`).

The dominant dycore horizontal stencils (5th-order flux advection, 6th-order
numerical diffusion) are domain-decomposed across a `jax.sharding.Mesh` via
`jax.shard_map`, with a collective `jax.lax.ppermute` periodic-ring halo
exchange, and are **bit-identical** across partition counts on a fake 8-device
CPU mesh.

## What is new (vs. the v0.11 DGX `pmap` substrate)

The v0.11 work (`proofs/multigpu_dgx/`, `runtime/sharding.py`) partitions a real
operational `State` host-side and traces `run_forecast_operational` under a
leading device axis with `jax.pmap`. This sprint adds the current
single-program SPMD primitive — `jax.shard_map` over a named `jax.sharding.Mesh`
— for the dycore stencils directly:

- `src/gpuwrf/runtime/shard_map_dycore.py`:
  - `make_x_mesh(P)` — 1-D x-decomposition `Mesh` over `P` (fake or real) devices.
  - `periodic_ring_halo_x(...)` — collective `ppermute` ghost-cell exchange,
    called *inside* the `shard_map` body (mesh axis resolved by name).
  - `sharded_flux5_advection_x`, `sharded_sixth_order_diffusion_x` — global
    field in, sharded across the mesh, halo-refreshed, stencil applied locally,
    trimmed to owned interior, gathered to a global array.
  - `single_device_*` — the bit-identity reference kernels.

Nothing is wired into the default forecast path. Importing the module is inert;
`select_forecast_runner(ShardingConfig.disabled()) is run_forecast_operational`
still holds. The just-merged compile-speed lane (`runtime/xla_autotune.py`,
`runtime/compile_cache.py`) was **not** touched.

## Proven (fake CPU mesh)

1. **Partition-invariance bit-identity (load-bearing gate).** P=2, P=4, P=8 all
   reproduce the P=1 single-shard result with max abs diff **exactly 0.0** for
   both stencils. The domain decomposition introduces zero numerical change.
2. **`ppermute` halo correctness on a known analytic field.** A periodic
   sinusoid `sin(2*pi*i/nx)` sharded and halo-exchanged reconstructs every ghost
   cell bit-identically to the analytic value at the wrapped global index
   (max diff 0.0, P=2 and P=4).
3. **Decomposition exactness.** Applying the identical local stencil on a
   periodically padded global array, then trimming, reproduces the global
   formula **bit-for-bit in eager mode** (`padded_eager_vs_global_eager` = 0.0).
   The sharded-vs-eager-global residual is ~9e-16 — pure XLA jit float
   reassociation (the same delta a plain single-device `jit` shows), independent
   of partition count.

Tests: `tests/parallel/test_shard_map_dycore.py` — 11 passed; full
`tests/parallel/` suite — 27 passed, 0 regressions.

## Hardware honesty — REAL multi-GPU is UNMEASURED

This workstation has **one** physical RTX 5090. This proof is CPU fake-device
only and **cannot** measure:

- Real H200/Blackwell NVLink/NVSwitch bandwidth, latency, or NCCL transport;
- Collective/compute overlap or strong/weak scaling efficiency;
- Absence of host/device transfers inside a full real GPU timestep loop;
- Multi-node fabric behaviour.

Therefore any per-watt or whole-Earth-at-1km throughput claim that depends on
multi-GPU scaling stays **PROJECTED**, never **MEASURED**. The validatable
deliverable here is the fake-mesh bit-identity of the decomposed numerics.

## Reproduce

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
taskset -c 0-3 python proofs/v013/multigpu_fakemesh.py \
  --output proofs/v013/multigpu_fakemesh.json

PYTHONPATH=src JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
taskset -c 0-3 pytest -q tests/parallel/test_shard_map_dycore.py
```

`GPUWRF_JAX_CACHE=0` avoids a stale cross-machine persistent XLA AOT cache
(harmless `prefer-no-gather` SIGILL-warning noise on this host); results are
identical with or without it.

## Carry-over / next

- Real multi-GPU measurement is hardware-blocked here; defer to a DGX/H200 host
  using the smoke checklist in `proofs/multigpu_dgx/README.md`.
- This decomposition covers periodic-x horizontal stencils. Specified/nested
  lateral-boundary decomposition and the y-axis 2-D mesh remain future work
  (same boundary caveat as the v0.11 `pmap` path, which rejects
  `run_boundary=True`).
- Extending the `shard_map` path to the full coupled acoustic substep is the
  natural follow-on; the stencil-level bit-identity here de-risks it.
