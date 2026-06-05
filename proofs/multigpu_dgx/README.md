# Multi-GPU / DGX Simulation Proofs

Date: 2026-06-05
Branch: `worker/gpt/multigpu-dgx`

## Status

This lane adds an optional, default-off x-sharding substrate and fake-device
verification for DGX-style execution. The default operational forecast path
remains the default and is selected as the exact existing function object when
`ShardingConfig.enabled=False`.

This lane does not claim a complete full-forecast sharded dycore integration.
The committed evidence covers:

- default-off graph invariance;
- opt-in `lax.ppermute` periodic x-halo exchange;
- State x partition/merge for mass and staggered x-face leaves;
- representative horizontal dynamics operators run on x shards;
- ppermute halo refresh followed by sharded operator execution on an 8-device
  fake CPU mesh.

## Reproduction Commands

CPU fake-device simulation only; do not use the GPU lock wrapper for these:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
taskset -c 0-27 python scripts/verify_multigpu_dgx_sim.py \
  --devices 8 --check all --output proofs/multigpu_dgx/s4_all_fake8.json
```

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
taskset -c 0-27 pytest -q \
  tests/parallel/test_sharding_config.py \
  tests/parallel/test_halo_exchange.py \
  tests/parallel/test_sharded_horizontal_ops.py
```

Any real-GPU command must be wrapped:

```bash
/tmp/wrf_gpu_run.sh env PYTHONPATH=src taskset -c 0-27 pytest -q tests/parallel
```

## Proof Objects

- `s1_flag_off_graph.json`: disabled sharding selects the existing
  `run_forecast_operational` function; HLO op count matches the reference at
  145 ops and contains zero collective/SPMD tokens.
- `s2_halo_exchange.json`: fake-device periodic x-halo exchange matches global
  slices for widths 1-4 over mass-grid and x-face staggered State leaves.
- `s3_horizontal_operators.json`: halo-fed local x operators match global
  formulas for flux5, sixth-order diffusion, acoustic x divergence, and
  acoustic x-face pressure `dpn`.
- `s4_e2e_fake8.json`: starts from unfilled x shards, refreshes halos with
  `lax.ppermute`, runs sharded operators, and matches single-domain references
  on an 8-device fake CPU mesh.
- `s4_all_fake8.json`: consolidated fake 8-device proof covering flag-off,
  halo, operator, and ppermute-plus-operator checks.
- `s5_scaling_projection.json`: analytical projection only, not a hardware
  measurement.

## What Simulation Proves

- The sharding mode is opt-in and default-off.
- Disabled sharding does not add collective ops to the compiled default graph.
- The fake pmap axis, rank ordering, and `lax.ppermute` send/receive topology
  work for 8 local devices.
- Periodic x halos are correct for widths 1-4.
- The selected sharded horizontal operators reproduce owned outputs from the
  current single-domain formulas within recorded tolerances.
- Column-local physics remains structurally shard-friendly because no tested
  physics coupler needs cross-column communication in these sharding helpers.

## What Simulation Cannot Prove

- Real H200/NVLink bandwidth, latency, or collective overlap.
- NCCL transport behavior.
- Host/device transfer absence inside the full real timestep loop.
- Full operational sharded forecast parity, because this lane stops at
  representative horizontal operators plus halo plumbing.
- Multi-node launcher behavior, InfiniBand fabric behavior, or strong scaling.

## Real-DGX Smoke Checklist

1. Confirm hardware and environment:
   - `nvidia-smi -L` shows 8 H200-class GPUs.
   - `nvidia-smi topo -m` shows expected NVLink/NVSwitch topology.
   - JAX sees 8 GPU devices and x64 is enabled.
2. Default-off proof:
   - run `--check flag-off` on one GPU and on all 8 visible GPUs;
   - verify 145 HLO ops, reference/disabled HLO match, and zero collectives.
3. Halo and operator smoke:
   - run the parallel pytest suite through `/tmp/wrf_gpu_run.sh`;
   - run `--check all` on 8 GPUs and compare max diffs to committed fake proof.
4. Transfer and collective audit:
   - capture Nsight Systems trace;
   - verify collectives correspond only to documented halo exchanges;
   - verify no host/device transfers occur inside timestep loops.
5. Scaling measurement:
   - run 1, 2, 4, 8 GPU weak and strong scaling fixtures;
   - save profiler artifacts before reporting any measured speedup.
6. Multi-node extension:
   - configure `jax.distributed.initialize` with stable coordinator/process ids;
   - repeat halo/operator parity across at least two nodes;
   - do not report multi-node scaling until InfiniBand/NCCL traces are saved.
