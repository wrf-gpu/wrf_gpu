# Plan: Optional Multi-GPU / DGX Sharding Mode

Date: 2026-06-05
Author: GPT-5.5 xhigh frontrunner
Worktree: `.claude/worktrees/multigpu-dgx`
Branch: `worker/gpt/multigpu-dgx`
Base: `d054954` (`README: add consolidated 'Roadmap -- delta to a complete WRF v4 port' table`)
Scope: make multi-GPU domain decomposition an optional opt-in path for single-node DGX H200-class systems and a documented multi-node extension path, without changing the default single-GPU compiled graph.

## Decision

Recommendation: **GO, but only as an optional mode with a hard flag-off graph-invariance gate.**

The current codebase is structurally ready for an opt-in sharding layer because the persistent runtime state is an ADR-002 SoA pytree, the dycore already routes through `apply_halo(state, halo)`, and most expensive physics adapters are column-batched. The work must not touch the single-GPU default hot path beyond inert imports or host-level optional entry points. When sharding is disabled, the compiled graph for the default single-GPU entry must have identical JAXPR/HLO operation counts and no collective operations relative to the unsharded path.

No DGX is available in this sprint. Acceptance therefore depends on a simulated verification ladder:

- fake multi-device CPU mesh via `XLA_FLAGS=--xla_force_host_platform_device_count=N`;
- `jax.lax.ppermute` halo-pattern tests under `pmap`/SPMD;
- sharded-vs-global numerical equality on representative horizontal operators and a small coupled step where feasible;
- explicit real-DGX smoke checklist for later hardware.

This simulation can prove partitioning logic, halo send/receive directionality, periodic edge handling, local operator equality to global operators, and absence of default-path overhead. It cannot prove NVLink/InfiniBand performance, H200 memory bandwidth, NCCL transport behavior, or strong-scaling efficiency. Those remain projections until real hardware profiling exists.

## Non-Goals

- Do not make sharding the default.
- Do not claim DGX speedup without real DGX profiler artifacts.
- Do not rewrite the dycore or physics fidelity to fit sharding.
- Do not add host/device transfers inside timestep loops.
- Do not widen tolerances, clamp fields, or replace WRF-facing validation with JAX-vs-JAX happy paths.
- Do not tag or merge; manager owns integration.

## File Ownership And Collision Control

Owned by this lane:

- `src/gpuwrf/contracts/halo.py`
- new sharding module under `src/gpuwrf/runtime/` or `src/gpuwrf/parallel/`
- tests and proof scripts for sharding simulation
- sharded-operator variants and helpers

Shared hot files, edit only if needed and flag in commit message:

- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/contracts/grid.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/physics_couplers.py`

The preferred integration shape is a host-level optional entry point and sharding config object so `run_forecast_operational` remains the default compiled function. Any shared edit must be additive, static, and default-off.

## Sprint Plan

| Step | Goal | Files | Go / No-Go Gate | What Simulation Proves | Estimate | Early Validability |
|---|---|---|---|---|---:|---|
| **S1. Architecture + proof harness first** | Freeze optional-mode ABI, define `ShardingConfig(enabled=False)`, implement HLO/JAXPR op-count regression for flag-off default, and create fake-device runner skeleton. | `.agent/decisions/PLAN-MULTIGPU-DGX.md`, new `runtime/sharding.py`, `tests/parallel/*`, `scripts/verify_multigpu_dgx_sim.py` | **GO** only if `enabled=False` calls the same default function or lowers to identical op counts with zero `collective-permute`/`all-gather`/`all-reduce`. **NO-GO** if any disabled flag branch appears in HLO. | Default zero overhead; fake device count is usable; proof artifacts are reproducible on CPU. | 1 sprint, 400-800 LOC, medium risk | First commit after plan can run CPU-only op-count test. |
| **S2. Halo exchange + sharded State plumbing** | Replace the halo body with a real opt-in path while preserving no-op identity by default. Add x-domain State partition/merge helpers, ppermute-based periodic halo exchange for array leaves, and direction tests. | `contracts/halo.py`, `runtime/sharding.py`, `tests/parallel/test_halo_exchange.py` | **GO** if default `apply_halo(state, HaloSpec(...)) is state` still passes and sharded periodic halo slabs match global periodic slices for widths 1-4 on fake 2/4/8 devices. **NO-GO** if single-GPU `apply_halo` allocates or traces ops. | Rank topology, send/receive direction, periodic wrap, staggered x-face ownership, and SoA partition/merge correctness. | 1-2 sprints, 700-1400 LOC, high risk | Halo unit tests do not need the full forecast. |
| **S3. Sharded horizontal operators** | Add opt-in sharded variants for the current periodic horizontal stencil surface: flux-form advection, explicit diffusion, acoustic PGF face-pair/dpn helpers. Keep existing functions as global reference. | new sharded operator module(s), possible narrow imports in dynamics tests | **GO** if sharded outputs match unsharded global outputs bitwise or within predeclared fp64 round-off tolerance for deterministic fixtures. **NO-GO** if an operator needs a halo wider than declared or silently falls back to whole-domain all-gather. | Local halo width sufficiency for `jnp.roll` stencils, flux divergence correctness at shard seams, acoustic PGF seam equality. | 2-3 sprints, 1200-2500 LOC, high risk | Each operator compares isolated arrays before runtime integration. |
| **S4. Runtime opt-in integration + column physics** | Add an optional sharded forecast entry that partitions the state, runs sharded dycore kernels, executes column-local physics on local shards without collectives, and merges outputs for validation. Keep the default entry untouched. | new runtime entry, tests/proof script; only minimal `operational_mode.py` export if needed | **GO** if fake-device sharded one-step/short-step result equals single-GPU global result within stated tolerance, and flag-off graph test still passes. **NO-GO** if physics adapters introduce cross-shard communication or if boundary handling is unclear for real cases. | End-to-end wiring on fake devices; column-local physics per-shard execution; no default regression. | 2-4 sprints, 1500-3500 LOC, very high risk | Dry physics-off step first, then physics-on column-only checks. |
| **S5. Hardening, projections, and real-DGX checklist** | Produce final proof objects, documented simulation limits, transfer/collective audit, rough scaling projection, and a real-DGX smoke checklist for single-node 8xH200 and multi-node. | `proofs/multigpu_dgx/*`, docs/checklist, scripts | **GO** if proof object lists commands, device counts, op counts, max diffs, and unsupported evidence. **NO-GO** if any performance number is reported as measured without hardware. | Reproducible confidence statement for later DGX run; clear boundary between simulated correctness and real interconnect performance. | 1 sprint, 300-700 LOC/docs, medium risk | Checklist can be reviewed before hardware exists. |

## Acceptance Gates

1. `git log -1 --oneline` at start equals `d054954`.
2. Plan is committed before implementation; `/tmp/v0110_dgx.plan` records commit hash and step count.
3. Flag-off default path:
   - default public entry remains available and default;
   - `enabled=False` path calls the same compiled function or lowers to identical JAXPR/HLO op count;
   - HLO contains no `collective-permute`, `all-gather`, `all-reduce`, `all-to-all`, `sharding`, or `partition-id` tokens in the flag-off proof.
4. CPU fake-device simulation:
   - run with `PYTHONPATH=src XLA_FLAGS=--xla_force_host_platform_device_count=N taskset -c 0-27 ...`;
   - cover at least N=2 and N=4; N=8 if local CPU memory permits.
5. Halo correctness:
   - width 1-4;
   - periodic east/west and north/south where implemented;
   - staggered x-face merge/ownership proof.
6. Numerical correctness:
   - sharded operator output equals global operator output bitwise where no collective associativity changes occur;
   - otherwise predeclared tolerance is fp64 round-off only and recorded before the run.
7. GPU commands:
   - any real-GPU command must be wrapped in `/tmp/wrf_gpu_run.sh <cmd>`;
   - CPU fake-device tests must not use the wrapper.

## Real-DGX Smoke Checklist

Run only when hardware exists:

1. Confirm environment:
   - `nvidia-smi -L` shows 8 H200-class GPUs;
   - JAX sees 8 GPU devices;
   - `jax_enable_x64=True`;
   - NCCL/NVLink topology visible (`nvidia-smi topo -m`).
2. Single-GPU default:
   - run the flag-off HLO regression on one GPU;
   - verify op counts and no collectives match the committed proof.
3. Single-node 8-GPU fake-to-real smoke:
   - run halo unit tests on real GPU devices with the GPU lock wrapper;
   - run sharded operator parity on a small deterministic grid;
   - run a dry physics-off short forecast;
   - run column-physics-on smoke if memory allows.
4. Transfer/collective audit:
   - capture Nsight Systems trace;
   - verify no host/device transfers inside timestep loops;
   - verify collectives correspond only to documented halo exchanges.
5. Scaling measurement:
   - run 1, 2, 4, 8 GPU weak and strong scaling fixtures;
   - report as real hardware measurement only after profiler artifacts are saved.
6. Multi-node extension:
   - initialize `jax.distributed` with stable coordinator/process ids;
   - repeat halo and operator parity across at least two nodes;
   - do not claim multi-node scaling until InfiniBand/NCCL traces exist.

## Rollback

Because the mode is optional, rollback is straightforward:

- remove the new sharding module, tests, scripts, and docs;
- revert any shared-file flag/export edits;
- leave `contracts/halo.py` no-op default behavior intact.

If the flag-off graph-invariance gate fails at any point, stop implementation, commit only the failing proof object and diagnosis, and ask the manager for a merge decision.
