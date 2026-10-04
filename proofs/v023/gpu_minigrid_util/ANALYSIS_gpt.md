# GPT analysis: GPU utilization options for Tenerife mini-grid

## Scope

This is analysis only. I did not run any GPU forecast, did not take `/tmp/wrf_gpu2_gpu.lock`, and did not edit source code. I read the v023 brief, project rules, local skills, the current nested runtime/orchestration code, AOT key code paths, and the cited v017/v021/v022 performance evidence.

The objective is to find the most practical way to improve throughput for many independent Tenerife-only 2-nest runs, with minimal kernel invasion and with exactness/validation risks stated honestly.

## Evidence anchors

- Current runtime state is already after the big host fixes: async nested output is default, `cuda_async` allocator is set, root-step sync is the default instead of per-advance blocking, AOT cheap-key exists, and fused nested cascade is default where the topology qualifies.
- The current `State` and `OperationalCarry` are JAX pytrees of normal rank-2/rank-3 array leaves. They are stackable, but much of the dycore/nesting code assumes unbatched leaf rank internally.
- `build_child_boundary_package` and `interp._gather` assume field ranks of 2 or 3. A direct leading batch axis inside those functions would break or require broad rank-aware edits. An outer `vmap` around the existing unbatched function is the low-invasion path.
- `OperationalNamelist` has static/grid config in aux data and device leaves for tendencies/metrics/static physics fields. Homogeneous cases with identical geometry can share most static config and differ in carry/state/boundary data plus clock-base.
- AOT cheap keys include carry aval shapes. Any leading batch size `B` creates a distinct compiled executable/cache entry. Dates should not cause recompiles if the existing date-blind clock-base discipline is preserved.
- Existing fused cascade logic only fuses a non-root parent with leaf children. A pure 2-nest `d01 -> d02` Tenerife case will not automatically get that exact fused cascade unless the root case is handled specially.
- Prior v0.19 leaf-vmap evidence is negative, but it batched heterogeneous tiny leaf nests with different shapes. That is not the same problem as homogeneous independent cases.
- v021/v022 evidence says small grids are latency/occupancy limited, larger grids improve GPU economics, and full large-grid claims should be bounded by the measured current roofline, not by theoretical 5090 FLOPs.

## Ranked options

| Rank | Option | Kernel invasion | Expected throughput effect | VRAM pressure | Exactness risk | Recommendation |
|---:|---|---|---|---|---|---|
| 1 | Homogeneous independent-case batching with outer `vmap` over whole 2-nest cases | Low to medium. Runtime/orchestration/AOT/output changes, no dycore stencil rewrite | Best. Manufactures larger effective grid work and amortizes launches/scheduler overhead across cases | Linear in dynamic state/work arrays; shared code/static data in one process | Low to medium until bit gate proves it | Primary path |
| 2 | Multiple standalone JAX processes with CUDA MPS or concurrent streams | Very low source invasion | Useful fallback, but weaker. No kernel coalescing, duplicate contexts/executables, allocator contention | High because each process duplicates more memory | Low, because each case can use the existing standalone executable | Fallback if strict byte identity blocks `vmap` |
| 3 | Host pipeline and output pipeline tightening | Low | Small by itself. Current code already has async output/root-step sync; helps keep a batch queue full | Low | Low | Pair with rank 1, not a main lever |
| 4 | Spatial stitching: pack many independent grids into one larger grid with gaps | Medium to high | Theoretically creates scale, practically wastes cells and risks cross-case contamination | Wastes area in guard regions | High | Do not pursue for production exactness |
| 5 | Pad/enlarge the physical d01/d02 domains and crop | Low to medium config work | Can raise utilization but pays for useless physics cells | Potentially large | Medium to high unless boundary/crop proof is strong | Only as a diagnostic or if science wants the larger domain anyway |
| 6 | Persistent kernels, Pallas/Triton rewrites, dycore kernel surgery | High | Could help in principle, but this violates the "minimal kernel invasion" direction | Unknown | High | Out of scope for this sprint |

## Winner

The best practical candidate is homogeneous independent-case batching: run `B` independent Tenerife cases in one compiled JAX program, with the same geometry and cadence, using an outer `vmap` so the existing per-case dycore and boundary code still sees normal unbatched ranks.

The important distinction is where the batch axis lives. Do not teach the whole dycore that every state leaf has rank+1 as a first step. Instead, build batched wrappers that call the existing single-case functions under `jax.vmap`. Inside the mapped function, `theta`, `u`, `v`, boundary packages, and interpolated fields keep their normal shapes. XLA receives a batched program, but Python code and rank assertions remain close to today's path.

This is the only option that directly attacks the mini-grid utilization problem while preserving the current validated numerical code. It creates scale without inventing artificial spatial coupling and without requiring low-level kernel rewrites.

## Prior-art reconciliation

The v0.19 `vmap` dead end does not directly rule this out.

That prior attempt batched heterogeneous leaves of one nested run. The leaves had different domain shapes, were very small, and the implementation had to work around ragged geometry and leaf-specific cadence. The ledger records it as about 2.5x slower and notes that the seven leaves were not same-shape.

The Tenerife case is different:

- every batch member is the same domain tree shape, same grid dimensions, same nesting ratio, same output cadence, and same static geometry;
- cases are independent, so there is no cross-batch boundary feedback;
- an outer `vmap` avoids padding heterogeneous leaves into one shape;
- the batch can target the full 2-nest cascade, not only tiny same-level leaf kernels.

The prior failure still warns about XLA codegen. A batched HLO can fuse differently, use more registers, or compile to worse kernels. Therefore this should be treated as the top candidate, not as proven. The first GPU gate after the lock is free must compare B=2 against two standalone serial runs for bit identity, wall time, VRAM, and AOT warm-hit behavior.

## Implementation sketch

1. Add a batch entry point in `nested_pipeline.py`, not in the dycore. Load `B` cases with `_load_domains`, assert identical domain names, dimensions, nesting ratios, map factors, physics suite, time step, history cadence, and static treedefs.

2. Stack per-domain `OperationalCarry` objects with `jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), ...)`. Keep shared `OperationalNamelist`/geometry values unbatched when they are truly identical. Per-case dates should enter through the existing clock-base path; per-case ICs and lateral-boundary state live in the batched carry/state.

3. In `domain_tree.py`, create batched advance wrappers around the existing unbatched functions. Conceptually:

   ```python
   batched_advance = jax.vmap(
       lambda carry, start, clock_base: _advance_chunk(
           carry,
           shared_namelist,
           start,
           clock_base,
           n_steps=n_steps,
           cadence=cadence,
       ),
       in_axes=(0, 0, 0),
   )
   ```

   If all cases use the same relative step counters, `start` can be scalar/shared and `in_axes=(0, None, 0)`. If absolute forcing phase can differ by date, keep the relevant clock/phase inputs batched.

4. Batch the live boundary forcing by vmapping the existing unbatched package builder:

   ```python
   batched_bdy = jax.vmap(
       lambda child_state, parent_state: build_child_boundary_package(
           child_state,
           parent_state,
           edge.weights,
           bdy_width,
       ),
       in_axes=(0, 0),
   )
   ```

   This is key because `build_child_boundary_package` and `interp._gather` assume rank-2/rank-3 fields internally.

5. For the 2-nest case, add a specific batched root cascade path: advance d01 one parent step, build d01-to-d02 boundary packages, advance d02 for the child ratio, all inside the batched wrapper. The current generic fused-cascade path skips root parents, so relying only on existing `_fusable_parent` will leave throughput on the table for a pure `d01 -> d02` mini-grid.

6. Output should split the batched carry into per-case views at history cadence and feed the existing writer per output directory. That preserves the current NetCDF surface and keeps validation simple. Later, training-oriented outputs could write a batched archive directly, but that should not be part of the first correctness sprint.

7. AOT should key explicitly by `(geometry, suite, B, precision mode, cadence/topology)`. Since cheap keys already hash aval shapes, B will naturally produce distinct blobs. Production should choose one or two fixed B values, for example B=4 and a B=2 fallback, instead of compiling arbitrary batch sizes.

## Quantitative estimate

This is an estimate only; no v023 GPU measurement was run.

Memory anchors from current proof history:

- 9 km d01 standalone around 4.7 GiB.
- Full 1 km single-domain/current all-island type runs are around 18 GiB peak in the existing resource notes.
- Retained 72 h Canary L2 d02 nested profile reached about 29.8 GiB.
- v017 active all-7 traces were commonly in the 15.8-22.3 GiB range.

For a small Tenerife 2-nest with d02 only a few hundred cells per side, the plausible B depends on actual d02 dimensions:

| Assumption | Effective columns per case | Estimated per-case dynamic peak | Plausible B on 32 GiB 5090 | Notes |
|---|---:|---:|---:|---|
| Small mini d02, about 150-180 squared plus smaller d01 | about 25k-40k | about 3-5 GiB | 4-6 | B=4 likely reaches useful scale |
| Medium mini d02, about 220-260 squared plus smaller d01 | about 55k-75k | about 6-9 GiB | 2-3 | B=3 may already resemble a large grid |
| Full all-island/current 1 km style domain | 150k+ | about 18 GiB | 1 | batching probably does not fit |

Assume a practical usable budget of roughly 24-26 GiB after desktop/driver/allocator/fragmentation margin. A single-process batched executable should share some static/executable memory, so the incremental cost per additional case should be lower than launching fully separate processes, but the dynamic state/work arrays scale close to linearly with B.

Throughput should be reasoned in columns and kernels, not theoretical TFLOPs. The measured project roofline says small 129 squared grids can be slower than CPU, while large 1 km grids are more plausibly in the 1.6x-2.7x GPU-vs-CPU band, centered near about 2x for current code. Batching does not exceed that roofline; it helps a mini-grid behave more like the larger-grid regime.

A useful sizing rule is:

```text
effective_columns = B * (d01_columns + d02_columns)
target useful regime = roughly 120k-250k effective columns
```

Examples:

- If one Tenerife 2-nest case is about 30k effective columns, B=4 gives about 120k and B=6 gives about 180k.
- If one case is about 60k effective columns, B=2 gives about 120k and B=3 gives about 180k.
- If one case is already above 150k columns, batching is probably memory-limited and unnecessary.

Expected same-GPU throughput gain versus serial standalone execution:

- B=2: about 1.5x-1.9x cases/hour if the single case is genuinely underfilled and memory fits.
- B=4: about 2.8x-3.5x cases/hour in the small-mini case.
- B=6-8: about 4x-6x cases/hour only if VRAM and compile size remain healthy.

These are intentionally below linear B scaling. Device work still grows, register pressure may worsen, output remains per-case, and XLA may not collapse launches as much as hoped.

## Spatial stitching estimate

Spatial stitching is unattractive because the guard region needed for bit-identical independent forecasts is not a fixed halo. Numerical influence propagates every timestep through dynamics, diffusion, advection, pressure solve approximations, boundary relaxation, and physics tendencies. A fixed seam buffer can protect only a short interval unless the forecast is periodically restarted from clean per-tile boundaries.

Even with an unrealistically fixed guard width `w`, the area waste is material:

```text
area_overhead = ((n + 2w)^2 / n^2) - 1
```

For `n=160, w=10`, overhead is about 26.6 percent per tile. For `n=128, w=10`, it is about 32.1 percent. A guard width large enough to prevent multi-hour contamination would be far larger than 10 cells and would destroy the throughput argument. This also complicates per-case map factors, land masks, static fields, lateral boundaries, and nest forcing. It is the wrong tradeoff for a project that requires real WRF fixtures and near-identical outputs.

## Concurrent processes / MPS

Concurrent standalone processes are the best fallback if strict byte identity blocks `vmap`. They preserve the existing executable per case and require little or no source work. The downside is that they duplicate JAX/XLA runtime state, compiled executables, allocator pools, NetCDF output state, and much of the static memory. They also do not coalesce kernels; they only let CUDA interleave work from multiple contexts.

A reasonable expectation is B=2, maybe B=3, if the mini-grid is small enough. This could improve cases/hour by roughly 1.2x-2.5x, but it is less predictable than a single batched process and more likely to hit VRAM fragmentation. It is still worth keeping as an operational escape hatch because its exactness story is strongest.

## Failure modes to test explicitly

- Batching changes HLO fusion enough to lose bit identity versus standalone. There should be no reductions over the batch axis, but XLA may still change per-element fusion or contraction boundaries.
- The batched compiled module grows too large or compiles too slowly. B should be fixed to a small set of production values.
- VRAM limits B below the useful regime. If B=2 is the maximum and one case is very small, MPS may be comparable.
- Output materialization serializes the run. Existing async writer helps, but per-case NetCDF writing can become the dominant host-side cost for short mini-grid cases.
- Date-specific physics/static fields create treedef mismatches across cases. The loader must assert homogeneous structure before stacking.
- Direct rank+1 plumbing accidentally enters `build_child_boundary_package`, interpolation, or physics routines that assume rank-2/rank-3 fields. The implementation should vmap around those routines instead.
- The current generic fused-cascade topology does not fuse root `d01 -> d02`; without a special batched root cascade, a two-domain mini-grid may stay more host-orchestrated than intended.
- AOT cache churn appears if operators choose arbitrary B values or mix subtly different namelist/static configurations.

## Acceptance gates for a future implementation

When the GPU lane is available, the first implementation should pass these gates before broader rollout:

1. B=2, 1 forecast-hour, same two init-days also run as two standalone serial cases. Compare wrfout hashes and full-field deltas per case.
2. Confirm one warm AOT hit for the same `(geometry, B)` on a second run with different dates.
3. Record peak VRAM and compile RSS for B=1, B=2, B=4 if B=4 fits.
4. Capture an Nsight or equivalent utilization summary showing fewer/larger effective kernels or materially improved busy fraction.
5. Measure cases/hour including output, not only device step timing.

## Decision

Proceed with homogeneous independent-case `vmap` batching as the next design/implementation sprint if the project wants better mini-grid throughput. Keep MPS/concurrent standalone execution as the conservative fallback if the bit-identity gate fails. Do not spend sprint time on spatial stitching or persistent-kernel rewrites for this objective.

## Handoff

- Objective: analyze GPU utilization strategies for many independent Tenerife 2-nest mini-grid forecasts without GPU execution or source edits.
- Files changed: `proofs/v023/gpu_minigrid_util/ANALYSIS_gpt.md` only.
- Commands run: read project constitution/agents, read v023 brief, read local profiling/design/review/reporting skills, inspect `domain_tree.py`, `nested_pipeline.py`, `state.py`, `operational_state.py`, `operational_mode.py`, boundary/interp code, AOT cheap-key/precompile paths, and cited v017/v021/v022 proof artifacts.
- Proof objects produced: this analysis markdown.
- Unresolved risks: exact B depends on actual Tenerife mini-grid dimensions and VRAM profile; `vmap` bit identity and XLA codegen must be measured after the GPU lane is free.
- Next decision needed: approve or reject a follow-up implementation sprint for batched homogeneous 2-nest orchestration, with B=2/B=4 correctness and utilization gates.
