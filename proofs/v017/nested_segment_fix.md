# v0.17 Nested `_advance_chunk` Compile Bound Fix

Status: CODE-ONLY, GPU validation pending by manager.
Branch: `worker/gpt/v017-nested-segment`.

## Root Cause

The nested runtime already segments the outer forecast by output hour
(`src/gpuwrf/integration/nested_pipeline.py:730-777`), but each segment delegates
per-domain advancement through `run_operational_domain_tree`.

Inside the recursive runtime, a parent domain with children advances one parent
step, forces child boundaries, then recurses children (`src/gpuwrf/runtime/domain_tree.py:296-314`).
Leaf domains advance for the parent-grid-ratio subcycle
(`src/gpuwrf/runtime/domain_tree.py:286-293`). The operational adapter then calls
`_advance_chunk` with that interval length and the domain radiation cadence
(`src/gpuwrf/runtime/domain_tree.py:327-339`).

Before this patch, `_advance_chunk` was declared as:

`@partial(jax.jit, static_argnames=("n_steps", "cadence"))`

at `src/gpuwrf/runtime/operational_mode.py:4198`. That made every distinct
`(carry/namelist shape+static aux, n_steps, cadence)` a separate XLA cache key.
The all-7 run showed repeated slow alarms for `jit__advance_chunk` and no wrfout
before kill:

`/mnt/data/wrf_gpu_validation/v017_canary_all7_full_par_20260615T100321Z/all7_full.log`

The static `n_steps` part is the unbounded failure mode: any output/coupling/final
tail interval length variation forces another full compile instead of reusing the
domain executable.

## Fix

`_advance_chunk` is now a single jitted dynamic loop:

- `src/gpuwrf/runtime/operational_mode.py:4198-4237`
- `start_step`, `n_steps`, and the explicit `cadence` argument are traced `int32`
  scalars.
- The old static-length `jnp.arange(...); lax.scan(...)` is replaced with
  `jax.lax.fori_loop(0, n_steps, body, carry)`.
- The production nested adapter passes `n_steps` and `cadence` as `int32` JAX
  scalars (`src/gpuwrf/runtime/domain_tree.py:333-339`).

No XLA flags were added. No cache-key-affecting environment policy was added.

I did not batch parent domains into longer fixed chunks: that would skip the WRF
live-nest order of "parent step -> build child boundary -> child subcycle" and
would not be bit-identical. This patch keeps those exact host-recursion boundaries
and removes only the interval-length static key.

## Bit-Identity Argument

The per-step body is unchanged:

- global step index remains `start_step + offset`;
- radiation still fires exactly when `step_index % cadence == 0`;
- `_physics_boundary_step(...)` is called once per model step in the same order;
- live parent-to-child boundary construction still occurs at the same parent-step
  points in `run_domain_tree_callbacks`;
- Noah-MP held radiation still uses the namelist's actual `radiation_cadence_steps`
  for the `radt/2` held-time offset.

The only lowering change is `scan` over a static index vector -> dynamic `fori_loop`
over the same ordered offsets. CPU regression
`tests/test_v013_compile_perf2.py::test_dynamic_loop_matches_static_scan_reference_bitwise`
pins this with a radiation-cadence branch: dynamic loop output equals the former
static scan reference bit-for-bit for varying starts, lengths, and cadences.

I did not run GPU validation in this worker lane.

## Expected Compile Count

For the Canary all-7 case:

- domains: d01..d09;
- distinct `(e_we,e_sn)` shapes from namelist: 8, with d06 and d07 both 40x40;
- static namelist/grid/radiation/Noah-MP metadata may keep d06 and d07 as separate
  executable keys despite identical dimensions, so the honest expectation is
  approximately 8-9 `jit__advance_chunk` cold compiles, not growth with every
  output/coupling/final-tail interval.

The important release gate is boundedness: changing `n_steps` no longer adds a
new `_advance_chunk` executable. After those bounded cold compiles, the run should
reach the first forecast hour and write wrfout files.

## Manager GPU Validation

Script:

`proofs/v017/run_canary_all7_nested_segment_validation.sh`

It runs a 1-hour all-7 nested forecast:

- `--max-dom 9 --hours 1`;
- CPU pinned to cores `0-3`;
- GPU serialized through `scripts/run_gpu_lowprio.sh`;
- writes `summary.txt` with `jit__advance_chunk` compile-alarm count, slow-operation
  completion count, and wrfout count.

Expected manager result: rc 0, bounded `jit__advance_chunk` compile alarms
(about 8-9 on a cold cache), and `9` wrfout files for the 1-hour validation.

## CPU-Only Checks Run

All commands were run with `CUDA_VISIBLE_DEVICES=''`, `JAX_PLATFORMS=cpu`, and
`taskset -c 0-3`.

- `python -m py_compile src/gpuwrf/runtime/operational_mode.py src/gpuwrf/runtime/domain_tree.py tests/test_v013_compile_perf2.py`
- `pytest -q tests/test_v0110_domain_tree.py`
- `pytest -q tests/test_v013_compile_perf2.py -k 'advance_chunk_pattern or python_int_start_step or int32_and_pyint or dynamic_n_steps or dynamic_loop'`
- `pytest -q tests/test_v013_compile_perf2.py`
- `git diff --check`

Results: all passed.
