# v0.21.0 Fused AOT 9-Nest Phase Bound

Date: 2026-06-26
Branch: `worker/opus/vnext-parallel-compile`
Base commit: `36f11c2e`
Verdict: **K = 2, fixed small bounded.**

## Question

Before any recompile, bound the number of distinct fused d02 aval phases and explain why the second fused d02 key differs from the first:

- phase A / captured: `ed303b8cbbf3`
- phase B / missing: `1db7b2e1d0dc`

## CPU-Only Evidence

No GPU probe, model run, or compile was used for this diagnosis.

The saved phase-bound proof object is:

- `proofs/v021/canary_gate/logs/fused_aot_9nest_phase_bound.log`

Key facts from that log:

- `bounded_K: 2`
- `boundary_leaf_count: 88`
- `boundary_slots: State slots 38..48 are u/v/theta/qv/ph/mu/w/p/pb/phb/mub_bdy`
- `phase_A_ed303: d02 parent boundary leaves already T=2; d03..d09 child boundary leaves still initial T=1`
- `phase_B_expected: after first fused substep, build_child_boundary_package has rewritten every d03..d09 child boundary leaf to T=2; later substeps stay T=2`
- `why_not_unbounded: build_child_boundary_package reads old_leaf_full[-1] and always returns jnp.stack([old,new], axis=0), exactly two time slices regardless of prior time-axis length`

The phase A boundary time-axis counts reconstructed from the saved `ed303` metadata are:

| Domain | Boundary leaves | Leading time axis |
| --- | ---: | ---: |
| d02 | 11 | 2 |
| d03 | 11 | 1 |
| d04 | 11 | 1 |
| d05 | 11 | 1 |
| d06 | 11 | 1 |
| d07 | 11 | 1 |
| d08 | 11 | 1 |
| d09 | 11 | 1 |

The initial `State` contract creates boundary packages with a leading time axis of 1:

- `src/gpuwrf/contracts/state.py:77`
- `src/gpuwrf/contracts/state.py:122`

The nest boundary builder then replaces a child boundary package with exactly two time slices:

- `src/gpuwrf/nesting/boundary_construction.py:371`
- `src/gpuwrf/nesting/boundary_construction.py:387`
- `src/gpuwrf/nesting/boundary_construction.py:395`
- `src/gpuwrf/nesting/boundary_construction.py:439`

The important shape operation is:

```python
ref = old_leaf_full[-1]
...
return jnp.stack([old, new], axis=0)
```

That consumes the previous last ring and produces a fresh two-time package. It does not append to the previous leading axis.

## Reason

This is a real intra-root multi-phase fused d02 aval, not a gate `root_steps` artifact and not cheap-key nondeterminism.

Phase A is the first fused d02 cascade entry inside a d01 root step. At that point, d01 has already forced d02, so d02's own boundary leaves are two-time packages (`T=2`). The downstream children d03..d09 have not yet been forced inside this fused cascade and still carry their initial `wrfinput` boundary packages (`T=1`).

During the first fused d02 substep, the cascade calls `build_child_boundary_package` for each child. That function rewrites each child boundary leaf as `[old_child_ring, new_parent_target]`, so every d03..d09 boundary leaf becomes `T=2`.

Phase B is the next fused d02 substep. It now sees d02 `T=2` and every child d03..d09 also `T=2`, so its aval signature differs from phase A and maps to the second key `1db7b2e1d0dc`.

The set is bounded because the child boundary update always returns a two-time package. It never carries forward a growing time axis. Later substeps remain in the same phase B shape.

## Bound

`K = 2` distinct fused d02 aval phases:

1. Phase A: d02 boundary leaves are `T=2`, d03..d09 boundary leaves are still `T=1`.
2. Phase B: d02 and d03..d09 boundary leaves are all `T=2`.

There is no evidence of per-substep or per-root-step unbounded shape growth. Scalar substep/start/clock values are traced scalar values and do not alter aval shape.

## Minimal Gate Redesign

Treat fused d02 AOT as a stable finite multi-phase set:

- Cold run may discover exactly the bounded phase set, with assertion `K <= small_N` and observed `K = 2`.
- Warm run must load the same fused phase keys from `source=aot_blob`.
- Gate must fail on any `fallback:fused-cached-call-error`.
- Gate must fail if cold or warm discovers keys outside the bounded set.
- Digest identity, sane `s/step`, and flat memory remain required.

This is shippable if the one re-gate proves both phase keys are warm-loadable and identity-clean. It is not an unbounded AOT thrash problem.

## Commands

- CPU phase-bound analysis:
  - `JAX_PLATFORMS=cpu PYTHONPATH=src python ... > proofs/v021/canary_gate/logs/fused_aot_9nest_phase_bound.log`
- Cheap validation after reverting key-perturbing debug edits:
  - `python -m py_compile src/gpuwrf/runtime/aot_cheap_key.py src/gpuwrf/runtime/aot_precompile.py proofs/v021/canary_gate/perstep_timing_driver.py`
  - `PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest tests/test_aot_executable.py::test_fused_cascade_cached_calls_are_keyed_by_aval_signature -q`
  - `git diff --check`
