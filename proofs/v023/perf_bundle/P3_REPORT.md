# P3 Report - 2-Domain Root Fusion CPU Proof

## Change

`src/gpuwrf/runtime/domain_tree.py` now allows the fused cascade gate for the exact flat 2-domain one-way root case, `d01 -> d02`. The gate remains fail-closed for deeper roots, leaves, missing edge weights, non-positive ratios, child subtrees, and feedback-enabled trees.

Targeted tests were added in `tests/test_v0110_domain_tree.py` for:

- accepting the flat 2-domain root fusion gate;
- rejecting feedback on that root path;
- preserving scheduler events/step counters for a fused root cascade;
- default fused-factory behavior for the 2-domain root case.

## CPU Validation

Command:

```bash
taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false GPUWRF_NESTED_AOT=0 PYTHONPATH=src python proofs/v023/perf_bundle/p3_root_fusion_cpu_proof.py
```

Result:

- JAX backend: `cpu`
- JAX devices: `cpu:0`
- CPU affinity: cores `4-31`
- geometry: `d01 -> d02`, ratio `3`, `root_steps=1`
- eager/fused events identical:
  - `("advance", "d01", 1, 1, 1)`
  - `("force", "d01", "d02", 1)`
  - `("advance", "d02", 1, 3, 3)`
- `own_steps` identical: `{"d01": 1, "d02": 3}`
- byte identity: all carry leaves matched shape, dtype, and bytes
  - `d01`: `71` leaves, sha256 `c4c09186f8e24f89e9503edf3f78ba9fc29b2f7ab706409428d1fda53bf48146`
  - `d02`: `71` leaves, sha256 `0eb789693024a0ead1bec2317262d64ef56db240caee7f5c3408545a89b78ca0`
- default-on decision: fused 2-domain output is byte-identical to unfused, so no opt-in gate is required for the flat one-way 2-domain root case.

Proof object: `proofs/v023/perf_bundle/P3_CPU_PROOF.json`

## HLO / Program Proof

The unfused CPU path lowers as three separate top-level XLA programs for one root step:

- `p3_unfused_parent_advance`
- `p3_unfused_boundary_force`
- `p3_unfused_child_advance`

The fused CPU path lowers as one top-level XLA program:

- `p3_fused_root_cascade`

Counts from `P3_CPU_PROOF.json`:

- top-level programs: unfused `3`, fused `1`, delta `-2`
- HLO `ENTRY` count: unfused sum `3`, fused `1`
- raw HLO launch-marker estimate: unfused sum `174`, fused `169`

HLO artifacts:

- `proofs/v023/perf_bundle/hlo/p3_unfused_parent_advance.txt`
- `proofs/v023/perf_bundle/hlo/p3_unfused_boundary_force.txt`
- `proofs/v023/perf_bundle/hlo/p3_unfused_child_advance.txt`
- `proofs/v023/perf_bundle/hlo/p3_fused_root_cascade.txt`
- full HLO copies under `proofs/v023/perf_bundle/hlo/full/`

## Tests

Command:

```bash
taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_v0110_domain_tree.py -k 'fusable_parent or fused_root or fused_factory'
```

Result: `9 passed, 19 deselected`.

## GPU Perf Deferred

No GPU command was run for P3. The expected GPU win is fewer top-level dispatches and less host orchestration per 2-domain root step: parent advance + boundary force + child advance become one root-cascade executable. The required validation-phase measurement is a warm GPU launch-count and wall-time comparison; this report makes no GPU wall-time claim.

## Remaining Limits

- Proof is CPU-only by contract.
- Fusion is enabled only for the exact flat one-way 2-domain root tree.
- Feedback and deeper root trees remain on the eager path.
