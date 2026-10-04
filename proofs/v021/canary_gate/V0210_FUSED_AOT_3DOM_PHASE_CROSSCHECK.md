# v0.21.0 Fused AOT 3-Domain Phase Cross-Check

Date: 2026-06-26
Branch: `worker/opus/vnext-parallel-compile`
Verdict: **Existing 3-domain PASS does not satisfy the redesigned bounded-phase gate.**

## Question

Check whether the already-passed 3-domain fused+AOT gate proves the same two-phase warm-reuse mechanism now required for the 9-nest gate.

## Result

It does not fully prove the redesigned invariant.

The old 3-domain logs do show the same bounded two-key cold discovery:

- cold resident fused keys: `4c14e73fbb57`, `6db965ea2ff5`
- `BIT_ID` digest matches in cold and warm
- warm `REF_COMPARE ref_equal=True`

But they also show the old single-slot cached-call failure mode and do not show both phase keys loading in the fresh warm process:

- cold log contains `fallback:fused-cached-call-error`
- warm loaded fused keys: only `4c14e73fbb57`
- warm did not log `6db965ea2ff5 source=aot_blob`
- the old 3-domain driver used difference mode, not the fixed-shape 9-nest gate mode

Stored validator output:

- `proofs/v021/canary_gate/logs/fused_aot_3dom_redesigned_validation.json`
- verdict: `FAIL`

## Interpretation

The old 3-domain gate is useful supporting evidence that the fused cascade can have a bounded two-key phase set and still produce an exact digest match. It is **not** sufficient release evidence for the redesigned bounded-phase gate because it predates the aval-signature cached-call map and contains the very cached-call error the redesigned gate must reject.

The binding proof still needs the 9-nest re-gate under current code.

## Re-Gate Plan

Use a fresh clean cache for the binding 9-nest gate.

Reason: the existing `ed303` blob is plausibly valid, but it lives in a stopped diagnostic cache that only contains phase A. The release gate should avoid stale-pollution ambiguity and prove clean cold-to-warm behavior from a new cache.

The updated 9-nest runner now:

- creates a timestamped cache by default: `/tmp/gpuwrf_v0210_fused_aot_9nest_gate_<UTC>`
- logs `MARKER:GATE_CACHE` in both cold and warm logs
- expects exactly `K=2` fused d02 phase keys
- asserts the bounded set is `<=4`
- fails on `fallback:fused-cached-call-error`, fused AOT exec errors, fused lower/compile exceptions, or warm `jit_fused_jit` compile markers
- requires warm `source=aot_blob` keys to equal the cold resident fused phase set
- requires fixed-mode `SUMMARY`, finite sane `s_per_fc_hour`, `vram_flat=True`, `rss_flat=True`, `BIT_ID`, and warm `REF_COMPARE ref_equal=True`

Expected operational cost: cold compiles both fused phases, then fresh warm loads both from AOT. This is the defensible release gate, even if it costs roughly two phase compiles.

## Cheap Checks

- `python -m py_compile proofs/v021/canary_gate/validate_fused_aot_9nest_gate.py proofs/v021/canary_gate/perstep_timing_driver.py src/gpuwrf/runtime/domain_tree.py src/gpuwrf/runtime/aot_cheap_key.py src/gpuwrf/runtime/aot_precompile.py`
- `bash -n proofs/v021/canary_gate/run_v0210_fused_aot_gate_9nest.sh`
- `PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest tests/test_aot_executable.py::test_fused_cascade_cached_calls_are_keyed_by_aval_signature -q`
