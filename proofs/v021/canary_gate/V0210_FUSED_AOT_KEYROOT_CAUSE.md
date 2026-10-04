# v0.21.0 Fused AOT 9-Nest Key Root Cause

Date: 2026-06-26
Branch: `worker/opus/vnext-parallel-compile`
Base commit: `36f11c2e`
Verdict: **KWARM-vs-FIXED root_steps hypothesis rejected.**

## Question

Explain the two observed fused d02 cheap keys without triggering another GPU compile:

- first/captured: `ed303b8cbbf3`
- second/missing: `1db7b2e1d0dc`

## Evidence

No GPU probe or model re-run was used for this diagnosis.

The failed 9-nest gate was already in fixed mode:

- `fused_aot_9nest_capture.log:1`: `mode=fixed fixed_root_steps=1 fixed_repeats=5`
- `fused_aot_9nest_capture.log:15`: first fused miss, `cheap_key=ed303b8cbbf3`
- `fused_aot_9nest_capture.log:16`: first fused capture, `cheap_key=ed303b8cbbf3`
- `fused_aot_9nest_capture.log:17`: second fused miss, `cheap_key=1db7b2e1d0dc`
- There is no intervening `CALL FIXED_FIRST` line, so both fused keys occurred inside the first `root_steps=1` fixed call.

The saved `ed303` metadata shows the fused executable input contract:

- `fused_aot_9nest_ed303_meta_contract.log:295`: `arg_count=6`
- `fused_aot_9nest_ed303_meta_contract.log:306`: `child_start_count=7`
- `fused_aot_9nest_ed303_meta_contract.log:8`: `scalar_leaf_count=40`
- `fused_aot_9nest_ed303_meta_contract.log:211`: `shape_dimension_12_count=0`

The reconstructed top-level fused args are:

1. parent `OperationalCarry`
2. tuple of 7 child carries
3. scalar `parent_start`
4. tuple of 7 scalar `child_starts`
5. parent clock base
6. tuple of 7 child clock bases

There is no `root_steps` argument in the fused d02 AOT key input contract, and no saved input aval shape carries dimension `12`.

The current aval-map code also proves this is not identical-aval key nondeterminism:

- `domain_tree.py:1602`: computes `sig = _aval_signature(args)`
- `domain_tree.py:1603`: checks `cached_calls.get(sig)`
- `domain_tree.py:1607`: returns the cached callable on a signature hit
- `domain_tree.py:1620`: computes `_fused_cascade_cheap_key(...)` only after a signature miss

Because `ed303` had just been compiled and stored under its aval signature, the later computation of `1db7` means the next fused call had a different runtime aval signature. With identical root_steps + identical avals, the code would have returned `cached_call(*args)` before computing a second cheap key.

## Conclusion

The two-key failure is **not** caused by `KWARM=12` warmup shape versus `FIXED_ROOT_STEPS=1` timing shape.

It is also **not** genuine cheap-key nondeterminism for identical avals.

The remaining explanation is a real intra-root fused phase change: during one fixed `d01` root step, the d02 fused cascade advances multiple d02 substeps. After the first captured fused substep, the next d02 fused substep presents a different runtime aval signature, so it correctly maps to a different cheap key (`1db7...`).

## Minimal Fix Direction Before Recompile

Do **not** spend a compile cycle just changing `FIXED_ROOT_STEPS=KWARM`; that cannot make `ed303` and `1db7` collapse to one executable, because both keys already appear inside `fixed_root_steps=1`.

Minimal safe direction:

- Treat fused d02 as a finite multi-phase AOT set, not a guaranteed single executable.
- Keep the `domain_tree.py` aval-signature cached-call map and unit test; that is still the right fix for single-slot cached-call eviction.
- Change the 9-nest gate expectation from "exactly one fused key" to "cold discovers a stable finite set of fused phase keys; warm loads the same set from `source=aot_blob`; zero `fallback:fused-cached-call-error`; no unbounded new keys; digest/timing/memory pass."

Only if the project requires one fused executable should the next implementation normalize the carry avals across d02 substep phases. That is a model/runtime shape-stability change, not a gate `root_steps` configuration fix.

## Commands

- CPU metadata read:
  - `JAX_PLATFORMS=cpu PYTHONPATH=src python ... > proofs/v021/canary_gate/logs/fused_aot_9nest_ed303_meta_contract.log`
- Cheap validation:
  - `python -m py_compile src/gpuwrf/runtime/aot_cheap_key.py src/gpuwrf/runtime/aot_precompile.py proofs/v021/canary_gate/perstep_timing_driver.py`
  - `PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest tests/test_aot_executable.py::test_fused_cascade_cached_calls_are_keyed_by_aval_signature -q`
  - `git diff --check`
