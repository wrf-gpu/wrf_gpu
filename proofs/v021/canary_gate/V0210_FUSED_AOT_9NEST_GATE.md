# v0.21.0 Fused AOT 9-Nest Gate

Date: 2026-06-26
Branch: `worker/opus/vnext-parallel-compile`
Base commit: `36f11c2e`
Verdict: **FAIL - 9-nest fused AOT cached-call thrash is not eliminated.**

## Objective

Fix the fused 9-nest + AOT cached-call thrash bug, add a fixed-shape 9-nest gate, rerun the mandatory 9-nest gate, and report whether warm fused execution loads from the AOT blob without recompile/rethrash.

## Result

| Gate check | Result | Evidence |
| --- | --- | --- |
| Cold fused AOT captures first executable | PASS | `fused_aot_9nest_capture.log:16` captured `cheap_key=ed303b8cbbf3` |
| No fused key thrash during cold fixed-shape run | **FAIL** | `fused_aot_9nest_capture.log:17` immediately requested second `cheap_key=1db7b2e1d0dc` |
| Warm `fused/d02 loaded=true source=aot_blob` | NOT RUN | stopped per contract after second fused key appeared |
| No `fallback:fused-cached-call-error` | PASS for attempted cold run | count `0` in cold log |
| Cold/warm bit-identical digest | NOT RUN | no `BIT_ID` / `REF_COMPARE`; warm process not started |
| Sane `s/step` / memory flat | NOT RUN | no `SUMMARY`; stopped before timing summary |

The unit-level cache fix works for alternating runtime aval signatures, but the real 9-nest fused gate still produces multiple fused AOT cheap keys even in the fixed-shape gate mode. I stopped the run before compiling the second huge fused executable, as required by the sprint contract.

## Files Changed

- `src/gpuwrf/runtime/domain_tree.py`
  - Added `_aval_signature(args)` and replaced the single fused cached-call slot with a bounded map keyed by runtime aval signature.
  - AOT-loaded and newly compiled fused callables are stored under their exact runtime aval signature.
  - Cached-call failures now evict only the matching signature entry.
- `tests/test_aot_executable.py`
  - Added `test_fused_cascade_cached_calls_are_keyed_by_aval_signature`, which alternates two child-carry shapes and verifies A/B/A uses two serializations rather than three.
- `proofs/v021/canary_gate/perstep_timing_driver.py`
  - Added `CANARY_SINGLE_SHAPE_TIMING=1` fixed-shape mode with `CANARY_FIXED_ROOT_STEPS` and `CANARY_FIXED_REPEATS`.
  - Fixed mode repeats the same initial carry/root-step call and emits fixed-mode `SUMMARY`, `BIT_ID`, and `REF_COMPARE` markers when it completes.
- `proofs/v021/canary_gate/run_v0210_fused_aot_gate_9nest.sh`
  - Added an in-lock `timeout 4h`.
  - Enabled fixed-shape 9-nest gate defaults: `CANARY_FIXED_ROOT_STEPS=1`, `CANARY_FIXED_REPEATS=5`.
  - Allowed `CANARY_INPUT_DIR` override because the canonical default symlinks are broken on this workstation.

## Commands Run

- `PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest tests/test_aot_executable.py::test_fused_cascade_cached_calls_are_keyed_by_aval_signature -q`
  - PASS: `1 passed in 6.73s`
- `PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest tests/test_aot_executable.py -q`
  - PASS: `28 passed in 4.69s`
- `python -m py_compile src/gpuwrf/runtime/domain_tree.py proofs/v021/canary_gate/perstep_timing_driver.py`
  - PASS
- `bash -n proofs/v021/canary_gate/run_v0210_fused_aot_gate_9nest.sh`
  - PASS
- `git diff --check`
  - PASS
- Gate command:
  - `CANARY_INPUT_DIR=<DATA_ROOT>/wrf_downscale/canary_all7/run_cadvariant ./proofs/v021/canary_gate/run_v0210_fused_aot_gate_9nest.sh`
  - FAIL: stopped after second fused AOT key appeared in cold fixed-shape run.

## Proof Objects Produced

- `proofs/v021/canary_gate/logs/fused_aot_9nest_capture.log`
  - `MARKER:PROCESS_START ... mode=fixed fixed_root_steps=1 fixed_repeats=5`
  - First fused miss: `cheap_key=ed303b8cbbf3`
  - First fused capture: `source=fallback:fused-jit-compiled+aot-captured ... cheap_key=ed303b8cbbf3`
  - Second fused miss: `cheap_key=1db7b2e1d0dc`
  - Counts: fused missing `2`, fused captured `1`, fused cached-call error `0`, summary `0`, bit id `0`.
- `proofs/v021/canary_gate/logs/9nest_gate_after_fix_outer.log`
  - Outer copy of the same failed gate run.
- No `proofs/v021/canary_gate/logs/fused_aot_9nest_warm_load.log` was produced for the final run because the gate was stopped before warm execution.

## Unresolved Risks

- The aval-signature cached-call map fixes the single-slot eviction bug in isolation, but it does not guarantee one fused AOT cheap key for the real 9-nest schedule.
- The remaining key churn appears upstream of cached-call execution: the fused cascade is being asked for a second compiled/AOT artifact with `cheap_key=1db7b2e1d0dc` after successfully capturing `ed303b8cbbf3`.
- The fixed-shape gate was not able to produce warm-load, bit-identity, timing, or memory-flat evidence.

## Next Decision Needed

Do not ship this as the v0.21 fused 9-nest AOT gate. The next debugging step should identify why the real fixed-shape 9-nest call generates a second fused cheap key after the first fused capture. The likely next probe is to log the fused `ckey` inputs/aval signature and own-step/loop metadata for `ed303b8cbbf3` versus `1db7b2e1d0dc`, then either make the AOT cache key include the true intended phase dimension or normalize the fused schedule so the fixed-shape gate has one executable.
