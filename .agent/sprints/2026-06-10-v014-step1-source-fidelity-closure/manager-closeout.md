# Manager Closeout

Date: 2026-06-10 01:07 WEST
Manager: Codex
Worker: GPT-5.5 xhigh in tmux `0:3`

## Outcome

Closed as a successful narrowing sprint, not as a parity fix.

Verdict:

`STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_MYNN_DRIVER_SOURCE_OUTPUT`

The sprint implemented the missing qv source leaf and WRF
`conv_t_tendf_to_moist` handling for `rad_rk_tendf=1`, then proved those are
secondary. The remaining blocker is now a single hard MYNN driver/kernel
source-output mismatch.

## Proof Objects

- `proofs/v014/step1_source_fidelity_closure.py`
- `proofs/v014/step1_source_fidelity_closure.json`
- `proofs/v014/step1_source_fidelity_closure.md`
- `.agent/reviews/2026-06-10-v014-step1-source-fidelity-closure.md`
- refreshed `proofs/v014/step1_dry_source_leaf_fix.*`
- refreshed `proofs/v014/step1_part2_source_leaves_split.json`

## Commands Run

Manager reran:

- `python -m py_compile proofs/v014/step1_source_fidelity_closure.py proofs/v014/step1_dry_source_leaf_fix.py proofs/v014/step1_part2_source_leaves_split.py tests/test_v014_dry_source_leaf_wiring.py src/gpuwrf/coupling/physics_couplers.py src/gpuwrf/runtime/operational_mode.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_dry_source_leaf_wiring.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_source_fidelity_closure.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_dry_source_leaf_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_part2_source_leaves_split.py`
- `python -m json.tool` on all three JSON proof objects
- `git diff --check`

## Merge Decision:

Merge the source changes and proof artifacts as the new v0.14 frontier. Do not
start TOST, Switzerland, broad FP32, or broad memory validation. The gate is
still blocked on MYNN source output.

## Unresolved Risks

- JAX MYNN source outputs are about an order of magnitude too weak compared to
  WRF at the same Step-1 boundary.
- The next fix likely requires direct WRF MYNN driver instrumentation and/or
  a JAX MYNN kernel semantics fix.

## Next Sprint

Escalate the remaining hard blocker to Fable/Mythos after `/compact`: emit one
WRF MYNN driver hook at Step 1 around `module_bl_mynnedmf_driver`, including
input columns/fluxes/turbulence state before MYNN, raw `dth1/dqv1` after
`mynnedmf_post_run`, and module-em mass-scaled `RTHBLTEN/RQVBLTEN`; compare
that exact boundary to JAX `_mynn_column_from_state` /
`step_mynn_pbl_column`, then fix the MYNN source mismatch if local.
