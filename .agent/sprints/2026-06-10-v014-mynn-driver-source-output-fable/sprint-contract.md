# Sprint Contract: V0.14 MYNN Driver Source-Output Fix

Date: 2026-06-10 01:17 WEST
Owner: Fable/Mythos in tmux `0:1`
Manager: `worker/gpt/v013-close-manager`

## Objective

Fix, or prove one exact blocker for, the remaining Step-1 source-fidelity
divergence: JAX MYNN driver/kernel source outputs are about an order of
magnitude too weak versus WRF.

Endpoint is the strict Step-1 proof collapsing, or a single precise
WRF-anchored blocker at the MYNN driver/kernel boundary.

## Accepted Prior Proof

Primary proof:

- `proofs/v014/step1_source_fidelity_closure.md`
- verdict:
  `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_MYNN_DRIVER_SOURCE_OUTPUT`

Key facts:

- Strict WRF after-conv vs current JAX dry `T_TENDF`: max_abs
  `2457.578397008898`, RMSE `21.364579991779515`.
- JAX mass-coupled MYNN `RTHBLTEN`: max_abs `260.83156991819124`.
- WRF mass-coupled `RTHBLTEN`: max_abs `2522.90576171875`.
- JAX mass-coupled qv source: max_abs `0.045505018412171354`.
- WRF `QV_TEND`: max_abs `0.4930315017700195`.
- Same-boundary scalar inputs are close, so the order-10 error is not dry-state
  input mismatch: `T` max_abs `5.788684885033035e-05`, `QV`
  `5.969281098756885e-08`, `P` `0.0390625`.
- Radiation-held-rate and WRF `conv_t_tendf_to_moist` are now secondary.

## Required Work

Use the fastest rigorous path. This is a hard debug/fix sprint; do not split it
into tiny micro-prompts.

1. Emit one disposable WRF Step-1 MYNN driver hook around
   `module_bl_mynnedmf_driver`:
   - input columns/fluxes/turbulence state immediately before MYNN;
   - raw `dth1/dqv1` or equivalent source arrays after `mynnedmf_post_run`;
   - module-em mass-scaled `RTHBLTEN/RQVBLTEN`.
2. Compare that exact WRF boundary to JAX `_mynn_column_from_state` and
   `step_mynn_pbl_column` outputs. Include enough metadata to rule in/out:
   timestep semantics, column orientation, PBL top/masks, vertical grid,
   surface flux inputs, exchange coefficients/turbulence state, and any source
   scaling constants.
3. If local, fix the JAX MYNN adapter/kernel source semantics. Preserve the
   high-performance architecture: no production CPU-WRF dependency, no
   host/device transfer inside timestep loops, no broad dycore rewrite.
4. Rerun the strict Step-1 proofs. If not closed, produce one exact MYNN
   boundary blocker and the shortest next proof/fix route.

## File Ownership

Allowed production files:

- `src/gpuwrf/physics/mynn_pbl.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/runtime/operational_mode.py`
- narrow adjacent MYNN/surface-layer adapter files if the proof requires it.

Avoid `src/gpuwrf/dynamics/core/**` unless the proof unexpectedly shows the
contract is wrong. Do not edit memory/FP32 scaffolding, TOST, Switzerland, GPU
run scripts, or release docs.

Allowed proof/test files:

- `proofs/v014/mynn_driver_source_output_fix.py`
- `proofs/v014/mynn_driver_source_output_fix.json`
- `proofs/v014/mynn_driver_source_output_fix.md`
- updates to existing Step-1 proof artifacts if rerun;
- `.agent/reviews/2026-06-10-v014-mynn-driver-source-output-fix.md`;
- focused tests under `tests/`.

Disposable WRF instrumentation must be generated in scratch or proof temp
paths, not committed into pristine WRF sources.

## Acceptance Gates

Required, manager-rerunnable:

```bash
python -m py_compile proofs/v014/mynn_driver_source_output_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/mynn_driver_source_output_fix.py
python -m json.tool proofs/v014/mynn_driver_source_output_fix.json >/tmp/mynn_driver_source_output_fix.validated.json
python -m py_compile proofs/v014/step1_source_fidelity_closure.py proofs/v014/step1_dry_source_leaf_fix.py proofs/v014/step1_part2_source_leaves_split.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_source_fidelity_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_dry_source_leaf_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_part2_source_leaves_split.py
python -m json.tool proofs/v014/step1_source_fidelity_closure.json >/tmp/step1_source_fidelity_closure.validated.json
python -m json.tool proofs/v014/step1_dry_source_leaf_fix.json >/tmp/step1_dry_source_leaf_fix.validated.json
python -m json.tool proofs/v014/step1_part2_source_leaves_split.json >/tmp/step1_part2_source_leaves_split.validated.json
git diff --check
```

If production code changes, add/update focused CPU tests and run them.

Pass target:

- strict Step-1 after-conv vs JAX dry `T_TENDF` nested-interior max_abs
  `<= 1.0e-3`, RMSE `<= 1.0e-5`, or one narrower WRF-anchored MYNN blocker.

## Completion Marker

Print exactly:

`FABLE MYNN_DRIVER_SOURCE_OUTPUT DONE - see proofs/v014/mynn_driver_source_output_fix.md`
