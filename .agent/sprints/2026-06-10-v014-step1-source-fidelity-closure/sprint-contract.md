# Sprint Contract: V0.14 Step-1 Source-Fidelity Closure

Date: 2026-06-10 00:47 WEST
Owner: GPT-5.5 xhigh worker in tmux
Manager: `worker/gpt/v013-close-manager`

## Objective

Close, or reduce to one strictly narrower WRF-anchored blocker, the remaining
Step-1 `T_TENDF` source-fidelity gap.

This is one coherent fix sprint, not a micro-split sprint. The previous sprint
proved that merely exposing a JAX MYNN theta delta as `RTHBLTEN` is insufficient.
The endpoint is still the strict Step-1 proof collapsing, not a partial local
unit test.

## Accepted Prior Proofs

- `proofs/v014/step1_part2_source_leaves_split.md`
- `proofs/v014/step1_dry_source_leaf_fix.md`

Current facts:

- WRF `update_phy_ten` closes exactly as `T_TENDF = pre + active RTH`.
- WRF `conv_t_tendf_to_moist` closes to roundoff and equals the accepted
  `after_first_rk_step_part2` surface.
- Patched JAX dry `T_TENDF` is active but too small: max_abs
  `260.83156991819124`.
- WRF top active source `RTHBLTEN` has max_abs `2522.90576171875`.
- Final WRF after-conv vs patched JAX dry residual remains max_abs
  `2457.575215120763`, RMSE `21.445918959761645`.
- Forcing radiation only moves max_abs to `2454.161554535577`, so held
  `RTHRATEN` is secondary to MYNN source fidelity.
- WRF `conv_t_tendf_to_moist` contributes max_abs `224.50967407226562`, RMSE
  `4.572429855170764`.

## Required Work

Use the fastest rigorous CPU-only path. You may fix all three blockers if the
evidence remains local and performance-compatible:

1. Split MYNN PBL adapter/kernel inputs and outputs against WRF
   `RTHBLTEN/RQVBLTEN` at the accepted Step-1 boundary. Determine why the JAX
   source is about an order of magnitude too weak. Check timestep semantics,
   perturbation/full-theta semantics, moist/dry theta semantics, mass coupling,
   MYNN input state, surface-layer inputs, and any WRF source-scaling constants.
2. Seed or refresh held `RTHRATEN` at the same Step-1 boundary if required, but
   do not over-focus there: the forced-radiation falsifier says it is secondary.
3. Implement WRF `conv_t_tendf_to_moist` / `QV_TEND` handling before feeding
   `DryPhysicsTendencies.t_tendf`, using the formula already proven by WRF
   surfaces.
4. Keep the high-performance GPU architecture intact: no production CPU-WRF
   dependency, no host/device transfer inside timestep loops, no broad dycore
   rewrite, no material full-domain temporaries beyond the existing source
   leaves unless measured/justified.
5. If closure is impossible in this sprint, produce one narrower blocker with
   evidence that all three current blockers were tested/ranked. Do not stop at
   "MYNN differs"; name the next exact WRF/JAX boundary or source formula.

## File Ownership

Allowed production files:

- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/runtime/operational_state.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/physics/mynn_pbl.py`
- narrow adjacent MYNN/surface-layer adapter files only if the proof shows the
  source mismatch is there.

Avoid `src/gpuwrf/dynamics/core/**` unless you prove `rk_addtend_dry` is the
wrong contract. Do not edit memory/FP32 scaffolding, GPU run scripts, TOST,
Switzerland, or release docs.

Allowed proof/test files:

- `proofs/v014/step1_source_fidelity_closure.py`
- `proofs/v014/step1_source_fidelity_closure.json`
- `proofs/v014/step1_source_fidelity_closure.md`
- updates to `proofs/v014/step1_dry_source_leaf_fix.{py,json,md}` if reused;
- updates to `proofs/v014/step1_part2_source_leaves_split.{py,json,md}` if
  reused;
- `.agent/reviews/2026-06-10-v014-step1-source-fidelity-closure.md`
- focused tests under `tests/`.

## Acceptance Gates

Required, manager-rerunnable:

```bash
python -m py_compile proofs/v014/step1_source_fidelity_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_source_fidelity_closure.py
python -m json.tool proofs/v014/step1_source_fidelity_closure.json >/tmp/step1_source_fidelity_closure.validated.json
python -m py_compile proofs/v014/step1_dry_source_leaf_fix.py proofs/v014/step1_part2_source_leaves_split.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_dry_source_leaf_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_part2_source_leaves_split.py
python -m json.tool proofs/v014/step1_dry_source_leaf_fix.json >/tmp/step1_dry_source_leaf_fix.validated.json
python -m json.tool proofs/v014/step1_part2_source_leaves_split.json >/tmp/step1_part2_source_leaves_split.validated.json
git diff --check
```

If production code changes, add or update focused CPU tests and run them.

Pass target:

- Strict Step-1 `after_conv_t_tendf_to_moist` vs JAX dry `T_TENDF` nested
  interior max_abs `<= 1.0e-3`, RMSE `<= 1.0e-5`, or one strictly narrower
  WRF-anchored blocker explaining why that target cannot yet be met.

## Completion Marker

Print exactly:

`GPT SOURCE_FIDELITY_CLOSURE DONE - see proofs/v014/step1_source_fidelity_closure.md`
