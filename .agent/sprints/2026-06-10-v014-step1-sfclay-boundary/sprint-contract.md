# Sprint Contract: V0.14 Step-1 Surface-Layer Boundary Closure

Date: 2026-06-10 02:34 WEST
Owner: GPT-5.5 xhigh worker in tmux
Manager: `worker/gpt/v013-close-manager`

## Objective

Close, or reduce to one strictly narrower WRF-anchored blocker, the current
Step-1 surface-layer flux/input boundary divergence that feeds MYNN.

This is a whole-debug sprint. Start from Fable/Mythos' proven conclusion, but do
not blindly assume it is the final bug. If the surface-layer hypothesis is
wrong, prove that and return the next highest-probability blocker with evidence.
If it is local and performance-compatible, implement the fix.

## Accepted Prior Proofs

- `proofs/v014/mynn_driver_source_output_fix.{py,json,md}`
- `.agent/reviews/2026-06-10-v014-mynn-driver-source-output-fix.md`
- `proofs/v014/step1_source_fidelity_closure.{py,json,md}`
- `proofs/v014/step1_dry_source_leaf_fix.{py,json,md}`
- `proofs/v014/step1_part2_source_leaves_split.{py,json,md}`

Current facts:

- MYNN source-output order-10 deficit was fixed by WRF-equivalent first-call
  `mym_initialize` level-2 equilibrium QKE initialization.
- With WRF inputs and WRF-init QKE, JAX MYNN reproduces WRF raw `RTHBLTEN`
  (`strong_ratio_median=0.9982`, `corr=1.0000`, `rmse=2.6e-06`) and `RQVBLTEN`
  (`strong_ratio_median=0.9735`, `corr=0.9998`).
- Strict Step-1 after-conv residual improved but remains red: max_abs
  `1497.6112512148795`, RMSE `13.468453371786723`.
- The remaining Fable-attributed blocker is surface-layer boundary mismatch:
  `ustar` bias `-0.077`, max `0.176`; `HFX` RMSE `24.6 W/m^2`; `QFX` bias
  `-2.1e-5`; land `TSK` up to `8.3 K`; `ZNT` up to `0.97 m`.
- Existing evidence says JAX starts `sfclayrev` from `ustar=0`, while WRF has a
  first-call/`flag_iter`/UST first-guess path; identical-input ocean columns can
  still show about 4x ustar deficits.
- WRF's MYNN cold-start init path used an uninitialized local `rmol`; deterministic
  rmol-pinned WRF truth exists and should be the preferred strict target.

## Required Work

Use the fastest rigorous CPU-only path. Prefer focused hooks/savepoints over
slow full-runtime log chasing.

1. Emit or reuse an exact WRF Step-1 surface-driver hook around
   `module_sf_mynn`/`sfclayrev` for `TSK/ZNT/UST/HFX/QFX` in/out, including
   `flag_iter`, initial `UST`, roughness, skin-temperature, lowest-level winds,
   lowest-level thermodynamics, and any fields needed to reproduce WRF's
   first-call behavior. Disposable WRF source edits are allowed only in scratch;
   archive the patch diff under `proofs/v014/`.
2. Compare that boundary to JAX `surface_adapter` / `physics.surface_layer`
   using the current Step-1 live-nest replay state. Do not compare JAX-vs-JAX
   only; at least one decisive WRF-origin boundary is required.
3. If local, port the WRF-compatible surface-layer behavior into the JAX path:
   first-call `flag_iter`/UST first-guess semantics, `TSK`/skin-temperature
   sourcing, `ZNT`/roughness sourcing, or any exact adjacent adapter issue found.
4. Rerun the MYNN source-output decomposition and strict Step-1 proofs. Gate on
   production JAX inputs moving toward WRF-input case-B levels, and ideally the
   strict Step-1 after-conv residual collapsing.
5. If the surface fix does not close the residual, return one exact next blocker
   and rank at least three tested hypotheses. Do not stop at "surface differs";
   identify the next concrete WRF/JAX boundary, source formula, or state leaf.

## File Ownership

Allowed production files:

- `src/gpuwrf/physics/surface_layer.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/runtime/operational_state.py`
- narrow adjacent State/config/input-loader files only if required to source
  `TSK`, `ZNT`, or first-call surface-layer state faithfully.

Avoid `src/gpuwrf/dynamics/**`, memory/FP32 scaffolding, GPU run scripts, TOST,
Switzerland, and release docs. Do not use Fable/Mythos or Hermes.

Allowed proof/test files:

- New primary proof: `proofs/v014/step1_sfclay_boundary_fix.py`
- New proof outputs: `proofs/v014/step1_sfclay_boundary_fix.{json,md}`
- WRF patch archive: `proofs/v014/step1_sfclay_boundary_fix_wrf_patch.diff`
- updates to `proofs/v014/mynn_driver_source_output_fix.{py,json,md}` if reused;
- updates to strict Step-1 proof artifacts if rerun;
- `.agent/reviews/2026-06-10-v014-step1-sfclay-boundary.md`
- focused tests under `tests/`.

## Acceptance Gates

Required, manager-rerunnable:

```bash
python -m py_compile proofs/v014/step1_sfclay_boundary_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_sfclay_boundary_fix.py
python -m json.tool proofs/v014/step1_sfclay_boundary_fix.json >/tmp/step1_sfclay_boundary_fix.validated.json
python -m py_compile proofs/v014/mynn_driver_source_output_fix.py proofs/v014/step1_source_fidelity_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/mynn_driver_source_output_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_source_fidelity_closure.py
python -m json.tool proofs/v014/mynn_driver_source_output_fix.json >/tmp/mynn_driver_source_output_fix.validated.json
python -m json.tool proofs/v014/step1_source_fidelity_closure.json >/tmp/step1_source_fidelity_closure.validated.json
git diff --check
```

If production code changes, add or update focused CPU tests and run them.

Pass target:

- Preferred: strict Step-1 `after_conv_t_tendf_to_moist` vs JAX dry `T_TENDF`
  nested-interior max_abs `<= 1.0e-3`, RMSE `<= 1.0e-5`.
- Acceptable narrowing: one WRF-anchored blocker strictly later or narrower than
  surface-layer boundary, with proof that the surface-layer discrepancy is fixed
  or bounded and a manager-rerunnable proof object.

## Performance Constraints

- Keep the GPU-native design intact: no CPU-WRF dependency, no host/device
  transfer inside timestep loops, no full-domain materialization beyond existing
  JAX arrays unless measured and justified.
- Any first-call state added to the runtime must be resident, static-shape, and
  compatible with JIT/vmap.
- This sprint is CPU-only unless the manager explicitly grants a short GPU probe.

## Proof Object

Primary proof object:

- `proofs/v014/step1_sfclay_boundary_fix.json`

It must include:

- WRF hook provenance and patch hash.
- WRF-vs-JAX surface input/output metrics for `TSK/ZNT/UST/HFX/QFX` and flux
  handles consumed by MYNN.
- MYNN decomposition before/after using production JAX inputs.
- Strict Step-1 result or exact next blocker.
- A compact ranked hypothesis table, max 8 rows.

## Handoff Requirements

Write:

- `proofs/v014/step1_sfclay_boundary_fix.md`
- `.agent/reviews/2026-06-10-v014-step1-sfclay-boundary.md`
- sprint `worker-report.md`, `tester-report.md`, `reviewer-report.md`,
  `manager-closeout.md`, and `memory-patch.md` drafts if you finish fully.

Completion marker:

`GPT STEP1_SFCLAY_BOUNDARY DONE - see proofs/v014/step1_sfclay_boundary_fix.md`

Also send the marker to manager tmux window `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_SFCLAY_BOUNDARY DONE - see proofs/v014/step1_sfclay_boundary_fix.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
