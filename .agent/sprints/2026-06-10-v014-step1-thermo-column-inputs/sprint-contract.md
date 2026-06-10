# Sprint Contract: V0.14 Step-1 Thermodynamic Column Inputs

Date: 2026-06-10 03:55 WEST
Owner: GPT-5.5 xhigh worker in tmux
Manager: `worker/gpt/v013-close-manager`
Base commit: `cdfdbbc2 v014 fix tsk znt sfclay sourcing`

## Objective

Close, or reduce to one strictly narrower WRF-anchored blocker, the remaining
Step-1 `sfclay_mynn` input mismatch after `TSK/ZNT/MAVAIL` sourcing was fixed.

This is a whole-debug sprint. The target is the exact surface-driver input
boundary, not a broad forecast comparison. Use the fastest rigorous method:
reuse the existing WRF hook and JAX proof where possible, add a narrower hook or
savepoint only if it removes ambiguity, then implement a local fix only when the
boundary evidence proves it.

## Accepted Prior Proofs

- `proofs/v014/mynn_driver_source_output_fix.{py,json,md}`
- `proofs/v014/step1_sfclay_boundary_fix.{py,json,md}`
- `proofs/v014/step1_tsk_znt_sourcing_fix.{py,json,md}`
- `proofs/v014/step1_tsk_znt_sourcing_fix_wrf_patch.diff`
- `.agent/reviews/2026-06-10-v014-step1-tsk-znt-sourcing.md`

Current facts:

- MYNN cold-start QKE is fixed and the MYNN kernel is exonerated with WRF inputs.
- WRF MYNN surface first-call semantics are fixed.
- `TSK/ZNT/MAVAIL` are fixed/proven at the exact `sfclay_mynn` input hook:
  `TSK` max_abs `0.0 K`, `ZNT` max_abs `1.1920928910669204e-08 m`, `MAVAIL`
  max_abs `1.1920928966180355e-08`.
- Strict Step-1 after-conv residual remains red: max_abs
  `1497.6112467075195`, RMSE `13.252694871222973`.
- Current WRF-anchored input deltas at `sfclay_mynn`:
  - `th_phy(kts)` max_abs `5.490148027499686 K`, RMSE `4.596847297193302`;
  - derived `t_phy(kts)` max_abs `5.521345498302992 K`, RMSE `4.614221839008816`;
  - `p_phy(kts)` max_abs `292.8203125 Pa`, RMSE `45.931279429597595`;
  - `u/v/qv(kts)` are near roundoff.

## Required Work

1. Read the prior TSK/ZNT proof and inspect how it compares WRF
   `sfclay_mynn_in__{th_phy,t_phy,p_phy,dz8w}` against JAX
   `_surface_column_view`.
2. Localize whether the `th_phy/t_phy/p_phy/dz8w` mismatch is caused by:
   - state sourcing before `_surface_column_view`;
   - `_surface_column_view` formulas (`theta`, pressure, temperature, dz);
   - an indexing/staggering/orientation mismatch in the hook/proof;
   - a missing WRF perturbation/base conversion after earlier Step-1 fixes;
   - or a strictly later surface-layer algebra issue.
3. If local and performance-compatible, fix production code. Prefer
   state/coupler/source corrections over proof-only transformations. Do not
   insert clamps or CPU-WRF dependencies.
4. Produce a primary proof object:
   `proofs/v014/step1_thermo_column_inputs.{py,json,md}`.
5. Rerun `proofs/v014/step1_tsk_znt_sourcing_fix.py` and the strict
   source-fidelity/MYNN proof chain enough to prove the new blocker status.
6. If this boundary is not fixable in one sprint, return one exact narrower
   blocker, a ranked hypothesis table, and the fastest next command.

## File Ownership

Allowed production files:

- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/contracts/state.py`
- narrow adjacent state/input-loader files only if the proof shows the source is
  there.

Avoid `src/gpuwrf/dynamics/**`, broad memory/FP32 scaffolding, GPU run scripts,
TOST, Switzerland, release docs, and broad refactors. Do not use Hermes or
Fable/Mythos.

Allowed proof/test files:

- `proofs/v014/step1_thermo_column_inputs.py`
- `proofs/v014/step1_thermo_column_inputs.{json,md}`
- WRF patch archive if a new/changed hook is used:
  `proofs/v014/step1_thermo_column_inputs_wrf_patch.diff`
- focused tests under `tests/` if production code changes;
- updates to current v0.14 Step-1 proof artifacts only when rerun evidence
  requires them;
- `.agent/reviews/2026-06-10-v014-step1-thermo-column-inputs.md`

## Acceptance Gates

Required, manager-rerunnable:

```bash
python -m py_compile proofs/v014/step1_thermo_column_inputs.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_thermo_column_inputs.py
python -m json.tool proofs/v014/step1_thermo_column_inputs.json >/tmp/step1_thermo_column_inputs.validated.json
python -m py_compile proofs/v014/step1_tsk_znt_sourcing_fix.py proofs/v014/step1_source_fidelity_closure.py proofs/v014/mynn_driver_source_output_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_tsk_znt_sourcing_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_source_fidelity_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/mynn_driver_source_output_fix.py
python -m json.tool proofs/v014/step1_tsk_znt_sourcing_fix.json >/tmp/step1_tsk_znt_sourcing_fix.validated.json
python -m json.tool proofs/v014/step1_source_fidelity_closure.json >/tmp/step1_source_fidelity_closure.validated.json
python -m json.tool proofs/v014/mynn_driver_source_output_fix.json >/tmp/mynn_driver_source_output_fix.validated.json
git diff --check
```

If production code changes, add or update focused CPU tests and run them.

Pass target:

- Preferred: strict Step-1 `after_conv_t_tendf_to_moist` vs JAX dry `T_TENDF`
  nested-interior max_abs `<= 1.0e-3`, RMSE `<= 1.0e-5`.
- Acceptable narrowing: exact WRF-anchored blocker strictly later or narrower
  than `th_phy/t_phy/p_phy/dz8w` sourcing, with proof that these inputs are fixed
  or bounded.

## Performance Constraints

- CPU-only unless the manager explicitly grants a short low-VRAM GPU probe.
- No host/device transfer inside timestep loops.
- No new dynamic-shape arrays.
- Any new runtime state must remain static-shape and resident.

## Handoff Requirements

Write:

- `proofs/v014/step1_thermo_column_inputs.md`
- `.agent/reviews/2026-06-10-v014-step1-thermo-column-inputs.md`
- concise worker/test/review closeout drafts if complete.

Completion marker:

`GPT STEP1_THERMO_COLUMN_INPUTS DONE - see proofs/v014/step1_thermo_column_inputs.md`

Notify manager pane `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_THERMO_COLUMN_INPUTS DONE - see proofs/v014/step1_thermo_column_inputs.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
