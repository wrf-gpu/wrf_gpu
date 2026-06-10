# Sprint Contract: V0.14 Step-1 TSK/ZNT Surface-Input Sourcing

Date: 2026-06-10 03:00 WEST
Owner: GPT-5.5 xhigh worker in tmux
Manager: `worker/gpt/v013-close-manager`

## Objective

Close, or reduce to one strictly narrower WRF-anchored blocker, the current
Step-1 `TSK/ZNT` surface input sourcing mismatch before `sfclay_mynn`.

This is a whole-debug sprint. The previous sprint fixed WRF MYNN surface
first-call semantics, but strict Step-1 remains red. Do not assume TSK/ZNT is
the final bug without a WRF hook; prove or refute it at the exact surface-driver
boundary.

## Accepted Prior Proofs

- `proofs/v014/step1_sfclay_boundary_fix.{py,json,md}`
- `.agent/reviews/2026-06-10-v014-step1-sfclay-boundary.md`
- `proofs/v014/mynn_driver_source_output_fix.{py,json,md}`
- `proofs/v014/step1_source_fidelity_closure.{py,json,md}`

Current facts:

- MYNN cold-start QKE is fixed.
- WRF MYNN surface first-call semantics are fixed: UST first guess, `MOL=0`,
  land `QSFC=qv/(1+qv)`, Li_etal_2010 z/L seed.
- UST RMSE improved `0.08667703917523994 -> 0.02954126268295198`.
- qv-flux RMSE improved `1.9833425562981398e-05 -> 1.442591864492997e-05`.
- strict after-conv `T_TENDF` remains red: max_abs `1497.6112467075195`,
  RMSE `13.296448784742802`.
- surviving surface-input residuals: TSK max_abs `8.344940187890643 K`,
  ZNT max_abs `0.9737602076530456 m`.

## Required Work

Use the fastest rigorous CPU-only path:

1. Emit a tiny WRF Step-1 surface-driver hook around
   `module_surface_driver/module_sf_mynn` on the current d02 Step-1 case.
   Capture incoming `TSK/ZNT/UST/QSFC/MOL` and outgoing `UST/HFX/QFX/ZNT`.
   Disposable WRF source edits are allowed only in scratch; archive a patch diff
   under `proofs/v014/`.
2. Compare exact WRF arrays against JAX `_surface_column_view` inputs and
   `surface_layer_with_diagnostics(..., first_timestep=True)` outputs.
3. If local, fix TSK/ZNT sourcing in the JAX path. Likely areas include
   `src/gpuwrf/integration/d02_replay.py`, land-state/static lower-boundary
   loaders, roughness derivation, and surface adapter state construction.
4. Rerun `step1_sfclay_boundary_fix.py`, `step1_source_fidelity_closure.py`, and
   `mynn_driver_source_output_fix.py`.
5. If TSK/ZNT is not the blocker, return one exact narrower blocker and a ranked
   hypothesis table. Do not stop at a broad "surface differs".

## File Ownership

Allowed production files:

- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/io/land_state.py`
- `src/gpuwrf/io/gwdo_static.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/physics/surface_layer.py`
- narrow adjacent State/config/input-loader files only if required to source
  TSK/ZNT faithfully.

Avoid `src/gpuwrf/dynamics/**`, memory/FP32 scaffolding, GPU run scripts, TOST,
Switzerland, release docs, and broad writer refactors. Do not use Fable/Mythos
or Hermes.

Allowed proof/test files:

- New primary proof: `proofs/v014/step1_tsk_znt_sourcing_fix.py`
- New proof outputs: `proofs/v014/step1_tsk_znt_sourcing_fix.{json,md}`
- WRF patch archive: `proofs/v014/step1_tsk_znt_sourcing_fix_wrf_patch.diff`
- updates to `proofs/v014/step1_sfclay_boundary_fix.{py,json,md}`;
- updates to `proofs/v014/step1_source_fidelity_closure.{py,json,md}`;
- updates to `proofs/v014/mynn_driver_source_output_fix.{py,json,md}` if needed;
- `.agent/reviews/2026-06-10-v014-step1-tsk-znt-sourcing.md`
- focused tests under `tests/`.

## Acceptance Gates

Required, manager-rerunnable:

```bash
python -m py_compile proofs/v014/step1_tsk_znt_sourcing_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_tsk_znt_sourcing_fix.py
python -m json.tool proofs/v014/step1_tsk_znt_sourcing_fix.json >/tmp/step1_tsk_znt_sourcing_fix.validated.json
python -m py_compile proofs/v014/step1_sfclay_boundary_fix.py proofs/v014/step1_source_fidelity_closure.py proofs/v014/mynn_driver_source_output_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_sfclay_boundary_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_source_fidelity_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/mynn_driver_source_output_fix.py
python -m json.tool proofs/v014/step1_sfclay_boundary_fix.json >/tmp/step1_sfclay_boundary_fix.validated.json
python -m json.tool proofs/v014/step1_source_fidelity_closure.json >/tmp/step1_source_fidelity_closure.validated.json
python -m json.tool proofs/v014/mynn_driver_source_output_fix.json >/tmp/mynn_driver_source_output_fix.validated.json
git diff --check
```

If production code changes, add or update focused CPU tests and run them.

Pass target:

- Preferred: strict Step-1 `after_conv_t_tendf_to_moist` vs JAX dry `T_TENDF`
  nested-interior max_abs `<= 1.0e-3`, RMSE `<= 1.0e-5`.
- Acceptable narrowing: exact WRF-anchored blocker strictly later or narrower
  than TSK/ZNT sourcing, with proof that TSK/ZNT is fixed or bounded.

## Performance Constraints

- CPU-only unless the manager explicitly grants a short GPU probe.
- No production CPU-WRF dependency.
- No host/device transfer inside timestep loops.
- Any new runtime state must be static-shape and resident.

## Handoff Requirements

Write:

- `proofs/v014/step1_tsk_znt_sourcing_fix.md`
- `.agent/reviews/2026-06-10-v014-step1-tsk-znt-sourcing.md`
- sprint report drafts if complete.

Completion marker:

`GPT STEP1_TSK_ZNT_SOURCING DONE - see proofs/v014/step1_tsk_znt_sourcing_fix.md`

Notify manager pane `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_TSK_ZNT_SOURCING DONE - see proofs/v014/step1_tsk_znt_sourcing_fix.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
