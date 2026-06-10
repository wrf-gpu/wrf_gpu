# Sprint Contract: V0.14 Step-1 MYNN Source Coupling

Date: 2026-06-10 WEST
Owner: GPT-5.5 xhigh worker in tmux
Manager: `worker/gpt/v013-close-manager`
Base commit: `fc3c9fd9 v014 bound sfclay output algebra`

## Objective

Close, or reduce to one strictly narrower WRF-anchored blocker, the remaining
Step-1 dry source divergence after `SFCLAY1D_mynn` output algebra was bounded.

The current blocker is believed to be in MYNN/PBL source coupling after fixed
surface outputs. Treat that as the leading hypothesis, not a fact. Build the
fastest rigorous proof loop that compares exact WRF MYNNEDMF inputs, raw
post-driver source arrays, and JAX `mynn_adapter_with_source_leaves` outputs.
If the leading hypothesis is false, use the same context to rank and cheaply
test the next hypotheses rather than returning only a negative result.

## Accepted Prior Proofs

- `proofs/v014/step1_sfclay_output_algebra.{py,json,md}`
- `proofs/v014/step1_sfclay_output_algebra_wrf_patch.diff`
- `.agent/reviews/2026-06-10-v014-step1-sfclay-output-algebra.md`
- `proofs/v014/mynn_driver_source_output_fix.{py,json,md}`
- `proofs/v014/step1_source_fidelity_closure.{py,json,md}`
- `proofs/v014/step1_tsk_znt_sourcing_fix.{py,json,md}`
- `proofs/v014/step1_thermo_column_inputs.{py,json,md}`

Current facts:

- MYNN cold-start QKE is fixed and the MYNN kernel is exonerated with WRF
  inputs.
- `TSK/ZNT/MAVAIL`, WRF `phy_prep` thermodynamic inputs, and MYNN surface-layer
  output algebra are now bounded at their WRF hooks.
- Surface boundary after the latest fix:
  - `UST` max_abs `0.0007252174862408534`, RMSE `1.53999402707944e-05`.
  - `HFX` max_abs `0.2643125302157898`, RMSE `0.022548398654638105`.
  - `QFX` max_abs `6.468560998136325e-08`, RMSE `3.002727253934746e-08`.
  - `BR` max_abs `0.01166976922050278`, RMSE `0.0003583716190119449`.
  - `rho` max_abs `0.00018143653869628906`, RMSE `7.786468426065368e-06`.
- Strict Step-1 after-conv `T_TENDF` remains red:
  - max_abs `847.1446969755725`;
  - RMSE `9.627208432391289`;
  - worst cell `{'i': 64, 'j': 37, 'k': 2}`.

## Required Work

1. Add or rerun a disposable WRF hook around `module_pbl_driver` /
   `module_bl_mynnedmf_driver` after the fixed MYNN surface outputs. Archive the
   exact patch under `proofs/v014/step1_mynn_source_coupling_wrf_patch.diff`.
2. Capture the WRF-side source boundary with enough raw arrays to distinguish
   adapter/coupling bugs from true MYNN kernel bugs. At minimum:
   - exact MYNNEDMF input fluxes and surface fields: `UST`, `HFX`, `QFX`,
     `RHO`, `FLHC`, `FLQC`, `CHS`, `CHS2`, `CQS2`, `MOL/RMOL`, `PBLH`, `EL_PBL`;
   - first-call turbulent state: `QKE`, any relevant `EXCH_H/M`, eddy or mixing
     arrays if present;
   - raw post-driver thermal/moisture source arrays such as `dth1`, `dqv1`, or
     the closest WRF-local equivalent before `module_em` mass scaling;
   - final mass-scaled `RTHBLTEN` and `RQVBLTEN` as used by the Step-1 dry
     source path.
3. Compare against the JAX call path that currently feeds Step-1:
   `_mynn_column_from_state`, `_surface_fluxes_from_state`, and
   `mynn_adapter_with_source_leaves`.
4. Localize whether the order-847 residual is caused by:
   - source sign/unit/mass-scaling semantics;
   - lowest-layer indexing or stagger/unstagger mismatch;
   - wrong density/mu/geopotential column passed into MYNN;
   - missing first-call MYNN initialization field;
   - accidental state mutation before/after source capture;
   - or another strictly later/non-MYNN source boundary.
5. If evidence proves a local, performance-compatible bug, fix production code
   in the allowed files and add focused regression tests. Do not introduce
   host/device transfers in timestep loops, CPU-only fallbacks, dynamic-shape
   arrays, broad refactors, or correctness clamps.
6. If full closure is not possible in this sprint, return one exact narrower
   WRF-anchored blocker, a ranked hypothesis table with rejected hypotheses, and
   the fastest next command.

## File Ownership

Allowed production files:

- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/physics/mynn_pbl.py`
- `src/gpuwrf/runtime/operational_mode.py`

Allowed only with explicit proof need:

- a tiny adjacent adapter/test helper in `src/gpuwrf/physics/surface_layer.py`.

Avoid dycore, memory/FP32, TOST, Switzerland/demo validation, release packaging,
GPU run scripts, and broad documentation. Do not use Hermes or Fable/Mythos.

Allowed proof/test files:

- `proofs/v014/step1_mynn_source_coupling.py`
- `proofs/v014/step1_mynn_source_coupling.{json,md}`
- `proofs/v014/step1_mynn_source_coupling_wrf_patch.diff`
- focused tests under `tests/` if production code changes;
- updates to current Step-1 proof artifacts only when rerun evidence requires
  them;
- `.agent/reviews/2026-06-10-v014-step1-mynn-source-coupling.md`.

## Acceptance Gates

Required, manager-rerunnable:

```bash
python -m py_compile proofs/v014/step1_mynn_source_coupling.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_mynn_source_coupling.py
python -m json.tool proofs/v014/step1_mynn_source_coupling.json >/tmp/step1_mynn_source_coupling.validated.json
python -m py_compile proofs/v014/step1_sfclay_output_algebra.py proofs/v014/step1_source_fidelity_closure.py proofs/v014/mynn_driver_source_output_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_sfclay_output_algebra.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_source_fidelity_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/mynn_driver_source_output_fix.py
python -m json.tool proofs/v014/step1_sfclay_output_algebra.json >/tmp/step1_sfclay_output_algebra.validated.json
python -m json.tool proofs/v014/step1_source_fidelity_closure.json >/tmp/step1_source_fidelity_closure.validated.json
python -m json.tool proofs/v014/mynn_driver_source_output_fix.json >/tmp/mynn_driver_source_output_fix.validated.json
git diff --check
```

If production code changes, add/update focused CPU tests and run them. Suggested
baseline:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_mynn_surface_layer_regressions.py tests/test_m6_surface_layer_kernel.py tests/test_v014_dry_source_leaf_wiring.py
```

Pass target:

- Preferred: strict Step-1 `after_conv_t_tendf_to_moist` vs JAX dry `T_TENDF`
  nested-interior max_abs `<= 1.0e-3`, RMSE `<= 1.0e-5`.
- Acceptable narrowing: exact WRF-anchored blocker strictly later or narrower
  than MYNN/PBL source coupling, with proof that MYNN input fluxes and raw
  post-driver source arrays are fixed or bounded.

## Performance Constraints

- CPU-only unless the manager explicitly grants a short low-VRAM GPU probe.
- Preserve GPU-native, vectorized JAX paths.
- No host/device transfer inside timestep loops.
- No new dynamic-shape arrays or Python loops on runtime grid size.
- No correctness clamps that hide divergence.

## Handoff Requirements

Write:

- `proofs/v014/step1_mynn_source_coupling.md`
- `.agent/reviews/2026-06-10-v014-step1-mynn-source-coupling.md`
- concise worker/test/review closeout drafts if complete.

Completion marker:

`GPT STEP1_MYNN_SOURCE_COUPLING DONE - see proofs/v014/step1_mynn_source_coupling.md`

Notify manager pane `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_MYNN_SOURCE_COUPLING DONE - see proofs/v014/step1_mynn_source_coupling.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
