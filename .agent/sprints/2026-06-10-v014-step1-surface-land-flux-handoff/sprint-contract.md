# Sprint Contract: V0.14 Step-1 Surface/Land Flux Handoff

Date: 2026-06-10 WEST
Owner: GPT-5.5 xhigh worker in tmux
Manager: `worker/gpt/v013-close-manager`
Base commit: `919334c0 v014 narrow mynn source to land flux handoff`

## Objective

Close, or strictly narrow, the Step-1 heat/moisture flux handoff divergence
between WRF surface/land physics and the MYNN driver.

The accepted prior sprint proved that MYNN raw source units are not the primary
blocker when WRF inputs and WRF-initialized QKE are provided. The remaining
strong signal is WRF-internal: between `SFCLAY1D_mynn` output and
`module_bl_mynnedmf_driver` input, UST is preserved but HFX/QFX are changed.
Find where and why, then fix the JAX production path if local and
performance-compatible.

## Accepted Prior Proofs

- `proofs/v014/step1_mynn_source_coupling.{py,json,md}`
- `proofs/v014/step1_mynn_source_coupling_wrf_patch.diff`
- `.agent/reviews/2026-06-10-v014-step1-mynn-source-coupling.md`
- `.agent/sprints/2026-06-10-v014-step1-mynn-source-coupling/manager-closeout.md`

Current facts:

- Strict after-conv `T_TENDF` remains red:
  - max_abs `438.5379097262689`
  - RMSE `5.4654420375782955`
  - worst cell `{'i': 74, 'j': 39, 'k': 1}`
- WRF MYNN inputs plus WRF initialized QKE exonerate the MYNN kernel/source
  units:
  - raw `RTHBLTEN` max_abs `0.00026206000797283305`
  - RMSE `2.5971191677632803e-06`
  - corr `0.9999580118448544`
- WRF `SFCLAY1D_mynn` output -> WRF MYNN-driver input:
  - UST max_abs `4.998779168374767e-12`
  - HFX max_abs `277.80298614281253`, RMSE `23.78077473308822`
  - QFX max_abs `1.4684322196e-05`, RMSE `1.0634310887382864e-06`

## Required Work

1. Add or rerun disposable WRF hooks immediately before and after the surface
   land-flux update that sits between surface-layer output and MYNN input.
   Prior evidence suggests `module_surface_driver` with
   `sf_surface_physics=4`, but treat that as a hypothesis. Confirm the actual
   WRF path and namelist values from the hook, not from memory.
2. Capture enough WRF-side raw arrays to distinguish:
   - surface-layer output handles (`UST`, HFX, QFX, LH, FLHC/FLQC if present,
     CHS/CHS2/CQS2 if present);
   - land-surface update outputs (`HFX`, QFX, LH, `TSK`, `GRDFLX`, soil
     moisture/temperature updates if local to the flux change);
   - MYNN-driver input handles immediately before `module_bl_mynnedmf_driver`;
   - relevant namelist switches (`sf_surface_physics`, `sf_sfclay_physics`,
     `bl_pbl_physics`, Noah/Noah-MP selection, land/ocean masks).
3. Compare WRF hook surfaces against the current JAX Step-1 path. Explicitly
   test whether the JAX path is skipping an LSM/land flux overlay, using the
   wrong namelist field, applying the overlay in the wrong order, or using a
   stale surface state.
4. If evidence proves a local bug, fix production code. Keep edits scoped and
   GPU-native: no host/device transfers in timestep loops, no CPU-WRF
   dependency, no dynamic-shape runtime arrays, and no clamps that hide
   divergence.
5. Rerun `proofs/v014/step1_mynn_source_coupling.py` after any fix. Preferred
   closure is strict Step-1 `after_conv_t_tendf_to_moist` vs JAX dry `T_TENDF`
   max_abs `<= 1.0e-3`, RMSE `<= 1.0e-5`. Acceptable narrowing is an exact
   WRF-anchored blocker strictly later or narrower than the surface/land flux
   handoff, with a ranked hypothesis table and fastest next command.

## File Ownership

Allowed production files:

- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/coupling/driver.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/physics/noah*.py`
- `src/gpuwrf/physics/surface_layer.py`
- `src/gpuwrf/coupling/scan_adapters.py`

Only touch files needed by proof. Avoid dycore, FP32, memory layout,
TOST, Switzerland/demo validation, release packaging, and broad docs. Do not
use Hermes or Fable/Mythos.

Allowed proof/test files:

- `proofs/v014/step1_surface_land_flux_handoff.py`
- `proofs/v014/step1_surface_land_flux_handoff.{json,md}`
- `proofs/v014/step1_surface_land_flux_handoff_wrf_patch.diff`
- focused tests under `tests/` if production code changes
- updates to current Step-1 proof artifacts only when rerun evidence requires
  them
- `.agent/reviews/2026-06-10-v014-step1-surface-land-flux-handoff.md`

## Acceptance Gates

Required, manager-rerunnable:

```bash
python -m py_compile proofs/v014/step1_surface_land_flux_handoff.py proofs/v014/step1_mynn_source_coupling.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_surface_land_flux_handoff.py
python -m json.tool proofs/v014/step1_surface_land_flux_handoff.json >/tmp/step1_surface_land_flux_handoff.validated.json
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_mynn_source_coupling.py
python -m json.tool proofs/v014/step1_mynn_source_coupling.json >/tmp/step1_mynn_source_coupling.validated.json
git diff --check
```

If production code changes, add/update focused CPU tests and run the relevant
subset. At minimum include existing MYNN/surface regressions if touched:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_mynn_surface_layer_regressions.py tests/test_m6_surface_layer_kernel.py tests/test_v014_dry_source_leaf_wiring.py tests/test_v014_mynn_coldstart_init.py
```

## Performance Constraints

- CPU-only unless the manager explicitly grants a short low-VRAM GPU probe.
- Preserve GPU-native, vectorized JAX paths.
- No host/device transfer inside timestep loops.
- No new dynamic-shape arrays or Python loops on runtime grid size.
- No correctness clamps.

## Handoff Requirements

Write:

- `proofs/v014/step1_surface_land_flux_handoff.md`
- `.agent/reviews/2026-06-10-v014-step1-surface-land-flux-handoff.md`
- concise worker/test/review closeout drafts if complete.

Completion marker:

`GPT STEP1_SURFACE_LAND_FLUX_HANDOFF DONE - see proofs/v014/step1_surface_land_flux_handoff.md`

Notify manager pane `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_SURFACE_LAND_FLUX_HANDOFF DONE - see proofs/v014/step1_surface_land_flux_handoff.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
