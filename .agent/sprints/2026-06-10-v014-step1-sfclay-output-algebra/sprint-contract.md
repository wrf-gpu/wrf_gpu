# Sprint Contract: V0.14 Step-1 SFCLAY Output Algebra

Date: 2026-06-10 04:29 WEST
Owner: GPT-5.5 xhigh worker in tmux
Manager: `worker/gpt/v013-close-manager`
Base commit: `bdf68332 v014 fix sfclay thermo column inputs`

## Objective

Close, or reduce to one strictly narrower WRF-anchored blocker, the remaining
Step-1 MYNN surface-layer output mismatch after the `sfclay_mynn` input tuple
was fixed.

This is a whole-debug sprint. The target is now inside WRF
`module_sf_mynn.F` / `SFCLAY1D_mynn`, not the upstream coupler. Use a narrow
internal WRF hook to freeze the algebra boundary, compare against
`surface_layer_with_diagnostics` on the fixed input tuple, then implement a
local fix only when the evidence proves it.

## Accepted Prior Proofs

- `proofs/v014/mynn_driver_source_output_fix.{py,json,md}`
- `proofs/v014/step1_tsk_znt_sourcing_fix.{py,json,md}`
- `proofs/v014/step1_thermo_column_inputs.{py,json,md}`
- `proofs/v014/step1_source_fidelity_closure.{py,json,md}`
- `.agent/reviews/2026-06-10-v014-step1-thermo-column-inputs.md`

Current facts:

- MYNN cold-start QKE is fixed and the MYNN kernel is exonerated with WRF inputs.
- MYNN first-call surface semantics are fixed.
- `TSK/ZNT/MAVAIL` and WRF `phy_prep` thermodynamic surface inputs are fixed at
  the exact `sfclay_mynn` hook.
- Fixed input maxima:
  - `th_phy(kts)` max_abs `6.71089752017906e-05 K`;
  - `t_phy(kts)` max_abs `0.013577942721781255 K`;
  - hydrostatic `p_phy(kts)` max_abs `0.015625 Pa`;
  - `dz8w(kts)` max_abs `0.00018988715282830526 m`;
  - `psfc` max_abs `0.015625 Pa`.
- Surface outputs remain red after fixed inputs:
  - `UST` max_abs `0.01231782267117762`, RMSE `0.0007831552182476275`;
  - `HFX` max_abs `27.09163832864155`, RMSE `1.3972156996573475`;
  - `QFX` max_abs `2.744275103194571e-07`, RMSE `8.62190066790179e-08`;
  - `BR` max_abs `2.0`, RMSE `0.7681227693986273`.
- Strict Step-1 after-conv residual remains red: max_abs
  `847.1445725702908`, RMSE `9.56593990212596`.

## Required Work

1. Add a disposable WRF internal hook inside `module_sf_mynn.F` /
   `SFCLAY1D_mynn` for the current d02 Step-1 case. Capture enough internal
   arrays/scalars to localize the output algebra, at minimum:
   `thx`, `thgb`, `br`, `zol`, `psim`, `psih`, `psit`, `ust`, `hfx`, `qfx`,
   plus any upstream locals needed to explain them (`zq`, `za`, `wspd`, `mol`,
   `rmol`, `qsfc`, `qgh`, `ch`, `ck` if present).
2. Compare the exact WRF internal values against JAX
   `surface_layer_with_diagnostics(_surface_column_view(state, grid),
   first_timestep=True)` on the fixed input tuple.
3. Localize whether the mismatch is caused by:
   - a MYNN-SL stability-function branch or lookup table;
   - first-call `zol`/`rmol`/`mol` seeding still missing at an internal point;
   - `thgb/thx` or 2 m/surface temperature algebra;
   - `wspd`/roughness/height constants;
   - unit/sign convention for heat/moisture flux;
   - or a strictly later MYNN/PBL source coupling issue.
4. If local and performance-compatible, fix production `surface_layer.py` (or
   a narrow adjacent surface constant/helper) and focused tests.
5. Produce `proofs/v014/step1_sfclay_output_algebra.{py,json,md}` and WRF hook
   archive `proofs/v014/step1_sfclay_output_algebra_wrf_patch.diff`.
6. Rerun the current Step-1 proof chain enough to prove the new blocker status.
7. If not fixable in one sprint, return one exact narrower blocker, a ranked
   hypothesis table, and the fastest next command.

## File Ownership

Allowed production files:

- `src/gpuwrf/physics/surface_layer.py`
- `src/gpuwrf/physics/surface_constants.py`
- `src/gpuwrf/coupling/physics_couplers.py` only if diagnostics need a tiny
  adapter adjustment.

Avoid dycore, memory/FP32 scaffolding, GPU run scripts, TOST, Switzerland,
release docs, and broad refactors. Do not use Hermes or Fable/Mythos.

Allowed proof/test files:

- `proofs/v014/step1_sfclay_output_algebra.py`
- `proofs/v014/step1_sfclay_output_algebra.{json,md}`
- `proofs/v014/step1_sfclay_output_algebra_wrf_patch.diff`
- focused surface tests under `tests/`;
- updates to current v0.14 Step-1 proof artifacts when rerun evidence requires
  them;
- `.agent/reviews/2026-06-10-v014-step1-sfclay-output-algebra.md`

## Acceptance Gates

Required, manager-rerunnable:

```bash
python -m py_compile proofs/v014/step1_sfclay_output_algebra.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_sfclay_output_algebra.py
python -m json.tool proofs/v014/step1_sfclay_output_algebra.json >/tmp/step1_sfclay_output_algebra.validated.json
python -m py_compile proofs/v014/step1_thermo_column_inputs.py proofs/v014/step1_tsk_znt_sourcing_fix.py proofs/v014/step1_source_fidelity_closure.py proofs/v014/mynn_driver_source_output_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_thermo_column_inputs.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_tsk_znt_sourcing_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_source_fidelity_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/mynn_driver_source_output_fix.py
python -m json.tool proofs/v014/step1_thermo_column_inputs.json >/tmp/step1_thermo_column_inputs.validated.json
python -m json.tool proofs/v014/step1_tsk_znt_sourcing_fix.json >/tmp/step1_tsk_znt_sourcing_fix.validated.json
python -m json.tool proofs/v014/step1_source_fidelity_closure.json >/tmp/step1_source_fidelity_closure.validated.json
python -m json.tool proofs/v014/mynn_driver_source_output_fix.json >/tmp/mynn_driver_source_output_fix.validated.json
git diff --check
```

If production code changes, add/update focused CPU tests and run them.

Pass target:

- Preferred: strict Step-1 `after_conv_t_tendf_to_moist` vs JAX dry `T_TENDF`
  nested-interior max_abs `<= 1.0e-3`, RMSE `<= 1.0e-5`.
- Acceptable narrowing: exact WRF-anchored blocker strictly later or narrower
  than surface-layer output algebra, with proof that `UST/HFX/QFX/BR` are fixed
  or bounded at the WRF internal hook.

## Performance Constraints

- CPU-only unless the manager explicitly grants a short low-VRAM GPU probe.
- No host/device transfer inside timestep loops.
- No new dynamic-shape arrays.
- Keep the surface path JAX/JIT-compatible and vectorized.

## Handoff Requirements

Write:

- `proofs/v014/step1_sfclay_output_algebra.md`
- `.agent/reviews/2026-06-10-v014-step1-sfclay-output-algebra.md`
- concise worker/test/review closeout drafts if complete.

Completion marker:

`GPT STEP1_SFCLAY_OUTPUT_ALGEBRA DONE - see proofs/v014/step1_sfclay_output_algebra.md`

Notify manager pane `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_SFCLAY_OUTPUT_ALGEBRA DONE - see proofs/v014/step1_sfclay_output_algebra.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
