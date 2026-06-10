# Sprint Contract: V0.14 Fable NoahMP Step-1 Closure

Date: 2026-06-10 WEST
Owner: Fable/Mythos in tmux `0:1`
Manager: `worker/gpt/v013-close-manager`
Base commit: current manager HEAD plus the accepted surface/land flux proof
artifacts

## Objective

Close the current v0.14 Step-1 grid-parity blocker as one whole task.

The prior GPT sprint proved that WRF's heat/moisture flux handoff into MYNN is
not mysterious: `SFCLAY1D_mynn` output equals `PRE_NOAHMP`, NoahMP is the exact
HFX/QFX change point, and `POST_NOAHMP` equals the MYNN driver input. The
remaining blocker is that the JAX Step-1 live-nest/source-capture path is built
with NoahMP disabled or missing the WRF-derived NoahMP land/static/radiation
state.

This is intentionally **not** a micro-run. The endpoint should be a roadmap
checkbox: strict Step-1 fixed and proven, or the exact remaining blocker proven
narrower than the current NoahMP-disabled configuration.

## Current Evidence

Accepted proof objects:

- `proofs/v014/step1_mynn_source_coupling.{py,json,md}`
- `proofs/v014/step1_surface_land_flux_handoff.{py,json,md}`
- `proofs/v014/step1_surface_land_flux_handoff_wrf_patch.diff`
- `.agent/reviews/2026-06-10-v014-step1-surface-land-flux-handoff.md`
- `.agent/sprints/2026-06-10-v014-step1-surface-land-flux-handoff/manager-closeout.md`

Critical facts:

- Strict after-conv `T_TENDF` remains red:
  - max_abs `438.5379097262689`
  - RMSE `5.4654420375782955`
- MYNN raw source units are exonerated when fed WRF inputs plus WRF initialized
  QKE:
  - raw `RTHBLTEN` max_abs `0.00026206000797283305`
  - RMSE `2.5971191677632803e-06`
  - corr `0.9999580118448544`
- WRF handoff boundaries:
  - `SFCLAY -> PRE_NOAHMP` HFX max_abs `5.000003966415534e-09`
  - `SFCLAY -> PRE_NOAHMP` QFX max_abs `4.999459505238002e-16`
  - `PRE_NOAHMP -> POST_NOAHMP` HFX max_abs `277.80298614000003`
  - `PRE_NOAHMP -> POST_NOAHMP` QFX max_abs `1.4684322196e-05`
  - `POST_NOAHMP -> MYNN` HFX/QFX/UST max_abs `0.0`
- JAX Step-1 proof config currently reports:
  - `use_noahmp=False`
  - `sf_surface_physics=None`
  - `inputs_have_noahmp_land=False`

## Required Work

1. Read the current Step-1 proof builders and production runtime path. Identify
   where the live-nest/source-capture construction drops or disables NoahMP
   state/configuration for the d02 Step-1 proof.
2. Wire the WRF-derived NoahMP land/static/radiation state into the JAX Step-1
   production/proof path with `sf_surface_physics=4` semantics where the WRF
   fixture requires it. Preserve GPU-native structure: no CPU-WRF dependency,
   no host/device transfer inside timestep loops, no dynamic-shape runtime
   arrays, and no clamps that hide divergence.
3. If the first local fix exposes additional immediately adjacent
   configuration/state bugs on the same Step-1 closure path, fix them in the
   same sprint rather than returning a micro-blocker.
4. Rerun strict proof gates and update proof artifacts. Preferred release gate:
   strict Step-1 `after_conv_t_tendf_to_moist` vs JAX dry `T_TENDF` max_abs
   `<= 1.0e-3`, RMSE `<= 1.0e-5`.
5. If that cannot be achieved, produce an exact WRF-anchored proof of the
   remaining blocker. It must be narrower than "JAX NoahMP disabled/missing
   land/static state", include a ranked hypothesis table, and name the fastest
   next proof command. A broad "more debugging needed" result is not acceptable.

## File Ownership

Allowed production files, if needed:

- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/coupling/driver.py`
- `src/gpuwrf/coupling/scan_adapters.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/physics/noah*.py`
- `src/gpuwrf/physics/noahmp/**`
- `src/gpuwrf/physics/surface_layer.py`
- state/config contracts only if strictly necessary for NoahMP land/static
  leaves

Allowed proof/test files:

- `proofs/v014/step1_mynn_source_coupling.*`
- `proofs/v014/step1_surface_land_flux_handoff.*`
- new focused `proofs/v014/*noahmp*step1*` artifacts if useful
- focused tests under `tests/`
- `.agent/reviews/2026-06-10-v014-fable-noahmp-step1-closure.md`

Do not touch TOST, Switzerland/Gotthard, Grid-Delta Atlas, release packaging,
broad FP32/memory work, or unrelated dycore code unless you prove the Step-1
NoahMP closure directly requires it.

## Acceptance Gates

Required, manager-rerunnable:

```bash
python -m py_compile proofs/v014/step1_mynn_source_coupling.py proofs/v014/step1_surface_land_flux_handoff.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_mynn_source_coupling.py
python -m json.tool proofs/v014/step1_mynn_source_coupling.json >/tmp/step1_mynn_source_coupling.validated.json
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_surface_land_flux_handoff.py
python -m json.tool proofs/v014/step1_surface_land_flux_handoff.json >/tmp/step1_surface_land_flux_handoff.validated.json
git diff --check
```

If production code changes, add/update focused CPU tests and run the relevant
subset. At minimum, include any touched MYNN/surface/NoahMP tests and:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_mynn_surface_layer_regressions.py tests/test_m6_surface_layer_kernel.py tests/test_v014_dry_source_leaf_wiring.py tests/test_v014_mynn_coldstart_init.py
```

## Performance Constraints

- CPU proof work is preferred. Use GPU only for short, low-VRAM sanity probes if
  necessary and only after checking no long validation job is running.
- Preserve whole-state device residency and vectorized JAX paths.
- No host/device transfer inside timestep loops.
- No CPU-WRF runtime dependency in production.
- No new dynamic-shape arrays on runtime grid size.
- No correctness clamps or tolerance masking.

## Handoff Requirements

Write:

- `.agent/reviews/2026-06-10-v014-fable-noahmp-step1-closure.md`
- updated proof Markdown/JSON for every rerun proof
- focused test(s), if production code changes

Handoff format:

- objective
- files changed
- commands run
- proof objects produced
- unresolved risks
- next decision needed, if any

Completion marker to manager pane `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'FABLE NOAHMP_STEP1_CLOSURE DONE - see .agent/reviews/2026-06-10-v014-fable-noahmp-step1-closure.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
