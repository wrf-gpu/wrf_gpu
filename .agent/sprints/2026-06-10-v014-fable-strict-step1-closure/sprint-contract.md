# Sprint Contract: V0.14 Fable Strict Step-1 Closure

Date: 2026-06-10 WEST
Owner: Fable/Mythos in tmux `0:1`
Manager: `worker/gpt/v013-close-manager`
Base commit: `94fe5d5f`

## Objective

Close the current v0.14 strict Step-1 grid-parity blocker as one endpoint-sized
task.

Current committed verdict:
`NOAHMP_STEP1_WIRED_STRICT_RED_NARROWED_TO_RADIATION_FORCING_INTO_NOAHMP`.

Preferred endpoint:

- strict Step-1 green in `proofs/v014/noahmp_step1_closure.py`
  (`max_abs <= 1.0e-3`, `rmse <= 1.0e-5`) after production/proof fixes; or
- acceptable fallback: an exact WRF-anchored blocker narrower than the current
  two-lane split, with ranked hypotheses, proof artifacts, performance/safety
  implications, and the fastest next command.

Do not return a micro-blocker. If you prove the first suspected bug wrong, use
that context to continue toward the whole endpoint.

## Current Evidence

Accepted artifacts:

- `.agent/reviews/2026-06-10-v014-fable-noahmp-energy-closure.md`
- `proofs/v014/noahmp_land_tile_energy_closure.{py,json,md}`
- `proofs/v014/noahmp_step1_closure.{py,json,md}`
- `proofs/v014/moist_theta_physics_consumer_audit.{json,md}`
- `proofs/v014/rrtmg_step1_forcing_parity.{py,json,md}`
- `.agent/sprints/2026-06-10-v014-fable-noahmp-energy-closure/manager-closeout.md`

Critical facts:

- NoahMP land-tile energy is closed. The JAX energy solve is exact under WRF
  NMPIN; the production `noahmp_coupler.assemble_noahmp_forcing` now decouples
  WRF moist theta `theta_m` to dry theta before Exner conversion.
- Strict Step-1 still fails: max_abs `1489.5135568470864`, RMSE
  `12.146876720723487`.
- Worst cell is water: Fortran `(i=66, j=37, k=3)`, WRF `-2457.6`, JAX `-968.1`.
  NoahMP does not run there.
- Ranked current lanes:
  1. `surface_layer.py` / sfclay-MYNN water path likely has the same
     `theta_m -> dry T` bug through `_potential_to_temperature`.
  2. RRTMG Step-1 forcing parity: GLW bias about `+17.44 W/m2`, SWDOWN RMSE
     about `2.76 W/m2`, mass-coupled RTHRATEN max_abs about `19.4`.
- Station TOST and Switzerland-GPU remain blocked until grid divergence is no
  longer radical or is formally bounded.

## Required Work

1. Read governing docs and accepted artifacts listed above.
2. First attack the surface-layer/sfclay-MYNN water-path theta boundary:
   prove/refute the suspected `theta_m -> theta_dry` fix with WRF-anchored
   hooks or minimal same-input harnesses, then implement if local.
3. Re-run strict Step-1 after any local fix. If strict is still red, continue
   to RRTMG Step-1 forcing parity rather than stopping at a narrow note.
4. For RRTMG, build the smallest WRF-anchored forcing hook/comparator needed to
   decide whether the remaining GLW/SWDOWN/RTHRATEN residual is an input/derived
   profile mismatch, an optical/gas/top-buffer convention mismatch, or a kernel
   bug. Fix if local and safe; otherwise return a narrower blocker with exact
   proof and a release-scope recommendation.
5. Preserve GPU-native performance structure. No clamps, tolerance widening,
   CPU-WRF runtime dependency, or host/device transfer inside timestep loops.

## File Ownership

Allowed production files if proven needed:

- `src/gpuwrf/physics/surface_layer.py`
- `src/gpuwrf/coupling/scan_adapters.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/physics/rrtmg_lw.py`
- `src/gpuwrf/physics/rrtmg_sw.py`
- `src/gpuwrf/physics/rrtmg_constants.py`
- focused helper/constants files if a shared dry/moist conversion helper is
  demonstrably needed.

Allowed tests:

- `tests/test_m6_surface_layer_kernel.py`
- `tests/test_v014_mynn_surface_layer_regressions.py`
- `tests/test_v014_mynn_coldstart_init.py`
- `tests/test_v014_dry_source_leaf_wiring.py`
- `tests/test_m5_rrtmg_*.py`
- new focused tests under `tests/`

Allowed proof/review files:

- `proofs/v014/noahmp_step1_closure.*`
- new `proofs/v014/surface_layer_theta_decoupling.*`
- new `proofs/v014/rrtmg_step1_forcing_closure.*`
- `.agent/reviews/2026-06-10-v014-fable-strict-step1-closure.md`

Do not edit:

- TOST, Switzerland/Gotthard, Grid-Delta Atlas, FP32/memory lanes.
- Unrelated dycore/runtime code unless the proof shows the Step-1 closure
  directly requires it.

## Acceptance Gates

Required:

```bash
python -m py_compile proofs/v014/noahmp_step1_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/noahmp_step1_closure.py
python -m json.tool proofs/v014/noahmp_step1_closure.json >/tmp/noahmp_step1_closure.validated.json
git diff --check
```

If surface-layer production changes, also run:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_m6_surface_layer_kernel.py tests/test_v014_mynn_surface_layer_regressions.py tests/test_v014_mynn_coldstart_init.py tests/test_v014_dry_source_leaf_wiring.py
```

If RRTMG production changes, also run relevant RRTMG tests, at minimum:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_m5_rrtmg_gate.py tests/test_m5_rrtmg_tier1.py tests/test_m5_rrtmg_intermediate_oracles.py tests/test_rrtmg_topographic_coupling.py
```

Add focused proof scripts and JSON validation for any new proof object.

## Constraints

- CPU proof work preferred; no GPU needed unless the manager explicitly approves.
- Keep output context-sparing.
- Respect the existing dirty worktree; do not stage/revert unrelated files.
- If you need to touch both surface-layer and RRTMG, keep the causal proof clear
  enough that the manager can split or accept the commit safely.

## Handoff Requirements

Write:

- `.agent/reviews/2026-06-10-v014-fable-strict-step1-closure.md`
- proof JSON/Markdown for any new localizations/fixes
- updated `noahmp_step1_closure` proof outputs
- focused tests if production changed

Handoff must include objective, files changed, commands run, proof objects,
unresolved risks, next decision needed, and whether TOST/Switzerland-GPU remain
blocked.

Completion marker to manager pane `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'FABLE STRICT_STEP1_CLOSURE DONE - see .agent/reviews/2026-06-10-v014-fable-strict-step1-closure.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
