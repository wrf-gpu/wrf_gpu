# GPT v0.6.0 Noah-Classic Operational Coupler Handoff

## objective

Wire Noah-classic (`sf_surface_physics=2`) into the operational JAX scan's land-coupling path, mirroring the Noah-MP surface-layer -> LSM -> atmosphere feedback pattern, while preserving Noah-classic savepoint parity and keeping unsupported selections fail-closed.

## files changed

- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/src/gpuwrf/coupling/noahclassic_surface_hook.py`
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/src/gpuwrf/runtime/operational_mode.py`
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/src/gpuwrf/runtime/operational_state.py`
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/src/gpuwrf/coupling/scan_adapters.py`
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/proofs/v060/noah_coupler_smoke.py`
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/proofs/v060/noah_coupler_smoke.json`
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/proofs/v060/gen_noah_coupler_report.py`
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/proofs/v060/noah_coupler_report.json`
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/proofs/v060/gen_scanwire_report.py`
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/proofs/v060/scanwire_smoke.py`
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/proofs/v060/scanwire_report.json`
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/proofs/v060/forecast_gate_harness.py`
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/proofs/v060/forecast_gate_readiness.json`

## implementation summary

- Added `NoahClassicStatic`, `NoahClassicLandState`, and `NoahClassicRadiation` plus the operational `noahclassic_surface_step` hook.
- The hook consumes surface-layer exchange handles (`ustar`, `theta_flux`, `qv_flux`, `rhosfc`), builds WRF-derived SFLX forcing, advances the 4-layer Noah-classic carry, and writes land `TSK`/top `SMOIS`/moisture availability/roughness/HFX-derived flux handles back to `State`.
- Extended `OperationalCarry` with `noahclassic_land` and `noahclassic_rad`.
- Extended `OperationalNamelist` with explicit `noahclassic_static`, `noahclassic_land`, and optional held radiation.
- Updated `_resolve_operational_suite` so explicit `sf_surface_physics=2` resolves only with the required Noah-classic bundle; missing bundle remains fail-closed.
- Threaded Noah-classic after the selected surface-layer adapter and before MYNN in the physics boundary step, matching the Noah-MP coupling order.
- Updated M9 diagnostic overlay so Noah-classic land HFX/LH/TSK reports come from the latest land carry over land tiles.

## commands run

- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m py_compile src/gpuwrf/coupling/noahclassic_surface_hook.py src/gpuwrf/runtime/operational_state.py src/gpuwrf/runtime/operational_mode.py src/gpuwrf/coupling/physics_dispatch.py proofs/v060/noah_coupler_smoke.py proofs/v060/gen_noah_coupler_report.py proofs/v060/scanwire_smoke.py proofs/v060/gen_scanwire_report.py proofs/v060/forecast_gate_harness.py`
- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src:. pytest -q tests/v060/test_noahclassic_parity.py --tb=short`
- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src:. pytest -q tests/test_v060_physics_dispatch.py --tb=short`
- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=src:. python proofs/v060/noah_coupler_smoke.py --out proofs/v060/noah_coupler_smoke.json`
- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=src:. python proofs/v060/gen_noah_coupler_report.py`
- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src:. python proofs/v060/forecast_gate_harness.py --validate --out proofs/v060/forecast_gate_readiness.json`
- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src:. python proofs/v060/gen_scanwire_report.py`
- `git diff --check`

## proof objects produced

- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/proofs/v060/noah_coupler_report.json`: `overall_pass=true`, parity `PASS`, smoke `pass=true`, dispatcher `gpu_runnable=true`.
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/proofs/v060/noah_coupler_smoke.json`: CPU JAX operational physics-slot scan, `sf_surface_physics=2`, finite state/land, nonzero HFX/QFX/GRDFLX feedback, water-tile land carry unchanged, bounded soil-water relative change `0.0005391933283350839`.
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/proofs/v060/scanwire_report.json`: regenerated rollup, `overall_pass=true`, Noah-classic counted as scan-wired with explicit bundle.
- `/home/enric/src/wrf_gpu2/.claude/worktrees/v060-noahcoupler-gpt/proofs/v060/forecast_gate_readiness.json`: regenerated readiness text; canonical combo_2 remains not scan-runnable because it selects host-NumPy YSU and does not assemble a Noah-classic bundle.

## unresolved risks

- The smoke proof is a CPU JAX physics-slot `lax.scan`, not a full dycore forecast segment or GPU forecast gate. GPU was intentionally not used.
- Full real-run initialization still needs production assembly of `NoahClassicStatic`/`NoahClassicLandState` from WRF/metgrid/history inputs for canonical combo_2.
- Noah-classic precipitation forcing is currently zero in the coupling assembler, matching the limited smoke scope; hydrology feedback from operational precipitation still needs explicit plumbing.
- CH is inferred from surface-layer flux handles when the surface-layer adapter does not expose CH directly.
- Restart/checkpoint serialization for `noahclassic_land`/`noahclassic_rad` is not wired yet.

## next decision needed

Decide whether the manager wants the next sprint to plumb real-run Noah-classic bundle initialization and restart I/O, or to first unblock canonical combo_2 by rewriting YSU for JAX scan traceability.
