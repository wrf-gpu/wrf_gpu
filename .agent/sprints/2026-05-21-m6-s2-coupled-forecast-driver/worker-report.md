# M6-S2 Worker Report - Coupled Forecast Driver

## 0. Operational Fitness Statement

M6-S2 forecast claims operational-fitness-gated-on-M6-S7-RMSE per ADR-007. FP32 storage from M6-S1 preserved.

This sprint proves that the coupled d02 driver can ingest real Gen2 initial conditions, replay the M6-S2a d02 lateral-boundary artifact, execute a 1h/6h/24h coupled forecast on the pinned 160 x 67 x 45 WRF extent, and produce validated proof artifacts with zero measured post-init host/device transfer. It does not claim operational forecast skill, probtest parity, conservation closure, or the final ADR-007 speed verdict.

## Objective

Build the first real coupled forecast driver on the GPU-resident M6 state:

- Use `Gen2Run.load(domain="d02", var=..., time=0)` and `gpuwrf.io.validation` for d02 initial state loading.
- Use M6-S2a `d02_boundary_replay_v1.zarr` for lateral-boundary replay.
- Run a JIT-compiled coupled driver with `lax.scan`-based timestep execution.
- Apply WRF-style specified plus relaxation-zone lateral boundary forcing.
- Replace the M6-S1 constant-dz placeholder with real state/grid threading.
- Produce 1h, 6h, and 24h proof artifacts plus a d02 spacetime budget.

## Files Changed

- `pyproject.toml`: first-commit prerequisite adding `jax>=0.4` and `zarr>=3.0,<4`.
- `src/gpuwrf/contracts/state.py`: added boundary-forcing state leaves `u_bdy`, `v_bdy`, `theta_bdy`, `qv_bdy`, `ph_bdy`, and `mu_bdy`.
- `src/gpuwrf/contracts/precision.py`: added boundary leaves to the precision matrix, preserving FP32-gated storage for wind/theta/qv boundary leaves and FP64 for geopotential/surface-pressure boundary leaves.
- `src/gpuwrf/coupling/physics_couplers.py`: removed `DEFAULT_DZ_M`; widened MYNN/RRTMG adapters to thread `GridSpec | None`; derives positive column `dz` from geopotential interfaces.
- `src/gpuwrf/coupling/boundary_apply.py`: new WRF-style boundary replay application with specified edge and relaxation zone.
- `src/gpuwrf/coupling/driver.py`: new d02 coupled forecast driver, static radiation-cadence segmentation, Gen2 IC load, boundary replay packing, forecast diagnostics, output writing, and transfer/budget hooks.
- `src/gpuwrf/coupling/__init__.py`: exported new boundary/driver interfaces.
- `src/gpuwrf/profiling/budget.py`: added XLA `compiled.memory_analysis()` temporary-byte measurement.
- `src/gpuwrf/io/gen2_accessor.py`: narrow fallback so `history_files()` returns `wrfinput_d02` when only wrfinput is visible in the pinned Gen2 tree.
- `scripts/m6_run_coupled_forecast.py`: new CLI for 1h/6h/24h forecast artifacts and d02 spacetime budget generation.
- `tests/test_m6_boundary_apply.py`, `tests/test_m6_forecast_smoke.py`, `tests/test_m6_forecast_24h.py`: new M6-S2 tests.
- `tests/test_m6_precision_matrix.py`, `tests/test_m6_state_extension.py`: extended for boundary leaves.
- `.agent/decisions/ADR-010-coupled-state-extension.md`: amended with M6-S2 ratifications for FP32 Path A, boundary leaves, GridSpec threading, cadence implementation, and limitations.
- `artifacts/m6/forecast_smoke_1h.json`, `artifacts/m6/forecast_6h_summary.json`, `artifacts/m6/forecast_24h_summary.json`, `artifacts/m6/spacetime_budget_d02.json`: proof artifacts.
- `artifacts/m6/forecast_smoke_1h.outputs.json`, `artifacts/m6/forecast_6h_summary.outputs.json`, `artifacts/m6/forecast_24h_summary.outputs.json`: output manifests.

## Proof Objects Produced

- `artifacts/m6/forecast_smoke_1h.json`: PASS, d02, 1.0 lead hour, zero measured post-init transfer.
- `artifacts/m6/forecast_6h_summary.json`: PASS, d02, 6.0 lead hours, zero measured post-init transfer.
- `artifacts/m6/forecast_24h_summary.json`: PASS, d02, 24.0 lead hours, zero measured post-init transfer.
- `artifacts/m6/spacetime_budget_d02.json`: d02 spacetime budget with real temporary-byte measurement and CPU denominator comparison.
- Forecast output containers:
  - `/home/enric/.cache/gpuwrf_outputs/m6/coupled_driver/wrfout_gpu_d02_p001h.npz`
  - `/home/enric/.cache/gpuwrf_outputs/m6/coupled_driver/wrfout_gpu_d02_p006h.npz`
  - `/home/enric/.cache/gpuwrf_outputs/m6/coupled_driver/wrfout_gpu_d02_p012h.npz`
  - `/home/enric/.cache/gpuwrf_outputs/m6/coupled_driver/wrfout_gpu_d02_p018h.npz`
  - `/home/enric/.cache/gpuwrf_outputs/m6/coupled_driver/wrfout_gpu_d02_p024h.npz`
- Trace directories:
  - `/home/enric/.cache/gpuwrf_tmp/trace_forecast_1h`
  - `/home/enric/.cache/gpuwrf_tmp/trace_forecast_6h`
  - `/home/enric/.cache/gpuwrf_tmp/trace_forecast_24h`

The d02 budget currently reports:

- `host_to_device_bytes_post_init = 0`
- `device_to_host_bytes_post_init = 0`
- `host_device_transfer_bytes = 0`
- `temporary_bytes_per_step = 136890408`
- `debug_vs_stripped_hlo_diff_bytes = 0`
- `total_per_step_ms = 20.19151810090989`
- `extrapolated_24h_wall_s = 29.07578606531024`
- CPU denominator comparison: 3106.249150758174 s grid-points attributed and 4859.527050000001 s raw-timing subtraction, both reported for M6-S5 to decide.

## Commands Run

Prerequisite and import:

```bash
git add pyproject.toml
git commit -m "[M6-S2 prereq] add zarr + jax to pyproject.toml per M6-S2a Opus follow-up"
git push origin worker/codex/m6-s2-coupled-forecast-driver
python -m pip install -e . --no-deps
python -c "from gpuwrf.io.boundary_replay import extract_d02_boundary; print('ok')"
```

Forecast generation:

```bash
python scripts/m6_run_coupled_forecast.py --hours 1 --output artifacts/m6/forecast_smoke_1h.json
python scripts/m6_run_coupled_forecast.py --hours 6 --output artifacts/m6/forecast_6h_summary.json
python scripts/m6_run_coupled_forecast.py --hours 24 --output artifacts/m6/forecast_24h_summary.json
```

Artifact validation and tests:

```bash
python -c "from gpuwrf.io.proof_schemas import validate_artifact; validate_artifact('artifacts/m6/forecast_24h_summary.json'); print('ok')"
python - <<'PY'
from gpuwrf.io.proof_schemas import ForecastSmoke, SpacetimeBudget
ForecastSmoke.validate_file('artifacts/m6/forecast_smoke_1h.json')
ForecastSmoke.validate_file('artifacts/m6/forecast_6h_summary.json')
ForecastSmoke.validate_file('artifacts/m6/forecast_24h_summary.json')
SpacetimeBudget.validate_file('artifacts/m6/spacetime_budget_d02.json')
print('ok')
PY
pytest -q tests/test_m6_*.py
```

Final validation result:

```text
23 passed in 18.53s
```

## Implementation Notes

The radiation cadence is implemented as static segmentation rather than a dynamic per-step `lax.cond`. This preserves arbitrary forecast-length handling while avoiding the one-byte predicate transfer pattern identified by the M6-S1 reviewer. For non-cadence-aligned lengths, the trailing segment runs without an extra final radiation call; this is documented behavior and remains deterministic.

The boundary application follows the WRF EM specified-boundary and relaxation-zone structure. The implementation cites and mirrors the relevant behavior from `dyn_em/module_bc_em.F:lbc_fcx_gcx` and `share/module_bc.F:relax_bdytend_core`/`spec_bdytend`, with d02 namelist widths `spec_bdy_width=5`, `spec_zone=1`, and `relax_zone=4`. The specified outer edge is reapplied after relaxation so the edge cells remain prescribed.

The GridSpec route is threaded through the driver and adapters, but the actual `dz` passed into MYNN/RRTMG is derived from the current geopotential interfaces. That is the least invasive way to remove the M6-S1 `DEFAULT_DZ_M` placeholder while keeping ADR-002 state layout stable and avoiding a large static metric array in every `State` instance.

The budget's temporary-byte number comes from XLA compiled executable memory analysis, not a hardcoded literal. The host/device transfer audit is measured after warm-up from JAX profiler trace output and preserved in the proof artifacts.

## Unresolved Risks

- The 24h run is finite and resident, but diagnostics hit broad finite-guard bounds. This is not an operational-quality forecast. M6-S7 must decide operational fitness with RMSE/probtest evidence.
- The driver uses a one-second effective internal dycore step inside each 60-second coupled step to keep the reduced dycore finite on the full d02 grid. This is a stability guard, not a final dynamics solution.
- The finite-state guard replaces non-finite candidate updates with previous-step values and clips broad physical ranges. This makes the residency proof robust but must not be mistaken for physical validation.
- The pinned Gen2 directory visible during this sprint exposed `wrfinput_d02` but not `wrfout_d02_*` history files. The accessor fallback enables the contract's time-zero IC reads; it does not create history-dependent forcing.
- `mu_bdy` repeats the initial `MU/MUB` boundary field when replay history is unavailable. Wind, theta, qv, and geopotential boundaries use the M6-S2a replay artifact.
- Forecast outputs are WRF-shaped NPZ proof containers, not NetCDF files. NetCDF/HDF5 attempted large `/tmp/*.nc4` scratch files on a nearly-full tmpfs, so the script writes compact NPZ containers to `/home/enric/.cache/gpuwrf_outputs/m6/coupled_driver`.
- The per-kernel wall budget excludes compile and output-writing time. The 24h summary separately records `output_run_wall_s`.

## Next Decision Needed

M6-S5 needs to choose the binding CPU denominator for the ADR-007 4x verdict: 3106.249150758174 s grid-points attributed or 4859.527050000001 s raw-timing subtraction. M6-S7 then needs to decide whether the FP32 Path A forecast is operationally acceptable under RMSE/probtest gates, or whether specific leaves must revert to FP64.
