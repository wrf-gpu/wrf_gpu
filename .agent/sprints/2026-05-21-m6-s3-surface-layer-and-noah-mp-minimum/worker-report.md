# M6-S3 Worker Report - Surface Layer + Bounded Noah-MP Minimum

Worker: Codex gpt-5.5
Branch: `worker/codex/m6-s3-surface-layer-and-noah-mp-minimum`
Date: 2026-05-21
Worktree: `/tmp/wrf_gpu2_m6s3`

## Objective

Add the smallest honest surface layer and Noah-MP minimum that makes `U10/V10/T2/Q2` operationally meaningful for M6: MM5 sfclay Monin-Obukhov surface-layer diagnostics and flux handles, with prescribed Gen2 land state rather than a prognostic Noah-MP port.

## Summary

Implemented a vectorized JAX MM5 sfclay-style surface-layer kernel in `src/gpuwrf/physics/surface_layer.py`, fed by new prescribed Noah-MP/land-state helpers in `src/gpuwrf/physics/noah_mp.py` and `src/gpuwrf/io/land_state.py`. The coupled `surface_adapter` now calls the real kernel and continues to cast `ustar/theta_flux/qv_flux/tau_u/tau_v/rhosfc/fltv` through the FP64 precision registry.

Also extended WRF-like NPZ forecast output with `U10`, `V10`, `T2`, `Q2`, `UST`, `HFX_KIN`, and `QFX_KIN`, so the M6-S2 driver now emits surface diagnostics from the real surface-layer path.

Important limitation: the pinned local Gen2 run has no `wrfout_d02_*` files. It only has `wrfinput_d02` for d02. Therefore prescribed radiation tendencies are unavailable and full before/after operational RMSE deltas at 1h/6h/12h/24h are blocked. I recorded this honestly in the artifacts instead of manufacturing a full-driver RMSE claim.

## AC Evidence

AC1 - Surface-layer scope memo: PASS.
Created `.agent/decisions/ADR-012-m6-surface-layer-scope.md` before model-code edits. It records Option A prescribed land, included MM5 sfclay features, excluded prognostic Noah-MP features, Gen2 data source, direct `ZNT` absence, and WRF source line mappings.

AC2 - Radiation-conditioning feasibility: BLOCKED with proof.
Created `artifacts/m6/radiation_conditioning_feasibility.json`. The artifact reports `history_file_count=0` for real `wrfout_d02_*` and `RTHRATEN/RTHRATSW/RTHRATLW` unavailable. Decision field: `M6-S3 deviation: RRTMG online remains required`.

AC3 - Real Monin-Obukhov surface-layer kernel: PASS.
Created `src/gpuwrf/physics/surface_constants.py` and `src/gpuwrf/physics/surface_layer.py`. The kernel implements WRF sfclay input preparation, saturation mixing ratio, bulk Richardson number, regime-dependent stability functions, friction velocity, 10m/2m diagnostics, heat/moisture fluxes, momentum stress, density, and virtual heat flux. It returns the M5-S2.x `SurfaceFluxes` contract in FP64.

AC4 - Noah-MP subset: PASS for Option A prescribed.
Created `src/gpuwrf/physics/noah_mp.py`. It packages `TSK/SMOIS/SH2O/TSLB/IVGTYP/ISLTYP/LU_INDEX/XLAND/LANDMASK/LAKEMASK/SST` into a bounded prescribed state. It does not call or claim `NOAHMP_SFLX` parity.

AC5 - Static land/SST/geog provenance: PASS with caveat.
Created `src/gpuwrf/io/land_state.py` and `artifacts/m6/land_state_manifest.json`. Manifest includes SHA-256 for `wrfinput_d02`, variable inventory, summaries, and roughness derivation. `ZNT` is absent locally; `CM/CH` are present but zero, so roughness uses a documented VEGFRA/land-water surrogate.

AC6 - Coupled into M6-S2 driver: PASS.
Updated `src/gpuwrf/coupling/physics_couplers.py.surface_adapter` to call `gpuwrf.physics.surface_layer.surface_layer`. Updated `src/gpuwrf/coupling/driver.py` output path to include surface diagnostics. A 1h coupled smoke forecast passed and wrote `U10/V10/T2/Q2/UST`.

AC7 - Tier-1 vs WRF harness oracle: PASS for harness linkage and focused test.
Created `scripts/wrf_sfclay_harness.f90` and `scripts/wrf_sfclay_harness_build.sh`. Build artifact: `artifacts/m6/wrf_sfclay_harness_build.txt`. `nm` artifact: `artifacts/m6/wrf_sfclay_harness_nm.txt`, containing `module_sf_sfclay_sfclay1d`. Focused pytest compares JAX outputs against the WRF-linked harness at strict tolerance and passes.

AC8 - Operational delta artifact: PARTIAL/BLOCKED for full lead RMSE.
Created `artifacts/m6/surface_operational_delta.json`. It reports lead-0 deltas against `wrfinput_d02` truth:
- `U10`: RMSE improved by `-0.026082609377303945 m s-1`.
- `V10`: RMSE improved by `-0.10684549525484888 m s-1`.
- `T2`: RMSE improved by `-0.09437492135558345 K`.
- `Q2`: RMSE degraded by `+0.00040216490500692736 kg kg-1`.
Full 1h/6h/12h/24h before/after RMSE is blocked because no `wrfout_d02_*` truth exists in the pinned local Gen2 tree.

AC9 - Honest accounting/schema: PASS.
Added `SurfaceLayerArtifact` to `src/gpuwrf/io/proof_schemas.py` and registry aliases for `radiation_conditioning_feasibility`, `surface_operational_delta`, and `land_state_manifest`. `scripts/m6_gate_surface_layer.py` validates all three.

AC10 - ADR-012 + ADR-013: PASS.
Created `.agent/decisions/ADR-012-m6-surface-layer-scope.md` and `.agent/decisions/ADR-013-m6-noah-mp-subset.md`, with WRF source citations.

## F-S3 Prerequisites

F-S3-1 - sanitize_state ON/OFF:
PARTIAL. I produced lead-0 diagnostic deltas where `sanitize_state` is not involved and ran a 1h coupled smoke with sanitize ON through the new surface adapter. I did not produce a sanitize OFF 1h/6h/12h forecast series in this turn. The artifact records this explicitly as partial. The 1h sanitize-ON run passed with finite state and 0 post-init transfers.

F-S3-2 - real d02 mu_bdy history or interior-only waiver:
WAIVED TO INTERIOR-ONLY. The local pinned Gen2 path has no `wrfout_d02_*` history. `Gen2Run.history_files("d02")` falls back to `wrfinput_d02`; this cannot surface a real `mu_bdy` time series. I documented that M6-S8 must restrict operational comparison to interior diagnostics unless a Gen2 run with real d02 history is supplied.

F-S3-3 - FP64 surface adapter boundary:
PASS. The adapter still stores `ustar/theta_flux/qv_flux/tau_u/tau_v/rhosfc/fltv` via `_field_dtype(...)`, and the precision registry keeps all seven as FP64. `tests/test_m6_state_extension.py` remains passing.

## Commands Run

- `pytest -q tests/test_m6_surface_layer_kernel.py tests/test_m6_noah_mp_prescribed.py tests/test_m6_proof_schemas.py` -> `8 passed in 3.35s`
- `./scripts/wrf_sfclay_harness_build.sh` -> built `/tmp/wrf_gpu2_m6s3/data/scratch/wrf_sfclay_harness`, SHA `b9160dc06c764050cac44ce04f9e16ea81021d017df23bedab3d57466926c0cc`
- `python scripts/m6_run_surface_layer.py` -> wrote radiation, land-state, and operational-delta artifacts
- `python scripts/m6_gate_surface_layer.py` -> `PASS`, with radiation `BLOCKED`, land state `PASS`, operational delta `PARTIAL`
- `python .agent/skills/writing-gpu-kernels/scripts/static_kernel_check.py src/gpuwrf/physics/surface_layer.py src/gpuwrf/physics/noah_mp.py src/gpuwrf/coupling/physics_couplers.py` -> `ok: true`
- `PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true pytest -q tests/test_m6_*.py` -> `28 passed in 32.06s`
- `python scripts/m6_run_coupled_forecast.py --hours 1 --output artifacts/m6/forecast_smoke_1h_surface.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/surface_layer --audit-steps 1` -> `PASS`, 0 H2D, 0 D2H, output contains `U10/V10/T2/Q2/UST`

## Proof Objects Produced

- `artifacts/m6/radiation_conditioning_feasibility.json`
- `artifacts/m6/land_state_manifest.json`
- `artifacts/m6/surface_operational_delta.json`
- `artifacts/m6/wrf_sfclay_harness_build.txt`
- `artifacts/m6/wrf_sfclay_harness_nm.txt`
- `artifacts/m6/wrf_sfclay_harness_sha256.txt`
- `artifacts/m6/forecast_smoke_1h_surface.json`
- `artifacts/m6/forecast_smoke_1h_surface.outputs.json`
- `/home/enric/.cache/gpuwrf_outputs/m6/surface_layer/wrfout_gpu_d02_p001h.npz`

## Files Changed

- `.agent/decisions/ADR-012-m6-surface-layer-scope.md`
- `.agent/decisions/ADR-013-m6-noah-mp-subset.md`
- `src/gpuwrf/physics/surface_constants.py`
- `src/gpuwrf/physics/surface_layer.py`
- `src/gpuwrf/physics/noah_mp.py`
- `src/gpuwrf/io/land_state.py`
- `src/gpuwrf/io/proof_schemas.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/coupling/driver.py`
- `scripts/wrf_sfclay_harness.f90`
- `scripts/wrf_sfclay_harness_build.sh`
- `scripts/m6_run_surface_layer.py`
- `scripts/m6_gate_surface_layer.py`
- `tests/test_m6_surface_layer_kernel.py`
- `tests/test_m6_noah_mp_prescribed.py`
- `tests/test_m6_proof_schemas.py`
- M6 artifacts listed above

## Unresolved Risks

- No `wrfout_d02_*` history exists in the local pinned Gen2 run, so d02 radiation tendencies, real time-varying d02 land state, and full operational lead RMSE remain blocked.
- Direct `ZNT` is absent. Roughness is a documented surrogate from available prescribed fields, not a WRF-history roughness replay.
- F-S3-1 sanitize OFF is not fully closed. The new surface path is finite in the 1h sanitize-ON smoke and lead-0 diagnostics avoid sanitize, but the requested ON/OFF forecast attribution remains a follow-up.
- The M5/MYNN internal `mynn_surface_stub.surface_layer` remains a separate M5 hook because MYNN column state does not carry prescribed land fields. The M6 coupled surface adapter now uses the real kernel and emits diagnostics, but MYNN-internal bottom-boundary flux use still needs a later state-threading decision.

## Next Decision Needed

Manager/reviewer should decide whether to accept M6-S3 as a partial-but-useful surface-layer fold-in under the local-data limitations, or require a new Gen2 fixture with real `wrfout_d02_*` before closing AC2/AC8/F-S3-1 fully.
