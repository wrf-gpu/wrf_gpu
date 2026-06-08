# 2026-06-08 GPT v0.13 Implementation Review

## Objective

Independent GPT-5.5 xhigh review/debug pass over v0.13 "Validate & Accelerate" implementations: WSM7, MRF, GFS/old-MM5 surface layer, slab LSM, GSFC SW/LW, radiation `rad_rk_tendf`, moisture advection wiring, RRTMG chunking, two-way feedback dedup, compile knobs, and scheme registry/catalog/dispatch consistency.

## Bugs Fixed

### FIXED MAJOR: `ra_sw_physics=0` / `ra_lw_physics=0` were catalog-accepted but operationally rejected or fell through to RRTMG

- Files: `src/gpuwrf/runtime/operational_mode.py:2424`, `src/gpuwrf/runtime/operational_mode.py:2884`, `src/gpuwrf/runtime/operational_mode.py:2917`, `src/gpuwrf/runtime/operational_mode.py:2378`, `src/gpuwrf/runtime/operational_mode.py:2694`, `src/gpuwrf/runtime/operational_mode.py:3276`
- Symptom: the public registry/catalog accepted disabled radiation modes (`ra_sw_physics=0`, `ra_lw_physics=0`), but `_SCAN_WIRED_OPTIONS` omitted `0`. If admitted, the old tendency dispatch would fall through to RRTMG because only `1`/`2` were special-cased for SW and only `1` was special-cased for LW.
- WRF reference: WRF runs SW and LW drivers independently; disabled components must contribute no heating/flux, not default RRTMG.
- Fix: add `0` to the operational radiation allowlist; return exact zero RTHRATEN for disabled SW/LW components; zero disabled components in held Noah-MP/Noah-classic radiation seeds and M9 radiation diagnostics.
- Tests: `tests/test_rrtm_lw_operational_wiring.py:207`, `tests/test_rrtm_lw_operational_wiring.py:213`, `tests/test_rrtm_lw_operational_wiring.py:220`, `tests/test_rrtm_lw_operational_wiring.py:230`, `tests/test_cdudhia_sw_operational_wiring.py:198`.

### FIXED MAJOR: slab LSM reference-only entry overclaimed `gpu_gate_ready=True`

- Files: `src/gpuwrf/coupling/physics_dispatch.py:263`, `src/gpuwrf/coupling/physics_dispatch.py:266`, `src/gpuwrf/coupling/physics_dispatch.py:349`
- Symptom: `resolve_physics_suite({"sf_surface_physics": 1})` returned `land_surface.gpu_runnable=True` and `gpu_gate_ready=True` even though the catalog and operational resolver correctly describe slab LSM as reference-only until a TSLB/radiation/static land hook exists.
- WRF reference: WRF slab LSM owns 5-layer soil temperature state and surface radiation/static forcing. The current operational `State` does not carry that contract.
- Fix: mark the slab `SchemeEntry` as `gpu_runnable=False`, preserving reference metadata but preventing the dispatcher gate from overclaiming operational readiness.
- Test: `tests/test_v060_physics_dispatch.py:122`.

## Substantial Issues For Manager

### MAJOR: selected GFS/old-MM5 surface-layer diagnostics do not feed PBL adapters that need full surface diagnostics

- Files: `src/gpuwrf/coupling/scan_adapters.py:418`, `src/gpuwrf/coupling/scan_adapters.py:443`, `src/gpuwrf/coupling/scan_adapters.py:919`, `src/gpuwrf/coupling/scan_adapters.py:929`, `src/gpuwrf/coupling/scan_adapters.py:1074`, `src/gpuwrf/coupling/physics_dispatch.py:422`
- WRF reference: `/home/enric/src/wrf_pristine/WRF/dyn_em/module_first_rk_step_part1.F:594` calls `surface_driver`, which writes `HFX/QFX/BR/PSIM/PSIH`; `/home/enric/src/wrf_pristine/WRF/dyn_em/module_first_rk_step_part1.F:1113` then passes those same fields into `pbl_driver`.
- Symptom: v0.13 GFS/old-MM5 surface adapters write only B2 kinematic flux handles. `_pbl_surface_forcing()` ignores the selected surface-layer adapter and re-derives revised-MM5 diagnostics via `surface_layer_with_diagnostics(...)`. `mrf_pbl_adapter()` explicitly consumes that revised-MM5 forcing. The resolver accepts `bl_pbl_physics=99` with `sf_sfclay_physics=3` or `91` and reports `gpu_gate_ready=True`, even though MRF receives revised-MM5 `BR/PSIM/PSIH/U10/V10` rather than GFS/old-MM5 diagnostics.
- Minimal repro:
  `JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-23 python - <<'PY'\nfrom gpuwrf.coupling.physics_dispatch import resolve_physics_suite\nfor sf in (3, 91):\n    s = resolve_physics_suite({'bl_pbl_physics': 99, 'sf_sfclay_physics': sf})\n    print(s.pbl.option, s.surface_layer.option, s.gpu_gate_ready)\nPY`
  prints `99 3 True` and `99 91 True`.
- Recommended fix: add an operational surface-diagnostics carry/contract (`HFX`, `QFX`, `BR`, `PSIM`, `PSIH`, `U10`, `V10`, `ZNT`, `PBLH` as needed) produced by the selected surface-layer adapter and consumed by PBL adapters, or fail-close unvalidated PBL/surface pairs until that carry exists. Do not silently keep re-deriving revised-MM5 for non-revised-MM5 selections.

### MAJOR: WSM7 is described as reference-only in sprint scope/tests but catalog behavior is fail-closed at namelist validation

- Files: `src/gpuwrf/contracts/physics_registry.py:60`, `src/gpuwrf/contracts/physics_registry.py:180`, `src/gpuwrf/contracts/physics_registry.py:191`, `src/gpuwrf/io/scheme_catalog.py:167`, `src/gpuwrf/io/scheme_catalog.py:297`, `src/gpuwrf/io/scheme_catalog.py:949`, `tests/test_wsm7_savepoint_parity.py:173`, `tests/test_wsm7_savepoint_parity.py:184`
- WRF reference: `/home/enric/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON:469` defines `qh`; `/home/enric/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON:3038` defines `wsm7scheme mp_physics==24` as `moist:qv,qc,qr,qi,qs,qg,qh`; `/home/enric/src/wrf_pristine/WRF/phys/module_mp_wsm7.F:75` takes `qh`.
- Symptom: the WSM7 kernel and savepoint test correctly show a `qh` state replacement and `hail_acc`, but `ACCEPTED_MP_PHYSICS` omits `24`, `MOIST_SPECIES` omits `qh`, and `classify_scheme("mp_physics", 24)` returns `RECOGNIZED_FAIL_CLOSED`, not `REFERENCE_ONLY`. `validate_namelist({"physics": {"mp_physics": [24]}})` raises immediately. This is safe fail-closed behavior, but it contradicts the v0.13 scope wording "reference-only pending qh State leaf" and prevents even a validator-accepted reference-only selection.
- Minimal repro:
  `JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-23 python - <<'PY'\nfrom gpuwrf.io.scheme_catalog import classify_scheme\nfrom gpuwrf.io.namelist_check import validate_namelist\nprint(classify_scheme('mp_physics', 24).status.value)\ntry:\n    validate_namelist({'physics': {'mp_physics': [24]}})\nexcept Exception as e:\n    print(type(e).__name__)\nPY`
  prints `recognized_fail_closed` and `UnsupportedSchemeError`.
- Recommended fix: choose one contract. If WSM7 should be reference-only, add `mp_physics=24` to the accepted/reference-only catalog without making it operational, and extend registry tests so reference-only does not imply State/dycore moisture advectability. If it should remain fail-closed until `qh` exists everywhere, update v0.13 scope/docs to avoid calling it reference-only.

## Oracle Integrity And Slop-Pattern Findings

- Checked oracle provenance for WSM7, MRF, GFS/old-MM5/slab surface batch, GSFC SW, and GSFC LW. The build scripts copy from `/home/enric/src/wrf_pristine/WRF`; recorded source hashes match the pristine modules checked locally for `module_mp_wsm7.F`, `module_bl_mrf.F`, `module_sf_gfs.F`, `module_sf_sfclay.F`, `module_sf_slab.F`, `module_ra_gsfcsw.F`, and `module_ra_goddard.F`. GSFC-LW applies a visibility-only shim after recording the pristine checksum.
- No JAX-vs-JAX self-compare or happy-path oracle was found in the checked v0.13 proof artifacts. RRTMG chunking proofs are correctly framed as bit-inertness/VRAM proofs against alternate JAX chunk configurations, not WRF-fidelity oracles.
- No new masking-clamp slop was found in the fixed paths. Existing radiation/PBL clamps reviewed here appear to be WRF-style physical bounds or existing safety guards, not v0.13 proof substitutes.

## Commands Run

- `JAX_PLATFORMS=cpu PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 taskset -c 0-23 python -m pytest tests/test_v060_physics_dispatch.py -q` -> `14 passed`
- `JAX_PLATFORMS=cpu PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 taskset -c 0-23 python -m pytest tests/test_rrtm_lw_operational_wiring.py tests/test_cdudhia_sw_operational_wiring.py -q` -> `18 passed`
- `JAX_PLATFORMS=cpu PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 taskset -c 0-23 python -m pytest tests/test_v060_physics_dispatch.py tests/test_scheme_catalog_fail_closed.py tests/test_namelist_check.py tests/test_wsm7_savepoint_parity.py tests/test_v013_t3_surface_lsm_wiring.py tests/test_v013_mrf_operational.py tests/test_rrtm_lw_operational_wiring.py tests/test_cdudhia_sw_operational_wiring.py tests/dynamics/test_moisture_advection_operational.py tests/test_v013_compile_perf2.py -q` -> `150 passed`

## Files Changed

- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/physics_dispatch.py`
- `tests/test_rrtm_lw_operational_wiring.py`
- `tests/test_cdudhia_sw_operational_wiring.py`
- `tests/test_v060_physics_dispatch.py`
- `.agent/reviews/2026-06-08-gpt-v013-impl-review.md`

## Proof Objects Produced

- This review document.
- CPU pytest proof: targeted dispatch/radiation tests and selected v0.13 registry/wiring suite, all passing as listed above.

## Unresolved Risks

- The PBL/surface-diagnostics mismatch can make non-default surface-layer selections operationally misleading for PBLs that require full surface diagnostics.
- WSM7 cannot become operational without adding `qh` and hail accumulators across State, dycore advection, lateral boundaries, restart/I/O, registry, and wrfout mapping.
- I did not run GPU profiler/VRAM jobs per instruction; GPU performance claims remain tied to existing v0.13 proof JSONs.

## Next Decision Needed

Decide whether to fix the PBL/surface issue by (a) adding a selected-sfclay diagnostics carry, or (b) fail-closing unvalidated PBL/surface pairings until that carry exists. Also decide whether WSM7 should be validator-accepted `REFERENCE_ONLY` or stay `RECOGNIZED_FAIL_CLOSED` until `qh` State support lands.
