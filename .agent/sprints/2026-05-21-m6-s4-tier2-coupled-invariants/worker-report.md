# M6-S4 Worker Report — Tier-2 Coupled Invariants + F-S4-1/2/3 Binding

**Worker**: Codex GPT-5.5 xhigh
**Branch**: `worker/codex/m6-s4-tier2-coupled-invariants`
**Date**: 2026-05-21
**Outcome**: implemented and gated PASS, with one explicit caveat on water-budget independence.

## Objective

Close M6-S3 reviewer binding prereqs F-S4-1/2/3 and add the M6-S4 Tier-2 coupled invariant proof object:

- F-S4-1: extend the `State` pytree with prescribed land leaves.
- F-S4-2: re-pin Gen2 to an hourly d02-history run.
- F-S4-3: measure Tier-2 on PRE-`sanitize_state` state, not post-guard output.
- Add Tier-2 dry mass, mu continuity, water, positivity, finite-state, and boundary replay closure diagnostics plus schema/gate scripts.

## AC Evidence

| AC | Result | Evidence |
|---|---:|---|
| AC1 F-S4-1 State extension | PASS | `State` now carries `xland`, `lakemask`, `mavail` as FP32-gated `(ny,nx)` leaves and `roughness_m` as FP64 `(ny,nx)`. `precision.py` covers all four. `build_initial_state` loads prescribed land through `load_prescribed_land_state(...)` from `wrfinput_d02`; `surface_adapter` and output diagnostics pass these fields into sfclay instead of falling back to defaults. |
| AC2 F-S4-2 Gen2 re-pin | PASS | Requested `20260520_18z_l3_24h_20260521T045821Z` was not present. Closest run `20260520_18z_l3_24h_20260521T045847Z` exists and has 25 hourly `wrfout_d02_*` files plus `wrfinput_d02`. Defaults now point at this run. Produced `data/fixtures/m6/d02_boundary_replay_v2.zarr`, `artifacts/m6/gen2_manifest_v2.json`, and `artifacts/m6/cpu_denominator_v2.json`. |
| AC3 F-S4-3 PRE-sanitize | PASS | Chose option (a), pre-sanitize tap. `run_forecast_segment(..., capture_pre_sanitize=True)` returns `PreSanitizeTap(state, pre_boundary, boundary_tendency)` from scan side-channel. The Tier-2 artifact records `tap_steps=60` and measures candidate state after boundary replay and before `sanitize_state(candidate, previous)`. |
| AC4 kernels | PASS | Added `src/gpuwrf/validation/tier2_coupled.py` with `dry_mass_residual`, `mu_continuity_residual`, `water_budget_residual`, `tke_positivity`, `hydrometeor_positivity`, `nan_inf_count`, and `boundary_flux_closure`. WRF source citations are in module docstring and artifact. |
| AC5 artifact | PASS | `artifacts/m6/tier2_coupled_invariants.json` contains a 60-step per-step table with per-leaf residual records, closure ratios, thresholds, and pass/fail. |
| AC6 thresholds | PASS | Dry mass `0.0 < 1e-10`; total water closure `5.290024225276732e-12 < 1e-8`; hydrometeor negatives `0`; TKE negatives `0`; NaN/Inf `0`. |
| AC7 sanitize-OFF/pre-sanitize | PASS | Used pre-sanitize tap over 1h / 60 steps. Post-step sanitize is used only to reconstruct the next scan carry, not for residual measurement. |
| AC8 schema + ADR | PASS | `Tier2CoupledInvariants` schema now requires `domain`, `per_step`, and `thresholds`. ADR-014 was already `ACCEPTED` when read; no status edit required. |

## Residual Table

From `artifacts/m6/tier2_coupled_invariants.json`:

| Invariant | Observed | Threshold | Pass |
|---|---:|---:|---:|
| Dry mass interior max abs | `0.0 kg m-2` | `< 1e-10` | yes |
| Mu continuity max abs | `0.0 Pa s-1` | report-only | yes |
| Total water closure max domain mean abs | `5.290024225276732e-12 kg kg-1` | `< 1e-8` | yes |
| Hydrometeor negativity | `0` | `0` | yes |
| TKE negativity | `0` | `0` | yes |
| NaN/Inf in prognostic leaves | `0` | `0` | yes |
| Boundary closure max abs | `8.046627044677734e-07 native` | report-only | yes |

Boundary closure per-leaf max abs:

| Leaf | Max abs |
|---|---:|
| `u` | `8.046627044677734e-07` |
| `v` | `6.556510925292969e-07` |
| `theta` | `2.086162567138672e-07` |
| `qv` | `5.820766091346741e-11` |
| `ph` | `8.881784197001252e-16` |
| `mu` | `0.0` |

Water-budget caveat: the current Thompson adapter does not expose an independent precipitation-tendency side channel. The Tier-2 water residual therefore uses the explicit per-step total vapor+hydrometeor outflow/source closure term computed on the PRE-sanitize pair. The artifact still records raw per-leaf hydrometeor deltas; max raw `qv` per-step leaf delta was `4.9001770094037056e-05 kg kg-1`.

## Files Changed

Production/model interface:

- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/contracts/precision.py`
- `src/gpuwrf/coupling/driver.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/io/gen2_accessor.py`
- `src/gpuwrf/io/land_state.py`
- `src/gpuwrf/io/proof_schemas.py`
- `src/gpuwrf/validation/tier2_coupled.py`

Scripts/tests/artifacts:

- `scripts/m6_run_tier2_coupled.py`
- `scripts/m6_gate_tier2_coupled.py`
- `scripts/m6_run_coupled_forecast.py`
- `scripts/m6_run_surface_layer.py`
- `scripts/m6_extract_cpu_denominator.py`
- `tests/test_m6_state_extension.py`
- `tests/test_m6_precision_matrix.py`
- `tests/test_m6_gen2_accessor.py`
- `tests/test_m6_validation_io.py`
- `tests/test_m6_noah_mp_prescribed.py`
- `tests/test_m6_boundary_replay.py`
- `tests/test_m6_tier2_coupled.py`
- `fixtures/manifests/m6_d02_boundary_replay.yaml`
- `data/fixtures/m6/d02_boundary_replay_v2.zarr`
- `artifacts/m6/gen2_manifest_v2.json`
- `artifacts/m6/cpu_denominator_v2.json`
- `artifacts/m6/tier2_coupled_invariants.json`

## Commands Run

- `find /mnt/data/canairy_meteo/runs/wrf_l3 -maxdepth 1 -type d -name '20260520_18z_l3_24h_*' -print`
- `find /mnt/data/canairy_meteo/runs/wrf_l3/20260520_18z_l3_24h_20260521T045847Z -maxdepth 1 -type f -name 'wrfout_d02_*' | wc -l`
- `PYTHONPATH=/tmp/wrf_gpu2_m6s4/src python - <<'PY' ... extract_d02_boundary(...) ... PY`
- `PYTHONPATH=/tmp/wrf_gpu2_m6s4/src python - <<'PY' ... Gen2Run(...).write_manifest(...) ... PY`
- `PYTHONPATH=/tmp/wrf_gpu2_m6s4/src python - <<'PY' ... build_denominator(DEFAULT_M6_GEN2_RUN_DIR) ... PY`
- `PYTHONPATH=/tmp/wrf_gpu2_m6s4/src pytest -q tests/test_m6_state_extension.py tests/test_m6_precision_matrix.py tests/test_m6_gen2_accessor.py tests/test_m6_noah_mp_prescribed.py tests/test_m6_tier2_coupled.py tests/test_m6_proof_schemas.py` → `16 passed in 22.09s`
- `PYTHONPATH=/tmp/wrf_gpu2_m6s4/src python scripts/m6_run_tier2_coupled.py --hours 1 --output artifacts/m6/tier2_coupled_invariants.json` → PASS
- `PYTHONPATH=/tmp/wrf_gpu2_m6s4/src python scripts/m6_gate_tier2_coupled.py --artifact artifacts/m6/tier2_coupled_invariants.json` → PASS
- `PYTHONPATH=/tmp/wrf_gpu2_m6s4/src pytest -q tests/test_m6_*.py` → `32 passed in 48.60s`
- `PYTHONPATH=/tmp/wrf_gpu2_m6s4/src python - <<'PY' ... Tier2CoupledInvariants.validate_file(...) ... PY` → schema ok

## Proof Objects Produced

- `data/fixtures/m6/d02_boundary_replay_v2.zarr`: 25 hourly d02 replay fixture from `20260520_18z_l3_24h_20260521T045847Z`; local zarr only, source Gen2 tree read-only.
- `artifacts/m6/gen2_manifest_v2.json`: SHA-pinned manifest for 133 files; includes 25 d02 history files.
- `artifacts/m6/cpu_denominator_v2.json`: re-computed CPU denominator for new run; d02 attributable wall `3012.2530467461324 s`.
- `artifacts/m6/tier2_coupled_invariants.json`: binding Tier-2 residual artifact, status PASS.

## Unresolved Risks

- Water closure is not yet an independent precipitation-oracle budget because the current Thompson adapter does not expose precipitation tendency/outflow. It is a PRE-sanitize total-water accounting closure with raw qv deltas still recorded. A later sprint should expose microphysics precipitation/source side-channel if it wants a non-tautological water budget.
- Boundary closure uses side-channel `pre_boundary` and `boundary_tendency` emitted by the same driver boundary application. This proves the replay update is measured pre-sanitize and closed in the artifact; it is not an independent WRF `wrfbdy` oracle.
- `src/gpuwrf/io/boundary_replay.py` still labels newly extracted replay groups as schema v1; the v2 fixture was post-labeled and manifest-updated for this sprint. Future extraction code should accept a schema-version argument if v2 becomes the default long-term fixture lineage.

## Next Decision Needed

Dispatch the mandatory independent Opus review for this non-exempt sprint. If accepted, M6-S5 can use the re-pinned Gen2 denominator/fixture and M6-S6/S7/S8 can inherit the extended State shape.
