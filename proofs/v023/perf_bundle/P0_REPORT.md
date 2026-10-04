# P0 GPT Report: M9 Radiation CPU-Proven Reduction

## Objective

Execute `P0_GPT_BRIEF.md` CPU-only: prove a bit-identical M9 radiation reduction with HLO and CPU field compare, without using GPU lane 0:2.

## Implementation

- Added M9-only RRTMG flux-slice solvers:
  - `solve_rrtmg_sw_m9_flux_slices`
  - `solve_rrtmg_lw_m9_flux_slices`
- Routed `compute_m9_diagnostics` through the internal `_m9_flux_slices_only=True` path.
- Added `compute_m9_selected_diagnostics` for output-subset DCE.
- Gated the selected/DCE path conservatively:
  - pure SW M9 requests are expanded to the complete SW family and use the selected JIT;
  - full output, mixed requests, LW requests, and surface-field requests use the full M9 path.
- Threaded `variable_subset` through daily and nested output diagnostics, including the Noah-MP carry-backed path.

## Default Byte-Identity Contract

The default forecast/physics paths are unchanged. The new SW/LW flux-slice code is reached only by the output-time M9 diagnostic route (`compute_m9_diagnostics` -> `_m9_flux_slices_only=True`) or by the output-subset selected wrapper. It is not called from the timestep physics solvers.

Default full M9 output remains byte-identical to `origin/main`: `P0_DEFAULT_FULL_FIELD_COMPARE.json` compares all 17 M9 writer fields from the current default path against `origin/main` and reports PASS. The only reduced-HLO selected path is gated to pure SW M9 output subsets; mixed, LW, and surface-field requests fall back to the full M9 diagnostic path.

## CPU Proof Commands

All commands were run with `taskset -c 4-31`, `JAX_PLATFORMS=cpu`, `JAX_PLATFORM_NAME=cpu`, `CUDA_VISIBLE_DEVICES=`, `JAX_ENABLE_COMPILATION_CACHE=0`, and `PYTHONPATH=src:.`.

```bash
pytest -q tests/test_rrtm_lw_operational_wiring.py::test_m9_rrtmg_diagnostic_uses_scoped_512_tile_cap tests/test_v0201_training_output_subset.py::test_training_output_subset_env_opt_in tests/test_v0201_training_output_subset.py::test_m9_subset_attrs_expand_only_requested_radiation_dependencies tests/test_v0222_output_pipeline.py::test_materialize_stage_honours_radiation_carry_source_flag tests/test_v014_noahmp_nested_pipeline.py::test_nested_noahmp_default_keeps_output_time_m9_resolve tests/test_v014_noahmp_nested_pipeline.py::test_nested_noahmp_opt_in_uses_carry_and_omits_unavailable_fluxes
```

Result: `6 passed`.

```bash
pytest -q tests/test_v0201_training_output_subset.py::test_default_none_output_is_value_identical tests/test_v0201_training_output_subset.py::test_subset_restricts_to_named_set_plus_mandatory_coords
```

Result: `2 passed`.

```bash
python proofs/v023/perf_bundle/p0_m9_cpu_proof.py
```

Result: PASS.

```bash
python -m compileall -q src/gpuwrf
git diff --check
```

Result: PASS.

## Proof Objects

- `P0_DEFAULT_FULL_FIELD_COMPARE.json`: PASS. Current default full M9 output is byte-identical to `origin/main` for 17 M9 writer fields: `t2`, `u10`, `v10`, `psfc`, `swdown`, `glw`, `pblh`, `tsk`, `swdnb`, `swupb`, `lwdnb`, `lwupb`, `swdnt`, `swupt`, `lwdnt`, `lwupt`, `swnorm`.
- `P0_CPU_FIELD_COMPARE.json`: PASS. Reduced SW-family selected path is byte-identical to `origin/main` full M9 for `swdown`, `swdnb`, `swupb`, `swdnt`, `swupt`, `swnorm`.
- `P0_HLO_SUMMARY.json`: PASS. Full-before to SW-family-after HLO reductions:
  - chars: `1626763 -> 576030` (`-1050733`)
  - dynamic-update-slice: `17 -> 6`
  - while: `166 -> 52`
  - tuple: `7397 -> 2272`
  - reduce: `672 -> 338`
- HLO/text and field bundles:
  - `P0_before_full.hlo`, `P0_before_full_fields.npz`, `P0_before_full_summary.json`
  - `P0_after_full.hlo`, `P0_after_full_fields.npz`, `P0_after_full_summary.json`
  - `P0_after_subset.hlo`, `P0_after_subset_fields.npz`, `P0_after_subset_summary.json`

## Deferred GPU Validation

No GPU command was run. GPU perf/VRAM confirmation remains deferred to the validation plan on lane 0:2:

- compare `origin/main` vs this branch on the real nested output boundary;
- enable `GPUWRF_NEST_PERF_TIMERS=1`;
- capture M9 diagnostic wall time, VRAM high-water, and output byte identity;
- verify no host/device transfer is introduced inside timestep loops.

## Risks

- The HLO reduction is proven only for pure SW M9 output subsets. CPU probes showed that narrowing mixed, LW, or surface outputs can introduce last-bit differences, so those paths deliberately fall back to full M9.
- No GPU speedup, VRAM reduction, or B200 behavior is claimed in this report.
