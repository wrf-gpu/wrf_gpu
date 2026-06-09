# Worker Report

## Summary:

The worker found and emitted the first useful WRF `d02` step-6000 boundary where
current-step dry source/save-family leaves exist while the dry native state is
still the same as the full pre-RK step-entry state. The WRF hook succeeded and
the source/save instrumentation gap is closed.

Final verdict:
`SOURCE_SAVE_BOUNDARY_READY_NO_JAX_WRAPPER_FULL_DOMAIN_PATCH_AND_SCALAR_OLD_LIMITER`.

## Files Changed

- `proofs/v014/source_save_boundary_hook.py`
- `proofs/v014/source_save_boundary_hook.json`
- `proofs/v014/source_save_boundary_hook.md`
- `proofs/v014/source_save_boundary_hook_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_sources.py`
- `proofs/v014/same_input_single_rk_parity_sources.json`
- `proofs/v014/same_input_single_rk_parity_sources.md`
- `.agent/reviews/2026-06-09-v014-source-save-boundary.md`

No production `src/gpuwrf/**` files were changed.

## Commands Run

- CPU-WRF scratch copy/build/run commands recorded in
  `proofs/v014/source_save_boundary_hook.json`.
- `python -m py_compile proofs/v014/source_save_boundary_hook.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/source_save_boundary_hook.py`
- `python -m json.tool proofs/v014/source_save_boundary_hook.json >/tmp/source_save_boundary_hook.validated.json`
- `python -m py_compile proofs/v014/same_input_single_rk_parity_sources.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity_sources.py`
- `python -m json.tool proofs/v014/same_input_single_rk_parity_sources.json >/tmp/same_input_single_rk_parity_sources.validated.json`
- `git diff -- src/gpuwrf`

## Proof Objects Produced

- `proofs/v014/source_save_boundary_hook.json`
- `proofs/v014/source_save_boundary_hook.md`
- `proofs/v014/source_save_boundary_hook_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_sources.json`
- `proofs/v014/same_input_single_rk_parity_sources.md`
- `.agent/reviews/2026-06-09-v014-source-save-boundary.md`

## Findings

The accepted WRF boundary is after `first_rk_step_part1`,
`first_rk_step_part2`, and `rk_tendency`, but before `relax_bdy_dry`,
`rk_addtend_dry`, `spec_bdy_dry`, `small_step_prep`, and `advance_uv`.

The hook emitted dry source/save records for `MASS_SOURCE`, `MASS2D_SOURCE`,
`U_SOURCE`, `V_SOURCE`, `WPH_SOURCE`, and `MOIST_OLD_QV`. Dry source/save leaves
needed by `DryPhysicsTendencies` are present, including `ru_tendf`, `rv_tendf`,
`rw_tendf`, `ph_tendf`, `t_tendf`, `mu_tendf`, `h_diabatic`, and save-family
fields.

The dry native state is preserved exactly on overlap against the prior full
pre-RK savepoint: `220609` compared values, worst max abs `0.0`.

## Unresolved Risks

- The strict JAX comparison did not run.
- The repo lacks a proof-only full-domain loader/wrapper that constructs
  `State`, `OperationalCarry`, `OperationalNamelist`, `GridSpec`,
  `DycoreMetrics`, and `DryPhysicsTendencies` from WRF-emitted fields only.
- Full-domain same-boundary promoted carry/boundary leaves are not emitted:
  `t_2ave`, `ww`, `mudf`, `muave`, `muts`, `ph_tend`, `mu_save`, `ww_save`,
  `rthraten`, active physics carry, and boundary leaves.
- The hook is a 17x17 patch with one conservative 8-cell-halo-valid mass cell;
  the existing post-RK/pre-halo truth is not a full-domain/full-vertical truth
  surface.
- `scalar_old` is not valid at this WRF boundary under the current namelist, so
  the next proof needs either a dry-only wrapper or a consistent old-field
  strategy.

## Next Decision Needed

Open a wrapper/truth-surface sprint. Preferred target: emit a full-domain,
full-vertical same-boundary WRF source/save plus post-RK truth surface and build
the narrowest proof-only JAX wrapper that can execute `_rk_scan_step_with_pre_halo_capture`
without mixing JAX-produced carry with WRF source leaves.
