# V0.14 Same-Input Single-RK Parity Sources

Verdict: `SOURCE_SAVE_BOUNDARY_READY_NO_JAX_WRAPPER_FULL_DOMAIN_PATCH_AND_SCALAR_OLD_LIMITER`.

No strict same-input JAX comparison was run. The WRF source/save boundary is consistent, but the current proof cannot build the required JAX input contract from WRF-emitted fields only.

## Boundary

- Source/save boundary ready: `True`.
- Hook records: `{'MASS2D_SOURCE': 289, 'MASS_SOURCE': 12716, 'MOIST_OLD_QV': 12716, 'U_SOURCE': 13464, 'V_SOURCE': 13464, 'WPH_SOURCE': 13005}`.
- Accepted ordering: `after first_rk_step_part1/part2 and rk_tendency; before relax_bdy_dry, rk_addtend_dry, spec_bdy_dry, small_step_prep, advance_uv`.

## Blocker

- Missing wrapper: proof-only loader/wrapper that maps WRF-emitted full-domain source_save_after_rk_tendency records into State, OperationalCarry, OperationalNamelist/GridSpec/DycoreMetrics, and DryPhysicsTendencies, then calls _rk_scan_step_with_pre_halo_capture
- Missing field surface: full-domain same-boundary promoted carry leaves are not WRF-emitted: t_2ave, ww, mudf, muave, muts, ph_tend, mu_save, ww_save, rthraten, active physics carry, and boundary leaves; source hook emits the dry source/save family only
- Patch/truth limitation: source hook is a 17x17 patch with one 8-cell-halo-valid mass cell, while the existing post-RK/pre-halo truth emits only K1 mass/U/V and kstag 0/1 W/PH records, not a full 44-level full-domain State output
- Existing JAX checkpoint used: `False`.
