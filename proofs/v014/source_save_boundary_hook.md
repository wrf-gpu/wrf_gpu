# V0.14 Source/Save Boundary Hook

Verdict: `SOURCE_SAVE_BOUNDARY_HOOK_READY`.

## Output

- Files: `2`.
- Unique records: `{'MASS_SOURCE': 12716, 'MASS2D_SOURCE': 289, 'U_SOURCE': 13464, 'V_SOURCE': 13464, 'WPH_SOURCE': 13005, 'MOIST_OLD_QV': 12716}`.
- Duplicate tile overlap max delta: `95.91573333740234`.
- Patch valid mass cells after 8-cell halo: `1`.

## Boundary

- Position: `after first_rk_step_part1, first_rk_step_part2, and rk_tendency; before relax_bdy_dry, rk_addtend_dry, spec_bdy_dry, small_step_prep, and advance_uv`.
- Before first dry/acoustic mutation: `True`.
- Raw source/save leaves present: `True`.
- Missing dry source leaves: `[]`.
- Step-entry dry state preserved on overlap: `True`.

The hook closes the WRF source/save instrumentation gap. It does not by itself provide a full-domain JAX wrapper.
