# V0.14 Same-Input Single-RK Parity Full

Verdict: `FULL_PRE_RK_JAX_LOADER_BLOCKED_RK_FIXED_SOURCE_BOUNDARY`.

No JAX-vs-WRF same-input comparison was run. The proof blocks before JAX execution because the WRF inputs are not a complete same-input RK boundary.

## Parsed WRF Inputs

- Full pre-RK records: `{'MASS_FULL': 12716, 'MOIST_FULL': 76296, 'SCALAR_FULL': 38148, 'U_FULL': 13464, 'V_FULL': 13464, 'WPH_FULL': 13005}`.
- Post-RK/pre-halo truth records: `{'MASS_K1': 289, 'U_K1': 306, 'V_K1': 306, 'WPH_KSTAG01': 578}`.
- Candidate valid mass cells after 8-cell halo: `1`.

## Blocker

- Missing exact boundary: current-step source/save surface after WRF has produced ru_tendf, rv_tendf, rw_tendf, ph_tendf, t_tendf, mu_tendf, h_diabatic, u_save, v_save, w_save, ph_save, and t_save, but before any dynamics state update that changes the one-step initial state used by the JAX wrapper.
- Missing fields: `['ru_tendf', 'rv_tendf', 'rw_tendf', 'ph_tendf', 't_tendf', 'mu_tendf', 'h_diabatic', 'u_save', 'v_save', 'w_save', 'ph_save', 't_save', 'moist_old', 'scalar_old']`.

## Next Action

Add a second accepted WRF source boundary, or move the proof boundary, so the same file set contains current-step DryPhysicsTendencies/save-family leaves from WRF. Then construct OperationalCarry/DryPhysicsTendencies and call _rk_scan_step_with_pre_halo_capture on CPU.
