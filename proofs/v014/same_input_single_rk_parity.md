# V0.14 Same-Input Single-RK Parity

Verdict: `SAME_INPUT_TENDENCY_INPUT_BLOCKED_PRE_RK_FULL_NATIVE_STATE_RK_TENDF_AND_HISTORY_SOURCE_FIELDS`.

No JAX-vs-WRF same-input comparison was run. Running it with the current files would be a weak comparison, because the WRF pre-RK input does not contain the state and tendency/source inputs needed by the JAX RK boundary.

## Parsed Inputs

- Pre-RK surface records: `{'MASS_K1': 289}`.
- Pre-RK fields: `{'MASS_K1': ['T_THM', 'T_OLD', 'T_HIST_SRC', 'P', 'PB', 'MU_NEW', 'MU_OLD', 'MUB']}`.
- Post-RK/pre-halo surface records: `{'MASS_K1': 289, 'U_K1': 306, 'V_K1': 306, 'WPH_KSTAG01': 578}`.
- Optional final-calc surface adds missing inputs: `False`.

## Blocker

- Missing full pre-RK native state: `['U on native x-stagger for all vertical levels', 'V on native y-stagger for all vertical levels', 'W on native vertical faces for all vertical levels', 'PH and PHB on native vertical faces for all vertical levels', 'T/P/PB full mass-column, not only K1', 'QV and active moisture/scalar state needed by the operational carry', 'surface/coupling leaves carried by State when physics/boundary are active']`.
- Missing JAX base tendency leaves: `['u', 'v', 'w', 'theta', 'qv', 'p', 'ph', 'mu']`.
- Missing WRF RK-fixed physics/source leaves: `['ru_tendf', 'rv_tendf', 'rw_tendf', 'ph_tendf', 't_tendf', 'mu_tendf', 'h_diabatic', 'u_save', 'v_save', 'w_save', 'ph_save', 't_save']`.
- Missing wrapper: `['proof-only loader that maps a WRF full pre-RK savepoint into OperationalCarry', 'paired OperationalNamelist/GridSpec/DycoreMetrics/boundary_config from the same WRF case', 'call boundary to feed WRF DryPhysicsTendencies into _rk_scan_step_with_pre_halo_capture']`.

## Patch Width

- Status: `NOT_PRIMARY_BLOCKER`.
- Candidate valid mass cells with an 8-cell halo if full state existed: `1`.
- Patch width is not the primary verdict because the proof is blocked before any stencil-valid scoring can begin.

## Next Action

Add a CPU WRF pre-RK hook at solve_em after grid%itimestep increment and before the RK loop that emits the full native-staggered step-entry state plus WRF rk_tendency/rk_addtend_dry inputs: U,V,W,T,P,PB,PH,PHB,MU,MUB,QV/moisture full columns; rk1/_old history/source fields; ru_tendf, rv_tendf, rw_tendf, ph_tendf, t_tendf, mu_tendf, h_diabatic, u_save, v_save, w_save, ph_save, t_save; and the boundary/carry leaves needed by _rk_scan_step_with_pre_halo_capture. Then add a proof-only JAX loader/wrapper that constructs OperationalCarry and feeds those exact WRF tendency leaves before scoring halo-valid cells.

This result supports blocked instrumentation, not upstream drift, final-RK PGF/mass-wind, or theta/tendency source by itself.
