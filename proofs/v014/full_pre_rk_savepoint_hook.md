# V0.14 Full Pre-RK Savepoint Hook

Verdict: `FULL_PRE_RK_HOOK_BLOCKED_RK_FIXED_SOURCE_UNAVAILABLE_AT_STEP_ENTRY`.

## Output

- Files: `2`.
- Unique records: `{'MASS_FULL': 12716, 'U_FULL': 13464, 'V_FULL': 13464, 'WPH_FULL': 13005, 'MOIST_FULL': 76296, 'SCALAR_FULL': 38148}`.
- Duplicate tile overlap max delta: `0.0`.
- Patch valid mass cells after 8-cell halo: `1`.

## Sufficiency

- Full native dry state present: `True`.
- Active moisture present: `True`.
- Strict same-input ready: `False`.
- Missing for strict same-input: `['ru_tendf', 'rv_tendf', 'rw_tendf', 'ph_tendf', 't_tendf', 'mu_tendf', 'h_diabatic', 'u_save', 'v_save', 'w_save', 'ph_save', 't_save', 'moist_old', 'scalar_old']`.

The hook lands at the required step-entry boundary. At that exact point WRF has not yet produced current-step `*_tendf` or `*_save` inputs, so the downstream parity proof must fail closed unless a later accepted source boundary is provided.
