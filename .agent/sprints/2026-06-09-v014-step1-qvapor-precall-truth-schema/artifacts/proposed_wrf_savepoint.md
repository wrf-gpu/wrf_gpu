# Proposed WRF Savepoint: Step-1 Pre-Call QVAPOR

Boundary: `before_first_rk_step_part1_call` in `dyn_em/solve_em.F::solve_em`.

Place the capture at the existing pre-call hook, after `IF (coupler_on) CALL cpl_settime( curr_secs2 )` and before `CALL first_rk_step_part1`.

Minimal schema change:

- Keep mass fields: `['T_STATE', 'P_STATE', 'PB', 'MU_STATE', 'MUB', 'MUT']`.
- Add mass field: `QVAPOR` from `REAL(moist(i,k,j,P_QV),KIND=8)`.
- Keep W/PH fields unchanged: `['W_STATE', 'PH_STATE', 'PHB']`.
- Add optional header: `moist_index_qv P_QV`.

Shape contract:

- Mass `QVAPOR`: `[44, 66, 159]` in assembled `(z,y,x)` order.
- W/PH remains `[45, 66, 159]`.
- Tile policy: `mass_owned_single_owner_no_overlap, same as accepted pre-call text`.

Acceptance checks:

- All emitted files have surface before_first_rk_step_part1_call, domain_id 2, step 1, rk_step 1.
- MASS_PREPART schema includes QVAPOR after MUT or otherwise with unambiguous field name.
- Tile count and tile bounds match the accepted pre-call T/P/PB/MU/MUB/PH/PHB/W text set.
- Existing T_STATE/P_STATE/PB/MU_STATE/MUB/MUT/W_STATE/PH_STATE/PHB values remain numerically identical to the accepted pre-call dump.
- QVAPOR has shape [44,66,159] after assembly, finite values, and no reuse of post-RK/pre-halo artifacts.
- Validator reports gpu_used=false and same_boundary_qvapor_truth_exists=true only for this boundary.
