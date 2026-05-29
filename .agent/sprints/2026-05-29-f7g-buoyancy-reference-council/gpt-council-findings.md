# F7G GPT council findings

## 1. WRF answers to Q1-Q3

### Q1: WRF's ideal IC and `calc_p_rho_phi` are the same discrete relation

WRF does not rely on relaxation for the idealized hydrostatic rebalance. It constructs signed eta metrics with `znw` decreasing from 1 to 0, then sets `dnw(k)=znw(k+1)-znw(k)` and `rdnw(k)=1/dnw(k)`, so both are negative for normal eta ordering (`WRF/dyn_em/module_initialize_ideal.F:625-627`, `:711-713`).

The base-state geopotential is integrated with that signed `dnw`:

`phb(k+1) = phb(k) - dnw(k)*(c1h(k)*mub+c2h(k))*alb(k)` (`WRF/dyn_em/module_initialize_ideal.F:977-983`).

The idealized thermal/cold perturbation then recomputes full inverse density from the EOS, stores `al=alt-alb`, and re-integrates perturbation geopotential at fixed column mass:

`ph'(k+1) = ph'(k) - dnw(k)*(((c1h*mub+c2h)+(c1h*mu'))*al(k) + (c1h*mu')*alb(k))` for the warm bubble (`WRF/dyn_em/module_initialize_ideal.F:1103-1118`, `:1121-1129`) and Straka/`grav2d_x` (`WRF/dyn_em/module_initialize_ideal.F:1278-1298`, `:1305-1313`).

After initialization, `start_em` explicitly derives `al` and `p` from `ph` using the same `calc_p_rho_phi` relation, not a different operator: for hypsometric option 1, `al = -1/((c1h*mub+c2h)+(c1h*mu')) * (alb*c1h*mu' + rdnw*(ph'(k+1)-ph'(k)))` (`WRF/dyn_em/start_em.F:819-828`), then `p` is computed from the EOS (`WRF/dyn_em/start_em.F:842-868`). The timestep diagnostic routine uses the same form: `al=-1/(c1*muts+c2)*(alb*c1*mu + rdnw*(ph(k+1)-ph(k)))` and then the EOS pressure (`WRF/dyn_em/module_big_step_utilities_em.F:1023-1030`, `:1082-1088`).

Algebraically, with `M=(c1h*(mub+mu')+c2h)` and WRF-signed `rdnw=1/dnw`, the init recurrence gives:

`rdnw*Delta ph' = -M*al - c1h*mu'*alb`

Substituting that into `calc_p_rho_phi` gives:

`al_calc = -(c1h*mu'*alb + rdnw*Delta ph')/M = al`

So the recurrence is the exact discrete inverse by construction. The suspected mismatch is real only if JAX uses positive `|dnw|`/`1/|dnw|` while applying the WRF formula unchanged. In that convention, the inverse must be sign-adapted; otherwise a balanced `ph'` diagnoses `al` with the wrong sign.

### Q2: `pg_buoy_w` and `advance_w` are consistent by staging, not by a hidden `muave`

WRF builds the large-step vertical PGF/buoyancy tendency in `rk_tendency` by calling `pg_buoy_w(rw_tend, p, cqw, mu, mub, ...)` once per RK stage (`WRF/dyn_em/module_em.F:1361-1368`). In dry air, `pg_buoy_w` adds `g*(rdn(k)*(p(k)-p(k-1)) - c1f(k)*mu')/msfty` on interior faces and the corresponding top-face formula (`WRF/dyn_em/module_big_step_utilities_em.F:2539-2549`, `:2553-2572`). This is stage pressure `grid%p`, not a synthetic theta pressure and not the live pressure recomputed inside every acoustic substep.

The small-step variables are then converted into work deltas. On RK1, `small_step_prep` copies current fields to the `_1` time level, sets `MU_2=0`, stores `t_save=t_2`, and replaces `t_2` by `(c1h*muts+c2h)*t_1 - (c1h*mut+c2h)*t_2`; for a fixed-mass rest state this is zero (`WRF/dyn_em/module_small_step_em.F:125-190`, `:259-264`). It similarly stores `ph_save=ph_2` and replaces `ph_2` with `ph_1-ph_2`, also zero at RK1 rest (`WRF/dyn_em/module_small_step_em.F:268-277`).

`muave` is the small-step mass-work running average, not the physical column perturbation mass. `advance_mu_t` sets old `MUAVE=MU`, advances the work `MU`, computes `MUTS=Mut+MU`, and then sets `MUAVE=.5*((1+epssm)*MU_new+(1-epssm)*MU_old)` (`WRF/dyn_em/module_small_step_em.F:1102-1108`). For a stationary fixed-mass thermal with `mu'=0` and no mass tendency, `muave=0`.

`t_2ave` is also a small-step work quantity. `advance_mu_t` first saves the current work theta into `t_ave` before updating it (`WRF/dyn_em/module_small_step_em.F:1138-1144`). `advance_w` then forms `t_2ave=.5*((1+epssm)*t_2+(1-epssm)*t_2ave)` and normalizes it with `(c1h*Muave*t0)/((c1h*Muts+c2h)*(t0+t_1))` (`WRF/dyn_em/module_small_step_em.F:1341-1344`). Therefore, for a fixed-mass rest IC on the first acoustic substep, `t_2ave=0` and `muave=0`; the initial `theta'` is not supposed to be carried as a direct `c2a*alt*t_2ave` buoyancy source.

The `advance_w` buoyancy-looking term is a small-step work correction:

`rdn(k)*(c2a(k)*alt(k)*t_2ave(k)-c2a(k-1)*alt(k-1)*t_2ave(k-1)) - c1f(k)*muave`

(`WRF/dyn_em/module_small_step_em.F:1435-1455`, `:1477-1489`; top face at `:1492-1502`). It does not need nonzero `muave` to balance a `mu'=0` initialized thermal. The stage hydrostatic reference is already in `grid%al/grid%p/grid%ph`; the small-step solve evolves deviations around that reference and finishes geopotential with the solved coupled `w` (`WRF/dyn_em/module_small_step_em.F:1581-1586`).

### Q3: WRF-correct fix

Do not iterate the IC and do not invent a synthetic `p_buoy`. WRF already has a closed-form discrete inverse.

The next sprint should make the JAX idealized path obey the WRF-signed relation end to end:

1. Restore a single WRF vertical-metric convention for WRF-shaped dycore operators: `dnw=znw(k+1)-znw(k)`, `rdnw=1/dnw`, `dn=0.5*(dnw(k)+dnw(k-1))`, `rdn=1/dn`, with the normal values negative as in WRF (`WRF/dyn_em/module_initialize_ideal.F:711-720`). If the project keeps positive layer thicknesses in geometry helpers, introduce explicit `wrf_dnw/wrf_rdnw/wrf_rdn` adapters and pass those to every WRF formula; do not mix positive metrics with unmodified WRF signs.

2. Make the IC rebalance and `calc_p_rho_phi` exact inverses under that convention. With WRF-signed metrics, keep the WRF formulas literally. With positive metrics, the nonhydrostatic inverse must be sign-adapted to `al=(rdnw_abs*Delta ph' - c1h*mu'*alb)/(c1h*muts+c2h)`, not `-(c1h*mu'*alb + rdnw_abs*Delta ph')/(...)`.

3. Reproduce `start_em` post-init recomputation before the first RK tendency: derive `al` and `p` from `ph_1`, `mu_1`, `mub`, `alb`, signed `rdnw`, and the EOS (`WRF/dyn_em/start_em.F:819-868`). Store that as the stage `grid%p` equivalent.

4. Build `pg_buoy_w` exactly once per RK stage from that stage `grid%p` and stage `mu`, matching `module_em.F:1361-1368` and `module_big_step_utilities_em.F:2553-2572`. Do not recompute `pg_buoy_w` inside each acoustic substep from the live `calc_p_rho` work pressure.

5. Fix the small-step work references. At a new WRF small-step stage, `t_2ave` must represent the work-theta average saved/updated by `advance_mu_t`, not the full initialized theta field. For an RK1 fixed-mass rest thermal, the first-substep `t_2ave` and `muave` must be zero by the cited WRF lines. Feeding initial `theta'` through `t_2ave` is a JAX reference bug and double-counts the thermal relative to WRF.

## 2. Decisive root cause

Both symptoms are JAX-side issues, but they are not two independent WRF mechanisms:

- The 19x `pg_buoy_w(grid%p)` artifact is a discrete-inverse/sign mismatch. WRF's init recurrence and `calc_p_rho_phi` are exact inverses because WRF `dnw/rdnw` are signed. JAX currently documents and constructs positive `dnw/rdnw` in the idealized metric path while applying WRF formulas that require signed `rdnw`; that flips the diagnosed `al/p` relation for a hydrostatically rebalanced `ph'`.

- The claimed large-step/small-step "reference mismatch" is not present in WRF. The mismatch is in JAX staging: treating `t_2ave` as if it should carry the initialized `theta'`, and/or feeding live small-step pressure into `pg_buoy_w`, violates WRF's work-variable staging. WRF's `muave` is only a small-step mass-work average; it is not supposed to become nonzero just to balance a `mu'=0` thermal.

## 3. Exact next-sprint fix spec

Implement the signed-metric/inverse fix first, then the staging fix:

1. Add a WRF-signed vertical metric view and route WRF-faithful operators through it: `diagnose_pressure_al_alt`, `core/calc_p_rho.py`, `pg_buoy_w_dry`, horizontal nonhydrostatic PGF, `advance_w_wrf`, and idealized IC rebalance. The acceptance criterion is that an IC-generated `ph'` round-trips through the WRF `calc_p_rho_phi` algebra to the same `al` used to build it.

2. In the idealized IC builder, stop hiding the convention with `abs(dnw)` unless all downstream WRF operators receive the matching sign adapter. The code should make the equation visibly equivalent to WRF: `ph(k+1)=ph(k)-wrf_dnw(k)*(...)`.

3. In the operational acoustic path, compute a stage `rw_tend_pg_buoy` before the acoustic substep scan from stage `p`/`mu`, and carry it unchanged through all acoustic substeps. Keep `calc_p_rho(step=iteration)` for WRF's substep pressure/density refresh and smdiv memory only; it is not the source of a new `pg_buoy_w` each substep.

4. Replace the initial/full-theta `t_2ave` carry with WRF work-variable semantics. After `small_step_prep` on an RK1 rest thermal, the old and new theta work values are zero, so `advance_w` sees `t_2ave=0`; later substeps may produce nonzero `t_2ave` only from actual small-step theta evolution.

5. Delete/disable the F7F workaround comments and switches that say live small-step pressure is the correct `pg_buoy_w` source. WRF source says the large-step call is `module_em.F:1361-1368`; the small-step `calc_p_rho` call is later in the acoustic loop (`WRF/dyn_em/solve_em.F:4161-4208`).

## 4. Falsifiable checks

1. Algebraic round-trip check: for warm-bubble and Straka IC columns, compute `al_init=alt_full-alb`, build `ph'`, then diagnose `al_calc` with the WRF-signed `calc_p_rho_phi` formula. Required: `max_abs(al_calc-al_init) <= 1e-12` in fp64 interior columns.

2. Pressure/source check: after the `start_em`-equivalent recomputation, `pg_buoy_w(grid%p)` must not show the 19x artifact. For the fixed-mass hydrostatically balanced rest diagnostic, require direct vertical `rw_tend` residual near zero (`max_abs <= 1e-10 m s^-2` in the no-motion column test) or, for an explicitly unbalanced analytic-buoyancy oracle, require `pg_buoy_w(grid%p)/analytic in [0.9, 1.1]`. Do not accept a 9x/19x ratio.

3. Small-step reference check: first acoustic substep from RK1 fixed-mass rest IC must have `max_abs(muave)=0`, `max_abs(t_2ave)=0` before `advance_w` term B, and `pg_buoy_w` must be the fixed stage array, not recomputed from `calc_p_rho_step`.

4. WRF fixture check: instrument WRF for the same ideal cases and compare JAX first-stage `al`, `p`, `rw_tend` from `pg_buoy_w`, `muave`, and `t_2ave` against WRF savepoints at the corresponding boundaries.

F7G_GPT_COMPLETE
