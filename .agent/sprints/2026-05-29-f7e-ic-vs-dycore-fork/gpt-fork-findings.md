# F7E GPT Fork Findings

## 1. WRF Ground Truth

### Q1: WRF idealized init

WRF does **not** implement the warm-bubble / density-current perturbation by iterating a nonzero dry-mass perturbation. It perturbs theta, recomputes inverse density, and rebalances geopotential at fixed column dry mass.

Evidence:

- `WRF/dyn_em/module_initialize_ideal.F:1010-1016`: before the bubble, WRF computes dry column mass from the dry sounding and sets `grid%MU_1 = pd_surf-grid%p_top-grid%MUB`, `grid%MU_2 = grid%MU_1`, `grid%MU0 = grid%MU_1 + grid%MUB`. In dry idealized cases this is `mu'=0`.
- `WRF/dyn_em/module_initialize_ideal.F:1026-1045`: WRF initializes perturbation pressure `grid%p` by integrating the hydrostatic vertical-momentum pressure relation from `mu`. In dry `mu'=0`, this leaves `p'=0`.
- `WRF/dyn_em/module_initialize_ideal.F:1103-1118`: `quarter_ss` adds the theta bubble to `grid%t_1/t_2`, then recomputes `grid%alt` and `grid%al` from the unchanged `grid%p+grid%pb`.
- `WRF/dyn_em/module_initialize_ideal.F:1121-1130`: the `quarter_ss` "rebalance hydrostatically" loop updates `grid%ph_1`, `grid%ph_2`, and `grid%ph0`. It does not update `grid%MU_1`, `grid%MU_2`, or `grid%p`.
- `WRF/dyn_em/module_initialize_ideal.F:1148-1174`: `squall2d_x` uses the same fixed-mass theta perturbation plus hydrostatic geopotential rebalance.
- `WRF/dyn_em/module_initialize_ideal.F:1193-1219`: `squall2d_y` uses the same pattern.
- `WRF/dyn_em/module_initialize_ideal.F:1278-1313`: `grav2d_x` / Straka cold bubble uses the same pattern: perturb theta, recompute `alt/al`, rebalance `ph`, fixed mass.
- `WRF/test/em_quarter_ss/namelist.input:69` and `:90` set `hybrid_opt=0` and `non_hydrostatic=.true.`; `:116` selects `ideal_case=2`.
- `WRF/test/em_squall2d_x/namelist.input:61`, `:79`, `:108` and `WRF/test/em_squall2d_y/namelist.input:61`, `:79`, `:108` likewise select pure-sigma nonhydrostatic ideal cases.
- `WRF/test/em_grav2d_x/namelist.input:61`, `:78`, `:103` selects pure-sigma nonhydrostatic `ideal_case=6`.

Answer: the perturbation is theta-only in mass (`mu'` remains zero for dry cases), but **not** theta-only in geometry. WRF rebalances `ph_1/ph_2/ph0` hydrostatically after changing theta. There is no explicit bubble-driven `mu'` iteration and no `mu' != 0` cancellation mechanism.

### Q2: `pg_buoy_w` semantics

WRF `pg_buoy_w` is the dry vertical pressure-gradient plus dry-mass perturbation term. In dry air it is exactly:

`g * ( rdn(k) * (p(k)-p(k-1)) - c1f(k) * mu' ) / msfty`

Evidence:

- `WRF/dyn_em/module_big_step_utilities_em.F:2541-2549`: WRF documents the dry form as `g * [rdn(k)*{p(i,k,j)-p(i,k-1,j)} - c1(k)*mu(i,j)]`.
- `WRF/dyn_em/module_big_step_utilities_em.F:2564-2571`: the interior-face implementation applies that formula.
- `WRF/dyn_em/module_big_step_utilities_em.F:2555-2561`: the top-face implementation uses the same pressure-gradient/mass split with the one-sided top pressure difference.
- `WRF/dyn_em/module_em.F:1361-1368`: `rk_tendency` calls `pg_buoy_w(rw_tend, p, cqw, mu, mub, ...)`.
- `WRF/dyn_em/solve_em.F:1848-1859`: `rk_tendency` receives `grid%p`, `grid%mu_2`, `grid%al`, `grid%alt`, and `grid%pb` after `rk_step_prep`.

For a WRF-balanced dry bubble, `mu'=0` and `p'` is not an arbitrary theta-derived pressure. `p` is WRF's perturbation pressure diagnostic consistent with the hydrostatically rebalanced `ph/al`. Therefore the `pg_buoy_w` term should not produce the observed frozen `0.615 m/s^2` direct forcing from a fixed theta bubble. If fed a theta-perturbed column with base `ph` and a theta-derived `p'`, the WRF formula will faithfully accelerate that imbalance.

### Q3: p' diagnosis and where the imbalance lives

WRF's p' diagnosis is field-consistency sensitive: it first gets inverse-density perturbation from geopotential/mass, then pressure from the equation of state. That means an unbalanced theta-only/base-ph column will produce a large p' artifact, while WRF's hydrostatically rebalanced bubble does not.

Evidence:

- `WRF/dyn_em/module_big_step_utilities_em.F:1023-1030`: nonhydrostatic `calc_p_rho_phi` computes `al` from `ph(i,k+1)-ph(i,k)` and `mu`.
- `WRF/dyn_em/module_big_step_utilities_em.F:1056-1087`: it computes perturbation pressure from theta and `al+alb`, then subtracts `pb`.
- `WRF/dyn_em/module_small_step_em.F:522-528`: acoustic `calc_p_rho` does the same split for the small-step work variables: `al` from `ph`, then `p` from the linearized EOS.
- `WRF/dyn_em/module_small_step_em.F:548-567`: only after that does WRF apply `smdiv` pressure memory.
- `WRF/dyn_em/solve_em.F:2628-2670`: WRF calls `calc_p_rho(... step=0 ...)` after `small_step_prep`.
- `WRF/dyn_em/solve_em.F:4164-4206`: WRF calls `calc_p_rho(... step=iteration ...)` after each acoustic substep.
- `WRF/dyn_em/module_small_step_em.F:187-190`: RK1 `small_step_prep` saves `MU_2` then sets the work `MU_2=0`.
- `WRF/dyn_em/module_small_step_em.F:262-276`: RK1 work `t_2` and `ph_2` are differences between reference and current states, so a rest initial state starts the acoustic work arrays at zero.

Conclusion for Q3: the 9.4x forcing lives in inconsistent p' construction, not in the WRF `pg_buoy_w` formula. A theta-perturbed column with base `ph` is not WRF's idealized IC. But the current JAX path also has an operator-side p' over-count that can recreate the same bad pressure even after `ph` is rebalanced.

## 2. Verdict

**VERDICT: BOTH, with the decisive correction being field-consistent WRF p'/ph semantics, not a `mu'` iteration and not a `pg_buoy_w` coefficient change.**

What is true:

- The original "theta perturbation + base ph + base p" setup is an IC bug relative to WRF. WRF changes theta and then rebalances `ph_1/ph_2/ph0`.
- `mu'=0` is correct for the dry warm-bubble and Straka ideal cases. Hypothesis A's nonzero-`mu'` cancellation story is false.
- JAX's `pg_buoy_w_dry` stencil matches WRF's dry formula: `src/gpuwrf/dynamics/core/advance_w.py:118-126` corresponds to `WRF/dyn_em/module_big_step_utilities_em.F:2564-2571`.
- The active JAX over-count is the synthetic absolute `p_buoy` path, not the WRF formula. `src/gpuwrf/runtime/operational_mode.py:664-681` derives an absolute pressure from theta/ph and `src/gpuwrf/dynamics/core/acoustic.py:523-531` gives that synthetic `p_buoy` priority over the acoustic pressure. That is not what WRF passes to `pg_buoy_w`.

What is false:

- False A detail: WRF does not make `mu' != 0` to cancel `rdn*dp'`.
- False B detail: WRF does not require a theta-only/base-ph column to make `pg_buoy_w ~= g*theta'/theta0`. A WRF-balanced dry bubble can have zero direct initial `pg_buoy_w` while still evolving through the coupled WRF pressure/geopotential and horizontal PGF terms.

## 3. Fix Spec For Next Opus Sprint

Implement WRF field-consistent idealized IC and pressure diagnostics. Do not tune `pg_buoy_w_dry`.

1. In `src/gpuwrf/ic_generators/idealized.py`, make the WRF fixed-mass balance explicit and testable:
   - Keep `mu_perturbation = 0` for dry warm bubble and density current.
   - Keep initial dry `p_perturbation` as WRF's fixed-mass pressure diagnostic, zero for the dry pure-sigma neutral-base cases.
   - After applying theta perturbation, recompute `alt_full = EOS(theta_full, pb+p)` and `al = alt_full - alb`.
   - Integrate perturbation geopotential from the lower boundary with WRF's hydrostatic recurrence from `module_initialize_ideal.F:1123-1129` / `:1305-1313`.
   - Store `ph_perturbation = ph_full - phb`, `ph_total = phb + ph_perturbation`.
   - Remove or rewrite comments that say base `ph` plus theta is the intended buoyancy source; WRF's intended source is the balanced perturbation fields.

2. In `src/gpuwrf/runtime/operational_mode.py`, remove the synthetic `p_buoy_abs` construction at `:664-681`.
   - `pg_buoy_w_dry` should consume WRF's actual perturbation pressure diagnostic, not a second theta-derived pressure.
   - Set `p_buoy=None`, or set it to `state.p_perturbation` only if that field is the WRF `grid%p` diagnostic from the current RK stage.
   - Keep `calc_p_rho_wrf` / `calc_p_rho_step` as the source for small-step pressure, matching `WRF/dyn_em/module_small_step_em.F:522-528`.

3. In `src/gpuwrf/dynamics/core/rk_addtend_dry.py`, fix `_absolute_diagnostics` at `:93-133` to stop deriving `p_abs` from absolute theta.
   - Use `state.p_perturbation` for WRF `p`.
   - Compute `al` from `ph_perturbation` and `mu_perturbation`.
   - Keep `alt` from EOS and `php` from full `ph`.
   - This preserves WRF horizontal PGF inputs without inventing a vertical pressure source.

4. Add a focused parity proof:
   - Warm-bubble and density-current ICs: `max_abs(mu_perturbation) == 0`.
   - WRF-balanced ICs: `calc_p_rho_wrf(step=0)` work `p` and `al` are near zero at RK1 because work arrays are zero.
   - WRF large-step `pg_buoy_w_dry(state.p_perturbation, mu_perturbation)` has no frozen `0.615 m/s^2` warm-bubble forcing.
   - No host/device transfer inside timestep loops.

## 4. Falsifiable Check

The fix is proven only if all of these pass:

1. `proofs/f7e/rwtend_after_fix.json`: warm bubble with WRF-balanced fixed-mass IC has `max_abs_c1f_mu_term = 0`, `max_abs_pg_term` not producing `0.615 m/s^2`, and `max_abs_rw_phys_m_s2 < 0.01` for the direct stage-constant `pg_buoy_w` source.
2. A deliberately bad control IC with theta perturbation but base `ph` reproduces the large p' artifact, proving the checker can fail.
3. Warm bubble remains finite through 500 s with no linear `max|w| ~= 0.615*t` growth.
4. Straka density current remains finite through 900 s and the front-position proof is in the expected WRF benchmark neighborhood, approximately 15 km on the contracted grid.

F7E_FORK_COMPLETE
