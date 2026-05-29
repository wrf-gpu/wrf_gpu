# F7D GPT Verification: Acoustic Pressure Formulation

## 1. WRF Ground Truth

### Q1: What state does WRF `calc_p_rho` operate on each substep?

WRF does **not** pass absolute prognostic theta/phi/mu directly as the small-step work arrays. `small_step_prep` explicitly converts prognostics into coupled perturbation work variables: its description says it computes "coupled perturbation variables for the small timestep" and gives `mu*u" = mu(t)*u(t)-mu(*)*u(*)` as the pattern (`module_small_step_em.F:107-111`).

For RK step 1, WRF copies current fields into the RK reference family (`u_1=u_2`, `v_1=v_2`, `t_1=t_2`, `w_1=w_2`, `ph_1=ph_2`) at `module_small_step_em.F:128-169`, sets `MUTS=MUB+MU_2` at `module_small_step_em.F:172-175`, saves the physical perturbation mass in `MU_SAVE`, then zeros the small-step mass work array `MU_2` at `module_small_step_em.F:187-190`. It then forms coupled work arrays such as `t_2=(c1h*MUTS+c2h)*t_1-(c1h*MUT+c2h)*t_2` and `ph_2=ph_1-ph_2` at `module_small_step_em.F:259-277`; for RK1 these initially collapse to zero work differences.

For later RK steps, `small_step_prep` stores the RK reference full column mass in `MUTS=MUB+MU_1`, saves the stage-entry perturbation mass in `MU_SAVE=MU_2`, and converts `MU_2` to the work delta `MU_1-MU_2` at `module_small_step_em.F:196-215`. The key point is that the work `mu` is a delta, while the total dry mass used by the acoustic equations is separately carried as `MUT`/`MUTS`.

`calc_p_rho` is called immediately after `small_step_prep` with `grid%mu_2` as WRF's `mu` argument and `grid%muts` as WRF's `mut` argument (`solve_em.F:2628-2635` and `solve_em.F:2658-2665`). It is called again after each acoustic substep with the same state family: `grid%mu_2`, `grid%muts`, `grid%t_2`, `grid%t_save`, and `grid%ph_2` (`solve_em.F:4164-4171` and `solve_em.F:4194-4201`).

The exact nonhydrostatic pressure diagnostic is:

- `al(i,k,j)=-1./(c1h(k)*Mut(i,j)+c2h(k))*(alt(i,k,j)*(c1h(k)*mu(i,j)) + rdnw(k)*(ph(i,k+1,j)-ph(i,k,j)))` (`module_small_step_em.F:522-523`).
- `p(i,k,j)=c2a(i,k,j)*(alt(i,k,j)*(t_2(i,k,j)-(c1h(k)*mu(i,j))*t_1(i,k,j)) / ((c1h(k)*Mut(i,j)+c2h(k))*(t0+t_1(i,k,j)))-al(i,k,j))` (`module_small_step_em.F:527-528`).

So the accurate formulation is: WRF diagnoses **small-step perturbation pressure/inverse-density from coupled work deltas**, but the denominator is the **current full small-step dry-mass total** `MUTS = MUT + MU_work`, not a bare RK-entry delta. This is neither "absolute p-prime directly in `p`" nor "delta-only with no full-total mass path."

### Q2: Does WRF accumulate a small-step pressure response, or is buoyancy only frozen once per stage?

WRF has both:

1. A **frozen large-step vertical PGF/buoyancy tendency** assembled before the acoustic loop.
2. A **small-step acoustic feedback loop** that updates mass/theta/geopotential and recomputes work `p`/`al` every acoustic substep.

The frozen large-step vertical forcing is `pg_buoy_w`. `rk_tendency` calls `pg_buoy_w(rw_tend, p, cqw, mu, mub, ...)` when nonhydrostatic (`module_em.F:1361-1368`). The routine documents the dry form as `(1./msft)*g*[rdn(k)*(p(k)-p(k-1))-(c1(k)*mu)]` (`module_big_step_utilities_em.F:2547-2549`) and implements it at interior faces as `rw_tend += (1/msfty)*g*(cq1*rdn(k)*(p(i,k,j)-p(i,k-1,j)) -(c1f(k)*muf(i,j))-...)` (`module_big_step_utilities_em.F:2564-2571`). That `p` is the stage-entry perturbation pressure passed into `rk_tendency` (`solve_em.F:1848-1859`), not the later small-step work `p`.

The small-step loop then advances `u/v`, `mu/theta/ww`, `w/ph`, and only after that recomputes `p/al` for the next substep: `advance_uv` starts at `solve_em.F:3065-3153`, `advance_mu_t` updates `MU`, `MUTS`, `MUAVE`, `MUDf`, `ww`, and theta at `module_small_step_em.F:1102-1108` and `module_small_step_em.F:1138-1171`, `advance_w` updates `w` and `ph` at `solve_em.F:3824-3901`, and then `calc_p_rho(step=iteration)` recomputes `grid%al, grid%p` at `solve_em.F:4161-4208`.

The work `p` is used directly by `advance_uv` in the horizontal small-step PGF. WRF builds `dpxy` from `ph`, `alt`, `p`, `pb`, and `al` at `module_small_step_em.F:828-831`, adds the nonhydrostatic pressure-column term using `p` at `module_small_step_em.F:836-863`, and applies it to `u` at `module_small_step_em.F:866-869`; the `v` path mirrors this at `module_small_step_em.F:902-942`.

`advance_w` does **not** take `p` as an argument. Its vertical acoustic restoring path uses the updated `ph`, `t_2ave`, `muave`, `muts`, and frozen `c2a/alt` coefficients. The explicit large-step `rw_tend` enters at `module_small_step_em.F:1477`; the implicit pressure/geopotential term uses `rhs/ph` differences at `module_small_step_em.F:1478-1485`; the vertical buoyancy/restoring term uses `rdn(k)*(c2a*alt*t_2ave difference) - c1f(k)*muave` at `module_small_step_em.F:1486-1489`; and `ph` is advanced from the solved `w` at `module_small_step_em.F:1581-1584`. Thus WRF's vertical feedback is not "recompute `pg_buoy_w` from work `p` each substep"; it is the implicit `advance_w` coupling plus the refreshed `calc_p_rho` work pressure for the next horizontal PGF.

### Q3: Is the frontrunner's framing correct?

The diagnosis is **partly right about the split, but wrong about the WRF fix**.

Correct parts:

- WRF's small-step `p` is a work-array perturbation pressure, and it can be near zero at RK-stage entry for a balanced or slowly evolving stage. That matches `small_step_prep` converting state to work deltas (`module_small_step_em.F:107-111`, `module_small_step_em.F:187-190`, `module_small_step_em.F:259-277`) and `calc_p_rho` operating on those work arrays (`module_small_step_em.F:522-528`).
- WRF's `pg_buoy_w` is a large-step tendency assembled before the acoustic loop from the stage-entry `p` and `mu` (`module_em.F:1361-1368`; `module_big_step_utilities_em.F:2564-2571`). It is not recomputed inside the acoustic substep loop.

Incorrect parts:

- WRF does **not** make the small-step `p` be the absolute perturbation pressure by feeding `mu_save + mu_work` directly as the `mu` work argument. The numerator still uses the work `mu`, work `theta`, and work `ph`; the full-total part is the `MUTS` denominator and mass coupling in `advance_mu_t`.
- The absence of absolute `p` in small-step `calc_p_rho` is not, by itself, a missing restoring loop. WRF refreshes `MUTS=MUT+MU` in `advance_mu_t` (`module_small_step_em.F:1102-1107`), updates `theta` (`module_small_step_em.F:1138-1171`), updates `w/ph` through the implicit vertical solve (`module_small_step_em.F:1477-1489`, `module_small_step_em.F:1581-1584`), and then recomputes work `p/al` for the next substep (`solve_em.F:4161-4208`; `module_small_step_em.F:522-528`).
- The proposed "make work `p` absolute p-prime" would contradict the WRF split and likely double-count the large-step PGF/buoyancy already applied through `horizontal_pressure_gradient` and `pg_buoy_w` (`module_em.F:1325-1335`, `module_em.F:1361-1368`).

## 2. Verdict

**PARTIAL.**

The frontrunner correctly identified that JAX has a pressure/mass formulation problem around the small-step acoustic loop, and correctly observed that the WRF large-step buoyancy source uses stage-entry absolute perturbation pressure. But the proposed root framing, "WRF small-step `calc_p_rho` should diagnose absolute p-prime instead of work pressure," is refuted by WRF source.

The WRF-faithful target is:

- keep `pg_buoy_w` as a once-per-RK-stage large-step `rw_tend` source from stage-entry absolute `p` and `mu`;
- keep small-step `p` as a work-array perturbation pressure;
- ensure the work-pressure diagnostic uses the WRF full small-step total mass `MUTS = MUT_stage_total + MU_work` and the live coupled work `theta/ph/mu` updated each acoustic substep.

## 3. JAX Divergence And Fix Spec

### Divergence

JAX currently has two separate issues.

First, comments and state naming claim WRF fidelity while the mass total passed to `calc_p_rho` is not WRF's `MUTS` total. In `small_step_prep_wrf`, `mut = _base_mu(state)` (`src/gpuwrf/dynamics/core/small_step_prep.py:171`), `mu_work = ...` (`src/gpuwrf/dynamics/core/small_step_prep.py:175`), and `muts = mut + mu_work` (`src/gpuwrf/dynamics/core/small_step_prep.py:176`). In WRF, `grid%mut` is the full stage dry mass from `calculate_full(mut,mub,mu)` (`module_em.F:184-187`; assignment `rfield=rfieldb+rfieldp` at `module_big_step_utilities_em.F:3912-3916`), and `MUTS` is that full stage mass plus the small-step work delta (`module_small_step_em.F:1102-1107`). JAX's `prep.mut` is therefore `MUB`, not WRF `MUT`.

Second, JAX's pressure diagnostic uses that wrong mass total. `calc_p_rho_wrf` passes `mut=prep.mut` into `_calc_al_p` (`src/gpuwrf/dynamics/core/calc_p_rho.py:109-120`), and `acoustic_substep_core` passes `mut=uv_state.mut` into `calc_p_rho_step` (`src/gpuwrf/dynamics/core/acoustic.py:599-615`). WRF's corresponding call passes `grid%muts` as the `Mut` denominator (`solve_em.F:2628-2635`, `solve_em.F:4164-4171`). The JAX `_calc_al_p` formula itself matches WRF lines 522-528 structurally (`src/gpuwrf/dynamics/core/calc_p_rho.py:73-87`), but it is being fed the wrong total-mass field.

The lines that intentionally make JAX `p` a work/delta pressure are `mu_work` and `theta_work` construction in `small_step_prep_wrf` (`src/gpuwrf/dynamics/core/small_step_prep.py:175-198`) and `_calc_al_p` consuming `mu_work`, `ph_work`, and `theta_work` (`src/gpuwrf/dynamics/core/calc_p_rho.py:51-87`). That work-pressure property is WRF-consistent. The divergence is the missing WRF full-total `MUT/MUTS` semantics, not the fact that work `p` is a work pressure.

JAX does refresh `p` after each substep: `acoustic_substep_core` advances `mu/theta/ww` (`src/gpuwrf/dynamics/core/acoustic.py:488-502`), advances `w/ph` (`src/gpuwrf/dynamics/core/acoustic.py:539-589`), then calls `calc_p_rho_step` and stores `p_rho.p`, `p_rho.al`, and `p_rho.pm1` (`src/gpuwrf/dynamics/core/acoustic.py:599-638`). However, because the total mass semantics are wrong, that refresh is not the WRF `MUTS` path. Separately, `p_for_buoy = uv_state.p_buoy if ... else uv_state.p` (`src/gpuwrf/dynamics/core/acoustic.py:523-531`) keeps the large-step buoyancy source frozen; that is WRF-consistent in cadence, provided `p_buoy` is computed from the correct stage-entry absolute diagnostics.

### Exact Fix Spec For Next Opus Sprint

Do **not** change source in this verifier task. For the implementation sprint:

1. Fix `SmallStepPrepState` mass semantics in `src/gpuwrf/dynamics/core/small_step_prep.py`.
   - Add/keep an explicit base dry mass field if needed, but make `prep.mut` mean WRF `grid%mut`: full stage-entry dry mass `MUB + MU_current`.
   - Compute `mu_work` exactly as WRF work `MU_2` after `small_step_prep`: RK1 `0`; later RK steps `MU_ref - MU_current`.
   - Compute `prep.muts = prep.mut + mu_work`, which gives WRF `MUTS` (`MUB+MU_current+MU_work`).
   - Recompute `muu/muv` from `prep.mut` and `muus/muvs` from `prep.muts`, matching `module_small_step_em.F:172-207`.
   - Build `theta_work`, `u_work`, `v_work`, and `w_work` with the WRF current/stage mass pairs, matching `module_small_step_em.F:238-276`.

2. Fix `calc_p_rho` call sites.
   - In `calc_p_rho_wrf`, pass `mut=prep.muts` into `_calc_al_p`, not the stage-entry/base `prep.mut`.
   - In `acoustic_substep_core`, pass the live `muts_new` into `calc_p_rho_step` as the WRF `Mut` denominator, not `uv_state.mut`.
   - Rename `_calc_al_p`'s `mut` parameter to `mut_total` or `muts_total` to prevent this bug from recurring.
   - Keep `_calc_al_p`'s numerator as work variables: `mu_work`, `ph_work`, `theta_work`. Do not replace it with absolute perturbation `mu`/`theta`/`ph`.

3. Fix downstream users of `mut`.
   - `advance_mu_t_wrf` should continue to satisfy `mu_work_old = inputs.muts - inputs.mut`, but now `inputs.mut` must be WRF full stage-entry dry mass.
   - `calc_coef_w_wrf_coefficients` must receive WRF `grid%mut` full stage-entry mass, as in `solve_em.F:2676-2681`, not `MUB`.
   - `advance_w_wrf` must receive WRF `mut` full stage-entry mass and live `muts`, matching `module_small_step_em.F:1178-1185`.
   - `small_step_finish_wrf` reconstruction denominators/numerators should be audited after the mass semantic change against `module_small_step_em.F:379-430`.

4. Keep the WRF split for vertical buoyancy.
   - Keep `p_buoy` as the stage-entry absolute perturbation-pressure source for `pg_buoy_w`, matching `module_em.F:1361-1368` and `module_big_step_utilities_em.F:2564-2571`.
   - Do not feed the substep work `p` into `pg_buoy_w` as the primary buoyancy source.
   - Do not make the acoustic work `p` be absolute p-prime. The restoring loop is: `advance_uv(work p/al)` -> `advance_mu_t(mu/theta/ww)` -> `advance_w(w/ph, t_2ave, muave)` -> `calc_p_rho_step(work p/al from live MUTS/theta/ph)` -> next substep `advance_uv`.

### Falsifiable Checks

Minimum proof objects for the implementation sprint:

1. A source-parity unit test for RK1 `small_step_prep + calc_p_rho(step=0)` showing work `mu/theta/ph` are zero and work `p/al` are zero for a stage-entry rest state, while independently computed stage absolute `p_buoy` can be nonzero for a warm/cold bubble. This proves the implementation preserves WRF's split.
2. A one-column or small 2-D acoustic-step probe showing that after `advance_mu_t` and `advance_w`, `calc_p_rho_step` changes work `p/al` from their step-0 values and updates `pm1`; the next `advance_uv` must consume that changed work `p`.
3. A WRF fixture/savepoint comparison at substep granularity for `mu_2`, `muts`, `t_2`, `ph_2`, `p`, `al`, `w_2`, and `rw_tend` through at least two acoustic substeps.
4. End-to-end idealized gates: `flat_rest` remains exact; warm bubble and Straka stay finite past the current 80-100 s failure window; `max|w|` must stop coherent linear growth and either saturate/oscillate or match the WRF fixture envelope. Do not accept a clamp/masking fix; require pressure/mass substep traces as proof.

F7D_VERIFY_COMPLETE
