# F7 Plan Critique

## Findings

1. **F7.A has a dangerous lifetime bug in the contract wording.** The contract says RK2/RK3 must consume the RK1-produced `mu_save`. That is not what WRF does. `module_small_step_em.F:172-215` saves current `MU_2` into `MU_SAVE` every RK stage; the cross-stage family is `MU_1`, `T_1`, `U_1`, `V_1`, `W_1`, and `PH_1`, copied on RK1 and reused later. If the worker implements "carry RK1 `mu_save`" literally, the fix is wrong. The right F7.A target is: carry RK1 `_1` fields across RK stages, and treat `*_save`/`mu_save` as per-stage restoration state.

2. **F7.A is missing a hard acceptance criterion for the actual first failure.** F6 says the first pure-dycore critical violation is step 1, RK1, substep 1, `advance_mu_t`, `theta_sanity_bounds`; `muts_mut_work_mu_consistency` and `theta_mass_residual` also fire on that same first substep. The current F7.A AC only requires acoustic u/v delta > 0 and no RK2 `theta_1` invariant. That can pass while the original RK1/substep1 mass-theta failure still exists.

3. **`advance_w_wrf` is not a blocker for the first `advance_uv_wrf`, but `calc_p_rho(step=0)` and correct `dts_rk` probably are.** WRF order is `calc_p_rho(step=0)`, `calc_coef_w`, then substep `advance_uv -> advance_mu_t -> advance_w -> calc_p_rho(step=iteration)`. So `advance_w` can stay in F7.B for the first substep, but `advance_uv` consumes pressure/geopotential/density fields that should have been prepared by `calc_p_rho(step=0)`. Also current `_acoustic_scan` deletes `dt_stage` and uses `dt_s / acoustic_substeps`; RK1 should use `dts_rk = dt/3` with one small step. F7.A does not make this a clear AC.

4. **The proof gates are too weak for a WRF-faithful rewrite.** "acoustic uv max delta > 0" only proves some u/v code ran. It does not prove WRF pressure-gradient signs, mass coupling, map-factor handling, nonhydrostatic `dpn`, or divergence damping are correct. The three F6 unit tests are useful smoke tests, but they are not a WRF fixture, analytic oracle, or conservation proof.

5. **The 2-3 day estimate is optimistic.** A real F7.A includes separate `_1` and `_save` lifetimes, RK-stage prep/finish transformations for u/v/w/theta/ph/mu, `c2a`, `muus/muvs`, `dts_rk` plumbing, and a substantial `advance_uv` port. That is more like 3-5 days if done carefully, plus proof work. It might be 2-3 days only if the worker ships a partial periodic/no-boundary approximation and calls it done.

## Q1 - Is the F7.A -> F7.B -> F7.C -> F7.D sequence right?

Mostly right as a bounded decomposition, but F7.A is not quite the right first unit as written.

`advance_w_wrf` should not move before `advance_uv_wrf`: WRF calls `advance_uv` first in each acoustic substep, then `advance_mu_t`, then `advance_w`. The first F6 critical failure is before `advance_w` would run, so moving all of `advance_w` into F7.A is not the minimal way to attack the first failure.

But `calc_p_rho(step=0)` should move earlier, or at least be represented by a real preparatory pressure state in F7.A. `advance_uv` is a pressure-gradient update; doing it before implementing the pressure/inverse-density cadence risks producing a nonzero u/v delta that is numerically meaningless. `calc_p_rho(step=iteration)` and the full `advance_w`/ph update can stay in F7.B, but the loop-entry pressure state is part of the `advance_uv` contract.

`rk_addtend_dry` should not be folded into F7.A unless the team freezes a proper large-step tendency bundle now. It is not a cheap append-only detail; it defines what `ru_tend`, `rv_tend`, `rw_tend`, `t_tend`, `ph_tend`, and `mu_tend` mean before the acoustic step. For pure-dycore/physics-off debugging it is probably not the first blocker, but the F7.A API should not make F7.C harder.

Recommended order:

1. F7.0 or F7.A preflight: correct RK stage descriptors (`dt_rk`, `dts_rk`, substep count) and prove the `_1` vs `*_save` lifetime model on a one-step harness.
2. F7.A: `small_step_prep_wrf`/`finish_wrf`, loop-entry `calc_p_rho(step=0)` or explicit pressure-prep equivalent, `advance_uv_wrf`, and proof that the original RK1/substep1 theta/mass failures clear.
3. F7.B: full `advance_w_wrf`, ph update, `calc_p_rho(step=iteration)`, and vertical acoustic proof.
4. F7.C: `rk_addtend_dry` plus WRF-shaped dry tendency/advection merger.
5. F7.D: `sumflux`, scalar tendency cadence, and scalar boundary cadence.

## Q2 - Is bounded better than one large sprint?

Yes, bounded is the right call. A single 1000+ LOC sprint would be faster only in the fake sense: fewer handoffs, more hidden coupling, and a much higher chance of a green self-comparison that still fails WRF behavior. F5/F6 already show that local "fixes" can satisfy narrow tests while the state lifetimes remain wrong.

The problem is not bounded vs mega; the problem is the current boundaries and proof gates. Bounded sprints are valuable only if each one has a falsifiable proof object. F7.A's current proof target is too permissive. It should require the first F6 pure-dycore critical violation to move past step 1/RK1/substep1, `muts_mut_work_mu_consistency` to clear at that point, and a WRF or analytic check for at least the prep/advance_uv state.

## Q3 - Is the F7.A contract overscoped/underscoped?

Both.

Overscoped for 2-3 days:

- Full `small_step_prep`/`finish` plus full `advance_uv` is not a small surgical patch.
- `advance_uv` lines 654-942 are not just "add pressure gradient"; they include large-step tendencies, nonhydrostatic pressure terms, mass/map coupling, moisture factors, divergence damping, and boundary-sensitive loop bounds.
- `speedup_estimate.json` is premature for a correctness rewrite unless it is clearly labeled as a rough regression check, not a GPU performance claim.

Underscoped for correctness:

- It does not explicitly fix `dts_rk`; current code uses `dt_s / acoustic_substeps` even for RK1, while WRF RK1 uses `dt/3`.
- It does not include loop-entry `calc_p_rho(step=0)`, even though `advance_uv` depends on those fields.
- It does not mention `calc_mu_uv_1` before `small_step_finish`, which F5 lists as required after acoustic substeps.
- It does not clearly separate RK1 `_1` fields from per-stage `*_save` fields.
- Its AC does not require the first F6 theta/mass violation to clear.
- It lacks a WRF savepoint or analytic oracle for `small_step_prep`/`advance_uv`; "uv delta > 0" is not enough.

The AC list should be changed before F7.A is judged complete.

## Q4 - Are we missing a cheaper first move?

Yes, but not the suggested `mu_save` carry hack.

An inline "carry `mu_save` across RK2/RK3" hack is the wrong cheap move because the first critical failure occurs at RK1/substep1 before RK2 exists, and because WRF does not carry RK1 `mu_save` across stages. That hack might hide or move the RK2 saved-state invariant, but it would not honestly test the architecture diagnosis.

The cheaper useful move is a one-day diagnostic spike:

- Patch only the RK1 acoustic entry path in a throwaway branch or scratch worktree.
- Give `advance_mu_t` a WRF-shaped RK1 small-step work state: `_1` fields from the physical origin, per-stage `*_save` from the candidate, `MUTS = MUB + MU_2`, `MU_2` work value initialized to zero, and coupled theta/u/v work arrays.
- Fix `dts_rk` for RK1 to `dt/3`.
- Optionally add a minimal `advance_uv` call using the existing pressure fields, but label it as a diagnostic, not F7.A completion.
- Re-run F6 combination `a` for one step and require the RK1/substep1 theta/mass failures to move.

That would answer in hours whether the first failure is mainly work-state ownership or whether missing `advance_uv`/pressure is already fatal. It is not production code, but it is a better risk reducer than starting a 2-3 day module rewrite blind.

## Q5 - Honest verdict

Score: **6/10**.

The plan is pointed in the right direction and bounded sprints are better than a mega-sprint, but F7.A as written can produce a misleading partial success: it confuses `mu_save` lifetime, defers pressure prep that `advance_uv` needs, omits `dts_rk`/first-failure ACs, and relies on weak proof gates.

If I were changing it, I would insert a one-day F7.0 diagnostic spike, correct the `_1` vs `*_save` language, move loop-entry `calc_p_rho(step=0)` or equivalent pressure-prep into F7.A, and harden AC4 to require the original F6 RK1/substep1 theta/mass violations to clear under combination `a`.

F7_CRITIQUE_COMPLETE
