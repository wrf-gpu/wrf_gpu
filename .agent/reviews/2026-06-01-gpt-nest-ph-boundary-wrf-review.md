# GPT-5.5 Review: Nested `ph_2` Boundary Handling in WRF

Date: 2026-06-01
Branch reviewed: `worker/opus/final-verdict`
Mode: read-only design review; no GPU run; no production-code edits.

## Verdict

WRF does not leave perturbation geopotential `ph_2` free at nested lateral boundaries. In pristine WRF, `ph_2` is part of the specified/relaxation lateral-boundary set for `specified` and `nested` domains. Its boundary tendencies are produced before the RK acoustic loop, consumed inside the acoustic `advance_w` small step through `ph_tend`, and the outer specified zone is updated each acoustic substep by the special mass-coupled `spec_bdyupdate_ph` routine. WRF also applies a final `spec_bdy_final` drift correction to `ph_2`, but that final correction is not the primary or sole ph boundary mechanism.

Therefore the failed JAX end-of-step hydrostatic ph overwrite/nudge is not WRF-faithful and its blow-up mechanism is consistent with WRF's source structure: it injects `ph` outside the coupled `w/ph` acoustic state.

The proposed direction, "add a ph boundary tendency inside the acoustic small step coupled with w," is WRF-faithful in concept only if it mirrors WRF's mass-coupled `ph_b*`/`ph_bt*` tendency path plus `spec_bdyupdate_ph`. It is not WRF-faithful if implemented as an ad hoc per-substep relaxation toward a freshly diagnosed hydrostatic `ph'` target without WRF's coupled boundary arrays, mass factors, specified-zone update, and staged target cadence.

## Findings

1. **Critical: WRF forces `ph_2` at nested boundaries.**
   `relax_bdy_dry` and `spec_bdy_dry` both include `ph`/`ph_tend` and `ph_bxs/ph_bxe/ph_bys/ph_bye` plus `ph_btxs/ph_btxe/ph_btys/ph_btye`. The calls are active under `(config_flags%specified .or. config_flags%nested)`, including nests.

2. **Critical: the ph boundary tendency is acoustic-loop state, not an end-step value repair.**
   `advance_w` advances the implicit `w` and `ph` equations together and includes `ph_tend` in the RHS before the tridiagonal `w` solve. After that solve, WRF updates `ph` from the solved `w`. The outer specified boundary is then mass-coupled-adjusted with `spec_bdyupdate_ph` every acoustic substep before `calc_p_rho` recomputes pressure.

3. **High: WRF's nest forcing path constructs ph boundary arrays and tendencies.**
   Online nesting calls `med_nest_force`, couples parent and child fields, interpolates parent data into the nest boundary, and explicitly calls `bdy_interp` for `grid%ph_2`, filling child `ph_b*` and `ph_bt*`.

4. **High: hydrostatic recomputation is WRF-justified at init/vertical-nesting/rebalance points, not as an arbitrary late correction.**
   Real initialization and the vertical-nesting force path recompute `ph_2` hydrostatically from mass/thermo fields. During integration, WRF still constrains boundary `ph_2` through the ph boundary tendency machinery.

5. **Medium: the current JAX comments around `force_geopotential=False` are materially wrong relative to pristine WRF.**
   The local comment says WRF does not independently overwrite nest geopotential from interpolated parent fields. That is too broad. WRF does not do a raw end-step decoupled overwrite as the failed attempt did, but it does carry `ph_2` in nested boundary arrays/tendencies and updates boundary `ph` inside the acoustic integration.

## Exact WRF Path

### 1. Boundary tendency construction includes `ph_2`

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_bc_em.F:161-178`

- `relax_bdy_dry` signature includes `ph_tendf`, `ph`, `ph_bxs/ph_bxe/ph_bys/ph_bye`, and `ph_btxs/ph_btxe/ph_btys/ph_btye`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_bc_em.F:274-290`

- WRF mass-weights `ph` with `mut` via `mass_weight(ph, mut, rfield, c1f, c2f, ...)`.
- It then calls `relax_bdytend_tile(rfield, ph_tendf, ph_b*, ph_bt*, 'h', ...)`.
- This means relaxation is applied to mass-coupled full-level geopotential, not to a decoupled post-step `ph'` value.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_bc_em.F:320-344`

- For nests, WRF also relaxes `w` with the same boundary machinery. This matters because `w` and `ph` are the coupled acoustic pair.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_bc_em.F:413-428`

- `spec_bdy_dry` signature includes `ph_tend` and the same `ph_b*`/`ph_bt*` boundary fields.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_bc_em.F:495-502`

- `spec_bdy_dry` calls `spec_bdytend(ph_tend, ph_b*, ph_bt*, 'h', ...)`.

`/home/enric/src/wrf_pristine/WRF/share/module_bc.F:1430-1491`

- `spec_bdytend` normalizes variable names and sets `ktf = kte` for variable `'h'`, so full-level `ph` is included through the top full level.

`/home/enric/src/wrf_pristine/WRF/share/module_bc.F:1492-1543`

- `spec_bdytend` writes `field_tend` directly from the side-specific boundary tendency arrays in the outer specified zone.

`/home/enric/src/wrf_pristine/WRF/share/module_bc.F:1221-1291`

- `relax_bdytend_core` also sets `ktf = kte` for `'h'`.

`/home/enric/src/wrf_pristine/WRF/share/module_bc.F:1293-1427`

- Relaxation-zone tendencies are added as `fcx * residual - gcx * residual_laplacian`, using `field_bdy + dtbc * field_bdy_tend - field`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:938-965`

- On `rk_step == 1`, WRF calls `relax_bdy_dry` when `specified` or `nested`.
- The actual call passes `ph_save` as `ph_tendf`, `grid%ph_2` as `ph`, and all `grid%ph_b*` / `grid%ph_bt*` arrays.

`/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:968-981`

- `rk_addtend_dry` merges `ph_tendf` into the RK-stage `ph_tend`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:983-1005`

- `spec_bdy_dry` is then called for `specified` or `nested`, setting the specified-zone `ph_tend` from the `ph_bt*` arrays.

Conclusion for question 1: **yes, WRF applies lateral-boundary treatment to `ph_2` at nested boundaries. `ph_2` is in both the relaxed and specified sets. It is not simply diagnosed or left to interior dynamics during integration.**

### 2. `ph_2` is advanced inside the acoustic `w/ph` step

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:125-168`

- On RK step 1, WRF copies `ph_2` to `ph_1` along with `w_1`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:268-277`

- `small_step_prep` saves `ph_save = ph_2` and converts `ph_2` into the small-step work delta `ph_1 - ph_2`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:1498-1518`

- Each acoustic substep calls `advance_w(... grid%ph_2, ph_save, grid%phb, ph_tend, ...)`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1178-1192`

- `advance_w` receives `ph`, `ph_1`, `phb`, and `ph_tend`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1262`

- The routine description states that `advance_w` advances the implicit `w` and geopotential equations.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1273-1282`

- For `specified` or `nested`, the lateral outer rows/columns are excluded from the interior implicit `advance_w` loops. They are handled by the boundary update path.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1312-1319`

- The ph RHS includes `ph_tend` inside the acoustic step:
  `rhs(i,k+1) = dts*(ph_tend(i,k+1,j) + .5*g*(1.-epssm)*w(...))`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1321-1368`

- WRF then adds vertical ph advection and the previous `ph` work value.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1401-1431`

- The `w` RHS uses the ph RHS and pressure/geopotential terms in the same implicit solve.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1433-1443`

- WRF performs the Thomas forward/back solve for `w`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1460-1464`

- WRF updates `ph` from the solved `w`:
  `ph = rhs + msfty*.5*dts*g*(1.+epssm)*w/(c1f*muts+c2f)`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_bc_em.F:17-25`

- `spec_bdyupdate_ph` is a special ph-specific specified-zone updater.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_bc_em.F:89-93`, `:108-112`, `:127-131`, `:145-149`

- `spec_bdyupdate_ph` updates boundary `field` with `field_tend`, `mu_tend`, `muts`, `c1/c2`, and `ph_save`. This is the mass-coupled correction that preserves consistency with changing `mu`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:1583-1597`

- After `advance_w` inside each acoustic small step, WRF calls `spec_bdyupdate_ph(ph_save, grid%ph_2, ph_tend, mu_tend, grid%muts, c1f, c2f, dts_rk, ...)` for `specified` or `nested`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:1598-1618`

- For `nested`, WRF also applies `spec_bdyupdate` to `grid%w_2` using `rw_tend`. For `specified`, it uses zero-gradient `w`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:1623-1636`

- Immediately after the ph/w boundary updates, WRF calls `calc_p_rho(... grid%ph_2, ...)` for the current acoustic iteration.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:397-403`

- At small-step finish, WRF reconstructs full variables and adds `ph_save` back to `ph_2`.

Conclusion for question 2: **if a ph boundary tendency exists, it enters the acoustic loop through `advance_w` and `spec_bdyupdate_ph`, not only as a large-step or end-step correction.**

### 3. Final boundary correction exists but is not the whole mechanism

`/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:4624-4626`

- WRF enters final boundary forcing for `specified` or `nested` after the RK work.

`/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:4687-4698`

- It calls `spec_bdy_final` for `grid%ph_2`.

`/home/enric/src/wrf_pristine/WRF/share/module_bc.F:2066-2086`

- `spec_bdy_final` says it forces boundary values to boundary-file values to avoid drift from roundoff.

`/home/enric/src/wrf_pristine/WRF/share/module_bc.F:2139-2143`

- For variable `'h'`, `mucouple` remains true and `msfcouple` is false.

`/home/enric/src/wrf_pristine/WRF/share/module_bc.F:2146-2158`, `:2164-2177`, `:2182-2193`, `:2199-2209`

- It computes `bfield = field_bdy + dtbc * field_bdy_tend` and writes `field = bfield / (c1*mu+c2)` for ph.

This final step is faithful only as the last drift correction on top of the in-loop ph forcing. It is not evidence that WRF uses a lone end-of-step hydrostatic overwrite.

### 4. Nest forcing and hydrostatic recomputation

`/home/enric/src/wrf_pristine/WRF/frame/module_integrate.F:409-430`

- Parent integration calls `med_nest_force(parent, nest)` and then recursively integrates the child nest over the parent time interval.

`/home/enric/src/wrf_pristine/WRF/share/mediation_force_domain.F:111-134`

- `med_force_domain` couples parent and nested domains before interpolation.

`/home/enric/src/wrf_pristine/WRF/dyn_em/couple_or_uncouple_em.F:270-286`

- Coupling multiplies `ph_2`, `w_2`, and `t_2` by the relevant mass factors. In particular, `grid%ph_2(i,k,j) = grid%ph_2(i,k,j) * mutf_2(i,k,j)`.

`/home/enric/src/wrf_pristine/WRF/share/mediation_force_domain.F:160-177`

- WRF calls `interp_domain_em_part1` and then `force_domain_em_part2`.

`/home/enric/src/wrf_pristine/WRF/external/RSL_LITE/force_domain_em_part2.F:141-148`

- In vertical refinement, the intermediate grid has parent horizontal resolution and nest vertical resolution. The code says the nest base state is recalculated.

`/home/enric/src/wrf_pristine/WRF/external/RSL_LITE/force_domain_em_part2.F:154-160`

- WRF uncouples `t_2` and moisture used to calculate `ph_2`.

`/home/enric/src/wrf_pristine/WRF/external/RSL_LITE/force_domain_em_part2.F:165-202`

- WRF recomputes base pressure, `t_init`, `alb`, and integrates base geopotential `phb` hydrostatically.

`/home/enric/src/wrf_pristine/WRF/external/RSL_LITE/force_domain_em_part2.F:205-246`

- WRF integrates hydrostatic pressure/inverse-density perturbation fields down from the top.

`/home/enric/src/wrf_pristine/WRF/external/RSL_LITE/force_domain_em_part2.F:248-257`

- For `hypsometric_opt == 1`, WRF recomputes `grid%ph_2` hydrostatically from mass and inverse density.

`/home/enric/src/wrf_pristine/WRF/external/RSL_LITE/force_domain_em_part2.F:263-275`

- For `hypsometric_opt == 2`, WRF computes total ph hydrostatically and subtracts `phb`.

`/home/enric/src/wrf_pristine/WRF/external/RSL_LITE/force_domain_em_part2.F:285-299`

- WRF then re-couples `ph_2`, `t_2`, and moisture before nest-boundary interpolation.

`/home/enric/src/wrf_pristine/WRF/inc/nest_forcedown_interp.inc:82-105`

- The forcedown include explicitly calls `bdy_interp` for `grid%ph_2`, filling `ngrid%ph_2`, `ngrid%ph_bxs/ph_bxe/ph_bys/ph_bye`, and `ngrid%ph_btxs/ph_btxe/ph_btys/ph_btye`.

`/home/enric/src/wrf_pristine/WRF/share/interp_fcn.F:2299-2320`

- `bdy_interp` takes boundary values and boundary tendencies; the comments identify starting values and tendencies.

`/home/enric/src/wrf_pristine/WRF/share/interp_fcn.F:2578-2615`

- `bdy_interp1` computes `bdy_t* = rdt * (interpolated_current - nfld)` and then stores `bdy_* = nfld`.

`/home/enric/src/wrf_pristine/WRF/inc/nest_interpdown_interp.inc:237-267`

- Same-vertical initial nest interpolation directly interpolates `ph_2` and `phb`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_initialize_real.F:3765-3774`

- Real initialization sets bottom total/base geopotential and initializes `ph_2`.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_initialize_real.F:3810-3818`

- Real initialization integrates base `phb` hydrostatically.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_initialize_real.F:3921-3958`

- It integrates hydrostatic pressure and inverse density down from the top.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_initialize_real.F:3962-3972`

- For `hypsometric_opt == 1`, real initialization computes perturbation `ph_2` hydrostatically.

`/home/enric/src/wrf_pristine/WRF/dyn_em/module_initialize_real.F:3973-3988`

- For `hypsometric_opt == 2`, it computes ph via the alternative hydrostatic equation and subtracts `phb`.

I did not find a `med_nest_egress` symbol in this WRF tree. The relevant online-nesting path is `med_nest_force` -> `med_force_domain` -> generated RSL_LITE force/interp code. Feedback is child-to-parent after integration and does not replace the parent-to-child lateral boundary constraint.

Conclusion for question 3: **WRF recomputes child/intermediate `ph_2` hydrostatically in real init and vertical-nesting/rebalance contexts. During integration, child boundary `ph_2` is constrained by parent/nest boundary arrays and tendencies, including `ph_b*`/`ph_bt*`, inside the acoustic boundary path.**

## Assessment of Proposed JAX Fix

Proposed: add a `ph'` boundary-zone relaxation tendency inside the acoustic small step coupled with `w`, target = verified hydrostatic inverse of `calc_p_rho_phi`, mirroring existing `apply_normal_bdy_work` for normal momentum.

This is directionally right because it moves the ph correction into the only place WRF can safely accept it: the `w/ph` acoustic update. It is not enough to say "inside the loop" though. The WRF-faithful object is not a generic hydrostatic value nudge; it is a mass-coupled ph boundary tendency plus specified-zone update:

- Build stage/cadence boundary arrays equivalent to `ph_bxs/ph_bxe/ph_bys/ph_bye` and `ph_btxs/ph_btxe/ph_btys/ph_btye`.
- Treat those as coupled ph boundary values/tendencies, consistent with WRF's `couple_or_uncouple_em` and `mass_weight(ph, mut, ..., c1f, c2f)`.
- Add relaxation-zone contribution to `ph_tendf`/`ph_tend` with the WRF `fcx/gcx` residual and full-level `'h'` extent.
- Pass `ph_tend` into `advance_w` every acoustic substep.
- After `advance_w`, apply the WRF `spec_bdyupdate_ph` mass-coupled formula in the specified zone using `ph_save`, `mu_tend`, current `muts`, `c1f/c2f`, and `dts_rk`.
- For nested domains, update `w` boundary state in the same acoustic cadence; WRF does this for nests.
- Recompute `calc_p_rho` after the ph/w boundary updates, as WRF does.

Using a hydrostatic inverse target is defensible only as the source of the boundary target when the raw parent-interpolated `ph` leaves are known inconsistent with the child mass/thermo column. WRF itself just carries `ph_2` in the nesting boundary machinery, and in vertical-nesting/rebalance paths it recomputes hydrostatic `ph_2` before coupling/interpolation. The faithful adaptation is therefore: **derive the hydrostatic ph target at boundary-ingest/force-cadence time if needed, then feed it through WRF's ph boundary tendency machinery.** Do not recompute a moving target from the mutable acoustic state every substep.

## Pitfalls

- **Mass coupling and units.** WRF relaxes a mass-coupled ph field in `relax_bdy_dry`, and `spec_bdyupdate_ph` divides by `c1f*muts+c2f` while correcting for `mu_tend` and `ph_save`. A decoupled `ph'` tendency with the right shape but wrong coupling can still excite pressure errors.

- **Outer-row ownership.** `advance_w` excludes outer lateral rows/columns for `specified`/`nested`; `spec_bdyupdate_ph` owns that specified zone. An implementation that both advances and overwrites the same cells will not match WRF.

- **Full-level vertical extent.** For variable `'h'`, WRF sets `ktf = kte`, so the top full level is included. Do not apply mass-level extents.

- **`w` coupling.** For nests, WRF updates `w` boundary state alongside `ph`. A ph-only correction can leave acoustic memory inconsistent and recreate the resonance seen in the failed end-step attempts.

- **`ph_save` and acoustic work arrays.** WRF's `ph_2` inside the loop is a work delta from `small_step_prep`, not the final full perturbation ph. The update formula depends on `ph_save`.

- **Divergence damping and pressure memory.** `calc_p_rho` follows the ph/w boundary update every substep and updates the `pm1`/`smdiv` pressure memory. Abrupt ph forcing outside this order can inject a persistent pressure-mode error.

- **Relaxation strength.** WRF's `fcx/gcx` additive tendency is calibrated for mass-coupled consistent boundary fields. The existing JAX `apply_normal_bdy_work` uses a stronger convex blend for a documented decoupled side-history approximation. Copying that strength to ph would not be WRF-faithful by default and is risky because ph directly feeds pressure and the implicit w solve.

- **Hydrostatic target consistency.** The target must use the same base fields, terrain anchor, `c1f/c2f`, `c1h/c2h`, moisture convention, top treatment, and hypsometric option as the dycore diagnostic. A bottom or top off-by-one in ph creates column-integrated pressure error.

- **Target cadence.** WRF boundary tendencies are formed from boundary start value plus time tendency over the parent/boundary cadence. Re-deriving target from live child state every acoustic substep creates a moving attractor, not a WRF boundary tendency.

## Recommended Faithful Approach

Implement the smallest WRF-shaped ph boundary path, not another end-of-step fix:

1. Stage `ph_b*`/`ph_bt*` equivalents for the d03 nest at the same point where the normal-momentum boundary work targets are staged. If raw parent ph is inconsistent, compute a hydrostatic ph target from the forced boundary mass/thermo fields at boundary-ingest cadence, then store it as the boundary target/tendency.

2. Add a ph boundary tendency field to the acoustic state and merge it into the stage `ph_tend` before the acoustic scan, matching WRF's `relax_bdy_dry` + `rk_addtend_dry` + `spec_bdy_dry` order.

3. Extend the acoustic substep with a WRF `spec_bdyupdate_ph` equivalent after `advance_w_wrf` and before `calc_p_rho_step`.

4. Keep the final `spec_bdy_final`-like ph correction optional and secondary. It should only remove drift after the in-loop path exists; it must not be the main correction.

5. Validate with a WRF fixture, not JAX-vs-JAX self-comparison: boundary-ring ph, w, `calc_p_rho` pressure, `pm1`, and T2/PSFC response should be compared at substep or short-interval cadence. A first acceptance gate should show no nonfinite `w/ph/p` growth through hour 1 before pursuing 6 h or 24 h d03 skill.

Simpler alternatives:

- **Constrain only at interp/init:** not sufficient. WRF keeps ph constrained at the lateral boundary during integration. The observed free-boundary d03 drift is exactly the failure mode this leaves open.

- **Relax only `mu/theta` and let ph follow hydrostatically:** not WRF-faithful for the nonhydrostatic EM integration. `ph` is a prognostic acoustic variable in WRF and is explicitly in the boundary tendency set. This can be a diagnostic experiment, not the final faithful fix.

- **Re-reference diagnostic pressure/T2 only:** may be a temporary symptom workaround, but it does not correct dycore state and should not close this boundary-faithfulness issue.

## Commands Run

All source-inspection commands were run read-only with `taskset -c 0-3` where applicable. No GPU commands were run.

- `sed -n` reads of `PROJECT_CONSTITUTION.md`, `AGENTS.md`, the three requested RCA/context reviews, relevant sprint contracts, and project-local review/physics skills.
- `git branch --show-current`
- `git status --short --branch`
- `rg`/`nl -ba | sed -n` reads in `/home/enric/src/wrf_pristine/WRF` for:
  - `dyn_em/module_bc_em.F`
  - `dyn_em/solve_em.F`
  - `dyn_em/module_small_step_em.F`
  - `dyn_em/couple_or_uncouple_em.F`
  - `share/module_bc.F`
  - `share/interp_fcn.F`
  - `share/mediation_force_domain.F`
  - `frame/module_integrate.F`
  - `external/RSL_LITE/force_domain_em_part2.F`
  - `inc/nest_forcedown_interp.inc`
  - `inc/nest_interpdown_interp.inc`
  - `dyn_em/module_initialize_real.F`
- Read-only JAX orientation reads for:
  - `src/gpuwrf/coupling/boundary_apply.py`
  - `src/gpuwrf/dynamics/core/acoustic.py`
  - `src/gpuwrf/runtime/operational_mode.py`

## Proof Objects Produced

- This review: `.agent/reviews/2026-06-01-gpt-nest-ph-boundary-wrf-review.md`

No executable proof object was produced because the requested task was read-only source design review with no GPU run.

## Unresolved Risks

- The precise hydrostatic target formula in JAX still needs a line-by-line consistency audit against the active dycore's `calc_p_rho_phi` inverse, including bottom/top indexing, base-state ph, moisture convention, and `hypsometric_opt`.
- The existing normal-momentum boundary path uses a calibrated stronger blend for decoupled side-history replay. Reusing that abstraction for ph without restoring WRF's mass-coupled tendency semantics could stabilize the wrong thing or destabilize the acoustic pair.
- The pristine source tree contains local instrumentation comments in `solve_em.F` after the cited core small-step area, but the cited boundary and acoustic code paths are unaffected.

## Next Decision Needed

Opus should decide whether to implement the WRF-shaped ph boundary-tendency path as a dycore change now, or defer it and use a clearly labeled diagnostic-pressure workaround for T2 only. The faithful dycore fix is the in-loop mass-coupled ph boundary tendency plus `spec_bdyupdate_ph`, not another end-of-step hydrostatic overwrite.
