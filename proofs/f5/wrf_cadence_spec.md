# F5 WRF Cadence Spec

This proof object maps the WRF RK3, acoustic small-step, and dry/scalar update cadence onto the current JAX runtime. All WRF behavioral claims below include `file:line` references from the inspected WRF source.

## Section 1 - Top-level RK3+small-step driver (from solve_em.F)

### 1.1 Runge-Kutta loop shape

- `solve_em.F` describes the RK loop as a three-stage corrector where advection is evaluated at the start of the time step and re-evaluated on each corrector, while physics is evaluated on RK1 and saved for reuse on each RK pass (`solve_em.F:1423-1438`).
- The top-level RK loop is `Runge_Kutta_loop: DO rk_step = 1, rk_order` (`solve_em.F:1447`).
- For `rk_step == 1`, WRF sets `dt_rk = dt / 3.0`, `dts_rk = dt_rk`, and `number_of_small_timesteps = 1` (`solve_em.F:1472-1475`).
- For `rk_step == 2`, WRF sets `dt_rk = 0.5 * dt`, `dts_rk = dts`, and `number_of_small_timesteps = num_sound_steps / 2` (`solve_em.F:1476-1479`).
- For `rk_step == 3`, WRF sets `dt_rk = dt`, `dts_rk = dts`, and `number_of_small_timesteps = num_sound_steps` (`solve_em.F:1480-1483`).

### 1.2 RK1-only physics and stored tendencies

- WRF enters an RK1-only block when `rk_step == 1` (`solve_em.F:1693`).
- The RK1-only comments identify the non-timesplit physics tendencies as RK1 work that is saved into tendency arrays for later RK passes (`solve_em.F:1699-1709`).
- `first_rk_step_part1` writes dry physics tendency arrays including `ru_tendf`, `rv_tendf`, `rw_tendf`, `t_tendf`, `ph_tendf`, and `mu_tendf` (`solve_em.F:1715-1724`).
- `first_rk_step_part2` continues the RK1-only dry tendency path using the same dry tendency arrays (`solve_em.F:1772-1780`).
- The RK1-only physics block ends before the per-stage `rk_tendency` call (`solve_em.F:1830-1848`).

### 1.3 Per-RK dry dynamics and fixed physics addition

- WRF calls `rk_tendency` every RK stage after the RK1-only physics block, passing dry state, mass, save, and tendency arrays (`solve_em.F:1837-1864`).
- The `rk_tendency` call receives the dry mass fields `mu_1`, `mu_2`, `mut`, `muu`, `muv`, and `mub` (`solve_em.F:1849-1859`).
- After per-stage dry tendencies are formed, WRF calls `rk_addtend_dry` for dry prognostic fields (`solve_em.F:2130-2143`).
- `module_em.F` states that `rk_addtend_dry` combines physics tendencies computed on the first RK substep with dynamics tendencies computed on each RK substep (`module_em.F:1711-1716`).
- In `rk_addtend_dry`, WRF adds RK1-stored u/v/w/ph/theta/mu tendency arrays into the current per-stage dry tendencies, with map-factor and mass-coupling rules that differ by field (`module_em.F:1735-1782`).

### 1.4 Boundary placement inside the RK loop

- After `rk_addtend_dry`, WRF applies dry specified/nested boundary tendencies inside the RK stage through `spec_bdy_dry` (`solve_em.F:2188-2210`).
- After `advance_uv`, WRF applies specified/nested boundary work for u and v inside the acoustic small-step path (`solve_em.F:3225-3234`, `solve_em.F:3267-3287`).
- After `advance_mu_t`, WRF applies specified/nested boundary work for theta, mu, and muts inside the acoustic small-step path (`solve_em.F:3621-3635`, `solve_em.F:3658-3679`).
- After `advance_w`, WRF applies specified/nested boundary work for geopotential and w inside the acoustic small-step path (`solve_em.F:3993-4010`, `solve_em.F:4102-4124`).
- After the RK loop completes, WRF applies final specified boundary updates for u, v, w, theta, geopotential, and mu (`solve_em.F:7922-7999`).

### 1.5 Acoustic small-step envelope and scalar placement

- Before entering the acoustic small-step loop, WRF comments list the small-step order as `small_step_prep`, `calc_coef_w`, then repeated `advance_uv`, `advance_mu_t`, `advance_w`, and `calc_p_rho`, followed by `small_step_finish` (`solve_em.F:2371-2389`).
- WRF calls `small_step_prep` before the acoustic loop in the CPU and GPU paths (`solve_em.F:2544-2562`, `solve_em.F:2572-2588`, `solve_em.F:2605-2623`).
- WRF calls `calc_p_rho(... step=0)` before acoustic substeps begin (`solve_em.F:2628-2640`, `solve_em.F:2645-2655`, `solve_em.F:2658-2670`).
- WRF calls `calc_coef_w` before acoustic substeps begin (`solve_em.F:2676-2687`, `solve_em.F:2705-2716`).
- The acoustic loop is `small_steps : DO iteration = 1, number_of_small_timesteps` (`solve_em.F:3065`).
- Within each acoustic substep, WRF calls `advance_uv` first (`solve_em.F:3088-3105`, `solve_em.F:3135-3152`).
- Within each acoustic substep, WRF calls `advance_mu_t` after u/v advancement and u/v boundary handling (`solve_em.F:3398-3415`, `solve_em.F:3435-3452`).
- Within each acoustic substep, WRF calls `advance_w` after theta/mass advancement and theta/mass boundary handling (`solve_em.F:3837-3855`, `solve_em.F:3880-3898`).
- WRF accumulates small-step averaged fluxes with `sumflux` after `advance_w` (`solve_em.F:4048-4061`, `solve_em.F:4080-4093`).
- WRF calls `calc_p_rho(... step=iteration)` after `advance_w` and related boundary work in each acoustic substep (`solve_em.F:4164-4176`, `solve_em.F:4181-4191`, `solve_em.F:4194-4206`).
- After all acoustic substeps, WRF calls `calc_mu_uv_1` to update `muus` and `muvs` from `muts` (`solve_em.F:4383-4406`).
- After all acoustic substeps, WRF calls `small_step_finish` (`solve_em.F:4412-4428`, `solve_em.F:4430-4445`, `solve_em.F:4448-4462`).
- WRF performs scalar mixing and scalar tendency work after the dry acoustic path using `rk_scalar_tend` (`solve_em.F:5060-5061`, `solve_em.F:5161-5186`).
- WRF applies scalar boundary tendencies only on RK1 in the inspected branch (`solve_em.F:5207-5210`).
- WRF ends the RK loop after scalar and related per-stage work (`solve_em.F:6766`).

## Section 2 - WRF subroutine responsibilities and ordering

### 2.1 Stage preparation

| WRF routine | Responsibility | Required cadence |
| --- | --- | --- |
| `rk_step_prep` | Computes full dry mass, face masses, coupled momentum, omega, moisture coefficients, inverse density, and geopotential diagnostics before RK stage work (`module_em.F:137-155`, `module_em.F:184-225`). | Before per-stage `rk_tendency` in the top-level RK stage (`solve_em.F:1837-1864`). |
| `rk_tendency` | Zeros dry tendency/save arrays, computes stage dry advection, pressure-gradient, buoyancy, damping, and selected RK1 diffusion/relaxation terms (`module_em.F:580-638`, `module_em.F:855-1143`, `module_em.F:1226-1388`, `module_em.F:1488-1641`). | Once per RK stage after RK1 physics and before `rk_addtend_dry` (`solve_em.F:1837-1864`, `solve_em.F:2130-2143`). |
| `rk_addtend_dry` | Adds RK1-stored fixed physics tendencies to current per-stage dry tendencies and applies field-specific map/mass coupling (`module_em.F:1711-1782`). | Once per RK stage after `rk_tendency` (`solve_em.F:2130-2143`). |

### 2.2 Small-step preparation

| WRF routine | Responsibility | Required cadence |
| --- | --- | --- |
| `small_step_prep` | On RK1, copies current physical perturbation prognostics from `_2` to `_1`; every RK stage, builds small-step mass states, saves current `_2` values, and converts `_2` arrays into coupled perturbation work arrays (`module_small_step_em.F:125-190`, `module_small_step_em.F:196-215`, `module_small_step_em.F:230-285`). | Once per RK stage before `calc_p_rho(step=0)`, `calc_coef_w`, and the acoustic substep loop (`solve_em.F:2544-2623`). |
| `calc_p_rho` | Computes perturbation inverse density, perturbation pressure, geopotential-related state, and divergence damping pressure memory (`module_small_step_em.F:492-504`, `module_small_step_em.F:522-563`). | Once before acoustic substeps with `step=0`, then once after each acoustic substep with `step=iteration` (`solve_em.F:2628-2670`, `solve_em.F:4164-4206`). |
| `calc_coef_w` | Computes implicit vertical momentum/geopotential coefficients using dry mass, hydrostatic coefficients, and `c2a` (`module_small_step_em.F:608-649`). | Once per RK stage before the acoustic substep loop (`solve_em.F:2676-2716`). |

### 2.3 Acoustic substep body

| WRF routine | Responsibility | Required cadence |
| --- | --- | --- |
| `advance_uv` | Advances coupled perturbation u/v work arrays using large-step tendencies and small-step pressure-gradient terms (`module_small_step_em.F:654-729`, `module_small_step_em.F:802-942`). | First operation inside each acoustic substep (`solve_em.F:3065`, `solve_em.F:3088-3152`). |
| `advance_mu_t` | Advances perturbation dry mass, total small-step mass `muts`, weighted mass `muave`, `mudf`, small-step omega `ww`, and coupled theta work arrays (`module_small_step_em.F:969-1043`, `module_small_step_em.F:1071-1171`). | After `advance_uv` and u/v boundary work in each acoustic substep (`solve_em.F:3398-3452`). |
| `advance_w` | Advances implicit vertical velocity and geopotential equations with terrain, pressure, density, mass, theta, damping, and Thomas-solve terms (`module_small_step_em.F:1178-1271`, `module_small_step_em.F:1315-1584`). | After `advance_mu_t` and theta/mass boundary work in each acoustic substep (`solve_em.F:3837-3898`). |
| `sumflux` | Accumulates small-step averaged mass/momentum fluxes after w/geopotential work (`solve_em.F:4048-4061`, `solve_em.F:4080-4093`). | After `advance_w` inside each acoustic substep. |
| `calc_p_rho(step=iteration)` | Refreshes perturbation pressure, inverse density, and divergence damping memory from the updated small-step work state (`module_small_step_em.F:492-563`). | Last required diagnostic step after each acoustic substep and related boundary work (`solve_em.F:4164-4206`). |

### 2.4 Stage finish and scalar work

| WRF routine | Responsibility | Required cadence |
| --- | --- | --- |
| `calc_mu_uv_1` | Recomputes full u/v face dry masses from `muts` (`module_big_step_utilities_em.F:209-211`, `module_big_step_utilities_em.F:221-234`, `module_big_step_utilities_em.F:274-287`). | Once after all acoustic substeps and before `small_step_finish` (`solve_em.F:4383-4406`). |
| `small_step_finish` | Reconstructs physical perturbation prognostic variables from coupled small-step work arrays, restores saved geopotential and omega state, handles final-RK theta diabatic treatment, and restores physical perturbation dry mass (`module_small_step_em.F:364-430`). | Once after all acoustic substeps in an RK stage (`solve_em.F:4412-4462`). |
| `rk_scalar_tend` | Computes scalar tendencies after dry acoustic dynamics, using averaged fluxes and stage mass fields (`solve_em.F:5060-5061`, `solve_em.F:5161-5186`). | After `small_step_finish` inside the RK stage. |

## Section 3 - Field lifetime and representation across the cadence

| Boundary | WRF representation and lifetime |
| --- | --- |
| Before `small_step_prep` | `u_2`, `v_2`, `w_2`, `t_2`, `ph_2`, and `mu_2` are the current physical perturbation prognostic fields entering the RK stage; on RK1 WRF copies those fields into `u_1`, `v_1`, `w_1`, `t_1`, `ph_1`, and `mu_1` (`module_small_step_em.F:125-167`). |
| RK1 `small_step_prep` mass setup | On RK1, WRF sets `muts = mub + mu_2`, copies `muu` and `muv` into `muus` and `muvs`, saves `mu_2` into `mu_save`, and then sets `mu_2 = 0.0` for the small-step work variable (`module_small_step_em.F:172-190`). |
| RK2/RK3 `small_step_prep` mass setup | On later RK stages, WRF sets `muts = mub + mu_1`, computes `muus` and `muvs` from `mub + mu_1`, saves `mu_2` into `mu_save`, and replaces `mu_2` with `mu_1 - mu_2` as the small-step work variable (`module_small_step_em.F:196-215`). |
| Coupled u/v/theta/w/geopotential work state | WRF saves current `u_2`, `v_2`, `t_2`, `w_2`, and `ph_2`, then rewrites those arrays into coupled perturbation work arrays before the acoustic loop (`module_small_step_em.F:241-276`). |
| `c2a` lifetime | WRF computes `c2a` during `small_step_prep` from pressure, base pressure, inverse density, and thermodynamic constants (`module_small_step_em.F:230-234`), then `calc_coef_w` consumes `c2a` for vertical implicit coefficients (`module_small_step_em.F:624-649`). |
| `ww_save` lifetime | WRF saves the pre-small-step `ww` value in `ww_save` during `small_step_prep` (`module_small_step_em.F:285`), then `small_step_finish` restores `ww` from `ww_save` (`module_small_step_em.F:401-403`). |
| During `advance_uv` | WRF advances coupled perturbation u/v arrays with large-step dry tendencies and small-step pressure-gradient terms; `advance_uv` comments identify `ph`, `alt`, `p`, `al`, and `pb` as not coupled, while `muu`, `muv`, and `mu` provide mass coupling (`module_small_step_em.F:727-729`, `module_small_step_em.F:802-831`, `module_small_step_em.F:854-942`). |
| During `advance_mu_t` | WRF advances `mu` as the perturbation dry-mass work array, computes `muts` as total small-step dry mass, computes `muave` as a weighted mass state, updates `mudf`, updates `ww`, and advances coupled theta work arrays (`module_small_step_em.F:1071-1171`). |
| During `advance_w` | WRF treats the u/v/w arrays sent to the vertical solve as coupled variables, uses `muave`, `muts`, and `mut` in the implicit w/geopotential equations, and updates `ph` from the solved w and RHS (`module_small_step_em.F:1323-1331`, `module_small_step_em.F:1341-1395`, `module_small_step_em.F:1477-1584`). |
| After all substeps | WRF recomputes face masses from `muts` with `calc_mu_uv_1` and then reconstructs physical perturbation prognostic variables in `small_step_finish` (`solve_em.F:4383-4462`, `module_small_step_em.F:364-430`). |
| After `small_step_finish` | WRF restores `mu_2` by adding back `mu_save`, so `mu_2` returns to a physical perturbation mass representation (`module_small_step_em.F:430`). |

## Section 4 - Coupling, pressure, and tendency semantics that the rewrite must preserve

### 4.1 Dry mass and map-factor coupling

- WRF computes full dry mass at u/v faces in `calc_mu_uv` from base dry mass plus perturbation dry mass (`module_big_step_utilities_em.F:51-53`, `module_big_step_utilities_em.F:64-79`, `module_big_step_utilities_em.F:125-131`).
- WRF couples momentum by multiplying velocities by full dry mass and applying map factors: u uses `(c1h*muu+c2h)/msfu`, v uses `(c1h*muv+c2h)*msfv_inv`, and w uses `(c1f*mut+c2f)/msft` (`module_big_step_utilities_em.F:359-394`).
- WRF's generic `couple` helper couples variables with dry-air column mass, with u, v, w, theta/geopotential, and scalar branches using field-specific locations and map factors (`module_big_step_utilities_em.F:555-628`).
- WRF's `decouple` helper reverses dry-mass coupling for u, v, w, theta, and scalar variables with field-specific mass locations (`module_big_step_utilities_em.F:4791-4845`).

### 4.2 Pressure, inverse density, and geopotential cadence

- WRF computes perturbation inverse density and pressure in `calc_p_rho` from geopotential, dry mass, theta, and stage/work fields (`module_small_step_em.F:492-504`, `module_small_step_em.F:522-528`).
- WRF seeds pressure memory when `step == 0` and applies divergence damping pressure correction when `step > 0` (`module_small_step_em.F:548-563`).
- WRF advances geopotential inside `advance_w` from phi-equation RHS terms and the implicit w solve, not as a separate post-hoc diagnostic (`module_small_step_em.F:1315-1395`, `module_small_step_em.F:1477-1584`).
- WRF's `save_ph_mu` copies `ph_2` and `mu_2` to `ph_1` and `mu_1` on RK1, then saves and converts `ph_2` and `mu_2` into perturbation work arrays (`module_small_step_em.F:1997-2033`).
- WRF's `restore_ph_mu` restores saved geopotential and dry-mass perturbation values after temporary work-array use (`module_small_step_em.F:2074-2081`).

### 4.3 Physics and scalar coupling

- WRF mass-couples physics tendencies because the equations are flux-form while physics tendencies are uncoupled (`module_em.F:4419-4421`).
- WRF multiplies radiation theta tendency by `c1*mut+c2` (`module_em.F:4433-4439`).
- WRF multiplies cumulus u, v, theta, and qv tendencies by dry mass terms (`module_em.F:4452-4455`).
- WRF multiplies PBL u, v, and theta tendencies by dry mass terms (`module_em.F:4576-4578`).
- WRF couples FDDA u, v, and theta tendencies with `muu`, `muv`, and `mut` (`module_em.F:4626-4648`).
- WRF couples scalar and tracer tendencies with dry mass (`module_em.F:4670-4684`).
- WRF includes diabatic heating in `rk_addtend_dry` through the theta tendency expression that uses `h_diabatic` and dry mass (`module_em.F:1770-1773`).
- WRF applies final-RK theta handling in `small_step_finish` by subtracting a diabatic heating term before decoupling theta (`module_small_step_em.F:418-422`).

## Section 5 - JAX gap map and concrete rewrite changes

The current JAX runtime has useful pieces, but it does not yet preserve the WRF cadence above. The rewrite sprint should treat the following as concrete build items.

| WRF requirement | Current JAX location | Gap | Concrete JAX change |
| --- | --- | --- | --- |
| RK stage sizes are 1, half, and full acoustic counts with `dts_rk = dt/3` on RK1 and `dts` on later stages (`solve_em.F:1472-1483`). | `_rk_scan_step` uses factors and substep counts, but `_acoustic_scan` deletes `dt_stage` and uses `namelist.dt_s / acoustic_substeps` for every stage (`src/gpuwrf/runtime/operational_mode.py:563-620`). | RK1 small-step time step is not represented as WRF `dts_rk = dt/3`. | Add an explicit RK stage descriptor carrying `dt_rk`, `dts_rk`, and `number_of_small_timesteps`, and pass `dts_rk` into every acoustic substep. |
| Physics tendencies are computed on RK1 and reused by every RK pass (`solve_em.F:1693-1780`, `module_em.F:1711-1716`). | Physics adapters run after `_rk_scan_step` in `_physics_boundary_step_with_limiter_diagnostics` (`src/gpuwrf/runtime/operational_mode.py:721-771`). | Physics cadence is outside the WRF RK loop. | Move physics tendency production into an RK1 tendency bundle and feed the fixed bundle into each per-stage dry tendency addition. |
| `rk_tendency` recomputes stage dry dynamics each RK pass, including advection, pressure-gradient, geopotential, buoyancy, damping, and selected RK1 diffusion/relaxation terms (`solve_em.F:1837-1864`, `module_em.F:855-1143`, `module_em.F:1226-1388`, `module_em.F:1488-1641`). | `_rk_scan_step` combines `compute_advection_tendencies` and `_horizontal_pressure_gradient_tendencies` (`src/gpuwrf/runtime/operational_mode.py:587-620`). | The JAX advection is reduced, periodic, and not the WRF flux-form mass-coupled tendency path (`src/gpuwrf/dynamics/advection.py:1-18`, `src/gpuwrf/dynamics/advection.py:39-113`, `src/gpuwrf/dynamics/advection.py:162-238`). | Replace reduced advection with WRF-shaped dry tendency construction, including `rhs_ph`, buoyancy, damping, and RK1-only terms behind feature flags. |
| `rk_addtend_dry` adds fixed RK1 physics tendencies to stage dynamics with field-specific map and mass coupling (`module_em.F:1711-1782`). | No direct JAX equivalent is present in the operational scan (`src/gpuwrf/runtime/operational_mode.py:587-620`). | Per-stage fixed-tendency addition is missing. | Implement a dry tendency merger that mirrors `rk_addtend_dry` for u, v, w, theta, ph, and mu arrays. |
| `small_step_prep` saves physical perturbation prognostics and converts `_2` arrays into coupled perturbation work arrays before the acoustic loop (`module_small_step_em.F:125-285`). | `_with_save_family` snapshots current state and resets `muave`/`muts` in a simplified way (`src/gpuwrf/runtime/operational_mode.py:395-411`); `_mass_couple_theta_before_advance` handles only theta coupling (`src/gpuwrf/dynamics/core/acoustic.py:177-190`). | Save-family and work-array semantics are not WRF-equivalent. | Build a `small_step_prep_wrf` state transform that updates `_1`, save arrays, `muts`, `muus`, `muvs`, `mu_save`, `ww_save`, `c2a`, and coupled u/v/w/theta/ph/mu work arrays. |
| `calc_p_rho(step=0)` runs before substeps, and `calc_p_rho(step=iteration)` runs after each substep (`solve_em.F:2628-2670`, `solve_em.F:4164-4206`, `module_small_step_em.F:492-563`). | `diagnose_pressure_al_alt` can be called, but the operational acoustic core also has `_diagnose_pressure` as a simplified stub (`src/gpuwrf/dynamics/acoustic_wrf.py:232-261`, `src/gpuwrf/dynamics/core/acoustic.py:193-215`). | Pressure, inverse-density, and divergence damping memory do not match WRF cadence. | Implement `calc_p_rho_wrf` with `step` and pressure-memory state, call it before the acoustic scan and after every substep. |
| `calc_coef_w` uses WRF `c2a` and dry-mass fields before the acoustic loop (`solve_em.F:2676-2716`, `module_small_step_em.F:608-649`). | `_operational_acoustic_substep_core` calls `calc_coef_w_wrf_coefficients` per substep without passing real `c2a`, so the helper defaults `c2a` to ones (`src/gpuwrf/runtime/operational_mode.py:532-560`, `src/gpuwrf/dynamics/acoustic_wrf.py:614-676`). | Vertical implicit coefficients are computed at the wrong cadence and with incomplete thermodynamic input. | Compute coefficients once per RK stage from `small_step_prep` outputs and pass them through the acoustic scan. |
| `advance_uv` is the first acoustic substep operation and includes large-step tendencies plus small-step pressure-gradient terms (`solve_em.F:3088-3152`, `module_small_step_em.F:654-942`). | The operational acoustic core calls `advance_mu_t_core` first and has no WRF `advance_uv` call in `acoustic_substep_core` (`src/gpuwrf/dynamics/core/acoustic.py:218-272`). | Acoustic order is wrong and u/v pressure-gradient substep work is missing. | Implement `advance_uv_wrf` and call it before `advance_mu_t_wrf` in every acoustic substep. |
| `advance_mu_t` updates `mu`, `muts`, `muave`, `mudf`, `ww`, and theta work arrays after u/v work (`solve_em.F:3398-3452`, `module_small_step_em.F:969-1171`). | `advance_mu_t_wrf` exists and is called through `advance_mu_t_core` (`src/gpuwrf/dynamics/mu_t_advance.py:68-161`, `src/gpuwrf/dynamics/core/acoustic.py:157-160`). | The local kernel is useful, but the caller feeds simplified prep/save state and runs it before missing `advance_uv` (`src/gpuwrf/runtime/operational_mode.py:443-529`, `src/gpuwrf/dynamics/core/acoustic.py:218-272`). | Keep the kernel, but rewire inputs from WRF-equivalent `small_step_prep_wrf` and place it after `advance_uv_wrf`. |
| `advance_w` advances implicit w and geopotential equations with full WRF RHS and Thomas solve (`solve_em.F:3837-3898`, `module_small_step_em.F:1178-1584`). | `w_solve_core`/Thomas pieces exist, but `_advance_geopotential` and `_ph_tend_increment` are simplified stubs in the operational acoustic core (`src/gpuwrf/dynamics/tridiag_solve.py:13-55`, `src/gpuwrf/dynamics/core/acoustic.py:193-215`, `src/gpuwrf/dynamics/core/acoustic.py:254-272`). | Full WRF `advance_w` RHS, pressure-density coupling, terrain lower boundary, damping, and geopotential update are missing. | Implement `advance_w_wrf` around the existing Thomas solver, with WRF RHS assembly and `ph_tend` handling. |
| WRF accumulates `sumflux` after `advance_w` and runs scalar tendency work after dry acoustic dynamics (`solve_em.F:4048-4093`, `solve_em.F:5060-5186`). | The operational runtime does not carry WRF `ru_m`, `rv_m`, and `ww_m` flux accumulation into a scalar pass in the RK stage (`src/gpuwrf/runtime/operational_state.py:31-59`, `src/gpuwrf/runtime/operational_mode.py:587-620`). | Scalar cadence and small-step averaged flux dependencies are absent. | Add flux accumulators to operational carry, update them after `advance_w_wrf`, and run WRF-shaped scalar tendencies after `small_step_finish_wrf`. |
| Boundary updates occur after dry tendency addition, after each acoustic variable group, and at final RK completion (`solve_em.F:2188-2210`, `solve_em.F:3225-3287`, `solve_em.F:3621-3679`, `solve_em.F:3993-4124`, `solve_em.F:7922-7999`). | Boundary handling currently occurs outside the RK/acoustic cadence in `_physics_boundary_step_with_limiter_diagnostics` (`src/gpuwrf/runtime/operational_mode.py:721-771`). | Boundary cadence is not WRF-equivalent. | Introduce boundary hooks at WRF cadence points, initially no-op for periodic tests and active for specified/nested boundary fixtures. |
| `small_step_finish` reconstructs physical perturbation fields and restores `mu_2`, `ph`, and `ww` after the acoustic loop (`solve_em.F:4412-4462`, `module_small_step_em.F:364-430`). | `_carry_from_acoustic_core` stores current `next_state` into save arrays and writes `mu_save=acoustic.mu`, which is not the WRF finish/restore sequence (`src/gpuwrf/runtime/operational_mode.py:489-529`). | Finish-state lifetime and save arrays are wrong. | Implement `small_step_finish_wrf` as the only exit from the acoustic work representation back to physical perturbation prognostics. |

## Section 6 - Direct answers for the rewrite sprint

### 6.1 When are stage-start fields saved?

- On RK1, WRF saves the physical perturbation stage-start fields by copying `u_2`, `v_2`, `w_2`, `t_2`, `ph_2`, and `mu_2` into the `_1` arrays inside `small_step_prep` (`module_small_step_em.F:125-167`).
- On every RK stage, WRF saves current `u_2`, `v_2`, `t_2`, `w_2`, `ph_2`, and `mu_2` into `u_save`, `v_save`, `t_save`, `w_save`, `ph_save`, and `mu_save` before converting `_2` fields into small-step work arrays (`module_small_step_em.F:172-215`, `module_small_step_em.F:241-276`).
- WRF separately saves pre-small-step `ww` into `ww_save` (`module_small_step_em.F:285`).

### 6.2 When are large-step tendencies recomputed?

- WRF recomputes dry dynamic stage tendencies by calling `rk_tendency` once per RK stage (`solve_em.F:1837-1864`).
- The WRF driver comment states that advection is re-evaluated on each corrector while physics is evaluated on RK1 and saved (`solve_em.F:1423-1438`).
- WRF combines the per-stage dynamic tendencies with RK1-stored fixed physics tendencies in `rk_addtend_dry` during each RK stage (`solve_em.F:2130-2143`, `module_em.F:1711-1782`).

### 6.3 Where do physics tendencies enter?

- WRF computes non-timesplit physics tendencies only in the RK1 block (`solve_em.F:1693-1780`).
- WRF mass-couples physics tendencies because the dynamics equations are flux-form and physics tendencies are uncoupled (`module_em.F:4419-4421`).
- WRF feeds RK1-stored physics tendencies into each RK stage through `rk_addtend_dry` (`module_em.F:1711-1782`).
- WRF includes theta diabatic heating in `rk_addtend_dry` and applies final-RK theta diabatic handling in `small_step_finish` (`module_em.F:1770-1773`, `module_small_step_em.F:418-422`).

### 6.4 When are `mu`, `muts`, `muave`, and `mudf` physical fields versus work fields?

- Before `small_step_prep`, `mu_2` is the physical perturbation dry-mass prognostic field entering the RK stage, and WRF copies it to `mu_1` on RK1 (`module_small_step_em.F:125-167`).
- During `small_step_prep`, WRF turns `mu_2` into a small-step perturbation work variable by setting it to zero on RK1 or to `mu_1 - mu_2` on later RK stages (`module_small_step_em.F:172-215`).
- During `advance_mu_t`, WRF advances the `mu` work variable, computes `muts = mut + mu`, computes weighted `muave`, and updates `mudf` (`module_small_step_em.F:1071-1112`).
- After acoustic substeps, WRF uses `muts` to recompute face masses with `calc_mu_uv_1` (`solve_em.F:4383-4406`).
- At `small_step_finish`, WRF restores physical perturbation dry mass by adding `mu_save` back into `mu_2` (`module_small_step_em.F:430`).

### 6.5 When are perturbation pressure and geopotential diagnosed or advanced?

- WRF computes `c2a` in `small_step_prep` from base plus perturbation pressure and inverse density (`module_small_step_em.F:230-234`).
- WRF diagnoses perturbation pressure and inverse density with `calc_p_rho(step=0)` before acoustic substeps (`solve_em.F:2628-2670`, `module_small_step_em.F:492-563`).
- WRF advances geopotential inside `advance_w` during each acoustic substep (`solve_em.F:3837-3898`, `module_small_step_em.F:1315-1395`, `module_small_step_em.F:1477-1584`).
- WRF refreshes perturbation pressure and inverse density after each acoustic substep with `calc_p_rho(step=iteration)` (`solve_em.F:4164-4206`, `module_small_step_em.F:492-563`).

### 6.6 Which arrays are total, perturbation, or coupled at routine boundaries?

- At the `small_step_prep` entry, `_2` prognostic arrays are physical perturbation fields, and `mub`, `muu`, `muv`, and `mut` provide base/full dry mass context (`module_small_step_em.F:72-91`, `module_small_step_em.F:125-167`).
- At the acoustic-loop entry, WRF has converted `_2` arrays into coupled perturbation work arrays and prepared `muts`, `muus`, and `muvs` as total small-step dry mass states (`module_small_step_em.F:172-215`, `module_small_step_em.F:241-276`).
- At `advance_uv`, WRF treats `u` and `v` as coupled perturbation work arrays, while pressure, inverse density, geopotential, and base pressure inputs are not coupled (`module_small_step_em.F:727-729`, `module_small_step_em.F:823-831`, `module_small_step_em.F:854-862`, `module_small_step_em.F:897-936`).
- At `advance_mu_t`, WRF treats theta as a mass-coupled work array, `mu` as a perturbation work array, `muts` as total small-step mass, and `ww` as map-coupled small-step omega (`module_small_step_em.F:1071-1171`).
- At `advance_w`, WRF uses coupled u/v/w work arrays, perturbation geopotential work state, perturbation pressure, base pressure/geopotential inputs, and total/work dry-mass fields in the implicit vertical solve (`module_small_step_em.F:1204-1236`, `module_small_step_em.F:1323-1395`, `module_small_step_em.F:1477-1584`).
- At `small_step_finish` exit, WRF decouples and restores u, v, w, theta, geopotential, omega, and perturbation dry mass back to physical prognostic representations (`module_small_step_em.F:364-430`).
