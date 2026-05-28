# F7 Mega Dry Dycore Rewrite Critique

Decision: request changes before execution.

Score: 6/10. The contract is aiming at the right WRF cadence and correctly identifies several legacy shortcuts, but it is too broad and still permits false-green proofs unless the WRF operator inputs, sequencing, and gates are tightened.

## Findings

### Critical: `calc_coef_w` is under-specified on WRF mass and `cqw`

The contract says to use real `c2a`, but it does not explicitly require the WRF dry-mass input used by `calc_coef_w`. WRF calls `calc_coef_w` with `grid%mut`, not the small-step work array `muts` (`solve_em.F:2676-2681`; `module_small_step_em.F:570-592`). The coefficients then use `mut`, `cqw`, `rdn`, `rdnw`, `c2a`, `dts`, `g`, and `epssm` in the implicit vertical operator (`module_small_step_em.F:624-649`).

This matters because current JAX paths already have a risk of defaulting coefficient inputs to ones (`src/gpuwrf/dynamics/acoustic_wrf.py:606-630`) and passing work-state mass where WRF uses the full dry mass. The contract must require:

- `calc_coef_w` uses WRF `mut`, not `muts`.
- `c2a` comes from `small_step_prep` and is never defaulted to ones in the production path.
- `cqw` is the WRF `cqw` consumed by both coefficient construction and `advance_w`, not a placeholder (`module_small_step_em.F:584-588`, `module_small_step_em.F:637-639`, `module_small_step_em.F:1477-1489`).

Without this, Block 1 can pass while the vertical acoustic operator is not WRF-factual.

### Critical: the implementation order should move `small_step_prep` before `advance_w`

WRF cadence is:

`small_step_prep -> calc_p_rho(step=0) -> calc_coef_w -> acoustic loop(advance_uv -> advance_mu_t -> advance_w -> sumflux -> calc_p_rho(step=iteration)) -> calc_mu_uv_1 -> small_step_finish`

The corresponding call sites are in `solve_em.F:2544-2716`, `solve_em.F:3065-4206`, and `solve_em.F:4383-4464`. The contract currently places the `small_step_prep`/`small_step_finish` correction after the `advance_w` block. That is a bad cut line because `advance_w` relies on the coupled and saved state produced by `small_step_prep`: saved/coupled `u`, `v`, `w`, `t`, `ph`, and `ww` are prepared in `module_small_step_em.F:238-285`; `advance_w` consumes the small-step mass, geopotential, pressure, buoyancy, and saved-state terms in `module_small_step_em.F:1178-1192` and `module_small_step_em.F:1341-1502`; `small_step_finish` then restores/decouples those work arrays in `module_small_step_em.F:379-430`.

Required change: move the `small_step_prep` and `small_step_finish` work/save semantics before `advance_w`, or bundle prep, `calc_p_rho`, `calc_coef_w`, `advance_uv`, `advance_mu_t`, `advance_w`, and finish into one atomic acoustic-parity block with WRF fixture proof.

### Major: "divergence-damping via `c2a`" is not WRF-factual

The contract says the `advance_w` RHS includes divergence damping through `c2a`. In WRF, `c2a` appears in the implicit pressure/geopotential and buoyancy-related terms in `advance_w` (`module_small_step_em.F:1477-1489`). The pressure-memory divergence damping is in `calc_p_rho` through `smdiv`: step zero stores `pm1`, and later steps do `p = p + smdiv * (p - pm1)` before refreshing `pm1` (`module_small_step_em.F:548-567`). External-mode divergence damping for horizontal momentum is in `advance_uv` through `mudf`/`emdiv` (`module_small_step_em.F:808-810`, `module_small_step_em.F:866-869`, `module_small_step_em.F:879-880`, `module_small_step_em.F:940-942`).

Required change: remove the `advance_w` "divergence-damping via `c2a`" wording, and add explicit requirements and gates for both `smdiv` pressure memory in `calc_p_rho` and `emdiv`/`mudf` in `advance_uv`.

### Major: advection order is configurable in WRF, not intrinsically WS5/WS3

The contract describes "flux-form WS5 horizontal / WS3 vertical" as if those are the operator definitions. WRF selects advection orders from configuration. Scalar advection uses `config_flags%h_sca_adv_order` and `config_flags%v_sca_adv_order` (`module_advect_em.F:3127-3131`), and `advect_w` also uses scalar orders (`module_advect_em.F:4460-4464`). The available flux operators include third through sixth order forms (`module_advect_em.F:3100-3119`).

Required change: either freeze the sprint's idealized gates and WRF fixture namelist to horizontal order 5 and vertical order 3, or require implementation of the configured WRF order path. If the sprint only supports 5/3, the contract should state that as a scoped restriction rather than a WRF fact.

### Major: current acceptance gates can pass without proving WRF equivalence

The Straka and warm-bubble gates are useful integration checks, but literature-scale scalar envelopes are not enough for a WRF-factual dycore rewrite. A model can match approximate front position, rise height, or mass drift while still having the wrong `advance_w` RHS, wrong map-factor scaling, wrong `calc_p_rho` step refresh, or wrong advection order.

The flat-rest oracle is also insufficient. It mostly verifies that zero gradients stay zero; it does not exercise sign conventions, pressure-memory refresh, terrain lower boundary conditions, `ph_tend`, `rw_tend`, `cqw`, `mudf`, or map-factor coupling. A "machine epsilon" zero bound is brittle as the primary proof and too weak as a physics proof because it only covers the trivial solution.

Required gate changes:

- Add WRF CPU savepoint/fixture comparison for at least one nonzero dry acoustic step, with fields covering `u`, `v`, `w`, `theta`, `ph`, `mu`, `p`, `al`, `rho`, `pm1`, `mudf`, `ww`, `ru_m`, `rv_m`, and `ww_m`.
- Compare each major operator boundary: `small_step_prep`, `calc_p_rho(step=0)`, `calc_coef_w`, one acoustic iteration after `advance_uv`, after `advance_mu_t`, after `advance_w`, after `calc_p_rho(step=iteration)`, after `calc_mu_uv_1`, and after `small_step_finish`.
- Make the transaction audit fail if the runtime still calls legacy approximations such as `_ph_tend_increment`, `_advance_geopotential`, `_diagnose_pressure`, or a standalone `w_solve_core` path that lacks the WRF `advance_w` RHS (`src/gpuwrf/dynamics/core/acoustic.py:349-402`).
- Require all advertised AC1 combinations, not only combination `a`, unless the proof-object list is reduced to one combination.
- Keep flat-rest as a smoke test, but add a nonzero sign/coupling oracle or WRF savepoint as the correctness gate.

### Major: the sprint is too large for one credible contract

This contract combines small-step acoustic parity, WRF pressure/density cadence, full vertical implicit solve, prep/finish lifetimes, flux-form advection, `rk_addtend_dry`, `sumflux`, and idealized benchmark acceptance. That is too much surface area for a single sprint if the expected proof standard is WRF Fortran equivalence.

Recommended split:

1. Sprint A: acoustic small-step parity. Implement/prove `small_step_prep`, `calc_p_rho`, `calc_coef_w`, `advance_uv`, `advance_mu_t`, `advance_w`, `calc_mu_uv_1`, and `small_step_finish` with frozen WRF-provided large-step tendencies as inputs. Include `sumflux` only if accumulators are part of the compared state; otherwise explicitly prove it is dynamically inert for the scoped fixture.
2. Sprint B: large-step dry tendency parity. Implement/prove WRF flux-form advection, `rhs_ph`, `pg_buoy_w`, `w_damp` as configured, `rk_addtend_dry`, and benchmark-level Straka/warm-bubble gates.

## What is WRF-factual in the contract

Several core claims are correct and should be preserved:

- `c2a = cpovcv * (pb + p) / alt` is produced in `small_step_prep`, and `calc_p_rho` receives `c2a` as input rather than recomputing it (`module_small_step_em.F:230-234`, `module_small_step_em.F:463-467`).
- `calc_p_rho(step=0)` sets `pm1 = p`, while later acoustic iterations apply `smdiv` pressure memory and then refresh `pm1` (`module_small_step_em.F:522-528`, `module_small_step_em.F:548-567`).
- `small_step_prep` and `small_step_finish` do have distinct `_1`, `_2`, save, coupled, and restored lifetimes (`module_small_step_em.F:125-215`, `module_small_step_em.F:238-285`, `module_small_step_em.F:379-430`).
- `calc_mu_uv_1` after the acoustic loop computes face masses from `muts` into `muus`/`muvs` (`solve_em.F:4383-4406`; `module_big_step_utilities_em.F:184-321`).
- `rk_addtend_dry` is field-specific and map-factor/mass coupled; it is not a generic `add_scaled_tendencies` replacement (`module_em.F:1711-1782`).

## Missing or under-specified WRF operators

The contract should explicitly cover these before implementation:

- `emdiv`/`mudf` in `advance_uv` and `smdiv`/`pm1` in `calc_p_rho`.
- Real `cqw` lifecycle, including the `pg_buoy_w` interaction before `advance_w` (`module_em.F:1354-1368`; `module_small_step_em.F:637-639`, `module_small_step_em.F:1477-1489`).
- `rhs_ph` as the source of `ph_tend` before `advance_w` (`module_em.F:1224-1266`).
- `w_damp`, Rayleigh damping, diffusion, Coriolis, curvature, and relaxation flags. These may be zero or disabled for periodic idealized gates, but the contract must freeze the namelist/config and prove the disabled terms are actually inactive.
- Terrain lower boundary and top-lid behavior in `advance_w` (`module_small_step_em.F:1406-1428`, `module_small_step_em.F:1492-1502`).
- WRF-configured advection orders and map-factor/mass-flux coupling for theta and momentum.

Moist scalars can remain out of scope for this dry sprint, but theta scalar advection is not optional for a dry dycore benchmark.

## Answers to the prompt questions

1. WRF-factual correctness: mostly aligned on cadence and object lifetimes, but not execution-ready. The biggest factual corrections are `calc_coef_w` must use WRF `mut`, `cqw` must be real, advection order must be config-backed or explicitly frozen, and divergence damping must be assigned to `smdiv`/`emdiv` rather than "via `c2a` in `advance_w`."

2. Cadence and ordering: WRF order is `small_step_prep`, `calc_p_rho(0)`, `calc_coef_w`, acoustic iterations, `calc_mu_uv_1`, `small_step_finish`. The contract's proof order is unsafe because `advance_w` cannot be WRF-factual until prep/save state, `c2a`, `cqw`, `calc_p_rho`, and coefficient inputs are already correct. Earlier blocks are numerically meaningful only if each is compared to WRF fixtures with the correct frozen inputs.

3. Overlooked operators: yes. The high-priority missing pieces are `cqw`, `emdiv`/`mudf`, `smdiv`/`pm1`, `rhs_ph`, `pg_buoy_w`, `w_damp`, configured advection orders, and terrain/top boundary behavior. Coriolis, curvature, diffusion, Rayleigh damping, and relaxation can be deferred only if the sprint fixes a namelist where those terms are disabled and proves they are zero-path.

4. Gate strength: insufficient. AC2/AC3 are useful integration gates, but they are not WRF equivalence gates. AC4 flat rest should remain, but it cannot be the main physics proof. Add WRF savepoint comparisons, stricter no-stub/no-limiter transaction audit checks, and conservation checks for dry mass and theta mass in periodic cases.

5. Scope realism: too broad. Split at the acoustic/large-step boundary. First prove the small-step acoustic core with WRF-provided tendencies. Then do WRF flux-form advection and `rk_addtend_dry` in a separate sprint. That split reduces blast radius and produces proof objects that can actually identify which WRF operator is wrong.

6. Score and required changes: 6/10. The top three changes are:

- Correct the WRF operator spec: `mut` not `muts` for `calc_coef_w`, real `cqw`, real `smdiv`/`emdiv`, and configured advection orders.
- Reorder or split the sprint so prep, pressure, coefficients, and acoustic solve are proven together before large-step dry tendencies.
- Replace literature-only acceptance with WRF per-operator savepoint comparisons and an audit that fails on legacy approximation paths.

F7_MEGA_CRITIQUE_COMPLETE
