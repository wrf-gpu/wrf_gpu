# ADR-020 — c2 JAX/WRF Dycore Architecture Skeleton

Date: 2026-05-22
Author: c2-A1 worker (codex)
Status: PROPOSED after spike absorption. User authorized the sprint on 2026-05-22; this ADR records the implementation skeleton and must not be treated as operational closure until manager/reviewer approval.
Scope: c2-A1 architecture skeleton for WRF-compatible JAX dycore representation and scan carry.

## Decision

Adopt a c2 dycore architecture that separates prognostic `State`, static `BaseState`, lateral `BoundaryState`, and static `GridSpec.metrics: DycoreMetrics`. `State` carries explicit total and perturbation forms for pressure, geopotential, and dry-column mass: `p_total/p_perturbation`, `ph_total/ph_perturbation`, and `mu_total/mu_perturbation`. `BaseState` carries `pb`, `phb`, `mub`, `t0`, and `theta_base`. `DycoreMetrics` carries map factors, hybrid-eta coefficients, vertical inverse spacings, non-hydrostatic pressure-interpolation coefficients `cf1/cf2/cf3/fnm/fnp`, and terrain slopes `dzdx/dzdy` at mass points plus `dzdx_u/dzdy_v` at momentum faces. This closes Opus review R1 by making the WRF `module_small_step_em.F:663,698-711` coefficients static grid data. Implement new modules under `src/gpuwrf/dynamics/` for metrics, hybrid eta, damping, hyperdiffusion, limiters, WRF-shaped acoustic scan, and orchestration.

## Rationale

The c1 methodology review and architecture scout agree that remaining stability mechanisms are not isolated operator patches. WRF map factors, hybrid-eta coefficients, smdiv pressure memory, diffusion, Rayleigh damping, and limiters cross data contracts and timestep composition. Pace and ICON4Py demonstrate the architecture pattern: named metric/config/intermediate state and named stabilizer modules. Dinosaur/NeuralGCM supports the JAX style: pytrees plus pure `lax.scan` composition.

The numerical-stability spike is now incorporated. Its flat warm-bubble run became nonfinite at step 76 (150 s), the mountain case at step 36 (70 s), and brute-force `smdiv=0.1` plus a top Rayleigh sponge did not move the flat failure past step 76. The spike conclusion is therefore formulation-first: c2 must encode base-state/perturbation decomposition and well-balanced terrain-following PGF from day 1; damping hooks are stabilizers around a correct operator, not the architectural fix.

## Constraints

- No line-by-line port from Pace, ICON4Py, HOMME, NeuralGCM, or WRF.
- WRF `dyn_em` remains the numerical oracle for formulas.
- Map factors and hybrid-eta coefficients stay out of `State`.
- `p_perturbation`, `ph_perturbation`, and `mu_perturbation` are first-class leaves. c2 PGF code must not reconstruct them late from total-minus-base inside the stencil.
- `pb`, `phb`, `mub`, `t0`, and `theta_base` are `BaseState` leaves loaded once from initialization fixtures or analytic oracles.
- Terrain slopes `dzdx/dzdy` and face slopes `dzdx_u/dzdy_v` are `DycoreMetrics` leaves derived from `terrain_height`; terrain must not be passive metadata only.
- Previous pressure and accumulators are explicit scan carry.
- This ADR does not authorize production physics retuning, sanitize masking, MPI, or multi-GPU work.
- `smdiv`, Rayleigh damping, and hyperdiffusion are secondary stabilizers. Defaults should stay close to WRF-scale damping (`smdiv` about 0.1, Rayleigh only in an upper sponge) after the operator is well-balanced.

## Intermediate Fields Policy

Per Opus review R3 and R7, `al` (perturbation inverse density), `alt` (total inverse density), and humidity coupling factors `cqu/cqv` are scan-carried intermediates, not `State` leaves. `al` and `alt` are computed once per acoustic substep from `(p_perturbation, theta, ph_perturbation, mu_perturbation)` using WRF's `calc_p_rho_phi`/`calc_alt` call pattern (`module_em.F:217,242,1326,1340`). c2-A2 must build them inside the `acoustic_wrf.acoustic_substep_scan` carry tuple, so they cannot become stale redundant state.

## Well-Balanced PGF Requirement

c2-A2 horizontal pressure-gradient implementation must follow the WRF small-step structure in `dyn_em/module_small_step_em.F:828-862` for x-momentum and `:902-936` for y-momentum. The operator must:

- use stored perturbation pressure `p_perturbation` directly;
- combine `ph/ph_perturbation`, `p_perturbation`, `pb`, `al/alt` analogues, hybrid mass coefficients, and map-factor ratios as WRF does in the `dpxy` terms;
- use the WRF-canonical implicit terrain cancellation via the three-term decomposition `dpxy = M*rdx*(c1h*muu + c2h)*(d(ph)/dx + alt_avg*d(p_perturbation)/dx + al_avg*d(pb)/dx)` for x and the same `rdy` form for y. At hydrostatic rest with hydrostatic base plus hydrostatic perturbation, these three terms cancel exactly without explicit slope subtraction. WRF reference: `module_small_step_em.F:828-831` (x) and `:902-905` (y). Per Opus review R4, c2-A2 must not add a separate explicit slope-subtraction term. The `dzdx`/`dzdy` arrays in `DycoreMetrics` are reserved outside this PGF cancellation path for terrain-following advection/diffusion metric corrections and separately documented `dpn`-related construction.
- name the map-factor ratios exactly: x-PGF uses `msfux/msfuy`; y-PGF uses `msfvy/msfvx`. Per Opus review R6, these ratios follow WRF `module_small_step_em.F:821-826,886-891` to preserve coupled momentum form.
- add the non-hydrostatic correction fourth term when `non_hydrostatic` is active: `dpxy += M*rdx*d(php)/dx * (rdnw*d(dpn)/deta - 0.5*c1h*mu_avg)` for x, with the corresponding `M*rdy*d(php)/dy` form for y. Here `php` is hydrostatic perturbation pressure at half-levels, computed from `phb + ph_perturbation` at half-faces using `fnm/fnp`, and `dpn` is face pressure built with `cf1/cf2/cf3` near boundaries and `fnm/fnp` in the interior. Without this Opus review R5 term, the PGF misses the non-hydrostatic correction stressed by warm-bubble cases. WRF reference: `module_small_step_em.F:854-863` (x) and `:928-937` (y), with `dpn` construction at `:836-851` and `:910-925`.
- apply humidity coupling factors `cqu/cqv` to the final PGF tendency, matching `module_small_step_em.F:868,942`. Per Opus review R7, c2-A2 implements these as scan-carried intermediates under the same policy as `al/alt`.
- keep mass divergence damping `mudf/mudf_xy` distinct from `smdiv`; its exact c2 treatment is TBD pending c2-A2 prototyping.

WRF `module_small_step_em.F:557-565` remains the source anchor for `smdiv` pressure memory, but the spike shows that this damping term does not replace the well-balanced PGF.

## Evidence

Evidence is produced by the c2-A1 proof objects:

- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/metrics.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/hybrid_eta.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/scan_transfer_audit.md`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/limiter_conservation.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/integration_warm_bubble.json`
- `origin/worker/codex/m6x-numerical-stability-spike:.agent/sprints/2026-05-22-m6x-numerical-stability-spike/worker-report.md`
- `.agent/sprints/2026-05-22-gemini-c2-architecture-review/response.md`

## Consequences

The architecture now has a stable place for WRF metrics, terrain slopes, base-state fields, perturbation fields, and scan diagnostics. The cost is that c2 implementation sprints must fill new modules instead of incrementally patching the old c1 acoustic/advection functions.

## Revisit

Revisit if c2-A1 proof objects show XLA cannot keep `DycoreMetrics` and scan carry resident, if WRF fixture loading cannot populate the schema, if c2-A2 cannot implement the WRF `module_small_step_em.F:828-862,902-936` PGF without host/device transfer inside timestep loops, or if the manager/reviewer rejects the ADR-002 amendment patch.
