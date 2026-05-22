# ADR-020 — c2 JAX/WRF Dycore Architecture Skeleton

Date: 2026-05-22
Author: c2-A1 worker (codex)
Status: PROPOSED after spike absorption. User authorized the sprint on 2026-05-22; this ADR records the implementation skeleton and must not be treated as operational closure until manager/reviewer approval.
Scope: c2-A1 architecture skeleton for WRF-compatible JAX dycore representation and scan carry.

## Decision

Adopt a c2 dycore architecture that separates prognostic `State`, static `BaseState`, lateral `BoundaryState`, and static `GridSpec.metrics: DycoreMetrics`. `State` carries explicit total and perturbation forms for pressure, geopotential, and dry-column mass: `p_total/p_perturbation`, `ph_total/ph_perturbation`, and `mu_total/mu_perturbation`. `BaseState` carries `pb`, `phb`, `mub`, `t0`, and `theta_base`. `DycoreMetrics` carries map factors, hybrid-eta coefficients, vertical inverse spacings, and terrain slopes `dzdx/dzdy` at mass points plus `dzdx_u/dzdy_v` at momentum faces. Implement new modules under `src/gpuwrf/dynamics/` for metrics, hybrid eta, damping, hyperdiffusion, limiters, WRF-shaped acoustic scan, and orchestration.

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

## Well-Balanced PGF Requirement

c2-A2 horizontal pressure-gradient implementation must follow the WRF small-step structure in `dyn_em/module_small_step_em.F:828-862` for x-momentum and `:902-936` for y-momentum. The operator must:

- use stored perturbation pressure `p_perturbation` directly;
- combine `ph/ph_perturbation`, `p_perturbation`, `pb`, `al/alt` analogues, hybrid mass coefficients, and map-factor ratios as WRF does in the `dpxy` terms;
- subtract the terrain-following hydrostatic slope correction equivalent to `(g/alpha) * dzdx * dp/deta` and `(g/alpha) * dzdy * dp/deta`;
- apply map-factor scaling at the appropriate momentum faces;
- use perturbation inverse density (`al`) and total inverse density (`alt`) analogues explicitly, rather than hiding them in an unreviewed pressure update.

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
