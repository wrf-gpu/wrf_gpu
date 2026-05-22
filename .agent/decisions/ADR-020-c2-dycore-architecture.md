# ADR-020 — c2 JAX/WRF Dycore Architecture Skeleton

Date: 2026-05-22
Author: c2-A1 worker (codex)
Status: DEFERRED-PROPOSED. User authorized the sprint on 2026-05-22; this ADR records the implementation skeleton and must not be treated as operational closure. Final ADR commitments are delayed pending the parallel numerical-stability spike report at `/tmp/wrf_gpu2_main_cp/.agent/sprints/2026-05-22-m6x-numerical-stability-spike/worker-report.md` or branch `worker/codex/m6x-numerical-stability-spike`.
Scope: c2-A1 architecture skeleton for WRF-compatible JAX dycore representation and scan carry.

## Decision

Adopt a c2 dycore architecture that separates prognostic `State`, static `BaseState`, lateral `BoundaryState`, and static `GridSpec.metrics: DycoreMetrics`. Implement new modules under `src/gpuwrf/dynamics/` for metrics, hybrid eta, damping, hyperdiffusion, limiters, WRF-shaped acoustic scan, and orchestration.

## Rationale

The c1 methodology review and architecture scout agree that remaining stability mechanisms are not isolated operator patches. WRF map factors, hybrid-eta coefficients, smdiv pressure memory, diffusion, Rayleigh damping, and limiters cross data contracts and timestep composition. Pace and ICON4Py demonstrate the architecture pattern: named metric/config/intermediate state and named stabilizer modules. Dinosaur/NeuralGCM supports the JAX style: pytrees plus pure `lax.scan` composition.

## Constraints

- No line-by-line port from Pace, ICON4Py, HOMME, NeuralGCM, or WRF.
- WRF `dyn_em` remains the numerical oracle for formulas.
- Map factors and hybrid-eta coefficients stay out of `State`.
- Previous pressure and accumulators are explicit scan carry.
- This ADR does not authorize production physics retuning, sanitize masking, MPI, or multi-GPU work.
- Base-state-vs-perturbation decomposition commitments for individual variables are pending the numerical-stability spike, especially Gemini §4 findings.
- Whether sloping-surface metric terms must be carried in `GridSpec` from day 1 is pending the numerical-stability spike.

## Evidence

Evidence is produced by the c2-A1 proof objects:

- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/metrics.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/hybrid_eta.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/scan_transfer_audit.md`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/limiter_conservation.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/integration_warm_bubble.json`

## Consequences

The architecture now has a stable place for WRF metrics and scan diagnostics. The cost is that c2 implementation sprints must fill new modules instead of incrementally patching the old c1 acoustic/advection functions.

## Revisit

Before accepting this ADR, incorporate the numerical-stability spike findings on base-state/perturbation decomposition and sloping-surface metric terms. After acceptance, revisit if c2-A1 proof objects show XLA cannot keep `DycoreMetrics` and scan carry resident, if WRF fixture loading cannot populate the schema, or if the manager/reviewer rejects the ADR-002 amendment patch.
