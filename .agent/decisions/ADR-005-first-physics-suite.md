# ADR-005 - First Physics Suite for M5

Date: 2026-05-20
Author: M5-S0 research-scout draft (Codex gpt-5.5 xhigh); manager finalization pending
Status: accepted by manager pending explicit user approval at M5-S0 closeout
Scope: M5 first physics scheme selection for the Canary 3 km v0 path
Reversibility: reversible. This chooses the first M5 implementation target, not a permanent physics-suite lock. If the chosen scheme fails the M5 stop/go gate, the manager can pick a different first scheme after reviewer-visible evidence.

## Decision

Decision: Select Thompson microphysics as the first M5 physics scheme.

Selected scheme: Thompson 2008 microphysics using WRF v4.7.1 `mp_physics=8` semantics as the Tier 1 behavior target. M5-S1 should implement a constrained WRF-compatible Thompson column subset first, with the variable vocabulary and fixture boundary chosen so later expansion remains compatible with the WRF scheme rather than becoming an unrelated toy microphysics package.

This ADR does not select the complete v0 operational physics suite. It only picks the first physics package to implement and validate. MYNN PBL, RRTMG radiation, Noah-MP land surface, and any cumulus/shallow-convection choice remain future M5-S2..N or M6/M7 decisions.

## Per-Canary Rationale

Per-Canary rationale: The Canary 3 km target needs explicit precipitation physics early. The forecast regime includes maritime trade-wind flow, inversion-capped shallow cloud, windward terrain lifting, and frequent convective showers. Microphysics directly controls hydrometeor source terms, phase partitioning, fallout, evaporation, latent heating, and precipitation reaching the surface. Those variables are visible in the first forecast products the project will care about: precipitation occurrence/intensity, cloud water/ice realism, and the heating feedback into the dycore.

PBL/surface-layer physics is arguably the most important category for trade-wind inversion and near-surface wind/humidity. It is not selected first because MYNN-style schemes introduce surface-flux dependencies, vertical turbulent diffusion solves, and stronger coupling to surface/static data. Radiation is also critical for daily heating and cloud feedback, but RRTMG brings spectral tables, gas/aerosol assumptions, solar geometry, and cyclical time-of-day coupling. Noah-MP would force a land/surface/static-geog proof object before the project has proven any real physics kernel. Thompson is therefore the best first M5 scheme: operationally meaningful, column-isolatable, and hard enough to test the ADR-001 JAX risk honestly.

## JAX Implementability

JAX implementability: ADR-001 selects JAX as the primary backend and makes the first real M5 physics scheme the decisive stop/go test for the M2 analytic-column surrogate. The M2 JAX column artifact passed with 1 kernel launch, 22 registers/thread, and 0 local memory on the analytic column fixture, under the documented fallback-profiler limitation. Full Thompson is much harder: deep conditional pathways for saturation, freezing/melting, collection, autoconversion/accretion, sedimentation, and hydrometeor threshold behavior can inflate registers or cause XLA to split kernels.

The first implementation attempt should still be pure JAX: `jax.jit`, `jax.numpy`, `jax.lax.cond`, `jax.lax.where`, and bounded `jax.lax.scan` over vertical levels where needed. The worker should avoid a monolithic all-in-one fused function if resource use climbs; staging around saturation adjustment, conversion/accretion, and sedimentation is allowed if the total launch count remains under the M5 gate. Expected dry-run target for a staged Thompson column is 6-9 launches, 96-128 registers/thread, and 0-256 B local memory. This is an estimate, not a performance claim.

Pallas/Triton is not pre-authorized for Thompson by this ADR. If the real profile trips ADR-001's fallback criteria, the manager must open `.agent/decisions/ADR-001-FALLBACK-thompson.md` or equivalent, with the failed JAX profile, attempted JAX restructurings, and reviewer cross-model challenge.

## M5 Stop/Go Gate Dry-Run Readiness

M5 stop/go gate dry-run readiness: Thompson is ready for the M5 gate because its validation and failure modes are concrete.

Tier 1 fixture readiness: WRF v4.7.1 can provide a column savepoint for the Thompson call boundary once the manager chooses the exact call site. The fixture should include temperature/pressure/moisture, hydrometeor mixing ratios, relevant number concentrations if present, density/exner or equivalent thermodynamic inputs, timestep, and before/after tendencies or state deltas.

Tier 2 invariant readiness: required checks are hydrometeor non-negativity, bounded vapor/condensate tendencies, total water budget residual within documented tolerance, finite latent heating, and no NaN/Inf. These checks match `VALIDATION_STRATEGY.md` without inventing scheme-specific tolerances after the fact.

Profile readiness: the first Thompson profile must report kernel launches, registers/thread, local memory bytes, occupancy, transfer count, and profiler limitation if Nsight counters remain blocked. GO means local memory <= 256 B, registers/thread <= 128, kernel launches <= 10, and correctness passing. FALLBACK means local memory > 256 B, registers/thread > 200, or XLA emits >50 launches. Intermediate gray-zone results, such as 129-200 registers with correctness passing, require manager judgment and likely one JAX restructuring attempt before a fallback memo.

## Consequences

Positive consequences: M5 starts with a scheme that matters to Canary precipitation and directly tests real branchy WRF column physics under the selected JAX backend. The validation story is compact: WRF column fixture, water budget, positivity, and profile gate. The result will give the manager evidence for later MYNN and RRTMG sequencing.

Negative consequences: The first physics sprint does not solve the marine PBL/inversion problem, even though that may be the highest-impact category for wind and cloud. It also does not establish the diurnal radiation cycle or land-surface state. Those gaps must remain visible in M5/M6 planning.

## Risks

Risks: Thompson may exceed JAX register or local-memory thresholds. The WRF call boundary may be harder to isolate cleanly than expected. Sedimentation and number-concentration handling may force a narrower first subset than reviewers want. A microphysics-first path can overfit the project to precipitation while delaying PBL evidence that Canary operations will eventually need. The manager must not describe Thompson success as "the first physics suite is operationally sufficient"; it is only the first package in the suite.

## Trigger for Revisiting

Trigger for revisiting: Revisit this ADR if the Thompson Tier 1 fixture cannot be generated from the WRF v4.7.1 baseline, if the JAX implementation fails correctness after a bounded restructuring attempt, if the M5 profile reaches ADR-001 STOP conditions even after the per-scheme Triton fallback, or if reviewer/domain feedback shows that a first PBL scheme is mandatory before any microphysics proof is operationally useful. Revisit is also required if M5-S1 narrows the implementation so much that it no longer represents WRF Thompson `mp_physics=8` semantics.

## Deferred Schemes

MYNN2.5/MYNN-EDMF PBL is deferred to a later M5 sprint because it needs surface-layer fluxes and vertical mixing solver evidence. RRTMG SW+LW is deferred because table governance, solar geometry, and gas/cloud optical inputs would obscure the first JAX physics-kernel signal. Noah-MP is deferred until the surface/land/SST/static-geog proof object is in scope. Cumulus parameterization is deferred because the v0 3 km domain should first rely on explicit convection plus microphysics, with shallow-convection need revisited after coupled validation.

## Evidence Pointers

Local evidence: ADR-001 backend selection and M5 stop/go gate, ADR-002 state layout, `PROJECT_PLAN.md` M5 notes, `ROADMAP.md` M5-S0 entry, M3 closeout proof inventory, and M2 column artifacts. External-prior basis from training knowledge: WRF ARW Technical Note physics inventory; Thompson et al. microphysics; MYNN/MYNN-EDMF PBL family; RRTMG radiation; Noah/Noah-MP land-surface documentation.
