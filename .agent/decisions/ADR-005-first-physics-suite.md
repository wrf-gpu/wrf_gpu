# ADR-005 - First Physics Suite for M5

Date: 2026-05-20
Author: M5-S0 research-scout draft (Codex gpt-5.5 xhigh); manager finalization pending
Status: accepted by manager pending explicit user approval at M5-S0 closeout. Codex cross-model critical-review (`REVIEW-codex-ADR-005/critical-review.md` 2026-05-20) returned `Accept with required fixes`; all 5 findings applied in this revision before M5-S1 dispatch (see "Cross-Model Challenge" section below).
Scope: M5 first physics scheme selection for the Canary 3 km v0 path
Reversibility: reversible. This chooses the first M5 implementation target, not a permanent physics-suite lock. If the chosen scheme fails the M5 stop/go gate, the manager can pick a different first scheme after reviewer-visible evidence.

## Decision

Decision: Select Thompson microphysics as the first M5 physics scheme.

Selected scheme: Thompson 2008 microphysics using WRF v4.7.1 `mp_physics=8` semantics as the Tier 1 behavior target.

**Thompson-first is a SEQUENCING decision, NOT an operational-sufficiency claim.** The Canary 3 km regime is most-strongly limited by PBL/inversion physics (see Per-Canary Rationale §21 and the scout-report's Canary-fit ranking which puts PBL/surface-layer above microphysics for operational impact). Thompson is selected first because it is the most tractable first real-physics stop/go test for the JAX backend (ADR-001) and the M1 fixture infrastructure, NOT because precipitation microphysics is the highest-value missing operational component for Canary. M5/M6 communications and ADRs MUST NOT describe Thompson success as "the first physics suite is operationally sufficient." MYNN-EDMF PBL is the explicit FOLLOW-ON HOOK and is reserved as M5-S2 (placeholder sprint folder to be created at M5-S1 closeout); trade-wind inversion and low-cloud realism risks remain visible in M5/M6 planning and ADR-005 cannot be used to defer that work.

This ADR aligns with the project's Gen2 target operational stack (`PROJECT_PLAN.md:176`): Thompson + MYNN-EDMF + Noah-MP + RRTMG. Thompson is the first package in that stack; the others follow in M5-S2..N and M6/M7 as their preconditions (surface-coupling for Noah-MP; spectral/time-of-day infrastructure for RRTMG; flux/diffusion infrastructure for MYNN) become available.

### Minimum frozen Thompson target (REQUIRED for M5-S1 dispatch per critical-review Blocker #1)

The M5-S1 sprint contract MUST implement at least the following Thompson subset, no narrower:

- **WRF call boundary**: `module_mp_thompson` `mp_gt_driver` column-loop body (the per-column inner of WRF's Thompson driver). The fixture savepoint captures inputs/outputs at this driver boundary.
- **Prognostic hydrometeor species (REQUIRED in scope)**: water vapor `qv`, cloud water `qc`, rain `qr`, cloud ice `qi`, snow `qs`, graupel `qg`. All six are MUST-include — narrowing below this set is not WRF-compatible Thompson.
- **Number concentration fields (REQUIRED in scope)**: `Ni` (ice number), `Nr` (rain number). These are the two number-concentrations Thompson 2008 prognoses. Cloud-water and snow/graupel are typically diagnostic in Thompson 2008; if WRF v4.7.1's `mp_physics=8` build path makes any additional number concentration prognostic, that one is also REQUIRED.
- **Sedimentation**: **OUT of M5-S1 scope.** Sedimentation is the most complex Thompson path (variable terminal velocities, sub-stepping, sub-cycling for stability). M5-S1 implements source/sink terms WITHOUT sedimentation; sedimentation is a dedicated M5-S1.x or M5-S2-prerequisite sprint. The Tier 1 fixture MUST be generated with `iwmrf=0` or equivalent flag that disables sedimentation at fixture-generation time so the GPU implementation can match.
- **Tier 1 fixture variables**: input `(T, p, qv, qc, qr, qi, qs, qg, Ni, Nr, ρ or π, dt)`; output `(qv, qc, qr, qi, qs, qg, Ni, Nr, T)` after one Thompson call. Tolerances `tolerance_abs=1e-10`, `tolerance_rel=1e-8` at fp64 for the hydrometeor mixing ratios; `tolerance_abs=1e-3`, `tolerance_rel=1e-6` for number concentrations (Thompson's saturation/freezing thresholds are conditional and small numerical drift accumulates).
- **Tier 2 invariants**: hydrometeor non-negativity; total water budget closure `|Σq_water_t - Σq_water_0| ≤ 1e-8 * Σq_water_0`; finite latent heating `|Δθ_latent| < 100 K`; no NaN/Inf.

If any of the above CANNOT be implemented in M5-S1, M5-S1 manager MUST open a fix-cycle amendment to ADR-005 and re-dispatch this critical-review BEFORE narrowing the contract further. This prevents the "constrained subset that still claims progress" failure mode flagged in critical-review Blocker #1.

This ADR does not select the complete v0 operational physics suite. It only picks the first physics package to implement and validate. MYNN PBL, RRTMG radiation, Noah-MP land surface, and any cumulus/shallow-convection choice remain future M5-S2..N or M6/M7 decisions per the deferred-schemes section below.

## Per-Canary Rationale

Per-Canary rationale: The Canary 3 km target needs explicit precipitation physics early. The forecast regime includes maritime trade-wind flow, inversion-capped shallow cloud, windward terrain lifting, and frequent convective showers. Microphysics directly controls hydrometeor source terms, phase partitioning, fallout, evaporation, latent heating, and precipitation reaching the surface. Those variables are visible in the first forecast products the project will care about: precipitation occurrence/intensity, cloud water/ice realism, and the heating feedback into the dycore.

PBL/surface-layer physics is arguably the most important category for trade-wind inversion and near-surface wind/humidity. It is not selected first because MYNN-style schemes introduce surface-flux dependencies, vertical turbulent diffusion solves, and stronger coupling to surface/static data. Radiation is also critical for daily heating and cloud feedback, but RRTMG brings spectral tables, gas/aerosol assumptions, solar geometry, and cyclical time-of-day coupling. Noah-MP would force a land/surface/static-geog proof object before the project has proven any real physics kernel. Thompson is therefore the best first M5 scheme: operationally meaningful, column-isolatable, and hard enough to test the ADR-001 JAX risk honestly.

## JAX Implementability

JAX implementability: ADR-001 selects JAX as the primary backend and makes the first real M5 physics scheme the decisive stop/go test for the M2 analytic-column surrogate. The M2 JAX column artifact passed with 1 kernel launch, 22 registers/thread, and 0 local memory on the **analytic column fixture** (`artifacts/m2/jax/column_profile.json:27-31`), but the artifact also records **fallback-derived profiling and profiler limitations** (`artifacts/m2/jax/column_profile.json:30`). Full Thompson is structurally a different problem: deep conditional pathways for saturation, freezing/melting, collection, autoconversion/accretion, and hydrometeor threshold behavior can inflate registers or cause XLA to split kernels.

**Critical-review Major #3 — applied:** Any kernel-launch / register / local-memory numbers in this ADR are **hypotheses for risk-prior purposes only**, not readiness evidence. The M2 analytic-column profile is NOT a Thompson profile and does not validate Thompson's JAX viability. The **first real Thompson profile JSON** (the artifact `artifacts/m5/thompson_profile.json` to be produced by M5-S1) is the ONLY readiness proof object that the M5 stop/go gate accepts, consistent with ADR-001's M5 gate definition (`ADR-001-backend-selection.md:131-139`).

The first implementation attempt should still be pure JAX: `jax.jit`, `jax.numpy`, `jax.lax.cond`, `jax.lax.where`, and bounded `jax.lax.scan` over vertical levels where needed. The worker should avoid a monolithic all-in-one fused function if resource use climbs; staging around saturation adjustment, conversion/accretion, and (if sedimentation were in scope, which it is not for M5-S1) sedimentation is allowed if the total launch count remains under the M5 gate.

**Hypothetical dry-run target** (NOT a readiness claim — to be confirmed or falsified by `artifacts/m5/thompson_profile.json`): for a staged Thompson column with sedimentation OUT of scope, estimated 6-9 launches, 96-128 registers/thread, and 0-256 B local memory. These numbers are derived from training-knowledge intuition about Thompson's branch depth, NOT measured. They exist to set worker expectations and to flag the structural risk that Thompson may need restructuring or fallback. The M5 stop/go gate uses MEASURED numbers from the first real Thompson profile, not these estimates.

Pallas/Triton is not pre-authorized for Thompson by this ADR. If the real profile trips ADR-001's fallback criteria, the manager must open `.agent/decisions/ADR-001-FALLBACK-thompson.md` or equivalent, with the failed JAX profile, attempted JAX restructurings, and reviewer cross-model challenge.

## M5 Stop/Go Gate Dry-Run Readiness

M5 stop/go gate dry-run readiness: Thompson is ready for the M5 gate because its validation and failure modes are concrete.

Tier 1 fixture readiness: WRF v4.7.1 can provide a column savepoint for the Thompson call boundary once the manager chooses the exact call site. The fixture should include temperature/pressure/moisture, hydrometeor mixing ratios, relevant number concentrations if present, density/exner or equivalent thermodynamic inputs, timestep, and before/after tendencies or state deltas.

Tier 2 invariant readiness: required checks are hydrometeor non-negativity, bounded vapor/condensate tendencies, total water budget residual within documented tolerance, finite latent heating, and no NaN/Inf. These checks match `VALIDATION_STRATEGY.md` without inventing scheme-specific tolerances after the fact.

Profile readiness: the first Thompson profile must report kernel launches, registers/thread, local memory bytes, occupancy, transfer count, and profiler limitation if Nsight counters remain blocked. **GO** means local memory <= 256 B, registers/thread <= 128, kernel launches <= 10, AND correctness passing (Tier 1 + Tier 2 from the frozen target above). **FALLBACK** means local memory > 256 B, registers/thread > 200, or XLA emits >50 launches.

**Gray-zone rule (Critical-review Major #4 — applied; NOT discretionary):** Intermediate results (129-200 registers with correctness passing; OR local memory 257-512 B; OR launches 11-50) trigger a **mandatory NON-discretionary sequence** before any GO-or-FALLBACK call:

1. **At least one documented JAX restructuring attempt** (e.g. split the monolithic kernel at a major branch point; replace `lax.cond` with `jnp.where` where the branches are short; loop-unroll vs `lax.scan`). The attempt MUST produce a SECOND profile JSON for comparison.
2. **Occupancy/local-memory evidence** from the restructured profile MUST be reported alongside the original numbers.
3. **Cross-model reviewer signoff** (Codex `gpt-5.5` xhigh as critical-reviewer) on a gray-zone memo that documents both profiles, the restructuring tried, and the reviewer's call: proceed-without-fallback OR open per-scheme fallback ADR.
4. If the reviewer recommends proceed-without-fallback in the gray zone, that decision goes to the human arbiter at M5-S1 closeout — not silently absorbed by manager judgment.

This removes the "manager judgment" loophole flagged by critical-review Major #4. Manager has clear procedural authority on GO and FALLBACK; gray-zone requires a documented restructuring + cross-model signoff + human arbiter visibility.

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

Local evidence: ADR-001 backend selection and M5 stop/go gate (`ADR-001-backend-selection.md:131-139`); ADR-002 state layout; **`PROJECT_PLAN.md:176` Gen2 operational target stack = Thompson + MYNN-EDMF + Noah-MP + RRTMG** (the project-specific target stack that this ADR chooses the FIRST package of, not a generic-WRF-practice argument); `PROJECT_PLAN.md` M5 notes; `ROADMAP.md` M5-S0 entry; M3 closeout proof inventory; M2 column artifacts (which are dry-run-priors only per Critical-review Major #3).

External-prior basis from training knowledge: WRF ARW Technical Note physics inventory; Thompson et al. 2008 microphysics (`mp_physics=8` semantics); MYNN/MYNN-EDMF PBL family (deferred — M5-S2 placeholder); RRTMG radiation (deferred — M5/M6 boundary); Noah/Noah-MP land-surface documentation (deferred — M7 surface coupling).

## Cross-Model Challenge

Codex `gpt-5.5` xhigh critical-review of 2026-05-20 (file: `.agent/decisions/REVIEW-codex-ADR-005/critical-review.md`) returned Decision: `Accept with required fixes`.

### Codex's findings — verbatim summary (full text in critical-review.md)

> **Top three structural concerns:**
> 1. The Thompson recommendation is defensible as a sequencing choice, but the ADR's strongest argument is implementation/validation tractability, not highest Canary operational impact. The ADR must be explicit that Thompson-first is a first real-physics stop/go package, not a claim that precipitation microphysics is the highest-value missing operational component.
> 2. The selected target is under-specified. "Thompson 2008 microphysics using WRF v4.7.1 `mp_physics=8` semantics" is specific enough as a scheme name, but "a constrained WRF-compatible Thompson column subset first" leaves the M5-S1 worker room to implement a toy-like subset and still claim progress.
> 3. The JAX implementability case is appropriately cautious in places, but still leans too heavily on the M2 analytic-column surrogate.

Five findings total: 1 blocker (frozen Thompson target before M5-S1 dispatch), 3 majors (sequencing-not-sufficiency framing + MYNN follow-on hook; M2 numbers as hypothesis not readiness; non-discretionary gray-zone rule), 1 minor (cite PROJECT_PLAN:176 directly).

Counter-proposals from critical-review:
- Accept Thompson-first if ADR tightened.
- Alternative: MYNN2.5/MYNN-EDMF PBL-first if optimizing for Canary operational impact rather than first-kernel tractability. Critical-reviewer accepted Thompson-first as a sequencing choice rather than recommending rejection.
- Lower-risk: WSM6 microphysics if Thompson boundary proves unworkable. Not chosen because WSM6 moves away from the target operational stack.

### Manager response — all 5 applied

- **Blocker #1 (frozen Thompson target)**: Selected-scheme section expanded with a "Minimum frozen Thompson target" subsection naming the WRF call boundary (`module_mp_thompson` `mp_gt_driver` column-loop body), prognostic species (qv/qc/qr/qi/qs/qg + Ni/Nr), sedimentation OUT of M5-S1 scope (deferred to dedicated sub-sprint), Tier 1 fixture variables + tolerances, Tier 2 invariants. M5-S1 manager cannot narrow further without amending ADR-005 and re-dispatching critical-review.
- **Major #2 (sequencing not sufficiency + MYNN hook)**: Selected-scheme paragraph expanded with explicit "SEQUENCING decision, NOT operational-sufficiency claim" framing; MYNN-EDMF declared the explicit follow-on hook as M5-S2 placeholder.
- **Major #3 (M2 numbers as hypothesis)**: JAX-Implementability section reworded; estimated 6-9 launches / 96-128 registers / 0-256 B local memory explicitly labeled as "Hypothetical dry-run target — NOT a readiness claim — to be confirmed or falsified by `artifacts/m5/thompson_profile.json`." The first real Thompson profile is the only accepted readiness proof.
- **Major #4 (non-discretionary gray-zone rule)**: Profile-readiness section adds an explicit 4-step gray-zone procedure (restructuring attempt + 2nd profile + cross-model signoff + human arbiter visibility) replacing the "manager judgment" loophole.
- **Minor #5 (cite PROJECT_PLAN:176)**: Evidence-pointers section now cites the project's Gen2 target stack directly, framing this ADR as project-specific not generic-WRF.

No manager counter-dissent recorded. All Codex findings were fair catches.
