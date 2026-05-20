# Sprint Contract — M5-S0 First-Physics-Suite Selection

Sprint ID: `2026-05-20-m5-s0-physics-scheme-selection`
Milestone: M5 — First Physics Suite (this is the decision-gate sprint that PRECEDES implementation; see `ROADMAP.md` M5-S0 NEW entry).
Worker: research-scout (Codex `gpt-5.5` `xhigh`) — research-and-writing role, no model code changes.
Tester: not applicable (research/decision sprint; tester role explicitly waived in scout-report.md).
Reviewer: opus-reviewer (Codex `gpt-5.5` `xhigh`) — judges whether the recommendation is evidence-backed and operationally relevant.
Critical-review: opus-reviewer (Codex `gpt-5.5` `xhigh`) — second-opinion on the chosen scheme.
Approval status: pending scout dispatch.

## Objective

Pick the **first physics scheme** that M5-S1 will implement, with a recorded evidence-backed rationale tying the choice to the Canary operational target. Output: `scout-report.md` (the brief) + draft `ADR-005-first-physics-suite.md` (the decision, manager-finalizable). The M2 column-physics analog is **NOT** by itself a commitment to any scheme — this sprint makes the explicit operational choice.

Per user directive of 2026-05-19 (delegation): manager dispatches without per-decision approval; escalation only if no scheme can be defended. Per `feedback_manager_autonomy.md`.

## Non-Goals

- **No implementation.** Zero code in `src/gpuwrf/`. M5-S1 is the implementation sprint that uses this decision.
- **No multiple-scheme commitment.** Pick ONE first scheme. Sibling schemes are M5-S2..N future work.
- **No M5-S1 contract drafting.** That happens in a separate sprint after this scout's recommendation is reviewed.
- **No new ADRs other than ADR-005 draft.** The halo ADR (project_state_layout memory) is separate scope.

## File Ownership

Scout may create or edit only these paths:

### Scout deliverables (this sprint folder)
- `.agent/sprints/2026-05-20-m5-s0-physics-scheme-selection/scout-report.md` (new — the brief)
- `.agent/sprints/2026-05-20-m5-s0-physics-scheme-selection/agent_success.json` (new)
- `.agent/sprints/2026-05-20-m5-s0-physics-scheme-selection/manager-closeout.md` (template — manager will finalize)

### ADR draft
- `.agent/decisions/ADR-005-first-physics-suite.md` (new — scout drafts technical body; manager finalizes; codex critical-review on a separate pass)

Any change outside this list requires manager approval. **Do not touch any code, any test, any artifact, any existing ADR, or any governance file.** This is research+writing only.

## Inputs (mandatory read order)

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `PROJECT_PLAN.md` §8 (M5 stricter gates) + §11.6 (IC/BC = AIFS)
4. `PROJECT_SCOPE.md` — Canary operational target
5. `.agent/milestones/ROADMAP.md` M5 (the §M5-S0 entry exists; this contract IS that S0 sprint)
6. `.agent/milestones/M5-first-physics-suite.md`
7. `ARCHITECTURE_PRINCIPLES.md`, `VALIDATION_STRATEGY.md`, `PRECISION_POLICY.md`, `PERFORMANCE_TARGETS.md`
8. ADR-001 (`.agent/decisions/ADR-001-backend-selection.md`) — JAX primary, Triton escape hatch
9. ADR-002 (`.agent/decisions/ADR-002-state-layout.md`) — SoA, C-grid, fp64, halo placeholder
10. `.agent/decisions/MILESTONE-M3-CLOSEOUT.md` — current proof-object inventory
11. M2 column-physics work: `artifacts/m2/*/column_profile.json` + `artifacts/m2/*/maintainability.md` (the M2 column candidates — what each backend could handle in column physics)
12. The relevant skills: `.agent/skills/researching-prior-art/SKILL.md`, `.agent/skills/validating-physics/SKILL.md`, `.agent/skills/maintaining-memory/SKILL.md`
13. **External knowledge expected from the scout** (no internet access required; codex's training knowledge should suffice):
    - **WRF physics options inventory**: ARW Tech Note §8 (Physics) — what schemes exist and what each does
    - **Operational considerations for Canary 3 km daily forecasts**: maritime subtropical climate, frequent convective showers, importance of marine PBL, coastal upwelling not relevant at 3 km, terrain-induced precipitation, trade-wind inversion
    - **Common WRF operational stacks**: e.g. Thompson (microphysics) + MYNN2.5 (PBL) + RRTMG (radiation) + Noah-MP (surface) is one widely-deployed combo
    - **JAX-implementability concerns**: schemes with deep conditional branching (e.g. Thompson microphysics is conditional-heavy; MYNN PBL has many branches; radiation has nested integrals) — what's likely to register-spill on Blackwell

## Acceptance Criteria

All must hold for closeout. Numbered for reviewer traceability.

### 1. scout-report.md (the brief)

1.1. ≥1500 bytes, must include the literal token `Summary:` and `Decision:`.

1.2. Section: **Candidate inventory.** Tabulate at least these candidate categories with one paragraph each:
- Microphysics (e.g. Thompson, WSM6, Morrison-Gettelman, P3)
- Planetary Boundary Layer / surface layer (e.g. MYNN2.5, YSU, MYJ, ACM2)
- Radiation (e.g. RRTMG SW+LW, Goddard, CAM)
- Land surface (e.g. Noah-MP, Noah, RUC)
- Convection (if applicable at 3 km — likely "explicit" not "parameterized" at this resolution, but cumulus may still be needed at outer domain)
For each: what it computes, why operational WRF deployments tend to use it, what its primary computational cost is (arithmetic intensity vs branching vs memory).

1.3. Section: **Canary-specific operational fit.** Rank the categories by criticality for the Canary 3 km daily forecast given:
- Maritime subtropical climate (trade winds, inversion-capped boundary layer, frequent shallow convective showers)
- 3 km grid spacing (cumulus often "grey zone"; explicit convection viable for most cases; shallow-cumulus / shallow-convection parameterization may be needed)
- Terrain-induced precipitation on windward coasts
- No SST coupling required at M5 (deferred per M7 surface-coupling sprint)

1.4. Section: **JAX implementability.** For each top-3 candidate, assess:
- Estimated kernel-launch count for a column step (target ≤ 10 per ADR-001's M5 gate)
- Estimated register pressure (target ≤ 128 per ADR-001)
- Local memory pressure (target ≤ 256 B)
- Branching depth (deeply-conditional schemes are hard for XLA to fuse; per ADR-001 trigger: trip → per-scheme Triton fallback ADR)
- Whether the scheme is implementable in pure `jax.jit` + `jax.numpy` + `jax.lax`, or whether it needs Pallas/Triton

1.5. Section: **Recommendation.** ONE first scheme picked by name + version (e.g. "Thompson 2008 with `imp_physics=8` semantics" or "MYNN2.5 EDMF" — be specific enough that M5-S1's worker can implement it). Justify in ≤300 words tying to:
- Operational impact for Canary 3 km daily (which forecast variable does this primarily improve?)
- JAX implementability (register/launch pressure should pass M5 gate without Triton escape)
- Bakeoff-validation tractability (we need a tier-1 fixture for this scheme — does WRF + M1 fixture infrastructure permit generating one?)

1.6. Section: **Out-of-scope notes.** What schemes are explicitly DEFERRED to M5-S2..N and why (e.g. "RRTMG is critical but cyclical-time-of-day work; defer to M5-S2 after the diurnal pattern is required").

1.7. Section: **Risks + open questions.** Anything the scout couldn't resolve from training knowledge that would need a domain expert or live research.

### 2. ADR-005 draft (`.agent/decisions/ADR-005-first-physics-suite.md`)

2.1. ≥1500 bytes, must include `Decision:`, `Selected scheme:`, `Per-Canary rationale:`, `JAX implementability:`, `M5 stop/go gate dry-run readiness:`, `Risks:`, `Trigger for revisiting:`.

2.2. Worker drafts the technical body. Manager finalizes after sprint reviewer Accept. Codex critical-review then runs in a separate folder (`REVIEW-codex-ADR-005/`) per ADR-001/002/003 pattern.

2.3. **Status field** initially reads "accepted by manager pending explicit user approval at M5-S0 closeout" — same gating pattern as ADR-001/002/003. (User has authorized overnight progress without per-decision approval per 2026-05-19 directive; but the constitutional human-approval gate stays for irreversible-architecture ADRs. Manager may proceed to M5-S1 implementation in the user-asleep window; user gets a final-approval pass at M5-S0 closeout in the morning.)

2.4. Reversibility: assess; first physics scheme is **reversible** (we can pick another after evaluation), so this is NOT irreversible by `.agent/rules/architecture-decision-policy.md`. The "human approval" framing is courtesy + audit-trail, not a constitutional hard-gate.

### 3. Cross-AI critical-review readiness

3.1. The ADR-005 draft and scout-report MUST be self-contained enough that an independent reviewer can challenge the scheme choice without re-doing the literature work.

3.2. Scout records in `agent_success.json`: `sprint_count: 1`, `reviewer_rejections_before_handoff: 0` (none yet), `escalation_events: 0` (or [...] if any).

### 4. Hygiene

4.1. No code touched. No `src/`, no `tests/`, no `scripts/`, no `artifacts/`, no `fixtures/`, no existing ADR/governance/skill files.
4.2. Worktree is `/tmp/wrf_gpu2_m5_scout` (separate from the main worktree to enable parallel execution with the M4-S1 worker). Branch: `scout/codex/m5-s0-physics-scheme-selection` (scout creates + pushes).

## Validation Commands

```bash
# Scout self-checks (run from /tmp/wrf_gpu2_m5_scout)
wc -c .agent/sprints/2026-05-20-m5-s0-physics-scheme-selection/scout-report.md
wc -c .agent/decisions/ADR-005-first-physics-suite.md
grep -c -E 'Decision:|Selected scheme:|Per-Canary rationale:|JAX implementability:' .agent/decisions/ADR-005-first-physics-suite.md
git status --short
git branch --show-current
```

## Proof Object

- `scout-report.md` ≥ 1500 bytes with `Summary:` + `Decision:` tokens.
- `ADR-005-first-physics-suite.md` ≥ 1500 bytes with all required tokens.
- `agent_success.json`.
- Both committed on branch `scout/codex/m5-s0-physics-scheme-selection` and pushed.

## Risks

- **Scout picks by analogy to deepthink/GPT-5.5 brief** instead of operational reality. Mitigation: AC 1.3 requires Canary-specific operational fit; reviewer will flag if the rationale is generic.
- **Scout doesn't have Canary-specific knowledge** beyond training knowledge. Mitigation: AC 1.7 requires explicit open-questions list so the manager can spot gaps.
- **M5 stop/go gate (≤10 launches, ≤128 regs, ≤256 local mem) too tight for any real scheme.** Per ADR-001 this is the **trigger** for per-scheme Triton fallback — not a sprint failure. ADR-005 should explicitly state expected pressure and which fallback applies if the gate trips.

## Handoff Requirements

- Scout pushes to branch `scout/codex/m5-s0-physics-scheme-selection` in `/tmp/wrf_gpu2_m5_scout`.
- On scout completion, manager:
  - Reviews scout-report.md and ADR-005 draft
  - Finalizes ADR-005 body
  - Dispatches Codex critical-review on ADR-005 in `.agent/decisions/REVIEW-codex-ADR-005/`
  - Applies findings
  - Merges scout branch into main
  - Per overnight-autonomy directive: proceeds to M5-S1 implementation WITHOUT waiting for user explicit approval (will flag the ADR-005 acceptance in MORNING-REPORT.md for user's awareness)

## When done

Type `/exit` to close the codex REPL — the dispatcher wrapper will detect exit, run completion helper, kill the tmux window, and send-keys a summary back to the manager.
