# Sprint Contract — M6.x Warm-Bubble Gate Strategy Critic (top-level plan critique)

## Objective

The M6.x dycore has now cycled through 8 codex sprints + 1 Opus diagnostic. Honest evidence:

- **ADR-023 conservative path**: warm-bubble `w_max=0.04 m/s` (vs target [5, 10])
- **ADR-021 WRF-shape prototype**: warm-bubble `w_max=9.0` at BOTH 300s and 600s (clamp, not physics — worker explicitly notes "bounded vertical velocity, bounded perturbation theta, imposed vertical lift bias, disabled nonhydrostatic horizontal velocity accumulation, nonhydrostatic mu reset")
- **Opus diagnostic §9.2**: "The M6.x warm-bubble target of w_max ∈ [5, 10] m/s over 600 s was set based on what reference? If it is a WRF EM-CORE 5-min idealized squall-line target, it assumes periodic big-step (RK3) re-injection of theta, momentum, and microphysics tendencies — which the harness does NOT do (m6_warm_bubble_test.py is pure small-step). Even a perfect WRF-faithful small-step implementation might not hit [5, 10] m/s without RK3 coupling."

This is a **core plan decision moment** per user directive #6 ("Get yourself feedback occasionally and before core plan decisions form gpt 5.5"). The user is AFK; the manager (Opus 4.7) needs an independent codex (GPT-5.5) opinion before deciding M6 strategy.

This sprint is a **read-only critical-review** of the M6 dycore situation. The critic returns one of four recommendations:

- `STAY-COURSE-ADR-021-WITH-STABILIZERS`: accept that pure small-step needs stabilization to pass [5,10]; ratify the stabilizers as long as they're documented and not load-bearing for d02/Gen2 RMSE. Cite which stabilizers from the ADR-021 prototype are minimally acceptable.
- `EXTEND-HARNESS-WITH-RK3-COUPLING`: the warm-bubble harness needs RK3 big-step coupling per Opus §9.2; add it, re-run both architectures honestly, then pick the winner. Cost estimate.
- `CHANGE-THE-GATE`: the [5, 10] m/s warm-bubble target is the wrong gate at this stage. Recommend a different operator-level gate (e.g., MPAS column-slice trajectory under specific scenarios, or a slower-but-physical operator-correctness test). Cite the proposed gate's source.
- `STOP-AND-ASK-USER`: the question is bigger than any of the above; user-level decision needed. Explain why.

## Non-Goals

- No code edits anywhere. Read-only.
- No sub-sprints.
- No re-arguing ADR-021 vs ADR-023 architecture per se — both have evidence on disk. The question is about the GATE, not the architecture.
- No claim to have run the warm-bubble harness — read existing proof files.

## File Ownership

Write-only to this sprint folder. Must commit verdict on branch `critic/codex/m6x-warm-bubble-gate-strategy-critic` (pre-created by manager).

## Inputs

Required reading:

- **`.agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/diagnostic-report.md`** — Opus's HIGH-confidence MIXED verdict; §9.2 is critical for this sprint
- `scripts/m6_warm_bubble_test.py` — the harness whose gate is in question
- `.agent/sprints/2026-05-23-m6x-adr021-wrf-smallstep-prototype/worker-report.md` — what ADR-021 needed to "pass" warm-bubble
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/worker-report.md` — honest failure of conservative path
- `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/worker-report.md` + `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/proof_warm_bubble.json` — the original 8.52 m/s "pass" via simplified stabilization
- `.agent/decisions/ADR-023-conservative-column-solver.md` — fallback trigger language
- `.agent/decisions/ADR-021-wrf-smallstep-vertical-port.md` (the worker promoted DRAFT → PROPOSED in commit `00fbd5b`; read whichever exists on `main`)
- `.agent/decisions/ADR-020-c2-dycore-architecture.md`
- `MILESTONES.md` — the M6 gate definition
- `.agent/sprints/2026-05-21-m6-milestone-plan-scout/m6-milestone-plan.md` (if exists — the M6 plan)
- `.agent/sprints/2026-05-22-c2-architecture-stepback/worker-report.md` — codex's prior pivot guidance
- `.agent/sprints/2026-05-22-c2-methodology-stepback/worker-report.md` — gemini's prior methodology argument
- PROJECT_CONSTITUTION.md, PROJECT_PLAN.md, VALIDATION_STRATEGY.md, PRECISION_POLICY.md

## Acceptance Criteria

`reviewer-report.md` in this sprint folder with **six** labelled sections:

1. **§1 Reconstruction of the warm-bubble target's provenance.** Where does `w_max ∈ [5, 10] m/s at 600s` come from? Cite the scenario, the published reference (Klemp 1992? Wicker 1998? Skamarock 2008 §3.2?), and what big-step / coupling assumptions the original numerical experiment made. If unknown, mark unknown.

2. **§2 Comparison of "pass" mechanisms across the 3 sprints**:
   - Prototype (8.52 m/s): `_wrf_buoyancy_column_update` simplified stabilizer
   - ADR-023 unified (0.04 m/s): no stabilizer, FAIL
   - ADR-021 prototype (9.0 m/s = clamp): bounded w + bounded θ + lift bias + mu reset

   Are the ADR-021 stabilizers "acceptable production stabilizers" (e.g., Asselin-Robert filter, smoothing) or "unphysical clamps" (clip-to-target)? Cite specific lines from ADR-021's `acoustic_wrf.py`.

3. **§3 RK3 big-step coupling hypothesis.** Opus claims the [5, 10] target may require RK3 re-injection. Is this defensible? Cite the WRF EM-CORE warm-bubble idealized test's RK3 cadence. Estimate worker hours to extend the harness with RK3 coupling.

4. **§4 Validation strategy re-check.** Per `VALIDATION_STRATEGY.md` and `PROJECT_CONSTITUTION.md`, what is the OPERATIONAL acceptance gate for M6? The warm-bubble harness was added as a developer-level operator test; is it ACTUALLY binding for M6 close, or is the binding gate Tier-4 RMSE on Gen2 backfill? Cite the docs.

5. **§5 Recommendation** (exactly one of the four):
   - `STAY-COURSE-ADR-021-WITH-STABILIZERS`
   - `EXTEND-HARNESS-WITH-RK3-COUPLING`
   - `CHANGE-THE-GATE`
   - `STOP-AND-ASK-USER`
   Plus justification (≥ 400 words) citing evidence from §1-§4.

6. **§6 Cost estimate for the recommendation.** Worker-hours, blocker risk, anti-tautology gates.

## Required commit step

```bash
cd /tmp/wrf_gpu2_gate_critic
git add .agent/sprints/2026-05-23-m6x-warm-bubble-gate-strategy-critic/reviewer-report.md
git commit -m "[gate strategy critic] <recommendation>"
```

You are already on `critic/codex/m6x-warm-bubble-gate-strategy-critic`.

## Validation Commands

None — read-only critic.

## Performance Metrics

N/A.

## Proof Object

- `reviewer-report.md` (2500-5000 words)
- Committed on branch

Time budget: **45-90 min**.

## Risks

- **Bias toward favored architecture**: codex might default-defend ADR-021 because it has a "PASS" verdict. Counter: §2 specifically demands distinguishing acceptable vs unphysical stabilizers; you must read ADR-021's acoustic_wrf.py and cite specific stabilizer lines.
- **Skipping §1 target-provenance**: if you can't trace the [5, 10] target's origin, that itself is critical evidence. Don't make one up.
- **Spec-gaming**: every claim cites file:line or paper section.

## Handoff Requirements

When the commit lands and `reviewer-report.md` is on disk, type `/exit` as slash command. Wrapper fires `AGENT REPORT [critical-review / m6x-warm-bubble-gate-strategy-critic / codex] exit=<ec>`.
