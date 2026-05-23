# Sprint Contract — M6 Close Strategy Plan Critic

## Objective

The user told the manager (Claude Opus 4.7) at ~07:00 UTC: "the 4 open questions in the morning report are too technical — formulate a plan with measurable goals and ask GPT-5.5 whether this is the best strategic way; have GPT-5.5 give its own version of the plan with explanations and rationale. Also think about diagnostic tools / sidecars that give more diagnostic power. And get inspiration from similar projects (WRF, MPAS, Pace, ICON4Py) and the WRF code itself. Always think: what is the correct solution + what is the easiest way to the correct solution?"

The manager has drafted a 5-sprint plan at `manager-plan-m6-close-strategy.md` in this folder. This sprint asks the Codex critic (GPT-5.5 xhigh) to:

1. **Critique** the manager's draft plan with cited evidence
2. **Write your own version** of the plan with explicit rationale per step
3. **Compare and contrast** — where do you agree, where do you differ, why
4. **Identify diagnostic sidecars** the manager missed
5. **Mine WRF/MPAS/Pace/ICON4Py source** for specific patterns/stabilizers that would inform the operator changes

## Non-Goals

- No code edits anywhere. Read-only review.
- No sub-sprints.
- No critique of ADR-021 vs ADR-023 architecture itself — the manager's plan stays neutral on that; either could be the answer. The critic should evaluate the *plan*, not re-litigate the architectural decision.
- No vague "this looks fine" responses — every claim must cite file:line.

## File Ownership

Write-only to this sprint folder. Must commit your verdict on branch `critic/codex/m6x-close-strategy-plan-critic` (pre-created by manager).

## Inputs

Required reading:

- **`manager-plan-m6-close-strategy.md`** in THIS folder — the plan you're evaluating
- `MORNING-REPORT.md` (at repo root) — current project state + 4 open questions
- `.agent/SPRINT-TRACKER.md` — full session ledger
- `.agent/decisions/ADR-023-conservative-column-solver.md` — current architecture
- `.agent/decisions/ADR-024-warm-bubble-gate-policy.md` — gate policy change
- `.agent/decisions/ADR-021-wrf-smallstep-vertical-port-DRAFT.md` — opposing alternative
- `.agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/diagnostic-report.md` — Opus's MIXED verdict (especially §7 conclusions + §9 open questions)
- `.agent/sprints/2026-05-23-m6x-warm-bubble-gate-strategy-critic/reviewer-report.md` — prior CHANGE-THE-GATE verdict (especially §5 + §6)
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/worker-report.md` — what the unified path actually does + remaining stabilizers
- `.agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/worker-report.md` — current FAIL_PHYSICAL_BOUNDS verdict
- `.agent/sprints/2026-05-21-m6-milestone-plan-scout/m6-milestone-plan.md` — the original M6 plan that defined the milestone gates
- `MILESTONES.md` § M6 — the M6 gate definition
- `VALIDATION_STRATEGY.md` — Tier-1/2/3/4 pyramid
- `ADR-007` precision policy — operational RMSE budgets
- `PROJECT_CONSTITUTION.md` — what cannot change

Source mining (cite specific line ranges):

- WRF `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F` — small-step canonical
- WRF `module_em.F` — large-step / RK3 outer
- MPAS `/mnt/data/canairy_meteo/artifacts/wsm6_gpu_port/MPAS_wsm6_GPU_for_CAG_clean/MPAS-Model-5.3/src/core_atmosphere/dynamics/mpas_atm_time_integration.F` — especially `dss` Rayleigh (`:2184-2193`), tridiag coef builder (`:1589-1656`)
- Pace `Pace@6a46e69` (public via curl if not local) — `dyn_core.py`, `del2cubed.py`, `ray_fast.py`
- ICON4Py `ICON4Py@3934f68` (public via curl if not local) — `solve_nonhydro.py`, `vertically_implicit_dycore_solver.py`
- Dinosaur `Dinosaur@59a0197` — `time_integration.py` IMEX

Current state evidence:

- `src/gpuwrf/dynamics/acoustic_wrf.py` — the current unified ADR-023 operator
- `scripts/m6_warm_bubble_test.py` — the new operator-sanity harness
- `scripts/diagnostic_warm_bubble_vs_slice.py` — the diagnostic script Opus already built
- `.agent/sprints/2026-05-23-m6x-adr023-d02-boundary-replay-1h/` — d02 scaffolding (worker halted earlier; `scripts/m6_d02_boundary_replay_1h.py`, `src/gpuwrf/integration/d02_replay.py`, `tests/test_m6x_d02_boundary_replay.py`)

## Acceptance Criteria

Produce `reviewer-report.md` in this sprint folder with **seven** labeled sections:

1. **§1 Critique of the manager's draft plan.** For each of the 5 sprints (S1-S5) and the cross-cutting inspiration mining + risk register, identify:
   - What's right about the structure
   - What's missing, vague, or wrong
   - Specific file:line citations for every claim
   - Where the manager's measurable goals are NOT actually measurable (e.g., "5-8h" is not a measurable goal; "RMSE < 2× Gen2 spread" IS, if Gen2 spread is defined)

2. **§2 Your own version of the plan.** Write a complete 3-7 sprint sequence with:
   - Goal statement per sprint (single sentence)
   - Measurable acceptance criteria (numerical thresholds, named tests, file:line proof targets)
   - Time budget per sprint
   - Dependency order
   - Rationale (≥ 100 words per sprint) explaining why this sprint, why now, why this scope
   - Diagnostic sidecars / tooling deliverables identified per sprint
   - Inspiration sources (WRF/MPAS/Pace/ICON4Py file:line) cited per sprint

3. **§3 Diff vs manager's plan.** Side-by-side comparison table. For each manager sprint, name the closest sprint in your plan and identify the key differences. For sprints in your plan with no manager analog (or vice versa), say so. ≥ 300 words.

4. **§4 Diagnostic sidecars audit.** List EVERY diagnostic capability that would clarify the dycore situation. Include:
   - The 4 the manager named (field RMSE timeline, spatial divergence map, conservation tracker, bound violation tracer)
   - Any the manager missed
   - Build order (cheapest-most-valuable first)
   - For each: what question it answers, expected output, estimated build hours
   Aim for ≥ 8 sidecars total. Most should reuse the pattern from `scripts/diagnostic_warm_bubble_vs_slice.py`.

5. **§5 Source-mining table.** For each operator concern (mu_continuity_increment limiter, 0.38 buoyancy scale, 1.35 omega-to-w metric, hyperdiffusion, Rayleigh damping, divergence damping, time-averaging like t_2ave/ww/muave), cite:
   - The MPAS or WRF source line that defines the canonical form
   - The Pace or ICON4Py source line that ports it (if applicable)
   - Whether ADR-023, ADR-021, or both currently implement it correctly
   - The minimum fix to bring our code in line with canonical
   ≥ 6 rows in the table.

6. **§6 Risk re-assessment.** What does the manager's risk register miss? Specifically address:
   - What if neither ADR-023 nor ADR-021 produces Tier-4 PASS within budget?
   - What is the project's exit strategy if the dycore takes another 4+ sprints?
   - Is there an existing third path (e.g., port a working JAX dycore from another project as the substrate) the manager has not considered?

7. **§7 Recommendation** (exactly one):
   - `RATIFY-MANAGER-PLAN` — manager's plan is the best path. List the 1-2 amendments needed.
   - `RATIFY-CRITIC-PLAN` — your plan is materially better. Cite the key advantages.
   - `HYBRID` — synthesize specific sprints from both. Specify which from each.
   - `NEITHER` — both have a fundamental flaw. Specify the third option.

   Plus a single paragraph manager-facing summary (≤ 200 words) that the manager can paste into the next status update.

## Required commit step (mandatory)

```bash
cd /tmp/wrf_gpu2_strategy_critic
git switch -c critic/codex/m6x-close-strategy-plan-critic  # already on it
git add .agent/sprints/2026-05-23-m6x-close-strategy-plan-critic/reviewer-report.md
git commit -m "[strategy plan critic] <verdict>"
```

## Validation Commands

None — read-only critic.

## Performance Metrics

N/A — strategy critic.

## Proof Object

- `reviewer-report.md` (4000-7000 words; this is the longest critic sprint in the project — it's worth depth)
- Committed on branch `critic/codex/m6x-close-strategy-plan-critic`

Time budget: **90-150 min**.

## Risks

- **Bias toward the manager's plan**: Stockholm syndrome. Counter: §3 requires explicit diff and §2 requires writing a complete alternative plan from scratch.
- **Vague critique**: every claim cites file:line.
- **Missing the "correct + easiest" framing**: the recommendation in §7 must explicitly evaluate against BOTH correctness AND ease, not just one.
- **Spec-gaming**: the verifiability triple applies. Numerical claims cite file:line; project-named entities (WRF subroutines, MPAS variables, Pace classes) must exist where cited.

## Handoff Requirements

When the commit lands and `reviewer-report.md` is on disk, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [critical-review / m6x-close-strategy-plan-critic / codex] exit=<ec>`.
