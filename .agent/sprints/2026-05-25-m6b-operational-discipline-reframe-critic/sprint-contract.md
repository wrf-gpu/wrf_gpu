# Sprint Contract — Operational-Mode Discipline Reframe Critic (codex GPT-5.5, full-state)

## Objective

Per principal standing order "if stuck, take a step back, plan how to check other angles, try new things, ask GPT about ideas and plan changes" — this critic sprint reviews whether the **Critic Amendment #1 rule** that operational-mode must compose its own variants of validation operators (no imports of `acoustic_loop.py`/`dycore_step.py`/`coupled_step.py`) is still serving the project, given 5 cascading defects in 4 fix sprints.

## Context (history of operational-mode defects)

Validation ladder (M6B0-R → M6B6, 7/7 PASS, worst delta 0.0 bitwise) used helpers `acoustic_wrf.calc_coef_w_wrf_coefficients`, `mu_t_advance.advance_mu_t_wrf`, `tridiag_solve.thomas_*_scan`, `small_step_scratch.update_*`, `acoustic_loop.acoustic_substep_wrf`, `dycore_step.dycore_step_wrf`, `coupled_step.coupled_timestep_wrf`. These are validation-mode-only helpers per Critic Amendment #1 (drafted in PROJECT_PLAN §14.5.1 to protect GPU-optimized-core primacy).

Operational mode was specified to **compose its own WRF-shaped variants** of these operators (copy/adapt with WRF citation per defect). The plan worked for M6B0-R, M6B1, M6B2 individually. But on real Gen2 IC with full RK3 timestep composition, defect cascade:

1. **M6b first attempt (BLOCKER)** — operational strict-subset carry missing 6 scratch families (theta-bounds at 10s)
2. **carry-expansion fix (PARTIAL)** — promoted scratch; 10s probe PASS but worker noted "direct theta promotion went nonfinite at 70s"
3. **M6b RETRY (BLOCKER)** — theta=1343K at step 62 with carry-expanded operational
4. **bisection-then-RK1-fix (synthetic IC)** — found "RK1 acoustic loop omission"; fix `879ef56` works on synthetic-IC but fails on real Gen2 IC
5. **RK1+D2H combined (PARTIAL)** — D2H lift PASS, RK1 parity FAIL on real Gen2 IDs at step 1; B6 still 0.0
6. **sanity check + real-IC bisection** — both name 4 defects: advance_mu_t commit drop + W recompute + dt_sub + ph_tend formula
7. **4-defect fix (FAIL)** — applied all 4; step-1 delta 5.63e14, nonfinite at step 10; **5th defect named**: operational fed raw `state.u/v` while WRF passes mass-coupled `ru_m/rv_m/ww_m`

Throughout: validation-mode `coupled_timestep_wrf` is **0.0 bitwise on 10 timesteps physics+boundary on**.

## The strategic question

Should the **"operational composes its own variants" rule** stay binding, or should we **lift validation `coupled_step.coupled_timestep_wrf` into a shared core that operational imports**, then add the operational-only concerns (carry pruning, fusion, precision downcast, kernel selection) AROUND it?

The pilot-vs-core principle from the principal directive:
> "An efficient gpu version that is close to wrf in terms of results but very well gpu optimized. Pilots are ok if they can be optimized but if it incompatible with a gpu-optimized core, the solution is just wrong for this project."

Was Critic Amendment #1's "operational composes its own variants" a *pilot rule* that's now empirically failing? Or is the fix-by-fix path the right one, with patience?

## Non-Goals

- NO code edits.
- NO sub-sprint dispatch.
- NO ADR promotion.
- NO re-opening of the B-direct validation ladder (it's bitwise-locked).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_reframecritic` on branch `critic/codex/m6b-operational-discipline-reframe`.

Write-only:
- `.agent/sprints/2026-05-25-m6b-operational-discipline-reframe-critic/reviewer-report.md` (2500-5000 words)

Read-only.

## Inputs (mandatory; ALL of these — this is the full-state package)

1. `PROJECT_PLAN.md §14.5.1` (the Amendment #1 rule + binding invariants)
2. `feedback_gpu_optimized_core_primacy.md` (memory — the principal's GPU-optimized-core directive)
3. `.agent/sprints/2026-05-24-m6-speed-vs-bitwise-critic/reviewer-report.md` (the speed-vs-bitwise critic that ratified §14.5.1)
4. All M6b sprint reports in order (carry-fix, RETRY, bisection, RK1-fix, RK1+D2H acceptance, sanity, real-IC bisection, 4-defect fix)
5. `src/gpuwrf/dynamics/coupled_step.py` (validation `coupled_timestep_wrf` — 0.0 bitwise reference)
6. `src/gpuwrf/runtime/operational_mode.py` (operational composer with 5 known defects)
7. `.agent/decisions/ADR-026-operational-mode-design-PROPOSED.md`
8. `.agent/decisions/ADR-027-d2h-invariant-clarification-DRAFT.md`

## Acceptance Criteria

`reviewer-report.md` with **6 labelled sections**:

### §1 — Defect-cascade diagnosis

Is the 5-defect cascade consistent with "operational has a hidden 6th, 7th, 8th defect waiting" OR with "operational mode is structurally diverging from validation in a way that the 'compose own variants' rule cannot bridge"?

Cite specific worker reports.

### §2 — The 'compose own variants' rule

Was Amendment #1 the right rule at the time it was drafted? What was the failure mode it was trying to prevent (CPU-shape creep into operational)? Is that failure mode still real, or has the discipline of the validation ladder made it moot?

### §3 — Pilot-vs-core trade

Per principal directive, pilots are OK if optimizable; incompatible-with-GPU-core pilots are wrong. Validation `coupled_step.coupled_timestep_wrf` is currently labelled validation-only. Is the validation helper:
- (a) A pilot that's incompatible with a GPU-optimized core (then it MUST stay validation-only and operational must compose its own — current state)
- (b) A pilot that IS optimizable (then operational can import it and add GPU concerns as wrappers)

For each call site in `coupled_step.coupled_timestep_wrf` and its transitive dependencies, judge optimizability: can it be `lax.scan`'d, kernel-fused, precision-downcast in operational mode without changing call semantics?

### §4 — Proposed restructure (if you recommend reframe)

If reframe: propose a specific module restructure. E.g.:
- `src/gpuwrf/dynamics/core/` (NEW) — shared pure-numerical operators imported by BOTH validation and operational
- `src/gpuwrf/dynamics/validation_wrappers.py` — wraps core with savepoint emission, sanitizer-friendly modes
- `src/gpuwrf/runtime/operational_mode.py` — wraps core with carry pruning, fusion, precision, kernel selection

Show how the 5 defects collapse if validation's `coupled_timestep_wrf` is imported into operational vs hand-composed.

### §5 — Risk of staying the course

If we stay with "operational composes own variants" and dispatch the 5th-defect fix (ru_m/rv_m/ww_m): estimate the number of remaining hidden defects. Best evidence: how many distinct operator-interface mismatches exist between `operational_mode.py` and `coupled_step.py`?

### §6 — Recommendation

ONE of:
- `STAY-THE-COURSE` — dispatch ru_m/rv_m/ww_m fix; estimate N more defect rounds; provide kill gate
- `REFRAME-TO-SHARED-CORE` — propose specific restructure; provide 1-sprint impl plan; cite WRF + Amendment #1 supersession
- `HYBRID` — keep validation helpers validation-only but lift the COMPOSITION pattern (RK loop / acoustic loop structure) into a shared template; operational instantiates with its own operators
- `PIVOT-TO-VALIDATION-AS-OPERATIONAL` — ship validation `coupled_step` as the operational dycore; add the GPU-optimization wrapper around it

Plus one paragraph dissent.

## Validation Commands

None — read-only critic.

## Risks

- Spec-gaming: every claim cites file:line.
- Confirmation bias: do not just rubber-stamp the manager's "reframe" preference; argue both sides honestly.
- Sunk-cost protection: do not minimize the 5-defect history.

## Handoff Requirements

Commit + `/exit`. Manager reads + dispatches per recommendation.

Time budget: **60-120 min**.
