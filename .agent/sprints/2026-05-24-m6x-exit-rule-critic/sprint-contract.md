# Sprint Contract — M6.x Exit-Rule Critic (full-state GPT-5 critique)

## Objective

S2.1-redo (commit `4b97743`) finally produced a real Gen2 d02 1h baseline on the current ADR-023 unified operator. The numbers are catastrophic:

| Field | 1h RMSE | Gen2 24h noise floor | Ratio |
|---|---|---|---|
| T2  | 136.885 K | 0.628 K | 218× |
| U10 | 106.419 m/s | 1.456 m/s | 73× |
| V10 | 102.232 m/s | 1.591 m/s | 64× |

Plus theta = 550 K at step 3600 (post-sanitize); 17.2 BILLION nonfinite candidates per 1h run; the operator is hemorrhaging nonfinites and only the tanh sanitizer + `_mu_continuity_increment` are keeping it finite.

The strategy critic's exit-rule from the HYBRID plan said: "after S3 + 1 localized fix, if neither path passes Tier-3, force a manager decision memo with three bounded options: WRF scratch hybrid / full WRF small-step / third-path substrate scout."

This sprint is the **GPT-5 critique with full state** to decide: is this **exit-rule territory**, or is S3-real (`_mu_continuity_increment` replacement + sourced damping) plausibly sufficient?

## Non-Goals

- No code changes. Read-only.
- No sub-sprint dispatch.
- No promotion of any ADR.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_exitcritic` on branch `critic/codex/m6x-exit-rule-critic`.

Write-only:
- `.agent/sprints/2026-05-24-m6x-exit-rule-critic/reviewer-report.md`

Read-only everywhere else.

## Inputs

You MUST read all of these (they form the full-state package the manager has assembled):

**The new evidence (read first):**
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/worker-report.md`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/findings_real.md`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/s3_input_memo_real.md`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_field_rmse_timeline.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_sanitizer_audit.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_bound_violation_tracer.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_operator_term_budget_tracer.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_vertical_column_phase_space.json`

**The HYBRID plan + exit rule:**
- `.agent/sprints/2026-05-23-m6x-close-strategy-plan-critic/reviewer-report.md` (the HYBRID verdict + exit rule)
- `.agent/decisions/manager-reflections/PLAN-REFLECTION-2026-05-24.md`

**The architecture state:**
- `.agent/decisions/ADR-023-conservative-column-solver.md` (PROPOSED — current operator)
- `.agent/decisions/ADR-024-warm-bubble-gate-policy.md` (PROPOSED — gate policy)
- `.agent/decisions/source_mining_operator_table.md` (S1 — operator term provenance)

**Prior architectural dead-ends (proven non-viable):**
- `.agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/worker-report.md` (ADR-021 strip = theta blowup 22,000 K at step 1)

**Cleanup state (post-S3-narrow):**
- `.agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/worker-report.md` (stabilizers 28→20 experiment, 8→37 source)

**Reference projects (for third-path option):**
- Critic's prior §6 risk re-assessment in `.agent/sprints/2026-05-23-m6x-close-strategy-plan-critic/reviewer-report.md` (Dinosaur IMEX, ICON4Py port options)

## Acceptance Criteria

`reviewer-report.md` with **6 labelled sections**:

### §1: Diagnosis of the new baseline numbers

What's the most likely root cause of:
- 218× T2 RMSE vs Gen2
- 17 BILLION sanitized nonfinites per 1h run
- theta = 550 K at step 3600

Is this:
(a) A specific operator bug (e.g., missing factor of 100 in a coefficient, sign error in horizontal-vertical coupling)
(b) A boundary-forcing mismatch (e.g., wrong unit conversion when reading Gen2 wrfbdy)
(c) Fundamental architectural inadequacy (the conservative column solver + small carry genuinely cannot replicate WRF acoustic+thermo dynamics)
(d) An instrumentation artifact (the field-RMSE-timeline sidecar is over-counting)

Cite specific lines in the proof JSONs + operator code.

### §2: Is S3-real plausibly sufficient?

S3-real's plan (from prior s3_input_memo_real.md): replace `_mu_continuity_increment` with WRF MUAVE/MUTS/ww sourced damping; replace `MPAS_OMEGA_TO_W_METRIC = 1.35` with per-level metric; demote `MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = 0.38` to slice-only.

Could this realistically improve T2 RMSE from 136 K to within 5× Gen2 noise floor (~3 K) in 1-2 sprints? Or is the gap so large that these targeted fixes are insufficient?

### §3: Exit-rule status

The HYBRID exit-rule fires when "neither path passes Tier-3 within budget." The current ADR-023 path is far from passing anything; ADR-021 is dead (22,000 K theta blowup). Does the exit-rule fire now, or are we still pre-exit-rule because S3-real hasn't been tried yet?

### §4: Recommended next action

ONE of:
- `CONTINUE-HYBRID-WITH-S3-REAL`: dispatch S3-real as planned. List which fixes likely give the largest gain. Worst-case fallback if S3-real also fails.
- `DISPATCH-OPERATOR-BUG-HUNT`: 218× RMSE smells like a specific bug (factor of 100? unit error?). Dispatch a bug-hunt sprint before any architectural change. Specify what to grep for.
- `DISPATCH-THIRD-PATH-SCOUT`: time for ADR-025 / third-path scout (Dinosaur IMEX, ICON4Py port, or other). Justify why ADR-023 + ADR-021 are both exhausted.
- `M6-DYCORE-BLOCKER-MEMO`: write the formal blocker memo + escalate to the user. Specify what the user must decide.

### §5: Bug-hunt grep targets (if §4 is OPERATOR-BUG-HUNT)

Specific code locations to investigate: file:line ranges in `src/gpuwrf/dynamics/`. Suspect patterns: unit-conversion factors near 100, missing/extra factors of g, wrong sign in coupling. Estimated hours to disprove or confirm.

### §6: Open questions for the manager

What evidence would change your recommendation? What additional diagnostic from S1 sidecars would refine the answer?

## Validation Commands

None — read-only critic.

## Performance Metrics

N/A.

## Proof Object

- `reviewer-report.md` (3000-5000 words)
- Committed on branch `critic/codex/m6x-exit-rule-critic`

Time budget: **60-120 min**.

## Risks

- **Premature exit-rule call**: jumping to blocker memo before trying S3-real wastes a known fix path.
- **False-positive bug hunt**: not every catastrophic number is a single-bug-fix. The architecture step-back may indeed be the right call.
- **Spec-gaming**: every claim cites file:line OR proof JSON path.

## Handoff Requirements

Commit + `/exit`. Wrapper sends AGENT REPORT to manager pane (session 2 via patched dispatch wrapper).
