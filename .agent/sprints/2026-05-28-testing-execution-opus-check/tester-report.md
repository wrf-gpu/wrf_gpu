# Tester Report — Sprint #4: Testing-Execution Opus Check

**Role**: tester (Claude Opus 4.7, acting as sonnet-test-engineer)
**Sprint**: `2026-05-28-testing-execution-opus-check`
**Branch**: `tester/opus/testing-execution-opus-check`
**Worktree**: `/tmp/wrf_gpu2_op4check`
**Generated**: 2026-05-28

## Objective

Read every proof object from Sprint #3 RE-DO
(`.agent/sprints/2026-05-27-testing-plan-execution-redo/`), audit each
verdict against the on-disk evidence and the `test_plan_revised.md`
thresholds, triage SKIP/FAIL items against the `novelty_bounds.md` claim
ceiling, and render a binding publishability verdict for v0.0.1.

This is a pure-judgement sprint per the contract (no code changes, no
GPU runtime, no test execution, CPU pinning `taskset -c 0-3`). The
role-prompt template's generic "rerun validation commands / add tests"
instructions are explicitly superseded by the contract's pure-judgement
hard rules.

## Deliverables produced

| AC | File | Purpose |
|---|---|---|
| AC1 | `per_test_review.md` | Per-proof verdict review (10 items + cross-cutting honesty audit). |
| AC2 | `skip_fail_triage.md` | MUST FIX / DOCUMENT / OUT_OF_SCOPE classification per SKIP/FAIL. |
| AC3 | covered inside AC1 (necessary-vs-nice-to-have mapping table) and AC4 (rationale section). |
| AC4 | `publishability_decision.md` | **Binding verdict: PUBLISHABLE_AS_IS** under Option-2 framing precondition; must-do list; Limitations wording. |
| AC5 | `paper_rewrite_input.md` | Tight lift-and-drop summary for Sprint #5 (Results, Limitations, "what this does NOT claim", Abstract template, title shortlist). |
| AC6 | `tester-report.md` (this file) | Tester-role wrap-up with Decision token. |

All deliverables are inside the contract's allowed scope
(`.agent/sprints/2026-05-28-testing-execution-opus-check/**`). No files
outside the sprint directory were touched. No code under `src/`,
`scripts/`, or `publish/` was modified. No governance files were touched.

## Inputs reviewed

Read in order per the role prompt:

1. `PROJECT_CONSTITUTION.md` — immutable end goal anchored on Canary 3 km/1 km
   GPU-resident regional NWP; "physics correctness precedes speed claims".
2. `AGENTS.md` — operating rules (no done claim without proof object;
   honest decision-oriented reports).
3. `CLAUDE.md` — project-local skills authoritative; no destructive auto-accept.
4. The sprint-contract for this sprint (`sprint-contract.md`).
5. Sprint #3 RE-DO sprint-contract.md + aggregate_report.{md,json} +
   worker-report.md + all 10 HIGH-priority proof JSONs.
6. `novelty_bounds.md` (the load-bearing claim ceiling).
7. `PAPER-REWRITE-FRAMING-MEMO.md` (the editorial brief Sprint #5
   inherits).
8. `test_plan_revised.md` (the threshold table).
9. `multi_agent_framing.md` and `opus_pre_dispatch_spot_check.md` from the
   history-research sprint (contextual).

## Methodology

- For each of the 10 proof objects: ask "does the verdict match the
  numbers on disk?", "is the SKIP/FAIL token honestly characterised?",
  "is the threshold-vs-value comparison correct?", "do referenced
  artifacts exist?". No GPU runs, no fresh measurements; pure on-disk
  audit.
- For the triage: anchor every classification to the
  `novelty_bounds.md` Option-2 claim, not to wishful-thinking framing.
- For the publishability verdict: weigh the principal's stated intent
  ("finish this perfectly clean now") against the honest skill gap
  the proof objects reveal; refuse both rubber-stamp and perfectionism.

## Key findings — short form

1. **No fabricated evidence**. Every verdict in the aggregate report
   matches the underlying JSON numbers. Honesty notes inside each JSON
   accurately describe what was and was not done. The worker report
   does not outrun the proof objects.

2. **One PASS, four FAIL, five SKIP**. The PASS (DETERMINISM-REPEAT) is
   real and bitwise. Of the four FAILs:
   - SAVEPOINT-PARITY-DEEP is a depth-stretch miss (step-100 PASSES,
     step-1000/10000 not run). Not a correctness regression.
   - CONSERVATION-MASS-24H has Canary 24 h uncorrected drift 4.81e-6
     (below the 1e-5 corrected-drift threshold). Operationally healthy;
     formal closed-domain gate is what failed.
   - CONSERVATION-ENERGY-24H has a finite, bounded proxy diagnostic
     (3.09 % drift); missing the CPU envelope is what failed.
   - CANARY-MULTIDAY-SIDE-BY-SIDE is the single strongly negative
     result: GPU RMSE is +161 % to +378 % vs CPU on T2/U10/V10 across
     3 complete days. This is openly disclosed in the JSONs and the
     aggregate report and is reproduced verbatim in
     `skip_fail_triage.md` and `paper_rewrite_input.md`.
   The five SKIPs are all "no GPU forecast runner under sprint scope"
   tokens; IC builders exist and are finite-checked.

3. **The Option-2 claim ceiling is the only defensible novelty wording**.
   Under Option 2 ("source-open WRF-compatible Python/JAX/XLA
   workstation prototype with whole-state device residency"), the
   necessary evidence (determinism, savepoint parity to v0.0.1 depth,
   D2H-invariant, Canary pipeline functioning) is on disk. Under
   Option 1 ("first fully source-open full-physics WRF GPU port") the
   same evidence would be insufficient.

4. **No item is publication-blocking under Option 2**. Of the nine
   non-PASS items, all nine are DOCUMENT-as-known-gap; zero are
   MUST FIX; zero are OUT_OF_SCOPE. The Canary skill regression
   triggers mandatory placement in Abstract + Results + Limitations +
   Discussion but does not block publication under the framing memo's
   explicitly-accepted honest gap.

## Risks, gaps, edge cases I considered

- **Risk of rubber-stamp**: the principal said "finish this perfectly
  clean now" and the framing memo accepts the skill gap. A timid
  reviewer would parrot "PUBLISHABLE_AS_IS" without surfacing the
  precondition. I mitigated by making the Option-2 framing precondition
  binding inside the verdict and by enumerating the must-do list (M-1
  through M-7) and the verbatim Limitations text.

- **Risk of perfectionism**: a strict reader could demand the warm-bubble
  integrator, the CPU envelope, and the 14-day Canary corpus before any
  release. Under Option 2 framing this is not the published claim, so the
  evidence base is sufficient. The publishability decision explicitly
  rejects DEFER_PUBLICATION for that reason.

- **Risk of misreading PASS coverage**: the DETERMINISM-REPEAT proof is
  a 1 h Canary segment, not a full 24 h pipeline. The framing memo and
  `paper_rewrite_input.md` mandate the precise wording so a reader does
  not over-claim.

- **Risk that the Canary skill regression masks a deeper defect**: the
  per-step bitwise savepoint parity to 100 steps (passes) plus the
  bitwise reproducibility (passes) together suggest the defect is not
  in the dycore but downstream — surface-flux coupling and theta-guard
  saturation, as the framing memo states. This is consistent with the
  proof-object pattern (dycore guardrails green; operational skill red).

- **Edge case checked**: the conservation_mass_24h.json contains an
  embedded skill-diff sub-object that duplicates the Canary multiday
  numbers. I confirmed those numbers are inherited from the canary case
  source manifest and are not a fabricated re-measurement; the verdict
  for mass-24h is set by mass-conservation thresholds, not by the
  embedded skill diff.

- **Edge case checked**: `determinism_repeat.json` has
  `pipeline_verdict: PIPELINE_PARTIAL` for each run and short
  `forecast_wall_s` ≈ 5.88 s. The PARTIAL classification corresponds to
  a single 19:00:00 wrfout per run, not a 24 h pipeline. Determinism
  is therefore correctly demonstrated over that one-hour segment; I
  required the paper rewrite to phrase it precisely.

## Tests added or run

Per the sprint contract's hard rule "no code changes; pure judgement +
writing", I did not add tests, did not modify `tests/`, did not run any
fresh GPU execution, and did not re-run the publication-test harness.
The sprint contract supersedes the generic role-prompt template on this
point. All deliverables are markdown judgement artifacts inside
`.agent/sprints/2026-05-28-testing-execution-opus-check/`.

## Fixtures used

None. This sprint reads existing on-disk proof objects and produces
markdown judgements; no fixture binaries were touched. The `data/`
symlink and binary fixtures under `/mnt/data/wrf_gpu2/` were not
accessed.

## Handoff

- **objective**: render a binding publishability verdict for the v0.0.1
  release.
- **files changed**: five new markdown files in
  `.agent/sprints/2026-05-28-testing-execution-opus-check/`
  (per_test_review.md, skip_fail_triage.md, publishability_decision.md,
  paper_rewrite_input.md, tester-report.md).
- **commands run**: none (no GPU, no Python, no shell beyond Read/Write
  via the harness; CPU pinning unused because no compute occurred).
- **proof objects produced**: the five markdown deliverables enumerated
  above; the binding artifact is `publishability_decision.md`.
- **unresolved risks**: the verdict is contingent on Sprint #5 honouring
  the Option-2 framing precondition. If Sprint #5 drifts toward Option-1
  wording, the v0.0.1 evidence base does not support the resulting
  paper claim and Sprint #5's output should be rejected by its tester.
- **next decision needed**: dispatch Sprint #5 (paper rewrite) with
  `publishability_decision.md` and `paper_rewrite_input.md` as binding
  inputs. After Sprint #5, run Sprint #6 (final quality gate + release
  audit). Then v0.0.1 tag and PDF render.

## Decision

Decision: **PUBLISHABLE_AS_IS** for v0.0.1, **under the binding
precondition** that Sprint #5 (paper rewrite) adopts the
`novelty_bounds.md` Option-2 framing verbatim, mirrors the
`PAPER-REWRITE-FRAMING-MEMO.md` directives exactly, and places the
Canary skill regression in Abstract + Results + Limitations +
Discussion as specified.

The decision is honest about gaps, refuses both rubber-stamp and
perfectionism, and lets the v0.0.1 timeline proceed. The detailed
must-do list, the verbatim Limitations text, and the Sprint #5
lift-sheet are recorded in
`publishability_decision.md` and `paper_rewrite_input.md`.
