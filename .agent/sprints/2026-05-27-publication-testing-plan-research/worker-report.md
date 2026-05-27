# Worker Report - Publication Testing Plan Research

Summary: PLAN_READY. I produced the community acceptance criteria memo, gap analysis, and executable test plan for positioning `wrf_gpu` as an open-source JAX/Python GPU-native WRF-compatible port. The plan is deliberately stricter than the current paper draft: it treats the 22.26x speedup, zero inter-kernel D2H, bitwise restart, and B6 savepoint parity as real systems evidence, but it also records that the one-day side-by-side AEMET skill comparison still fails and cannot support an operational replacement claim.

## Five-Line Priority Summary

1. HIGH: add three idealized cases: warm bubble, density current, and mountain wave.
2. HIGH: add 24 h dry-mass and energy budget proofs.
3. HIGH: add at least one stock WRF ideal benchmark with CPU WRF reference.
4. HIGH: extend the one-day AEMET comparison to a 7-10 case multi-regime Canary corpus.
5. HIGH: make the public release auditable with license, citation, install docs, tutorial, CI, and proof manifest.

## Files Changed

- `.agent/sprints/2026-05-27-publication-testing-plan-research/community_acceptance_criteria.md`
- `.agent/sprints/2026-05-27-publication-testing-plan-research/gap_analysis.md`
- `.agent/sprints/2026-05-27-publication-testing-plan-research/test_plan.md`
- `.agent/sprints/2026-05-27-publication-testing-plan-research/worker-report.md`

## Commands Run And Output

Context reads, all read-only:

- `sed -n '1,240p' PROJECT_CONSTITUTION.md`
  - stdout: project constitution loaded; non-negotiables include physics before speed, GPU residency, WRF-compatible interfaces, and rule/contract governance.
  - stderr: empty.
- `sed -n '1,260p' AGENTS.md`
  - stdout: repository instructions loaded; local `.agent/skills` authoritative; do not use legacy global `wrf-gpu-port`.
  - stderr: empty.
- `sed -n '1,260p' CLAUDE.md`
  - stdout: project-local skills and non-destructive action rules loaded.
  - stderr: empty.
- `sed -n '1,300p' PROJECT_PLAN.md`
  - stdout: project plan loaded; key validation, M6/M7, and operational-mode constraints reviewed.
  - stderr: empty.
- `sed -n '1,300p' .agent/milestones/ROADMAP.md`
  - stdout: milestone proof-object expectations loaded.
  - stderr: empty.
- `sed -n '1,260p' .agent/goals/M1-DONE.md`
  - stdout: active goal read only; not modified.
  - stderr: empty.
- `sed -n '1,320p' .agent/sprints/2026-05-27-publication-testing-plan-research/sprint-contract.md`
  - stdout: accepted file ownership and hard rules; no code, no measurements, no remote push.
  - stderr: empty.
- `taskset -c 0-3 sed -n '1,260p' .agent/skills/writing-gpu-kernels/SKILL.md`
  - stdout: kernel validation and transfer-discipline guardrails loaded.
  - stderr: empty.
- `taskset -c 0-3 sed -n '1,320p' .agent/skills/writing-execplans/SKILL.md`
  - stdout: execution-plan structure loaded.
  - stderr: empty.
- `taskset -c 0-3 git status --short --branch`
  - stdout: `## worker/gpt/publication-testing-plan-research`
  - stderr: empty.
- `taskset -c 0-3 rg --files ...` and `find ...`
  - stdout: located research brief, M6/M7 closeouts, M7 skill/speed proof objects, test coverage, fixture manifests, and related sprint reports.
  - stderr: empty.
- `taskset -c 0-3 sed -n ... publication/research_brief/english_brief.txt`
  - stdout: research brief citations and NWP validation sections reviewed.
  - stderr: empty.
- `taskset -c 0-3 rg -n "22\\.26|speedup|D2H|restart|AEMET|B6|savepoint|V3" ...`
  - stdout: found current proof references, including M7 amendment, paper draft, and M6/M7 reports.
  - stderr: empty.
- `taskset -c 0-3 sed -n ... .agent/decisions/MILESTONE-M7-CLOSEOUT*.md`
  - stdout: original M7 closeout and amendment reviewed; amendment supersedes inflated 156.82x claim and records skill failure.
  - stderr: empty.
- `taskset -c 0-3 sed -n ... .agent/sprints/2026-05-27-m7-skill-fix-iter2/worker-report.md`
  - stdout: current corrected speedup `22.25579686534753` and failed skill gate reviewed.
  - stderr: empty.
- `taskset -c 0-3 sed -n ... .agent/sprints/2026-05-27-m7-profiler-window-fix/worker-report.md`
  - stdout: D2H audit PASS reviewed.
  - stderr: empty.
- `taskset -c 0-3 sed -n ... .agent/sprints/2026-05-27-m7-restart-continuity/worker-report.md`
  - stdout: bitwise restart proof reviewed.
  - stderr: empty.
- `taskset -c 0-3 sed -n ... .agent/sprints/2026-05-25-m6b6-coupled-step-parity/worker-report.md`
  - stdout: B6 coupled-step parity PASS and validation-only scope reviewed.
  - stderr: empty.
- `taskset -c 0-3 sed -n ... .agent/sprints/2026-05-27-gemini-agy-review/review.md`
  - stdout: independent review concerns loaded; validation sample size and publication framing are major blockers.
  - stderr: empty.

Web research:

- Search queries for WRF idealized cases, NCAR/WRF verification, METplus, FSS, SAL, PyCECT, ARW technical note, Pace, ICON GPU, SCREAM, DCMIP, FAIR/public-release norms.
  - stdout: source candidates used only where aligned with the local brief; unverified items in the written plan are marked `[verify before use]`.
  - stderr: not applicable.

Validation after edits:

- `taskset -c 0-3 git diff --check`
  - exit 0; stdout empty; stderr empty.
- `taskset -c 0-3 wc -c .agent/sprints/2026-05-27-publication-testing-plan-research/worker-report.md`
  - most recent stdout before this report-line update: `8307 .agent/sprints/2026-05-27-publication-testing-plan-research/worker-report.md`
  - stderr: empty.
- `taskset -c 0-3 rg -n "Summary:|PLAN_READY" .agent/sprints/2026-05-27-publication-testing-plan-research/worker-report.md`
  - final stdout: line 3 contains `Summary: PLAN_READY`; lines 95-96 are this command/output entry.
  - stderr: empty.
- `taskset -c 0-3 LC_ALL=C rg -n "[^[:ascii:]]" .agent/sprints/2026-05-27-publication-testing-plan-research`
  - exit 127; stdout empty; stderr: `taskset: failed to execute LC_ALL=C: No such file or directory`. Command form was invalid because `LC_ALL=C` must be passed through `env`.
- `taskset -c 0-3 env LC_ALL=C rg -n "[^[:ascii:]]" .agent/sprints/2026-05-27-publication-testing-plan-research`
  - exit 0; stdout showed only pre-existing non-ASCII in `sprint-contract.md`.
  - stderr: empty.
- `taskset -c 0-3 env LC_ALL=C rg -n "[^[:ascii:]]" .agent/sprints/2026-05-27-publication-testing-plan-research/community_acceptance_criteria.md .agent/sprints/2026-05-27-publication-testing-plan-research/gap_analysis.md .agent/sprints/2026-05-27-publication-testing-plan-research/test_plan.md .agent/sprints/2026-05-27-publication-testing-plan-research/worker-report.md`
  - exit 1; stdout empty; stderr empty. This means no non-ASCII matches in the new deliverables.

## Proof Objects Produced

- `.agent/sprints/2026-05-27-publication-testing-plan-research/community_acceptance_criteria.md`
- `.agent/sprints/2026-05-27-publication-testing-plan-research/gap_analysis.md`
- `.agent/sprints/2026-05-27-publication-testing-plan-research/test_plan.md`
- `.agent/sprints/2026-05-27-publication-testing-plan-research/worker-report.md`

## Risks

- Some citation targets from the supplied research brief appear to require final verification before manuscript use; the plan marks those as `[verify before use]`.
- The test plan gives executable commands for scripts that the next execution sprint must write; this sprint was explicitly research and planning only.
- CPU WRF reference generation may exceed the 24 GPU-hour target even though the GPU workload stays under budget.
- The current project evidence is strong for systems behavior but still weak for meteorological skill; this is a publication framing risk, not a wording issue.

## Handoff

- objective: research community expectations and write a concrete publication-grade testing plan for `wrf_gpu`.
- files changed: four files listed above, all inside the sprint directory.
- commands run: listed above with summarized stdout/stderr.
- proof objects produced: the three required planning artifacts plus this report.
- unresolved risks: citation verification, CPU reference availability, missing idealized benchmarks, missing energy budget, missing multi-regime skill corpus.
- next decision needed: dispatch an execution sprint that freezes file ownership for the HIGH-priority tests in `test_plan.md`; do not publish operational claims until the skill gap is resolved or explicitly framed as a limitation.
