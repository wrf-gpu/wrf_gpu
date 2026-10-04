---
name: wrf-gpu-reviewer
description: Use when reviewing WRF-GPU changes and evidence.
---

# Scoped reviewer

Read `docs/agent-workflow/manager/SKILL.md` from the repository root. Review the
sealed candidate against its base SHA and task contract; do not mutate the
candidate while claiming independent review.

- Check the entire scoped diff, changed-file list and commit metadata. Flag
  scope creep, private identifiers, credentials, absolute local paths and
  unlicensed copied instructions before public publication.
- Check WRF semantics, precision, scheduling, output fields, scientific gates
  and regression coverage. Frozen-limit changes, masking and unexplained clamps
  are blockers, not ways to repair a failing comparison.
- Reconcile claims with raw tester evidence: exact source/input revision,
  commands, exits, missing tests and disclosed known failures. Keep historical
  measurements, current-release validation and projections distinct.
- For benchmarks, check comparable workloads, timing endpoints, cache states,
  concurrency, memory and energy boundaries against `docs/release/METHODS.md`.
- Check portable instructions against real repository paths and command options.
  Do not require a private orchestrator or a particular host to manage a fix.
- Return blocking findings with file/line, reason and evidence; separately list
  suggestions and the bounded scope accepted. If reviewing your own work,
  explicitly label self-review rather than claiming independent approval.

The manager owns integration and release authorization. Reviewer acceptance of
one diff is not proof of full forecast validation or a completed public release.
