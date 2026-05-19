# Goal Condition — M2 Done

Used by the self-paced `/loop` manager to detect M2 (Backend Bakeoff) completion. End state is **objective and machine-checkable**. M2 is done when `python scripts/check_m2_done.py` returns `{"ok": true}`.

## Single-command status check

```bash
python scripts/check_m2_done.py
```

Returns JSON. M2 is done when top-level `ok` is `true` and `errors` is empty.

## Explicit assertions

All must hold:

### A. Repository hygiene
- `python scripts/validate_agentos.py` → `ok: true`.
- `pytest -q` → all pass (target: 45 → 60+, as candidates may add backend-specific tests).
- M1 artifacts still on `main` (no regression of fixtures, schema, CLI).

### B. Sprint completeness
For every `dir = .agent/sprints/2026-*-m2-*/`: `python scripts/close_sprint.py <dir>` → `ok: true`.

### C. Bakeoff coverage (the M2 substance — from `PROJECT_PLAN.md §5` and `.agent/milestones/ROADMAP.md M2`)

The six candidate families: `jax`, `triton`, `gt4py`, `kokkos`, `cupy_or_numba`, `cuda_tile`.

For **each** candidate, either:
- Both `artifacts/m2/<candidate>/stencil_profile.json` AND `artifacts/m2/<candidate>/column_profile.json` exist and match `PERFORMANCE_TARGETS.md` schema (kernel_launches, occupancy_pct, registers_per_thread, local_memory_bytes, host_device_transfer_bytes, wall_time_s, artifact_paths), **AND** `artifacts/m2/<candidate>/correctness.json` against the M1 fixtures with `pass: true`,
- **OR** for any candidate that cannot produce a profile: `artifacts/m2/<candidate>/<problem>_failure.json` per the candidate-failure schema in ROADMAP.md M2 (blocker_category, blocker_summary, logs, reviewer_decision, remediation), with `reviewer_decision in {covered, excluded, escalated}`.

Plus, per candidate (whether passed or failed):
- `artifacts/m2/<candidate>/maintainability.md` (≤300 words; build complexity, error legibility, debugger story).
- `artifacts/m2/<candidate>/agent_success.json` (sprint_count, reviewer_rejections, escalation_events).

### D. ADR-001 (backend lock)
- `.agent/decisions/ADR-001-backend-selection.md` exists, ≥2000 bytes, includes literal tokens: `Decision:`, `Selected backend:`, `Dissent`, `Evidence summary`.
- The selected backend is one of the six candidate families (or explicitly "deferred" with rationale).
- A Codex cross-model critical review of the ADR exists at `.agent/decisions/REVIEW-codex-ADR-001.md` per the manager-autonomy directive.

### E. M2 closeout
- `.agent/decisions/MILESTONE-M2-CLOSEOUT.md` exists with: candidate matrix, profiler-evidence summary, agent-success summary, ADR-001 pointer, residual risks, recommended M3 start date.
- `.agent/milestones/M2-backend-bakeoff.md` Reviewer Decision field is `Accepted`.

### F. Cross-AI verification provenance
- Each per-candidate sprint's `tester-report.md` was produced by **Claude Opus 4.7** (not codex). Verifiable by grepping `via claude` in completion-helper history OR the role-prompts/tester.md header naming the AI.

### G. Bounds (so a stuck loop can't run forever)
- Per-candidate retry cap: 5 worker attempts before manager writes `.agent/decisions/BLOCKER-m2-<candidate>.md` and excludes that candidate from ADR-001 (with reviewer_decision = excluded).
- Total wall time since M2 loop start ≤ 48 hours. After 48 h, write `.agent/decisions/M2-TIMEOUT.md` summarizing state and stop.
- Total spend on subordinate agent calls ≤ approximately 200 calls. After ~200, write a cost report and stop.

## What the loop should NOT do

- Do not start M3 work. M2 is the only goal of this run.
- Do not change the constitution, scope, spec, architecture principles, validation strategy, precision policy, performance targets, or this goal file.
- Do not commit binary fixture data, profiler binaries (`*.ncu-rep`, `*.nsys-rep`, `*.sqlite`), or any file >100 KB into git. Binary profiler outputs live at `data/profiler_artifacts/`; the committed `*.json` is the parsed summary.
- Do not auto-merge to `main` until reviewer Accepts the per-candidate sprint (the S1/S2/S3 M1 pattern).
- Do not exceed the §G bounds without writing the marker file and stopping.

## Escalation triggers (stop loop, write `BLOCKER-*.md`)

- A candidate's toolchain has no Blackwell/RTX 5090 (cc120) support and cannot be remediated by version-bumping → write candidate-failure.json with `reviewer_decision: excluded`, do NOT stop the whole loop (continue with other candidates).
- Multiple candidates fail simultaneously due to a shared infrastructure issue (CUDA toolkit missing, HDF5 incompatibility, etc.) → stop the loop.
- The M1 oracle regresses (a candidate breaks an M1 fixture) → stop the loop.
- ADR-001 deadlock after cross-model review: two candidates score within 5% on profiler metrics and neither has a clear maintainability winner → write `.agent/decisions/BLOCKER-adr-001-deadlock.md` requesting human-arbiter tiebreak.

Marker file format same as M1-DONE.md §Escalation triggers.
