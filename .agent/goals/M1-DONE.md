# Goal Condition — M1 Done

Used by the self-paced `/loop` manager to detect M1 completion. The end state is **objective and machine-checkable**: every assertion below must hold simultaneously for M1 to be considered complete. The manager runs `scripts/check_m1_done.py` each turn and stops the loop when its `ok` field is `true`.

## Single-command status check

```bash
python scripts/check_m1_done.py
```

Returns JSON. M1 is done when the top-level `ok` is `true` AND `errors` is empty.

If `scripts/check_m1_done.py` does not yet exist, fall back to the explicit assertions below.

## Explicit assertions (used if the script is unavailable)

All must hold:

### A. Repository hygiene
- `python scripts/validate_agentos.py` → JSON `{"ok": true, "errors": []}`.
- `pytest -q` → all pass.
- `python scripts/repo_status_snapshot.py` → JSON `dirty_files` contains only paths inside sprint folders that are themselves complete.

### B. Sprint completeness (all M1 sprints closed)
For every directory matching `.agent/sprints/2026-*-m1-*/`:
- `python scripts/close_sprint.py <dir>` → JSON `{"ok": true}`.

### C. M1 proof objects (from `MILESTONES.md` M1 + `ROADMAP.md` M1 + `PROJECT_PLAN.md §7`)
- `fixtures/manifests/schema.yaml` exists.
- `fixtures/manifests/schema.json` exists (JSON-Schema mirror).
- `fixtures/manifests/fixture-manifest-template.yaml` validates: `python scripts/validate_fixture_manifest.py fixtures/manifests/fixture-manifest-template.yaml` exits 0.
- `docs/fixture-storage-policy.md` exists, non-empty.
- `src/gpuwrf/validation/compare_fixture.py` exists; `python -m gpuwrf.validation.compare_fixture --help` exits 0.
- At least **one analytic stencil micro-fixture manifest** in `fixtures/manifests/` with `source: analytic` and validates against the schema.
- At least **one analytic column micro-fixture manifest** in `fixtures/manifests/` with `source: analytic` and validates against the schema.
- At least **one Canary WRF-derived fixture manifest** in `fixtures/manifests/` with `source: wrf-derived`, validates against the schema, references `wrf_version: 4.7.1`, and points to data under `data/fixtures/` (the symlink to `/mnt/data/wrf_gpu2/fixtures/`).

### D. Reviewer + tester independence
- Every M1 sprint folder contains `reviewer-report.md` and `tester-report.md`, both non-empty, both committed by different agent runs (worker, then tester, then reviewer — three separate `worker/gpt/...`, `tester/sonnet/...`, `reviewer/opus/...` branches per the branch policy).
- Every reviewer report contains the literal token `Decision: Accept` or `Decision: Accept with required fixes` (followed by completed-fix evidence).

### E. M1 closeout artifact
- `.agent/milestones/M1-wrf-oracle-fixtures-plan.md` "Reviewer Decision" field reads `Accepted` (was `Pending`).
- `.agent/decisions/MILESTONE-M1-CLOSEOUT.md` exists and includes: summary, list of sprint closeouts, list of proof objects with paths, residual risks, recommended next milestone start date.

### F. Bounds (so a failed loop can't run forever)
- Total per-sprint codex/claude retries ≤ 5. After 5, the sprint is marked **stalled**; the manager either rewrites the contract once and retries, or escalates by stopping the loop with a `BLOCKER-*.md` file in `.agent/decisions/`.
- Total elapsed wall time since loop start ≤ 12 hours. After 12 h, write `.agent/decisions/M1-TIMEOUT.md` summarizing state and stop.
- Total spend on subordinate agent calls ≤ approximately 100 codex/claude calls. After ~100, write a cost report and stop.

## What the loop should NOT do

- Do not start M2 (backend bakeoff) under any circumstance. M1 is the *only* goal of this run.
- Do not modify the constitution, scope, spec, architecture principles, validation strategy, precision policy, performance targets, or this goal file.
- Do not commit binary fixture data into git. All binary fixtures live under `data/` (symlinked to `/mnt/data/wrf_gpu2/`). Only manifests + ≤100 KB sample slices are tracked.
- Do not auto-merge to `main`. Sprint branches stay open until the manager closeout sprint integrates them in a single integration commit.
- Do not exceed the §F bounds without writing the corresponding marker file and stopping.

## Escalation triggers (stop and write a `BLOCKER-*.md` file)

Stop the loop and report by leaving a marker file under `.agent/decisions/` if any of these occur:
- A sprint cannot be defined without an architectural decision unforeseen at plan time.
- A worker repeatedly hallucinates fixtures with no WRF baseline grounding.
- The `/mnt/data/` storage is exhausted (`df -h /mnt/data` shows <50 GB free).
- WRF baseline binary at `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` is missing or non-executable.
- AgentOS validation fails after a sprint closeout and the manager cannot repair it in one retry.

Marker file format:
```markdown
# BLOCKER — <one-line summary>

Date: <UTC date>
Sprint: <sprint-id or "none">
Class: scope / architecture / data / infrastructure / agent-failure
Evidence: <file paths, command outputs>
Manager's attempted mitigation: <what was tried>
Recommended human action: <what the user needs to do or decide>
```
