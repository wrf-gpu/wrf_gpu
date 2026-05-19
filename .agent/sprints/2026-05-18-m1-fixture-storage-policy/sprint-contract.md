# Sprint Contract

Sprint ID: `2026-05-18-m1-fixture-storage-policy`
Milestone: M1 — WRF Oracle & Fixtures
Sequence: S1 (first implementation sprint of M1)
Reviewer: opus-reviewer (required pre-implementation per `.agent/rules/sprint-lifecycle.md` step 3 — this sprint touches validation behavior)
Worker: gpt-kernel-worker (Codex `gpt-5.5` `high`)
Tester: sonnet-test-engineer
Approval status: **AMENDED 2026-05-19 (attempt 2) — reviewer Decision = Reject on attempt 1; see `reviewer-report.md`; amended acceptance criteria below address the two blockers and the file-size-command note.**

### Amendment log
- **2026-05-19 attempt 2**: amended acceptance criterion #1 to require `wrf_version` non-empty when `source == "wrf-derived"` in *both* schema and Python validator; amended Validation Commands to fix the malformed `git ls-files | xargs -I{} stat ...` invocation (use `-print0 | xargs -0 stat -c '%s %n'`); archived attempt-1 work as `worker-report.attempt1.md`.

## Objective

Lay the smallest foundation that lets every later sprint produce, store, validate, and consume fixtures consistently:

1. Pin a **fixture manifest schema** (`fixtures/manifests/schema.yaml` + a JSON-Schema or pydantic validator) extending today's `fixture-manifest-template.yaml`. Fields, types, units, tolerance metadata, source-commit, checksum, license-notes — all frozen for the duration of M1.
2. Define and commit `docs/fixture-storage-policy.md`: naming convention, sha256 checksum policy, what may be committed (manifests, ≤100 KB sample slices, JSON/CSV smoke arrays), what must stay out of git (NetCDF, GRIB, Zarr, binary dumps, profiler dumps), and the external-storage path/URL convention.
3. Skeleton a **comparison harness CLI** at `src/gpuwrf/validation/compare_fixture.py` that, given a manifest and a candidate array, emits a Tier-1 pass/fail JSON record in the schema of `PERFORMANCE_TARGETS.md`-adjacent style. Skeleton means: argparse + manifest load + tolerance check + JSON emit. **No backend code. No actual GPU work. No WRF extraction.**
4. Extend `scripts/validate_agentos.py` (or add a sibling `scripts/validate_fixture_manifest.py`) to validate every committed manifest against the pinned schema as part of CI.

## Non-Goals

- No analytic fixture content (that is sprint S2).
- No column-physics fixture (S3).
- No Canary WRF run or WRF extraction tooling (S4).
- No backend choice; no GPU code; no model code of any kind.
- No tolerance values — only the metadata fields that hold them.
- No new role definitions, no new skills, no new rules.

## File Ownership

Worker may create or edit only these paths in this sprint:

- `fixtures/manifests/schema.yaml` (new)
- `fixtures/manifests/schema.json` (new, JSON-Schema mirror, optional but encouraged)
- `fixtures/manifests/fixture-manifest-template.yaml` (edit: re-point as a thin example that validates against the new schema; filename preserved)
- `docs/fixture-storage-policy.md` (new)
- `.gitignore` (edit: add patterns implementing the storage-policy git-exclusion rules — large fixture file types and external-artifact cache paths)
- `src/gpuwrf/validation/__init__.py` (new if missing)
- `src/gpuwrf/validation/compare_fixture.py` (new)
- `scripts/validate_fixture_manifest.py` (new)
- `tests/test_fixture_manifest_schema.py` (new)
- `tests/test_compare_fixture_skeleton.py` (new)
- `pyproject.toml` (edit only if a new dev-dep like `pydantic` or `jsonschema` is needed; explain in worker report)

Any change outside this list requires manager approval.

## Inputs

- `fixtures/manifests/fixture-manifest-template.yaml` (current placeholder)
- `INTERFACE_CONTRACTS.md` § `FixtureManifest`
- `VALIDATION_STRATEGY.md` § Tier 1
- `PROJECT_PLAN.md` §6 (validation architecture) and §7 (M1 exit criteria)
- `.agent/skills/building-wrf-oracles/SKILL.md`
- `.agent/milestones/M1-wrf-oracle-fixtures-plan.md`

## Acceptance Criteria

All must hold for closeout.

### Schema (`fixtures/manifests/schema.yaml`)

1. Top-level required fields: `fixture_id`, `source` (enum: `analytic` | `wrf-derived`), `source_commit`, `wrf_version` (nullable when `source == "analytic"`; **required AND non-empty (minLength 1) when `source == "wrf-derived"`** — enforced by *both* `fixtures/manifests/schema.yaml` / `schema.json` AND by `validate_manifest()` in `src/gpuwrf/validation/compare_fixture.py`; the test `tests/test_fixture_manifest_edge_cases.py::test_wrf_derived_requires_non_empty_wrf_version` must pass), `scenario`, `created_utc`, `tier` (enum: `1` | `2` | `3` | `4`), `precision_reference` (enum: `fp64` | `fp32` | `bf16` | `fp16`), `generation_command`, `external_uri` (nullable), `sample_slice_path` (nullable; ≤100 KB), `git_commit`, `license_notes`, `variables` (list, non-empty), `files` (list, may be empty for purely external manifests).
2. Each `variables[*]` entry requires: `name`, `units`, `shape` (list of int), `staggering` (enum: `mass` | `u` | `v` | `w` | `m`-stagger appropriate to grid; document in schema), `dtype`, `tolerance_abs` (float, per-variable), `tolerance_rel` (float, per-variable), `tolerance_rationale` (string, ≤200 chars, explaining how the value was chosen), and `tier_overrides` (nullable mapping tier→{tolerance_abs, tolerance_rel} for variables whose tolerance differs by validation tier).
3. Each `files[*]` entry requires: `path`, `checksum_sha256`, `bytes` (int), `external` (bool — `true` if `path` is a URI rather than a relative repo path).
4. Tolerances are **never** top-level — always per-variable. Top-level tolerance fields must be rejected by the validator.
5. The schema must be expressible as JSON-Schema in `schema.json` so non-Python tooling (codex, CI) can validate manifests without importing Python.

### Storage policy (`docs/fixture-storage-policy.md`)

6. Answers, with explicit rules: naming convention, sha256 checksum policy, what may be committed (manifests + ≤100 KB sample slices + JSON/CSV smoke arrays), what must stay out of git (NetCDF, GRIB, Zarr, binary dumps, profiler dumps), external storage path/URI convention (local path *or* placeholder S3-style URI), retention rules.
7. `.gitignore` extended with patterns implementing the "must stay out of git" rule.

### Validator (`scripts/validate_fixture_manifest.py`)

8. `python scripts/validate_fixture_manifest.py <path>` exits 0 on the template; exits non-zero with a human-readable error citing the violating field/path on a deliberately broken manifest.

### Comparison CLI (`gpuwrf.validation.compare_fixture`) — surface frozen by this sprint

9. CLI argument surface (exact names, frozen by this sprint and reusable by M2):
   - `--manifest <path>` (required) — manifest YAML or JSON path.
   - `--candidate <path>` (required) — candidate NumPy `.npz` or `.npy` file mapping variable names to arrays.
   - `--reference <path>` (optional) — reference array file; if omitted, the manifest's `sample_slice_path` is used.
   - `--tier <int>` (optional, default `1`) — selects tier-specific tolerance overrides.
   - `--out <path>` (optional, default stdout) — where the JSON record is written.
10. Output JSON record (exact field set, frozen):
    ```json
    {
      "fixture_id": "...",
      "tier": 1,
      "pass": true,
      "variables": [
        {
          "name": "T",
          "pass": true,
          "shape_ok": true,
          "max_abs_diff": 0.0,
          "max_rel_diff": 0.0,
          "tolerance_abs": 1e-6,
          "tolerance_rel": 1e-6,
          "violation_index": null
        }
      ],
      "first_failure": null,
      "command": "...",
      "schema_version": "1"
    }
    ```
    On a multi-variable failure, `first_failure` names the variable with the largest tolerance breach (by `max_rel_diff / tolerance_rel`, with `max_abs_diff` as tiebreaker), and `variables[*].pass` flags each individually.
11. Behavior: load manifest, validate against schema, load candidate + reference arrays, check shape matches the manifest declaration, compute max abs/rel diff per variable, apply per-variable (and per-tier override) tolerance, emit the JSON record. **No backend code, no fancy NumPy logic beyond `np.max(np.abs(...))` and `np.max(np.abs((a-b)/(|b|+eps)))`, no I/O beyond what is listed.**
12. `python -m gpuwrf.validation.compare_fixture --help` prints meaningful usage including all five flag names above.

### Test suite

13. `tests/test_fixture_manifest_schema.py` covers schema validation: positive case (template validates), negative cases (missing required field, top-level tolerance rejected, malformed `tier_overrides`, bad checksum format, oversized sample slice).
14. `tests/test_compare_fixture_skeleton.py` covers CLI: identity case (`pass: true`), single-variable failure (`pass: false`, `first_failure` populated), shape mismatch (returns non-zero exit code), multi-variable case with one passing and one failing.
15. `pytest -q` passes overall.

### CI and repo hygiene

16. `python scripts/validate_agentos.py` passes.
17. `python scripts/repo_status_snapshot.py` JSON output's `dirty_files` is either empty *or* every entry is a path inside the contract-owned set defined under *File Ownership*. The worker report includes the parsed `dirty_files` list and asserts membership.
18. No committed file exceeds 100 KB. The worker report includes `git ls-files | xargs -I{} stat -c '%s {}' | sort -nr | head -5` to demonstrate.

## Validation Commands

```bash
python scripts/validate_agentos.py
python scripts/validate_fixture_manifest.py fixtures/manifests/fixture-manifest-template.yaml
python -m gpuwrf.validation.compare_fixture --help
pytest -q                                          # must report 0 failures (the wrf_version edge case must now pass)
python scripts/repo_status_snapshot.py             # dirty_files must be a subset of contract-owned paths
git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5   # asserts no committed file > 100 KB
git diff --stat $(git rev-parse HEAD)
```

All commands run from repo root. Output of each is included in the worker report.

## Performance Metrics

Not applicable — this sprint does no GPU work. Zero profiler artifacts expected. Any GPU artifact appearing here is a contract violation and the reviewer must reject the sprint.

## Proof Object

- Diff (limited to the files listed in *File Ownership*).
- Worker report at `.agent/sprints/2026-05-18-m1-fixture-storage-policy/worker-report.md` with: files changed, commands run with their output, decisions made, deviations from contract (none expected), limitations, next risk.
- Tester report at the same folder's `tester-report.md` with independently-rerun validation commands and any added edge-case tests.
- Reviewer report at the same folder's `reviewer-report.md` with severity-ranked findings and decision.
- All schema/policy/CLI files cited above.

## Risks

- **Schema overfitting to current placeholder template.** Mitigation: schema must support both analytic and WRF-derived fixtures from the start. Reviewer challenge: walk through how a Canary WRF fixture (S4) would express itself; if it can't, the schema fails review.
- **CLI skeleton creeping into real comparison logic.** Mitigation: contract explicitly says "skeleton." Reviewer rejects any backend-specific or NumPy-fancy logic beyond bounded tolerance compare.
- **Dependency bloat.** Mitigation: if `pydantic` or `jsonschema` is added, worker reports the marginal install cost and justifies vs. a hand-rolled validator.
- **Schema change after M1 close.** Mitigation: any later schema change requires ADR per `.agent/rules/architecture-decision-policy.md`.

## Handoff Requirements

- Worker pushes to branch `worker/gpt/m1-s1-fixture-storage-policy` (per `.agent/rules/branch-and-worktree-policy.md` naming — `worker/gpt/<sprint>` is the project convention; this slot is filled by Codex but the branch namespace is `gpt`).
- Worker opens no merge until reviewer + tester reports are on disk.
- Manager closeout writes `manager-closeout.md` in the sprint folder, sets reviewer decision in M1-plan to "S1 accepted," and recommends sprint S2 (analytic stencil micro-fixture) as next.
- Memory patch is **not** expected from this sprint — no new project knowledge that wasn't already in the constitution.
