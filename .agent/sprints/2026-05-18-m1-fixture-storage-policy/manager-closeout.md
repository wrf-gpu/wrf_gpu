# Manager Closeout

Sprint: `2026-05-18-m1-fixture-storage-policy` (M1 S1)
Closed: 2026-05-19
Cycles: 2 worker attempts, 1 tester pass, 2 reviewer passes (binding Accept on attempt 2)

## Outcome

Successful close. The fixture-oracle foundation is in place and the next M1 sprints (analytic stencil, analytic column, Canary WRF-derived) can build on a frozen schema and a working comparison CLI.

## Proof Objects

- `fixtures/manifests/schema.yaml` — pinned manifest schema with required + per-variable fields, including `wrf_version` non-empty constraint when `source == "wrf-derived"`.
- `fixtures/manifests/schema.json` — JSON-Schema mirror.
- `fixtures/manifests/fixture-manifest-template.yaml` — re-pointed to validate against the new schema.
- `docs/fixture-storage-policy.md` — naming, checksums, commit/exclusion rules, external `data/` convention, retention.
- `.gitignore` — extended file-extension blocklist + `data/` symlink exclusion.
- `src/gpuwrf/validation/compare_fixture.py` — Tier-1 NumPy comparison CLI skeleton with 5 frozen flags (`--manifest --candidate --reference --tier --out`) and JSON output schema.
- `scripts/validate_fixture_manifest.py` — CI-callable manifest validator.
- `tests/test_fixture_manifest_schema.py` + `tests/test_compare_fixture_skeleton.py` + `tests/test_fixture_manifest_edge_cases.py` — 25 tests total, all passing.
- Reports: `worker-report.attempt1.md`, `worker-report.md` (attempt 2), `tester-report.md`, `reviewer-report.attempt1.md`, `reviewer-report.md` (attempt 2 Accept).
- Branches: `worker/gpt/m1-fixture-storage-policy` (final integration), `tester/sonnet/m1-fixture-storage-policy`, `reviewer/opus/m1-fixture-storage-policy`.

## Merge Decision

Merge Decision: **Accept and integrate into main**. The worker branch `worker/gpt/m1-fixture-storage-policy` (HEAD: reviewer attempt-2 commit) carries the full sprint integration — worker code (attempt 1 + attempt 2 fix), tester tests, and reviewer reports. Merge this branch into `main` as the sprint integration commit.

## Scope Changes

None. S1 stayed inside the contracted scope on both attempts. Attempt 1 missed one validator/schema parity case (caught by tester, confirmed by reviewer); attempt 2 fixed it inside the same scope. No GPU code, no WRF extraction, no fixture payloads, no profiler artifacts.

## Lessons

1. **Schema and Python validator parity is not automatic.** Worker attempt 1 implemented the schema correctly but the validator treated `wrf_version` as "string, possibly empty." Tester caught this immediately by writing the edge-case test. Future schema-defining sprints should add an explicit "schema-validator parity" acceptance criterion up front so it doesn't need a fix cycle.
2. **The contract's `git ls-files | xargs -I{} stat -c '%s {}' | sort -nr` was malformed.** `xargs -I{} stat -c '%s {}'` expands `{}` only in the format string, not as a stat argument — so stat received no path and printed "missing operand." Both worker and tester independently discovered + worked around this. Corrected to `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5` in the amendment. Note for future sprint-contract authors: stress-test every validation command verbatim before committing the contract.
3. **The reviewer Decision token enforcement (require literal `Decision:`) caught a real semantic difference.** When the reviewer wrote `## Decision\n\nDecision: Accept`, both forms were present; close_sprint detected the token correctly.

## Next Sprint

S2: `m1-analytic-stencil-fixture` — produce one tiny 3D advection-diffusion analytic fixture manifest + payload (≤100 KB sample slice) + comparison-CLI round-trip test. Manager opens this on the next turn.
