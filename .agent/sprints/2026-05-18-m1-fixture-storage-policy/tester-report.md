# Tester Report

## Tests Added Or Run
- Re-ran every validation command listed in the sprint contract from repo root on branch `tester/sonnet/m1-fixture-storage-policy`.
- Added `tests/test_fixture_manifest_edge_cases.py` with edge coverage for:
  - WRF-derived manifests with empty `wrf_version`.
  - CLI default reference loading from `sample_slice_path`.
  - Tier-specific tolerance override selection.
  - Rejection of `.npy` candidate input for multi-variable manifests.
  - Missing candidate variable error reporting.

## Results
- `python scripts/validate_agentos.py` passed: `ok: true`, 31 required files checked, 13 skills checked.
- `python scripts/validate_fixture_manifest.py fixtures/manifests/fixture-manifest-template.yaml` passed.
- `python -m gpuwrf.validation.compare_fixture --help` passed and showed all five frozen flags: `--manifest`, `--candidate`, `--reference`, `--tier`, `--out`.
- Baseline `pytest -q` before tester edge tests passed: `20 passed in 1.08s`.
- `pytest -q tests/test_fixture_manifest_edge_cases.py` failed intentionally on the new contract edge: `1 failed, 4 passed`.
- Full `pytest -q` after tester edge tests failed: `1 failed, 24 passed`.
- `python scripts/repo_status_snapshot.py` passed with `ok: true`; dirty files include the new tester test plus pre-existing sprint-control and `.claude` files.
- Exact contract command `git ls-files | xargs -I{} stat -c '%s {}' | sort -nr | head -5` is malformed and prints repeated `stat: missing operand`, so it is not a valid proof. Corrected check `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5` shows only pre-existing PDFs exceed 100 KB; largest code file is `src/gpuwrf/validation/compare_fixture.py` at 15549 bytes.
- `git diff --stat $(git rev-parse HEAD)` produced no output before adding untracked tester files because the worker implementation is already committed at `HEAD`.

## Fixtures Used
- `fixtures/manifests/fixture-manifest-template.yaml` for validator smoke validation.
- Synthetic NumPy arrays created in pytest temporary directories for candidate/reference comparison tests.
- One transient sample slice under `tests/tmp_edge_reference.npz` during `test_compare_uses_manifest_sample_slice_when_reference_omitted`; the test removes it in `finally`.

## Gaps
- Contract gap found: `validate_manifest()` accepts `source: wrf-derived` with `wrf_version: ""`. The pinned `schema.yaml` and `schema.json` both express `minLength: 1`, and the contract requires `wrf_version` to be non-empty when `source == "wrf-derived"`, but the Python validator only checks that the value is a string. This would allow an invalid Canary WRF-derived fixture manifest through the actual CI-facing validator.
- The exact file-size proof command in the contract remains malformed; worker reported this too. The corrected command is adequate evidence, but the contract command itself should be patched in a future governance/contract update if reused.
- No physics or WRF-derived fixture behavior was evaluated because this S1 sprint explicitly excludes real fixtures and WRF extraction.

## Decision
Decision: Reject pending fix. The implementation is mostly sound for the S1 skeleton, but the actual Python validator must reject empty `wrf_version` for WRF-derived manifests before this sprint is accepted. The new failing test is `tests/test_fixture_manifest_edge_cases.py::test_wrf_derived_requires_non_empty_wrf_version`; after the worker fixes validator/schema parity, rerun `pytest -q` and the contract validation commands.
