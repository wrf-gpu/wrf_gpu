# Worker Report

## Summary
Summary: Implemented the M1 analytic stencil and analytic column fixture deliverables exactly within the worker-owned paths. The stencil fixture is a deterministic 32x16x8 periodic staggered-grid advection-diffusion update with fp64 NumPy reference output and stored face velocities. The column fixture is a deterministic 40-level moist thermodynamic source update with a moist-static-energy residual. Both fixtures validate against the S1 schema, round-trip through `gpuwrf.validation.compare_fixture`, and fail correctly when a candidate is perturbed beyond tolerance.

## Files Changed
- `fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml`
- `fixtures/manifests/analytic-column-thermo-v1.yaml`
- `fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`
- `fixtures/samples/analytic-column-thermo-v1.npz`
- `src/gpuwrf/fixtures/__init__.py`
- `src/gpuwrf/fixtures/analytic.py`
- `scripts/generate_analytic_fixtures.py`
- `tests/test_analytic_fixtures.py`
- `.agent/sprints/2026-05-19-m1-analytic-fixtures/worker-report.md`

## Commands Run
- `python scripts/validate_agentos.py`
  stdout: `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`
  stderr: empty
- `python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml`
  stdout: `fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml: ok`
  stderr: empty
- `python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-column-thermo-v1.yaml`
  stdout: `fixtures/manifests/analytic-column-thermo-v1.yaml: ok`
  stderr: empty
- `python scripts/generate_analytic_fixtures.py --seed 0 --out fixtures/samples/`
  stdout listed the two `.npz` sample paths and two manifest paths; stderr empty.
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate fixtures/samples/analytic-stencil-3d-advdiff-v1.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`
  stdout JSON included `"fixture_id": "analytic-stencil-3d-advdiff-v1"`, `"pass": true`, `"first_failure": null`; stderr empty.
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate fixtures/samples/analytic-column-thermo-v1.npz --reference fixtures/samples/analytic-column-thermo-v1.npz`
  stdout JSON included `"fixture_id": "analytic-column-thermo-v1"`, `"pass": true`, `"first_failure": null`; stderr empty.
- Perturbed-candidate check for both fixtures
  stdout: stencil returned `{"returncode": 1, "pass": false, "first_failure": "phi_initial"}`; column returned `{"returncode": 1, "pass": false, "first_failure": "temperature_initial"}`.
- `pytest -q`
  stdout: `29 passed in 3.28s`; stderr empty.
- `python scripts/check_m1_done.py`
  exit 1 as expected before later M1 closeout. Remaining errors: this sprint is not closed until tester/reviewer/manager reports are filled; no WRF-derived Canary manifest yet; no M1 closeout decision yet; M1 plan reviewer decision not accepted yet.
- `stat -c '%s %n' fixtures/samples/analytic-*.npz | sort -nr`
  stdout: `61080 fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`; `3890 fixtures/samples/analytic-column-thermo-v1.npz`.

## Proof Objects
- `fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml`
- `fixtures/manifests/analytic-column-thermo-v1.yaml`
- `fixtures/samples/analytic-stencil-3d-advdiff-v1.npz` (61,080 bytes)
- `fixtures/samples/analytic-column-thermo-v1.npz` (3,890 bytes)
- `tests/test_analytic_fixtures.py`
- Validation outputs above.

## Risks
- The analytic fixtures are simplified M2 oracles, not WRF parity evidence. This is aligned with the sprint non-goal of excluding real WRF data.
- `check_m1_done.py` still fails for expected downstream M1 reasons outside worker ownership: Canary WRF-derived fixture, tester/reviewer/manager reports, and milestone closeout.
- `.npz` is globally ignored by `.gitignore`; the two sample slices must be force-added because the contract requires these tiny samples in git.

## Handoff
- Objective: Produce two analytic fixture manifests and sample slices that exercise the S1 schema and comparison CLI.
- Files changed: listed above; no governance, goal, reviewer, tester, manager-closeout, or memory-patch files were modified.
- Commands run: listed above with outputs.
- Proof objects produced: two manifests, two sample slices, generator module, CLI wrapper, tests, and this report.
- Unresolved risks: no WRF-derived evidence in this sprint by contract; later tester/reviewer roles still need to run.
- Next decision needed: tester should validate the worker branch, then reviewer should decide whether to accept the analytic fixture proof objects.
