# Tester Report

## Tests Added Or Run

Added four edge-case tests in `tests/test_analytic_fixtures.py`:
- actual analytic candidate missing a required variable is rejected with return code 2 and the missing variable named;
- actual stencil fixture with a manifest-declared shape mismatch returns a clean comparison failure with `shape_ok: false`;
- missing candidate file is rejected with a load error;
- schema validation rejects a forbidden top-level `tolerance_abs` field.

Re-ran every validation command listed in the sprint contract from a clean shell after the test additions.

## Results

- `python scripts/validate_agentos.py` passed: `ok: true`, `errors: []`.
- Both analytic manifests validated with `python scripts/validate_fixture_manifest.py`.
- `python scripts/generate_analytic_fixtures.py --seed 0 --out fixtures/samples/` completed and rewrote the two sample paths and manifests deterministically.
- Both identity comparisons through `python -m gpuwrf.validation.compare_fixture` returned `"pass": true` and `"first_failure": null`.
- `pytest -q` passed: `33 passed in 3.43s`.
- `python scripts/check_m1_done.py` exited 1, as expected before M1 closeout. After this tester report was written, remaining errors are outside tester scope: reviewer/manager/memory reports still stubs, missing Canary WRF-derived fixture, missing milestone closeout, and M1 plan reviewer decision not accepted.
- Sample sizes remain within contract: stencil `61080` bytes, column `3890` bytes.

## Fixtures Used

- `fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml`
- `fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`
- `fixtures/manifests/analytic-column-thermo-v1.yaml`
- `fixtures/samples/analytic-column-thermo-v1.npz`

## Gaps

No blocking gap found in the analytic fixture deliverable. These fixtures are still Tier-1 analytic oracles only; they do not provide WRF-derived Canary evidence, GPU performance evidence, or timestep-loop transfer evidence, all of which are non-goals for this sprint. One residual validation limitation: the comparison CLI primarily enforces numeric tolerance and shape; dtype is declared in the manifest but not separately audited by the added tests.

Decision: Accept. The worker implementation satisfies the analytic-fixture sprint contract within tester-observed evidence, with expected downstream M1 gaps clearly outside this sprint.
