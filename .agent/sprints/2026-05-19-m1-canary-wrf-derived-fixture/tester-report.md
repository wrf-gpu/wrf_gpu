# Tester Report

## Tests Added Or Run

Added four tester edge-case checks in `tests/test_canary_wrf_fixture.py`:

- manifest `files[]` byte counts and sha256 values match the actual sample/full payloads;
- external `checksums.txt` covers the external full payload named by the manifest;
- malformed candidate missing `QVAPOR` exits cleanly with `candidate missing variable QVAPOR`;
- wrong-shaped `PB` candidate reports a failed comparison with `shape_ok: false`.

Re-ran every validation command from the sprint contract from the repo root on branch `tester/sonnet/m1-canary-wrf-derived-fixture`.

## Results

- Source wrfout readability: `source wrfout reachable`.
- `python scripts/validate_agentos.py`: pass, `ok: true`, no errors.
- `python scripts/extract_canary_wrf_fixture.py`: pass; regenerated `full.npz` 15967 bytes, `checksums.txt` 199 bytes, sample 14214 bytes, manifest 2641 bytes.
- `python scripts/validate_fixture_manifest.py fixtures/manifests/canary-wrf-d01-20260518T13-tslice-v1.yaml`: pass.
- `python -m gpuwrf.validation.compare_fixture ... --candidate sample --reference sample`: pass, JSON `pass: true`, `first_failure: null`, all six variables shape/tolerance clean.
- `pytest -q`: pass, `45 passed in 4.73s`.
- `python scripts/check_m1_done.py`: expected fail before role/manager closeout. Remaining errors are short stub reports/closeout and missing milestone closeout artifacts; there is no missing wrf-derived manifest error.
- Tracked file size check: largest tracked files remain existing PDFs / analytic sample; Canary sample is 14214 bytes.
- External fixture directory contains `full.npz` and `checksums.txt`.
- Source wrfout stat after validation: `enric:enric 664 9559615`; tests also hash-check that the source is not touched.

## Fixtures Used

- Source: `/mnt/data/canairy_meteo/runs/wrf_l3/20260517_18z_l3_24h_20260518T004341Z/wrfout_d01_2026-05-18_13:00:00`
- Manifest: `fixtures/manifests/canary-wrf-d01-20260518T13-tslice-v1.yaml`
- Sample: `fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz`
- External full payload: `data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/full.npz`
- External checksums: `data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/checksums.txt`

## Gaps

This is a Tier-1 fixture-plumbing validation only. It does not prove forecast skill, scientific representativeness of the subdomain, or final M4 dycore tolerances. I did not edit extractor or validation code by role constraint. I also did not resolve lifecycle closeout files; that remains reviewer/manager work after this report.

Decision: Accept. The WRF-derived fixture behaves as a valid M1 proof object under the contract commands and the added edge-case tests. Remaining `check_m1_done.py` failures are closeout workflow dependencies, not fixture implementation failures.
