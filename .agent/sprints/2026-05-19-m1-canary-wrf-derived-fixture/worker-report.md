# Worker Report

Summary: Implemented the first Canary WRF-derived M1 fixture from the existing Gen2 WRF L3 wrfout at `/mnt/data/canairy_meteo/runs/wrf_l3/20260517_18z_l3_24h_20260518T004341Z/wrfout_d01_2026-05-18_13:00:00`. The extractor slices one interior timestep block: mass fields `T`, `QVAPOR`, `P`, `PB` as 10x8x8, `U` as 10x8x9, and `V` as 10x9x8. Full payload is fp64 external storage; committed sample is fp32 and 14,214 bytes. Source stat before and after extraction stayed `enric:enric 664 9559615`.

## Files Changed

- `src/gpuwrf/fixtures/wrf_slice.py`
- `scripts/extract_canary_wrf_fixture.py`
- `fixtures/manifests/canary-wrf-d01-20260518T13-tslice-v1.yaml`
- `fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz`
- `data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/full.npz`
- `data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/checksums.txt`
- `tests/test_canary_wrf_fixture.py`
- `pyproject.toml` (`netCDF4>=1.6` added because the source wrfout is NetCDF-4 and the extractor imports `netCDF4.Dataset`)

## Commands Run

- `test -r ...wrfout... && echo "source wrfout reachable" || echo "MISSING SOURCE"` -> `source wrfout reachable`
- `stat -c '%U:%G %a %s %n' ...wrfout...` -> `enric:enric 664 9559615 /mnt/data/canairy_meteo/runs/wrf_l3/20260517_18z_l3_24h_20260518T004341Z/wrfout_d01_2026-05-18_13:00:00`
- `python scripts/validate_agentos.py` -> `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`
- `python scripts/extract_canary_wrf_fixture.py` -> wrote `full.npz` 15967 bytes, `checksums.txt` 199 bytes, sample 14214 bytes, manifest 2641 bytes.
- `python scripts/validate_fixture_manifest.py fixtures/manifests/canary-wrf-d01-20260518T13-tslice-v1.yaml` -> `...yaml: ok`
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/canary-wrf-d01-20260518T13-tslice-v1.yaml --candidate fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz --reference fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz` -> JSON `"pass": true`, `"first_failure": null`.
- Deliberate mutated candidate command using `/tmp/canary-wrf-d01-20260518T13-tslice-v1-mutated.npz` -> exit 1 expected, JSON `"pass": false`, `"first_failure": "T"`, `max_abs_diff` about `0.01000023`.
- `pytest -q` -> `41 passed in 4.39s`. An earlier parallel run with `check_m1_done.py` produced one temporary-file race in an existing edge-case test; the serial rerun passed.
- `python scripts/check_m1_done.py` -> exit 1, expected before tester/reviewer/manager closeout. Remaining errors were report/closeout gating only; no missing wrf-derived manifest error remained.
- `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5` -> largest tracked files unchanged; top entries are the pre-existing 1.5 MB PDF, 97,515 byte PDF, and 61,080 byte analytic sample.
- `ls -la data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/` -> `checksums.txt` 199 bytes and `full.npz` 15967 bytes.
- `stat -c '%s %n' fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz` -> `14214 fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz`

## Proof Objects

- Manifest: `fixtures/manifests/canary-wrf-d01-20260518T13-tslice-v1.yaml`
- Committed sample: `fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz`
- External payload: `data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/full.npz`
- External checksums: `data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/checksums.txt`
- Extractor and CLI: `src/gpuwrf/fixtures/wrf_slice.py`, `scripts/extract_canary_wrf_fixture.py`
- Tests: `tests/test_canary_wrf_fixture.py`

## Risks

- The subdomain is an interior convenience slice, not a forecast-skill reference. It is suitable for M1/M2 fixture plumbing and basic WRF-derived parity only.
- Tolerances are initial fp32-operational-output tolerances, not final dycore physics tolerances.
- `check_m1_done.py` remains red until other roles replace their reports and manager closeout artifacts; this worker did not touch those files by contract.

## Handoff

Objective: produce one WRF-derived Canary fixture without launching a new WRF job.
Files changed: listed above and limited to worker-owned paths.
Commands run: listed above with outputs.
Proof objects produced: manifest, sample, external payload, checksums, extractor, CLI, tests.
Unresolved risks: closeout depends on tester/reviewer/manager reports and acceptance; no memory patch expected.
Next decision needed: tester should validate independently, then reviewer decides accept/rework.
