# Manager Closeout

Sprint: `2026-05-19-m1-canary-wrf-derived-fixture` (M1 S3 — final M1 sprint)
Closed: 2026-05-19
Cycles: 1 worker, 1 tester (Accept), 1 reviewer (Accept). Zero fix cycles.

## Outcome

Clean single-pass close. First real WRF-derived fixture established, exercising the validator-enforced `wrf_version` non-empty rule put in place by S1's fix cycle. M1 is now satisfied on all proof objects.

## Proof Objects

- `fixtures/manifests/canary-wrf-d01-20260518T13-tslice-v1.yaml` (2,641 B) — `source: wrf-derived`, `wrf_version: "4.7.1"`, source-commit field records the wrfout sha256, per-variable tolerances + rationale.
- `fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz` (14,214 B) — in-tree sample slice, fp32.
- `data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/full.npz` (15,967 B) — external full payload, fp64.
- `data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/checksums.txt` (199 B) — sha256 manifest.
- `src/gpuwrf/fixtures/wrf_slice.py` — netCDF4-based extractor.
- `scripts/extract_canary_wrf_fixture.py` — idempotent CLI wrapper.
- `tests/test_canary_wrf_fixture.py` (145 lines, + 4 more tester edge cases) — schema validates, sample loads, sample ≤100 KB, source-wrfout untouched, round-trip identity-pass + perturbation-fail, wrf_version validator parity test.
- Source wrfout (`/mnt/data/canairy_meteo/runs/wrf_l3/20260517_18z_l3_24h_20260518T004341Z/wrfout_d01_2026-05-18_13:00:00`) verified untouched by stat before/after extraction.
- Pytest: 45/45 passing.

## Merge Decision

Merge Decision: **Accept and integrate into main**. Worker branch `worker/gpt/m1-canary-wrf-derived-fixture` (HEAD includes tester+reviewer commits via `reviewer/opus/m1-canary-wrf-derived-fixture`) merges to main via `git merge --no-ff` (S1/S2 pattern).

## Scope Changes

None. Worker added one dev-dep (`netCDF4>=1.6`) which was anticipated in the contract's File Ownership for `pyproject.toml` and justified in the worker report (source is NetCDF-4, requires HDF5-backed reader; `scipy.io.netcdf_file` only handles NetCDF-3). No GPU code, no fresh WRF job, no schema modification, no validator modification.

## Lessons

1. **Slicing from existing Gen2 runs is the right S3 strategy.** Per §11.6 decision: zero CPU time spent, oracle established in one sprint pass. Future WRF-derived fixtures should follow this pattern unless a specific scenario is missing from Gen2's run history.
2. **Variable staggering is non-trivial in WRF.** Worker correctly recorded staggering per variable (`T`, `QVAPOR`, `P`, `PB` on mass grid 8×8; `U` on u-grid 8×9; `V` on v-grid 9×8). The schema's per-variable `staggering` field paid off here — without it, downstream M2 candidates wouldn't know how to align grids.
3. **External-vs-sample dichotomy is sound.** Manifest references both: in-tree ≤100 KB sample for CI fast-loops, external full payload at `data/fixtures/...` for higher-fidelity tests. Symlink `./data → /mnt/data/wrf_gpu2/` keeps the policy enforceable.

## Next Sprint

**M1 closes with this sprint.** Manager next-action (this turn): write `.agent/decisions/MILESTONE-M1-CLOSEOUT.md`, flip M1 plan Reviewer Decision to Accepted, stop the loop, report to user. After user confirmation, M2 (backend bakeoff) opens.
