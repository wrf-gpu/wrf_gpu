# Sprint Contract

Sprint ID: `2026-05-19-m1-canary-wrf-derived-fixture`
Milestone: M1 — WRF Oracle & Fixtures
Sequence: S3 (final M1 sprint; produces the WRF-derived oracle proof object that M2 candidates will validate against)
Reviewer: opus-reviewer
Worker: gpt-kernel-worker (Codex `gpt-5.5` `high`)
Tester: sonnet-test-engineer
Approval status: opened 2026-05-19 by manager after S2 closeout (binding Accept).

## Objective

Produce one WRF-derived Canary fixture sliced from an existing Gen2 WRF L3 run — **without** launching a new CPU WRF job (per `PROJECT_PLAN.md §11.6` decision). Establishes the first real `source: wrf-derived` manifest with a non-empty `wrf_version`, exercising the validator-enforced rule that S1's fix cycle put in place.

**Source-of-record**: `/mnt/data/canairy_meteo/runs/wrf_l3/20260517_18z_l3_24h_20260518T004341Z/wrfout_d01_2026-05-18_13:00:00`
- WRF v4.7.1, Canary Islands L3 24h run, d01 outer domain, valid 2026-05-18 13:00 UTC.
- This is a known-good Gen2 operational output. Treat it as immutable input.

**Fixture deliverable**: `fixture_id: canary-wrf-d01-20260518T13-tslice-v1`
- A small spatial+vertical subdomain (target: 8×8 horizontal × 10 vertical levels) of 4–6 prognostic-like fields (e.g. `T`, `U`, `V`, `QVAPOR`, `P`, `PB`). Single timestep.
- Full slice payload stored externally at `data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/` (gitignored — symlinks to `/mnt/data/wrf_gpu2/fixtures/`).
- ≤100 KB sample slice committed in-tree at `fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz`.
- Manifest at `fixtures/manifests/canary-wrf-d01-20260518T13-tslice-v1.yaml`.

## Non-Goals

- **Do not** launch a fresh CPU WRF job. The Gen2 run is the source-of-record per the §11.6 decision.
- Do not slice multiple timesteps. One timestep is sufficient as the first oracle.
- Do not commit any binary file >100 KB into git. Full payload is external.
- Do not modify the schema or the comparison CLI. They are frozen from S1.
- No GPU code, no backend choice, no profiler artifacts.
- Do not interpret the fixture as a scientific reference for forecast skill — it is one timestep of one run.

## File Ownership

Worker may create or edit only these paths:

- `fixtures/manifests/canary-wrf-d01-20260518T13-tslice-v1.yaml` (new)
- `fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz` (new, ≤100 KB; can use fp32 downcasting on sample if needed for size)
- `data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/full.npz` (new, external storage; the manifest's `external_uri` points here)
- `data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/checksums.txt` (new, external sha256 manifest)
- `src/gpuwrf/fixtures/wrf_slice.py` (new — extractor: reads wrfout via netCDF4 or scipy.io.netcdf, slices subdomain, writes both full + sample .npz)
- `scripts/extract_canary_wrf_fixture.py` (new — CLI wrapper that runs the extraction)
- `tests/test_canary_wrf_fixture.py` (new — schema validates, sample loads, round-trip identity-pass + perturbation-fail, sample is ≤100 KB, full file exists at the external URI)
- `pyproject.toml` (edit only if a NetCDF dep — `netCDF4>=1.6` or `scipy>=1.10` — is needed; justify in worker report)

Any change outside this list requires manager approval. Worker must not modify `src/gpuwrf/validation/compare_fixture.py`, the schema, the storage policy, or any governance file.

## Inputs

- Source wrfout: `/mnt/data/canairy_meteo/runs/wrf_l3/20260517_18z_l3_24h_20260518T004341Z/wrfout_d01_2026-05-18_13:00:00` (verify it exists before starting; do not modify it).
- S1 outputs: `fixtures/manifests/schema.yaml`, `scripts/validate_fixture_manifest.py`, `src/gpuwrf/validation/compare_fixture.py`.
- S2 pattern: see `src/gpuwrf/fixtures/analytic.py` and `tests/test_analytic_fixtures.py` as a template — same shape of work, different source.
- `PROJECT_PLAN.md §11.6` for the IC/BC + AIFS context; S3 doesn't bind to AIFS yet (that's M3+) but the manifest's metadata should be accurate.
- `.agent/skills/building-wrf-oracles/SKILL.md`.

## Acceptance Criteria

All must hold for closeout.

### Schema/validator parity (S1 carry-forward)

1. Manifest validates against `fixtures/manifests/schema.yaml` via `python scripts/validate_fixture_manifest.py <path>` (exit 0).
2. Manifest has `source: wrf-derived`, `wrf_version: "4.7.1"` (**non-empty** — this is the explicit validator check from S1's fix cycle).
3. `source_commit` records the wrfout file path and its sha256 checksum.
4. `generation_command` records the exact `scripts/extract_canary_wrf_fixture.py` invocation that produced the artifacts.
5. Every variable has per-variable `tolerance_abs`, `tolerance_rel`, `tolerance_rationale`, `staggering`, `shape`, `dtype`, `units`. Tolerances chosen to reflect that this is single-precision-equivalent operational output, not bit-equal reference.

### Extraction integrity

6. The wrfout source file is **read-only** for the worker — verify with `stat -c '%U:%G %a'` before and after extraction; permissions must not change.
7. `data/fixtures/.../full.npz` exists, contains all selected variables at the agreed subdomain shape, fp64-cast for stable comparison.
8. `data/fixtures/.../checksums.txt` lists sha256 for every file the manifest references externally.
9. `fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz` exists and is ≤100 KB. May use fp32 for size; manifest's per-variable `dtype` must reflect what's actually in the sample.

### Round-trip via S1 CLI

10. `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/canary-wrf-d01-20260518T13-tslice-v1.yaml --candidate fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz --reference fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz` emits `{"pass": true, ...}`.
11. Same command with a deliberately mutated `--candidate` (worker constructs perturbing one variable by 10× its tolerance_abs) emits `{"pass": false, ...}` with the violating variable correctly identified.

### Test suite

12. `tests/test_canary_wrf_fixture.py` adds at minimum:
    - Source-wrfout-readability test (the path exists and is readable).
    - Manifest-validates test.
    - Sample-file-loads test (numpy can load it; shapes match manifest).
    - Sample-size test (≤100 KB).
    - CLI round-trip identity-pass test.
    - CLI round-trip perturbation-fail test.
    - **wrf_version validator parity test** (assert that the manifest *would* be rejected if `wrf_version` were empty — exercises S1's fix).
13. `pytest -q` passes on the full suite (target: 33 → ~40 tests).
14. The wrfout file remains untouched (`stat` hash check in test).

### CI / repo hygiene

15. `python scripts/validate_agentos.py` passes.
16. `python scripts/check_m1_done.py` reports `len(_manifests_with_source("wrf-derived")) >= 1`.
17. No committed file >100 KB beyond pre-existing PDFs (use the corrected command from the S1 lesson).

## Validation Commands

```bash
test -r "/mnt/data/canairy_meteo/runs/wrf_l3/20260517_18z_l3_24h_20260518T004341Z/wrfout_d01_2026-05-18_13:00:00" && echo "source wrfout reachable" || echo "MISSING SOURCE"
python scripts/validate_agentos.py
python scripts/extract_canary_wrf_fixture.py            # idempotent regeneration
python scripts/validate_fixture_manifest.py fixtures/manifests/canary-wrf-d01-20260518T13-tslice-v1.yaml
python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/canary-wrf-d01-20260518T13-tslice-v1.yaml --candidate fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz --reference fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz
pytest -q
python scripts/check_m1_done.py        # 'wrf-derived' error should clear
git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5
ls -la data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/
stat -c '%s %n' fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz
```

## Performance Metrics

Not applicable.

## Proof Object

- Diff (limited to File Ownership paths).
- Manifest + sample slice (in-tree).
- External full payload at `data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/`.
- Extractor module + CLI.
- Tests.
- worker-report.md / tester-report.md / reviewer-report.md / manager-closeout.md / memory-patch.md per lifecycle.

## Risks

- **Source wrfout missing or unreadable.** The path must exist. The first validation command verifies this. If missing, worker writes a `BLOCKER` file in the sprint folder and exits.
- **Sample size overrun.** A 4-variable × 8×8×10 fp32 slice ≈ 10 KB. Comfortable. Worst case: drop a variable or shrink the subdomain.
- **NetCDF dep choice.** `netCDF4` is the canonical reader but requires HDF5 system libs. `scipy.io.netcdf_file` works for NetCDF-3 only and may not handle compressed NetCDF-4. Worker picks one, justifies in report, adds to pyproject.toml.
- **Subdomain choice arbitrariness.** Worker picks a contiguous interior block (avoid lateral boundaries). Document the chosen indices in `tolerance_rationale`. Future S3-revisit sprints can refine.
- **Tolerance choices.** This is operational WRF output, not bit-true reference. Worker picks `tolerance_abs ≈ 1e-3 K` for temperature, similar order for others. These are starting values for M2 candidate comparison, not final M4 dycore tolerances.

## Handoff Requirements

- Worker pushes to branch `worker/gpt/m1-canary-wrf-derived-fixture`.
- Worker opens no merge until reviewer + tester reports are on disk.
- After reviewer Accept, manager writes closeout + merges into main (S1/S2 pattern), then writes M1 milestone closeout + flips the M1 plan's Reviewer Decision to Accepted, closing M1.
- Memory patch is not expected (no constitutional knowledge).

## Note on M1 closeout (manager-only, post-S3)

After this sprint closes:
1. Manager writes `.agent/decisions/MILESTONE-M1-CLOSEOUT.md` per `.agent/goals/M1-DONE.md §E`.
2. Manager edits `.agent/milestones/M1-wrf-oracle-fixtures-plan.md` to set `Reviewer Decision: Accepted`.
3. `python scripts/check_m1_done.py` returns `{"ok": true}`.
4. Manager stops the /loop (does not call ScheduleWakeup) and writes a user status report.
