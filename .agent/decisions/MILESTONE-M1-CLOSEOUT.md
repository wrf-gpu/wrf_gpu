# Milestone M1 Closeout — WRF Oracle & Fixtures

Date: 2026-05-19
Status: **CLOSED — all proof objects present, all sprints closed, all reviewer decisions Accept.**

## Summary

M1 establishes the project's correctness-oracle layer: a frozen fixture manifest schema, a comparison-harness CLI with a frozen public surface, an external-storage policy with git-exclusion rules, and three working fixtures (one analytic stencil, one analytic column, one Canary WRF-derived). M2 backend candidates can now consume these fixtures directly for correctness comparison.

## Closed Sprints

| ID | Outcome | Cycles |
|---|---|---|
| `2026-05-18-m1-fixture-storage-policy` (S1) | Accept (attempt 2) | 2 worker, 1 tester, 2 reviewer — 1 fix cycle on validator/schema parity |
| `2026-05-19-m1-analytic-fixtures` (S2) | Accept (attempt 1) | 1 worker, 1 tester, 1 reviewer — zero fix cycles |
| `2026-05-19-m1-canary-wrf-derived-fixture` (S3) | Accept (attempt 1) | 1 worker, 1 tester, 1 reviewer — zero fix cycles |

Bundled S2 (analytic stencil + analytic column in one sprint) per runbook "Sprint structure is not frozen" — saved one full role cycle vs. the 4-sprint default in §8 of PROJECT_PLAN.md.

## Proof Objects (all on `main`)

### Schema and machinery
- `fixtures/manifests/schema.yaml` + `fixtures/manifests/schema.json` (JSON-Schema mirror).
- `fixtures/manifests/fixture-manifest-template.yaml` (validates against the schema).
- `docs/fixture-storage-policy.md` (naming, checksums, commit/exclusion, external `data/` convention, retention).
- `.gitignore` extended to enforce the policy.
- `src/gpuwrf/validation/compare_fixture.py` (frozen 5-flag CLI surface: `--manifest --candidate --reference --tier --out`; frozen JSON output schema).
- `scripts/validate_fixture_manifest.py` (CI-callable validator).

### Fixtures (3 of 3 categories satisfied)
- **Analytic stencil**: `fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml` + `fixtures/samples/analytic-stencil-3d-advdiff-v1.npz` (61,080 B). 32×16×8 3D advection-diffusion, fp64, deterministic generator.
- **Analytic column**: `fixtures/manifests/analytic-column-thermo-v1.yaml` + `fixtures/samples/analytic-column-thermo-v1.npz` (3,890 B). 40-level moist thermo column, deterministic generator.
- **Canary WRF-derived**: `fixtures/manifests/canary-wrf-d01-20260518T13-tslice-v1.yaml` + `fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz` (14,214 B in-tree) + `data/fixtures/canary-wrf-d01-20260518T13-tslice-v1/full.npz` (15,967 B external). Sliced from Gen2 wrfout `wrf_l3/20260517_18z_l3_24h_20260518T004341Z/wrfout_d01_2026-05-18_13:00:00` (WRF v4.7.1).

### Tooling
- `src/gpuwrf/fixtures/analytic.py` (306 LOC, pure-NumPy generators).
- `src/gpuwrf/fixtures/wrf_slice.py` (NetCDF4-based extractor).
- `scripts/generate_analytic_fixtures.py`, `scripts/extract_canary_wrf_fixture.py` (idempotent CLI wrappers).
- `scripts/check_m1_done.py` (single-command M1 oracle).
- `scripts/dispatch_role.sh` (manager's universal sprint-role dispatcher).

### Test coverage
- 45 tests passing on `main` (up from 9 at M0 close).
- Coverage: schema validation positive + negative, CLI round-trip identity + perturbation for each fixture, generator determinism, source-wrfout immutability, wrf_version validator parity edge case, sample-size discipline.

### Governance artifacts
- `PROJECT_PLAN.md` (synthesis layer + §11 manager decisions).
- `.agent/milestones/ROADMAP.md` (per-milestone proof object lists).
- `.agent/goals/M1-DONE.md` + `.agent/goals/M1-MANAGER-RUNBOOK.md`.
- 6 sprint reports (3 worker, 3 tester, 3 reviewer + attempt-1 archives where fix cycles occurred).
- Codex cross-model bootstrap review: `.agent/decisions/REVIEW-codex-bootstrap-plan.md` + manager response.

## Residual Risks (not blockers; recorded for M2 / M3+)

- **dtype fidelity not enforced.** `compare_fixture.py:286` upcasts to fp64 for the numeric diff. Manifest `dtype` is informational. M2 contracts should enforce dtype parity when backend candidates start emitting fp32 / bf16. (S2 reviewer note.)
- **WRF-derived fixture is a single timestep convenience slice.** Adequate for M1/M2 oracle plumbing, not for forecast-skill validation. Future M3/M4 sprints will add multi-timestep slices and BC metadata. (S3 manager note.)
- **Tolerances on the Canary fixture are starting values for operational fp32 output**, not final dycore-precision tolerances. M4 will revise per-variable. (S3 worker note.)
- **The `dispatch_role.sh` "send-keys completion" mechanism requires single-threaded sprint execution.** Parallel sprints would interleave text into the manager prompt. M1 was single-threaded by design; M2 bakeoff may want parallelism — runbook needs updating before M2.

## Top Three Lessons

1. **Carrying schema-validator parity forward as an explicit acceptance criterion paid off.** S1 lost one round to it; S2 and S3 (which inherited the AC) had zero fix cycles. Future contracts that produce schema-conformant artifacts should embed this AC verbatim.
2. **Bundling small same-shape deliverables saves cycles.** S2 (analytic stencil + analytic column) ran as one sprint instead of two; saved 3 agent calls. The runbook's "Sprint structure is not frozen" guidance applies whenever deliverables share machinery and ownership.
3. **Reusing existing Gen2 runs for WRF-derived fixtures avoids hours of CPU WRF time per fixture.** The §11.6 decision worked: zero fresh WRF jobs, oracle established in one S3 pass.

## Recommended Next Milestone

**M2 — Backend Bakeoff** opens immediately on user confirmation. Per `PROJECT_PLAN.md §5`:
- Six candidate families: A JAX/XLA, B Triton, C GT4Py/DaCe, D Kokkos/C++, E CuPy or Numba, F Explicit CUDA C++ (CUDA Tile).
- Two shared bakeoff problems consuming M1 fixtures: 3D stencil (analytic-stencil-3d-advdiff-v1) + column (analytic-column-thermo-v1).
- Per-candidate proof object: profiler JSON OR candidate-failure artifact, correctness JSON against M1 fixtures, maintainability narrative ≤300 words, agent-success log.
- ADR-001 (backend lock) — irreversible architecture decision; manager exercises with Codex cross-model critical review per the manager-autonomy directive.

Manager intends to open M2's first sprint as a research-scout pass (Blackwell toolchain readiness for each candidate) before committing to implementation sprints.
