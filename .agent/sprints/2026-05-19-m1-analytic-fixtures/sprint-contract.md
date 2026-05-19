# Sprint Contract

Sprint ID: `2026-05-19-m1-analytic-fixtures`
Milestone: M1 — WRF Oracle & Fixtures
Sequence: S2 (consolidated: analytic stencil + analytic column, per runbook §B "Sprint structure is not frozen")
Reviewer: opus-reviewer
Worker: gpt-kernel-worker (Codex `gpt-5.5` `high`)
Tester: sonnet-test-engineer
Approval status: opened 2026-05-19 by manager after S1 closeout (binding Accept on reviewer attempt 2).

## Objective

Produce two analytic fixtures that exercise the S1 schema and CLI end-to-end, so M2 backend candidates have concrete correctness oracles to compare against. Both fixtures stay tiny (≤100 KB sample slice committed in-tree) and live entirely on the manager's `data/` symlink for any larger payload.

1. **Analytic stencil micro-fixture** (`fixture_id: analytic-stencil-3d-advdiff-v1`): a 32×16×8 staggered-grid 3D advection-diffusion update over a single timestep. Reference solution computed in NumPy at `fp64`. Tests fusion ergonomics in M2 candidates that consume this fixture.
2. **Analytic column micro-fixture** (`fixture_id: analytic-column-thermo-v1`): a single vertical column with N=40 levels of (T, qv, p) state, evolved one timestep under a simple analytic source (e.g. a closed-form moist-static-energy preserving operator). Tests register-pressure handling in M2 candidates.

Both fixtures must round-trip through `python -m gpuwrf.validation.compare_fixture` and report `pass: true` on the identity case, `pass: false` on a deliberately mutated case.

## Non-Goals

- No real WRF data (that is S3 / S4: Canary WRF-derived fixture).
- No backend selection, no GPU code, no profiler artifacts (M2/M3 territory).
- No new schema fields. The S1 schema is frozen; if a needed field is missing, file a manager note for an ADR in a separate sprint — do not edit the schema in this sprint.
- No tolerance-tuning beyond initial reasonable defaults (S1's `tolerance_abs`/`tolerance_rel` machinery is what gets used; pick conservative starting values).

## File Ownership

Worker may create or edit only these paths in this sprint:

- `fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml` (new)
- `fixtures/manifests/analytic-column-thermo-v1.yaml` (new)
- `fixtures/samples/analytic-stencil-3d-advdiff-v1.npz` (new, ≤100 KB — full state, small enough to commit)
- `fixtures/samples/analytic-column-thermo-v1.npz` (new, ≤100 KB)
- `src/gpuwrf/fixtures/__init__.py` (new if missing)
- `src/gpuwrf/fixtures/analytic.py` (new — pure-NumPy generators for both fixtures; deterministic from a seed; reproducible)
- `scripts/generate_analytic_fixtures.py` (new — CLI wrapper around the generators; rewrites both manifests + sample slices from scratch)
- `tests/test_analytic_fixtures.py` (new — generator determinism, manifest validates, CLI round-trips identity-pass + mutated-fail)
- `pyproject.toml` (edit only if a new dev-dep is absolutely required; explain in worker report; numpy and PyYAML should already cover this)

Any change outside this list requires manager approval.

## Inputs

- S1 outputs (all on main now): `fixtures/manifests/schema.yaml`, `fixtures/manifests/schema.json`, `scripts/validate_fixture_manifest.py`, `src/gpuwrf/validation/compare_fixture.py`.
- `PROJECT_PLAN.md §5` for problem shapes — Problem 1 stencil = 3D advection-diffusion, Problem 2 column = many local prognostics with branching. The analytic fixtures here are *the simplified analytic oracles* corresponding to those M2 problem shapes.
- `.agent/skills/building-wrf-oracles/SKILL.md`.

## Acceptance Criteria

All must hold for closeout.

### Schema/validator parity (carry-forward S1 lesson)

1. Both fixture manifests validate against `fixtures/manifests/schema.yaml` via `python scripts/validate_fixture_manifest.py <path>` (exit 0).
2. Each manifest's `source` is `analytic`, `wrf_version` is omitted or `null`, every variable has per-variable `tolerance_abs`, `tolerance_rel`, `tolerance_rationale`, `staggering`, `shape`, `dtype`, `units`. No top-level tolerance.

### Generator determinism

3. `python scripts/generate_analytic_fixtures.py --seed 0 --out fixtures/samples/` produces bit-identical `.npz` files on two consecutive runs (verify in test).
4. Generator code is pure-NumPy + Python stdlib only. No SciPy / xarray / JAX / torch / etc.

### Round-trip via S1 CLI

5. `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate fixtures/samples/analytic-stencil-3d-advdiff-v1.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz` emits `{"pass": true, ...}`.
6. The same command with a deliberately mutated `--candidate` (worker constructs a temporary `.npz` perturbing one variable by 10× its tolerance_abs) emits `{"pass": false, ...}` with the violating variable correctly identified.
7. Same two tests pass for `analytic-column-thermo-v1`.

### Sample-slice size discipline

8. Each `fixtures/samples/*.npz` is ≤100 KB. Verify via `stat -c '%s'`.

### Test suite

9. `tests/test_analytic_fixtures.py` adds at minimum:
   - Generator-determinism test (two runs, bit-identical).
   - Manifest-validates test (schema validator returns ok for both).
   - CLI round-trip identity test (positive case, both fixtures).
   - CLI round-trip perturbation test (negative case, both fixtures).
10. `pytest -q` passes on main+sprint changes (target: 25 → ~33 tests).

### CI / repo hygiene

11. `python scripts/validate_agentos.py` passes.
12. `python scripts/check_m1_done.py` reports `len(_manifests_with_source("analytic")) >= 2` (the function the M1 oracle uses).
13. No committed file >100 KB beyond pre-existing PDFs.

## Validation Commands

```bash
python scripts/validate_agentos.py
python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml
python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-column-thermo-v1.yaml
python scripts/generate_analytic_fixtures.py --seed 0 --out fixtures/samples/      # idempotent
python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate fixtures/samples/analytic-stencil-3d-advdiff-v1.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz
python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate fixtures/samples/analytic-column-thermo-v1.npz --reference fixtures/samples/analytic-column-thermo-v1.npz
pytest -q
python scripts/check_m1_done.py    # `errors` list should shrink by the analytic-manifest count
stat -c '%s %n' fixtures/samples/analytic-*.npz | sort -nr
```

## Performance Metrics

Not applicable — this sprint produces analytic data files, no GPU work, no kernels.

## Proof Object

- Diff (limited to File Ownership paths).
- Two new manifest files in `fixtures/manifests/`.
- Two new sample slice files in `fixtures/samples/`, each ≤100 KB.
- One generator module + one CLI script.
- One test module.
- worker-report.md / tester-report.md / reviewer-report.md / manager-closeout.md / memory-patch.md per lifecycle.

## Risks

- **Sample slice size overrun.** A 32×16×8×3 fp64 array is 12 KB; a column 40×3 fp64 is 1 KB. Comfortably under 100 KB. If multiple variables push it over, downcast to fp32 in the sample slice (manifest already supports mixed precision per-variable).
- **Determinism trap.** NumPy operations with implicit threading (e.g. BLAS in `np.linalg`) can be non-deterministic. Use explicit per-element operations and seeded `np.random.default_rng(0)`. The generator-determinism test enforces this.
- **Analytic oracle smell.** Conservative tolerances are fine for an analytic identity case. M2 candidates will use these fixtures for relative comparisons, not absolute correctness against physics. Worker should document the rationale in `tolerance_rationale` for each variable.
- **Bundling risk.** Two fixtures in one sprint means one bad generator could block both. Mitigation: worker implements them as two independent functions in `analytic.py` with separate tests.

## Handoff Requirements

- Worker pushes to branch `worker/gpt/m1-analytic-fixtures`.
- Worker opens no merge until reviewer + tester reports are on disk.
- Manager closeout integrates branches into main via `git merge --no-ff` (S1 pattern).
- Memory patch is not expected (no new constitutional knowledge).
