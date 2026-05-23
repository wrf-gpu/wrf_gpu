# Sprint Contract — M6.x S1: Diagnostic Foundation + Source-Mining Lock

## Objective

Critic-ratified HYBRID plan, Sprint S1. Build the diagnostic sidecar foundation AND the source-mining lock BEFORE any operator-code changes. The next sprint (S2: 1h d02 baseline) consumes these sidecars; the sprint after (S3: mu-limiter A/B) consumes the source-mining lock to anchor every operator change.

Two deliverables in one sprint:
1. **12 diagnostic sidecars** (read-only on production code; new scripts under `scripts/diagnostic_*.py`)
2. **`.agent/decisions/source_mining_operator_table.md`** — canonical WRF/MPAS/Pace/ICON4Py/Dinosaur line citations for every operator term that will be touched in S3

Critical: NO production-code edits. Sidecars only read from State / replay outputs / fixtures. Source-mining lock is documentation only.

## Non-Goals

- No edits to `src/gpuwrf/dynamics/`, `src/gpuwrf/contracts/`, `src/gpuwrf/physics/`. Read-only.
- No analytic R7 oracle, MPAS slice oracle, operator-sanity test changes.
- No d02 replay run (S2 does that).
- No mu-limiter change (S3 does that).
- No Tier-3 or Tier-4 work (S4/S5).
- No remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_s1_diag` on branch `worker/gpt/m6x-s1-diagnostic-foundation`.

Write-only:
- `scripts/diagnostic_bound_violation_tracer.py`
- `scripts/diagnostic_sanitizer_audit.py`
- `scripts/diagnostic_limiter_activation_tracker.py`
- `scripts/diagnostic_field_rmse_timeline.py`
- `scripts/diagnostic_spatial_divergence_map.py`
- `scripts/diagnostic_conservation_tracker.py`
- `scripts/diagnostic_boundary_ring_error_profiler.py`
- `scripts/diagnostic_vertical_column_phase_space.py`
- `scripts/diagnostic_operator_term_budget_tracer.py`
- `scripts/diagnostic_transfer_launch_timeline.py`
- `scripts/diagnostic_timestep_convergence_dashboard.py`
- `scripts/diagnostic_stabilizer_provenance_scanner.py`
- `.agent/decisions/source_mining_operator_table.md` (new)
- `tests/test_m6x_s1_diagnostic_sidecars.py` (new — smoke tests that each sidecar runs)
- `.agent/sprints/2026-05-23-m6x-s1-diagnostic-foundation/` — proofs + worker-report

Read-only everywhere else.

## Inputs

Required reading:
- **`.agent/sprints/2026-05-23-m6x-close-strategy-plan-critic/reviewer-report.md`** §4 (sidecar audit: all 12 with build cost + question answered + expected output) and §5 (source-mining table format)
- `.agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/diagnostic-report.md` — Opus's existing diagnostic script as reference pattern
- `scripts/diagnostic_warm_bubble_vs_slice.py` — Opus-built, reuse the pattern for sidecar style
- `scripts/diagnostic_gen2_rmse_baseline.py` — Gen2 baseline tool; data-loading patterns are reusable
- `scripts/m6_warm_bubble_test.py` — operator-sanity gate harness (bound-violation logic to extend)
- `src/gpuwrf/integration/d02_replay.py` — d02 replay scaffolding (source for sanitizer/limiter/replay diagnostics)
- `src/gpuwrf/dynamics/acoustic_wrf.py` — current operator (READ-ONLY here)
- `.agent/references/cpu-wrf-baseline.md` — Gen2 location
- WRF source `module_small_step_em.F:619-1597`, MPAS source `mpas_atm_time_integration.F:1589-2495`, Pace `dyn_core.py`+`del2cubed.py`+`ray_fast.py`, ICON4Py `solve_nonhydro.py`+`vertically_implicit_dycore_solver.py`, Dinosaur `time_integration.py` — for the source-mining lock table

## Acceptance Criteria

### Part A: 12 diagnostic sidecars

Each sidecar in `scripts/diagnostic_*.py` must:
1. Accept CLI args: `--input` (path to state/replay output/fixture) and `--output` (path to JSON proof)
2. Write a JSON proof with documented schema (top-level keys describe what's measured + units)
3. Run successfully against the current main's state (warm-bubble harness output, d02 replay scaffold, or synthetic fixture)
4. Be importable as a Python module (so the smoke test can call its `main(args)` function)
5. Have a module docstring with: purpose, answers, expected output schema, source citations for algorithmic choices

The 12 sidecars (from critic §4):

1. `diagnostic_bound_violation_tracer.py` — first `(field, step, time, value, bound, i, j, k)` where each bound violates
2. `diagnostic_sanitizer_audit.py` — per-step candidate nonfinite / clip / changed counts; first bad candidate step
3. `diagnostic_limiter_activation_tracker.py` — `_mu_continuity_increment` saturation fraction, max raw dmu, max bounded dmu, max-column location
4. `diagnostic_field_rmse_timeline.py` — RMSE, bias, max abs error for T2/U10/V10/qv2/w/theta by lead time vs Gen2 wrfout
5. `diagnostic_spatial_divergence_map.py` — `.npz` maps + JSON summary by boundary band, terrain quartile, land/sea, elevation
6. `diagnostic_conservation_tracker.py` — total mass / water / KE / dry static energy with source/sink + boundary terms
7. `diagnostic_boundary_ring_error_profiler.py` — RMSE by 0-5, 5-10, 10-20, interior grid-cell bands
8. `diagnostic_vertical_column_phase_space.py` — selected columns: time series + vertical profiles + phase portraits
9. `diagnostic_operator_term_budget_tracer.py` — per-term max/mean/L2 for buoyancy / pressure restoring / density coupling / theta transport / Rayleigh / smdiv / boundary forcing
10. `diagnostic_transfer_launch_timeline.py` — JAXPR callback-free flag, trace H2D/D2H bytes, launch count, peak memory
11. `diagnostic_timestep_convergence_dashboard.py` — norms by variable/lead/dt pair, convergence verdict (placeholder structure; populated in S4)
12. `diagnostic_stabilizer_provenance_scanner.py` — scans for named stabilizers in `src/gpuwrf/`, classifies as `source-backed`/`experiment-backed`/`reject`; extends warm-bubble anti-clamp scanner

### Part B: Source-mining lock

`.agent/decisions/source_mining_operator_table.md` with rows for at minimum:

- `_mu_continuity_increment` tanh limiter
- `MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = 0.38`
- `MPAS_OMEGA_TO_W_METRIC = 1.35`
- Hyperdiffusion / horizontal smoothing
- Rayleigh damping (W, top layer)
- Divergence damping (smdiv 2D + potential 3D)
- Time averaging: `t_2ave`, `ww`, `muave`, `muts`, `ph_tend`
- Tridiagonal vertical solve coefficients

Four columns:
1. Canonical source line (WRF or MPAS Fortran)
2. Pace / ICON4Py / Dinosaur port reference (if any)
3. ADR-023 / ADR-021 current status (correct / approximate / missing)
4. Minimum fix to bring code in line with canonical

≥ 8 rows. Every cell with `file:line` OR `(not locally accessible — cited from public commit X)`.

### Part C: Smoke test suite

`tests/test_m6x_s1_diagnostic_sidecars.py` — one test per sidecar:
1. Construct minimal synthetic input
2. Invoke sidecar `main()`
3. Assert output JSON has documented top-level keys
4. Do NOT validate diagnostic correctness in absolute terms (that's S2's job)

All smoke tests must PASS.

### Part D: No regression

```bash
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m3_transfer_audit.py -v
```
All pass.

### Part E: Worker report

`worker-report.md`:
- One-paragraph summary
- Per sidecar: purpose, input schema, output schema, build hours, source citations
- Source-mining table summary (row count, fix-needed status count)
- Files changed, commands, proof objects, risks, handoff

Branch commits on `worker/gpt/m6x-s1-diagnostic-foundation`. Multiple commits OK.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_s1_diag
pytest tests/test_m6x_s1_diagnostic_sidecars.py -v | tee .agent/sprints/2026-05-23-m6x-s1-diagnostic-foundation/proof_sidecar_smoke.txt
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-23-m6x-s1-diagnostic-foundation/proof_no_regression.txt
```

## Performance Metrics

None.

## Proof Object

- `proof_sidecar_smoke.txt` + `proof_no_regression.txt`
- 12 `scripts/diagnostic_*.py` + smoke test file + source-mining table
- `worker-report.md`

Time budget: **8-14 hours**. Single coherent sprint; don't split — the sidecars share JSON schema conventions and the source-mining lock is small once rows are identified.

## Risks

- **Scope creep**: sidecars are diagnostic capture tools. Sophistication belongs in interpretation, not the tool.
- **Schema drift**: all sidecars write JSON with consistent top-level structure. Use warm-bubble harness JSON as prototype.
- **Source-mining fabrication**: every `file:line` checked. If a project isn't locally accessible, mark `(cited from public commit X)`.
- **Spec-gaming**: M5 pattern — smoke tests must actually exercise each sidecar (not just import it).
- **Test fixture dependency**: synthetic minimal fixture in the smoke test if Gen2 unavailable.

## Handoff Requirements

When all 12 sidecars exist + import cleanly, source-mining table ≥ 8 rows on disk, smoke tests pass, no-regression passes, worker-report committed: type `/exit` as a slash command. Wrapper sends `AGENT REPORT [worker / m6x-s1-diagnostic-foundation / codex] exit=<ec>` to manager pane.

## Failure modes the manager will reject

- Modifying any file under `src/gpuwrf/`.
- Sidecars that aren't actually runnable (smoke tests must demonstrate).
- Source citations without `file:line` precision.
- Fewer than 12 sidecars or 8 source-mining table rows.
- Skipping smoke test suite.
- Self-promoting ADRs or modifying governance.
