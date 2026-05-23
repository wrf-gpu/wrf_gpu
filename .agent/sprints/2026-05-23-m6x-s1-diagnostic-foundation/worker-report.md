# Worker Report - M6.x S1 Diagnostic Foundation

Summary: Built the diagnostic sidecar foundation specified by the sprint contract: 12 read-only `scripts/diagnostic_*.py` tools, one synthetic-input smoke test suite, and the source-mining operator table. No files under `src/gpuwrf/` were modified. The sidecars all accept `--input` and `--output`, expose `main(args)`, and emit JSON using the shared top-level schema `schema_version`, `diagnostic`, `input`, `measurements`, `units`, `artifacts`, `status`, and `source_citations`. Smoke tests and the required no-regression bundle both passed.

## Sidecars

| Sidecar | Purpose | Input schema | Output schema | Build hours | Source citations |
|---|---|---|---|---:|---|
| `diagnostic_bound_violation_tracer.py` | First finite/physical-bound violation. | JSON `series`/`samples` plus optional `bounds`. | `measurements.first_violation`, `violations`, `first_nonfinite`. | 0.5 | `scripts/m6_warm_bubble_test.py:72-78`, `scripts/m6_warm_bubble_test.py:275-304` |
| `diagnostic_sanitizer_audit.py` | Pre-sanitize nonfinite/clip/change audit. | JSON `sanitizer_steps` or replay `diagnostics`. | `per_step`, `first_bad_candidate_step`, `totals`. | 0.4 | `src/gpuwrf/integration/d02_replay.py:313-322`, `src/gpuwrf/integration/d02_replay.py:468-487` |
| `diagnostic_limiter_activation_tracker.py` | Mu-limiter saturation evidence. | JSON `limiter_steps` with `raw_dmu` and `bounded_dmu`. | `per_step`, `max_saturation_fraction`, `limiter_active`. | 0.5 | `src/gpuwrf/dynamics/acoustic_wrf.py:456-475`, `src/gpuwrf/dynamics/acoustic_wrf.py:872-876` |
| `diagnostic_field_rmse_timeline.py` | Field RMSE/bias/max-error by lead. | JSON `leads[].forecast/reference` or replay `comparison`. | `timeline` by lead and field. | 0.5 | `src/gpuwrf/integration/d02_replay.py:515-546`, `scripts/diagnostic_gen2_rmse_baseline.py:1-9` |
| `diagnostic_spatial_divergence_map.py` | Error maps and strata by boundary/terrain/land/elevation. | JSON or `.npz` forecast/reference 2-D fields. | JSON summary plus sibling `.npz` error map. | 0.7 | `src/gpuwrf/integration/d02_replay.py:515-546`, `src/gpuwrf/integration/d02_replay.py:142-206` |
| `diagnostic_conservation_tracker.py` | Integral mass/water/KE/dry-static drift. | JSON `states[]` with totals or arrays. | `time_series`, `max_abs_relative_drift`. | 0.5 | `.agent/milestones/ROADMAP.md:78-98` |
| `diagnostic_boundary_ring_error_profiler.py` | RMSE by 0-5, 5-10, 10-20, interior bands. | JSON 2-D forecast/reference field. | `ring_rmse` list. | 0.4 | `src/gpuwrf/integration/d02_replay.py:99-140`, `src/gpuwrf/coupling/boundary_apply.py:31-77` |
| `diagnostic_vertical_column_phase_space.py` | Selected-column profiles and phase portraits. | JSON `columns[]` with `profiles` and `time_series`. | `columns[].vertical_profiles`, `phase_portraits`. | 0.5 | `scripts/diagnostic_warm_bubble_vs_slice.py:151-213`, `scripts/m6_warm_bubble_test.py:88-177` |
| `diagnostic_operator_term_budget_tracer.py` | Per-term tendency max/mean/L2 ranking. | JSON `terms` by named RHS component. | `per_term`, `extra_terms`, `ranking`. | 0.5 | `.agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/diagnostic-report.md:71-90`, `src/gpuwrf/dynamics/acoustic_wrf.py:735-747` |
| `diagnostic_transfer_launch_timeline.py` | Transfer/callback/launch/memory summary. | JSON replay proof with `transfer_audit`. | callback flag, H2D/D2H bytes, launches, peak memory. | 0.4 | `src/gpuwrf/integration/d02_replay.py:549-632` |
| `diagnostic_timestep_convergence_dashboard.py` | Placeholder Tier-3 dt-pair norm schema. | JSON `dt_pairs[]` with coarse/fine fields. | dt-pair norms and `PLACEHOLDER_PENDING_S4` verdict. | 0.4 | `.agent/milestones/ROADMAP.md:89-98`, `.agent/sprints/2026-05-23-m6x-close-strategy-plan-critic/reviewer-report.md:115-123` |
| `diagnostic_stabilizer_provenance_scanner.py` | Classify stabilizers as source-backed/experiment-backed/reject. | Source dir/file or JSON `source_files`. | `findings`, `classification_counts`. | 0.5 | `scripts/m6_warm_bubble_test.py:326-360`, `src/gpuwrf/dynamics/acoustic_wrf.py:456-475` |

## Source-Mining Table

Wrote `.agent/decisions/source_mining_operator_table.md` with 8 rows covering all contract-required operator concerns. Status counts in the table: 1 correct target-shape row, 5 approximate/blocked rows, 2 missing-as-accepted-production-evidence rows. WRF and MPAS citations were checked against local source; Pace, ICON4Py, and Dinosaur references were checked against `/tmp/wrf_gpu2_refs/` snapshots.

## Files Changed

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
- `tests/test_m6x_s1_diagnostic_sidecars.py`
- `.agent/decisions/source_mining_operator_table.md`
- `.agent/sprints/2026-05-23-m6x-s1-diagnostic-foundation/proof_sidecar_smoke.txt`
- `.agent/sprints/2026-05-23-m6x-s1-diagnostic-foundation/proof_no_regression.txt`
- `.agent/sprints/2026-05-23-m6x-s1-diagnostic-foundation/worker-report.md`

## Commands Run

`set -o pipefail; pytest tests/test_m6x_s1_diagnostic_sidecars.py -v | tee .agent/sprints/2026-05-23-m6x-s1-diagnostic-foundation/proof_sidecar_smoke.txt`

Output: 12 tests collected; all 12 passed in 0.05 s. Full stdout/stderr captured in `proof_sidecar_smoke.txt`.

`set -o pipefail; pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-23-m6x-s1-diagnostic-foundation/proof_no_regression.txt`

Output: 33 tests collected; all 33 passed in 32.59 s. Full stdout/stderr captured in `proof_no_regression.txt`.

## Proof Objects

- `.agent/sprints/2026-05-23-m6x-s1-diagnostic-foundation/proof_sidecar_smoke.txt`
- `.agent/sprints/2026-05-23-m6x-s1-diagnostic-foundation/proof_no_regression.txt`
- `.agent/decisions/source_mining_operator_table.md`
- `tests/test_m6x_s1_diagnostic_sidecars.py`
- 12 `scripts/diagnostic_*.py` sidecars listed above

## Risks

- These are S1 capture tools, not physical verdict tools. Several sidecars intentionally return `NO_*` or `STRUCTURE_ONLY` when supplied incomplete inputs.
- The timestep-convergence dashboard does not claim Tier-3 success; S4 owns actual convergence pass/fail evidence.
- The source table locks line citations and minimum fixes, but it does not approve any operator edit.
- The launch instruction says push, while the sprint contract Non-Goals says "No remote push." I followed the sprint contract and prepared a local branch commit only.
- Pre-existing untracked `scripts/dispatch_role_session2.sh` is outside file ownership and was left untouched.

## Handoff

Objective: build M6.x S1 diagnostic sidecars and source-mining lock before any operator changes.

Files changed: listed above; no `src/gpuwrf/` files edited.

Commands run: listed above with output summaries and full proof logs.

Proof objects produced: listed above.

Unresolved risks: diagnostic schemas may need small extensions after S2 emits real d02 replay artifacts, but the CLI/import/JSON contract is present and tested.

Next decision needed: manager can dispatch S2 baseline replay against these sidecars; S3 operator work should consume the source-mining table.
