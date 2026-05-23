# Worker Report — M6.x ADR-023 Public Scan Path Unification

Summary: Unified the public nonhydrostatic scan with the MPAS recurrence path, restored the missing warm-bubble slice fixture via generate-on-first-use plus manifest, and kept ADR-023 PROPOSED because the unified public warm-bubble path is finite but fails the 600 s warm-bubble target. Reviewer `b2f7a05` identified the path split at `.agent/sprints/2026-05-23-m6x-adr023-production-grade-reviewer/reviewer-report.md:70-76` and summarized the reject at lines 86-92.

## Files Changed

- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `tests/test_m6x_adr023_path_unification.py`
- `tests/test_m6x_adr023_production_grade.py`
- `tests/test_m6x_mpas_column_slice_oracle.py`
- `fixtures/manifests/mpas_column_slice_warm_bubble_2km.json`
- `.agent/decisions/ADR-023-conservative-column-solver.md`
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/launch_probe_unified.py`
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/worker-report.md`
- Proof text/json/csv artifacts in this sprint folder.

## What Changed

- Removed the separate `_wrf_buoyancy_column_update` callable and the positive-only updraft drag path.
- Removed the prototype-named `NONHYDROSTATIC_BUOYANCY_SCALE`; the MPAS recurrence uses `MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = 0.38`, matching the existing MPAS slice tendency scale.
- Routed `vertical_acoustic_update(..., pressure_scale=0.0)` and the public `non_hydrostatic=True` scan (`pressure_scale=-1.0`) through the same `_mpas_recurrence_vertical_update`.
- Plumbed `epssm` through the public scan; public warm-bubble sweep now differs across `epssm={0.0,0.1,0.3}`.
- Preserved legacy `base_state=None` pressure aliases so the c2 smdiv pressure-memory smoke test remains valid.
- Confirmed c2-A2 horizontal PGF and `mu_continuity_tendency` are untouched. The separate `_mu_continuity_increment` remains as a documented temporary stabilizer because removing it made the unified warm-bubble run nonfinite at step 2.

## Commands Run

- `pytest tests/test_m6x_adr023_path_unification.py -v | tee .../proof_unification_gate.txt` → `4 passed in 6.18s`.
- `pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_production_grade.py -v | tee .../proof_full_regression.txt` → `23 passed in 19.11s`.
- `pytest tests/test_m3_transfer_audit.py tests/test_m6x_c2_acoustic.py::test_acoustic_scan_jaxpr_has_scan_and_no_host_callbacks -v | tee .../proof_transfer_audit.txt` → `5 passed in 2.76s`.
- `rm -f data/fixtures/mpas_column_slice/warm_bubble_2km.npz` then `pytest tests/test_m6x_mpas_column_slice_oracle.py -v | tee .../proof_fixture_restore.txt` → `4 passed in 3.31s`; fixture regenerated locally.
- `python scripts/m6_warm_bubble_test.py --output .../proof_warm_bubble_unified.json | tee .../proof_warm_bubble_unified.txt` → `FAIL_TARGETS_NOT_MET`, finite through 600 s, `w_max=0.2890913445` at 300 s and `0.0409710985` at 600 s.
- Public-path epssm sweep inline proof → outputs differ; `w_max_600s` is `0.1682302069` for `epssm=0.0`, `0.0409710985` for `0.1`, and `0.0` for `0.3`.
- `nsys profile ... launch_probe_unified.py` plus `nsys stats ...` → unified `pressure_scale=-1.0` vertical recurrence still has `cuLaunchKernelEx_calls=67`, `cuMemcpyHtoDAsync_v2_calls=0`, `cuMemcpyDtoHAsync_v2_calls=0`, `cuMemcpyDtoDAsync_v2_calls=48`.

## Proof Objects

- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/proof_unification_gate.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/proof_full_regression.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/proof_transfer_audit.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/proof_fixture_restore.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/proof_warm_bubble_unified.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/proof_warm_bubble_unified.json`
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/proof_public_path_epssm_sweep.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/proof_launch_count_unified.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/artifacts/launch_probe_unified_stats_cuda_api_sum.csv`
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/artifacts/launch_probe_unified_stats_cuda_gpu_kern_sum.csv`
- `fixtures/manifests/mpas_column_slice_warm_bubble_2km.json`

## Risks

- Critical evidence: after path unification, the conservative recurrence does not pass the warm-bubble rung. It stays finite only with the documented temporary `mu` bound and still fails target amplitude/centroid behavior.
- The temporary mass limiter is still a validation stabilizer and must be replaced or ratified before d02 or 24 h replay.
- Launch count remains 67 with 48 device-to-device copies, so there is no speed claim.
- The binary `.npz` remains under gitignored `data/`; fresh checkouts regenerate it through the test path and manifest rather than committing binary fixture data.

## Handoff

Objective: close reviewer path-split and fixture-repro findings without adding new physics.

Files changed: listed above.

Commands run: listed above with outputs.

Proof objects produced: listed above.

Unresolved risks: unified warm-bubble failure, temporary `mu` limiter, launch-count/memcpy overhead.

Next decision needed: reviewer/manager must decide whether ADR-023 should fall back/amend before d02 replay, because the public conservative path is now unified but not warm-bubble production-ready.
