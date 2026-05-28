# Worker Report - M9.C Theta Comparator Audit

## Verdict

`M9C_COMPLETE`

The comparator is now trustworthy for theta convention handling. The corrected comparison did not reduce the current Canary 20260521 theta divergence: GPU wrfout `T` was detected as WRF-compatible perturbation theta in all 24 compared files, so the previous 77.36 K theta RMSE is a real downstream/model-output divergence target, not a 300 K comparator artifact.

## Objective

Audit and fix `scripts/operational_trace_compare.py` against WRF wrfout conventions, focused on theta absolute-vs-perturbation handling, then re-run the Canary 20260521 hourly trace and emit updated proof objects.

No `src/**` model code was modified.

## Files Changed

- `scripts/operational_trace_compare.py`
- `proofs/m9/operational_trace_hourly_v2.json`
- `proofs/m9/divergence_map_v2.json`
- `.agent/sprints/2026-05-28-m9c-theta-comparator-audit/comparator_audit.md`
- `.agent/sprints/2026-05-28-m9c-theta-comparator-audit/worker-report.md`

## Commands Run

- `sed`/`rg`/`ncdump` audits of `PROJECT_CONSTITUTION.md`, `AGENTS.md`, sprint contract, project-local skills, comparator, state contract, operational runtime, wrfout writer, WRF Registry, and the Canary WRF/GPU NetCDF headers.
- `taskset -c 0-3 python -m py_compile scripts/operational_trace_compare.py`
- `taskset -c 0-3 python scripts/operational_trace_compare.py --case 20260521 --gpu-pipeline-run .agent/sprints/2026-05-28-m9a-trace-harness/gpu_rerun/pipeline_run_20260521.json --wrf-root /mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z --hours 24 --output proofs/m9/operational_trace_hourly_v2.json`
- `taskset -c 0-3 python - <<'PY' ... PY` to generate `proofs/m9/divergence_map_v2.json` from v1/v2 traces.
- `python -m json.tool proofs/m9/divergence_map_v2.json`
- `python -m json.tool proofs/m9/operational_trace_hourly_v2.json`
- `taskset -c 0-3 pytest -q tests/savepoint/ tests/test_m6b6_coupled_step_parity.py` -> `6 passed, 3 xfailed in 462.63s`

## Proof Objects Produced

- `proofs/m9/operational_trace_hourly_v2.json`
  - Status: `FAIL`, as expected for a divergence trace.
  - Matched hours: 24.
  - GPU `T` detected as `perturbation_from_300K_base` for 24/24 hours.
  - WRF `T` detected as `perturbation_from_300K_base` for 24/24 hours.

- `proofs/m9/divergence_map_v2.json`
  - Viability verdict: `VIABLE`.
  - Theta old RMSE: `77.3589529727313 K`.
  - Theta new RMSE: `77.3589529727313 K`.
  - Headline reduction: `0.0%`.
  - New top-ranked likely defect: surface heat flux magnitude/sign or state coupling, followed by radiation/output and real theta divergence.

- `.agent/sprints/2026-05-28-m9c-theta-comparator-audit/comparator_audit.md`
  - Lists the required field-by-field GPU/WRF conventions, verdicts, and the exact theta converter fix.

## Unresolved Risks

- The current v2 trace is still wrfout-level, not a per-operator WRF Fortran trace.
- `P`, `PB`, and `PH` conventions were audited but not added to the 16-field M9 trace schema.
- T2, SWDOWN/GLW, HFX/LH, PBLH, and LU_INDEX remain materially divergent and need M10-M13 follow-up.

## Next Decision Needed

Proceed to M10/M11 with the corrected comparator. Do not spend M11 assuming a 300 K theta comparator artifact; the current evidence says theta divergence is real in the GPU output path.
