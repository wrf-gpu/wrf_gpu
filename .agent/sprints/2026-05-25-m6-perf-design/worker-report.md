# Worker Report — M6 Perf Design

## objective

Build the operational-mode runtime entry point, run the Stage 1.5 solver bakeoff, write ADR-026, and preserve the M6B validation-mode baseline without importing validation-only scratch into production carry.

## files changed

- `src/gpuwrf/runtime/__init__.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `scripts/m6_perf_design_canary_1h.py`
- `tests/test_m6_operational_mode_no_h2d.py`
- `tests/test_m6_operational_mode_parity_envelope.py`
- `.agent/decisions/ADR-026-operational-mode-design-DRAFT.md`
- `.agent/sprints/2026-05-25-m6-perf-design/artifacts/*`
- `.agent/sprints/2026-05-25-m6-perf-design/worker-report.md`

## commands run

- `PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6_perf_design_canary_1h.py --solver-bakeoff --samples 3`
- `GPUWRF_CUDA_PROFILER_RANGE=1 nsys profile --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop --output=.agent/sprints/2026-05-25-m6-perf-design/artifacts/proof_solver_bakeoff_nsight --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none python scripts/m6_perf_design_canary_1h.py --profile-only`
- `nsys stats --force-export=true --force-overwrite=true --report cuda_gpu_mem_size_sum --format json --output .agent/sprints/2026-05-25-m6-perf-design/artifacts/proof_solver_bakeoff_nsight_loop_mem .agent/sprints/2026-05-25-m6-perf-design/artifacts/proof_solver_bakeoff_nsight.nsys-rep`
- `PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6_perf_design_canary_1h.py --operational-smoke --hours 0.002777777777777778`
- `PYTHONPATH=src python scripts/m6_perf_design_canary_1h.py`

## proof objects produced

- `proof_solver_bakeoff.json`
- `proof_solver_bakeoff_nsight.nsys-rep`
- `proof_solver_bakeoff_nsight.sqlite`
- `proof_solver_bakeoff_nsight_loop_mem_cuda_gpu_mem_size_sum.json`
- `hlo/m6b2_lax_scan_thomas.hlo.txt`
- `hlo/pure_pcr.hlo.txt`
- `hlo/hybrid_pcr_thomas_refinement.hlo.txt`
- `hlo/xla_tridiagonal_solve_reference.hlo.txt`
- `proof_operational_smoke.json`
- `proof_acceptance_status.json`

## solver decision

Pure PCR measured fastest on the isolated d02-shape solve (`0.206 ms` vs Thomas `0.713 ms`) with ~1e-16 residuals and zero H2D/D2H in the Nsight-captured warmed solver loop. It is not promoted because the sprint contract requires cuSPARSE/cuSolverDx references and Tier-4 golden 1h before promotion. Operational mode therefore keeps Thomas as the conservative solver and records PCR as the M7 optimization candidate.

## unresolved risks

- Full Tier-4 golden 1h was not run for `run_forecast_operational`.
- Full 28-rank CPU WRF speed comparison was not rerun.
- Full forecast-loop Nsight no-transfer proof is still missing; current Nsight proof covers solver loop only.
- Direct cuSPARSE/cuSolverDx benchmark references are not implemented in this repo.

## next decision needed

Run a follow-up acceptance sprint that binds cuSPARSE/cuSolverDx or explicitly rejects them, runs full golden 1h Tier-4 + Tier-2 invariants, captures full forecast Nsight, then decides whether PCR can replace Thomas.
