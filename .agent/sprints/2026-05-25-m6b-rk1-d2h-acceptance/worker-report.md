# Worker Report - M6b RK1+D2H Acceptance

## objective

Integrate the D2H lift into `src/gpuwrf/runtime/operational_mode.py`, re-verify RK1 parity on Gen2 run IDs with available `wrfout_d02*`, confirm warmed inter-kernel D2H is zero, run the 1h x 3 operational acceptance harness with per-level theta bounds, and update ADR-027 to PROPOSED evidence.

## verdict

`BLOCKER - DO-NOT-CLOSE-M6`.

The D2H lift is successful: warmed Nsight reports `h2d_total=0` and `d2h_inter_kernel=0` after replacing dynamic RK-stage `jax.lax.switch` and radiation `jax.lax.cond` control flow with static operational sequencing.

M6 cannot close. RK1 parity fails on all four wrfout-rich Gen2 IDs at executed step 1, and the 1h x 3 harness fails per-level theta bounds before Tier-4 RMSE or spatial-divergence comparison can run.

## files changed

- `src/gpuwrf/runtime/operational_mode.py`
- `scripts/m6b_canary_1h_honest_v2.py`
- `tests/test_m6b_rk1_d2h_acceptance.py`
- `.agent/decisions/ADR-027-d2h-invariant-clarification-PROPOSED.md`
- `.agent/sprints/2026-05-25-m6b-rk1-d2h-acceptance/*` proof artifacts and this report

## implementation summary

- Replaced dynamic RK scan/switch with static RK1/RK2/RK3 sequencing. RK1 still runs one acoustic small step per `solve_em.F:1472-1475`.
- Replaced dynamic radiation cadence `lax.cond` inside the timestep scan with static forecast segmentation and one-step radiation segments.
- Updated the honest 1h harness defaults to wrfout-rich run IDs:
  `20260509_18z_l3_24h_20260511T190519Z`,
  `20260521_18z_l3_24h_20260522T072630Z`,
  `20260521_18z_l3_24h_20260522T133443Z`.
- Added acceptance tests that verify the D2H lift source shape and prevent a false `CLOSE-M6` when proof artifacts are blockers.
- Updated ADR-027 with the post-lift measured evidence: `d2h_inter_kernel=0`, `d2h_pre_kernel=25`.

## proof summary

| gate | artifact | result |
|---|---|---|
| RK1 parity on wrfout-rich IDs | `proof_rk1_parity_all_ids.json` | FAIL: all 4 IDs diverge at executed step 1 for requested steps 1 and 10; largest field `theta`, `max_abs_delta=1e300` |
| Required step-1 parity file | `proof_rk1_parity_step1.json` | FAIL on `20260521_18z_l3_24h_20260522T072630Z`; threshold `1e-10` |
| Required step-10 parity file | `proof_rk1_parity_step10.json` | FAIL on same ID; comparator still diverges at executed step 1; threshold `1e-8` |
| Warmed D2H | `proof_d2h_warmed_inter_kernel_zero.json` | PASS: `h2d_total=0`, `d2h_inter_kernel=0`, `d2h_pre_kernel=25` |
| 1h x 3 bounds | `proof_m6b_1h_runs.json`, `proof_bounds.json` | BLOCKER: `THETA_BOUNDS`; first bad steps 10, 35, 35 |
| Tier-4 RMSE | `proof_tier4_rmse.json` | NOT_RUN: blocked before valid 1h RMSE by theta bounds |
| Spatial divergence | `proof_spatial_divergence.json` | NOT_RUN: blocked before spatial audit by theta bounds |
| B6 regression | `proof_b6_regression.txt`, `proof_b6_regression_summary.json` | PASS: `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED`, `max_abs_delta: 0.0` present |
| New acceptance tests | `proof_acceptance_tests.txt` | PASS: 3 passed |
| Contracted pytest selection | `proof_pytest.txt`, `proof_pytest_summary.json` | FAIL: 213 passed, 5 failed |

## commands run

- `python -m py_compile src/gpuwrf/runtime/operational_mode.py scripts/m6b_canary_1h_honest_v2.py tests/test_m6b_rk1_d2h_acceptance.py`
- `PYTHONPATH=src pytest tests/test_m6b_fix_rk1_acoustic_loop.py::test_operational_rk1_dispatch_runs_one_acoustic_substep tests/test_m6b_rk1_d2h_acceptance.py::test_operational_mode_lifts_localized_dynamic_d2h_emitters tests/test_m6b_rk1_d2h_acceptance.py::test_honest_1h_defaults_use_wrfout_rich_gen2_ids -q`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6b_operational_vs_validation_compare.py --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --steps 1`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6b_operational_vs_validation_compare.py --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --steps 10`
- Single-process parity sweep over the four requested wrfout-rich IDs for requested steps 1 and 10, writing `proof_rk1_parity_all_ids.json`
- `taskset -c 0-3 nsys profile --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --output=.agent/sprints/2026-05-25-m6b-rk1-d2h-acceptance/proof_warmed env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 GPUWRF_CUDA_PROFILER_RANGE=1 python scripts/m6b_d2h_warmed_recapture.py --profile-steps 5`
- `env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6b_d2h_warmed_recapture.py --parse-rep .agent/sprints/2026-05-25-m6b-rk1-d2h-acceptance/proof_warmed.nsys-rep`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6b_canary_1h_honest_v2.py --runs 3 --hours 1`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6b6_coupled_step_compare.py --tier golden`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 pytest tests/test_m6b_rk1_d2h_acceptance.py -v`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 pytest tests/test_m6x_*.py tests/test_m3_*.py tests/test_m6b*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py -v`

## proof objects produced

- `proof_rk1_parity_all_ids.json`
- `proof_rk1_parity_step1.json`
- `proof_rk1_parity_step10.json`
- `proof_d2h_warmed_inter_kernel_zero.json`
- `proof_warmed.transfer_summary.json`
- `proof_m6b_1h_runs.json`
- `proof_bounds.json`
- `proof_tier4_rmse.json`
- `proof_spatial_divergence.json`
- `proof_b6_regression.txt`
- `proof_b6_regression_summary.json`
- `proof_acceptance_tests.txt`
- `proof_pytest.txt`
- `proof_pytest_summary.json`
- `proof_operational_mode_audit.json`
- `proof_performance.json`
- `proof_d2h_inheritance.json`

The raw `proof_warmed.nsys-rep` and exported `proof_warmed.sqlite` exist locally but are ignored by `.gitignore`; the committed transfer summary is `proof_warmed.transfer_summary.json`.

## unresolved risks

- RK1 parity is still not closed. The comparator reports divergence at executed step 1 on every requested wrfout-rich ID.
- The comparator proof text still carries the old localized-defect wording in its nested substep payload; the step-level parity result is the relevant acceptance blocker here.
- The 1h x 3 harness fails theta bounds before RMSE and spatial-divergence evidence can be produced.
- The contracted pytest selection fails 5 tests outside this sprint's write scope: four M3 contract tests and the old D2H warmed artifact-presence test for an ignored `.nsys-rep` in the older sprint directory.

## next decision needed

Dispatch a narrow operational composition sprint for the remaining RK1/theta defect before any M6 close attempt. Recommended M6 close decision: `DO-NOT-CLOSE-M6`.
