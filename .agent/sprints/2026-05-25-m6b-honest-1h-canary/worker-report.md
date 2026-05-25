# Worker Report — M6b Honest 1h Canary d02

## objective

Run the first sanitizer-off M6b Canary d02 operational-mode acceptance attempt on 3 pinned Gen2 run-IDs using `operational_mode.run_forecast_operational`, then recommend `CLOSE-M6` or `BLOCKER`.

## verdict

**M6 close recommendation: `BLOCKER`.**

The current operational path is not a valid 1h forecast candidate. All three pinned real Gen2 d02 states remained finite for the first 10-second operational step, but all violated theta bounds immediately, with extreme vertical-wind growth. Downstream 1h RMSE, spatial-divergence, and wall-clock gates were therefore not honestly claimable.

## per-run gate table

| run_id | completed | finite | theta range K | max \|u\|/\|v\|/\|w\| m/s | result |
|---|---:|---|---:|---:|---|
| `20260509_18z_l3_24h_20260511T190519Z` | 1 step / 10 s | yes | 15.75 .. 1.422e15 | 273.81 / 102.37 / 7.114e14 | `BLOCKER: THETA_BOUNDS` |
| `20260521_18z_l3_24h_20260522T072630Z` | 1 step / 10 s | yes | 25.66 .. 6.633e14 | 163.62 / 49.58 / 1.548e14 | `BLOCKER: THETA_BOUNDS` |
| `20260523_18z_l3_24h_20260524T004313Z` | 1 step / 10 s | yes | 55.31 .. 1.879e13 | 94.03 / 40.31 / 2.285e11 | `BLOCKER: THETA_BOUNDS` |

## acceptance gates

| Gate | Status | Evidence |
|---|---|---|
| Operational mode, not validation | PASS | `proof_operational_mode_audit.json`; no validation helper imports, callbacks, `device_get`, snapshots, or sanitizer tokens in `src/gpuwrf/runtime/operational_mode.py` |
| No nonfinite | PASS for attempted first step only | `proof_operational_runs.json`, `proof_bounds.json` |
| Theta bounded 200-400 K | FAIL | all 3 runs violate on first 10-second step |
| Wind maxima plausible | FAIL | all 3 runs violate via `w`; 2 also violate `u` and/or `v` |
| T2/U10/V10 RMSE | NOT RUN | blocked before valid 1h output; `proof_tier4_rmse.json` |
| Spatial-divergence audit | NOT RUN | blocked before valid RMSE field; `proof_spatial_divergence.json` |
| Wall-clock vs CPU WRF | BLOCKED | no valid 1h forecast wall-clock; `proof_performance.json` |
| Zero H2D/D2H | FAIL on profiled first-step reproduction; full 1h not verified | `proof_nsys_transfers_inside_loop.json`; H2D=0, D2H=53 in warmed captured range |

## caveats carried from perf-acceptance

- The 28-rank CPU WRF denominator remains recovered after the requested CPU WRF binary aborted in OpenACC; no clean 28-rank rerun is available here.
- HLO availability remains non-binding for this sprint; the blocker is physical-bounds failure before 1h completion.
- The previous 1.2x speedup tripwire was based on a quiescent acceptance state and must not be advertised as honest forecast speedup.

## files changed

- `scripts/m6b_canary_1h_honest.py`
- `tests/test_m6b_honest_acceptance.py`
- `tests/test_m6b_operational_no_h2d.py`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary/proof_*.json`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary/proof_nsys_transfers_inside_loop.txt`
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary/worker-report.md`

## commands run

- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6b_canary_1h_honest.py` → script returned 2 by design because the acceptance verdict is `BLOCKER`.
- `taskset -c 0-3 nsys profile --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --output=.agent/sprints/2026-05-25-m6b-honest-1h-canary/proof_nsys_operational_first_step env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 GPUWRF_CUDA_PROFILER_RANGE=1 python scripts/m6b_canary_1h_honest.py --profile-only`
- `nsys stats --force-export=true --force-overwrite=true --report cuda_gpu_mem_size_sum --format json --output .agent/sprints/2026-05-25-m6b-honest-1h-canary/proof_nsys_operational_first_step_mem .agent/sprints/2026-05-25-m6b-honest-1h-canary/proof_nsys_operational_first_step.nsys-rep`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 pytest tests/test_m6b_honest_acceptance.py tests/test_m6b_operational_no_h2d.py -v` → 5 passed.

## proof objects produced

- `proof_operational_runs.json`
- `proof_bounds.json`
- `proof_tier4_rmse.json`
- `proof_spatial_divergence.json`
- `proof_operational_mode_audit.json`
- `proof_profile_only.json`
- `proof_nsys_operational_first_step.nsys-rep` (local ignored raw profiler artifact)
- `proof_nsys_operational_first_step.sqlite` (local ignored export)
- `proof_nsys_operational_first_step_mem_cuda_gpu_mem_size_sum.json`
- `proof_nsys_transfers_inside_loop.json`
- `proof_nsys_transfers_inside_loop.txt`
- `proof_performance.json`

## unresolved risks

- The first-step blow-up prevents a valid 1h forecast output, so the required RMSE and spatial gates remain unevaluated, not passed.
- The Nsight transfer audit is only for the warmed first-step failure reproduction and already shows nonzero D2H; full 1h zero-transfer status is unverified.
- The root cause is not localized in this sprint. The evidence points at operational real-state coupling/vertical-acoustic behavior rather than validation-mode parity.

## next decision needed

Dispatch an operational-mode fix sprint for first-step real-state instability, with transfer audit included in the fix gate. M6 should remain open.
