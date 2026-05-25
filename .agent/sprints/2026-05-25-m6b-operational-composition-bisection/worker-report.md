# Worker Report - M6b Operational Composition Bisection

## objective

Bisect operational-mode composition against the locked validation coupled-step path on the real Gen2 d02 `20260523_18z_l3_24h_20260524T004313Z` initial condition, localize the first divergence by timestep/RK/acoustic/operator, name one defect with WRF source citation, and make no fix attempt.

## verdict

`OPERATIONAL-COMPOSITION-DEFECT-LOCALIZED-AT-RK1-ACOUSTIC-LOOP-OMISSION`.

Operational mode diverges from validation after the first requested timestep. The controlled substep bisection starts both paths from the same RK1 pre-acoustic candidate and then compares operational's RK1 output against one WRF-shaped validation acoustic substep. Operational runs no RK1 acoustic small step; validation changes `theta`, `mu`, `ww`, `w`, and scratch immediately.

## Stage 1 - step-level bisection

Proof: `proof_bisection_step_level.json`; command transcript: `proof_bisection_run.txt`.

| run_id | requested steps | executed to first divergence | first bad step | largest bad field | selected deltas |
|---|---:|---:|---:|---|---:|
| `20260523_18z_l3_24h_20260524T004313Z` | 70 | 1 | 1 | `theta` | `theta=1.0e300` nonfinite sentinel, `ph=195220.88016493057`, `mu=95000.0636501736`, `w=0.20655372496940294` |

The full endpoint comparison ran both `run_forecast_operational` and validation `coupled_timestep_wrf` on the same real IC. It stops at step 1 because the divergence threshold `1e-10` is already exceeded.

## Stage 2 - substep bisection

Proof: `proof_bisection_substep_level.json`.

| timestep | RK stage | acoustic substep | operational action | validation action | largest bad field |
|---:|---:|---:|---|---|---|
| 1 | 1 | 1 | advection candidate only; acoustic loop skipped | `acoustic_substep_wrf` applied to the same RK1 pre-acoustic candidate | `theta=5506.97802734375` |

Operator table at the same boundary:

| operator family | max delta | note |
|---|---:|---|
| `calc_coef_w` / W solve | `w=0.0017771519458527063` | first nonzero acoustic-family effect because the whole substep is absent operationally |
| `advance_mu_t` | `theta=5506.97802734375`, `mu=49.854842928446715`, `ww=15.217415205461393` | largest physical divergence |
| `scratch_updates` | `t_2ave=2753.529296875`, `ph_tend=55.06977844238281` | scratch follows from missing substep |

## Stage 3 - named defect and WRF citation

Defect: `src/gpuwrf/runtime/operational_mode.py::_rk_scan_step` dispatches RK stage 1 as `advance_stage(..., use_acoustic=False)`. That means no acoustic small-step operator runs in RK1.

WRF source says RK3 stage 1 must run a small-step loop:

- `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/solve_em.F:1447` starts `Runge_Kutta_loop`.
- `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/solve_em.F:1472-1475` sets RK3 stage 1 `number_of_small_timesteps = 1`.
- `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/solve_em.F:3065` starts `small_steps`.
- `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/solve_em.F:3435-3452` calls `advance_mu_t` inside `small_steps`.
- `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:1102-1108` updates `MU/MUDF/MUTS/MUAVE`; `:1162-1171` updates theta.

Recommended minimal fix for a follow-up sprint: make operational RK stage 1 execute the WRF-required acoustic small-step path instead of `use_acoustic=False`. Preserve the common RK1 pre-acoustic tendency candidate, run one WRF-shaped small step for RK1 per `solve_em.F:1472-1475`, and rerun this bisection before addressing later-stage defects. No fix was attempted here.

## files changed

- `scripts/m6b_operational_vs_validation_compare.py`
- `tests/test_m6b_operational_vs_validation_compare.py`
- `.agent/sprints/2026-05-25-m6b-operational-composition-bisection/proof_bisection_run.txt`
- `.agent/sprints/2026-05-25-m6b-operational-composition-bisection/proof_bisection_step_level.json`
- `.agent/sprints/2026-05-25-m6b-operational-composition-bisection/proof_bisection_substep_level.json`
- `.agent/sprints/2026-05-25-m6b-operational-composition-bisection/proof_no_regression.txt`
- `.agent/sprints/2026-05-25-m6b-operational-composition-bisection/worker-report.md`

## commands run

- `python -m py_compile scripts/m6b_operational_vs_validation_compare.py tests/test_m6b_operational_vs_validation_compare.py`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6b_operational_vs_validation_compare.py --gen2-run-id 20260523_18z_l3_24h_20260524T004313Z --steps 70 2>&1 | tee .agent/sprints/2026-05-25-m6b-operational-composition-bisection/proof_bisection_run.txt`
- `pytest tests/test_m6b_operational_vs_validation_compare.py -v`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_carry_expansion_*.py tests/test_m6b_honest_v2_*.py tests/test_m6b_operational_vs_validation_*.py -v 2>&1 | tee .agent/sprints/2026-05-25-m6b-operational-composition-bisection/proof_no_regression.txt`

## proof objects produced

- `proof_bisection_step_level.json`
- `proof_bisection_substep_level.json`
- `proof_bisection_run.txt`
- `proof_no_regression.txt` (`145 passed in 436.42s`)

## unresolved risks

- This sprint names the first composition defect only. Later defects are still possible after RK1 acoustic execution is restored.
- The step-level full coupled endpoint contains nonfinite-field deltas, recorded as `1.0e300` sentinel values. The controlled RK1 substep proof is the decisive localization artifact.
- The follow-up fix should re-run this comparator before changing the inline scratch formulas suspected by the carry-fix report.

## next decision needed

Dispatch `m6b-operational-composition-fix-rk1-acoustic-loop-omission`. Keep validation-mode code locked, and do not broaden the fix to later RK-stage cadence or scratch formulas until this first defect is corrected and re-bisected.
