# Worker Report - M6b Fix Carry Expansion

## objective

Promote `t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, and `_save` families from operational-undecided to operational-required with Tier-4/blocker evidence, add resident operational carry without importing validation-only helpers, run the 10 s three-run Gen2 bisection probe, and prove B6 golden validation parity remains 0.0.

## verdict

Dispatch recommendation: `READY-FOR-M6b-RETRY`, with one caveat below about the theta-bounds interpretation.

The carry-expanded operational path passes the 10 s bisection probe on the three pinned Gen2 run IDs with B6 acoustic cadence (`acoustic_substeps=10`) and sanitizer off. Validation-mode B6 golden parity remains passed with `max_abs_delta: 0.0` entries in `proof_b6_regression.txt`.

## promotion table

| Field family | Previous classification | New classification | Tier-4/blocker evidence |
|---|---|---|---|
| `t_2ave` | Undecided | Operational-required-with-Tier-4-evidence | Strict-subset operational mode failed all three real Gen2 runs after one 10 s step: `.agent/sprints/2026-05-25-m6b-honest-1h-canary/proof_bounds.json`; critic §1 identifies missing `t_2ave` carry. |
| `ww` | Undecided | Operational-required-with-Tier-4-evidence | Same blocker proof showed extreme `w_abs_max_m_s` up to `7.11e14`; critic §1 identifies missing `ww` recurrence. |
| `muave` | Undecided | Operational-required-with-Tier-4-evidence | Critic §1 ties the failure to absent mass running average; M6B3 proved scratch parity for `muave` at 0.0 but left operational undecided. |
| `muts` | Undecided | Operational-required-with-Tier-4-evidence | Critic §1 ties the failure to absent substep total mass; M6B3 proved scratch parity for `muts` at 0.0 but left operational undecided. |
| `ph_tend` | Undecided | Operational-required-with-Tier-4-evidence | Critic §1 identifies missing geopotential tendency accumulation; M6B3/M6B6 validation lane includes `ph_tend` with 0.0 parity. |
| `_save` family | Undecided | Operational-required-with-Tier-4-evidence | Critic §1 identifies missing RK/acoustic transition save state; current probe threads `u/v/w/t/ph/mu/ww_save` resident through the scan. |

## implementation summary

- Added `src/gpuwrf/runtime/operational_state.py` with `OperationalCarry`, promotion evidence, and initialization from Gen2/operational `State`.
- Updated `src/gpuwrf/runtime/operational_mode.py` to carry promoted scratch through the timestep/RK/acoustic scans.
- Did not import `gpuwrf.dynamics.acoustic_loop`, `dycore_step`, `coupled_step`, or `small_step_scratch`.
- Copied/adapted scratch formulas inline with WRF source anchors:
  - `module_small_step_em.F:1066-1175` for `ww`, `muave`, `muts`.
  - WRF small-step theta averaging for `t_2ave`.
  - WRF small-step geopotential tendency accumulation for `ph_tend`.
- Added `scripts/m6b_carry_expansion_probe.py` and `tests/test_m6b_carry_expansion_bounded.py`.

## 10 s probe results

Probe artifact: `proof_10s_probe.json` and `proof_10s_probe.txt`.

| run_id | finite | theta checked levels | theta range K | max \|u\|/\|v\|/\|w\| m/s | result |
|---|---|---:|---:|---:|---|
| `20260509_18z_l3_24h_20260511T190519Z` | yes | 0:30 | 290.33 .. 353.24 | 53.74 / 14.73 / 1.48 | PASS |
| `20260521_18z_l3_24h_20260522T072630Z` | yes | 0:30 | 288.81 .. 350.95 | 25.66 / 11.48 / 0.92 | PASS |
| `20260523_18z_l3_24h_20260524T004313Z` | yes | 0:30 | 290.61 .. 350.99 | 12.59 / 13.84 / 1.28 | PASS |

Caveat: the raw Gen2 initial full column already has upper-level theta maxima of about 492-500 K, so a literal full-column `<400 K` bound is impossible before any model step. The proof records `full_column_theta_max_k` and applies the bisection theta bound to lower 30 eta levels. This should be clarified before the full 1h retry gate.

## B6 regression

`taskset -c 0-3 ... python scripts/m6b6_coupled_step_compare.py --tier golden` passed. `proof_b6_regression.txt` records outcome `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED`, `passed: true`, and repeated `max_abs_delta: 0.0` field comparisons. Validation-mode files were not modified.

## no regression

Required regression command passed: `139 passed in 421.62s`.

## files changed

- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/runtime/operational_state.py`
- `scripts/m6b_carry_expansion_probe.py`
- `tests/test_m6b_carry_expansion_bounded.py`
- `.agent/sprints/2026-05-25-m6b-fix-carry-expansion/proof_10s_probe.json`
- `.agent/sprints/2026-05-25-m6b-fix-carry-expansion/proof_10s_probe.txt`
- `.agent/sprints/2026-05-25-m6b-fix-carry-expansion/proof_b6_regression.txt`
- `.agent/sprints/2026-05-25-m6b-fix-carry-expansion/proof_no_regression.txt`
- `.agent/sprints/2026-05-25-m6b-fix-carry-expansion/worker-report.md`

## commands run

- `python -m py_compile src/gpuwrf/runtime/operational_state.py src/gpuwrf/runtime/operational_mode.py`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6b_carry_expansion_probe.py --runs 3 --duration-s 10`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6b6_coupled_step_compare.py --tier golden`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_carry_expansion_*.py -v`

## proof objects produced

- `proof_10s_probe.json`
- `proof_10s_probe.txt`
- `proof_b6_regression.txt`
- `proof_no_regression.txt`

## unresolved risks

- Full-column theta `<400 K` is not a valid gate for these initialized Gen2 d02 columns because the initial upper levels already exceed it. The next sprint should either codify a level/window-specific theta bound or replace it with a WRF-consistent invariant.
- Operational prognostic theta/mu replacement from the WRF scratch transcription is not approved here; the sprint carries the promoted scratch resident and keeps prognostic theta/mu on the existing operational state path.
- D2H warmed re-capture remains deferred per the SPLIT-INTO-TWO verdict.

## next decision needed

Dispatch M6b warmed D2H recapture and M6b honest 1h retry, but clarify the theta-bounds window before treating a full-column `<400 K` check as a hard rejection.
