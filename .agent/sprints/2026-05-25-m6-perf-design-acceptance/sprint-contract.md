# Sprint Contract — M6-perf-design Acceptance Follow-up

## Objective

M6-perf-design (commit `e16ccc3`) built the operational-mode entry point + solver bakeoff infrastructure (Thomas vs PCR initial pass) but did NOT complete the hard acceptance gates per `PROJECT_PLAN §14.5.2`. This sprint closes the gates:

1. **Tier-4 golden 1h RMSE** on Canary d02 vs Gen2 baseline (T2/U10/V10 ≤ 5× noise floor)
2. **28-rank CPU WRF wall-clock baseline measurement** on the same 1h Canary 3km case (for comparison reference)
3. **Operational-mode full forecast-loop Nsight Systems trace** demonstrating zero H2D/D2H inside the timestep loop
4. **Speedup metric**: operational wall-clock < 28-rank CPU WRF by ≥1.2× (first-pass tripwire per Critic Amendment #6)
5. **cuSPARSE / cuSolverDx reference benchmark** for the solver bakeoff (was research-only in M6-perf-design; now ran as reference)
6. **ADR-026 promotion DRAFT → PROPOSED** with all open questions filled in

## Non-Goals

- NO modifications to validation-mode code (acoustic_wrf.py, mu_t_advance.py, tridiag_solve.py, small_step_scratch.py, acoustic_loop.py, dycore_step.py, coupled_step.py, savepoint_*.py — all LOCKED from M6B0-R/B1/B2/B3/B4/B5/B6).
- NO modifications to operational `wrf.exe`. Pre/post sha256 (`1ec3815...`).
- NO new operator semantics.
- NO sanitizer in operational path (validation can use sanitizer-on builds).
- NO bitwise WRF parity requirement (Tier-4 envelope per §14.5.1).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_perfacc` on branch `worker/gpt/m6-perf-design-acceptance`.

Write-only:
- `src/gpuwrf/runtime/operational_mode.py` (may extend; do NOT regress the 6/6 passing tests; do NOT import validation-only helpers — must compose its own operational variants per audit risk #6)
- `src/gpuwrf/runtime/cpu_wrf_baseline.py` (NEW) — orchestrator that runs the existing operational 28-rank CPU WRF on the chosen Canary d02 case and records wall-clock + `wrfout` for Tier-4
- `scripts/m6_perf_acceptance_run.py` (NEW)
- `scripts/m6_perf_solver_bakeoff_cusparse_ref.py` (NEW) — cuSPARSE/cuSolverDx reference benchmark for the solver bakeoff (read-only addition)
- `tests/test_m6_perf_acceptance.py` (NEW)
- `.agent/decisions/ADR-026-operational-mode-design-DRAFT.md` → finalize → `ADR-026-operational-mode-design-PROPOSED.md`
- `.agent/sprints/2026-05-25-m6-perf-design-acceptance/` — proofs + worker-report

Read-only everywhere else.

## Inputs (mandatory)

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6-perf-design/worker-report.md` (what was already built; gaps to close)
3. `.agent/sprints/2026-05-25-m6b6-coupled-step-parity/worker-report.md` (parity baseline; locked)
4. `.agent/sprints/2026-05-25-m6b-ladder-final-audit/audit_memo.md` (the 8 enumerated risks; do NOT trip them)
5. `PROJECT_PLAN.md §14.5 + §14.5.1 + §14.5.2`
6. `.agent/decisions/ADR-001-backend-selection.md` and `ADR-007-precision-policy.md`
7. `src/gpuwrf/runtime/operational_mode.py` (current state)
8. `data/fixtures/gen2_baseline/rmse_summary.csv` (Gen2 noise floor anchors)
9. Canairy operational WRF: `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` (CPU baseline binary; **READ-ONLY**)
10. Gen2 d02 fixtures: `/mnt/data/canairy_meteo/runs/wrf_l3/` (pick 3 pinned run-IDs for the 1h comparison)

## Acceptance Criteria (BINDING; all must PASS)

### Stage 1 — CPU WRF wall-clock baseline (MANDATORY)

`cpu_wrf_baseline.py`:
- Pick 1 Canary d02 case from Gen2 (pinned run-ID, e.g., `20260521_18z_l3_24h_20260522T072630Z`)
- Run 28-rank CPU WRF for 1h on this case
- Record wall-clock + output `wrfout_d02_1h_cpu_reference.nc`
- 28 cores reserved for this (DO NOT use Claude cores 0-3 for the run; use cores 4-31 per CPU budget memory)
- This is the **reference** for the speedup metric

Capture proof: `proof_cpu_wrf_baseline_walltime.txt` + `proof_cpu_wrf_baseline_run.log`.

### Stage 2 — Operational-mode 1h Canary run (MANDATORY)

Run `run_forecast_operational(state=canary_d02_ic, namelist=canary_d02_namelist, hours=1)` on the SAME Canary d02 case as Stage 1.
- Cores 0-3 (taskset) + GPU
- Record wall-clock + output `wrfout_d02_1h_jax_operational.nc`
- Sanitizer OFF
- Verify no nonfinite, theta bounded

Capture proof: `proof_operational_run.json` (per-step finiteness + bounds) + `proof_operational_walltime.txt`.

### Stage 3 — Nsight Systems full-loop trace (MANDATORY)

`nsys profile` of the operational 1h run. Filter for `cudaMemcpyHtoD` and `cudaMemcpyDtoH` **inside** the dycore region (between the timestep loop entry and exit).
- Expected: **ZERO** transfers inside the loop
- If non-zero: name the offending call; either lift out of loop or escalate

Capture: `proof_nsys_full_loop.nsys-rep` + `proof_nsys_transfers_inside_loop.txt` (grep summary).

### Stage 4 — Tier-4 RMSE envelope (MANDATORY)

Compare `wrfout_d02_1h_jax_operational.nc` against `wrfout_d02_1h_cpu_reference.nc` (Stage 1):
- Spatial-mean RMSE on T2 (≤ 3 K), U10 (≤ 7.5 m/s), V10 (≤ 7.5 m/s)
- Per-grid-point max abs delta (informational)
- Spatial-divergence audit (no single boundary/terrain artifact)

Capture: `proof_tier4_rmse.json` + `proof_tier4_spatial.json`.

### Stage 5 — Speedup gate (BINDING per §14.5.2)

Compute `speedup = wall_clock_cpu_28rank / wall_clock_jax_operational`. Must be ≥ 1.2× (first-pass tripwire).

If FAIL: identify dominant hotspot from Nsight; propose targeted optimization for one more perf-design sprint. **If second pass also fails, fire §14.5.2 architectural-reopening gate.**

Capture: `proof_speedup.json` + `proof_dominant_hotspot.txt`.

### Stage 6 — cuSPARSE/cuSolverDx solver bakeoff reference (MANDATORY)

Extend the existing solver bakeoff with cuSPARSE `gtsv` (and cuSolverDx if available) as reference benchmarks (not deployment candidates). For each:
- d02-scale (10500 columns × n=44) wall-clock per call
- Residual vs validated Thomas
- Memory traffic

Compare against Stage 6 bakeoff outputs from M6-perf-design (Thomas, PCR, hybrid). Updated `proof_solver_bakeoff_v2.json`.

### Stage 7 — ADR-026 promotion DRAFT → PROPOSED (MANDATORY)

Fill in ADR-026 open questions with measured evidence:
- Per-operator carry/precision/fusion/solver decisions table (with Stage 4 Tier-4 evidence + Stage 6 solver bakeoff evidence)
- Layout invariant + peak memory + 1km projected headroom (Critic Amendment #3)
- Compiled-region map with HLO/Nsight launch evidence (Critic Amendment #2)
- Precision authorization table (Critic Amendment #4)
- Measured path to M7 8-10× target (Critic Amendment #6) — hotspots, launch count, memory traffic, precision plan
- Operational-compatibility classification per Amendment #1 (every new field defaulting to validation-only or operational-approved-with-evidence)

Rename file: `ADR-026-operational-mode-design-DRAFT.md` → `ADR-026-operational-mode-design-PROPOSED.md`.

### Stage 8 — No regression

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py -v
```

All PASS. Validation-mode tests must NOT regress.

### Stage 9 — Worker report

`worker-report.md`: stages, Tier-4 RMSE table, speedup metric, Nsight summary, solver bakeoff comparison, ADR-026 status, files changed, **M6b dispatch recommendation** (`READY-FOR-M6b` if all gates pass).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_perfacc
# Stage 1 — CPU WRF baseline (DO NOT pin Claude cores 0-3; use cores 4-31 per CPU memory)
taskset -c 4-31 mpirun -np 28 /home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe 2>&1 | tee .agent/sprints/2026-05-25-m6-perf-design-acceptance/proof_cpu_wrf_baseline_run.log
# Stage 2 + 3 + 4 — operational mode + Nsight + Tier-4
taskset -c 0-3 nsys profile --output proof_nsys_full_loop python scripts/m6_perf_acceptance_run.py
# Stage 6 — cuSPARSE reference
taskset -c 0-3 python scripts/m6_perf_solver_bakeoff_cusparse_ref.py
# Stage 8 — no regression
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py -v
```

## Performance Metrics (BINDING)

- Speedup ≥ 1.2× (vs 28-rank CPU WRF)
- Zero H2D/D2H in timestep loop (Nsight verified)
- Tier-4 RMSE inside envelope (T2 ≤ 3 K, U10/V10 ≤ 7.5 m/s)

## Kill Gates

- Tier-4 RMSE outside envelope → route to operator-fix sprint
- Wall-clock loses to CPU WRF → one more perf-design sprint; if still slower, fire §14.5.2 architectural re-open
- Non-zero H2D/D2H in timestep loop → constitutional violation, must be lifted
- Imports validation-only helpers (acoustic_loop.py / dycore_step.py / coupled_step.py) into operational_mode.py → REJECT per audit risk #6
- Validation-mode regression → REJECT
- Operational sha256 changes → STOP

## Risks

- 28-rank CPU WRF run takes ~30-60 min wall-time for 1h Canary 3km; budget accordingly. Use 28-core CPU budget per memory rule (NOT cores 0-3).
- Nsight profile overhead may inflate operational wall-clock by 5-15%; capture both profiled and unprofiled times.
- cuSPARSE/cuSolverDx may not be installed; if not, document and run Thomas-only with HLO cost estimate vs PCR.

## Handoff Requirements

When all gates PASS + ADR-026 PROPOSED + worker-report committed: `/exit`. Manager dispatches **M6b honest 1h Canary** (already pre-drafted at `.agent/sprints/2026-05-25-m6b-honest-1h-canary/sprint-contract.md`).

## Failure modes the manager will reject

- Speedup claim without Nsight trace evidence
- Tier-4 claim without per-field RMSE table
- Skipping cuSPARSE reference benchmark
- ADR-026 PROPOSED without measured path to 8-10× M7 target
- Importing validation-only helpers into operational_mode.py
- Modifying validation-mode code or operational `wrf.exe`
