# Sprint Contract — M6b Operational Theta Path Fix

## Objective

Two prior fix sprints (acoustic-V, coftz-theta) landed but the operational replay still busts:
- V partial fix: V at step 46 = 11.5 m/s (was 103.7). V parity OK.
- coftz fix: landed in `vertical_implicit_solver.py` but **operational doesn't use coftz** — operational uses `dynamics/core/acoustic.py` → `advance_mu_t_wrf` (in `src/gpuwrf/dynamics/mu_t_advance.py`).

So the operational theta drift root cause is NOT coftz; it is in `advance_mu_t_wrf` OR the `dynamics/core/acoustic.py` composition of it.

Symptoms (pre-existing, post both prior fixes):
- 1h Canary on 20260521 with V fix: step 49 theta_upper_14_max = 1343.5 K (bound 700K) → FAIL.
- 1h Canary on 20260509: step 11 theta = 2.6×10¹² K → FAIL.
- Real-IC multi-step CPU parity: step 1 = 0.0 bitwise; step 2+ theta/t_2ave/ph_tend = 1e+300 nonfinite (V is 0.0 bitwise).
- GPU/CPU agree exactly within the validation path (it's not a GPU-only bug).

The coftz worker correctly diagnosed hypothesis H2 (`theta_face` source = **instantaneous theta** instead of WRF's `theta_1` running-average / theta_ave). Apply the same diagnosis to the operational path.

## Non-Goals

- NO modification to dynamics/core API surface that would break B6 ladder.
- NO retuning of bounds.
- NO touching V advection — that's already in workaround state.
- NO remote push.

## File Ownership

Worktree **already created** at `/tmp/wrf_gpu2_optheta` on branch `worker/gpt/m6b-operational-theta-fix`.
Your FIRST command: `cd /tmp/wrf_gpu2_optheta`.

Write-only (EXPANDED ownership):
- `src/gpuwrf/dynamics/mu_t_advance.py` (PRIMARY — `advance_mu_t_wrf` theta tendency uses `theta_1` for flux but updates with `theta_tend`; check whether `theta_1` is fresh or stale).
- `src/gpuwrf/dynamics/core/acoustic.py` (acoustic_substep_core — what theta does it feed forward between substeps?).
- `src/gpuwrf/dynamics/small_step_scratch.py` (build_scratch_state — what t_2ave is fed back into the next substep?).
- `src/gpuwrf/runtime/operational_mode.py` (if the bug is in how operational composes the substeps).
- `tests/test_m6b_operational_theta_fix.py` (NEW).
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/` — proofs + worker-report.md.

Read-only:
- Everything else.

## Inputs

1. This sprint contract.
2. `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/worker-report.md` — the coftz worker's diagnosis (H2 was right; fix landed in wrong file).
3. `.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/` — original V3-509 evidence.
4. `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/` — V partial fix.

## Hypothesis Space

1. **`advance_mu_t_wrf` theta-flux uses theta_1 stale**: At line 133 `face_theta = fnm * theta_1[k] + fnp * theta_1[k-1]`. If `theta_1` is the same array as `theta` (mutable) instead of a snapshot of theta at start of RK stage, after the first substep `theta_1` already drifted.
2. **`acoustic_substep_core` does not preserve `theta_1` between substeps**: Look at acoustic_substep_core (acoustic.py:185-233). It calls `advance_mu_t_core` with `state` — but does the input contain a stable `theta_1` or only `theta`?
3. **`build_scratch_state` feeds wrong `t_2ave` forward**: `t_2ave` is supposed to be a running average. If we feed `theta_new` instead of `0.5*(theta_new + theta_old)`, theta diverges.
4. **`_ph_tend_increment` formula has a unit error or sign error**: Currently `0.01 * theta_delta` — magic number; if WRF uses something different the ph_tend drifts.
5. **The `dts` substep time is wrong**: If `dts` is `dt` (full timestep) instead of `dt/n_substeps`, theta over-integrates by n_substeps×.

Document each in `hypothesis_notes.md`.

## Acceptance Criteria

### Stage 1 — Reproduce

Run `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 2` (after `cd /tmp/wrf_gpu2_optheta`). Confirm step 2 theta/t_2ave/ph_tend = 1e+300 nonfinite on baseline.

Write `proof_baseline_reproduces.json`.

### Stage 2 — Fix and validate

ALL of:
- Multi-step CPU parity 2/5/10 = 0.0 bitwise (`scripts/m6b_real_ic_operational_compare.py --steps 2` etc.). **No more nonfinite step 2.**
- B6 savepoint parity = 0.0 bitwise (`scripts/m6b6_coupled_step_compare.py --tier all`).
- 1h Canary on 20260509 + 20260521: theta in [200,700]K all 360 steps for all levels.
- `pytest -x` passes (modulo the pre-existing missing-fixture failure on `test_canary_wrf_fixture.py::test_full_external_file_exists_at_external_uri` — that's unrelated).

Write `proof_fix_validation.json`.

### Stage 3 — Regression test

`tests/test_m6b_operational_theta_fix.py`: load bad cell + run one operational acoustic substep + assert step 2 theta is finite and within [200, 700]K.

### Stage 4 — Worker report

`worker-report.md` with `Summary:`, what hypothesis matched, what was actually wrong, proofs, risks, handoff. >=400 bytes.

If you cannot satisfy all Stage 2 criteria with a small focused fix, write `Summary: BLOCKED — <reason>` and document the nearest-fit hypothesis + which line you'd recommend manager investigate.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_optheta
export OMP_NUM_THREADS=4
export PYTHONPATH="src"

# Baseline
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 2 > .agent/sprints/2026-05-26-m6b-operational-theta-fix/baseline_step2.txt 2>&1

# (Apply fix)

# Validation
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 2 > .agent/sprints/2026-05-26-m6b-operational-theta-fix/fixed_step2.txt 2>&1
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 5 > .agent/sprints/2026-05-26-m6b-operational-theta-fix/fixed_step5.txt 2>&1
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10 > .agent/sprints/2026-05-26-m6b-operational-theta-fix/fixed_step10.txt 2>&1
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all
taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .agent/sprints/2026-05-26-m6b-operational-theta-fix/v3_521/
taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z --output .agent/sprints/2026-05-26-m6b-operational-theta-fix/v3_509/
taskset -c 0-3 pytest tests/test_m6b_operational_theta_fix.py -v

git add -A && git commit -m "[operational theta fix] $(date -u +%FT%TZ)"
```

## Handoff

Per the universal worker spec.
