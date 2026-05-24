# Sprint Contract — M6.x S4-prep: Tier-3 Convergence Infrastructure

## Objective

S4 of the HYBRID plan needs Tier-3 controlled dt-convergence on an **idealized case** (independent of d02 replay). This sprint builds the infrastructure NOW so that when S2.2 + S2.1-redo + S3-real return, S4 can run immediately. Per critic's HYBRID plan §S4 + `VALIDATION_STRATEGY.md` Tier-3.

Concretely:
1. Pick an idealized case (e.g., `em_hill2d_x` 2-D x-z hill, or a flat warm-bubble, or hydrostatic-rest) that exists or can be ported
2. Build the dt-doubling test infrastructure: base/refined dt pairs, identical IC/BC, identical physics scope, identical solver primitives
3. Define norms per variable (U/V/W/theta/p/mu), lead windows (e.g., 0-60s, 60-300s), pass/fail criteria
4. Output schema: `artifacts/m6/tier3/tsc_envelope.json` (cited by critic in §S4)
5. Smoke-run the infrastructure on the current ADR-023 unified operator with a tiny case to verify it works (the verdict may be FAIL — that's OK, the goal is to have the test READY)

## Non-Goals

- No d02 replay (S2.2 covers that).
- No operator fix (S3-real handles it).
- No new physics schemes.
- No 24h or operational forecast.
- No remote push.
- No carry expansion or magic-number cleanup (S3-narrow already did the cleanup).
- **No "promote ADR-023 to ACCEPTED" attempt** — that's S6.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_s4prep` on branch `worker/gpt/m6x-s4prep-tier3-convergence-infra`.

Write-only:
- `scripts/m6_tier3_convergence_runner.py` (new) — orchestrator that runs the dt-doubling sweep on a given idealized case + emits JSON
- `src/gpuwrf/validation/tier3_envelope.py` (new) — pure-NumPy/JAX helper for norm computation + dt-pair comparison (read-only on operator)
- `tests/test_m6x_tier3_convergence_infra.py` (new) — smoke tests that the infrastructure runs + produces JSON in the expected schema
- `data/fixtures/tier3_idealized/case_definition.json` (new — IC + BC + grid + physics-toggle definition for the chosen case)
- `.agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/` — proofs + worker-report

Read-only everywhere else, especially `src/gpuwrf/dynamics/` (no operator edits).

## Inputs

Required reading:
- **`.agent/sprints/2026-05-23-m6x-close-strategy-plan-critic/reviewer-report.md`** §S4 (Tier-3 convergence spec from critic)
- `VALIDATION_STRATEGY.md` — Tier-3 definition
- `.agent/sprints/2026-05-21-m6-milestone-plan-scout/m6-milestone-plan.md` — original M6 Tier-3 spec (Section on M6-S6)
- `.agent/sprints/2026-05-21-m6-milestone-plan-scout/critical-review-codex.md` lines 54-59, 151-153 — prior warning that config-noise is NOT timestep convergence
- WRF source for `em_hill2d_x`: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/test/em_hill2d_x/` (if exists; otherwise document)
- `src/gpuwrf/timestep/`, `src/gpuwrf/dynamics/step.py`, `src/gpuwrf/dynamics/rk3.py` — existing RK3 stepper (read-only)
- `scripts/m6_warm_bubble_test.py` — working harness pattern for idealized case execution
- `scripts/diagnostic_timestep_convergence_dashboard.py` (S1) — has the schema scaffolding to extend

## Acceptance Criteria

### 1. Idealized case definition

`data/fixtures/tier3_idealized/case_definition.json` documents:
- Case name (e.g., `flat_warm_bubble_tier3` or `em_hill2d_x_tier3`)
- Grid: nx, ny, nz, dx, dy, dz_meta
- IC: theta perturbation form, base state, hydrostatic balance
- BC: periodic / open / damped
- Physics toggles (dycore-only OR dycore+microphysics)
- Default dt + refinement levels (e.g., dt, dt/2, dt/4)
- Total integration time (e.g., 60s or 300s; short enough to run multiple times)
- Variables to track for convergence: U, V, W, theta, p_perturbation, mu_perturbation

### 2. Convergence runner

`scripts/m6_tier3_convergence_runner.py`:
- Accepts `--case`, `--dt`, `--output` CLI args
- Loads case definition
- Runs the current ADR-023 unified path at the specified dt
- Records final-state arrays + per-checkpoint state at the documented norms
- Returns JSON conforming to the schema below

### 3. tsc_envelope.json schema

`artifacts/m6/tier3/tsc_envelope.json` produced by the runner has top-level keys:
```
{
  "artifact_type": "m6_tier3_tsc_envelope",
  "case": "<case name>",
  "config": {
    "boundary_mode": "<periodic|open|damped>",
    "physics": "<dycore_only|dycore_plus_micro>",
    "total_time_s": <number>,
    "variables": ["U","V","W","theta","p_perturbation","mu_perturbation"]
  },
  "dt_pairs": [
    {"dt_coarse": <num>, "dt_fine": <num>, "pair_index": 0}
  ],
  "checkpoints_s": [<lead in seconds>],
  "per_dt_run_metadata": [...],  // wall time, kernel launches, first nonfinite step
  "norms": {  // L2, Linf, RMSE per variable per checkpoint per dt
    "U": {...}, "V": {...}, ...
  },
  "convergence_verdict": "<PASS_TIER3|FAIL_DRIFT|FAIL_NONFINITE|FAIL_INSUFFICIENT_DT_PAIRS>",
  "rationale": "<short explanation citing the norm values>"
}
```

### 4. Smoke test passes

`tests/test_m6x_tier3_convergence_infra.py` invokes the runner on a tiny case (e.g., 4×4×8 grid, 4 timesteps, dt and dt/2) and asserts:
- JSON conforms to schema
- All `norms` arrays are finite
- `convergence_verdict` is one of the four allowed strings
- Runner completes within 60s wall on the tiny case

This test does NOT assert that the operator passes Tier-3 (that's S4's actual job after fixes land). It only asserts the infrastructure runs.

### 5. Smoke-run on current operator

Run the smoke case against the **current ADR-023 unified operator** (post-S3-narrow). Capture the JSON to `proof_tier3_smoke_current_state.json` and the verdict. **Honest report**: most likely `FAIL_DRIFT` or `FAIL_NONFINITE` given the operator's known mu issue. The honest result is the proof.

### 6. No regression

```bash
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_s3narrow_stabilizer_audit.py tests/test_m3_transfer_audit.py -v
```
All PASS.

### 7. Worker report

`worker-report.md` documenting:
- Case choice + rationale (cite the inputs)
- Runner architecture (high-level Python flow)
- Schema decisions
- Smoke-run verdict on current state with cited norms
- Files changed + commands + risks + handoff

### 8. Branch commits on `worker/gpt/m6x-s4prep-tier3-convergence-infra`.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_s4prep
python scripts/m6_tier3_convergence_runner.py --case flat_warm_bubble_tier3 --dt 1.0 \
  --output .agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_tier3_smoke_current_state.json \
  | tee .agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_tier3_smoke.txt
pytest tests/test_m6x_tier3_convergence_infra.py -v | tee .agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_smoke_test.txt
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_s3narrow_stabilizer_audit.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-24-m6x-s4prep-tier3-convergence-infra/proof_no_regression.txt
```

## Performance Metrics

- Smoke case wall time: ≤ 60s
- Transfer audit: 0 H2D/D2H in timestep loop

## Proof Object

- `scripts/m6_tier3_convergence_runner.py`
- `src/gpuwrf/validation/tier3_envelope.py`
- `tests/test_m6x_tier3_convergence_infra.py`
- `data/fixtures/tier3_idealized/case_definition.json`
- `proof_tier3_smoke_current_state.json` (+ `.txt`)
- `proof_smoke_test.txt`
- `proof_no_regression.txt`
- `worker-report.md`

Time budget: **4-8 hours**.

## Risks

- **Idealized case choice**: pick one that DOESN'T overlap with the warm-bubble harness (otherwise we're just re-running the operator-sanity gate). Either use `em_hill2d_x` (different setup) or a different theta-perturbation form for the warm-bubble (e.g., constant-N stratified column with smaller perturbation).
- **dt refinement may show non-convergence** on the current operator: that's expected and informative. Do NOT modify the operator to make it converge — that's S3-real's job.
- **`em_hill2d_x` may not be ported**: if it doesn't exist in the project, define a fresh idealized case and cite WRF's `em_hill2d_x` namelist as inspiration.
- **Schema drift**: lock the JSON schema in the smoke test so future S4 runs are comparable.
- **Spec-gaming**: don't pick a case so trivial that any operator "passes". The dt-doubling test must actually test convergence behavior.

## Handoff Requirements

When all proof files on disk, smoke test passes, no-regression green, worker-report committed: `/exit`. Wrapper sends AGENT REPORT to manager pane.

## Failure modes the manager will reject

- Modifying `src/gpuwrf/dynamics/` or operator-sanity test.
- Trivial case that obviously passes (e.g., pure rest state).
- Schema deviation from the documented spec.
- Skipping the smoke-run on current state.
