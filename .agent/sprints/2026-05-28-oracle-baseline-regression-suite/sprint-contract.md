# Sprint Contract — Oracle-Baseline Regression Suite (per principal directive 2026-05-28)

**Sprint ID**: `2026-05-28-oracle-baseline-regression-suite`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/oracle-baseline-regression`
**Worktree**: `/tmp/wrf_gpu2_oracle`
**Wall-time**: 4-12 h (target ≤ 1 day)
**GPU usage**: YES (for GPU side of comparison + smoke run)
**Sandbox**: `--sandbox danger-full-access`
**Complementary to**: the Opus instrumentation harness (running in parallel as background subagent; its output goes to `src/gpuwrf/diagnostics/comprehensive_harness.py`). USE its hooks if available; do not duplicate the instrumentation work.

## Why this sprint (principal motivation)

Principal directive 2026-05-28: "Unser aktueller Test (S2 real / Kanaren) ist zwar wichtig, aber bei Fehlern viel zu unspezifisch, was ewiges Debugging zur Folge hat. Die Idee: Einen Test (oder eine Test-Suite) entwickeln, den wir erst durch die CPU-basierte WRF-Version als Baseline (Oracle) durchjagen, und danach nach jedem Milestone durch die GPU-Version. Der Test soll messerscharf und exakt zeigen, ob die neu hinzugefügte Kopplung/Dynamik/Physik perfekt funktioniert und nichts anderes zerstört hat."

Translation: the current end-of-forecast skill test is too non-specific. We need an Oracle-baseline regression suite where CPU WRF is the truth-source. After each milestone, the GPU version runs through the suite and gets razor-sharp PASS/FAIL on whether the newly-added coupling works AND whether anything else broke.

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 statistically equivalent to CPU WRF v4 under TOST at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Objective

Build a multi-case Oracle-baseline regression suite with three layers:

1. **Oracle generation** (CPU WRF runs, possibly re-using existing ones)
2. **GPU regression harness** that runs the same cases and compares to Oracle field-by-field, time-by-time
3. **Milestone snapshot system** that lets the manager run `bash scripts/run_milestone_regression.sh M11` (or any milestone) and get a JSON saying "this milestone passes / fails / improves / regresses relative to the previous milestone."

## Required inputs (read in order)

1. `.agent/decisions/PROJECT-RESET-PLAN-FINAL.md` — binding goal + invariant ladder INV-1..11
2. `.agent/sprints/2026-05-28-diagnostic-harness/design.md` AND `src/gpuwrf/diagnostics/comprehensive_harness.py` — Opus agent deliverables (may not exist yet; check; if absent, design around the existing `scripts/operational_trace_compare.py`)
3. `scripts/operational_trace_compare.py` — hour-by-hour comparator (M9.A/M9.C)
4. `scripts/m6b6_coupled_step_compare.py` — dycore-only savepoint comparator (M6)
5. `tests/savepoint/` — current scaffold incl. PLACEHOLDER files
6. `proofs/m10/static_field_parity_after_fix.json` — example of bitwise-parity reporting schema
7. `proofs/m9/divergence_map_v2.json` — example of operator-level diagnosis schema
8. `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z/` — existing CPU WRF Oracle for Canary 20260521 24h
9. `/mnt/data/canairy_meteo/runs/wrf_l2/` — any L2 Oracle runs

## Acceptance

### AC1 — Oracle test-case manifest

`tests/regression/oracle_cases.yaml` lists every test case used in the suite. For each: case_id, type (real | idealized), CPU WRF run directory, expected hours, expected variables, tolerance class (BITWISE | EQUIVALENCE_TIGHT | EQUIVALENCE_LOOSE). Required cases:

| case_id | type | source | hours | tolerance class |
|---|---|---|---|---|
| canary_20260521_24h_d02 | real | existing CPU WRF run | 24 | EQUIVALENCE_TIGHT |
| canary_20260521_72h_d02 | real | run CPU WRF if not yet on disk (mark BLOCKED if not feasible) | 72 | EQUIVALENCE_LOOSE |
| idealized_warm_bubble | idealized | generate via WRF idealized test or analytic | short | BITWISE |
| idealized_density_current | idealized | generate via WRF idealized | short | BITWISE |
| canary_20260521_24h_d03 | real | existing CPU WRF run if available | 24 | EQUIVALENCE_TIGHT |

If any case requires a CPU WRF run that's not feasible in sprint scope, mark it BLOCKED in the manifest with the exact command needed to generate it. Do NOT block this sprint on Oracle generation for the harder cases.

### AC2 — Tolerance specification

`tests/regression/tolerances.yaml` defines per-field per-class tolerances. For example:
- BITWISE: max_abs_diff = 0, rmse = 0
- EQUIVALENCE_TIGHT: T2 RMSE ≤ 1.0K, U10 RMSE ≤ 0.5 m/s, theta_3D RMSE ≤ 3K, etc.
- EQUIVALENCE_LOOSE: 2× the TIGHT thresholds

Tolerances are not "what we currently achieve" — they are "what passing the TOST gate requires of each component test." The suite is allowed to FAIL today; the point is to give us a fixed target.

### AC3 — Regression-suite driver

`scripts/run_regression_suite.py`:
- Reads `oracle_cases.yaml` + `tolerances.yaml`
- For each case: runs GPU forecast via existing pipeline, captures GPU output, compares to Oracle per `tolerances.yaml`
- Writes per-case JSON to `proofs/regression/<case_id>_<milestone>.json` with full per-field per-output-time diff
- Writes aggregate `proofs/regression/aggregate_<milestone>.json` with the suite-level PASS/FAIL summary
- Optional `--milestone-snapshot M11` argument that ALSO emits `proofs/regression/snapshot_M11.json` (the durable artifact)

### AC4 — Milestone snapshot + regression check

`scripts/run_milestone_regression.sh M11`:
- Calls `scripts/run_regression_suite.py --milestone-snapshot M11`
- Compares snapshot_M11.json against snapshot_M10.json (or previous milestone snapshot)
- Emits `proofs/regression/regression_check_M10_to_M11.json` with: tests that newly PASS, tests that newly FAIL (= regression!), tests that improved within tolerance, tests that worsened within tolerance
- Exit code 0 if "no newly-failing tests AND aggregate pass-count did not decrease"; nonzero otherwise

### AC5 — CI integration

`tests/regression/test_regression_suite.py` — a pytest entry that runs `scripts/run_regression_suite.py --smoke` (a fast subset, e.g. canary_20260521 first 1h only) and asserts no schema error + at least N tests executed.

The full suite runs out-of-band (manual + post-milestone), not in every pytest run, but the smoke is part of `pytest tests/savepoint/ tests/regression/`.

### AC6 — Documentation

`tests/regression/README.md`:
- Explains the Oracle-baseline philosophy
- How to add a new case
- How to add a new tolerance class
- How a milestone reviewer uses the snapshot for cross-milestone regression check
- Example output excerpt

### AC7 — Smoke run

Execute `scripts/run_regression_suite.py --smoke` on the manager environment. Capture the output as `.agent/sprints/2026-05-28-oracle-baseline-regression-suite/smoke_output.json`. Document any setup steps needed.

### AC8 — Worker report

`.agent/sprints/2026-05-28-oracle-baseline-regression-suite/worker-report.md`:
- Standard format
- Verdict: `ORACLE_SUITE_COMPLETE` if AC1-AC7 all delivered; `ORACLE_SUITE_PARTIAL` with explicit gaps otherwise
- Headline: "<N> tests passing baseline today; <M> failing within tolerance; <K> blocked-pending-Oracle"

## Hard rules

1. **CPU pinning**: `taskset -c 0-3` on every command.
2. **GPU usage**: ALLOWED for the smoke (single 1h Canary run) under `--sandbox danger-full-access`.
3. **Files writable**: 
   - `tests/regression/**` (NEW directory)
   - `scripts/run_regression_suite.py`, `scripts/run_milestone_regression.sh` (NEW)
   - `proofs/regression/**` (NEW directory)
   - `.agent/sprints/2026-05-28-oracle-baseline-regression-suite/**`
4. **Files NOT writable**:
   - `src/gpuwrf/**` (no model code changes; this is test infrastructure)
   - `src/gpuwrf/diagnostics/comprehensive_harness.py` (Opus agent owns this; if Opus delivered it, READ + USE it; do not modify)
   - any other sprint's deliverables
   - governance files
5. **Coordination with parallel workers**: M11, M12, M13 are modifying `runtime/operational_mode.py`, `coupling/physics_couplers.py`, `io/wrfout_writer.py`. AVOID those files entirely.
6. **Manager repo ONLY** — do NOT touch `/home/enric/src/wrf_gpu/`.
7. **No remote push.**
8. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: oracle-baseline DONE exit=$?" Enter`.
9. **End with verdict**: `ORACLE_SUITE_COMPLETE` / `ORACLE_SUITE_PARTIAL` + one-line headline of suite size + current pass rate.

## Why this is razor-sharp

Today: a skill regression on the 24h Canary case tells us "something's wrong" with no localisation.
After this suite: a milestone snapshot tells us:
- "test `canary_20260521_d02 / theta_3D` regressed from 78K RMSE to 95K RMSE between M10 and M11 — this is a NEW failure."
- "test `idealized_warm_bubble / max_w` newly PASSES — M11 fix worked."
- "test `canary_20260521_d02 / HFX magnitude` was failing at M10 (mean 924) and still fails at M11 (mean 880) — improvement within tolerance but not yet passing."

The manager sees this and immediately knows what M12 needs to address vs what's safe to assume working.
