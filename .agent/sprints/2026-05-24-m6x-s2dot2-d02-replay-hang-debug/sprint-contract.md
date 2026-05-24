# Sprint Contract — M6.x S2.2: d02 Replay Hang Debug

## Objective

`scripts/m6_d02_boundary_replay_1h.py` hangs even at `--duration-s 1` — produces ZERO stdout/stderr in 1810 seconds (S2.1 worker measured). This is a fundamental infrastructure blocker. The replay needs to actually run for M6 close to be reachable.

This sprint's job: **find the hang root cause and fix it.** Could be:
- A missing dependency or import that triggers a circular wait
- A JAX compilation hang (infinite trace, missing concrete shape)
- A GPU context that fails to initialize
- A data loading hang on `/mnt/data/canairy_meteo/runs/` (file permissions? missing files? large I/O blocking?)
- A subprocess deadlock
- A `lax.scan` whose static_argnums isn't actually static at trace time

Strategy: instrument FIRST (add `print` statements + flush + timing), narrow with progressive `--duration-s 0` (no forecast, just startup) → `--duration-s 0.25` (one substep) → `--duration-s 1` (one RK3 step). Find the line where it hangs. Then fix.

## Non-Goals

- No changes to `src/gpuwrf/dynamics/`. Read-only on the operator.
- No new sidecars.
- No operator-fix attempts.
- No 1h or longer forecast in this sprint (we can't even get 1 second to work).
- No remote push.
- No spec-gaming: a "fix" that returns synthetic data is rejected.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_s2dot2_debug` on branch `worker/gpt/m6x-s2dot2-d02-replay-hang-debug`.

Write-only:
- `scripts/m6_d02_boundary_replay_1h.py` — instrument liberally with `print(..., flush=True)` + timing. Fix the hang.
- `scripts/m6_d02_baseline_run_instrumented.py` — adjust the probe orchestrator if needed (e.g., disable the probe-then-real two-pass; just run directly with timeout).
- `src/gpuwrf/integration/d02_replay.py` — instrument liberally. May fix small bugs (e.g., missing print flush, missing JAX block_until_ready, wrong static_argnums) but NO operator changes.
- `tests/test_m6x_d02_replay_hang_debug.py` (new) — a fast smoke test that calls the replay engine with synthetic minimal IC and asserts it returns within 60 seconds
- `.agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/` — proofs + worker-report

Read-only everywhere else, including all `src/gpuwrf/dynamics/`.

## Inputs

Required reading:
- **`.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/worker-report.md`** — what S2.1 found (1810s probe with zero output)
- **`.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/worker-report.md`** — what S2 found
- `scripts/m6_d02_boundary_replay_1h.py` — the script that hangs
- `scripts/m6_d02_baseline_run_instrumented.py` — the orchestrator (its probe-then-real two-pass might be the problem)
- `src/gpuwrf/integration/d02_replay.py` — the replay engine; has IC loading + RK3 stages + acoustic scans + physics + boundary lateral application
- `scripts/m6_warm_bubble_test.py` — known-working harness for comparison (this DOES run); diff the patterns
- `.agent/decisions/ADR-016-gen2-data-corpus.md` — Gen2 data access pattern

## Acceptance Criteria

### 1. Root cause identified

`worker-report.md` documents the specific hang location:
- File + line range where execution stops
- The kind of hang (compile loop, infinite trace, blocking I/O, subprocess wait, GPU init)
- Why it hangs (causal explanation, ideally with a minimal repro snippet)

### 2. Fix applied

`scripts/m6_d02_boundary_replay_1h.py --duration-s 1` runs to completion within 5 minutes (300s) on a clean checkout. Capture wall time in `proof_replay_runs.txt`.

### 3. Progressive duration test

Run with `--duration-s` ∈ {0, 0.25, 1, 60, 300}. Each must complete within reasonable budget (≤ 5min for short durations, ≤ 30min for 300s). Capture results in `proof_progressive.txt`.

### 4. Smoke test

`pytest tests/test_m6x_d02_replay_hang_debug.py -v` PASSES with the smoke test calling the engine with synthetic minimal IC and asserting return within 60s.

### 5. No regression

`pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_d02_boundary_replay.py tests/test_m3_transfer_audit.py -v` — all PASS (45+ tests).

### 6. Worker report

`worker-report.md` with: hang root cause + fix description + before/after timing + files changed + commands + proof objects + risks + handoff.

### 7. Branch commits on `worker/gpt/m6x-s2dot2-d02-replay-hang-debug`. Multiple commits OK — encourage WIP commits to preserve the investigation trail.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_s2dot2_debug
# Progressive duration test:
for D in 0 0.25 1 60 300; do
  echo "=== duration=$D ==="
  timeout 1800 python scripts/m6_d02_boundary_replay_1h.py --duration-s "$D" \
    --output .agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_run_d${D}.json \
    2>&1 | tee -a .agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_progressive.txt
done
# Smoke test:
pytest tests/test_m6x_d02_replay_hang_debug.py -v | tee .agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_smoke.txt
# No-regression:
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_d02_boundary_replay.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_no_regression.txt
```

## Performance Metrics

- Wall time at each duration: report
- Pre-fix: ∞ (hangs); post-fix: ≤ 300s for d=1, ≤ 30min for d=300

## Proof Object

- `proof_progressive.txt` — duration sweep results
- `proof_run_d*.json` — per-duration JSON output
- `proof_smoke.txt` — smoke test passes
- `proof_no_regression.txt`
- `worker-report.md` with root cause + fix description
- Branch `worker/gpt/m6x-s2dot2-d02-replay-hang-debug`

Time budget: **4-8 hours**. Could be quick if it's a missing flush; could be longer if JAX retrace is the issue.

## Risks

- **Hang root cause may be in src/gpuwrf/**: this contract says no operator code changes, but small fixes to `src/gpuwrf/integration/d02_replay.py` are allowed (it's the integration layer, not the dycore). Hard rule: NO changes to `src/gpuwrf/dynamics/`, `src/gpuwrf/contracts/`, `src/gpuwrf/physics/`.
- **Hang may be in JAX itself**: if JAX retraces forever on shape-polymorphic input, the fix is `jax.jit(..., static_argnums=(...))` correction. Document.
- **Data access may be the cause**: if `/mnt/data/canairy_meteo/runs/` is slow/missing, document and propose a synthetic-IC fallback as a deliberate path (not a covered-up substitute).
- **Spec-gaming**: a fix that silently returns synthetic data without running real replay = rejected.

## Handoff Requirements

When all proof files on disk, progressive duration test passes, smoke + no-regression green, worker-report committed: `/exit`. Wrapper fires AGENT REPORT to manager pane.

## Failure modes the manager will reject

- Modifying any file under `src/gpuwrf/dynamics/`, `src/gpuwrf/contracts/`, or `src/gpuwrf/physics/`.
- Returning synthetic data while claiming "real" output.
- Smoke test that doesn't actually call the replay engine.
- Skipping no-regression.
- Hang root cause described as "unclear" — every hang has a specific line/cause; find it.
