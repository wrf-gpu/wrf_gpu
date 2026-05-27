# Sprint Contract — Testing-Plan Execution RE-DO (Codex GPT-5.5)

**Sprint ID**: `2026-05-27-testing-plan-execution-redo`
**Created**: 2026-05-27 (post-crash re-execution; sprint #3 partial committed at `21ea2fc`/`36144bc`)
**Status**: READY
**Predecessor**:
- `.agent/sprints/2026-05-27-testing-plan-execution/` (EXECUTION_PARTIAL — all infrastructure landed, all GPU tests BLOCKED because `nvidia-smi` exit=124 at sprint time)
- `.agent/sprints/2026-05-27-testing-plan-critique/test_plan_revised.md` (the 13-test revised plan)

## Context (post-crash recovery)

The previous sprint #3 dispatch produced **all 15 pubtest_*.py orchestrators, 4 idealized IC builders, 2 unit-test files, and 18 proof JSON stubs**, but every heavy GPU test was honestly marked `BLOCKED` because the GPU was unreachable (`nvidia-smi` rc=124) at the time. The machine then crashed. Post-reboot, `nvidia-smi` responds in <1 s with healthy state (1.7 GB / 32.6 GB used at idle).

This sprint **re-executes the existing scripts against a healthy GPU** to replace the BLOCKED placeholder proof objects with real PASS/FAIL evidence.

## Objective

Re-run the HIGH-priority items from `test_plan_revised.md` using the now-existing pubtest infrastructure. Produce real proof objects with honest PASS/FAIL verdicts. The infrastructure exists; the GPU is healthy; this should be straight execution.

If a test really does fail at its threshold (e.g. the T2 skill regression on CANARY-MULTIDAY), record the failure plainly. The user wants honest evidence, not green theater.

## Scope — HIGH-priority items only (10 items, ≤17.4 GPU-hour budget)

The full spec is in `.agent/sprints/2026-05-27-testing-plan-critique/test_plan_revised.md`. The 10 items are:

1. IDEALIZED-WARMBUBBLE (Bryan & Fritsch 2002 reference)
2. IDEALIZED-DENSITY-CURRENT (Straka 1993)
3. IDEALIZED-MOUNTAIN-WAVE (Schaer 2002 primary; em_hill2d_x smoke)
4. CONSERVATION-MASS-24H (closed-domain ≤1e-10; Canary flux-corrected ≤1e-5)
5. CONSERVATION-ENERGY-24H (CPU envelope ±20%)
6. STABILITY-CFL-SWEEP
7. STABILITY-ACOUSTIC-SUBSTEP-SWEEP
8. DETERMINISM-REPEAT (with JAX_DETERMINISTIC_OPS if needed)
9. SAVEPOINT-PARITY-DEEP
10. CANARY-MULTIDAY-SIDE-BY-SIDE (≥14 days; per-variable thresholds — T2 expected to FAIL ±20%)

Reuse the existing `scripts/pubtest_*.py` orchestrators. The previous sprint scaffolded them but every GPU branch returned BLOCKED. **This sprint actually runs the GPU branch.**

Opportunistic MEDIUM-priority bonuses if budget allows:
- COMPILE-COLD-START-TIME (cheap)
- VRAM-FOOTPRINT-1KM-FRESH (cheap)

## Acceptance

- **AC1-AC10** per-test proof object at `.agent/sprints/2026-05-27-testing-plan-execution-redo/<test_id_lowercase>.json` with REAL verdicts (PASS / FAIL / SKIP_<reason>). No BLOCKED stubs.

- **AC11 — Real GPU usage**: total GPU-hours used recorded in `aggregate_report.md` (must be >0; the previous sprint's `0.0` is the trigger for re-dispatch).

- **AC12 — Invariant preservation**: 20260521 multi-step step-2 bitwise 0.0, B6 savepoint parity, restart bitwise, D2H/H2D = 0. These already-merged guardrails must still hold. Confirm with the existing pytest suite + the existing proof JSONs.

- **AC13 — Aggregate report**: `aggregate_report.md` + `.json` with: per-test verdict, GPU hours used per test, total hours, what passed cleanly, what failed and why, what the paper can now claim.

- **AC14 — Publish/scripts/ staging**: copy the HIGH-priority `scripts/pubtest_*.py` orchestrators that produced real evidence into `publish/scripts/` (NEW), with a `publish/scripts/README.md` listing each script's purpose, proof object path, and how to re-run. The user asked for the test scripts to live in the publish folder; this AC delivers that.

- **AC15 — Worker report** with verdict `EXECUTION_GREEN` (all 10 PASS or honest FAIL with threshold breach documented) / `EXECUTION_PARTIAL` (subset done; list what's left) / `EXECUTION_BLOCKED` (something fundamental still wrong; explain).

## Files Worker May Modify

- `.agent/sprints/2026-05-27-testing-plan-execution-redo/**`
- `publish/scripts/**` (NEW for AC14)
- `scripts/pubtest_*.py` — small targeted fixes only if a script bug blocks execution; document each edit in `worker-report.md`
- `tests/test_pubtest_*.py` — only if existing tests fail and need to align with the GPU path that worked

## Files Worker Must Not Modify

- `src/gpuwrf/runtime/operational_mode.py`, `coupling/physics_couplers.py`, `dynamics/**`, `contracts/**`, `validation/**` (model core frozen)
- `src/gpuwrf/fixtures/idealized_cases/**` (already in main, frozen)
- `publication/**` (paper untouched until sprint #5)
- `publish/paper/**`, `publish/tables/**`, `publish/figures/**`, `publish/manifest/**`, `publish/LICENSE_RECOMMENDATION.md`, `publish/README.md` — only `publish/scripts/` (NEW) is in scope
- governance files

## Hard Rules

1. **Real GPU execution required.** AC11 mandates >0 total GPU hours. A 0.0-hour BLOCKED re-run is a regression and must be flagged immediately.
2. **Honest verdicts.** Record threshold breaches plainly. T2 regression on CANARY-MULTIDAY is expected to FAIL ±20% — don't hide it.
3. **17.4 GPU-hour HIGH budget + 6.6 h reserve = 24 h cap.**
4. **Invariant preservation** — AC12 is a hard gate. If a test would break B6/restart/D2H/multi-step-bitwise, abort.
5. **CPU pinning**: `taskset -c 0-3` for orchestrators.
6. **Don't interfere with `~/src/canairy_meteo/Gen2/` nightly scheduler.**
7. **No remote push.** Local commit on `worker/gpt/testing-plan-execution-redo` only.
8. **Commit early, commit often.** If the machine crashes again, partial commits should preserve work. Aim to commit after each test (per-test JSON + script edits) rather than only at sprint end.

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 8-24 h
- Branch: `worker/gpt/testing-plan-execution-redo`
- Worktree: `/tmp/wrf_gpu2_testexec2`
- GPU usage: YES — ~15-17 GPU-hours expected.

## Why this matters

Sprint #4 (opus check) reads these proof objects and writes the publication-readiness verdict. Without real PASS/FAIL evidence here, sprint #5 (paper rewrite) cannot claim community-grade testing. The whole purpose of the user's pivot was to produce this evidence, so this sprint is the critical-path item to make the paper claim defensible.
