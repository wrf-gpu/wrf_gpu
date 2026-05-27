# Sprint Contract — Testing-Plan Execution (Codex GPT-5.5)

**Sprint ID**: `2026-05-27-testing-plan-execution`
**Created**: 2026-05-27 (sprint #3 in publication pipeline)
**Status**: READY
**Predecessor**: `.agent/sprints/2026-05-27-testing-plan-critique/test_plan_revised.md` (PLAN_REVISED, merged)

## Objective

Execute the **HIGH-priority** test items in `test_plan_revised.md` against the actual wrf_gpu codebase. Produce proof objects on disk for each test. This is the evidence that turns "the code exists" into "the code is proven to a publication-grade standard by community criteria." The next opus sprint (#4) will check the verdict against the pass/fail thresholds set in the revised plan.

The user has reserved the right to ship a paper that honestly characterises a partial result (per `PAPER-STRATEGIC-FRAMING.md`). The execution sprint's job is to produce HONEST PROOF OBJECTS, not to make the code pass. If a test fails the threshold, the proof object must record the failure plainly.

## Scope — HIGH-priority items only (10 items, 17.4 GPU-hour budget)

Read `.agent/sprints/2026-05-27-testing-plan-critique/test_plan_revised.md` for full specs. The HIGH-priority items are:

1. **IDEALIZED-WARMBUBBLE** — buoyant response (Bryan & Fritsch 2002 reference)
2. **IDEALIZED-DENSITY-CURRENT** — cold-pool propagation (Straka 1993 reference)
3. **IDEALIZED-MOUNTAIN-WAVE** — Schaer 2002 sinusoidal terrain primary; em_hill2d_x smoke
4. **CONSERVATION-MASS-24H** — dry-mass budget; closed-domain + Canary flux-corrected
5. **CONSERVATION-ENERGY-24H** — CPU envelope (±20% of CPU drift), not absolute 0.1%
6. **STABILITY-CFL-SWEEP** — CFL margin probe
7. **STABILITY-ACOUSTIC-SUBSTEP-SWEEP** — acoustic substep count sensitivity
8. **DETERMINISM-REPEAT** — full-pipeline determinism (with JAX_DETERMINISTIC_OPS if needed)
9. **SAVEPOINT-PARITY-DEEP** — B6 + deeper savepoints, current state of bitwise parity
10. **CANARY-MULTIDAY-SIDE-BY-SIDE** — ≥14 days, per-variable T2/U10/V10 vs CPU (T2 expected to fail per honest framing)

MEDIUM and LOW priority items are NOT in this sprint's scope; if budget permits, the execution sprint may opportunistically run COMPILE-COLD-START-TIME and VRAM-FOOTPRINT-1KM-FRESH as cheap MEDIUM bonuses.

## Acceptance

- **AC1 through AC10 — Per-test proof objects**: for each HIGH item, produce the proof object exactly as named in the revised plan, with the pass/fail verdict honestly stamped. Path pattern: `.agent/sprints/2026-05-27-testing-plan-execution/<test_id_lowercase>.json` plus optional `<test_id_lowercase>_summary.md` for tests where prose helps.

- **AC11 — Reuse before rewrite**: per the opus critique finding #9, `scripts/diagnostic_conservation_tracker.py` already implements mass + KE + dry-static-energy totals. `src/gpuwrf/validation/forecast_vs_obs.py:467` already has FSS. `scripts/m7_daily_pipeline.py` and `scripts/m7_gpu_vs_cpu_skill_diff.py` already implement the daily pipeline + skill diff. The execution sprint must **reuse these** rather than rewriting; if a tweak is needed, edit the existing script narrowly.

- **AC12 — Analytic IC builders**: per critique finding gap, no analytic IC builders for warm bubble / density current / Schaer mountain currently exist under `src/gpuwrf/fixtures/`. Write minimal builders for these three cases in `src/gpuwrf/fixtures/idealized_cases/` (NEW subpackage). Match published reference IC specs.

- **AC13 — WRF idealized CPU reference**: when needed, compile WRF idealized (`em_hill2d_x` as a smoke; analytic reference data for Schaer/Straka/Bryan-Fritsch reduces compile cost). Pin WRF commit hash in `proof.wrf_provenance` for each case.

- **AC14 — Aggregate report**: produce `.agent/sprints/2026-05-27-testing-plan-execution/aggregate_report.md` with: per-test PASS/FAIL verdict, total GPU hours used (must be within 17.4 + 6.6 reserve = 24h), what surprised you, what the publication can claim from this evidence.

- **AC15 — Worker report** with verdict `EXECUTION_COMPLETE` / `EXECUTION_PARTIAL` (with list of items completed) / `EXECUTION_BLOCKED` (with explicit blocker).

## Files Worker May Modify

- `src/gpuwrf/fixtures/idealized_cases/**` (NEW subpackage)
- `scripts/pubtest_*.py` (NEW orchestrator scripts; reuse existing diagnostic_conservation_tracker / m7_daily_pipeline / m7_gpu_vs_cpu_skill_diff where possible)
- `scripts/diagnostic_conservation_tracker.py` (narrow edits only if needed)
- `tests/test_pubtest_*.py` (NEW unit tests for the new IC builders + orchestrators)
- `.agent/sprints/2026-05-27-testing-plan-execution/**`

## Files Worker Must Not Modify

- `src/gpuwrf/runtime/operational_mode.py`, `coupling/physics_couplers.py`, `dynamics/**`, `contracts/**`, `validation/**` (model core frozen)
- `publication/**`, `publish/**` (paper untouched until sprint #5)
- Existing tests (don't delete; add new alongside)
- governance files
- `/mnt/data/canairy_meteo/**` (read-only)

## Hard Rules

1. **Honest proof objects** — record failures, don't hide them. The user's framing is explicit: honest result, not a passing-test theater.
2. **17.4 GPU-hour HIGH budget + 6.6h reserve**. Cap at 24 GPU-hours total.
3. **Reuse over rewrite** (AC11). The opus critique was explicit about which scripts to reuse.
4. **CPU pinning**: `taskset -c 0-3` for orchestrators; GPU work on RTX 5090; CPU WRF reference on cores 4-31.
5. **Don't interfere with `/home/enric/src/canairy_meteo/Gen2/` nightly scheduler.**
6. **Pin WRF commit hash** for any case that uses CPU WRF.
7. **Determinism** test may need `XLA_FLAGS` + `JAX_DETERMINISTIC_OPS`; if non-determinism is unavoidable, bound it (DETERMINISM-REPEAT proof object records the bound, not a fake pass).
8. **No remote push.** Local commit on `worker/gpt/testing-plan-execution` only.
9. **B6 + 20260521 multi-step bitwise + restart bitwise + D2H=0 must remain PASS** — the dycore is locked.

## Proof Objects

Per AC1-AC15 listed above. Plus aggregate_report.md, worker-report.md.

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 8-24 h (this is the heaviest sprint of the pipeline; 17.4 GPU-hr cost dominates)
- Branch: `worker/gpt/testing-plan-execution`
- Worktree: `/tmp/wrf_gpu2_testexec`
- GPU usage: YES — heavy. ~17 GPU-hours expected.

## What this enables

Sprint #4 (opus check) reads the proof objects and renders a publication-readiness verdict. Sprint #5 (paper rewrite) uses these proof objects as the evidence base for the new port-first abstract + Results section. Without this sprint, the paper can only claim what it already has (single-day Canary + skill regression). With this sprint, it can claim community-grade evidence across idealized cases, conservation, stability, determinism, and multi-day Canary.
