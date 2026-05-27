# Aggregate Report - Testing Plan Execution

Total GPU hours used: 0.0

| Test | Verdict | Status | Proof |
|---|---|---|---|
| IDEALIZED-WARMBUBBLE | BLOCKED | BLOCKED_NO_GPU_IDEALIZED_RUN | `idealized_warmbubble.json` |
| IDEALIZED-DENSITY-CURRENT | BLOCKED | BLOCKED_NO_GPU_IDEALIZED_RUN | `idealized_density_current.json` |
| IDEALIZED-MOUNTAIN-WAVE | BLOCKED | BLOCKED_NO_GPU_IDEALIZED_RUN | `idealized_mountain_wave.json` |
| CONSERVATION-MASS-24H | BLOCKED | BLOCKED_NO_24H_GPU_STATE_SERIES | `conservation_mass_24h.json` |
| CONSERVATION-ENERGY-24H | BLOCKED | BLOCKED_NO_CPU_ENVELOPE_OR_24H_GPU_SERIES | `conservation_energy_24h.json` |
| STABILITY-CFL-SWEEP | BLOCKED | BLOCKED_NO_GPU_STABILITY_RUNNER | `stability_cfl_sweep.json` |
| STABILITY-ACOUSTIC-SUBSTEP-SWEEP | BLOCKED | BLOCKED_NO_GPU_ACOUSTIC_SWEEP_RUNNER | `stability_acoustic_substep.json` |
| DETERMINISM-REPEAT | FAIL | FAIL_REQUIRED_THREE_RUNS_NOT_AVAILABLE | `determinism_repeat.json` |
| SAVEPOINT-PARITY-DEEP | FAIL | FAIL_INSUFFICIENT_SAVEPOINT_DEPTH | `savepoint_parity_deep.json` |
| CANARY-MULTIDAY-SIDE-BY-SIDE | FAIL | FAIL_INSUFFICIENT_GPU_CORPUS_AND_SINGLE_DAY_SKILL | `canary_multiday_skill.json` |

## What surprised me

- The CPU Canary inventory is sufficient for a 14-day window, but the checked-in GPU evidence is only single-day.
- The GPU preflight timed out, so no heavy HIGH-priority GPU execution could be started honestly.
- Existing B6/restart/determinism guardrail artifacts remain useful but do not satisfy the revised deeper gates.

## Publication claim supported from this sprint

This sprint supports only a partial-result framing: analytic IC builders and preflight proof objects exist, but the requested community-grade HIGH gates are mostly blocked or failing on available evidence.
