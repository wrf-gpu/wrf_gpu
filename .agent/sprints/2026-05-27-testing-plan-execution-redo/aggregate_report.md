# Aggregate Report - Testing Plan Execution Redo

Total GPU hours used: 1.226423

| Test | Verdict | GPU h | Status | Proof |
|---|---:|---:|---|---|
| IDEALIZED-WARMBUBBLE | SKIP_NO_IDEALIZED_GPU_FORECAST_RUNNER | 0.000000 | SKIP_NO_IDEALIZED_GPU_FORECAST_RUNNER | `idealized_warmbubble.json` |
| IDEALIZED-DENSITY-CURRENT | SKIP_NO_DENSITY_CURRENT_GPU_FORECAST_RUNNER | 0.000000 | SKIP_NO_DENSITY_CURRENT_GPU_FORECAST_RUNNER | `idealized_density_current.json` |
| IDEALIZED-MOUNTAIN-WAVE | SKIP_NO_MOUNTAIN_WAVE_GPU_FORECAST_RUNNER | 0.000000 | SKIP_NO_MOUNTAIN_WAVE_GPU_FORECAST_RUNNER | `idealized_mountain_wave.json` |
| CONSERVATION-MASS-24H | FAIL | 0.000000 | FAIL_MISSING_CLOSED_DOMAIN_AND_BOUNDARY_FLUX_CORRECTION | `conservation_mass_24h.json` |
| CONSERVATION-ENERGY-24H | FAIL | 0.000000 | FAIL_MISSING_CPU_ENVELOPE | `conservation_energy_24h.json` |
| STABILITY-CFL-SWEEP | SKIP_NO_WARMBUBBLE_GPU_RUNNER | 0.212751 | SKIP_NO_WARMBUBBLE_GPU_RUNNER | `stability_cfl_sweep.json` |
| STABILITY-ACOUSTIC-SUBSTEP-SWEEP | SKIP_NO_DENSITY_CURRENT_GPU_RUNNER | 0.264633 | SKIP_NO_DENSITY_CURRENT_GPU_RUNNER | `stability_acoustic_substep.json` |
| DETERMINISM-REPEAT | PASS | 0.004902 | PASS_THREE_RUN_BITWISE | `determinism_repeat.json` |
| SAVEPOINT-PARITY-DEEP | FAIL | 0.000000 | FAIL_INSUFFICIENT_SAVEPOINT_DEPTH | `savepoint_parity_deep.json` |
| CANARY-MULTIDAY-SIDE-BY-SIDE | FAIL | 0.744137 | FAIL_FIVE_DAY_OR_SKILL_GATE | `canary_multiday_skill.json` |

## What Passed Cleanly

- Real GPU execution was used for available Canary pipeline, determinism, and Canary surrogate stability runs.
- Determinism is PASS only if three independent 1h GPU pipeline runs compare bitwise at the final wrfout.

## What Failed Or Was Skipped

- Idealized warm-bubble, density-current, and Schaer mountain-wave remain SKIP_* because no reviewed GPU idealized forecast runner exists in this repo scope.
- Canary multiday side-by-side fails the requested five complete-day gate when fewer than five complete d02 history days are locally runnable, and may also fail variable skill thresholds.
- Conservation and deep savepoint gates fail unless their required closed-domain/CPU-envelope/depth evidence is present.

## Paper Claim Now Supported

The paper can claim that the publication-test harness was re-run on a healthy RTX 5090 path and produced real GPU evidence for the runnable Canary pipeline subset. It cannot claim community-grade idealized-case coverage, five complete Canary days, or deep 10000-step savepoint parity from this sprint.
