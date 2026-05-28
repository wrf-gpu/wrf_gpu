# Sprint Contract — F2: Idealized Cases (Skamarock Warm Bubble + Straka Density Current)

**Sprint ID**: `2026-05-28-f2-idealized-cases-skamarock-straka`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/f2-idealized-cases`
**Worktree**: `/tmp/wrf_gpu2_f2`
**Wall-time**: 3-5 days
**GPU usage**: YES
**Sandbox**: `--sandbox danger-full-access`

## Why this sprint

The community-standard tests for dycore correctness are **Skamarock & Klemp
1994 warm rising bubble** and **Straka 1993 density current**. Both have
analytical/numerical reference solutions and are sensitive to the exact
dycore bugs we suspect (missing advection, wrong mass conservation, theta
decoupling errors). A dycore that gets either case visibly wrong is broken
regardless of what self-compare tests say.

Currently the project has no idealized-case validation. This sprint adds
that floor — making future dycore changes auditable against published
reference figures.

## Binding goal (universal)

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72h RMSE on T2/U10/V10
**statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins
on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs (read in order)

1. `.agent/sprints/2026-05-28-agy-dycore-deep-review/findings.md` — primary directive on the dycore state
2. `src/gpuwrf/dynamics/core/` — dycore being tested
3. `src/gpuwrf/runtime/operational_mode.py` — call structure
4. `tests/savepoint/` — existing test patterns
5. Skamarock, W. C., & Klemp, J. B. (1994). "Efficiency and accuracy of the Klemp-Wilhelmson time-splitting technique." *Monthly Weather Review*, 122(11), 2623-2630 — warm bubble setup
6. Straka, J. M. et al. (1993). "Numerical solutions of a non-linear density current." *International Journal for Numerical Methods in Fluids*, 17(1), 1-22 — density current setup

(both are old enough to be findable via standard search; if blocked, reproduce the case setup from the WRF `test/em_quarter_ss/` example which uses the same Skamarock setup)

## Approach

### Phase 1 — Warm bubble setup (1-2 days)

a. Build an initial-condition generator in Python (CPU, no GPU needed) that produces a Skamarock 1994 warm-bubble IC:
   - 2D x-z slab, 20km wide × 10km deep
   - 250m grid spacing both directions
   - Background: stable stratification, θ₀ = 300K, dθ/dz = 0
   - Bubble: 2K positive perturbation in a centered 2km-radius sphere
   - U = V = W = 0
b. Convert IC to the JAX `OperationalState` pytree expected by `operational_mode.py`.
c. Run dycore for **500 seconds** (~5000 timesteps at Δt=0.1s) using `_physics_boundary_step` with physics disabled (radiation_cadence_steps=999999, microphysics_disabled, surface_layer_disabled, pbl_disabled).
d. Emit snapshots at 100s, 250s, 500s.

### Phase 2 — Compare to Skamarock 1994 reference (0.5 day)

a. The bubble should rise as a coherent thermal, deforming into a mushroom shape by 500s.
b. Maximum θ' should remain around 2K (no spurious amplification).
c. Maximum w should be O(10 m/s).
d. The thermal should be vertically symmetric in x (small horizontal drift = sign of broken advection).
e. Document each above as a pass/fail check in `proofs/f2/skamarock_bubble_verdict.md`.

### Phase 3 — Straka density current (1 day)

a. Build IC: 2D x-z slab, 50km wide × 6km deep, 100m grid.
b. Background: θ₀ = 300K, neutral, U=V=W=0.
c. Bubble: -15K cold perturbation in a 4km-radius semi-ellipse touching ground at center.
d. Run dycore for **900 seconds**.
e. Compare to Straka 1993 reference: 3 rotors should form behind the front, the front should travel ~15 km in 900s.
f. Document in `proofs/f2/straka_density_current_verdict.md`.

### Phase 4 — Make it a routine test (0.5 day)

a. Add `tests/idealized/test_warm_bubble.py` and `tests/idealized/test_density_current.py`.
b. Each runs the case and checks a small number of integrals (max w, mass conservation, front position) against reference values.
c. These become INV-7 + INV-12 (new invariant) candidates.

## Acceptance

- **AC1**: `tests/idealized/test_warm_bubble.py` runs end-to-end; produces a verdict + plots.
- **AC2**: `tests/idealized/test_density_current.py` runs end-to-end; produces a verdict + plots.
- **AC3**: Both verdicts written in `proofs/f2/` with honest comparison to published reference figures (even if our dycore fails — fail honestly with diagnostics).
- **AC4**: One paragraph in `proofs/f2/dycore_correctness_summary.md` answering: "given the warm bubble and density current outcomes, what is the most likely structural bug in our dycore?"
- **AC5**: No regression on existing tests.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3` for all Python work.
2. **GPU usage**: YES for the dycore runs.
3. **Files writable**: `tests/idealized/**`, `src/gpuwrf/ic_generators/` (NEW directory for IC generation utilities — don't put in `src/gpuwrf/dynamics/`), `proofs/f2/**`, `.agent/sprints/2026-05-28-f2-.../**`.
4. **Files NOT writable**: `src/gpuwrf/dynamics/**`, `src/gpuwrf/runtime/operational_mode.py`, governance, plan, ADRs.
5. **No remote push.**
6. **Manager repo ONLY**.
7. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: f2 DONE exit=$?" Enter`.
8. **End with verdict**: `F2_COMPLETE` if AC1-AC5 pass; `F2_PARTIAL` with explicit gaps.

## Out of scope

- Don't fix dycore bugs. Document them.
- Don't touch existing tests (this adds NEW tests).
- Don't run on real-Canary data (that's the operational pipeline).
