# ADR-024 — Warm-Bubble Gate Policy

Status: **ACCEPTED** (2026-05-24, per external consultation; supersedes the PROPOSED state of 2026-05-23. Warm-bubble is permanently a diagnostic / operator-sanity gate, never an architecture-acceptance target.)
Date: 2026-05-23 (proposed); 2026-05-24 (accepted)
Author: M6.x warm-bubble gate-redesign worker

## Context

The `m6x-warm-bubble-gate-strategy-critic` returned verdict `CHANGE-THE-GATE` at commit `c80b622`: the `[5, 10] m/s` warm-bubble amplitude band is not sourced for the current pure-small-step Gaussian harness. The critic found that two apparent passes were produced by prototype stabilizers rather than production physics: ADR-021 clamped `w` toward exactly 9 m/s, and the ADR-023 prototype used nonhydrostatic buoyancy scaling, updraft drag, and mu gating.

The Opus failure diagnostic showed why the old verdict was unsafe. The unified ADR-023 path reported a small positive `w_max`, but the hidden state had large signed/downward velocity, theta perturbation growth, pressure perturbation growth, and mu-limiter saturation. Its §9 question 2 also flags that the likely source lineage for the amplitude target assumes WRF EM-CORE big-step RK3 reinjection, not this isolated acoustic small-step loop.

The closest published/local WRF evidence is `em_squall2d_x`, which uses a materially different setup: 2-D x-z squall line, RK3, Kessler microphysics, fixed viscosity, damping/divergence controls, different grid spacing, and a configured acoustic substep cadence. That evidence is useful for designing a future sourced reference, but it does not justify using `[5, 10] m/s` as a pass/fail band for the current harness.

## Decision

The warm-bubble harness is now an **operator-sanity gate**, not an amplitude-pass gate. It reports the legacy positive `w_max` amplitude as a diagnostic only. The pass/fail verdict is limited to:

- `FAIL_FINITENESS` if the run becomes nonfinite, or if the R7 vertical-acoustic oracle or hydrostatic-rest prerequisite fails.
- `FAIL_PHYSICAL_BOUNDS` if theta, pressure, or mu perturbations exceed conservative bounds.
- `FAIL_ANTI_CLAMP_DETECTION` if the production path contains target-shaped warm-bubble clamps.
- `PASS_OPERATOR_SANITY` otherwise.

The current conservative bounds are intentionally broad: theta perturbation within `[-50, 50] K`, pressure perturbation within `[-50000, 50000] Pa`, and `mu_perturbation_max_Pa <= 50000`. These bounds are not tuned to force a pass; a follow-up sprint may tighten them with sourced reference evidence.

## Anti-Tautology Gates

Any future amplitude gate must satisfy all of the following before it can bind architecture decisions:

1. The amplitude envelope or trajectory must come from WRF, CM1, MPAS, or an analytic derivation external to the JAX operator under test.
2. A pass must fail if `w_max` is pinned to a threshold value, if negative `w` is clipped away, or if theta, pressure, or mu extrema leave documented physical bounds.
3. Accepted stabilizers must be named, sourced, and parameterized from the numerical scheme. A stabilizer that only moves this harness into `[5, 10] m/s` is rejected.
4. M6 close remains Tier-3 convergence plus initial Tier-4 consistency, conservation/bounds evidence, and clean transfer audit, not one warm-bubble amplitude number.

## Two-Stage Path

Stage 1 is this sprint: keep the current integration loop and grid setup, but replace the verdict with operator-sanity semantics and add a static anti-clamp scan.

Stage 2 is required only if the manager wants a convective amplitude gate later: build a sourced WRF/CM1/MPAS reference with matching grid, forcing, RK/acoustic cadence, damping, physics toggles, output timing, and tracked data/manifest artifacts. Only that reference may define a binding amplitude envelope.

## Consequences

The warm-bubble probe remains valuable because it catches missing buoyancy, sign errors, nonfinite evolution, pressure/mu blowups, and clamp-shaped passes. It no longer rewards an implementation for satisfying an unsourced `[5, 10] m/s` band.

ADR-023 remains PROPOSED. This ADR does not ratify any operator architecture, does not remove the documented `_mu_continuity_increment` limiter, and does not modify the R7 oracle, MPAS slice oracle, c2-A2 horizontal PGF, or `mu_continuity_tendency`.

## Proof Objects

- Harness and JSON policy: `scripts/m6_warm_bubble_test.py`
- New pytest gate: `tests/test_m6x_warm_bubble_operator_sanity.py`
- Current-state verdict: `.agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_current_state_verdict.json`
- No-regression and transfer-audit proofs in the same sprint folder
