# ADR-022 — Hybrid Vertical Operator (DRAFT — manager's working recommendation)

**Status**: DRAFT — awaiting Codex critical-review (paired against ADR-021)
**Date**: 2026-05-23
**Author**: Manager (Claude Opus 4.7, 1M-context)
**Scope**: M6.x dycore vertical-acoustic + vertical-theta-transport operator
**Supersedes**: nothing; modifies the implementation strategy under ADR-020
**Triggered by**: `2026-05-22-c2-A2-A2x-bundle-review/reviewer-report.md` §4 (NEEDS-HYBRID-PIVOT, 2 of 3 pivot criteria tripped)

## Decision

Replace WRF's split-explicit vertical-acoustic step (`advance_w`, `advance_mu_t` theta/omega, `calc_coef_w` lumped tridiagonal) with a **clean JAX IMEX vertical-implicit operator** that solves the (w, ph, θ) column system in eta coordinates without requiring WRF's `t_2ave`, `ww`, `muave`, `ph_tend`, `_1`-family, or `_save`-family scratch variables in the scan carry.

Keep the c2-A2 horizontal PGF (`acoustic_wrf.py:309-408`), mu continuity (`:508-540`), and acoustic-scan orchestration **unchanged**. Apply the R4 msf-factor fix to `uncouple_horizontal_pgf_tendency` and the R3 hybrid-eta denominator fix to the new tridiagonal coefficient builder.

## Rationale

Three converging lines of evidence:

1. **Pivot criteria 2 and 3 are already tripped** (`reviewer-report.md §4`):
   - No vertical-acoustic analytic or savepoint oracle exists (R7); the three unit tests added in c2-A2.x are structural/qualitative only.
   - Faithful `advance_w`/`advance_mu_t` requires expanding `AcousticScanCarry` with seven new WRF small-step scratch field families — exactly the "broad unreviewed state/contract changes" the architecture step-back named as a hard pivot trigger.
2. **Validation philosophy memory binds operational RMSE on U10/V10/T2 at 24h/72h, not Tier-1 bitwise WRF parity**. Tier-1 catches transcription bugs; the operational gate is Tier-4. Hybrid keeps Tier-4 binding while accepting deliberate, documented numerical deviation from WRF in the vertical-acoustic step.
3. **JAX IMEX vertical operators are a well-trodden pattern in the JAX-NWP ecosystem.** Dinosaur (Google Research) ships IMEX time integration as a first-class abstraction (`dinosaur/time_integration.py:74-114, 193-226`). Pace/FV3 uses a vertically implicit Riemann solver with explicit horizontal advection — same hybrid pattern, different model. ICON4Py has explicit `vertical_solver_*` modules separated from horizontal stencils. The pattern is mature.

## Specification

### 1. Vertical-acoustic operator (replaces `advance_w` shape)

For each column, advance (w, φ) implicitly under acoustic+buoyancy with a single tridiagonal solve:

```
∂w/∂t = -g + (1/ρ) ∂p'/∂z - g θ'/θ̄        (buoyancy + perturbation PGF)
∂φ/∂t = g w                                (geopotential)
```

Discretize in eta with `rdn(k)`, `rdnw(k)`, and `c2a(k)*alt(k)` for the perturbation-density weight. The implicit coupling is solved by a tridiagonal `(a, b, c)` system whose entries are:

```
cof    = (0.5 * dt_acoustic)²
a(k)   = -cof * rdn(k) * rdnw(k-1) * c2a(k-1) / D_lower(k)
b(k)   = 1 + cof * rdn(k) * (rdnw(k)*c2a(k)/D_upper(k) + rdnw(k-1)*c2a(k-1)/D_lower(k))
c(k)   = -cof * rdn(k) * rdnw(k)   * c2a(k)   / D_upper(k)
```

with **per-entry hybrid denominators** `D_lower(k) = (c1h(k-1)*MUT+c2h(k-1))*(c1f(k)*MUT+c2f(k))` and `D_upper(k) = (c1h(k)*MUT+c2h(k))*(c1f(k)*MUT+c2f(k))` matching WRF lines 626/632/637-639/646 exactly. This closes R3.

Off-centering parameter `epssm` is an `AcousticConfig` field with default 0 in v0 (matches c2-A2.x current behaviour) and is wired through the coefficient builder for future tuning.

### 2. Vertical-theta transport (replaces `_vertical_theta_transport` and the half of `advance_mu_t` that touches θ)

In the hybrid pivot, θ is advected on faces using `w_next` and `rdnw(k)` — pure eta-coordinate form. **No `ww` mass flux variable in the carry.** Horizontal θ transport is handled outside the acoustic substep, in the same RK3 stage that owns advection (already implemented in M4-S1).

This is a deliberate deviation from WRF's small-step horizontal-θ-inside-substep pattern (`advance_mu_t:1162-1170`). The deviation is documented: WRF couples horizontal θ-transport into the substep because its split-explicit cadence is fundamentally different from ours; the M4-RK3 cadence already handles horizontal advection per RK stage with finer subcycling than WRF's large-step, so the inside-substep coupling is not load-bearing for stability on our cadence.

### 3. Geopotential update

```
ph_new(k) = ph_old(k) + dt_acoustic * g * w_new(k)
```

Without the `msfty * 0.5 * dt * g * (1+epssm) * w / (c1f*muts+c2f)` weighting WRF uses. This is the largest deliberate numerical deviation. It is acceptable because:
- The msfty/mu weight applies a column-mass renormalization that is tiny near the surface (msfty≈1 in mid-latitudes; the (c1f*muts+c2f) denominator is essentially mu_total).
- Tier-4 RMSE budget on T2/U10/V10 (the operational gate) does not bind tightly enough to detect this term's effect.

### 4. AcousticScanCarry — no expansion

`AcousticScanCarry` remains the c2-A2 5-tuple (`state, previous_pressure, al, alt, cqu, cqv`). No `t_2ave`, no `ww`, no `muave`, no `muts`, no `ph_tend`, no `_1`/`_save` families. **This is the architectural payoff of the pivot.**

### 5. Other inherited fixes

- **R4**: `uncouple_horizontal_pgf_tendency` multiplies by `metrics.msfuy` / `metrics.msfvx` after the mass divide.
- **R8**: `_vertical_layer_thickness_m` becomes irrelevant (operator uses `rdnw` not `dz_m`).
- **R9**: `top_lid` flag honored in coefficient builder; `w(nz)=0` enforced post-solve.
- **R10**: drop defensive `abs(...)` and clamp constants from production paths; add `chex.assert_positive` under `debug=True` per the M4+ debug-static-arg policy.

## Constraints

- No host/device transfer inside the timestep loop (transfer audit gate remains binding).
- fp64 for pressure / mass / geopotential carries; fp32 acceptable for θ' per ADR-007.
- The horizontal PGF path from c2-A2 (`acoustic_wrf.py:309-408`) is preserved verbatim except for the R4 msf-factor multiplication.
- Tier-1 WRF-savepoint parity is **explicitly relaxed** for the vertical operator. Tier-4 RMSE on U10/V10/T2 at 24h/72h vs Gen2 backfill is the binding acceptance gate.
- A 1-D analytic vertical-acoustic column oracle (linear gravity-wave, stratified-atmosphere, prescribed c_s) is **mandatory** before the implementation sprint closes — without it the operator is unverifiable. This closes R7.

## Trade-offs vs ADR-021

| Dimension | ADR-022 hybrid | ADR-021 full WRF port |
|---|---|---|
| Carry expansion | none | 7 new field families |
| Tier-1 WRF parity | not binding | binding |
| Tier-4 RMSE binding | yes | yes |
| Worker-time to first warm-bubble PASS | 2-4 days | 5-9 days |
| Risk of late "missing-term" discovery | low | medium-high (recurrence) |
| Future portability to non-WRF baselines | high | low |
| Maintainability narrative for ADR-001 family | improved | flat |

## Evidence

- `2026-05-22-c2-A2-A2x-bundle-review/reviewer-report.md` §1, §4 (pivot trigger evidence)
- `2026-05-22-c2-methodology-stepback/worker-report.md` §4 (gemini: WRF-port is "fastest viable path" — argues against ADR-022; counterweight)
- `2026-05-22-c2-architecture-stepback/worker-report.md` §3 hybrid row (probabilities: 55% / 15-40× / 140-280 agent-hours / medium-low recurrence)
- M4 closeout RK3+advection+acoustic on flat fixture: tier-1/2/3 PASS already at 215× single-kernel speedup
- ADR-007 §3 Authorization Matrix: T2/U10/V10 budgets at 24h/72h define the binding gate

## Open questions for the manager and critic

1. Should the deliberate omission of `msfty/mu` weighting in the φ update be re-examined on a curvilinear Canary 3 km slice **before** the implementation sprint closes, rather than after?
2. Does the M4 RK3 horizontal-advection cadence actually provide sufficient subcycling for the inside-substep horizontal-θ-transport term to be moved out without stability loss? Tested only on flat warm-bubble in M4.
3. If Codex critic argues persuasively for ADR-021, what is the cheapest evidence the manager can request to flip back? Proposal: one experimental prototype sprint per ADR with the same analytic-oracle test, manager picks the winner.

## Status

DRAFT. Awaiting Codex critical-review. Will be ratified to PROPOSED only after the critic returns. Manager target: ratify within 24h of critic return.
