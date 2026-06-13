# v0.15 Long-Horizon Divergence Verdict — are the two carries run-aways?

**Date:** 2026-06-13 UTC · branch `worker/opus/v015-divergence-reframe` (from `b0978352`)
**Question:** Are v0.15's two strict-tolerance carries — Switzerland **RAINNC** and Canary
**QVAPOR** — genuine RUN-AWAYS (escalating GPU-vs-CPU divergence), or BOUNDED diagnostics
that merely exceed the tight per-cell tolerance?

**Verdict (measured): NOT run-aways. Both are BOUNDED / non-escalating over 72 h.**
They exceed the tight frozen per-cell tolerance (correctly carried to 0.16) but their
GPU-vs-CPU divergence saturates and stays within the field's own variability — a
tight-tolerance miss, not a stability failure.

This ADDS a second, defensible equivalence criterion. It does **not** move, loosen, or
hide the strict frozen tolerance: the strict 9/10 PASS/FAIL stays red/green exactly as
shipped (the over-tolerance field is still drawn RED).

## Criterion

The principal-opened **long-horizon non-escalating-divergence** criterion
(`.agent/decisions/REDUCED-PRECISION-EQUIVALENCE-AND-FP32-RIGOR.md §3`), implemented in
`proofs/perf/v015/fp32_oracles/divergence_growth_metric.py`:

- divergence d(t) = GPU-vs-CPU RMSE per lead (the exact numbers behind the shipped
  identity dashboards; read from each region's `atlas_grid_delta_summary.json`).
- **envelope** = the ORACLE's (CPU-WRF) own internal-variability scale = the CPU field's
  spatial standard deviation, averaged over the late third of the 72 h forecast. For
  accumulated RAINNC this grows with the field, so "bounded" means *the GPU-CPU
  difference does not escape the field's OWN growth/spread*, not "the difference is tiny".
- **run-away (ESCALATING)** = the divergence slope does NOT saturate AND the divergence
  breaches the envelope (5× the oracle's own spread). BOUNDED / BOUNDED_GROWTH = no
  run-away. The discriminator is the divergence SLOPE (late-window vs early-window),
  not the endpoint value.

## The two carried fields (called out explicitly)

| field | strict tol | overall RMSE / limit | divergence regime | run-away? | early→late slope (per h) | max / oracle-env |
|---|---|---|---|---|---|---|
| **Switzerland RAINNC** | **OVER (RED, 5.08×)** | 5.079 / 1.0 mm | **BOUNDED** | **NO** | 0.241 → 0.011 (ratio +0.046) | 1.13 |
| **Canary QVAPOR** | **OVER (RED, 1.44×)** | 1.442e-3 / 1.0e-3 | **BOUNDED** | **NO** | 4.17e-5 → −1.78e-6 (ratio −0.043) | 0.47 |

- **Switzerland RAINNC** rises 0.003 → ~6.6 mm RMSE during the precip events (h1–30)
  then **plateaus** (late slope is 4.6 % of the early slope). It sits at ~1.1× the precip
  field's own spatial spread. This is the expected accumulated-precip signature: GPU and
  CPU place precip events with slightly different timing/location, the accumulated
  difference builds up while it rains, then **freezes** once the events pass — it does not
  escalate. Bounded at envelope factor ≥ 2; saturating at ALL factors.
- **Canary QVAPOR** rises to ~1.7e-3 then **oscillates flat** (late slope ~0, even mildly
  negative). It stays at **0.47× the oracle's own moisture spread** — comfortably bounded
  at every envelope factor (1×–5×). Unconditionally bounded.

## All other fields

All 9 remaining hard-gate fields per region are also **non-escalating** (no run-away):
16 BOUNDED, plus 2 Canary fields (W, U10) classed BOUNDED_GROWTH. W/U10 are **diurnal
oscillations** (their final value is below their mid-run peaks; a single linear slope over
the last third catches a rising diurnal limb) — bounded within the envelope, not a
run-away. They pass the strict tolerance regardless.

## Robustness

The slope-based run-away test is envelope-independent: both carries have
`saturating = True` at envelope factors {1, 2, 3, 5} and late-slope tolerances
{0.10, 0.25, 0.40}. QVAPOR is BOUNDED at every factor; RAINNC is BOUNDED at factor ≥ 2
(and even at factor 1 its slope is saturating — the only thing that flips is the
magnitude test, max/env = 1.13 > 1, i.e. "1.1× the field's own spread", not a runaway
slope). **No field in either region ever shows an escalating (run-away) slope.**

## Artifacts

- Verdict JSON: `proofs/v015/long_horizon_divergence_verdict.json`
- Metric (gate): `proofs/perf/v015/fp32_oracles/divergence_growth_metric.py`
- Analysis driver: `proofs/perf/v015/fp32_oracles/v015_long_horizon_divergence_verdict.py`
- Dashboard panel generator: `scripts/build_long_horizon_divergence_panel.py`
- Panels (per region, adjacent to the unchanged identity dashboards):
  - `docs/assets/v015/identity_proof/switzerland_d01/long_horizon_divergence_panel.png`
  - `docs/assets/v015/identity_proof/canary_l2_d02/long_horizon_divergence_panel.png`

## Honest one-sentence verdict

v0.15's two carries (Switzerland RAINNC, Canary QVAPOR) are **NOT run-aways**: both
exceed the tight frozen per-cell tolerance but their GPU-vs-CPU divergence is bounded and
non-escalating over 72 h (saturating slope, within the oracle's own variability) —
carried to 0.16 for the tolerance, not a stability failure.
