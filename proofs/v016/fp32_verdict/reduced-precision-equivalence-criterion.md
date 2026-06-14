# Reduced-Precision (fp32) Equivalence Criterion + Scientific-Rigor Rules

**Status:** BINDING methodology for all fp32-operational / ADR-031 work (principal
directive 2026-06-13). The fp32 speedup is the project's biggest lever — it MUST
be validated scientifically so the claim cannot be debunked after several sprints.
Stay scientific; do not wildly jump between hypotheses.

## 1. The honest fp32 number (reconcile the prior findings — do NOT over-claim)

Three measurements exist; they are NOT contradictory once labeled correctly:
- **kernel-probe (earlier):** fp32 ÷1.8–2.4× dycore, fp32-BouLac 1.67×, "realistic
  multiplier well under 2 until kernel granularity is fixed." (production-ish)
- **viability sprint (newer):** **true all-fp32 core step = 4.3×** + ~2× VRAM —
  but explicitly a **COST PROXY** (x64 toggled off; numerics NOT validated;
  metrics stayed fp64).
- **Reconciliation (THE claim to make):** the viability worker's own
  production-realistic estimate (mass/pressure kept fp64 for conservation, plus
  boundary/GWD/NoahMP) is **~1.8–3×** — which OVERLAPS the earlier 1.67–2.4×.
  **Lead with ~1.8–3× production at standard grid (~3–6× at 1 km with the VRAM
  unlock).** Treat **4.3× as the cost-proxy CEILING (numerics-unvalidated)**, not
  the headline. Never publish 4.3× as the achieved number.

The earlier "different solution" the principal recalls = the kernel-probe's
"<2× until megakernel" view; it is consistent with the production ~1.8–3×, NOT
refuted by it. No hypothesis flip — both converge on ~2×-ish production fp32.

## 2. Soundness-FIRST (do not waste sprints on a doomed approach)

Before committing implementation sprints to fp32-operational, the ADR-031 scoping
MUST establish that the **acoustic-perturbation form is mathematically sound** —
i.e. that solving the acoustic small-step in perturbation form (departures from an
fp64 reference state) removes the catastrophic-cancellation that makes naive fp32
"detonate the acoustic." If there is a computational/mathematical reason it cannot
preserve the acoustic (e.g. the cancellation is intrinsic and not curable by a
reference-state split), STOP and report it — do not pour sprints into a doomed
rewrite. If sound, drive it hard. The principal is happy to invest sprints to TEST
this hypothesis, but not to waste them.

## 3. The reduced-precision EQUIVALENCE CRITERION (principal-opened, 2026-06-13)

For reduced precision, the gate is NOT required to be strict tight per-cell
bitwise/frozen-tolerance identity (fp32 cannot meet that by construction). The
acceptable, **more scientific** criterion the principal endorses:

> **Long-horizon non-escalating divergence:** over a long forecast (72 h+), the
> fp32 solution must NOT escape / diverge from the fp64-and/or-CPU-WRF oracle in
> an **escalating** fashion. Bounded, non-growing departure (the two solutions
> co-evolve within the oracle's own variability envelope) = scientifically
> equivalent. Escalating/runaway divergence = FAIL.

Rules for applying it:
- **Conservation-critical state stays fp64-locked** (mass, surface pressure, the
  conservation budgets) regardless — only the cancellation-safe working set goes
  fp32.
- Measure a **divergence-growth metric** vs the oracle across leads (e.g. RMSE(t)
  trend / Lyapunov-style growth rate), not just an endpoint snapshot. The test is
  whether the slope is bounded, not whether the value is tiny.
- This RELAXATION applies ONLY to reduced-precision lanes; the fp64 default path
  keeps the existing frozen-tolerance tiered gate. Document which gate each lane
  uses. Never silently relax the fp64 gate.
- This is a methodology ADR; formalize the exact metric + threshold when the
  fp32-operational work produces real long-horizon runs.

## 4. Process

Scientific, not thrashing: one ranked hypothesis (acoustic-perturbation-form
fp32), validated by (a) soundness analysis, (b) numerics validation on real runs
(NOT cost-proxy), (c) the long-horizon non-divergence gate, (d) the honest
production speedup measured. The joint Opus+GPT attack (GPT online ~tonight)
executes the ADR-031 blueprint under these rules. See
`KERNEL-OPTIMIZATION-FINDINGS-FINAL.md §8`, `ADR-031`, `proofs/perf/v015/viability/`.
