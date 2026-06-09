# V0.14 Debug-Method Critic

Date: 2026-06-09

Status: pending stable-memory review.

Fact:

- `.agent/reviews/2026-06-09-v014-debug-method-critic.md` was produced by
  Claude Opus xhigh as an independent read-only critique.
- The critique accepts native live-nest base initialization as a real correctness
  issue but rejects treating it as the V10/grid-field symptom closer without a
  direct falsifier.
- The review cites `proofs/v014/wind_mass_divergence_probe.md` as ranking
  dynamic wind/theta-carry divergence above static base-state mismatch for the
  V10 symptom.
- Static base fields such as `MUB` can dominate max-abs bisect headlines and
  mask dynamic-field localization.

Operational consequence:

- Source-port proofs must separate "base-state agreement improved" from
  "grid-field/V10 symptom closed".
- Before resuming TOST, run direct grid-field validation; before claiming a
  base port as the divergence fix, run an init-override or direct V10/grid-field
  proof.
