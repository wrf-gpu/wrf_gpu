# Worker Report: V0.14 GPT Moist-Theta Physics Consumer Audit

Summary: GPT-5.5 xhigh completed the read-only audit and produced the allowed
proof/review artifacts. No production source was edited by this sprint.

Objective:

- Audit production physics/coupling consumers of runtime `state.theta` after
  the active Fable/Mythos lane proved `state.theta` is WRF moist/coupled
  potential temperature `theta_m`.
- Identify which consumers must keep moist theta, which already decouple, and
  which must convert to dry theta/temperature at physics boundaries.

Files produced:

- `proofs/v014/moist_theta_physics_consumer_audit.json`
- `proofs/v014/moist_theta_physics_consumer_audit.md`
- `.agent/reviews/2026-06-10-v014-gpt-moist-theta-physics-consumer-audit.md`

Verdict:

`MULTIPLE_RUNTIME_PHYSICS_CONSUMERS_REQUIRE_DRY_DECOUPLING`.

Core rule:

```text
theta_dry = theta_m / (1 + (461.6 / 287.0) * qv_mixing)
theta_m   = theta_dry * (1 + (461.6 / 287.0) * qv_mixing)
```

Top result:

- State storage, LBC, feedback, and dycore transport should keep moist theta.
- Grid-backed MYNN/generic surface view already uses the correct dry-view plus
  moist-writeback pattern.
- NoahMP is the active primary bug, but the same boundary risk extends to Noah
  Classic, Thompson/scan microphysics, radiation inputs/tendency application,
  surface-layer scan adapters, PBL/MYJ, cumulus, GWDO, and legacy output
  diagnostics.

