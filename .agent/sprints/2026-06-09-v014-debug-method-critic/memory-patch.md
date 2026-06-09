# Memory Patch

Scope:

Project-memory update for v0.14 grid-debug sequencing after independent Opus
critique.

Evidence:

- `.agent/reviews/2026-06-09-v014-debug-method-critic.md` argues that native
  live-nest base initialization is a legitimate correctness fix but not yet
  proven to be the V10/grid-field symptom owner.
- The review cites `proofs/v014/wind_mass_divergence_probe.md` as ranking
  dynamic wind/theta-carry divergence ahead of static base-state mismatch.
- Static `MUB` max-abs can dominate bisect summaries and should not be used as
  the sole headline selector for dynamic divergence debugging.
- Required gate: a base-state source port must not be claimed as grid-parity
  closure without an init-override or direct V10/grid-field proof.

Proposed destination:

Create `.agent/memory/pending/2026-06-09-v014-debug-method-critic.md`.

Reviewer Status:

Pending. Promote after the active source sprint is resolved under this gate.
