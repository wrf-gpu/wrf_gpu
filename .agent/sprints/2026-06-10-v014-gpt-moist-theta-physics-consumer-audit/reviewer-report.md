# Reviewer Report: V0.14 GPT Moist-Theta Physics Consumer Audit

Decision: ACCEPT AS ROADMAP-CHANGING AUDIT, NOT AS A FIX.

The sprint stayed within the requested read-only scope and produced a compact
compatibility map. It correctly separates prognostic moist-theta storage and
dycore transport from physics-facing dry-theta/temperature semantics.

Accepted conclusions:

- `state.theta` should remain moist/coupled theta for state, LBC, feedback, and
  dynamics.
- Physics adapters that construct WRF-style `T`, dry theta, virtual theta,
  density, or dry theta tendencies need explicit dry conversion on input and
  moist recoupling on writeback.
- The grid-backed MYNN/generic surface pattern is the local model for safe API
  design.
- NoahMP is not the only consumer that needs attention; the audit names several
  families that must be fixed or gated before release claims.

Manager implication:

- Fable's current NoahMP fix can continue as the immediate Step-1 blocker.
- The next roadmap item after the NoahMP proof is a broader moist-theta adapter
  boundary closure, not TOST/Switzerland-GPU.

