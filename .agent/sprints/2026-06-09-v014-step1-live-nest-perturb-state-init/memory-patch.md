# Memory Patch Proposal: V0.14 Step-1 Live-Nest Perturb-State Init

Date: 2026-06-09 19:15 WEST

Reviewer Status: Pending. Opening sprint only.

Scope:

- `.agent/memory/pending/v014-grid-parity.md`

Reason:

- The predecessor localized the current `P/MU/W` residual to missing live-nest
  perturbation-state initialization before WRF `first_rk_step_part1`.
- This sprint should prove, fix, refute, or block that exact contract.

Evidence:

- Opening evidence:
  `proofs/v014/step1_first_rk_part1_p_state_split.json`.
- Closing evidence expected:
  `proofs/v014/step1_live_nest_perturb_state_init.json`.

Proposed Destination:

- `.agent/memory/pending/2026-06-09-v014-step1-live-nest-perturb-state-init.md`

Expected memory after close:

- Whether WRF live-nest `P/MU/W` perturbation-state initialization is the first
  remaining Step-1 bug.
- Any exact formula/source location or blocker.
- Before/after residuals if a source fix is applied.
