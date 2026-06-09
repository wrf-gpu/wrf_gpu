# Memory Patch Proposal: V0.14 Step-1 Start-Domain Perturbation Subsurface

Date: 2026-06-09 19:45 WEST

Reviewer Status: Pending. Opening sprint only.

Scope:

- `.agent/memory/pending/2026-06-09-v014-step1-start-domain-perturb-subsurface.md`

Reason:

- The predecessor localized `P/MU/W` residuals to WRF live-nest
  `start_domain(nest,.TRUE.)` perturbation-state initialization but did not have
  the exact internal `al/alt` and pre/post-`press_adj` surfaces needed for a
  production patch.

Evidence:

- Opening evidence:
  `proofs/v014/step1_live_nest_perturb_state_init.json`.
- Closing evidence expected:
  `proofs/v014/step1_start_domain_perturb_subsurface.json`.

Proposed Destination:

- `.agent/memory/pending/2026-06-09-v014-step1-start-domain-perturb-subsurface.md`

Expected memory after close:

- Whether the WRF `start_domain` internal surfaces close `P_STATE/MU_STATE`.
- Any exact formula/source location or remaining blocker.
- Before/after residuals if a source fix is applied.
