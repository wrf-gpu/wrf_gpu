# V0.14 Step-1 Current-MUB/Base-Input Split

Date: 2026-06-09 17:32 WEST

Sprint:
`.agent/sprints/2026-06-09-v014-step1-current-mub-base-input-split`.

Proof:
`proofs/v014/step1_current_mub_base_input_split.*`.

Verdict:
`STEP1_CURRENT_MUB_BASE_SPLIT_WRF_BLEND_UNIMPLEMENTED_OR_MISMATCHED`.

Important facts:

- WRF `adjust_tempqv` uses current `MUB` after `blend_terrain` and before the
  later `start_domain` base recompute.
- The previous JAX theta proof used final post-`start_domain` base `MUB` for
  that earlier `adjust_tempqv` call.
- WRF adjust hook current `MUB`: `86812.25`.
- JAX final base `MUB`: `86794.574960128695`.
- Proof-side direct WRF blend `MUB`: `86812.250452109511`.
- WRF pre-part1 final `MUB`: `86794.5703125`.
- WRF source formula is `p_new = p + c4h + c3h*mub + p_top`; the grouped
  `c3h*(mub+p_top)` formula is not WRF's `adjust_tempqv` formula.

Manager conclusion:

Next source sprint should add a transient live-nest adjust-base path for
theta/QV adjustment only, while keeping final post-`start_domain` BaseState for
step-entry. TOST, Switzerland, FP32 source landing, and memory source work stay
paused until the Step-1 theta/QV proof is rerun and either closes or names the
next boundary.
