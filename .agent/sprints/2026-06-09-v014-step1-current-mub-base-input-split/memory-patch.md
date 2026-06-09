# Memory Patch: V0.14 Step-1 Current-MUB/Base-Input Split

Date: 2026-06-09

Reviewer Status: APPROVED_FOR_PENDING_MEMORY

Record this sprint as closed with verdict
`STEP1_CURRENT_MUB_BASE_SPLIT_WRF_BLEND_UNIMPLEMENTED_OR_MISMATCHED`.

Reason:

- `step1_adjust_tempqv_intermediate` showed exact WRF/JAX agreement for saved
  inputs but a material current `mub`/`pb_new`/`p_new` mismatch of about
  `17.5 Pa`.
- `step1_current_mub_base_input_split` proves this is a boundary split:
  `adjust_tempqv` consumes transient post-`blend_terrain`/pre-`start_domain`
  current `MUB`, while the previous JAX theta proof used final
  post-`start_domain` base `MUB`.

Proof facts:

- WRF adjust hook current `MUB`: `86812.25`.
- JAX proof final base `MUB`: `86794.574960128695`.
- Proof-side direct WRF blend `MUB`: `86812.250452109511`.
- WRF pre-part1 final `MUB`: `86794.5703125`.
- WRF source formula: `p_new = p + c4h + c3h*mub + p_top`.

Next:

Open a source-changing sprint to compute transient post-blend/pre-`start_domain`
`MUB` for the `adjust_tempqv` theta/QV adjustment only, keep final BaseState
unchanged, and rerun Step-1 theta/QV proof plus field-level guard.
