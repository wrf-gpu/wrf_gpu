# Memory Patch: V0.14 Step-1 Current-MUB/Base-Input Split

Date: 2026-06-09

Reviewer Status: Pending. Opening sprint only.

Reason:

- `step1_adjust_tempqv_intermediate` showed exact WRF/JAX agreement for saved
  inputs but a material current `mub`/`pb_new`/`p_new` mismatch of about
  `17.5 Pa`.
- Production source patching is not authorized until this current-base-input
  mismatch is split to a specific WRF operation, JAX reconstruction, hook
  boundary, or blocker.

Expected memory after close:

- Record the first divergence surface for current `MUB/PB`.
- Record WRF and JAX numeric deltas for the target cell.
- Record the next justified source-changing sprint or exact blocker.
