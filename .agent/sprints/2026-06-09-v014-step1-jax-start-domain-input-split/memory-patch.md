# Memory Patch: V0.14 Step-1 JAX Start-Domain Input Split

Reviewer Status: `ACCEPTED_NO_STABLE_MEMORY_CHANGE`.

No stable memory change is proposed at close.

Accepted lesson:

- Reusing predecessor WRF truth surfaces was the right wall-clock method for
  this boundary and avoided an unnecessary new WRF instrumentation loop.
- The recurring live-nest debug pattern is now: close WRF source order first,
  split current JAX inputs by family, and do not patch production from a formula
  proof if the production inputs cannot reproduce the WRF truth array.

No update to stable project skills is required from this sprint beyond the
already-recorded manager rule to prefer focused truth surfaces and input-family
falsifiers over broad runtime chasing.
