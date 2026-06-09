# Memory Patch

Scope:

Project-memory update for the v0.14 grid-parity debug chain after previous-step
handoff bisection.

Evidence:

- `proofs/v014/previous_step_handoff_bisect.json` verdict is
  `BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE`.
- `final_reproducer_identity.all_target_fields_exact=true` proves the replay
  reproduces the bad checkpoint target leaves.
- The earliest captured surface at d02 completed step 5997 already has
  `all_target_fields_match_wrf_truth=false` and
  `static_base_fields_match_wrf_truth=false`.
- At that surface `MUB` max_abs is `1050.3046875`; `PB` max_abs is
  `1047.015625`.
- CPU live replay is blocked by `State.zeros` requiring GPU, but the required
  CPU validation command reuses the compact replay artifact and regenerates the
  repo proof objects.

Proposed destination:

Create `.agent/memory/pending/2026-06-09-v014-previous-step-handoff-bisect.md`.
After the earlier-source bisection closes, condense both into stable memory as
one grid-parity debugging fact.

Reviewer Status:

Pending. Do not promote to stable memory until the next sprint identifies
whether the first wrong source is native load, a replay segment, or a specific
handoff/hook boundary.
