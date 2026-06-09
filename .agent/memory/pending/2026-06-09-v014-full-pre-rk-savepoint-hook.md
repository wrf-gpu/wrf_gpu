# Pending Memory: V0.14 Full Pre-RK Savepoint Hook

Status: pending promotion after the source/save-boundary proof identifies the
correct WRF hook location.

Lesson:

- Full native CPU-WRF pre-RK state at `d02` step `6000` is now available from a
  validated scratch WRF hook.
- The hook output includes full dry state, active moisture, and scalar records;
  duplicate tile overlap max delta is `0.0`.
- Strict same-input JAX execution is still blocked because current-step
  `DryPhysicsTendencies`/save-family leaves are not available at the exact
  step-entry boundary.
- Do not feed zeros or JAX-generated tendencies into the proof; that would no
  longer be same-input.
- The next proof-enabling target is a WRF boundary after `*_tendf`,
  `h_diabatic`, `*_save`, `moist_old`, and `scalar_old` exist, but before the
  native state has been changed by dynamics.

Evidence:

- `proofs/v014/full_pre_rk_savepoint_hook.json`
- `proofs/v014/full_pre_rk_savepoint_hook.md`
- `proofs/v014/same_input_single_rk_parity_full.json`
- `proofs/v014/same_input_single_rk_parity_full.md`
- `.agent/reviews/2026-06-09-v014-full-pre-rk-savepoint-hook.md`
- `.agent/sprints/2026-06-09-v014-full-pre-rk-savepoint-hook/manager-closeout.md`
