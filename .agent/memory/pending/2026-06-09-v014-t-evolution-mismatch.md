# Pending Memory Patch: V0.14 T Evolution Mismatch

Scope:

Project-memory update for v0.14 same-state JAX-vs-WRF dynamic debugging after
the T history/source-attribution sprint.

Evidence:

- `proofs/v014/jax_t_history_source_attribution.json` reports
  `T_EVOLUTION_MISMATCH_CONFIRMED`.
- The proof uses the produced h10 pre-step carry checkpoint
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl` and
  confirms its hash/size match both the producer and canonical h10 compare
  records.
- The best JAX candidate for WRF history `T_HIST_SRC` is
  `captured_pre_halo_state.theta_minus_300`, still max_abs
  `3.3545763228707983`.
- The best JAX candidate for WRF `T_THM` is
  `captured_final_carry.t_2ave_minus_300`, still max_abs
  `3.677881697025043`.
- P/PB/MU/MUB are also divergent on the same patch, so `T` is not a lone
  writer/history-source artifact.

Proposed destination:

After independent review and after theta-evolution localization names the
failing stage/operator, add a concise entry to
`.agent/memory/stable/recurring-gotchas.md`:

- Once `JAX_MISMATCH_T` is compared at the h10 post-`after_all_rk_steps`
  pre-halo surface, do not spend further root-cause sprints on JAX history
  source remapping unless new evidence changes the candidate set. The current
  proof shows no inspected JAX theta/history candidate matches WRF
  `T_HIST_SRC` or `T_THM`; localize theta evolution by stage/operator instead.

Reviewer Status:

Pending. Do not apply to stable memory until a follow-up localization sprint
confirms the failing theta stage/operator or broader state boundary.
