# Pending Memory Patch: V0.14 Dynamic Layer Boundary

Scope:

Project-memory update for v0.14 same-state dynamic localization. This applies
when comparing CPU-WRF source-derived layers against GPU/JAX wrfout fields from
the h10 `d02` grid-parity investigation.

Evidence:

- `proofs/v014/wrf_same_state_marker_savepoint.json` proves the accepted h10
  post-RK marker is history-aligned against CPU h10 (`T/P/PB=0`,
  `U/V/W/PH <= 1.91e-6` max_abs).
- `proofs/v014/wrf_dynamic_term_localization.json` emits
  `final_stage_pre_small_step_finish` and `final_stage_post_small_step_finish`
  from disposable CPU-WRF scratch at the same h10 patch.
- `proofs/v014/wrf_dynamic_term_localization.md` shows
  `post_small_step_finish` is useful but not the final wrfout history surface:
  `P` differs from the post marker by about 1981 Pa max_abs and `V` by about
  57.95 m/s max_abs at that tile-local layer, while `PB/U/PH` are already
  aligned there.
- `proofs/v014/wrf_post_rk_refresh_localization.md` then confirms the green
  bridge: final `calc_p_rho_phi` closes `P`, and the state immediately after
  `dyn_em/solve_em.F::after_all_rk_steps` before RK halo exchanges is exact or
  roundoff-level against the accepted post marker and CPU h10 for
  `T/P/PB/U/V/W/PH/MU/MUB`.

Proposed destination:

After independent review, add a concise entry to
`.agent/memory/stable/recurring-gotchas.md`:

- In h10 same-state dynamic localization, do not compare JAX directly against
  WRF's tile-local `post_small_step_finish` layer as if it were wrfout history.
  It is not history-aligned for `P/V/W`. Use the green compare target
  immediately after `dyn_em/solve_em.F::after_all_rk_steps` and before RK halo
  exchanges; final `calc_p_rho_phi` closes `P`, and `after_all_rk_steps` closes
  `V/W`.

Reviewer Status:

Pending. Do not apply to stable memory until a reviewer approves this as a
general recurring gotcha and the JAX wrapper sprint confirms whether the same
surface is practical as the source-level compare target.
