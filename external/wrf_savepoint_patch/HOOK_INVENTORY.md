# Savepoint Wrapper Hook Inventory

Captured by sprint `2026-05-25-m6b-ladder-hygiene-cleanup` (Stage 2). The wrapper
module `dyn_em/savepoint_wrapper.F90` exposes **30** subroutines (12 zero-arg
legacy hooks plus 18 M6B1/M6B2/M6B3/M6B4 typed-arg hooks). Of these, **6 hooks
(3 pre/post pairs) from the hygiene baseline are wired into** `solve_em.F` and
**4 hooks (3 pre + 1 post)** are wired into
`module_small_step_em.F`; M6B4 adds **2 acoustic recurrence hooks wired into**
`solve_em.F` (12 total wired sites; 18 wrapper hooks declared but
not yet inserted in WRF source).

All hook **bodies are EMPTY** — i.e. the wrapper file contains zero in-timestep
HDF5 emission code. Production savepoint extraction is still performed by the
Python orchestrator `scripts/m6b0r_wrf_savepoint_extract.py` over wrfout slices.
Full ABI wiring is queued in sprint
`2026-05-25-m6b0r-fortran-hook-abi-followup`.

## Table

| # | Hook | ABI args | Body | Wired in `solve_em.F`? | Wired in `module_small_step_em.F`? | Active emission? | Defining sprint |
|---|------|----------|------|------------------------|------------------------------------|------------------|-----------------|
| 1 | `sp_calc_coef_w_pre` | 0 | EMPTY | YES (lines 2692, 2727) | NO | NO | M6B0-R |
| 2 | `sp_calc_coef_w_post` | 0 | EMPTY | YES (lines 2707, 2742) | NO | NO | M6B0-R |
| 3 | `sp_small_step_prep_post` | 0 | EMPTY | NO | NO | NO | M6B0-R |
| 4 | `sp_advance_mu_t_pre` | typed (rkstage, acstep, mu, mut, mudf, muts, muave, ww, theta) | EMPTY (typed args only) | YES (line 3426) | NO | NO (stub body) | M6B1 |
| 5 | `sp_advance_mu_t_post` | typed (+ ph_tend) | EMPTY (typed args only) | YES (line 3448) | NO | NO (stub body) | M6B1 |
| 6 | `sp_t_2ave_update_pre` | typed (rkstage, acstep, t_old, t_new, t_2ave) | EMPTY (typed args only) | NO (declared, not yet inserted) | NO | NO | M6B3 |
| 7 | `sp_t_2ave_update_post` | typed (rkstage, acstep, t_2ave) | EMPTY (typed args only) | NO (declared, not yet inserted) | NO | NO | M6B3 |
| 8 | `sp_ww_update_pre` | typed (rkstage, acstep, ww_old, ww_new) | EMPTY (typed args only) | NO (declared, not yet inserted) | NO | NO | M6B3 |
| 9 | `sp_ww_update_post` | typed (rkstage, acstep, ww_out) | EMPTY (typed args only) | NO (declared, not yet inserted) | NO | NO | M6B3 |
| 10 | `sp_muave_update_pre` | typed (rkstage, acstep, mu_old, mu_new, mut, muave, muts) | EMPTY (typed args only) | NO (declared, not yet inserted) | NO | NO | M6B3 |
| 11 | `sp_muave_update_post` | typed (rkstage, acstep, muave, muts) | EMPTY (typed args only) | NO (declared, not yet inserted) | NO | NO | M6B3 |
| 12 | `sp_ph_tend_accumulate_pre` | typed (rkstage, acstep, ph_tend, ph_tend_increment) | EMPTY (typed args only) | NO (declared, not yet inserted) | NO | NO | M6B3 |
| 13 | `sp_ph_tend_accumulate_post` | typed (rkstage, acstep, ph_tend) | EMPTY (typed args only) | NO (declared, not yet inserted) | NO | NO | M6B3 |
| 14 | `sp_substep_save_state_pre` | typed (rkstage, acstep, u, v, w, t, ph, mu, ww) | EMPTY (typed args only) | NO (declared, not yet inserted) | NO | NO | M6B3 |
| 15 | `sp_substep_save_state_post` | typed (rkstage, acstep, u_save, v_save, w_save, t_save, ph_save, mu_save, ww_save) | EMPTY (typed args only) | NO (declared, not yet inserted) | NO | NO | M6B3 |
| 16 | `sp_acoustic_substep_complete` | typed (rkstage, substep, mu, mut, mudf, muts, muave, ww, theta, ph_tend, u, v, w, ph, p, t_2ave) | EMPTY (typed args only) | YES (line 4398) | NO | NO (stub body) | M6B4 |
| 17 | `sp_acoustic_loop_complete` | typed (rkstage, mu, mut, mudf, muts, muave, ww, theta, ph_tend, u, v, w, ph, p, t_2ave) | EMPTY (typed args only) | YES (line 4406) | NO | NO (stub body) | M6B4 |
| 18 | `sp_advance_uv_post` | 0 | EMPTY | NO | NO | NO | M6B0-R |
| 19 | `sp_advance_w_rhs_ready` | 0 | EMPTY | NO | NO | NO | M6B0-R |
| 20 | `sp_advance_w_raw_w` | 0 | EMPTY | NO | NO | NO | M6B0-R |
| 21 | `sp_advance_w_tridiag_fwd_pre` | typed (rkstage, acstep, a, alpha, gamma, rhs) | EMPTY (typed args only) | NO | YES (line 1533) | NO (stub body) | M6B2 |
| 22 | `sp_advance_w_tridiag_fwd_post` | typed (rkstage, acstep, a, alpha, gamma, w_fwd) | EMPTY (typed args only) | NO | YES (line 1539) | NO (stub body) | M6B2 |
| 23 | `sp_advance_w_tridiag_back_pre` | typed (rkstage, acstep, gamma, w_fwd) | EMPTY (typed args only) | NO | YES (line 1540, paired with fwd_post) | NO (stub body) | M6B2 |
| 24 | `sp_advance_w_tridiag_back_post` | typed (rkstage, acstep, gamma, w_solved) | EMPTY (typed args only) | NO | YES (line 1555) | NO (stub body) | M6B2 |
| 25 | `sp_advance_w_rayleigh` | 0 | EMPTY | NO | NO | NO | M6B0-R |
| 26 | `sp_advance_w_ph_final` | 0 | EMPTY | NO | NO | NO | M6B0-R |
| 27 | `sp_calc_p_rho_post` | 0 | EMPTY | NO | NO | NO | M6B0-R |
| 28 | `sp_small_step_finish_post` | 0 | EMPTY | NO | NO | NO | M6B0-R |
| 29 | `sp_acoustic_substep_boundary` | 0 | EMPTY | NO | NO | NO | M6B0-R |
| 30 | `sp_rk_stage_boundary` | 0 | EMPTY | NO | NO | NO | M6B0-R |

## Summary statistics

- **Total declared hooks**: 30 (`grep -c "^  subroutine sp_" dyn_em/savepoint_wrapper.F90`)
- **Hooks called in `solve_em.F`** (post-M6B4 patch): 8 (calc_coef_w pre/post x2; advance_mu_t pre/post; acoustic_substep/loop_complete)
- **Hooks called in `module_small_step_em.F`** (post-hygiene patch): 4 (tridiag_fwd_pre/post; tridiag_back_pre/post)
- **Total wired sites**: 12
- **Wrapper hooks with NON-zero arg ABI**: 18 (the M6B1/M6B2/M6B3/M6B4 hooks)
- **Wrapper hooks with empty 0-arg ABI**: 10 (M6B0-R legacy)
- **Wrapper hooks with non-empty body**: 0 (all bodies are stubs; HDF5 emission is Python-orchestrated)

## What this means for ladder honesty

- Every wired hook's argument list reaches the wrapper, but the wrapper does
  nothing with it. The savepoint files used by M6B0-R/M6B1/M6B2/M6B3 comparators
  are produced by `scripts/m6b0r_wrf_savepoint_extract.py` reading the
  operational wrfout slice and rebuilding the small-step bundle in NumPy/JAX.
- The patch is therefore a **declared-ABI scaffolding artifact**, not an
  emission lane. The cumulative ladder audit captured this; this inventory makes
  it machine-readable.
- The 18 unwired hooks (`sp_t_2ave_update_*`, `sp_ww_update_*`,
  `sp_muave_update_*`, `sp_ph_tend_accumulate_*`, `sp_substep_save_state_*`,
  `sp_advance_uv_post`, `sp_advance_w_rhs_ready`, `sp_advance_w_raw_w`,
  `sp_advance_w_rayleigh`, `sp_advance_w_ph_final`, `sp_calc_p_rho_post`,
  `sp_small_step_finish_post`, `sp_acoustic_substep_boundary`,
  `sp_rk_stage_boundary`, `sp_small_step_prep_post`) are reserved for the
  Fortran-hook-ABI follow-up sprint; they remain declared in the wrapper so
  M6B4/B5/B6 sprints can add `CALL` sites incrementally without re-extending
  the wrapper.
- Body fill-in (real HDF5 emission) is gated on the queued
  `m6b0r-fortran-hook-abi-followup` sprint per audit recommendation.
