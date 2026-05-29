# F7-B Operational-dt Audit Summary (Block 1 damping)

Config: Gen2 d02 replay state, dt = 6 s, 4 sound steps, epssm = 0.5, combination
`a` (pure dycore, guards off), `taskset -c 0-3`, `cuda:0`, fp64.

WRF damping enabled (Gen2 d02 namelist coefficients):
- `w_damping = 1` — vertical-CFL limiter on the large-step `rw_tend`
  (`module_big_step_utilities_em.F:2714-2774`; `w_alpha=0.3`, `w_beta=1.0`).
- `damp_opt = 3` — implicit Rayleigh top-damping, `dampcoef=0.2`, `zdamp=5000 m`
  (`module_small_step_em.F:1559-1572`).
- divergence damping (verified active from Sprint A): `smdiv=0.1` pressure memory
  in `calc_p_rho`, `emdiv=0.01`/`mudf` external-mode in `advance_uv`.

## Result — physical-state max magnitudes per step (rk_step_start row)

| step | u_abs (m/s) | v_abs (m/s) | w_abs (m/s) |
| ---- | ----------- | ----------- | ----------- |
| 1 | 25.66 | 11.48 | 0.93 |
| 2 | 25.65 | 11.49 | 17.45 |
| 3 | 25.65 | 11.50 | 16.94 |
| 4 | 25.65 | 11.50 | 17.16 |
| 5 | 25.65 | 11.50 | 16.93 |
| 6 | 25.65 | 25.12 | 33.50 |
| 7 | 107.4 | 110.7 | 232.2 |
| 8 | blow-up | blow-up | blow-up |

`first_critical_violation` = `theta_sanity_bounds` at step 6 / RK3
(min θ = 198.9 K, just below the audit's conservative 200 K floor) — i.e. the
first flagged issue is a marginal theta excursion, NOT the acoustic ringing.

## Honest assessment vs AC3

- **Major improvement over Sprint A.** Sprint A (dampers off) detonated by step 4
  with `w ~ 1.5e4` m/s; with WRF damping ON the transients are **physical**
  (`w ~ 17 m/s`, `u ~ 25 m/s`) for the first ~5 steps — exactly the O(≤100 m/s)
  AC3 magnitude target, with no clamp/limiter (guards off).
- **AC3 not fully met.** A growing mode still escapes around step 6-7 and
  detonates by step 8, so `first_critical_violation` is not null over 12 steps.
  The residual growth is the missing operational numerical filter; the WRF
  6th-order monotonic diffusion (`diff_6th_opt=2`, `diff_6th_factor=0.12`) is
  implemented (`explicit_diffusion.py`) and wired into the audit
  (`--diff-6th-opt`), see `audit_operational_dt_diff6/`. Constant-K / Smagorinsky
  (`km_opt=4`) is not yet wired.

No masking clamp/cap/sanitizer was used; all damping is the WRF physics named
above with the cited coefficient sources.
