# Pending Memory: V0.14 Direct Grid After Live-Nest Base Proof

Status: pending promotion after Opus critic/debugger review and fresh
same-state current-code carry localization.

Lesson:

- Direct d02 h1-h12 grid comparison after the live-nest base-source fix verdict:
  `GRID_SYMPTOM_NOT_CLOSED`.
- The bounded GPU run itself is green (`L2_D02_GREEN`) and took total wall
  `1192.2986149120043` s, but the grid fields still diverge from CPU-WRF.
- Dynamic residuals remain large: `V10` RMSE `2.55039100124724`, worst h11 RMSE
  `4.277008742661733`; `PSFC` RMSE `517.1905702423264` Pa; `P` RMSE
  `230.30713670774634` Pa; `MU` RMSE `266.52491970646497`; `PH` RMSE
  `292.3872984317863`.
- Static/base payloads improved strongly, including exact C/DN/RDN/MAPFAC and
  lat/lon plus near-exact HGT/PHB, but PB/MUB are not exact and the dynamic
  symptom did not collapse.
- This proof agrees with `same_state_momentum_mass`: next source localization
  should be inside final RK pressure-gradient/mass-wind/theta-pressure coupling,
  not station TOST, output writer, or static-base plumbing.

Evidence:

- `proofs/v014/grid_after_live_nest_base.json`
- `proofs/v014/grid_after_live_nest_base.md`
- `.agent/reviews/2026-06-09-v014-grid-after-live-nest-base.md`
- `.agent/sprints/2026-06-09-v014-grid-after-base-direct/manager-closeout.md`
