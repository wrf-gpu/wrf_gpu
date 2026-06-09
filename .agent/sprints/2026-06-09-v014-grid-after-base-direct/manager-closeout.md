# Manager Closeout

## Outcome

The sprint is closed as a valid direct symptom proof with verdict
`GRID_SYMPTOM_NOT_CLOSED`.

The live-nest base-source work materially improved static/base payloads, but it
did not close the dynamic grid divergence. TOST remains paused.

## Proof Objects

- `proofs/v014/grid_after_live_nest_base.json`
- `proofs/v014/grid_after_live_nest_base.md`
- `proofs/v014/grid_after_live_nest_base.py`
- `proofs/v014/grid_after_live_nest_base/gpu_h12/`
- `.agent/reviews/2026-06-09-v014-grid-after-live-nest-base.md`

Key dynamic residuals over d02 h1-h12:

- `V10` RMSE `2.55039100124724`, worst h11 RMSE `4.277008742661733`
- `U10` RMSE `1.7111033260122948`
- `PSFC` RMSE `517.1905702423264` Pa
- `P` RMSE `230.30713670774634` Pa
- `MU` RMSE `266.52491970646497`
- `PH` RMSE `292.3872984317863`

Runtime:

- Total GPU wall: `1192.2986149120043` s
- Forecast-only wall: `1186.442607951998` s
- Device: `cuda:0`
- Peak VRAM: not recorded in committed runner artifact

## Merge Decision:

Merge proof/review artifacts only. Do not merge any release claim or TOST
resume decision from this sprint.

## Scope Changes

No production `src/` code changed. No TOST, Switzerland validation, FP32 work,
or memory source work was run.

## Lessons

The base/static bug was real but not the main closure. The direct h1-h12 grid
proof and the same-state momentum/mass proof now agree that the remaining
problem is dynamic and should be localized around final RK pressure-gradient,
mass-wind, and theta-pressure coupling.

## Next Sprint

Because two GPT debug sprints have now failed to prove closure on the same
complex grid-divergence problem, apply the manager cadence: dispatch one Opus
xhigh critic/debugger to challenge the evidence chain and propose the next
highest-yield localization/fix sprint before committing to a new root-cause
direction.
