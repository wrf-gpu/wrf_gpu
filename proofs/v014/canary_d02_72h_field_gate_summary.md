# V0.14 Canary d02 72h Field-Parity Gate Summary

Date: 2026-06-11 01:18 WEST
Owner: manager

## Run

- run root: `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z`
- CPU truth: `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- GPU output: `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
- branch/head at launch: `worker/gpt/v013-close-manager` / `5c2422acde15d7bf49ed1179877c56ae6f8bb8c8`
- command proof: `launch_env.txt` in the run root
- GPU rc: `0`
- grid compare rc: `0`
- paired leads: 72 hourly d02 outputs, h1 through h72

## Verdict

`PROCEED_BOUNDED_WITH_FOLLOWUP`.

The comparator verdict is formally `FAIL` because the frozen tolerance manifest
flags three fields: `MUB`, `PB`, and `QVAPOR`. This is not a stop signal for
the next release-gate run because the failures match the bounded h24
adjudication class and no new runaway or nonfinite class appeared.

## Evidence

- all compared fields have `finite_pair_fraction = 1.0`
- hard manifest fields passing at h72: `PSFC`, `RAINNC`, `T`, `T2`, `U`,
  `U10`, `V`, `V10`, `W`
- `MUB`/`PB` remain static and boundary-frame-confined:
  - `MUB` overall RMSE `9.27595`, max `250.672 Pa`
  - `MUB` interior-excluding-5-cell-frame RMSE `2.96e-4 Pa`, max `0.0078125 Pa`
  - `PB` overall RMSE `4.52089`, max `249.883 Pa`
  - `PB` interior-excluding-5-cell-frame RMSE `3.63e-4 Pa`, max `0.0078125 Pa`
- `QVAPOR` is the only hard dynamic miss:
  - overall RMSE `0.001452` vs manifest limit `0.001`
  - h24 RMSE `0.00129288`, h42 `0.00169938`, h56 worst `0.00175231`,
    h72 `0.00173738`
  - the signal saturates rather than amplifies day-over-day
- diurnal dynamic peaks are bounded:
  - `PSFC` peak RMSE falls from h18 `102.154 Pa` to h42 `52.601 Pa` and h66
    `17.037 Pa`
  - `P` peak RMSE falls from h18 `80.987 Pa` to h42 `51.889 Pa` and h66
    `20.432 Pa`
  - `LH` daytime peak rises modestly from h20 `163.726` to h44 `171.259` and
    h68 `186.875`, below the 1.3x h24-adjudication alarm threshold
  - `HFX` daytime peak falls from h20 `114.951` to h44 `97.453` and h68 `97.415`
  - `SWDOWN` daytime peak falls from h21 `92.534` to h45 `29.361` and h68
    `22.107`
- report-only precipitation-number residuals remain sparse/displaced-cell
  signals:
  - `QNICE` overall RMSE `540.042`, p99 `0`, max `529758.688`, worst h31
  - `QNRAIN` overall RMSE `8.419`, p99 `0`, max `3122.397`, worst h11
  - `RAINNC` mass field passes, RMSE `0.077835`

## Resources

- GPU forecast resource wall: `8244 s` from first to last resource sample
- total GPU memory peak: `21108 MiB / 32607 MiB`
- monitored GPU process RSS peak: `20950.1 MiB`
- GPU memory returned to idle after run (`2603 MiB` total at final sample)
- resource CSVs:
  - `resources/v014_canary_d02_72h_noahmp_lu16fix_gpu_usage.csv`
  - `resources/v014_canary_d02_72h_noahmp_lu16fix_process_usage.csv`
  - `resources/v014_canary_d02_72h_noahmp_lu16fix_system_memory.csv`

## Follow-Up

- Start the Switzerland/Gotthard 72h GPU field gate while the Canary
  boundary-frame `MUB/PB` seam is followed up separately.
- Do not silently relax tolerances. The final release must record either a
  targeted boundary-frame static-field policy decision or a closure fix.
- `QVAPOR` needs explicit final tolerance-policy/adjudication before tag if the
  same envelope appears in Switzerland or in the final Atlas dashboard.
