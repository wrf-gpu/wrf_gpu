# Sprint Contract: V014 Switzerland Post-LBC Residual

Date: 2026-06-11
Assignee: Fable high
Manager: Codex

## Objective

Root-cause and close the remaining Switzerland/Gotthard d01 72h field-parity
residual after the single-domain LBC clock fix. If the root cause is local and
safe, implement the fix, prove it, and commit on the worker branch. If not,
produce an exact proof-backed analysis and the smallest next fix plan.

This is an end-to-end task. Do not stop at "inspect QNRAIN"; determine whether
the residual is a real bug, a bounded physics/chaos class, a validation-policy
issue, or another driver/coupling defect.

## Starting Point

Merged fix:

- branch/head: `worker/gpt/v013-close-manager` /
  `eaff102c6f46a3863faa787f018975317b850823`
- proof:
  - `.agent/reviews/2026-06-11-v014-switzerland-field-parity-debug-fable.md`
  - `proofs/v014/switzerland_lbc_clock_root_cause.{py,json,md}`

Fixed Switzerland 72h rerun:

- run root: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_lbcclockfix_20260611T020428Z`
- CPU truth: `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`
- GPU output: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_lbcclockfix_20260611T020428Z/gpu_output`
- compare JSON: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_lbcclockfix_20260611T020428Z/switzerland_d01_72h_grid_compare.json`
- compare MD: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_lbcclockfix_20260611T020428Z/switzerland_d01_72h_grid_compare.md`
- GPU rc: `0`
- compare rc: `0`
- formal verdict: `FAIL`

Important fact already checked by the manager: the fixed run's `MU` boundary
ring now matches same-hour CPU truth exactly at h1/h2/h3/h6/h12/h18/h24/h30/
h36/h42/h48/h54/h60/h66/h72. The post-fix residual is not another frozen-LBC
clock bug.

## Observed Post-Fix Failure

The old +2380 Pa h72 pressure drift is much smaller but not fully gone. The
dominant formal failures now are:

- `RAINNC`: RMSE `5.833`, p99 `20.610`, worst h72.
- `PSFC`: RMSE `350.772 Pa`, bias `-171.041 Pa`, p99 `1347.996 Pa`,
  worst h72.
- `U`: RMSE `4.056`, p99 `15.647`, worst h72.
- `T`: RMSE `2.791 K`, p99 `10.578 K`, worst h72.
- `U10`: RMSE `2.619`, p99 `8.912`, worst h72.
- `V`: RMSE `2.977`, p99 `12.543`, worst h72.
- `V10`: RMSE `2.319`, p99 `8.280`, worst h72.
- `W`: RMSE `0.431`, p99 `1.571`, worst h72.
- `QVAPOR` now passes: RMSE `0.000594`.
- `T2` now passes: RMSE `1.315`.
- `PB/MUB` pass at bit-noise level.
- `TSK` is exactly equal; this is expected from hourly replay land refresh.

Large report-only fields:

- `QNRAIN`: RMSE `57140`, p99 `1556`, max `4.65178e7`, worst h19.
- `QNICE`: RMSE `57716`, p99 `273377`, max `2.08085e6`, worst h4.
- `PH`: RMSE `610.454`, bias `348.803`, p99 `2044.201`, worst h72.
- `MU`: RMSE `366.225`, bias `-192.788`, p99 `1380.543`, worst h72.
- `P`: RMSE `226.865`, bias `-131.537`, p99 `997.901`, worst h72.

Selected lead evolution:

- `PSFC`: h1 `28.98/-14.72`, h24 `29.34/-11.39`, h36 `37.91/7.63`,
  h42 `235.06/-187.40`, h60 `378.77/-334.43`, h72 `1119.66/-1029.85`.
- `MU`: h1 `27.95/-12.50`, h24 `53.71/-41.26`, h36 `48.65/-14.67`,
  h42 `260.63/-207.83`, h60 `407.95/-358.81`, h72 `1145.45/-1052.97`.
- `PH`: h1 `43.22/-7.40`, h24 `223.71/142.36`, h36 `458.16/302.43`,
  h42 `582.71/423.81`, h72 `1226.26/940.36`.
- `RAINNC`: h1 `0.003`, h12 `2.673`, h24 `5.061`, h48 `7.078`,
  h72 `8.002`.
- `QNRAIN` has huge sparse/displaced bursts at h18/h24/h66/h72.

Canary reference:

- Canary d02 72h is manager-adjudicated bounded:
  `proofs/v014/canary_d02_72h_field_gate_summary.md`.
- Canary has QNICE/QNRAIN report-only sparse signals too, but hard dynamic
  fields mostly pass except marginal/saturating `QVAPOR`.

## Required Method

1. Read the fixed compare JSON/MD, the LBC-clock proof, Canary bounded proof,
   and the relevant daily-pipeline/coupling/physics code.
2. Determine the earliest causal lead and field class. The formal h72 failures
   may be symptoms of an earlier precipitation/microphysics/PBL/radiation or
   diagnostic mismatch.
3. Explicitly falsify these candidate classes:
   - second LBC/boundary issue despite correct `MU` ring,
   - replay/land-refresh artifact (`TSK` exact, but other land/soil/snow fields
     may differ),
   - microphysics number-concentration explosion as root cause vs downstream
     displaced precipitation,
   - MYNN/PBL or radiation daytime residual accumulating after h36,
   - writer-only diagnostic issue vs prognostic state issue,
   - tolerance-policy issue/chaotic but bounded regime.
4. Use CPU-only probes first. Short GPU checks are allowed if they are the
   fastest proof path; no new long GPU campaign without manager review.
5. If a local fix is found, implement and prove it with the shortest meaningful
   gate.

## Acceptance Gate

One of:

- **Fix accepted**: source change plus proof object showing a short fixed
  Switzerland gate collapses the residual enough to justify a new full 72h
  rerun.
- **Bounded/adjudicated accepted**: no source change, but a rigorous proof that
  the residual is bounded/physical/chaotic under the release policy, with exact
  recommended tolerance/report treatment.
- **Root cause accepted**: exact subsystem/file/function identified with the
  next smallest fix sprint.

Output report:

`.agent/reviews/2026-06-11-v014-switzerland-post-lbc-residual-fable.md`

Commit any code/proof/report changes on the worker branch. Include objective,
files changed, commands run, proof objects, unresolved risks, and next decision.
