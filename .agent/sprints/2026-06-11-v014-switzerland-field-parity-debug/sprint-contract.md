# Sprint Contract: V014 Switzerland d01 Field-Parity Blocker

Date: 2026-06-11
Assignee: Fable high
Manager: Codex

## Objective

Root-cause and close the Switzerland/Gotthard d01 72h CPU-WRF vs GPU-JAX
field-parity blocker. If the root cause is local and safe, implement the fix,
produce proof objects, and commit on the worker branch. If it is not safe to
fix in this sprint, produce an exact, evidence-backed analysis with the smallest
next fix plan and no speculative hand-waving.

This is an end-to-end task, not a micro-run. Use the existing artifacts first,
then build the shortest probes needed to prove or falsify hypotheses.

## Current State

Canary L2 d02 72h has completed and is manager-adjudicated as
`PROCEED_BOUNDED_WITH_FOLLOWUP`:

- run root: `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z`
- proof: `proofs/v014/canary_d02_72h_field_gate_summary.md`
- formal misses: static d02 `MUB/PB` boundary-frame seam plus saturating
  `QVAPOR`; no runaway and all hard dynamic fields except `QVAPOR` pass.

Switzerland/Gotthard d01 72h has completed technically but failed the field
gate:

- run root: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_gpu_20260611T001250Z`
- CPU truth: `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`
- GPU output: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_gpu_20260611T001250Z/gpu_output`
- branch/head at launch: `worker/gpt/v013-close-manager` /
  `b7fb4cd84f7e53ab300bd86821aac2a8bbba6865`
- GPU rc: `0`
- grid compare rc: `0`
- compare JSON: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_gpu_20260611T001250Z/switzerland_d01_72h_grid_compare.json`
- compare MD: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_gpu_20260611T001250Z/switzerland_d01_72h_grid_compare.md`

Important contrast: Switzerland static fields are quiet. `PB` and `MUB` pass at
bit-noise level, unlike the Canary d02 boundary-frame seam.

## Observed Switzerland Failure

The comparator verdict is `FAIL` with 10 hard tolerance failures:

- `PSFC`: RMSE `1704.224 Pa`, bias `1401.175 Pa`, p99 `2962.680 Pa`,
  worst lead h63, positive drift.
- `T`: RMSE `10.805 K`, p99 `24.725 K`, worst h72.
- `RAINNC`: RMSE `5.368 mm`, p99 `21.175 mm`, worst h72.
- `V`: RMSE `9.598 m/s`, p99 `26.658 m/s`, worst h72.
- `U`: RMSE `8.215 m/s`, p99 `23.587 m/s`, worst h72.
- `V10`: RMSE `3.330 m/s`, p99 `9.634 m/s`, worst h16.
- `T2`: RMSE `2.712 K`, p99 `6.351 K`, worst h56.
- `U10`: RMSE `2.547 m/s`, p99 `7.919 m/s`, worst h66.
- `W`: RMSE `0.329 m/s`, p99 `1.286 m/s`, worst h9.
- `QVAPOR`: RMSE `0.001082`, p99 `0.003234`, worst h30.

Large report-only fields:

- `QNICE`: RMSE `49358.6`, p99 `218954`, max `2.08085e6`, worst h4.
- `QNRAIN`: RMSE `174.55`, p99 `269.45`, max `7925.73`, worst h11.
- `PH`: RMSE `2364.06`, bias `1711.24`, p99 `5634.47`, worst h72.
- `P`: RMSE `825.03`, bias `405.08`, p99 `2587.80`, worst h63.
- `MU`: RMSE `1646.49`, bias `1335.93`, p99 `2916.72`, worst h63.

Selected lead evolution:

- `PSFC` RMSE/bias: h1 `28.98/-14.72`, h12 `402.68/150.66`,
  h24 `757.38/632.13`, h48 `2161.28/2136.50`, h72 `2407.20/2380.01`.
- `MU` RMSE/bias: h1 `27.95/-12.50`, h24 `714.71/555.57`,
  h48 `2084.15/2054.60`, h72 `2332.76/2300.09`.
- `PH` RMSE/bias: h1 `43.22/-7.40`, h24 `1719.36/1341.30`,
  h48 `2868.81/2339.81`, h72 `3441.65/2830.78`.
- `T` RMSE: h1 `0.2865`, h12 `5.893`, h24 `10.555`, h72 `14.158`.
- `TSK` is exactly equal over the compared outputs, which is suspicious and
  may be useful for isolating whether the replay/physics coupling differs from
  the Canary live-nested path.

## Constraints

- Prefer CPU-only analysis first.
- Short GPU confirmation runs are allowed if they are the fastest proof path
  and use `scripts/run_gpu_lowprio.sh` with resource logging. Do not start a
  new long GPU campaign without manager review.
- Do not silently relax tolerances.
- Do not edit unrelated source or docs.
- Do not use Hermes/TG.
- Ignore any `/home/enric/src/canairy_waves` reports; they belong to another
  project.

## Required Method

1. Read the run roots, compare JSON/MD, `docs/GPU_RUNBOOK.md`, the CLI runner
   path, and the relevant `gpuwrf.cli run` / CPU-WRF replay initialization code.
2. Identify the earliest lead and first causal field/class, not just the h72
   symptom.
3. Compare Switzerland with the now-bounded Canary result to isolate whether
   this is:
   - CPU-WRF replay input-root / stale runner issue,
   - single-domain d01 initialization or boundary update issue,
   - physics selection/coupling mismatch,
   - missing or stale persisted state in the replay path,
   - diagnostic writer issue,
   - or a true kernel/numerics bug.
4. Build probes that directly falsify the top hypotheses. Favor high-signal
   probes over many slow reruns.
5. If a local fix is found, implement it with focused tests/proofs and commit
   on the worker branch.
6. If no fix is safe within the sprint, write the exact next action with the
   smallest required proof run.

## Acceptance Gate

One of:

- **Fix accepted**: a source change plus proof object showing the Switzerland
  h1/hN short gate or existing-output recompare closes the blocker, and a clear
  command for the manager to rerun the 72h gate.
- **Root cause accepted**: no source change, but a proof-backed report that
  identifies the precise failing subsystem/file/function and the next minimal
  fix sprint.

Output report:

`.agent/reviews/2026-06-11-v014-switzerland-field-parity-debug-fable.md`

Commit any code/proof/report changes on the worker branch. Include a concise
handoff: objective, files changed, commands run, proof objects, unresolved
risks, and next decision.
