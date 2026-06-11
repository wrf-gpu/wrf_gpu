You are Fable high working as an end-to-end debugger for `wrf_gpu2`.

Worktree/branch:

- You will be in a dedicated worktree on branch
  `worker/fable/v014-switzerland-field-parity-debug`.
- Verify `git log -1` and read
  `.agent/sprints/2026-06-11-v014-switzerland-field-parity-debug/sprint-contract.md`.

Goal:

Root-cause and close the Switzerland/Gotthard d01 72h CPU-WRF vs GPU-JAX
field-parity blocker. If the root cause is local and safe, implement the fix,
prove it, and commit. If it is not safe to fix now, produce the exact
proof-backed root-cause analysis and smallest next fix plan.

Critical artifacts:

- Switzerland GPU run root:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_gpu_20260611T001250Z`
- Switzerland CPU truth:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`
- Switzerland compare:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_gpu_20260611T001250Z/switzerland_d01_72h_grid_compare.json`
- Canary bounded comparison:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z`
- Manager Canary proof:
  `proofs/v014/canary_d02_72h_field_gate_summary.md`
- GPU runbook:
  `docs/GPU_RUNBOOK.md`

Observed Switzerland facts:

- GPU run technically completed: `gpu_rc=0`, 72 d01 forecast frames.
- Grid compare completed: `compare_rc=0`, formal verdict `FAIL`.
- Static fields are quiet: `PB` RMSE `0.000322`, `MUB` RMSE `0.000548`, both pass.
- Hard dynamic failures include `PSFC`, `T`, `RAINNC`, `U`, `V`, `U10`, `V10`,
  `T2`, `W`, `QVAPOR`.
- Pressure/mass drift is large and systematic:
  `PSFC` h1 RMSE/bias `28.98/-14.72`, h24 `757.38/632.13`,
  h48 `2161.28/2136.50`, h72 `2407.20/2380.01`.
  `MU` h72 RMSE/bias `2332.76/2300.09`, `PH` h72 `3441.65/2830.78`.
- `QNICE` and `QNRAIN` are huge early report-only signals (`QNICE` worst h4,
  `QNRAIN` worst h11), but do not assume they are the root cause until proven.
- `TSK` is exactly equal, which may help isolate a replay/physics/state path.

Rules:

- CPU-only analysis first. Use `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=` for
  probes unless a short GPU check is the fastest proof.
- Short GPU checks are allowed with `scripts/run_gpu_lowprio.sh` and resource
  logging. Do not start a new long GPU campaign.
- Do not use Hermes/TG.
- Ignore `/home/enric/src/canairy_waves` artifacts.
- Do not silently change tolerances.
- Avoid micro-reporting. Deliver the endpoint: fix+proof+commit or exact
  root-cause+next minimal fix.

Write final report to:

`.agent/reviews/2026-06-11-v014-switzerland-field-parity-debug-fable.md`

Commit your report and any code/proofs on your worker branch. End your output
with:

`FABLE SWITZERLAND_FIELD_PARITY_DEBUG DONE - see .agent/reviews/2026-06-11-v014-switzerland-field-parity-debug-fable.md`
