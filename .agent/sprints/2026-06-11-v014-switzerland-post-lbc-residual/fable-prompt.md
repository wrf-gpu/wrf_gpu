You are Fable high working as an end-to-end debugger for `wrf_gpu2`.

Worktree/branch:

- You will be in a dedicated worktree on branch
  `worker/fable/v014-switzerland-post-lbc-residual`.
- Verify `git log -1` and read
  `.agent/sprints/2026-06-11-v014-switzerland-post-lbc-residual/sprint-contract.md`.

Goal:

Root-cause and close the remaining Switzerland/Gotthard d01 72h field-parity
residual after the LBC-clock fix. This is not the old frozen-boundary bug: the
fixed run's `MU` boundary ring matches same-hour CPU truth exactly through h72.

Critical artifacts:

- fixed Switzerland 72h run root:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_lbcclockfix_20260611T020428Z`
- CPU truth:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`
- fixed compare:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_lbcclockfix_20260611T020428Z/switzerland_d01_72h_grid_compare.json`
- previous LBC-clock proof:
  `proofs/v014/switzerland_lbc_clock_root_cause.{py,json,md}`
- Canary bounded proof:
  `proofs/v014/canary_d02_72h_field_gate_summary.md`

Observed post-fix facts:

- GPU rc `0`, compare rc `0`, formal verdict `FAIL`.
- `PB/MUB` pass at bit-noise level.
- `QVAPOR` and `T2` now pass.
- Failures are concentrated in h42-h72 dynamic fields plus precipitation:
  `PSFC` h72 RMSE/bias `1119.66/-1029.85 Pa`,
  `MU` h72 `1145.45/-1052.97 Pa`,
  `PH` h72 `1226.26/940.36`,
  `T` h72 `7.127 K`,
  `U10/V10` h72 `5.435/5.122 m/s`,
  `RAINNC` h72 `8.002 mm`.
- `QNRAIN/QNICE` have huge sparse/report-only bursts; do not assume root cause
  until proven.
- `TSK` is exactly equal due replay land refresh.

Rules:

- CPU-only analysis first. Use `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=` for
  probes unless a short GPU check is the fastest proof.
- Short GPU checks are allowed through `scripts/run_gpu_lowprio.sh`; no long GPU
  campaign without manager review.
- Do not use Hermes/TG.
- Ignore `/home/enric/src/canairy_waves` artifacts.
- Do not silently change tolerances.
- Deliver an endpoint: fix+proof+commit, bounded adjudication with evidence, or
  exact root cause plus smallest next fix plan.

Write final report to:

`.agent/reviews/2026-06-11-v014-switzerland-post-lbc-residual-fable.md`

Commit your report and any code/proofs on your worker branch. End your output
with:

`FABLE SWITZERLAND_POST_LBC_RESIDUAL DONE - see .agent/reviews/2026-06-11-v014-switzerland-post-lbc-residual-fable.md`
