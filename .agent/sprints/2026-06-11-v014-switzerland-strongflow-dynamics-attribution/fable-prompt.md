You are Fable high, debugging wrf_gpu2 v0.14. You are in a dedicated worktree
on branch `worker/fable/v014-switzerland-strongflow-dynamics`, based on current
manager HEAD. First verify `git log -1`.

Read this sprint contract completely:

`.agent/sprints/2026-06-11-v014-switzerland-strongflow-dynamics-attribution/sprint-contract.md`

Goal: close the Switzerland d01 post-LBC strong-flow dry-dynamics mass-venting
release blocker end to end. Do not do a narrow micro-analysis. Either implement
and prove the local fix, or return a WRF-anchored exact root-cause proof that
lets the manager dispatch one final implementation sprint.

Important current state:

- Canary d02 72h is bounded/proceed.
- Switzerland d01 had an LBC clock bug; it is fixed and merged.
- The remaining Switzerland residual is NOT LBC, NOT microphysics, NOT writer,
  NOT accumulated chaos. Existing Fable proof says it is locally generated dry
  dynamics in the h36-h72 strong cross-Alpine flow regime, with roughly
  30-50 Pa/cell/h excess dry-mass venting from the GPU's own winds.
- Prime suspect is `top_lid=True` rigid top versus WRF's free/constant-pressure
  top, but do not anchor on that if the evidence points elsewhere.

Use short h36 storm-state A/B and term attribution. Avoid long 72h runs. GPU
short runs are allowed; coordinate by using only one GPU job at a time and
writing resource/log roots. If you edit model code, preserve the GPU-native
architecture: no host transfers in timestep loops, no clamps/masking as a
substitute for WRF-faithful dynamics.

Required output and acceptance gates are in the sprint contract. Commit your
report, proof object(s), and any fix to your worker branch. Then print exactly:

`FABLE SWITZERLAND_STRONGFLOW_DYNAMICS DONE - see .agent/reviews/2026-06-11-v014-switzerland-strongflow-dynamics-fable.md`

