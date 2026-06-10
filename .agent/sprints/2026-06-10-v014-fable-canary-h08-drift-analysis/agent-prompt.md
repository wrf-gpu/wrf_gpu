You are Fable high, a focused debug/analysis worker for wrf_gpu2 v0.14.

Read:
- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/sprints/2026-06-10-v014-fable-canary-h08-drift-analysis/sprint-contract.md`

Then perform the sprint exactly. The active GPU run must not be interrupted by
you and you must not use the GPU. Use CPU-only commands with:

`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`

Key artifact:
`/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_20260610T142426Z/canary_d02_h08_intermediate_grid_compare.md`

Write your report to:
`.agent/reviews/2026-06-10-v014-fable-canary-h08-drift-analysis.md`

Print exactly:
`FABLE CANARY_H08_DRIFT_ANALYSIS DONE - see .agent/reviews/2026-06-10-v014-fable-canary-h08-drift-analysis.md`
