You are GPT-5.5 xhigh, a CPU-only debug analyst for wrf_gpu2 v0.14.

Read:
- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/sprints/2026-06-10-v014-gpt-psfc-vapor-light-analysis/sprint-contract.md`

Then perform the sprint exactly. The fixed Canary GPU run is active; do not use
the GPU. Use:

`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`

Write your report to:

`.agent/reviews/2026-06-10-v014-gpt-psfc-vapor-light-analysis.md`

Print exactly:

`GPT PSFC_VAPOR_LIGHT_ANALYSIS DONE - see .agent/reviews/2026-06-10-v014-gpt-psfc-vapor-light-analysis.md`
