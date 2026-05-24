# Worker Report - M6.x Option E Shadow Scout

## Objective

Evaluate AceCAST and FahrenheitResearch/wrf-gpu-port as public-research-only shadow GPU-WRF benchmark candidates for Option E, without implementation, builds, purchases, dependency adoption, or remote push.

## Recommendation

`RECOMMEND-DEFER-E-LANE`

Neither candidate is ready to authorize as an execution lane from public evidence alone. AceCAST is the more credible operational product, with vendor support, validation framing, current releases, and strong A100 performance evidence, but the public terms do not include pricing, the latest public compatibility is WRF 4.6.0 rather than Canairy's WRF 4.7.1, and source-level/savepoint instrumentation access is unknown. FahrenheitResearch/wrf-gpu-port matches WRF 4.7.1 and RTX 5090/sm_120, but the README documents CPU physics, disabled GPU advection due to instability, CPU/GPU transfers at dynamics/physics boundaries, single-GPU-only status, no releases, and no public validation artifact beyond short-run claims.

What would change the calculus: AceCAST provides written license/cost terms, WRF 4.7.1 support or an acceptable 4.6.0 compatibility plan, explicit support for the Canairy namelist/physics stack, and permission to instrument savepoints; or wrf-gpu-port publishes reproducible build logs, validation artifacts against CPU WRF 4.7.1 for a real Canairy-sized case, transfer audits, and a stable multi-GPU/advection/physics roadmap.

## Dissent

The best argument against deferral is that Option E is explicitly business-continuity insurance, not primary architecture. On that basis, a narrow AceCAST evaluation could be justified now despite unknown commercial terms: request a 30-day trial, run the Advisor support check on the Canairy namelist, and stop immediately if WRF 4.7.1/version skew or licensing blocks the case. This would buy information quickly and might reveal an operational fallback before B-direct finishes.

## Savepoint-Harness Secondary Oracle Assessment

AceCAST: no. If AceCAST savepoints diverge from CPU WRF, it should not become a secondary oracle for M6B0. Public docs frame AceCAST validation as matching CPU WRF within tolerance, not as an independently validated physical reference, and the closed commercial implementation makes divergence attribution opaque unless TempoQuest grants source-level or instrumented savepoint access.

wrf-gpu-port: no. If wrf-gpu-port savepoints diverge from CPU WRF, it is evidence to debug that port, not a complementary oracle. Its public architecture leaves advection and all physics on CPU, has CPU/GPU updates at dynamics/physics boundaries, disables GPU advection for instability, and has no published savepoint-level validation artifacts.

## Files Changed

- `.agent/sprints/2026-05-24-m6x-option-e-shadow-scout/shadow_comparison.md`
- `.agent/sprints/2026-05-24-m6x-option-e-shadow-scout/worker-report.md`

## Commands Run

- `pwd && rg --files ...`
- `git status --short --branch`
- `sed -n ... PROJECT_CONSTITUTION.md`
- `sed -n ... AGENTS.md`
- `sed -n ... .agent/sprints/2026-05-24-m6x-option-e-shadow-scout/sprint-contract.md`
- `sed -n ... .agent/skills/researching-prior-art/SKILL.md`
- `sed -n ... .agent/skills/reporting-to-human/SKILL.md`
- `sed -n ... .agent/decisions/manager-reflections/PLAN-REFLECTION-2026-05-24-post-consultation.md`
- `sed -n ... .agent/decisions/blockers/M6-DYCORE-BLOCKER-MEMO.md`
- `sed -n ... .agent/references/cpu-wrf-baseline.md`
- `sed -n ... shadow_comparison.md`
- `sed -n ... worker-report.md`
- `git diff -- .agent/sprints/2026-05-24-m6x-option-e-shadow-scout/...`
- Web searches/opens for AceCAST docs, FahrenheitResearch/wrf-gpu-port, NVIDIA CUDA compute capability, WRF 4.7.1, WRF license, and NCAR/WRF forum GPU discussion.

## Proof Objects Produced

- `.agent/sprints/2026-05-24-m6x-option-e-shadow-scout/shadow_comparison.md`
- `.agent/sprints/2026-05-24-m6x-option-e-shadow-scout/worker-report.md`

## Unresolved Risks

- AceCAST pricing, final license terms, source/instrumentation rights, and Canairy namelist support are not public.
- AceCAST public latest WRF compatibility is WRF 4.6.0; Canairy is WRF 4.7.1.
- wrf-gpu-port performance and correctness claims are not independently reproduced here because the sprint forbids builds.
- wrf-gpu-port's CPU physics and host/device transfers conflict with the project's GPU-residency performance direction if promoted beyond shadow research.

## Next Decision Needed

Decide whether to keep Option E deferred, or authorize a tightly bounded commercial-discovery sprint for AceCAST only: license/cost request, Canairy namelist support check, WRF 4.7.1 compatibility question, and savepoint-instrumentation feasibility.
