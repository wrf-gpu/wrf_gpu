# Worker Report - M6 Perf PCR vs Thomas Scout

## objective

Research-only scout comparing serial Thomas, PCR, batched-Thomas, and external `gtsv` references for the WRF vertical-implicit acoustic solve on RTX 5090 / Blackwell at Canary d02 dimensions. Feed recommendation into M6-perf-design / ADR-026. No operator code was changed.

## recommendation

**BAKEOFF.** Do not switch to PCR from literature alone. Keep current batched Thomas as the operational default candidate for d02/1km until a small measured bakeoff compares:

- current JAX `lax.scan` Thomas over batched columns
- pure PCR with n=44/n=45 padding and masks
- one fixed PCR+Thomas hybrid split
- cuSPARSE/cuSolverDx `gtsv` as benchmark references only

Required bakeoff proof: HLO/cost analysis, Nsight Systems no-H2D/D2H trace, kernel/loop counts, `block_until_ready()` wall time, residuals vs Thomas on real Canary coefficients, and Tier-4 RMSE before any operational solver promotion.

## dissent

PCR may still win if XLA emits expensive scan-loop kernels or if the d01 case becomes occupancy-limited. NVIDIA's cuSolverDx uses a PCR+Thomas hybrid for no-pivot device tridiagonal solves, which is strong evidence that a hybrid can beat plain Thomas for some size/hardware combinations. The scout only rejects a blind switch; it does not reject PCR as a bakeoff candidate.

## files changed

- `.agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout/solver_comparison.md`
- `.agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout/worker-report.md`
- `.agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout/proof_collect_only.txt`

## commands run

- `pwd && rg --files -g 'PROJECT_CONSTITUTION.md' -g 'AGENTS.md' -g '.agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout/sprint-contract.md' -g '.agent/skills/**'`
- `git status --short --branch`
- `sed -n '1,220p' PROJECT_CONSTITUTION.md`
- `sed -n '1,220p' AGENTS.md`
- `sed -n '1,260p' .agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout/sprint-contract.md`
- `rg --hidden --files .agent/skills .agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout`
- `sed -n '1,240p' .agent/skills/researching-prior-art/SKILL.md`
- `sed -n '1,220p' .agent/skills/reporting-to-human/SKILL.md`
- `sed -n '1,220p' .agent/skills/profiling-nvidia-gpu/SKILL.md`
- `rg -n "14\\.5|14\\.5\\.1|14\\.5\\.2|vertical|implicit|operational" PROJECT_PLAN.md`
- `sed -n '243,286p' PROJECT_PLAN.md`
- `sed -n '1,260p' .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/worker-report.md`
- `sed -n '1,260p' src/gpuwrf/dynamics/tridiag_solve.py`
- `sed -n '1,260p' .agent/sprints/2026-05-25-m6-perf-design/sprint-contract.md`
- `rg -n "Part 3|d02|159|66|44|dimension|grid" .agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/env_audit_memo.md`
- `sed -n '72,90p' .agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/env_audit_memo.md`
- `sed -n '1,80p' data/fixtures/gen2_baseline/rmse_summary.csv`
- Web research via official/public URLs cited in `solver_comparison.md`
- `wc -w .agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout/solver_comparison.md`
- `pytest --collect-only 2>&1 | tail -3 | tee .agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout/proof_collect_only.txt`
- `git status --short`
- `git add .agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout/solver_comparison.md .agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout/worker-report.md .agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout/proof_collect_only.txt`
- `git commit -m "Add M6 PCR vs Thomas solver scout"`

## proof objects produced

- `solver_comparison.md` research memo with URL citations
- `worker-report.md`
- `proof_collect_only.txt`
- Git commit on branch `scout/codex/m6-perf-pcr-vs-thomas-scout`

## unresolved risks

- No performance measurements were run; all speedups are explicitly estimates.
- No real WRF acoustic coefficient diagonal-dominance audit was run.
- JAX Blackwell-specific `lax.scan` behavior must be measured in the repo environment; public JAX docs describe semantics and CUDA support, not this kernel's generated code.
- The contract's 1km phrase says "5.76M columns"; the memo corrects this to 5.76M cells / 96,000 columns.

## next decision needed

Manager should decide whether M6-perf-design includes a dedicated mini-bakeoff before ADR-026 solver selection, or keeps Thomas as the first operational solver and defers PCR/hybrid until a measured hotspot appears.
