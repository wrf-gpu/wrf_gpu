# Sprint Contract — Plan Critic (codex; user-mandated)

**Sprint ID**: `2026-05-22-plan-critic`
**Created**: 2026-05-22 ~00:00
**Status**: ACTIVE
**Trigger**: User directive 2026-05-21 ~23:35: "Periodically let your plan be criticized by gpt so that you are sure you still have the fastest and smartest path to the end goal."

## Objective

Independent codex critic on the manager's current project plan. Answer: is the manager on the **fastest, smartest path to the constitutional end goal** (Canary Islands 3km/1km daily operational forecast, ≥4× faster than 28-rank CPU WRF on RTX 5090, GPU-native, WRF-compatible-where-useful)?

## Acceptance

- **AC1 End-goal alignment**: read `PROJECT_CONSTITUTION.md`. Restate the end goal in your own words. Critique whether the current sprint state moves towards or away from it.
- **AC2 Critical-path correctness**: read the live sprint state (M6.x dycore in flight, M6.5-D1 closed, M6-S8 drafted, M6.x contingency archived, M7 plan in repo). Is the manager's identified critical path (M6.x → M6-S8 → M7) correct, or is there a faster path the manager missed?
- **AC3 Over-engineering audit**: identify any work in the last 5 sprints that did NOT contribute to the end goal. Should it have been done at all? Should it have been done smaller?
- **AC4 Missing work**: what's NOT being done that should be? Examples: AIFS-LAM data pipeline staging, operational scheduler, monitoring/alerting, output format compatibility, post-processing pipeline, validation against actual operational metrics.
- **AC5 Architectural risk audit**: M4 dycore is a reduced proxy. M6.x is completing it. The contingency designer scoped c1/c2/c3 backup options. Are there other architectural risks the manager has missed? (e.g., AIFS BC quality, sub-grid orography for 1km nest, sea-surface temperature treatment, ensemble-mode dispatch, hardware bottlenecks)
- **AC6 Next 2-3 sprints recommendation**: based on AC1-AC5, recommend the next 2-3 sprints in dispatch order. Be specific (sprint name, owner, wall, AC sketch).
- **AC7 Decision-quality audit**: review the manager's last 3 closeouts (M6-S5, M6-S6, M6.5-D1). Are the decisions reasonable? Any rejections or accepts that should have been the opposite?

## Hard rules

- Read-only critique
- No code
- Cite file:line for every concrete finding
- Be direct — manager wants criticism, not validation
- Be honest about uncertainty — say "I don't know" when applicable
- BEFORE `/exit`: `git add . && git commit && git push`

## Deliverable

`.agent/sprints/2026-05-22-plan-critic/critique-report.md` ≥3500 bytes covering AC1-AC7.

## Dispatch

- Worker: codex gpt-5.5 xhigh
- No reviewer (critic IS the review)
- Wall: 1-2h
- Worktree: `/tmp/wrf_gpu2_plancrit`
- Branch: `worker/codex/plan-critic-2026-05-21`

## End-goal context (the question this sprint is asking)

> Am I, the manager, driving this project on the fastest+smartest path to Canary 3km daily operational? If not, what's the better path?
