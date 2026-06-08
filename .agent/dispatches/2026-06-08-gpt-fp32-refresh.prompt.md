# GPT-5.5 xhigh Dispatch: FP32 Acoustic Feasibility Refresh for v0.14

You are a GPT-5.5 xhigh worker for `/home/enric/src/wrf_gpu2`.

Read in this order:
1. `/home/enric/src/wrf_gpu2/PROJECT_CONSTITUTION.md`
2. `/home/enric/src/wrf_gpu2/AGENTS.md`
3. `/home/enric/src/wrf_gpu2/.agent/skills/managing-sprints/SKILL.md`
4. `/home/enric/src/wrf_gpu2/.agent/skills/validating-physics/SKILL.md`
5. The prior GPT FP32 report:
   `/home/enric/src/wrf_gpu2/.claude/worktrees/gpt-fp32/.agent/reviews/2026-06-08-gpt-fp32-acoustic-feasibility.md`

Do not use the old global `wrf-gpu-port` skill. The repo-local rules are authoritative.

Context:
- The manager's v0.13 goal excludes FP32 acoustic productionization.
- If FP32 acoustic is theoretically possible, it should be put on a v0.14 roadmap only.
- The prior report concluded it is feasible via a perturbation/mixed-precision formulation, not a mathematical impossibility, but it was based on `237aceb5`.
- You must refresh this against the current v0.13 code/roadmap and all finished modules for compatibility implications.
- Orientation is critical: your worktree MUST be based on `worker/gpt/v013-close-manager` / `worker/opus/v0120-integration @237aceb5`. Verify `git log -1` before analysis. Do not treat stale branches such as `worker/opus/v013-t3-pbl @7fd92fd2` as current.

Task:
1. Validate/refute the prior conclusion: can an fp32 or mixed fp32 acoustic kernel avoid acoustic blow-up on GPU/JAX in principle?
2. If feasible, provide a v0.14 roadmap with staged proof gates and compatibility notes for current finished modules: dycore, nesting/two-way feedback, GWD, RRTMG tiling/clear-sky, MYNN/MYJ/Janjic, moisture advection, TOST, release docs.
3. If impossible, explain exactly what mathematical or JAX/GPU constraint makes it impossible.
4. Do not implement code. Do not run GPU jobs.

Required output:
- Write `.agent/reviews/2026-06-08-gpt-fp32-acoustic-refresh.md` in your worktree.
- Final line in the tmux pane must include: `GPT FP32 REFRESH DONE`.

Report format:
- objective
- files changed
- commands run
- verdict
- v0.14 roadmap if feasible
- compatibility matrix with current modules
- explicit v0.13 non-impact statement
- validation gates
- unresolved risks
- next decision needed, if any
