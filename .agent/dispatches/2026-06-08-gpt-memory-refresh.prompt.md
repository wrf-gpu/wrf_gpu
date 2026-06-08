# GPT-5.5 xhigh Dispatch: v0.13 Memory Refresh / Fix Selection

You are a GPT-5.5 xhigh worker for `/home/enric/src/wrf_gpu2`.

Read in this order:
1. `/home/enric/src/wrf_gpu2/PROJECT_CONSTITUTION.md`
2. `/home/enric/src/wrf_gpu2/AGENTS.md`
3. `/home/enric/src/wrf_gpu2/.agent/skills/managing-sprints/SKILL.md`
4. `/home/enric/src/wrf_gpu2/.agent/skills/profiling-nvidia-gpu/SKILL.md`
5. `/home/enric/src/wrf_gpu2/.agent/skills/writing-gpu-kernels/SKILL.md`
6. The prior GPT memory report:
   `/home/enric/src/wrf_gpu2/.claude/worktrees/gpt-mem-map/.agent/reviews/2026-06-08-gpt-analytic-memory-map.md`

Do not use the old global `wrf-gpu-port` skill. The repo-local rules are authoritative.

Context:
- The old Opus manager ran out of tokens after a memory-fix wave.
- A prior GPT memory map exists at branch `worker/gpt/v013-mem-map @7ce31a6b`, based on `237aceb5`.
- Later/parallel branches include:
  - `worker/opus/v013-2way-vram @8de39fd9`
  - `worker/opus/v013-compile-perf @92fc12f8`
  - `worker/opus/v013-skill-closure @25ab8d3e`
  - `worker/opus/v013-t3-microphysics @8714f1b4`
  - `worker/opus/v013-t3-wdm5 @81f1d0fb`
  - `worker/opus/v013-t3-cumulus @b8fefb3a`
  - `worker/opus/v013-t3-radiation @deadabd8` and `worker/opus/v013-t3-gsfc-lw @b8f58740`
  - `worker/opus/v013-t3-surface-lsm @ae105488`
- The manager checkout was reset after handoff to `worker/gpt/v013-close-manager`, based on `worker/opus/v0120-integration @237aceb5`, because `worker/opus/v013-t3-pbl @7fd92fd2` is an ancestor/stale branch.
- Do not assume branch names alone imply freshness; verify `git log -1`, merge-bases, and evidence files before classifying memory issues.
- TOST is time critical, but must wait only until no remaining memory fix can invalidate its pipeline validity.

Task:
1. Reconcile the prior memory map against the current v0.13 branch/parallel branches above.
2. Identify all "solvable" memory issues that should be part of v0.13, excluding FP32 acoustic (v0.14 only).
3. Categorize each issue:
   - MUST-FIX-BEFORE-TOST if it can change whether TOST fits/runs or materially changes memory efficiency in the TOST pipeline.
   - SAFE-NOW if bit-identical/default-inert and can be merged before/while TOST without invalidating correctness.
   - V0.14 if it changes validated dycore/physics semantics, needs new GPU correctness validation, or is not necessary for v0.13 closure.
4. If you find exactly one high-confidence SAFE-NOW memory fix that is small, disjoint, and provably bit-identical, you may implement it in your worktree and produce a proof object. Do not touch broad dycore or radiation kernels without a sprint contract. Do not consume the GPU unless the manager explicitly tells you.
5. Otherwise produce a concrete report only.

Required output:
- Write your deliverable to `.agent/reviews/2026-06-08-gpt-memory-refresh.md` in your worktree.
- If you edit code, also write a proof object under `proofs/v013/` and list the exact tests/commands.
- Final line in the tmux pane must include: `GPT MEMORY REFRESH DONE`.

Report format:
- objective
- files changed
- commands run
- proof objects produced
- updated memory ranking
- MUST-FIX-BEFORE-TOST list
- SAFE-NOW list
- V0.14 list
- whether TOST can start now, with reasoning
- unresolved risks
- next decision needed, if any
