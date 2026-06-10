You are Fable/Mythos, high-end reviewer for wrf_gpu2 v0.15 planning.

Repository: `/home/enric/src/wrf_gpu2`
Sprint contract:
`.agent/sprints/2026-06-10-v015-fable-kernel-efficiency-review/sprint-contract.md`

This sprint should be dispatched only after TOST and Switzerland compute are
running or complete and after your current v0.14 Step-1 closure sprint is done.

Read in order:
1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/decisions/V0150-ROADMAP-DRAFT.md`
5. the sprint contract above
6. the memory/FP32/proof files named in the contract

Objective:
Perform a thorough read-only review of all major compute and memory kernels and
produce a ranked v0.15 action list. Do not edit source. The end product is a
manager-usable action list with expected gain, complexity, risk, and proof gates
for each recommendation.

Scoring:
For every relevant candidate, classify compute gain, memory gain, complexity,
correctness risk, proof gates, and v0.15 recommendation. Complexity combines
what must change, what must be tested, numerical/WRF-fidelity risk, XLA/GPU risk,
and risk that the gain is speculative or already optimized away.

Output:
`.agent/reviews/2026-06-10-v015-fable-kernel-efficiency-review.md`
and optionally `proofs/v015/kernel_efficiency_review.json`.

No source edits, no GPU jobs, no commits.

When finished, send:

```bash
tmux send-keys -t 0:2 'FABLE V015_KERNEL_EFFICIENCY_REVIEW DONE - see .agent/reviews/2026-06-10-v015-fable-kernel-efficiency-review.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
