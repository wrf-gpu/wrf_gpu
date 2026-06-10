You are GPT-5.5 xhigh, debugger/fixer for wrf_gpu2 v0.14.

Repo: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`

Read first:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/sprints/2026-06-10-v014-step1-tsk-znt-sourcing/sprint-contract.md`
5. `proofs/v014/step1_sfclay_boundary_fix.md`
6. `.agent/reviews/2026-06-10-v014-step1-sfclay-boundary.md`

Mission:

Close, or reduce to one strictly narrower WRF-anchored blocker, the Step-1
`TSK/ZNT` surface input sourcing mismatch before `sfclay_mynn`.

Important facts:

- MYNN cold-start QKE is fixed.
- MYNN surface first-call semantics are fixed.
- Strict Step-1 remains red: max_abs `1497.6112467075195`, RMSE
  `13.296448784742802`.
- Current narrow blocker: TSK max_abs `8.344940187890643 K`; ZNT max_abs
  `0.9737602076530456 m`.
- Do not run TOST, Switzerland, broad FP32, broad memory, Hermes, or Fable.
- CPU-only unless manager later grants a short GPU probe.

Work style:

- Use a focused WRF hook and comparator. Do not guess from wrfout alone.
- If the TSK/ZNT hypothesis is wrong, prove it and name the next exact blocker.
- If the fix is local and performance-compatible, implement it.
- Keep top-level reports compact.

Deliver:

- `proofs/v014/step1_tsk_znt_sourcing_fix.py`
- `proofs/v014/step1_tsk_znt_sourcing_fix.json`
- `proofs/v014/step1_tsk_znt_sourcing_fix.md`
- `proofs/v014/step1_tsk_znt_sourcing_fix_wrf_patch.diff` if WRF hook changes
  are used.
- `.agent/reviews/2026-06-10-v014-step1-tsk-znt-sourcing.md`
- updated strict Step-1 proof artifacts if needed.

Run the acceptance gates from the sprint contract. If a gate is blocked, record
the exact blocker and fastest next command.

When done, print exactly:

`GPT STEP1_TSK_ZNT_SOURCING DONE - see proofs/v014/step1_tsk_znt_sourcing_fix.md`

Then notify manager pane:

```bash
tmux send-keys -t 0:2 'GPT STEP1_TSK_ZNT_SOURCING DONE - see proofs/v014/step1_tsk_znt_sourcing_fix.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
