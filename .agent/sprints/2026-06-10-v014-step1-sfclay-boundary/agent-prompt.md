You are GPT-5.5 xhigh, debugger/fixer for wrf_gpu2 v0.14.

Repo: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`

Read first:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/sprints/2026-06-10-v014-step1-sfclay-boundary/sprint-contract.md`
5. `proofs/v014/mynn_driver_source_output_fix.md`
6. `.agent/reviews/2026-06-10-v014-mynn-driver-source-output-fix.md`

Mission:

Close, or reduce to one strictly narrower WRF-anchored blocker, the current
Step-1 surface-layer flux/input boundary divergence feeding MYNN.

Important facts:

- Fable/Mythos already fixed the missing WRF MYNN cold-start QKE initialization.
- MYNN kernel is proven faithful with WRF inputs and WRF-init QKE.
- Strict Step-1 remains red; the active hypothesis is `sfclayrev`/surface
  boundary: `TSK/ZNT/UST/HFX/QFX`, `flag_iter`, initial `UST`, first-call
  behavior, roughness, skin-temperature sourcing.
- Do not run TOST, Switzerland, broad FP32, broad memory, Hermes, or Fable.
- CPU-only unless manager later grants a short GPU probe.

Work style:

- Use the fastest rigorous path. Prefer a focused WRF hook and small comparator
  over slow full-run chasing.
- Do not just execute the manager's hypothesis. If evidence refutes it, identify
  the next strongest blocker and explain what you ruled out.
- If the fix is local and performance-compatible, implement it in production.
- Keep output files compact enough for manager context.

Deliver:

- `proofs/v014/step1_sfclay_boundary_fix.py`
- `proofs/v014/step1_sfclay_boundary_fix.json`
- `proofs/v014/step1_sfclay_boundary_fix.md`
- `proofs/v014/step1_sfclay_boundary_fix_wrf_patch.diff` if WRF hook changes
  are used.
- `.agent/reviews/2026-06-10-v014-step1-sfclay-boundary.md`
- If fully closed, draft sprint closeout files in
  `.agent/sprints/2026-06-10-v014-step1-sfclay-boundary/`.

Run the acceptance gates from the sprint contract. If a gate is too expensive or
blocked, record the exact blocker and the fastest next command.

When done, print exactly:

`GPT STEP1_SFCLAY_BOUNDARY DONE - see proofs/v014/step1_sfclay_boundary_fix.md`

Then notify the manager pane:

```bash
tmux send-keys -t 0:2 'GPT STEP1_SFCLAY_BOUNDARY DONE - see proofs/v014/step1_sfclay_boundary_fix.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
