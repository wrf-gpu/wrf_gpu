You are GPT-5.5 xhigh, secondary validation/debug worker for wrf_gpu2 v0.14.

Repository: `/home/enric/src/wrf_gpu2`
Sprint contract:
`.agent/sprints/2026-06-10-v014-gpt-rrtmg-step1-forcing-parity/sprint-contract.md`

Objective:
Localize the secondary RRTMG Step-1 forcing residual without production source
edits. Fable is working on the primary NoahMP land-tile energy blocker, so keep
file ownership disjoint.

Read:
1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
4. the sprint contract above
5. `proofs/v014/noahmp_step1_closure.{py,json,md}`
6. relevant RRTMG/radiation coupling code

Known facts:
- LWDN/GLW bias is about `+17.44 W/m2`, both sides clear-sky.
- SWDOWN `+radt/2` convention is much better than lead-0.
- RTHRATEN residual is max_abs about `19.425`, RMSE about `2.488`.
- This is secondary to NoahMP land HFX but must be closed/bounded later.

Rules:
- No `src/gpuwrf/**` edits.
- No tests/source edits.
- No GPU, no TOST, no Switzerland, no Grid-Delta campaign, no FP32/memory work.
- Write only the proof/review files named in the contract.
- If you find an obvious source fix, describe it exactly; do not apply it.

Run the required gates. When finished, send:

```bash
tmux send-keys -t 0:2 'GPT RRTMG_STEP1_FORCING_PARITY DONE - see proofs/v014/rrtmg_step1_forcing_parity.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
