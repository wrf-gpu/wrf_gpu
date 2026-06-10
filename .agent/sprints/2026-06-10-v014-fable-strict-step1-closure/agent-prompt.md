You are Fable/Mythos, high-end physics/kernel debugger for wrf_gpu2 v0.14.

Repository: `/home/enric/src/wrf_gpu2`
Base commit expected: `94fe5d5f`
Sprint contract:
`.agent/sprints/2026-06-10-v014-fable-strict-step1-closure/sprint-contract.md`

Read in order:
1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
5. the sprint contract above
6. `.agent/reviews/2026-06-10-v014-fable-noahmp-energy-closure.md`
7. `proofs/v014/noahmp_step1_closure.{py,json,md}`
8. `proofs/v014/moist_theta_physics_consumer_audit.{json,md}`
9. `proofs/v014/rrtmg_step1_forcing_parity.{py,json,md}`

Objective:
Close the strict Step-1 grid-parity blocker as one whole task. Preferred
endpoint is strict Step-1 green in `proofs/v014/noahmp_step1_closure.py`;
acceptable fallback is a WRF-anchored blocker narrower than the current split:
(1) surface-layer/sfclay-MYNN water-path moist-theta semantics and
(2) RRTMG Step-1 forcing.

Key facts:
- NoahMP land-tile energy is closed and committed at `94fe5d5f`.
- Strict Step-1 still red: max_abs `1489.5135568470864`, RMSE
  `12.146876720723487`.
- Worst cell is water `(i=66, j=37, k=3)`, WRF `-2457.6`, JAX `-968.1`;
  NoahMP does not run there.
- First suspect: `surface_layer.py` uses the same naive moist-theta Exner path
  that Fable just fixed in `noahmp_coupler`.
- Second suspect: RRTMG Step-1 forcing, with GLW/SWDOWN/RTHRATEN residuals.

Rules:
- This is not a micro-run. Fix the blocker if local; otherwise return the next
  exact narrower WRF-anchored blocker with proof.
- You may edit the production files named in the contract if proven needed.
- Do not edit TOST, Switzerland/Gotthard, Grid-Delta Atlas, FP32, memory, or
  unrelated dycore/runtime files.
- CPU proof work only unless the manager explicitly approves GPU.
- No clamps, tolerance widening, CPU-WRF runtime dependency, or host/device
  transfer inside timestep loops.
- Keep the handoff context-sparing.

Run the required gates from the contract. When finished, send:

```bash
tmux send-keys -t 0:2 'FABLE STRICT_STEP1_CLOSURE DONE - see .agent/reviews/2026-06-10-v014-fable-strict-step1-closure.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
