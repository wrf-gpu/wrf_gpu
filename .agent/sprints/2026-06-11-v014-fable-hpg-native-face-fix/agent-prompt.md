You are Fable xhigh, hard-debug worker for wrf_gpu2 v0.14.

Worktree/branch: `/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`, branch `worker/fable/v014-hpg-native-face-fix`, based on the manager branch `worker/gpt/v013-close-manager`.

First verify `git branch --show-current` and `git log -1 --oneline`. Then read the sprint contract:

`.agent/sprints/2026-06-11-v014-fable-hpg-native-face-fix/sprint-contract.md`

Goal: close the remaining Switzerland d01 h36 field-parity blocker end to end. The blocker has been narrowed to WRF large-step horizontal-pressure-gradient native-face inputs after `rk_step_prep` / `rk_phys_bc_dry_1`, especially `pb_al` and `p_alt` on U/V faces. Existing GPT proofs already falsified local `muts`, `p`, `al`, and `alt` algebra cleanups as sufficient fixes.

You must either:

1. obtain WRF-native face truth, compare against JAX, implement the local WRF-faithful fix, prove h36 collapse, run required tests, and commit; or
2. produce a WRF-anchored exact-root proof that names the precise wrong face/input array and one concrete next implementation target.

This is not a micro-analysis sprint. If one hypothesis fails, continue to the next most likely native-face/staged-live mismatch until the endpoint is met or a true blocker is hit.

Hard rules:

- Do not use `ask-hermes`, Telegram, or any human notification bridge.
- Do not touch `/home/enric/src/canairy_waves`.
- Do not run a 72 h validation.
- Use short h36 probes and WRF/JAX face evidence.
- No clamps/masking.
- No host/device transfer inside timestep loops.
- If editing WRF for instrumentation, keep it reproducible and do not permanently dirty `/home/enric/src/wrf_pristine/WRF`.

Write the required report and proof:

- `.agent/reviews/2026-06-11-v014-fable-hpg-native-face-fix.md`
- `proofs/v014/switzerland_hpg_native_face_fix.py`
- `proofs/v014/switzerland_hpg_native_face_fix.json`

Commit all changes on `worker/fable/v014-hpg-native-face-fix`.

When done, print exactly:

`FABLE HPG_NATIVE_FACE_FIX DONE - see .agent/reviews/2026-06-11-v014-fable-hpg-native-face-fix.md`
