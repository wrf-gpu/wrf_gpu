# V0.14 GPT Acoustic-Substep Candidate Verifier

Date: 2026-06-11
Worker: GPT-5.5 xhigh
Worktree: `/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`

## Objective

Independently verify and, if safely possible, minimally advance the uncommitted
Fable acoustic-substep candidate fix.

The candidate is currently uncommitted in this worktree and changes the acoustic/operational path after Fable hit a model-limit prompt. Verify whether the candidate is a plausible WRF-faithful fix and whether the produced 2h Switzerland gates support accepting it for the next manager step.

## Current Candidate State

Fable has source edits in:

- `src/gpuwrf/coupling/boundary_apply.py`
- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/integration/daily_pipeline.py`
- `src/gpuwrf/runtime/operational_mode.py`

Key claimed fixes visible in the diff:

- WRF constants: `CP_D = 7*R_D/2 = 1004.5`, `GRAVITY_M_S2 = 9.81`.
- Fresh stage `ww` from `stage_velocities.rom` is used for `small_step_prep`, `rhs_ph`, and acoustic core state, instead of stale carried `ww`.
- `advance_w` surface boundary consumes just-advanced acoustic work-delta `u/v`, not frozen physical `u_1/v_1`.
- `w_damping` is moved to WRF's once-per-RK-stage large-step `rw_tend` location; per-substep damping is disabled for operational scan.
- Daily pipeline threads `diff_opt/km_opt` from real namelist.

Current run/log state at dispatch:

- `/tmp/acoustic_gate_forecast.log` shows `forecast result: status=PASS hours=2 output_dir=/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/gpu_output_acoustic_substep_fix`
- `/tmp/acoustic_chain2.log` shows `CHAIN: nokm stashed` and `CHAIN: km gate exit=0`; the `dt18/substeps=4` candidate may still be running.
- Main proof draft: `proofs/v014/switzerland_acoustic_substep_blocker.json`
- Proof script: `proofs/v014/switzerland_acoustic_substep_blocker.py`

## Required Work

1. Inspect the candidate diff for correctness risks, WRF-faithfulness, performance risks, and hidden non-identity/architecture regressions.
2. Inspect the proof script and JSON for whether it actually compares against WRF-native stage-boundary dumps, not JAX-vs-JAX self-compare.
3. When the existing GPU chain finishes, evaluate generated outputs and logs. Do not collide with it.
4. Run only targeted non-destructive checks:
   - `python -m py_compile` on changed/proof files
   - JSON validation
   - focused tests if they are cheap
   - GPU checks only through `scripts/run_gpu_lowprio.sh` and only if the lock is free or the running chain has finished
5. You may make small, reversible, non-destructive additions that help finish the proof:
   - verifier report,
   - tiny helper/analyzer scripts,
   - generated JSON/markdown proof summaries,
   - a proposed patch file if you find an obvious fix.
6. Do not overwrite or reshape Fable's active core source edits while Fable is still running. If a source change to a Fable-touched file is needed, either wait until Fable's GPU/process chain is done and make a minimal scoped edit, or write it as a separate patch proposal under `proofs/v014/` / `.agent/reviews/` for the manager/Fable to apply.
7. If you can cleanly finish the bug from here with minimal safe changes, do it and prove it. If the result looks weak or speculative, report `NEED_FABLE_AFTER_RESET` with the exact handoff.

## Output

Write:

`.agent/reviews/2026-06-11-v014-gpt-acoustic-substep-verifier.md`

End stdout with exactly:

`GPT ACOUSTIC_SUBSTEP_VERIFIER DONE - see .agent/reviews/2026-06-11-v014-gpt-acoustic-substep-verifier.md`

## Verdict Required

The report must answer:

- ACCEPT_FOR_MANAGER_GATE, REJECT, or NEED_FABLE_AFTER_RESET.
- Whether the Fable candidate materially collapses the h36->h37/h38 Switzerland residual versus the previous baseline.
- Whether the candidate introduces performance risks or host/device transfer risks.
- Which exact commands and proof files support the verdict.

## Constraints

- Do not run `ask-hermes`, Telegram, or human notification commands.
- Do not touch `/home/enric/src/canairy_waves`.
- Do not modify source.
- Do not interact with the Fable tmux window.
- Preserve GPU serialization; do not start a competing GPU process while Fable's chain is running.
