# V0.14 GPT Acoustic Stage-Mismatch Analysis

Date: 2026-06-11
Owner: manager
Assignee: GPT-5.5 xhigh in tmux
Status: prepared while Fable is session-limited until 12:30 WEST

## Objective

Read-only, WRF-native analysis of the remaining Switzerland/Gotthard h36->h37
p/ph acoustic-stage mismatch after the rejected Fable acoustic candidate.

The goal is to give Fable a sharper next starting point after reset, not to
change source.

## Context

Current Fable worktree:

`/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`

Durable manager handoff:

`/home/enric/src/wrf_gpu2/.agent/sprints/2026-06-11-v014-switzerland-acoustic-substep-continuation/manager-handoff.md`

GPT verifier report:

`/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix/.agent/reviews/2026-06-11-v014-gpt-acoustic-substep-verifier.md`

Key facts:

- `3d0b439c` fixed a real `hypsometric_opt=2` LOG-form HPG-input bug but it is
  not the venting blocker.
- The current uncommitted acoustic candidate is stable but does not collapse
  h36->h37 residual; GPT verdict is `NEED_FABLE_AFTER_RESET`.
- Stage compare is real WRF-native evidence and still shows large p/ph
  increment mismatch, especially:
  - `step1_stage1_vs_21602`
  - `step2_stage2_vs_21606`
  - tag `sub4_dt18_bcfix`
- WRF dumps live under:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_hpg_native_face/hpg_dumps`

## Required Work

1. Read the manager handoff, GPT verifier report, proof JSON/script, candidate
   diff, and relevant WRF/JAX acoustic source.
2. Determine whether the remaining p/ph mismatch is most likely:
   - pressure/phi refresh placement,
   - `calc_p_rho_phi`/EOS constant or base-state path,
   - `advance_mu_t` / divergence / `ww`,
   - vertical implicit solve / `calc_coef_w`,
   - `rhs_ph` or geopotential tendency staging,
   - boundary/halo application between stages,
   - time-step/substep cadence,
   - or another concrete lane.
3. Prefer evidence over speculation: cite exact JSON fields, WRF calls, source
   functions, and mismatch shape/timing.
4. If a small CPU-only helper is useful, write it under `proofs/v014/` and keep
   it analysis-only. Do not modify source.
5. Do not run GPU work unless the manager explicitly frees it; this sprint is
   primarily CPU/file analysis.

## Output

Write:

`proofs/v014/gpt_acoustic_stage_mismatch_analysis.md`

Report format:

- One-sentence verdict.
- Ranked hypothesis table with columns:
  `rank`, `lane`, `evidence`, `why it fits`, `why it may be wrong`,
  `next falsifier`.
- A compact "Fable next 3 actions" list, max 3 bullets.
- Any source edits proposed as a patch snippet only, not applied.

Print:

`GPT ACOUSTIC_STAGE_MISMATCH DONE - see proofs/v014/gpt_acoustic_stage_mismatch_analysis.md`

## Constraints

- Do not edit source files.
- Do not interact with Fable.
- Do not run `ask-hermes`, Telegram, or human notification commands.
- Do not touch `/home/enric/src/canairy_waves`.
- Do not start performance analysis.
