# V0.14 Fable Acoustic-Substep Blocker Sprint

Date: 2026-06-11
Worker: Fable xhigh
Branch/worktree: `worker/fable/v014-hpg-native-face-fix` at `/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`

## Objective

Close the remaining Switzerland d01 h36->h37 dry-mass/PSFC venting blocker end to end.

Find, fix, and prove the first remaining WRF-vs-JAX divergence in the acoustic-substep lane using the WRF-native stage-boundary dumps already produced by the HPG native-face sprint. If the acoustic-substep lane is conclusively refuted, deliver the exact next root class with proof and a ready-to-run next target.

## Current Evidence

- Fable commit `3d0b439c` fixed a real WRF-faithfulness bug: real cases need `hypsometric_opt=2` LOG-form `al/alt/p` diagnostics and LOG-form base `alb`.
- Native WRF HPG face truth shows the large-step HPG face terms now match WRF to roundoff-class relative error.
- The Switzerland h36->h37 blocker remains: full-physics excess outflux only collapsed about 1% (`-28.62 -> -28.33 Pa/cell/h`), so large-step HPG faces are not the blocker.
- The next likely lane is downstream of `rk_tendency`: acoustic substeps, including `advance_uv`, `advance_w`, `advance_mu`, `ww`/divergence, vertical implicit solve, and their per-stage state handoff.
- WRF stage-boundary truth is already captured in:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_hpg_native_face/hpg_dumps`
  with calls `21602`, `21603` for RK2/RK3 of step 7201 and `21604`-`21606` for step 7202.
- The primary proof file from the prior sprint is:
  `proofs/v014/switzerland_hpg_native_face_fix.json`
- Prior report:
  `.agent/reviews/2026-06-11-v014-fable-hpg-native-face-fix.md`

## Required Work

1. Build or extend debug instrumentation that exposes JAX operational per-stage states at the same boundaries as the WRF dumps.
2. Compare WRF-native stage-boundary increments against JAX for step 7201 and, if useful, step 7202.
3. Bisect the first divergent increment inside the acoustic lane, not just the final h37 field:
   - `advance_uv`
   - `advance_w`
   - `advance_mu`
   - `ww`/divergence
   - vertical implicit solve and pressure/phi refresh coupling
   - stage-boundary handoff between large-step tendency and acoustic solver
4. Implement the smallest WRF-faithful source fix if the root is found.
5. Prove the fix with staged evidence and an h36->h37 gate.

## Acceptance Gates

A fix is acceptable only if all are true:

- It is anchored against WRF-native truth, not a JAX-vs-JAX self-compare.
- It materially collapses the h36->h37 Switzerland venting blocker. Target: most of the old excess outflux removed; if not fully closed, quantify exactly what remains and why.
- It does not add clamps, masks, artificial damping, or host/device transfers inside timestep loops.
- It preserves the GPU-native performance architecture.
- It produces a clear proof object under `proofs/v014/` and a concise final report under `.agent/reviews/`.
- It runs focused tests and reports any pre-existing failures separately from regressions.

If no fix is found, the report must be equally useful:

- first divergence localized to the narrowest concrete function/state boundary achieved,
- hypotheses refuted,
- exact proof paths,
- next implementation target that can be handed off without rediscovery.

## Output

Produce:

- `.agent/reviews/2026-06-11-v014-fable-acoustic-substep-blocker.md`
- `proofs/v014/switzerland_acoustic_substep_blocker.json`
- any focused proof/debug script needed under `proofs/v014/`
- source commits on the worker branch if a source fix is implemented

End stdout with exactly:

`FABLE ACOUSTIC_SUBSTEP_BLOCKER DONE - see .agent/reviews/2026-06-11-v014-fable-acoustic-substep-blocker.md`

## Constraints

- Do not run `ask-hermes`, Telegram, or human notification commands.
- Do not touch `/home/enric/src/canairy_waves`.
- Do not start the performance audit yet; it waits until Switzerland 72h is green or explicitly bounded-accepted.
- Use the GPU only through the established serialized low-priority wrapper/lock when needed.
- Avoid micro-handoffs. This is a whole endpoint task: search, fix, prove, or produce an exact no-fix localization.
