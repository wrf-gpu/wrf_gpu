You are Fable xhigh continuing in the same worktree:

`/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`

Read and follow the sprint contract:

`.agent/sprints/2026-06-11-v014-fable-acoustic-substep-blocker/sprint-contract.md`

Your whole endpoint task:

Find, fix, and prove the remaining Switzerland d01 h36->h37 dry-mass/PSFC venting blocker. The previous sprint found and committed a real WRF-faithfulness fix (`hypsometric_opt=2` LOG-form `al/alt/p` plus LOG base `alb`), but it also proved that the large-step HPG native-face mismatch is NOT the venting blocker: h36->h37 excess outflux only collapsed about 1%.

Do not do a narrow partial task. Use the already captured WRF-native stage-boundary dumps in `/mnt/data/wrf_gpu_validation/v014_switzerland_hpg_native_face/hpg_dumps` to compare JAX operational stage-boundary states against WRF for step 7201/7202, localize the first divergent increment inside the acoustic-substep lane (`advance_uv`, `advance_w`, `advance_mu`, `ww`/divergence, vertical implicit solve, pressure/phi refresh coupling, stage handoff), implement the smallest WRF-faithful fix if found, and prove it with an h36->h37 gate.

If the acoustic-substep lane is refuted, produce a no-fix proof that narrows the next root class enough for a direct next sprint. Do not start performance audit work; that waits until Switzerland 72h is green or explicitly accepted.

Output required:

- `.agent/reviews/2026-06-11-v014-fable-acoustic-substep-blocker.md`
- `proofs/v014/switzerland_acoustic_substep_blocker.json`
- source commit(s) on this worker branch if you implement a fix

End stdout with exactly:

`FABLE ACOUSTIC_SUBSTEP_BLOCKER DONE - see .agent/reviews/2026-06-11-v014-fable-acoustic-substep-blocker.md`

Do not run ask-hermes/Telegram/human notification commands. Do not touch `/home/enric/src/canairy_waves`.
