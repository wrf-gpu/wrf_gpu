# V0.14 Switzerland Acoustic-Substep Continuation Handoff

Date: 2026-06-11
Owner: manager
Status: active correctness blocker; do not start performance audits or
Switzerland 72h GPU rerun yet.

## Current Truth

The Switzerland/Gotthard d01 h36->h37 strong-flow dry-mass/PSFC blocker is
still open.

Update after Fable continuation merge:

- Merged branch: `worker/fable/v014-hpg-native-face-fix`
- Merge commit on manager branch: `82f6b703`
- Landed subfix commits:
  - `3d0b439c` — real-case `hypsometric_opt=2` LOG-form HPG diagnostics.
  - `79b0c22e` — real-case `rhs_ph`, edge-faithful specified-domain stage
    omega, and WRF dycore constants.
- Proofs:
  - `.agent/reviews/2026-06-11-v014-fable-hpg-native-face-fix.md`
  - `.agent/reviews/2026-06-11-v014-fable-acoustic-continuation.md`
  - `proofs/v014/switzerland_hpg_native_face_fix.json`
  - `proofs/v014/switzerland_acoustic_continuation.json`

`79b0c22e` found and fixed the GPT-ranked real-case geopotential tendency lane:
`rhs_ph` was still the idealized 2nd-order/unit-map/periodic operator while WRF
uses map-factored order<=6 specified-boundary horizontal geopotential
advection. It also replaced periodic-wrap stage omega with
`stage_omega_specified` for specified/nested domains. Term-level WRF oracle
parity is machine-precision. Stage-boundary p/ph and band errors improve
substantially.

However the h36->h37 gate remains red:

| Run | h36->h37 residual Pa/cell/h | h36->h37 excess outflux Pa/cell/h |
|---|---:|---:|
| CPU truth | `+5.178443877551032` | n/a |
| hypso `3d0b439c` | `-27.697448979591826` | `-28.3281887755102` |
| rhs_ph/stage-omega `79b0c22e` | `-21.882908163265313` | `-27.203954081632645` |

Conclusion: accept and keep the subfixes, but do not run 72h yet. The next
root lane is now narrowed to stage-3/end-of-step wrapper cadence
(physics/moist/LBC/p-refresh interleaving vs WRF) plus residual lateral-band
amplifier.

Fable branch/worktree:

- Worktree: `/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`
- Branch: `worker/fable/v014-hpg-native-face-fix`
- Last committed worker fix: `3d0b439c`

Earlier proof chain:

- `3d0b439c` found and fixed a real WRF-faithfulness bug: real-case pressure
  diagnostics need WRF `hypsometric_opt=2` LOG-form `al/alt/p` and LOG-form
  base `alb`.
- Native WRF HPG face truth proves the large-step HPG face mismatch collapses
  to roundoff-class relative error.
- That fix is **not** the Switzerland venting blocker. h36->h37 excess outflux
  improved only about 1% (`-28.62 -> -28.33 Pa/cell/h`).

Fable then built an uncommitted acoustic-substep candidate with plausible
WRF-faithful changes:

- WRF constants `cp=1004.5`, `g=9.81`;
- fresh stage `ww` threading;
- work-delta `u/v` feed into `advance_w` surface BC;
- stage-level `w_damping`;
- real-case `diff_opt/km_opt` namelist threading.

GPT independently verified the candidate and rejected it as the release gate:

- Report:
  `/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix/.agent/reviews/2026-06-11-v014-gpt-acoustic-substep-verifier.md`
- Proof JSON:
  `/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix/proofs/v014/switzerland_acoustic_substep_blocker.json`
- Verdict: `NEED_FABLE_AFTER_RESET`

Key numbers from the proof JSON:

| Run | h36->h37 residual Pa/cell/h | h36->h37 excess outflux Pa/cell/h |
|---|---:|---:|
| CPU truth | `+5.178443877551032` | n/a |
| old `ec4d6769` | `-32.686352040816345` | `-28.614795918367335` |
| HPG native-face `3d0b439c` | `-27.697448979591826` | `-28.3281887755102` |
| acoustic candidate, no km | `-35.8594387755102` | `-28.798852040816314` |
| acoustic candidate | `-35.94119897959182` | `-28.81913265306123` |

Collapse metrics:

- `collapse_fraction_vs_old = -0.007140946777213886`
- `collapse_fraction_vs_hypso = -0.017330577730950925`

Conclusion: the acoustic candidate is stable enough to run 2h, but it does not
materially collapse the dry-mass/PSFC residual and should not be merged as the
manager gate.

## Next Fable Whole-Task Prompt

After Fable resets, send this as one complete task, not a micro-hypothesis:

```text
Continue the v0.14 Switzerland/Gotthard h36->h37 strong-flow dry-mass/PSFC
blocker from the current worktree.

Important manager/GPT verdict:
- Do not merge the current acoustic candidate as the release gate.
- GPT verified the candidate in
  .agent/reviews/2026-06-11-v014-gpt-acoustic-substep-verifier.md and found
  NEED_FABLE_AFTER_RESET.
- h36->h37 residual worsened versus the accepted HPG-native-face baseline:
  HPG 3d0b439c residual -27.697 Pa/cell/h, acoustic candidate -35.941.
- Stage compare is real WRF-native evidence and still shows large p/ph
  increment mismatch: start from sub4_dt18_bcfix, especially
  step1_stage1_vs_21602 and step2_stage2_vs_21606.

Whole endpoint:
Find, fix, and prove the remaining WRF-vs-JAX root cause for Switzerland
h36->h37. Keep useful parts of the acoustic proof harness. Revert or isolate
candidate source changes that are not proven by the hourly gate. Implement only
WRF-faithful fixes, no clamps/masks/tolerance hacks. Acceptance is material
collapse of h36->h37 excess dry-mass/PSFC residual versus both old ec4d6769 and
hypso 3d0b439c, plus WRF-native staged evidence for the first divergent
increment. If no fix is found, produce an exact next root class and a proof
object good enough for a direct next sprint.

Do not start performance audit work. Do not use ask-hermes/Telegram. Do not
touch /home/enric/src/canairy_waves.
```

## Manager Rules

- Do not start Canary/Switzerland maximal-speed performance audits until
  Switzerland is green or explicitly bounded/accepted.
- Do not start a Switzerland 72h GPU rerun until the h36 short gate is fixed or
  formally bounded.
- Treat `3d0b439c` as a real WRF-fidelity subfix candidate, but do not merge it
  blindly with the rejected acoustic candidate.
- Preserve GPT's proof report and JSON when cleaning up the worker branch.

## GPT Side-Analysis Addendum

Three GPT runners completed before the 12:20 WEST cutoff and produced durable
reports in the main repo:

- `proofs/v014/gpt_acoustic_stage_mismatch_analysis.md`
- `proofs/v014/gpt_acoustic_wrf_source_equation_audit.md`
- `proofs/v014/gpt_acoustic_stage_pattern_miner.md`
- `proofs/v014/gpt_acoustic_stage_pattern_miner_summary.json`

Consensus:

1. Do not attempt another production source fix from the current evidence.
   Add diagnostic captures first.
2. The first material mismatch is p/ph-first, not mass-first:
   `step1_stage1_vs_21602` interior increment RMSE is roughly `p=6.67 Pa`,
   `ph=1.51`, while `mu=0.0209 Pa` is still small.
3. The most likely next root lane is real-case geopotential tendency:
   `calc_ww_cp` / stage `ww` plus `rhs_ph` semantics for specified-boundary,
   map-factor, and higher-order WRF behavior. The current JAX implementation is
   still documented/structured as periodic/idealized in this area.
4. `advance_w` / vertical implicit solve is second: if `ww` and `ph_tend` prove
   matching before the acoustic loop, then capture first-substep `advance_w`
   RHS, terrain surface BC, tridiagonal result, and ph finish.
5. `calc_p_rho_step` / pressure refresh is third: if ph matches but p/alt/al
   diverge, compare post-`small_step_finish` and post-refresh surfaces.
6. Boundary-band errors are real and large, but current evidence says they are
   an amplifier rather than the first interior p/ph source.
7. Time-step cadence and `diff_opt/km_opt` do not explain the first stage
   mismatch; defer missing dt18 hourly gate until the first substep divergence
   is found.

Recommended diagnostic discriminator for Fable:

```bash
python proofs/v014/switzerland_acoustic_substep_blocker.py \
  --stage-compare --tag sub4_dt18_terms --dt 18 --substeps 4 --steps 2 \
  --capture-intra
python proofs/v014/switzerland_acoustic_substep_blocker.py --analyze
```

Required new captures before another production fix:

- WRF/JAX `ww` immediately after WRF `calc_ww_cp` / JAX
  `_stage_transport_velocities`;
- WRF/JAX `ph_tend` immediately after `rhs_ph`;
- WRF/JAX `rw_tend` after `pg_buoy_w` / `w_damp`;
- if those match, first acoustic-substep pre/post `advance_w`,
  tridiagonal solve, ph finish, and `calc_p_rho_step`.
