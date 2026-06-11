# Sprint Contract: v0.14 GPT Advance-W Term Split

Date: 2026-06-11
Owner: manager
Assignee: GPT-5.5 xhigh, tmux
Status: READY

## Objective

Independently diagnose and, if locally provable, fix the remaining Switzerland /
Gotthard h36->h37 dry-mass / PSFC field-parity blocker by splitting the first
wrong RK1 acoustic substep at WRF call `21601 -> 21602`.

This is a hypothesis-neutral debug sprint. Current evidence narrows the first
wrong state to either:

- outputs from `advance_mu_t` consumed by `advance_w`; or
- internals of `advance_w_wrf()` before `ph_next` is produced.

Treat that as the current evidence boundary, not as a mandated conclusion. Build
your own ranked hypothesis ledger, test it against WRF anchors, and solve the
whole assigned blocker if possible. If the evidence points outside
`advance_mu_t` / `advance_w`, reject the manager narrowing and prove the better
boundary.

## Current Known Facts

- Do not start a long Switzerland 72h GPU run. The long release gate is blocked
  until the h36 storm-state short gate collapses or is formally bounded.
- Canary L2 d02 72h is accepted as bounded/proceed; do not spend this sprint on
  Canary.
- CPU-WRF Switzerland 72h truth is complete at:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`.
- Latest manager branch tip at sprint creation:
  `9a8e9fe8 manager reduce worker polling cadence`.
- Already merged/proven:
  - LBC clock root cause.
  - HPG native-face / `hypsometric_opt=2` LOG-form diagnostics.
  - real-case `rhs_ph`, stage omega edge handling, WRF dycore constants.
  - WRF specified LBC cadence + order-5 specified-boundary advection
    degradation, flag-gated/default-off, with GPT verifier-required `w`
    wrapper correction.
  - GPU surface-`w` discriminator rejects the known decoupled-vs-WRF-coupled
    lower-boundary wind-feed deviation as first `phi/p` creator.
- Latest key numbers:
  - WRF call `21601` vs JAX h36 start: `mu/p/ph` max abs `0.0`;
    `alt` max abs `3.10e-06`; `al` max abs `5.35e-05`.
  - after one RK1 acoustic substep vs WRF call `21602`, interior RMSE:
    `mu=0.020896037745516495`,
    `p=1.1261975184532773`,
    `ph=0.4352639584631776`,
    `al=9.157037882376838e-05`,
    `alt=9.123246305212642e-05`.
  - specified stage omega interior RMSE vs WRF oracle:
    `5.79142152447787e-16`.
  - real-case `rhs_ph` port validation interior RMSE:
    `2.3951502769070958e-11`.
  - surface-`w` discriminator on GPU:
    `current_decoupled` and `wrf_coupled` both have `p` interior RMSE
    `1.126197518453275`, `ph` interior RMSE `0.43526395846317767`,
    improvement fraction `0.0`.

## Required Context

Read first:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/skills/validating-physics/SKILL.md`
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
- `.agent/reviews/2026-06-11-v014-gpt-switzerland-residual-root-cause.md`
- `proofs/v014/gpt_switzerland_residual_narrowing_summary.json`
- `proofs/v014/switzerland_advance_w_phi_discriminator.json`
- `proofs/v014/switzerland_advance_w_phi_discriminator.py`
- `proofs/v014/switzerland_acoustic_substep_blocker.json`
- `proofs/v014/switzerland_acoustic_continuation.json`
- `proofs/v014/switzerland_stage3_wrapper_cadence.json`
- `proofs/v014/gpt_stage3_wrapper_verifier.md`

Relevant source areas, not exclusive:

- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/dynamics/core/advance_w.py`
- `src/gpuwrf/dynamics/core/acoustic_wrf.py`
- `src/gpuwrf/dynamics/core/calc_p_rho.py`
- `src/gpuwrf/dynamics/core/small_step_prep.py`
- `src/gpuwrf/dynamics/core/rhs_ph.py`
- `src/gpuwrf/runtime/operational_mode.py`
- WRF pristine source under `/home/enric/src/wrf_pristine/WRF`, especially
  `dyn_em/module_small_step_em.F`, `dyn_em/solve_em.F`,
  `dyn_em/module_big_step_utilities_em.F`, and `dyn_em/module_advect_em.F`.

## Required Work

1. Build a compact independent hypothesis ledger before coding:
   - most plausible root classes;
   - evidence for / against each;
   - cheapest falsification or proof;
   - expected runtime / performance impact if fixed.
2. Use the fastest rigorous proof path. Prefer minimal savepoint / term-split /
   oracle loops over long forecasts. Avoid JAX-vs-JAX self-acceptance.
3. Split the first wrong boundary, in this order unless your ledger proves a
   better path:
   - post-`advance_mu_t` inputs to `advance_w`: `ww_new`, `muave_new`,
     `muts_new`, `theta_coupled`, `w`, `ph_tend`, `rw_tend`;
   - `advance_w_wrf()` internal terms: `rhs_seed`, vertical phi-advection
     contribution, explicit `rw_tend`, implicit pressure terms / coefficients,
     Thomas forward/back solve outputs, `w_solved`, and final `ph_next`;
   - if `ph_next` matches but `p/al/alt` do not, move to `calc_p_rho_step`.
4. If WRF-native terms are missing, create the minimal WRF-anchored dump or
   equivalent WRF-line-ported oracle needed to compare call `21602`. Do not
   corrupt the pristine WRF tree; keep any instrumentation patch/script clearly
   proof-only and record exact commands.
5. If you find a local WRF-faithful source defect, implement the smallest fix on
   your worker branch, add/adjust focused tests/proofs, and run the shortest gate
   that proves material improvement. Avoid clamps, tolerance relaxation, masking,
   artificial damping, or host/device transfers inside timestep loops.
6. Continue autonomously until one stop criterion is met. Do not stop after only
   one diagnostic if your context gives a clear next proof loop.

## Stop Criteria

Return control to the manager only when at least one is true:

- `FIXED`: local WRF-faithful fix implemented and short h36 / stage gate proves
  material improvement.
- `LOCAL_FIX_PROPOSED`: specific source defect and minimal patch found, but a
  manager-controlled GPU/time gate or merge coordination is needed.
- `NARROWED_NO_FIX`: first wrong operator / term / state boundary is isolated
  more tightly than the current handoff, with exact next proof/fix target.
- `REJECTED_PRIOR_BOUNDARY`: evidence proves the current
  `advance_mu_t` / `advance_w` boundary is wrong and identifies a better root
  class.
- `METHOD_LIMIT`: after at least three substantial proof attempts or one
  purpose-built term-split harness, remaining progress requires new external WRF
  artifacts, long GPU gate, scarce Fable escalation, or a larger architecture
  decision.

## Acceptance Gate

Successful close requires:

- proof object(s) under `proofs/v014/`;
- report under `.agent/reviews/`;
- exact commands run;
- source commit on the worker branch if code changed;
- focused test or proof validation;
- no long 72h GPU gate;
- no unproven performance-costly changes.

Partial success is acceptable only if it meets one stop criterion and provides a
stronger falsifiable narrowing than
`.agent/reviews/2026-06-11-v014-gpt-switzerland-residual-root-cause.md`.

## Output

Write:

- `.agent/reviews/2026-06-11-v014-gpt-advance-w-term-split.md`
- proof scripts / JSON / markdown under `proofs/v014/` as needed

If source changes are made, commit them on your worker branch.

Report must include:

- verdict: `FIXED`, `LOCAL_FIX_PROPOSED`, `NARROWED_NO_FIX`,
  `REJECTED_PRIOR_BOUNDARY`, or `METHOD_LIMIT`;
- compact hypothesis ledger;
- exact evidence chain and commands run;
- files changed and proof objects produced;
- h36/stage gate result if run;
- runtime / performance risk;
- next manager decision, max 8 bullets.

End stdout exactly:

`GPT ADVANCE_W_TERM_SPLIT DONE - see .agent/reviews/2026-06-11-v014-gpt-advance-w-term-split.md`

## Constraints

- Do not use `ask-hermes`, Telegram, or human notification commands.
- Do not touch `/home/enric/src/canairy_waves`.
- Do not dispatch Fable/Mythos or any other worker.
- Do not run long 72h GPU gates.
- Use GPU only for short focused proof gates and only if the GPU is free.
- Treat manager/Fable hypotheses as context, not truth.
- Manager will poll at roughly 15-minute cadence unless a DONE marker appears
  or resource safety requires earlier action.
