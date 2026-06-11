# V0.14 Fable Stage-3 / Wrapper-Cadence Closure Sprint

Date: 2026-06-11
Owner: manager
Assignee: Fable xhigh, tmux `0:7`
Status: prepared after merge `17c856c9`

## Objective

Close the remaining Switzerland/Gotthard d01 h36->h37 strong-flow
dry-mass/PSFC blocker end to end.

The accepted subfixes are now merged on the manager branch:

- `3d0b439c`: real-case `hypsometric_opt=2` LOG-form HPG diagnostics.
- `79b0c22e`: real-case `rhs_ph`, edge-faithful specified-domain stage omega,
  and WRF dycore constants.
- `82f6b703`: merge into manager branch.
- `17c856c9`: roadmap/handoff update.

These fixes are WRF-anchored and should be kept unless a later proof directly
contradicts them. They improve h36->h37 residual but do not close the gate:

- hypso baseline residual: `-27.697448979591826 Pa/cell/h`
- rhs_ph/stage-omega residual: `-21.882908163265313 Pa/cell/h`
- hypso excess outflux: `-28.3281887755102 Pa/cell/h`
- rhs_ph/stage-omega excess outflux: `-27.203954081632645 Pa/cell/h`

## Current Root Lane

The next narrowed lane is:

1. stage-3 / end-of-step wrapper cadence:
   physics/moist/LBC/p-refresh interleaving in JAX vs WRF's per-stage
   sequencing;
2. residual lateral-band amplifier:
   ring 0-7 mass/pressure/geopotential errors that still feed the depth-8
   budget surface.

Do not spend the sprint re-litigating HPG native faces, `hypsometric_opt`,
real-case `rhs_ph`, stage omega, `diff_opt/km_opt`, or dt18 cadence unless new
evidence directly requires it.

## Required Context

Read:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/sprints/2026-06-11-v014-switzerland-acoustic-substep-continuation/manager-handoff.md`
- `.agent/reviews/2026-06-11-v014-fable-acoustic-continuation.md`
- `proofs/v014/switzerland_acoustic_continuation.json`
- `proofs/v014/switzerland_acoustic_substep_blocker.json`
- the current source around:
  - `src/gpuwrf/runtime/operational_mode.py`
  - `src/gpuwrf/coupling/boundary_apply.py`
  - `src/gpuwrf/dynamics/core/rhs_ph.py`
  - `src/gpuwrf/dynamics/flux_advection.py`
  - WRF `solve_em.F`, `module_em.F`, `module_small_step_em.F`,
    `module_big_step_utilities_em.F` from `/home/enric/src/wrf_pristine/WRF`

## Required Work

1. Build the smallest diagnostic that proves where the remaining stage-3 /
   wrapper jump first appears:
   - after RK stage 3 but before wrapper writes,
   - p/phi refresh,
   - moist/physics state mutation,
   - LBC/nudge application,
   - halo/boundary handling,
   - or another exact boundary.
2. Compare against WRF-native or WRF-line-ported oracles wherever possible.
   Avoid JAX-vs-JAX self-acceptance.
3. Implement the smallest WRF-faithful source fix if the root is found.
4. Prove with:
   - focused term/stage evidence,
   - a h36->h37 short gate,
   - h36->h38 if the h37 change is promising,
   - focused tests.

## Acceptance Gate

A successful close requires:

- material collapse of the h36->h37 dry-mass/PSFC residual and excess outflux
  versus both old `ec4d6769` and merged rhs_ph/stage-omega state;
- no clamps, masks, tolerance relaxation, artificial damping, or host/device
  transfers inside timestep loops;
- proof objects under `proofs/v014/`;
- report under `.agent/reviews/`;
- source commit(s) on the worker branch;
- focused tests and JSON validation.

If the root is not fully fixed, produce a no-fix proof that is specific enough
for a direct next sprint and do not claim the blocker closed.

## Output

Write:

- `.agent/reviews/2026-06-11-v014-fable-stage3-wrapper-cadence.md`
- `proofs/v014/switzerland_stage3_wrapper_cadence.json`
- any focused proof script needed under `proofs/v014/`

End stdout exactly:

`FABLE STAGE3_WRAPPER_CADENCE DONE - see .agent/reviews/2026-06-11-v014-fable-stage3-wrapper-cadence.md`

## Constraints

- Do not run `ask-hermes`, Telegram, or human notification commands.
- Do not touch `/home/enric/src/canairy_waves`.
- Do not start performance audit work.
- Use the GPU only through the project lock / `scripts/run_gpu_lowprio.sh`.
- Keep Fable on whole endpoint tasks. Avoid micro-hypothesis replies.
