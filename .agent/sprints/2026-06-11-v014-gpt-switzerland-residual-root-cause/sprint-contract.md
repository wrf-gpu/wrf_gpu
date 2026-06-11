# Sprint Contract: v0.14 GPT Switzerland Residual Root Cause

Date: 2026-06-11
Owner: manager
Assignee: GPT-5.5 xhigh, tmux
Status: READY

## Objective

Independently diagnose and, if locally provable, fix the remaining
Switzerland/Gotthard h36->h37 dry-mass/PSFC field-parity blocker.

This is a hypothesis-neutral debug sprint. The manager's current leading
candidate is an interior acoustic `phi/p` mechanism, with `advance_w`,
`rw_tend`/`ph_tend`, and post-stage pressure refresh as plausible locations.
Treat that only as a candidate from prior evidence, not as a required
conclusion. You must build your own ranked hypothesis ledger from the artifacts
below, decide what evidence supports or falsifies each candidate, and pursue the
fastest rigorous proof path to a fix or a stronger narrowing.

Fable's and the manager's current interpretation is useful context, not a
constraint: Fable thinks the boundary-band lane is now mostly refuted as the
driver and that an interior hydrostatic `phi/p` mechanism is the next best
target; the manager agrees this is the strongest current clue. You may and
should reject that interpretation if your evidence points elsewhere.

## Current Known Facts

- Do not start a long Switzerland 72h GPU run. The mandatory long GPU gate is
  blocked until the h36 storm-state short gate collapses or is formally bounded.
- Canary L2 d02 72h is accepted as bounded/proceed; do not spend this sprint on
  Canary.
- CPU-WRF Switzerland 72h truth is complete at:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`.
- Already merged/proven lanes:
  - LBC clock root cause.
  - HPG native-face / `hypsometric_opt=2` LOG-form diagnostics.
  - real-case `rhs_ph`, stage omega edge handling, WRF dycore constants.
  - WRF specified LBC cadence + order-5 specified-boundary advection
    degradation, flag-gated/default-off, with GPT verifier-required `w`
    wrapper correction.
- Latest gate evidence says the boundary band is more WRF-faithful but not the
  venting driver:
  - CPU h36->h37 residual: `+5.178443877551032 Pa/cell/h`.
  - old `ec4d6769`: residual `-32.686352040816345`, excess outflux
    `-28.614795918367335`.
  - hypso `3d0b439c`: residual `-27.697448979591826`, excess
    `-28.3281887755102`.
  - rhs_ph `79b0c22e`: residual `-21.882908163265313`, excess
    `-27.203954081632645`.
  - + specified LBC cadence: residual `-20.302933673469383`, excess
    `-30.68188775510204`.
  - + cadence + advection degradation: residual `-21.064285714285717`,
    excess `-32.86951530612245`.
- Prior reports observe a broad interior hydrostatic-looking pair in the stage
  comparisons: `ph` sinks while `p` rises, with tiny `mu` mean error. This is an
  important clue but not a mandate.

## Required Context

Read first:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/skills/validating-physics/SKILL.md`
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
- `proofs/v014/gpt_stage3_wrapper_verifier.md`
- `.agent/reviews/2026-06-11-v014-fable-stage3-wrapper-cadence.md`
- `proofs/v014/switzerland_stage3_wrapper_cadence.json`
- `proofs/v014/switzerland_acoustic_continuation.json`
- `proofs/v014/switzerland_acoustic_substep_blocker.json`

Relevant source areas, not exclusive:

- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/dynamics/core/advance_w.py`
- `src/gpuwrf/dynamics/core/acoustic_wrf.py`
- `src/gpuwrf/dynamics/core/rhs_ph.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/boundary_apply.py`
- WRF pristine source under `/home/enric/src/wrf_pristine/WRF`, especially
  `dyn_em/module_small_step_em.F`, `dyn_em/solve_em.F`,
  `dyn_em/module_big_step_utilities_em.F`, and `dyn_em/module_advect_em.F`.

## Required Work

1. Build a compact hypothesis ledger before coding:
   - list the most plausible root classes;
   - evidence for and against each;
   - cheapest falsification/proof test;
   - expected performance impact if fixed.
2. Choose and execute the fastest rigorous proof path. Prefer minimal
   savepoint/oracle loops over long forecast runs. Use WRF-native dumps,
   WRF-line-ported oracles, conservation/budget checks, and existing h36
   artifacts wherever possible. Avoid JAX-vs-JAX self-acceptance.
3. If the strongest path is the interior `phi/p` clue, split the responsible
   stage/substep terms without assuming the root:
   - compare pre-operator inputs against WRF first;
   - then compare RHS terms, implicit coefficients/solve outputs, and final
     updates;
   - if those pass, move downstream to small-step finish, pressure/rho refresh,
     or wrapper state mutation.
4. If you find a local WRF-faithful source fix, implement the smallest fix on
   your worker branch, add/adjust focused tests/proofs, and run the shortest
   h36 gate that proves material improvement.
5. Continue autonomously until one of the stop criteria below is met. Do not
   stop merely because one diagnostic was run or because the manager's initial
   hypothesis was falsified; use your context to choose the next best proof
   loop.

## Stop Criteria

Return control to the manager only when at least one is true:

- `FIXED`: you have implemented a local WRF-faithful fix and proved material
  h36->h37 improvement with focused gates/tests.
- `LOCAL_FIX_PROPOSED`: you found a specific source defect and a minimal patch,
  but a gate needs manager-controlled GPU/time or merge coordination.
- `NARROWED_NO_FIX`: you have falsified enough candidates and isolated the
  first wrong operator/term/state boundary so the next sprint has a concrete,
  single proof/fix target.
- `REJECTED_PRIOR_HYPOTHESIS`: you can prove the Fable/manager leading
  `phi/p` interpretation is wrong and have a better evidenced root class.
- `METHOD_LIMIT`: after at least three substantial proof attempts or one
  purpose-built discriminator, the remaining path requires a new WRF-native dump,
  long GPU gate, scarce Fable escalation, or a larger architecture decision.

Do not claim progress based only on plausibility. Report excluded hypotheses
and why.

## Acceptance Gate

A successful close requires:

- the h36->h37 dry-mass/PSFC short gate materially collapses or is formally
  bounded against CPU truth and previous baselines;
- proof object(s) under `proofs/v014/`;
- source commit(s) if code changed;
- focused tests or proof commands recorded exactly;
- no clamps, masks, tolerance relaxation, artificial damping, or host/device
  transfers inside timestep loops;
- no long 72h GPU gate.

Partial success is acceptable only if it meets one of the stop criteria and
creates a stronger, falsifiable narrowing than the current evidence.

## Output

Write:

- `.agent/reviews/2026-06-11-v014-gpt-switzerland-residual-root-cause.md`
- proof scripts / JSON / markdown under `proofs/v014/` as needed

If source changes are made, commit them on your worker branch.

Report must include:

- verdict: `FIXED`, `LOCAL_FIX_PROPOSED`, `NARROWED_NO_FIX`, or `REJECTED_PRIOR_HYPOTHESIS`;
- compact hypothesis ledger;
- exact evidence chain and commands run;
- files changed and proof objects produced;
- h36 gate result if run;
- runtime/performance risk;
- next manager decision, max 8 bullets.

End stdout exactly:

`GPT SWITZERLAND_RESIDUAL_ROOT_CAUSE DONE - see .agent/reviews/2026-06-11-v014-gpt-switzerland-residual-root-cause.md`

## Constraints

- Do not use `ask-hermes`, Telegram, or human notification commands.
- Do not touch `/home/enric/src/canairy_waves`.
- Do not dispatch Fable/Mythos or any other worker.
- Do not run long 72h GPU gates.
- Use GPU only for short focused proof gates and only if the GPU is free.
- Treat manager hypotheses as candidates, not truth.
