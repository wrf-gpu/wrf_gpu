# S4/S7 Production Contract v2 - Perturbation-FP32 Carry + Fused Acoustic/Vertical Kernel

Status: v2 execution contract, revised after D adversarial critique
`proofs/v020/plan_critique/S4_S7_PLAN_CRITIQUE.md`; ready only after G0
returns GO under the sharpened criteria below.

Audience: S4/S7 frontrunner and reviewers.

Sources: `FINAL_FP32_SPRINT_PLAN.md` sections 1 and 5, `GPT_FP32_PLAN.md`
sections 4-7, `OPUS_P34_PLAN.md` point-3 L5, `GPT_P34_PLAN.md` P3.4/P5-P6,
the CPU prototype proof in `proofs/v020/fp32_proto/REPORT.md`, and the live
host-bubble diagnostic
`<USER_HOME>/src/wrf_downscale/reports/gpu_util_diagnostic_20260620.md`.
D critique v2 amendments are binding where they narrow earlier wording.

## Objective

Implement the combined v0.20 production workstream that makes the dycore
perturbation-authoritative in fp32 and then fuses the acoustic plus
vertical-implicit solve so the remaining fp64 work is register-local rather
than resident full-grid HBM traffic.

The target is not a broad fp32 cast pass. The target is:

- hot carry stores dynamic `p'`, `ph'`, `mu'`, and `w` as fp32 in the opt-in
  fp32 mode;
- explicit static/base `PB`, `PHB`, `MUB` and base gradients stay fp64;
- linear PGF pressure/geopotential gradients are evaluated as
  `grad(base)_fp64 + grad(perturbation)_fp32`;
- nonlinear EOS/log-alpha work is handled by fp64 bracket localization, not by
  gradient splitting;
- the <=5 cancellation brackets plus the tridiagonal solve stay fp64 in
  registers inside the fused acoustic/vertical path;
- `fp64_default` remains bit-identical and runnable throughout, including
  byte-identical storage and use of legacy total fields.

## Non-Goals

- No source edit before G0-GO.
- No default-path behavior change.
- No removal of stored `p_total`, `ph_total`, or `mu_total` from
  `fp64_default`, and no replacement of default-path total-minus-perturbation
  differencing unless an empirical wrfout byte-diff proves identity.
- No global fp32 totals for `p`, `ph`, or `mu`.
- No fp32 storage for derived cancellation fields `alt`, `al_prime`/`al`, or
  `php`; they are fp64-local derived fields or exit diagnostics only.
- No new clamps, masks, finite guards, or silent domain fallback.
- No Pallas/custom-kernel integration before an XLA-first proof shows the launch
  and HBM problem remains.
- No long forecast campaign before component savepoint/oracle gates pass.
- No single-domain success claim for nested all-7 acceptance.

## Prerequisites

1. G0-GO is recorded by the manager using the v2 decision rules in Gate 0.
   G0 must report both the aggressive fp32-solve ceiling and the production
   fp64-in-register-solve configuration; GO/KILL uses the production-config
   number.
2. Sprint-A S1 lands first only in its bit-safe form: remove exact
   bitwise-duplicate `p/ph/mu` leaves while keeping stored totals in
   `fp64_default`. The total-to-base-plus-perturbation reconstruction belongs to
   S4 mixed mode only, not the default path.
3. Existing v0.19.x nightly path remains runnable. Production changes happen on
   an opt-in worker/integration branch and under additive flags.
4. Baselines exist for the exact commit: current fp64 GPU path, CPU WRF fixture
   or WRF savepoint reference where available, and current all-7 max_dom=9.
5. Acceptance uses roadmap §8.5 as binding. `acceptance_bands.py` remains the
   synthetic-proof/default importable table only until the manager freezes
   `proofs/v020/ACCEPTANCE_TABLE.md`; production forecast gates must express
   thresholds relative to the fp64-GPU-vs-CPU-WRF gap where available and must
   include no material ACC/neighborhood skill drop.

## Sprint-A S1 Versus S4 Mixed-Mode Scope

This split is binding because it protects the v0.19.x runnability constraint.

- **Sprint-A S1, default path:** remove only exact bitwise-duplicate
  `p/ph/mu` leaves while keeping the stored totals and legacy default-path
  arithmetic. It may add dtype/source audits, but it must not make
  `fp64_default` reconstruct totals from base plus perturbation.
- **S1 gate:** empirical `fp64_default` wrfout byte-diff on the nightly all-7
  config, plus the dtype matrix. The dtype matrix alone is insufficient.
- **S4 mixed path:** in the opt-in fp32/mixed mode only, drop hot stored totals
  and reconstruct totals inside named fp64 brackets or at exits from explicit
  base `BaseState` leaves plus perturbations.
- **Base reads:** explicit `BaseState.pb`, `BaseState.phb`, and `BaseState.mub`
  are read in the mixed path. The default path keeps the existing total-field
  behavior unless a byte-identical proof authorizes a narrower cleanup.

## Never-FP32-Stored Derived Fields

The following are not fp32 storage targets:

- `alt` full inverse density;
- `al_prime`/`al` inverse-density perturbation or log-alpha difference;
- `php` mass-level full geopotential;
- bracket-local EOS/log-alpha intermediates and pressure powers;
- tridiagonal coefficients/RHS/solution while the production solve is the
  fp64-in-register configuration.

They may be recomputed in fp64 inside brackets, kept in registers or
column-local scratch by S7, or reconstructed at named exits. They must not
become resident fp32 full-grid arrays as an optimization.

## Allowed Files For The Production Sprint

This contract is documentation only. The later implementation sprint may touch
only files declared in its sprint contract. Expected production ownership:

- precision and state contracts:
  `src/gpuwrf/contracts/precision.py`, `src/gpuwrf/contracts/state.py`,
  `src/gpuwrf/runtime/operational_state.py`;
- timestep orchestration and acoustic scan:
  `src/gpuwrf/runtime/operational_mode.py`;
- dycore operators:
  `src/gpuwrf/dynamics/core/small_step_prep.py`,
  `src/gpuwrf/dynamics/core/small_step_finish.py`,
  `src/gpuwrf/dynamics/core/acoustic.py`,
  `src/gpuwrf/dynamics/core/advance_w.py`,
  `src/gpuwrf/dynamics/core/calc_p_rho.py`,
  `src/gpuwrf/dynamics/core/rk_addtend_dry.py`,
  `src/gpuwrf/dynamics/acoustic_wrf.py`;
- nesting/boundary base-state plumbing:
  `src/gpuwrf/nesting/boundary_construction.py`,
  `src/gpuwrf/runtime/domain_tree.py`;
- tests, scripts, and proof outputs under `tests/`, `scripts/`, and
  `proofs/v020/`.

Any expansion beyond this list requires a contract amendment before coding.

## Additive Modes And Reversibility

Required flags/modes:

- `fp64_default`: unchanged and bit-identical, with stored totals preserved.
- `mixed_perturb_fp32_v020` or equivalent: enables S4 storage/carry.
- `s7_fused_acoustic_vertical` or equivalent: enables fused acoustic/vertical
  execution. It depends on `mixed_perturb_fp32_v020` for the fp32 production
  path but must also support an fp64-debug comparison mode.

Every field demotion is reversible independently:

- `p_prime`: fp32 storage, fp64 bracket fallback.
- `ph_prime`: fp32 storage, fp64 bracket fallback.
- `mu_prime`: fp32 storage, optional compensated leaf, fp64 bracket fallback.
- `w`: fp32 storage only after the vertical-solve oracle passes; fp64 resident
  fallback remains.
- cumulative fields: fp32 storage plus compensation only after monotonic/finite
  gates pass; fp64 accumulator fallback remains.

No domain may silently switch back to a broad fp64 path. Any fallback must be
named in the proof as a failed compact-island result.

## S4 State Contract

### Hot Carry Leaves

The `fp64_default` hot carry keeps the current stored totals. The opt-in fp32
hot carry contains:

- `p_prime_fp32` / existing `p_perturbation` as authoritative dynamic pressure;
- `ph_prime_fp32` / existing `ph_perturbation` as authoritative dynamic
  geopotential;
- `mu_prime_fp32` / existing `mu_perturbation` as authoritative dynamic dry
  mass;
- `w_fp32` once gated, otherwise `w_fp64` with the same interface;
- optional compensation leaves for `mu_prime`, pressure memory, and cumulative
  diagnostics only where drift evidence demands them.

The hot carry does not contain `p_total`, `ph_total`, or `mu_total`.
That sentence applies to `mixed_perturb_fp32_v020` only; it is explicitly false
for `fp64_default`.

### Explicit Base Leaves

The mixed path receives and reads explicit fp64 base/reference leaves from the
existing base-state contract:

- `PB` / `BaseState.pb` / `p_base_fp64`;
- `PHB` / `BaseState.phb` / `ph_base_fp64`;
- `MUB` / `BaseState.mub` / `mu_base_fp64`;
- precomputed base gradients and static metric/base products needed by PGF,
  EOS, mass denominators, and boundary construction.

Mixed-mode base leaves are not recovered from `total - perturbation` in hot
functions. Totals are reconstructed only at named exits: wrfout, WRF savepoint,
boundary interchange that explicitly requires legacy totals, and diagnostic
output. Mixed-mode restart is different: it must persist perturbations natively
or fp64 totals, never fp32 totals that must later be differenced against fp64
base.

## S4 Operator Work Items

### `small_step_prep.py`

- Require explicit base state for mixed mode; leave `fp64_default` byte-identical.
- Build prep work arrays from `(base_fp64, perturb_fp32)` rather than total
  aliases only in the mixed path.
- Keep cancellation-sensitive prep brackets fp64 but cast outputs back to hot
  dtype at explicit boundaries.
- Gate: one-substep savepoint equality in fp64-debug mode, default wrfout
  byte-identity, and no forbidden total-minus-perturbation in mixed mode.

### `small_step_finish.py`

- Split hot-loop finish from compatibility finish.
- Hot-loop finish returns authoritative perturbations only.
- Compatibility finish reconstructs `p_total`, `ph_total`, and `mu_total` only
  at exits for mixed mode; default keeps stored totals.
- Gate: `fp64_default` bit-identical; mixed HLO scan carry has no full-grid
  total aliases.

### `diagnose_pressure_al_alt` and `calc_p_rho.py`

- Rewrite mixed pressure/EOS around explicit `PB/PHB/MUB` and dynamic
  perturbations.
- Use stable forms such as ratio, `log1p`, and `expm1` where they preserve WRF
  discrete algebra.
- Keep EOS/log-alpha cancellation bracket fp64-local. The EOS is nonlinear and
  is not split as `grad(base) + grad(perturbation)`; it is protected by
  reconstructing the required total inside a fp64 bracket.
- Treat `alt`, `al_prime`/`al`, and `php` as never-fp32-stored derived fields.
  S4 alone is expected to leave these fp64 islands resident until S7 fuses them
  into registers, so S4-alone wall/HBM wins are capped even when VRAM improves.
- Gate: pressure/EOS savepoints for hydrostatic rest, warm bubble, steep
  terrain, moist column, and real WRF columns.

### Acoustic and Large-Step PGF

Files: `acoustic.py::advance_uv_wrf`,
`rk_addtend_dry.py::_absolute_diagnostics`, and
`rk_addtend_dry.py::large_step_horizontal_pgf`.

- Split pressure/geopotential/mass terms into base and perturbation parts.
- Precompute or stage base gradients fp64 once per RK stage.
- Compute perturbation gradients in fp32.
- Evaluate the final PGF cancellation bracket in fp64-local form.
- Do not apply the linear-gradient split to EOS/log-alpha terms; those remain
  bracket-localized fp64 derived work.
- Gate: acoustic PGF and large-step PGF savepoints on flat, lee-wave,
  steep-ridge, high-peak, and boundary-ring cells.

### `advance_w.py` / Implicit `w`-`phi`

- Keep coefficient construction, vertical PGF/buoyancy bracket, and
  Thomas/PCR solve fp64-local first.
- Feed fp32 perturbation inputs at the island boundary.
- Return `w` to fp32 storage only after one-column and all-7 short ladder pass.
- Gate: one-column vertical oracle, hydrostatic/rest preservation, terrain
  watch columns, and d03 qke sentinel.

### Boundary Construction

- Boundary builders receive explicit child base leaves.
- Do not pack child base leaves via hot-path total-minus-perturbation in mixed
  mode. The current fp64/default path may keep existing total-derived behavior
  only while the empirical byte-diff remains green.
- Keep `ph`, `mu`, and `w` boundary leaves protected until boundary-ring tests
  pass.
- Gate: eager-vs-fused boundary package equality for fp64/default and
  skill/stability for mixed mode.

### Restart And Output Persistence

- Wrfout/output may reconstruct legacy totals at named exits.
- Mixed-mode restart must persist `p_prime`, `ph_prime`, `mu_prime`, `w` state
  natively with fp64 base leaves, or persist fp64 totals. It must not persist
  fp32 totals and then recover perturbations by `fp32_total - fp64_base`.
- Gate: restart-write -> restart-read -> restart-continue checks recovered
  perturbations against the pre-restart perturbations, not only reconstructed
  totals.

### Compensated Sums

- Use Kahan/Neumaier only for long-lived accumulation paths proven by drift
  probes: cumulative precipitation diagnostics, pressure memory, selected
  `mu_prime` or mass-divergence accumulation.
- Do not add compensation leaves for every field by default; that erases the
  storage win.
- Gate: finite, monotone where physically cumulative, basin/domain totals
  within cumulative carve-out bands, and no coupling failure in wind/temp/cloud.

## S7 Fused Acoustic + Vertical-Implicit Kernel Contract

### Live Host-Bubble Addendum

The live nested all-7 diagnostic measured mean GPU util `79%` over a 105 s
window, with roughly `21%` lost to host-bound bubbles. The GPU clocks stayed
high and the host was single-core saturated, so this is cascade glue blocking
the GPU, not throttling. The measured host levers are important but bounded:
they are expected to recover roughly `1.2-1.3x` on the all-7 nest if the bubbles
are closed. They do not replace the v0.20 priority: fp32
perturbation-authoritative storage at DRAM-bound scale plus the 1 km capability
unlock remains the big win.

S7 therefore has two structural layers:

- **S7a host-bubble removal:** async/overlapped dispatch, double-buffered
  parent/child boundary packages, fused inter-domain LBC/feedback, and async
  history output. These can ship independently where bit-identical.
- **S7b acoustic/vertical kernel fusion:** the original fused acoustic plus
  vertical-implicit solve target that keeps fp64 brackets and the tridiagonal in
  registers and compounds with S4 fp32 carry.

### Operators To Fuse

The S7 structural target is the acoustic/RK inner graph identified in
`OPUS_P34_PLAN` L5 and `GPT_P34_PLAN` P3.4:

- acoustic substep scan in `runtime/operational_mode.py`;
- `acoustic_substep_core`;
- `advance_uv_wrf` PGF/momentum update;
- `advance_mu_t` / dry-mass acoustic update;
- `rhs_ph` / geopotential tendency update where present;
- `advance_w_wrf` coefficient build, RHS, boundary terms, and vertical solve;
- `calc_p_rho_step` pressure/rho refresh;
- stage constants and base-gradient inputs for the RK acoustic stages.

The live host-bubble result extends S7 scope beyond the acoustic/vertical core:

- parent-to-child LBC package construction and handoff inside the fused cascade;
- child-to-parent feedback/nudging where enabled, or an explicit fail-closed
  proof when feedback is not supported by the fused path;
- double-buffered boundary packages so the next child/domain launch can be
  prepared while the current GPU work runs;
- async history-output buffering for the 9-domain wrfout write path so wrfout
  writes do not block GPU stepping on the step-loop thread.

The first integrated unit is one acoustic substep. The second is one full RK
stage. The final unit is the acoustic substep loop plus vertical implicit solve
for the configured stage.

The first host-bubble unit is bit-identical async history output. The second is
async dispatch/double-buffering around existing kernels. The third is fused
device-resident LBC/feedback inside the cascade.

### S7a Host-Bubble Removal

S7a is the bounded all-7 nest lever measured by the live diagnostic. It is
shippable ahead of fp32 when it preserves bit identity.
Its target numbers are refreshed from a live diagnostic replay on the current
commit, not assumed from the stale 2026-06-20 `79%` util / `21%` bubble sample.

#### Async/Overlapped Dispatch + Double-Buffering

- Keep the GPU queue non-empty by preparing and launching the next domain/step
  while current GPU work runs.
- Double-buffer parent-to-child LBC packages and child-to-parent feedback
  buffers so host preparation does not serialize the 9-domain cascade.
- Preserve data dependency order: parent advance must produce the LBC source
  consumed by the child advance; child feedback must consume the completed child
  state.
- Gate: bit-identical wrfout versus the synchronous path, no in-loop transfers,
  no peak-VRAM regression from extra in-flight buffers, and mean GPU util/bubble
  fraction improves in a replay of the live diagnostic on the current nightly
  all-7 config.

#### Device-Resident LBC/Feedback In The Fused Cascade

- Move remaining parent-to-child LBC and child-to-parent feedback/nudging glue
  into the fused cascade program where source dependencies are device-resident.
- This kills the single-host-core serialization identified by the diagnostic.
- Keep child base leaves child-owned; mixed fp32 mode must receive explicit base
  leaves and must not resurrect `total - perturbation` base recovery.
- Fail closed for feedback modes that are not implemented in the fused path.
- Gate: force/LBC/feedback host event count drops, host single-core bubble
  fraction drops, boundary-ring diagnostics are reported per domain, and all-7
  remains bit-identical for default/fp64 structural changes or skill-accepted
  for fp32 fast mode.

#### Async History Output

- Implement the task-101 path as a background-thread or buffered 9-domain
  wrfout writer.
- Step-loop GPU dispatch must not wait for synchronous NetCDF writes except at
  explicit flush/close/restart barriers.
- This is likely bit-identical and can ship independently of fp32.
- Gate: on the exact nightly all-7 config, wrfout files are byte- or
  value-identical, all expected d01-d09 output times are present/readable,
  restart consistency holds, memory used by queued output buffers is bounded,
  and the observed ~30 s write pauses disappear or move off the GPU-stepping
  critical path.

### Required fp64 In-Register Islands

The fused kernel must not materialize broad fp64 arrays for the following. They
remain fp64 in registers or column-local scratch:

1. horizontal PGF base-plus-perturbation cancellation bracket;
2. vertical PGF / hydrostatic / buoyancy cancellation bracket;
3. EOS/log-alpha/pressure refresh cancellation bracket;
4. dry-mass/mass-denominator or inverse-mass bracket;
5. boundary `ph/w` forcing bracket where nested terrain demands it;
6. the tridiagonal/Thomas/PCR vertical implicit solve.

The contract target is <=5 cancellation brackets plus the tridiagonal solve, as
framed in `FINAL_FP32_SPRINT_PLAN` and `OPUS_P34_PLAN`; if implementation splits
one bracket for correctness, the proof must name the split and its HBM/liveness
cost.

### XLA-Fusion-First Path

Before Pallas/custom kernels:

- reduce scan carry to evolving leaves only;
- close over fp64 stage constants and base gradients;
- raise or tune acoustic unroll only with compile-key/RSS measurement;
- split helper functions so fp64 brackets have explicit input/output dtypes;
- lower HLO for one substep and one RK stage;
- prove launch count, fp64 live arrays, convert count, and temp memory move.

Prediction before starting: the current `lax.scan` Thomas forward/back sweep is
not expected to fuse into the acoustic kernel under XLA-first. The expected
XLA-only win is launch/convert reduction in the surrounding elementwise graph.
If the tridiagonal remains a scan/loop-region kernel and the fp64 derived
islands remain HBM-resident, that is the predicted trigger for the Pallas ADR,
not a surprise failure.

Pass criteria for XLA-only S7:

- the nsys baseline reconciles the actual solve path and kernel name:
  current code defaults to `lax.scan` Thomas, while any
  `pcrGtsvBatchSharedMemKernel*` finding means a different XLA/cuSPARSE solve
  path was selected and must be named;
- the actual tridiagonal-kernel or scan-region count/time drops, or the proof
  shows it is not the remaining top cost;
- `loop_multiply_fusion_*` and `input_reduce_fusion_*` instance counts drop
  materially;
- no full-grid fp64 totals in scan carry;
- temp/argument bytes lower than the fp64 baseline;
- one-substep and one-RK savepoints pass.

If XLA-only fails exactly as predicted for the Thomas solve or fp64 islands,
write the exact cause and open the Pallas/custom-kernel ADR. Do not spend a
sprint rediscovering that `lax.scan` will not make the tridiagonal
register-resident.

### Pallas Fallback

Pallas is allowed only after the XLA-first proof fails and the manager approves
the custom-kernel ADR/governance step.

Pallas target shape:

- one block per column or small column tile;
- vertical dimension in registers/shared memory;
- fp32 perturbation storage at kernel boundaries;
- fp64 cancellation brackets in registers;
- tridiagonal solve local to the column;
- no host/device transfer inside the timestep loop;
- separate fp64-debug mode for oracle comparison.

Pallas integration gates:

- savepoint oracle for one substep and one RK stage;
- launch-count drop and top-kernel wall reduction in nsys;
- ncu/nsys register-spill and occupancy probe: registers per thread, achieved
  occupancy, local-memory load/store bytes, and spill stores for the fused
  kernel;
- no compile-cache/RSS blowup;
- all-7 short ladder before any 24 h run;
- documented fallback to the XLA/eager path.

## Validation Sequence

### Gate 0: G0-GO Precondition

No S4/S7 implementation begins before G0-GO is recorded. The G0 artifact must
state speed-GO, capability-GO, redirect, or kill. This contract executes only
under GO.

Required G0 A/Bs:

- **aggressive ceiling:** fp32 storage for `p'`, `ph'`, `mu'`, `w` and fp32
  tridiagonal/PCR solve;
- **production config:** fp32 storage for `p'`, `ph'`, `mu'`, `w` with the
  cancellation brackets and tridiagonal solve fp64 in registers or local
  scratch.

The GO decision uses the production-config number. The aggressive ceiling is
context only.

Speed-GO requires transient majority fp32-able, peak VRAM lower, and the
large-grid warm dycore/acoustic component on a >2x trajectory under the
production config, or a bottleneck-moved escape that is valid under the rule
below.

Speed-KILL is allowed only after profiler classification:

- **real kill:** fp32 did not shrink live/transient storage and dominant
  fp32/fp64 kernel walls are identical, reproducing the 1.106x signature;
- **not a kill:** fp32 did shrink storage/transient but wall is capped by launch
  count plus HBM-resident fp64 islands. This is GO-to-S7, not KILL.

Never KILL the fp32 path on an unfused island-resident measurement.

The bottleneck-moved GO escape is valid only if the new bottleneck is addressed
by S4/S7 or by a named scheduled follow-on. If it moves to a component outside
the funded scope with no scheduled fix, the outcome is REDIRECT to P3/P4,
multi-GPU, or that component's workstream, not Speed-GO.

### Gate 1: Static Source And HLO Audits

Required proof objects:

- empirical `fp64_default` wrfout byte-diff on the nightly all-7 config after
  the bit-safe S1 dedup and after each structural default-path change;
- forbidden-base-reconstruction audit for mixed hot functions:
  `p_total - p_perturbation`, `ph_total - ph_perturbation`,
  `mu_total - mu_perturbation`;
- HLO dtype/liveness report: fp64 ops, converts, scan carry leaves,
  argument/output/alias/temp bytes;
- transfer audit: no callback, no device-get, no H2D/D2H in the timestep loop;
- precision matrix test: `fp64_default` unchanged and mixed mode field dtypes
  match the contract. This matrix does not replace the empirical byte-diff.

### Gate 2: Real Fixture / Savepoint Closure

This closes the CPU-prototype synthetic-only caveat.

Required real proof cases:

- WRF or fp64-GPU savepoint for `small_step_prep` and `small_step_finish`;
- pressure/EOS savepoint for `diagnose_pressure_al_alt` and `calc_p_rho_step`;
- acoustic PGF savepoint for flat/ocean, lee-wave, steep-terrain, high-peak,
  and boundary-ring cells;
- `advance_w` one-column vertical solve oracle including steep terrain and
  nested boundary forcing;
- real all-7 d03 1 km qke sentinel case;
- restart/write/read perturbation persistence and recovery, plus reconstructed
  total consistency at exits.

Pass criteria:

- naive fp32-total baseline reproduces local cancellation failure where
  applicable;
- perturbation-authoritative path does not reproduce that failure;
- this local-cancellation test is the primary numerical veto and is not
  redundant with Gate 4 skill bands. The prototype already showed that a naive
  total-field path can pass bands while failing cancellation.
- fp64-debug compact path matches existing fp64 path to debug tolerance;
- mixed path stays within explicit savepoint tolerances and, where relevant,
  the current acceptance-band metrics.
- mixed-mode restart persists perturbations natively, and the
  restart-continue test recovers `p'`, `ph'`, `mu'`, and `w` to the pre-restart
  tolerance before checking reconstructed totals.

### Gate 3: Stability Ladder

Run each new S4/S7 increment through:

1. one acoustic substep;
2. one RK stage;
3. one full dynamics step;
4. 20 model steps;
5. 0.1 h;
6. 1 h;
7. 3 h;
8. 6 h;
9. 24 h;
10. 48 h;
11. 72 h;
12. 120 h final candidate.

The ladder is cadence-dependent. If P4 timestep or `n_sound` changes land
before S4/S7 integration, rerun the ladder under the new cadence rather than
reusing old fp32 drift evidence.

Abort on:

- NaN/Inf in protected fields;
- negative pressure or nonpositive dry mass;
- pressure/geopotential/mass boundary-ring growth;
- qke nonfinite or spike in d03-d09;
- horizontal/vertical wind runaway;
- cloud water/ice collapse or explosion;
- superlinear divergence growth in wind/temp/cloud;
- hidden fallback or in-loop transfer.

### Gate 4: Acceptance Bands And Skill

Use roadmap §8.5 as binding. `proofs/v020/fp32_proto/acceptance_bands.py`
is the current importable synthetic/default band spec until the manager freezes
`proofs/v020/ACCEPTANCE_TABLE.md`; production gates must reconcile with the
relative-to-fp64-gap rule:

- wind: `U10/V10`, lowest-level `U/V`, mid-level winds, boundary-ring winds;
- temperature: `T2`, lowest-model-level temperature, `THETA`, boundary-ring
  temperature;
- cloud/radiation: hydrometeors, cloud fraction, column condensate,
  `SWDOWN/LWDOWN`, radiative heating;
- pressure/mass/geopotential: `PSFC`, dry-mass drift, boundary-ring growth;
- `QVAPOR` and cumulative diagnostics: carve-outs from tight parity, not from
  finite/nonnegative/stability.

Additional binding requirements:

- thresholds expressed relative to the fp64-GPU-vs-CPU-WRF gap where available,
  so fp32 does not add a second port-sized error;
- no material ACC/neighborhood skill drop for wind/cloud at 72-120 h;
- dry-mass and total-water budget conservation checks by column/domain, with a
  stated tolerance. `QVAPOR` distribution has a parity carve-out, not a water
  conservation exemption;
- cumulative monotonicity and compensated-sum drift checks for cumulative
  carve-outs;
- existing limiter/clamp hit-counts, including EOS `_safe_pressure`,
  `_safe_alt`, and `jnp.maximum(..., 1e-12)` paths, reported before and after
  fp32. New guards remain forbidden.

Final production acceptance requires domain-by-domain and aggregate all-7
metrics at 24/48/72/120 h. The frozen grid-delta manifest is diagnostic only
once the skill gates pass.

### Gate 5: Performance And Memory Proof

Required post-G0/S7 artifacts:

- warm compile-excluded all-7 wall/fc-hour and large-grid component timing;
- current fp64 GPU baseline and CPU denominator where relevant;
- live host-bubble replay: mean/median GPU util, idle-bubble fraction,
  high-low transition rate, host core saturation, and output-pause attribution;
- nsys launch census: kernels per step, graph launches, top 20 kernels,
  actual tridiagonal solve kernel or scan-region names,
  `pcrGtsvBatchSharedMemKernel*` only if that path is active, and tiny-kernel
  counts;
- ncu/nsys register pressure probe for the fused/Pallas kernel: registers per
  thread, achieved occupancy, local-memory load/store bytes, and spills;
- host ledger: XLA dispatch, LBC package construction, feedback/nudging,
  history-output write time, and explicit sync counts;
- HLO memory analysis: argument/output/alias/temp bytes and fp64 islands;
- peak VRAM and live-array flatness across output groups;
- transfer audit;
- cold compile wall, peak host RSS, compile-key count;
- per-domain/per-component Amdahl table.

Performance claims are invalid without the exact remaining fp64-island matrix
and the reason each island remains. If fp32 storage shrinks but wall is capped
by unfused fp64 islands or launch count, the result is not an fp32 KILL; it is
the S7 target condition.

## Acceptance Criteria

The production S4/S7 workstream is complete only when all are true:

- `fp64_default` is bit-identical and still runnable, with stored totals kept
  byte-identical unless an empirical byte-diff proves a narrower cleanup.
- Mixed mode has no hot-loop fp32 total differencing.
- Explicit fp64 base leaves and fp32 perturbation leaves are the authoritative
  mixed-mode state contract.
- Real WRF/savepoint fixtures close the synthetic-only caveat from the CPU
  prototype.
- Perturbation fp32 passes the local cancellation veto, the cadence-correct
  stability ladder, and the §8.5 acceptance contract.
- Mixed-mode restart persists perturbations natively and restart-continue
  recovers perturbations, not only reconstructed totals.
- Dry-mass and water budgets close to declared tolerances.
- S7a lowers measured host bubbles without changing physics, or names the exact
  remaining host serialization.
- S7b lowers launch/HBM pressure with either XLA fusion or approved Pallas.
- Remaining fp64 islands are register-local or explicitly justified, with
  register spill/occupancy/local-memory evidence for fused kernels.
- Nested all-7 max_dom=9 passes through 24-120 h with wind/temp/cloud skill
  preserved.
- Proof artifacts name every fallback, every failed field demotion, and every
  remaining island.

## v2 Critique Resolution Ledger

- **B1:** Scope total removal to `mixed_perturb_fp32_v020` only. `fp64_default`
  keeps stored totals and legacy differencing; S1 removes only exact duplicate
  leaves and must pass an empirical nightly wrfout byte-diff. Mixed mode reads
  explicit `BaseState.pb/phb/mub`.
- **B2a:** G0 reports both aggressive fp32-solve and production
  fp64-in-register-solve A/Bs; the GO decision uses the production-config
  number.
- **B2b:** Speed-KILL requires profiler classification. Identical fp32/fp64
  walls plus no storage shrink is a real kill; storage shrink capped by launch
  count or HBM-resident fp64 islands is GO-to-S7, never KILL.
- **B2c:** Bottleneck-moved Speed-GO is valid only when the new bottleneck is
  handled by S4/S7 or by a named scheduled follow-on; otherwise the outcome is
  REDIRECT.
- **SF1:** `alt`, `al_prime`/`al`, and `php` are never-fp32-stored; EOS/log
  alpha are bracket-localized, not gradient-split. S4-alone VRAM can improve
  while wall/HBM remains capped until S7 registerizes these islands.
- **SF2:** Gate 2 savepoint local-cancellation is the primary numerical veto.
  It remains separate from skill bands because the naive total path passed the
  bands in the prototype while failing cancellation.
- **SF3:** `acceptance_bands.py` is demoted to synthetic/default importable
  bands until a frozen `ACCEPTANCE_TABLE.md`; production acceptance follows
  roadmap §8.5 relative-to-fp64-gap and ACC/neighborhood rules.
- **SF4:** XLA-first now predicts the `lax.scan` Thomas solve will not collapse
  into the acoustic kernel. The S7 nsys baseline must reconcile actual
  tridiagonal kernel names, including whether any `pcrGtsvBatchSharedMemKernel`
  path is active, before targeting a kernel.
- **SF5:** S7 gates include ncu/nsys register-pressure, achieved occupancy, and
  local-memory spill probes.
- **SF6:** Mixed restart persists perturbations natively with fp64 base, or
  fp64 totals; it must not recover perturbations from fp32 totals. A
  perturbation-recovery restart test is required.
- **SF7:** Gate 4 includes dry-mass and total-water budget conservation; QVAPOR
  has a parity carve-out, not a conservation carve-out.
- **N1:** Stability ladder evidence is cadence-dependent and must be rerun if
  P4 timestep or `n_sound` changes land first.
- **N2:** S7a async output/double-buffer bit-identity is proven on the exact
  nightly all-7 config, not only a reduced test.
- **N3:** Host-bubble targets are tied to a fresh diagnostic replay on the
  current commit, not the stale 2026-06-20 percentage.
- **N4:** Existing EOS/limiter clamp hit-counts are reported before and after
  fp32; new guards remain forbidden.

## Rollback

- Disable `s7_fused_acoustic_vertical` to return to the unfused mixed path.
- Disable async dispatch/double-buffering to return to synchronous cascade
  dispatch.
- Disable fused LBC/feedback to return to eager or existing fused-cascade glue.
- Disable async history output to return to synchronous wrfout writes.
- Disable individual field demotions to return `p'`, `ph'`, `mu'`, or `w` to
  fp64-in-bracket or fp64-resident fallback.
- Disable `mixed_perturb_fp32_v020` to return to `fp64_default`.
- Revert Pallas/custom-kernel registration independently of S4 state plumbing.
- If any rollback is used in a proof run, the report must identify it and cannot
  claim full S4/S7 success.

## Review Requirements

- Manager review before implementation starts post-G0-GO.
- Independent reviewer before Pallas/custom kernel integration.
- Reviewer must inspect HLO/liveness and savepoint artifacts before 24 h runs.
- Final closeout must include objective, files changed, commands run, proof
  objects, unresolved risks, and next decision needed.
