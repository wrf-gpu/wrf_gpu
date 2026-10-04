# GPT fp32/mixed-precision plan for v0.20.0

Date: 2026-06-21

Analyst: GPT-5 Codex, independent analyst 1 of 2

Scope: strategic analysis plus step-by-step implementation and validation plan
for the v0.20.0 fp32/mixed-precision kernel rewrite. I did not run GPU
benchmarks and did not edit model source code. This report is the proof object.

Primary target: a materially faster production kernel path, aiming at the
compact, perturbation-authoritative, explicit-fp64-island design. The goal is
not a small fp32-downcast win while the acoustic and pressure graph remains
fp64-dominant. The benchmark that must remain functional is nested all-7 with
`max_dom=9`.

## 1. Executive verdict

The v0.20 path should not be "turn on fp32" and should not be the older
conservative mixed plan where the full acoustic core remains fp64. Both are
known traps:

1. Naive global fp32 is numerically invalid. It destroys small pressure,
   geopotential, and dry-mass perturbations by representing them as differences
   of large totals in fp32.
2. Conservative mixed precision has repeatedly stalled near 1.0x to 1.1x
   because the hot step still carries broad fp64 state, full-grid total aliases,
   fp64 pressure/acoustic scratch, and fp64 conversions through the scan.
3. Thompson and other physics work that has been tested is often
   launch/bandwidth-bound, so changing arithmetic dtype by itself is not a large
   speed lever.
4. Current real-case production still defaults to `force_fp64=True`, and the
   `mixed_perturb_fp32` mode is contract/plumbing rather than a consumed hot-path
   implementation.

The only credible path to a large win is structural:

* Make `p_perturbation`, `ph_perturbation`, and `mu_perturbation`
  authoritative in the timestep.
* Carry explicit fp64 base/reference leaves outside the hot perturbation state.
* Remove `p_total`, `ph_total`, and `mu_total` full-grid aliases from the hot
  acoustic/RK carry.
* Reconstruct totals only at I/O, restart, lateral boundary interchange, and
  selected physics diagnostics.
* Keep fp64 islands local and auditable: the cancellation brackets, selected
  vertical solve pieces, mass/pressure accumulation guards, boundary ph/w
  forcing, surface/PBL sensitive fields, and accumulators.

The target is aggressive but falsifiable. If the speedup does not materialize,
the v0.20 proof must name the exact reason from HLO liveness, convert counts,
temp-memory budget, kernel-launch profile, and per-component timings. "fp32 did
not help" is not an acceptable conclusion unless it is tied to a measured cause,
for example broad fp64 liveness from remaining aliases, XLA promoting a whole
subgraph around a small bracket, launch-bound physics dominating the nested
step, or fp64 vertical/PGF islands consuming too much of the step.

## 2. Evidence read first

I read the required historical fp32/kernel documents and current source anchors,
including:

* `proofs/v020/fp32_analysis/BRIEF.md`
* `.agent/decisions/KERNEL-OPTIMIZATION-FINDINGS-FINAL.md`
* `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`
* `.agent/decisions/ADR-031-mixed-perturb-fp32-acoustic-DRAFT.md`
* `.agent/decisions/ADR-003-dycore-precision.md`
* `.agent/decisions/ADR-007-precision-policy.md`
* `.agent/decisions/2026-06-04-v0100-kernel-optimization-superplan.md`
* `PRECISION_POLICY.md`
* `FP32_GRADIENT_PROBLEM_OPEN_CHALLENGE.md`
* `proofs/perf/fp32_downcast_plan.md`
* `proofs/perf/fp32_downcast_spec.md`
* `proofs/v018/k1_fp32_stability_scoping.md`
* `proofs/thompson_perf/kernel_lever_summary.json`
* `proofs/thompson_perf/fp32_vs_fp64_oracle_diff.json`
* prior `2026-06-08-gpt-fp32-*` dispatches
* `tests/test_fp32_divergence_growth_metric.py`
* `tests/test_m6_precision_matrix.py`
* `scripts/precision_bench.py`

I also checked current implementation sites in:

* `src/gpuwrf/contracts/precision.py`
* `src/gpuwrf/contracts/state.py`
* `src/gpuwrf/runtime/operational_mode.py`
* `src/gpuwrf/runtime/operational_state.py`
* `src/gpuwrf/dynamics/core/small_step_prep.py`
* `src/gpuwrf/dynamics/core/small_step_finish.py`
* `src/gpuwrf/dynamics/core/acoustic.py`
* `src/gpuwrf/dynamics/core/advance_w.py`
* `src/gpuwrf/dynamics/core/calc_p_rho.py`
* `src/gpuwrf/dynamics/acoustic_wrf.py`
* `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
* `src/gpuwrf/dynamics/flux_advection.py`
* `src/gpuwrf/dynamics/explicit_diffusion.py`
* `src/gpuwrf/coupling/physics_couplers.py`
* `src/gpuwrf/physics/surface_layer.py`
* `src/gpuwrf/coupling/boundary_apply.py`

## 3. Why naive fp32 and older mixed plans stall

### 3.1 Global fp32 fails for a mathematically specific reason

The pressure-gradient and acoustic update operate on small perturbations sitting
on top of large base fields. The dangerous pattern is:

```text
small perturbation = fp32(total large field) - fp32(base large field)
```

or its equivalent:

```text
base = total - perturbation
later small quantity = (base + perturbation) - base
```

In fp32 this discards mantissa bits before the small signal is used. The historic
probes already showed the separation clearly: absolute-total fp32 pressure
cancellation was bad, while perturbation-form fp32 recovered the small signal by
roughly six orders of magnitude. This is the central numerical fact. It is not a
minor tolerance issue and cannot be fixed by widening acceptance bands.

Current code still contains hot-path total-minus-perturb reconstruction sites,
including:

* `_acoustic_core_state` and `_acoustic_core_state_from_prep` in
  `runtime/operational_mode.py`
* `_refresh_grid_p_from_finished` in `runtime/operational_mode.py`
* `small_step_prep_wrf`
* `small_step_finish_wrf`
* `diagnose_pressure_al_alt`
* `_absolute_diagnostics` and `large_step_horizontal_pgf`

Those sites are legitimate on the fp64 path. In a mixed fp32 path they are
either invalid or they force XLA to keep broad fp64 totals alive. v0.20 has to
remove them from the hot loop for the mixed mode, not just cast around them.

### 3.2 Conservative mixed precision keeps the expensive graph alive

The older downcast specs correctly identified safe fp32 fields:
`u`, `v`, `theta`, `qv`, hydrometeors, selected boundary leaves, and many physics
outputs. They also kept `p`, `ph`, `mu`, `w`, the acoustic internals, pressure
diagnostics, PGF, surface handles, and accumulators fp64.

That design is a useful negative control but not a 4x plan. It still leaves the
dominant dycore/acoustic graph in fp64. It also introduces casts at the boundary
of a broad fp64 island:

```text
fp32 state -> upcast into acoustic -> do almost everything fp64 -> downcast out
```

If the acoustic/RK carry still contains full fp64 `p_total`, `ph_total`,
`mu_total`, `p_perturbation`, `ph_perturbation`, `mu_perturbation`, `w`, `theta`
work arrays, `pm1`, `ph_1`, `t_2ave`, `w_save`, and pressure scratch, XLA has no
reason to produce a compact fp32-dominant kernel. The broad island becomes the
kernel.

### 3.3 Amdahl bound explains why broad islands cap speed

The invalid all-fp32 proxy measured around 4.3x on the relevant compute-bound
kernel. If the fp32 bulk is at most 4.3x faster and fraction `f` of the hot step
remains fp64-speed, the idealized speedup is:

```text
speedup = 1 / (f + (1 - f) / 4.3)
```

This means:

* `f = 0.30` caps speedup near 2.0x
* `f = 0.20` caps speedup near 2.6x
* `f = 0.10` caps speedup near 3.3x
* `f = 0.05` caps speedup near 3.8x
* `f = 0.02` caps speedup near 4.1x

Therefore a "massively faster" path must make the effective fp64 island cost
roughly single-digit percent of the dycore/acoustic step, or it must also win by
reducing liveness, temp memory, launch overhead, and rematerialization. Keeping a
large pressure/acoustic/w island fp64 guarantees a modest speedup regardless of
how many peripheral fields are downcast.

### 3.4 Existing source mechanics can silently defeat fp32

The current code has several precision-defeat mechanisms that v0.20 must audit:

* `daily_pipeline.py` sets the real path to `force_fp64=True`; this is currently
  intentional because direct fp32 perturbation loss detonates.
* `_enforce_operational_precision` upcasts all non-fp64 fields when
  `force_fp64=True`; when false it casts to the precision matrix.
* `State.replace` casts updates to the current field dtype by default. This is
  good for stability, but it can hide attempts to demote or promote fields
  unless `_cast=False` is used at intentional mode boundaries.
* The physics couplers now use `_output_dtype(state, field)` so force-fp64 truly
  stays fp64 through Thompson/MYNN/RRTMG, but that also means fp32 mode depends
  on live state dtype rather than the frozen matrix alone.
* Python `float64` literals, `jnp.asarray(..., dtype=jnp.float64)`, result-type
  promotion, scatter buffers, and metric helper dtypes have already defeated
  prior downcast attempts.

### 3.5 Some subcomponents are not fp64-ALU limited

The v0.15 kernel findings and Thompson proof objects show a repeated pattern:
many production steps are dominated by thousands of tiny dependent kernels,
launch overhead, memory traffic, and liveness rather than pure fp64 ALU rate.
Thompson sedimentation on a real tiled WRF fixture was about 85 percent
launch/bandwidth-bound; fp32 work dtype did not make it faster.

This matters strategically. v0.20 should not spend its main risk budget on
microphysics fp32 arithmetic as a speed lever. Physics precision still matters
for skill and VRAM, but the massive speedup path is compacting the dycore/acoustic
state and removing broad fp64 liveness.

## 4. Precision architecture for v0.20

### 4.1 Required invariant

For `mixed_perturb_fp32`, the timestep must be perturbation-authoritative:

```text
hot state carries p', ph', mu'
explicit fp64 base carries pb, phb, mub and selected derived base metrics
hot loop never recovers base as total - perturbation
hot loop never makes p_total/ph_total/mu_total full-grid aliases except at named exits
```

The hot loop may locally form fp64 totals inside a short bracket if the bracket
is proven necessary and the HLO proves it does not keep broad totals live across
the scan.

### 4.2 Target storage by variable group

Initial target storage for the mixed path:

| Group | Hot storage target | Notes |
| --- | --- | --- |
| `u`, `v` | fp32 | Upcast only for local PGF/coriolis brackets if proven necessary. |
| `theta` | fp32 | Store perturbation or absolute theta consistently; avoid fp64 `theta` work arrays except local diagnostics. |
| `qv`, hydrometeors, number concentrations, cloud fraction leaves | fp32 | Existing QVAPOR caveat applies; validate by skill and stability, not tight RMSE. |
| `p_perturbation`, `ph_perturbation`, `mu_perturbation` | fp32 initially, with optional fp64 compensation leaves for selected accumulators | These are authoritative. No totals in hot carry. |
| `p_base`, `ph_base`, `mu_base` and base-derived terrain/reference fields | fp64 read-only/static or slowly refreshed outside hot loop | Do not recompute from totals in the scan. |
| `w` | fp64 initially, demotion candidate after vertical-solve evidence | W is tightly coupled to ph and nesting boundary forcing. |
| `qke`, `qsq`, PBL sensitive fields | fp64 initially | Current precision matrix already promoted `qke` after a d03 1 km nonfinite signature. |
| Surface flux handles, accumulators, restart/wrfout totals | fp64 | Accumulators may be loose for parity but must not overflow or become unphysical. |
| Boundary `ph`, `mu`, `w` leaves and nested ph/w forcing | fp64 | Protect nested all-7. |
| Boundary `u`, `v`, `theta`, `qv` leaves | fp32 candidate | Validate by nested boundary stability. |

### 4.3 Initial fp64 islands

Start with these islands fp64, then shrink only with evidence:

1. Static base/reference fields: `pb`, `phb`, `mub`, terrain/base height,
   base inverse-density/log-alpha derivatives, and selected map/metric constants
   if they participate in cancellation.
2. Final PGF cancellation brackets in acoustic `advance_uv_wrf` and
   large-step `large_step_horizontal_pgf`.
3. Pressure/EOS refresh bracket in `diagnose_pressure_al_alt` and
   `calc_p_rho`, including `pm1`/`smdiv` pressure memory until compensated
   alternatives are proven.
4. `advance_w_wrf` coefficient build, buoyancy/PGF cancellation terms, and
   Thomas solve until a vertical-column oracle and nested forecast evidence
   pass.
5. Boundary `ph`/`w` forcing and nesting interpolation/feedback involving
   mass/geopotential.
6. Surface-layer flux inputs and PBL turbulent-energy fields, at least through
   the first stable nested 24h proof.
7. Restart/wrfout reconstruction and cumulative diagnostic writes.

The key is not that these islands are all tiny at first. The key is that each
island has an owner, a proof, and an HLO budget. Any island that expands by
promoting full acoustic state is a no-go.

### 4.4 HLO budget that defines "compact"

The mixed path should not be accepted as "compact explicit islands" until HLO
and profiler artifacts show all of the following:

* No `p_total`, `ph_total`, or `mu_total` full-grid arrays in the scan carry.
* No hot-path `total - perturbation` reconstruction for base fields.
* Full-grid fp64 arrays live across the acoustic scan limited to explicit base
  leaves, `w` if still locked, and named stateful accumulators.
* PGF/EOS/vertical fp64 temporaries are local to their brackets and do not
  become long-lived scan carry or multi-stage remat arenas.
* Convert counts are bounded and placed at island boundaries, not interleaved
  through every stencil expression.
* Argument/result bytes and temp memory drop substantially versus the current
  fp64 path. A practical early threshold is at least 40 percent lower dycore HLO
  temp/argument footprint before attempting long forecast gates.
* Host/device transfer audit remains clean: no in-loop transfer, no callback,
  no device get, no hidden debug materialization.

## 5. Step-by-step implementation plan

### Phase 0: freeze the contract and acceptance policy

Deliverables:

* A sprint contract that explicitly targets `mixed_perturb_fp32` compact
  islands, not broad acoustic upcast/downcast.
* A new mode name if useful, for example `mixed_perturb_fp32_v020`, so the old
  scaffold and the new compact implementation can be compared.
* Default production path remains fp64 until the full validation ladder passes.
* Static cache key includes the precision mode. Existing `acoustic_precision_mode`
  is already a static aux field; keep that property.
* A written no-go list:
  * no global fp32 totals for `p`, `ph`, `mu`
  * no posthoc tolerance widening
  * no clamps to hide divergence
  * no JAX-vs-JAX-only validation
  * no in-loop host/device transfers
  * no nested all-7 regression

Implementation notes:

* Keep `force_fp64=True` behavior byte-for-byte unless deliberately changing the
  mixed path.
* Add tests that assert the default fp64 path is unchanged at function-entry and
  output dtype boundaries.
* Treat the older `fp32_downcast_spec` as Stage 0 / negative-control evidence,
  not as the v0.20 target.

### Phase 1: build source/HLO audit tools before changing math

Deliverables:

* A source audit that fails the mixed-mode build if hot-path files contain
  forbidden base recovery patterns in mixed functions:
  * `p_total - p_perturbation`
  * `ph_total - ph_perturbation`
  * `mu_total - mu_perturbation`
  * forming `p_total`, `ph_total`, `mu_total` inside the scan except named
    reconstruction exits
* HLO dtype audit extensions:
  * count fp64 operations by component
  * count `convert` operations by source and destination dtype
  * count full-grid fp64 parameters/results/carry leaves
  * estimate temp memory and live fp64 arrays
  * flag broad fp64 promotion around local islands
* A per-component timing harness extending `scripts/precision_bench.py`:
  * acoustic substep
  * full RK dry step
  * large-step PGF
  * `advance_w`
  * `calc_p_rho`
  * Thompson/MYNN/RRTMG couplers
  * end-to-end nested all-7 step

No GPU run is required for this report, but the implementation sprint should
make these tools the first proof objects. Without them the team will rediscover
the 1.1x trap late.

### Phase 2: introduce compact hot-state representation

Deliverables:

* A mixed-mode hot-state type or wrapper that carries only perturbation-authority
  fields through the acoustic/RK scan.
* Explicit `BaseState` or `ReferenceState` inputs carrying fp64 base fields.
* Named conversion functions:
  * full `State` plus base -> compact mixed hot state
  * compact mixed hot state plus base -> full `State` only at exits
* A proof that the scan carry no longer includes full-grid `p_total`, `ph_total`,
  or `mu_total`.

Design:

```text
Full State at entry
  -> prepare explicit fp64 base/reference outside scan
  -> MixedHotState(
       u32, v32, theta32, qv32, hydrometeors32,
       p_pert32, ph_pert32, mu_pert32,
       w64 initially,
       selected accumulators/compensation leaves
     )
  -> acoustic/RK scan using explicit base
  -> reconstruct totals only for boundary, physics diagnostic, restart, wrfout
```

This phase is the main performance prerequisite. If HLO still shows the old
full `State` aliases live through the scan, later numerical work cannot produce
the desired speedup.

### Phase 3: plumb explicit base state through prep, finish, and refresh

Files/components:

* `small_step_prep_wrf`
* `small_step_finish_wrf`
* `diagnose_pressure_al_alt`
* `_acoustic_core_state_from_prep`
* `_refresh_grid_p_from_finished`
* restart/init/boundary reconstruction code

Actions:

1. Add mixed-mode call paths that require `BaseState` rather than falling back to
   `state.total - state.perturbation`.
2. Make fallback base reconstruction legal only on the fp64/default path or at
   named reconstruction boundaries.
3. Split finish into:
   * perturbation update finish for the hot loop
   * total reconstruction finish for exits
4. Preserve WRF alias semantics at exits: `p`, `p_total`, `p_perturbation`,
   `ph`, `ph_total`, `ph_perturbation`, `mu`, `mu_total`,
   `mu_perturbation` must still be consistent when a full `State` is produced.

Validation:

* Unit tests that mixed mode refuses to run without explicit base.
* Grep/audit no forbidden base recovery in mixed hot functions.
* One-substep equality to fp64 path when all fields are manually promoted to
  fp64 and the compact path uses fp64 storage. This isolates structural changes
  from precision changes.

### Phase 4: pressure/EOS in perturbation form

Files/components:

* `dynamics/acoustic_wrf.py::diagnose_pressure_al_alt`
* `dynamics/core/calc_p_rho.py`
* pressure refresh in `runtime/operational_mode.py`

Actions:

1. Rewrite mixed pressure diagnostics around explicit base:
   * base pressure `pb64`
   * perturbation pressure `p_prime32`
   * base geopotential `phb64`
   * perturbation geopotential `ph_prime32`
   * base dry mass `mub64`
   * perturbation dry mass `mu_prime32`
2. Use stable analytic forms where they remove cancellation:
   * `log1p(delta/base)` for log-alpha differences
   * ratio forms that keep `delta` explicit
   * `expm1` for small exponential differences if needed
3. Keep final EOS/log-alpha bracket fp64 initially.
4. Return authoritative `p_prime` and inverse-density perturbation data to fp32
   storage only after the bracket.
5. Keep `pm1` and `smdiv` pressure memory fp64 or compensated until drift probes
   pass.

Validation:

* Reuse and extend `calc_p_rho_fp32_oracle` and pressure cancellation probes.
* Column cases:
  * hydrostatic rest
  * warm bubble
  * steep terrain
  * moist column with high `qv`
* Required local result: perturbation-form fp32 must match the fp64 perturbation
  signal much closer than absolute-total fp32, without clamps.

### Phase 5: large-step and acoustic pressure-gradient force decomposition

Files/components:

* `dynamics/core/acoustic.py::advance_uv_wrf`
* `dynamics/core/rk_addtend_dry.py::_absolute_diagnostics`
* `dynamics/core/rk_addtend_dry.py::large_step_horizontal_pgf`

Actions:

1. Split PGF inputs into base and perturbation terms explicitly.
2. Precompute or cache base-only gradients in fp64 where they are time-invariant
   or slowly varying.
3. Carry perturbation gradients in fp32.
4. Evaluate only the final cancellation-sensitive bracket in fp64.
5. Return `ru_tend` and `rv_tend` to the live velocity dtype unless a local
   correction path is active.
6. Keep terrain terms fp64 until steep-terrain rest tests pass.

Do not repeat known anti-optimizations:

* Do not reintroduce pad-face pair rewrites that were already replaced with
  concat face construction.
* Do not hoist advance-w denominators just because it looks algebraically
  cleaner; prior evidence showed this was slower and not bit-identical due FMA
  differences.
* Do not rely on command buffers as the primary rescue; previous findings were
  wall-neutral/refuted.

Validation:

* Acoustic PGF oracle on flat and steep terrain.
* Rest-state preservation over many steps.
* Watch cells at terrain gradients and lateral boundaries.
* HLO proof that base gradients are not recomputed as full fp64 totals inside
  each acoustic substep.

### Phase 6: acoustic substep mixed execution

Files/components:

* `dynamics/core/acoustic.py::acoustic_substep_core`
* `advance_mu_t` path
* `advance_uv_wrf`
* `advance_w_wrf`
* `calc_p_rho_step`

Actions:

1. Run velocity/theta/moisture transport work in fp32 where it is not inside a
   named fp64 bracket.
2. Keep `mu` update in perturbation form:
   * `mu_prime32` as storage
   * optional fp64 compensation leaf if drift probes show mass accumulation
     growth
   * `muts`/`muave` fp64 until dry-mass conservation evidence permits demotion
3. Keep `w` fp64 initially.
4. Keep `ph` update split:
   * `ph_prime32` storage
   * fp64 local bracket for vertical pressure/geopotential update
   * nested ph/w boundary forcing fp64
5. Move all casts to explicit island boundaries.

Validation:

* One acoustic substep vs fp64 savepoint.
* One RK step vs fp64 savepoint.
* Divergence-growth metric: bounded or saturating divergence passes; escalating
  growth fails.
* HLO proof: no broad upcast of `u`, `v`, `theta`, `qv`, hydrometeors.

### Phase 7: vertical solve island, then shrink it

Files/components:

* `dynamics/core/advance_w.py`

Initial plan:

* Keep coefficient build, buoyancy/PGF terms, bottom/top boundary treatments,
  and Thomas scans fp64.
* Feed fp32 perturbation inputs into this island only at the boundary.
* Return `w` fp64 initially.

Shrink plan after stability:

1. Demote non-cancellation RHS/advection terms to fp32.
2. Keep Thomas coefficients fp64.
3. Test `w` storage fp32 only after 24h nested stability passes with fp64 `w`.
4. If fp32 `w` fails by drift rather than immediate blow-up, test a compact
   residual-correction design:
   * fp32 predicted `w`
   * fp64 residual correction per column
   * correction stored or applied at RK boundaries, not every stencil expression

Validation:

* `advance_w_fp32_oracle`
* vertical column hydrostatic/rest tests
* steep terrain and nested boundary ph/w forcing tests
* qke/PBL sensitivity after `w` demotion attempts

### Phase 8: non-acoustic advection, diffusion, and RK tendencies

Files/components:

* `dynamics/flux_advection.py`
* `dynamics/explicit_diffusion.py`
* RK tendency merge and coriolis paths

Actions:

1. Keep scalar advection outputs in the live field dtype unless a mass/metric
   bracket requires fp64 locally.
2. Ensure result-type logic does not promote entire fp32 advection results just
   because `rom`, metrics, or mass handles are fp64.
3. Treat mass multipliers/dividers as local fp64 brackets and cast the resulting
   tendency back to the target dtype.
4. Demote large-step coriolis after PGF is stable; it currently casts several
   fields to fp64 and can become a hidden broad island.

Validation:

* Conservation checks for dry mass and scalar mass.
* HLO count of fp64 advection/diffusion ops.
* 20-step and 1h ladder before any 24h run.

### Phase 9: physics coupling policy

Physics should follow skill/stability needs and known performance evidence:

* Thompson hydrometeors and number concentrations can remain fp32 storage.
  Existing dry oracle evidence showed zero fp32/fp64 diffs, but wet skill still
  needs validation. Do not expect Thompson fp32 arithmetic to be a major speed
  lever because the measured real fixture was launch/bandwidth-bound.
* RRTMG band optics are intrinsically fp32-like in WRF r4 logic, but the heating
  add and theta round trip should use live dtype at the coupling boundary.
* MYNN/PBL turbulent-energy fields stay fp64 initially. `qke` has a known d03
  1 km nonfinite history after hour 1 and should not be demoted in the first
  v0.20 pass.
* Surface layer inputs and flux handles stay fp64 initially. The current
  surface-layer code explicitly casts lowest-level `u`, `v`, `theta`, `qv`, and
  `p` to fp64; keep that until 24h/72h wind skill proves demotion is safe.
* Accumulators stay fp64 or use compensated accumulation. Their exact WRFv4
  parity is not the primary acceptance gate, but they must remain finite,
  nonnegative where physically nonnegative, and bounded.

Validation:

* Separate physics-only dtype probes are useful for safety, but production
  acceptance must be coupled 24-120h forecast skill, not isolated JAX-vs-JAX
  parity.

### Phase 10: nesting, boundaries, and all-7 `max_dom=9`

Nested all-7 is not an afterthought. It is the benchmark.

Rules:

1. Keep `ph`, `mu`, and `w` boundary leaves fp64 initially.
2. Keep nested ph/w relax/spec forcing fp64.
3. Allow `u`, `v`, `theta`, and `qv` boundary leaves to be fp32 only after
   boundary-ring tests pass.
4. Reconstruct totals at boundary exchange only, not inside every acoustic
   substep.
5. Two-way feedback:
   * feedback of mass/geopotential/vertical variables uses fp64
   * feedback of velocity/theta/moisture/cloud variables may use fp32 storage but
     should accumulate averages in fp64 if conservation or skill requires it
6. Every domain d01 through d09 must have stable dtype contracts and cache keys.
   A fix that only works on d01/d02 is not a v0.20 success.

Validation:

* Boundary-ring finite and physical checks at every nest.
* Domain-by-domain skill/stability metrics.
* d03 1 km qke watch explicitly included.
* All wrfout times present and readable.
* No domain silently falls back to fp64 broad mode unless the report calls it out
  as a failed compact-island proof.

### Phase 11: shrink islands one at a time

After the compact state and first stable mixed run, demote in this order:

1. PGF perturbation-gradient storage and surrounding arithmetic.
2. `calc_p_rho` non-cancellation surrounding arithmetic.
3. `advance_w` RHS/advection pieces while keeping coefficients/Thomas fp64.
4. `mu_prime` storage plus compensated fp32 accumulation if mass drift permits.
5. `ph_prime` storage with fp64 bracket and optional correction.
6. `w` storage.
7. surface-layer and PBL pieces only if wind/temp/cloud skill is insensitive.
8. qke last, and only if the d03 nonfinite signature has a clear fix.

Each demotion requires:

* local oracle pass
* HLO proof that it reduced fp64 work/liveness
* speed proof on the relevant component
* stability ladder through at least the next forecast tier

No demotion should be retained for a tiny speed gain if it materially risks
wind/temp/cloud skill. Conversely, no component should remain fp64 merely to
preserve tiny WRFv4 diffs that do not affect stability or 24-120h skill.

## 6. Validation plan and proof objects

### 6.1 Static and CPU-level validation

Before GPU runs:

* Precision matrix tests:
  * default fp64 path unchanged
  * mixed mode storage matches the planned matrix
  * `qke` remains locked unless a separate sprint changes it
* Source audit:
  * no forbidden total-minus-perturb base recovery in mixed hot functions
  * no hidden full-total reconstruction in acoustic scan
* HLO audit from `.lower(...).as_text()`:
  * fp64 op count by component
  * convert count and placement
  * fp64 carry leaves
  * temp-memory estimate
  * no callbacks/transfers
* Existing oracles:
  * pressure cancellation
  * `calc_p_rho`
  * `advance_w`
  * divergence-growth metric
* New oracles:
  * acoustic PGF perturb/base decomposition
  * large-step terrain PGF
  * nested boundary ph/w forcing
  * restart/wrfout reconstruction consistency

### 6.2 GPU profiling and timing once implementation exists

The implementation sprint should produce profiler artifacts, but none were run
for this report.

Required artifacts:

* Warm compiled timings with compile excluded.
* `nsys` step profile:
  * wall per step
  * device busy
  * kernel count per step
  * H2D/D2H transfer audit
* `ncu` or equivalent targeted kernels for:
  * acoustic substep
  * PGF
  * `advance_w`
  * `calc_p_rho`
  * Thompson/MYNN if they appear in the top profile
* Component timing table:
  * fp64 production
  * old conservative mixed negative control
  * compact mixed current
  * invalid all-fp32 proxy only as an upper-bound reference, not as acceptance

Performance gates:

* Early HLO gate: at least 40 percent lower dycore temp/argument footprint than
  current fp64 before attempting long nested forecasts.
* Component gate: acoustic/dycore large-grid warm speedup at least 2.5x before
  spending major validation time.
* Final target: nested all-7 `max_dom=9` materially faster, with the stretch
  target near 4x where dycore/acoustic dominates. If full nested all-7 cannot
  reach that because physics/launch dominates, the proof must show the exact
  Amdahl breakdown and the next structural bottleneck.

### 6.3 Forecast stability ladder

Every risky change should climb this ladder:

1. 1 acoustic substep
2. 1 RK step
3. 20 model steps
4. 0.1h
5. 1h
6. 3h
7. 6h
8. 24h
9. 48h
10. 72h
11. 120h

Abort early for:

* NaN/Inf in any non-carved-out prognostic field
* unphysical domain-wide explosion
* monotonic divergence growth in protected wind/temp/cloud fields
* nested boundary ring instability
* d03 qke nonfinite recurrence
* host/device transfer in the timestep loop
* HLO showing broad fp64 promotion that invalidates the compact-island claim

### 6.4 Reference comparisons

Use three reference classes:

1. CPU WRFv4 fixture output where available.
2. Current fp64 GPU production path.
3. Analytic/savepoint oracles for local components.

The frozen grid-delta manifest remains useful as a regression lens:

```text
proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json
scripts/compare_wrfout_grid.py
```

But for v0.20 fp32 acceptance it should be secondary. The primary gate is
forecast stability and 24-120h skill for wind, temperature, and cloud-related
layers. The new policy deliberately accepts less-tight margins versus WRFv4
where the forecast remains stable and operationally useful.

## 7. Concrete fp32 acceptance gates

These gates should be frozen before implementation begins. They are deliberately
not bit-tight. They prioritize no blow-up and preserve 24-120h wind, temperature,
and cloud skill.

Definitions:

* Baseline means CPU WRFv4 if available for that fixture and hour, otherwise the
  current fp64 GPU production run.
* Metrics are computed domain-by-domain and also on the nested all-7 aggregate.
* Boundary rings get separate checks because nested instability often appears
  there first.
* Hour tiers: 24h, 48-72h, 120h.
* Cumulative precipitation and similar accumulators are carved out from tight
  RMSE parity, but not from finite/nonnegative/bounded checks.
* QVAPOR gets a known-issue carve-out from tight RMSE, but not from finite,
  nonnegative, or wind/temp/cloud-skill consequences.

### 7.1 Universal hard gates

Fail immediately if any of these occur:

* NaN or Inf in `U`, `V`, `W`, `T`, `THETA`, `T2`, `P`, `PB`, `PH`, `PHB`,
  `MU`, `MUB`, `PSFC`, `QKE`, protected cloud fields, boundary ph/w leaves, or
  restart-critical state.
* Negative pressure or nonpositive dry mass.
* Surface pressure outside 50 kPa to 110 kPa unless the baseline itself is
  outside that range at the same grid cells.
* `T2` outside 180 K to 330 K unless the baseline itself is outside that range.
* Potential temperature outside 150 K to 1000 K unless baseline-matched.
* Horizontal wind max above 150 m/s on d01-d02 or above 200 m/s on high-resolution
  nests, unless the baseline has the same event within 10 percent.
* Vertical velocity max above 75 m/s below the model top or above 125 m/s at top
  sponge levels, unless baseline-matched.
* `qke` negative after existing model floors, nonfinite, or above 150 m2/s2
  unless baseline-matched.
* Any domain d01-d09 missing, producing unreadable wrfout, or dropping out of
  nested all-7 `max_dom=9`.

These are stability gates, not accuracy gates. Passing them does not prove
acceptance; failing them rejects the run.

### 7.2 Wind gates

Protected fields: `U10`, `V10`, lowest-model-level `U/V`, 850 hPa or nearest
mid-level wind where available, and boundary-ring winds.

Pass thresholds versus baseline:

| Lead | U10/V10 RMSE increase | U10/V10 bias abs | 95th percentile speed diff | Domain-mean speed drift |
| --- | ---: | ---: | ---: | ---: |
| 24h | <= 0.25 m/s | <= 0.20 m/s | <= 2.0 m/s | <= 0.20 m/s |
| 48-72h | <= 0.50 m/s | <= 0.35 m/s | <= 3.0 m/s | <= 0.35 m/s |
| 120h | <= 0.80 m/s | <= 0.60 m/s | <= 5.0 m/s | <= 0.60 m/s |

Additional trend gate:

* The divergence-growth metric for wind must be bounded or saturating. A
  superlinear growth signature fails even if a single-hour RMSE threshold passes.

### 7.3 Temperature gates

Protected fields: `T2`, lowest-model-level temperature, `THETA`, selected
pressure-level or model-level temperature slices, and boundary-ring temperature.

Pass thresholds versus baseline:

| Lead | T2 RMSE increase | T2 bias abs | 95th percentile T2 diff | Domain-mean theta drift |
| --- | ---: | ---: | ---: | ---: |
| 24h | <= 0.6 K | <= 0.3 K | <= 1.5 K | <= 0.3 K |
| 48-72h | <= 1.0 K | <= 0.6 K | <= 2.5 K | <= 0.6 K |
| 120h | <= 1.5 K | <= 1.0 K | <= 4.0 K | <= 1.0 K |

The gate is intentionally looser than old parity tolerances, but a coherent warm
or cold drift that would change forecast skill fails.

### 7.4 Cloud and radiation-related gates

Protected fields:

* `QCLOUD`, `QICE`, `QSNOW`, `QGRAUP`, cloud fraction fields, low/mid/high cloud
  cover diagnostics where available
* column-integrated cloud water and ice
* `SWDOWN`, `LWDOWN`, radiative heating tendency or `RTHRATEN`
* PBL/cloud coupling fields that affect low cloud and wind

Pass thresholds versus baseline:

| Lead | Domain cloud-fraction abs diff | Column cloud condensate normalized RMSE | SWDOWN/LWDOWN bias abs | Cloud-cover collapse/explosion |
| --- | ---: | ---: | ---: | --- |
| 24h | <= 0.10 | <= 25 percent | <= 30 W/m2 | fail if present |
| 48-72h | <= 0.15 | <= 40 percent | <= 50 W/m2 | fail if present |
| 120h | <= 0.20 | <= 60 percent | <= 70 W/m2 | fail if present |

Clouds are chaotic and often phase-sensitive, so the gate should include spatial
skill diagnostics rather than only pointwise RMSE:

* object count and area distribution for low cloud decks
* low-cloud fraction over Canary terrain and marine boundary layer zones
* cloud-top height distribution
* binary cloud/no-cloud FSS or neighborhood score at practical radii

### 7.5 Pressure, mass, and geopotential gates

These variables are not the highest user-facing skill target, but they are
stability-critical.

| Lead | PSFC RMSE increase | PSFC bias abs | Dry-mass relative drift | Geopotential boundary ring |
| --- | ---: | ---: | ---: | --- |
| 24h | <= 150 Pa | <= 75 Pa | <= 1e-4 | no growing ring mode |
| 48-72h | <= 300 Pa | <= 150 Pa | <= 3e-4 | no growing ring mode |
| 120h | <= 500 Pa | <= 250 Pa | <= 7e-4 | no growing ring mode |

If these pressure/mass gates are slightly exceeded but wind/temp/cloud skill is
excellent and no blow-up appears, the team may consider a documented exception.
But any exception must identify the physical reason and must not mask a growing
mode.

### 7.6 Moisture and QVAPOR gates

QVAPOR is a known-problem field, so do not require tight pointwise parity. Do
require stability:

* finite everywhere
* nonnegative after existing model limiters
* domain mean water vapor within 20 percent of baseline at 24h, 35 percent at
  72h, and 50 percent at 120h
* 99.9th percentile `QVAPOR` below 0.04 kg/kg unless baseline-matched
* no moisture drift that causes cloud, radiation, temperature, or wind gates to
  fail

Hydrometeors:

* finite and nonnegative after existing model floors
* no domain-wide cloud water/ice collapse or explosion
* cloud skill gates are authoritative over pointwise hydrometeor RMSE

### 7.7 Cumulative variable carve-out

Cumulative variables include precipitation accumulators and similar integrated
diagnostics. They are allowed looser parity because small phase changes can
produce large pointwise differences.

Hard requirements still apply:

* finite
* nonnegative where physically nonnegative
* monotonic where the WRF variable is cumulative
* basin/domain total not more than 3x baseline at 24h, 4x at 72h, or 5x at 120h
  unless baseline precipitation is near-zero and a documented phase shift
  explains the ratio
* no coupling back into wind/temp/cloud instability

These variables should be reported, not hidden. They just should not block a
massive fp32 speed win if the protected forecast-skill fields are stable.

## 8. Roadblocks requiring math or computational creativity

Ranked by expected payoff for the v0.20 goal:

### 1. Compact perturbation-native hot state

Payoff: highest.

Problem: current `State` carries total and perturbation aliases, and many helper
functions recover base from totals. Even if arithmetic is locally fp32, XLA can
keep the fp64 alias graph alive.

Idea: use a separate mixed hot-state type and explicit reconstruction exits. This
is an engineering and compiler-liveness problem as much as a numerical problem.

No-go evidence: HLO still shows broad fp64 scan carry or full-grid total aliases.

### 2. Base-gradient plus perturbation-gradient PGF

Payoff: very high.

Problem: pressure-gradient terms subtract large hydrostatic/base quantities.

Idea: precompute base-only fp64 gradients and evolve perturbation gradients
explicitly. Do the final bracket in fp64, not the whole stencil. Use stable
ratio/log forms to avoid constructing large totals.

Risk: terrain terms and vertical staggering make algebra easy to get subtly
wrong. Needs analytic/rest-state oracle and WRF fixture comparison.

### 3. HLO liveness control and barriers

Payoff: very high if XLA keeps widening local brackets.

Problem: a mathematically small fp64 island may compile as a broad fp64 remat
arena.

Ideas:

* island wrapper functions with explicit input/output dtypes
* `jax.lax.optimization_barrier` only where it demonstrably prevents destructive
  fusion or EFT invalidation
* donation/aliasing inspection
* manual split of giant functions if fusion keeps the wrong values live
* custom call/Pallas only if HLO cannot be constrained otherwise

Risk: barriers and splits can increase kernel count and undo gains. Every use
needs timing proof.

### 4. Compensated perturbation accumulators

Payoff: medium to high.

Problem: long acoustic/RK accumulation in fp32 can drift even when instantaneous
signals are represented correctly.

Ideas:

* Kahan/Neumaier for `mu_prime`, pressure memory, and selected theta/energy
  accumulations
* one fp64 correction leaf per field or per column
* periodic fp64 correction at RK boundaries rather than per stencil

Risk: extra registers and live arrays can erase speed. Use only where drift
probes demand it.

### 5. Stable EOS and log-alpha identities

Payoff: medium.

Problem: pressure/inverse-density diagnostics use total values but need small
differences.

Ideas:

* `log1p(delta/base)`
* `expm1` for small exponent differences
* ratio forms that separate base from perturbation
* local fp64 bracket around a compact scalar expression

Risk: WRF-faithful algebra can be changed accidentally. Needs savepoint oracle.

### 6. Vertical solve residual correction

Payoff: medium.

Problem: `advance_w` is likely the last large fp64 island because it couples
vertical pressure gradient, buoyancy, geopotential, boundary forcing, and a
Thomas solve.

Idea: fp32 predictor plus compact fp64 residual correction per column. This may
keep stability without carrying full fp64 `w` everywhere.

Risk: high math complexity and possible phase errors in nested terrain.

### 7. Double-single or EFT

Payoff: selective.

Problem: double-single is numerically strong but historical probes showed it can
be slower than native fp64 when applied broadly.

Use only if it replaces a much larger live fp64 array or keeps a local operation
inside fp32 hardware without broad promotion. Do not use as a general fp64
replacement.

### 8. Stochastic rounding

Payoff: uncertain.

Potentially useful for long drift in fp32 perturbations, but it adds complexity
and reproducibility risk. Consider only after deterministic perturbation and
compensation options are exhausted.

### 9. Pallas/custom kernels

Payoff: high only if XLA is the blocker.

If HLO proves that normal JAX cannot keep islands compact, a custom Pallas
kernel for PGF or acoustic substep may be justified. This should be a fallback
because it raises maintenance risk and must still support nested all-7.

## 9. How nested all-7 stays functional

Every phase must preserve nested all-7 functionality by design:

* Default fp64 path remains available until acceptance.
* Mixed mode must keep static cache separation by precision mode and domain.
* Boundary and feedback code gets explicit reconstruction exits. That keeps
  `p_total`, `ph_total`, and `mu_total` available to WRF-like boundary logic
  without forcing them into the acoustic hot carry.
* `ph`, `mu`, and `w` boundary leaves stay fp64 first.
* d03 1 km qke nonfinite history is an explicit sentinel.
* Every long validation run reports all domains d01-d09 separately.
* `max_dom=9` all-7 is the benchmark for final performance claims, not a
  side-case after a single-domain proof.

Suggested all-7 validation sequence:

1. Compile-only/lower-HLO all domains with mixed mode.
2. One-step nested all-7 smoke.
3. 20-step nested all-7 with boundary-ring checks.
4. 1h nested all-7.
5. 3h nested all-7.
6. 6h nested all-7.
7. 24h nested all-7.
8. 72h nested all-7.
9. 120h nested all-7 for final wind/temp/cloud skill.

No phase should be called done if it only works on a single domain or synthetic
fixture.

## 10. Decision gates for the sprint

Use these gates to avoid another partial fp32 sprint:

### Gate A: compactness before speed

Pass only if:

* HLO shows no full-grid total aliases in the hot scan carry.
* fp64 live arrays and temp memory drop materially.
* convert placement is explainable and local.

Fail reason must be exact:

* which field or function keeps fp64 live
* which helper reconstructs totals
* which fusion/remat decision widens the island

### Gate B: component speed before long forecasts

Pass only if:

* acoustic/dycore large-grid component is at least 2.5x faster warm-compiled, or
  the profiler proves a different component is now dominant.
* no transfer appears in-loop.

If speed remains 1.0x to 1.1x, stop and write the proof. Do not spend 120h
forecast time validating a broad-island path.

### Gate C: stability ladder

Pass each rung before climbing. Do not widen gates during the run.

### Gate D: skill acceptance

Final acceptance is wind/temp/cloud skill and stability through 24-120h, with
the per-variable gates in section 7. The frozen grid-delta manifest remains a
diagnostic comparator, not the primary veto when the new fp32 policy passes.

### Gate E: final performance claim

A v0.20 performance claim must include:

* nested all-7 `max_dom=9` wall/fc-hour
* current fp64 GPU baseline
* CPU WRF denominator when relevant
* compile-excluded warm timings
* profiler artifacts
* transfer audit
* HLO dtype/liveness artifact
* component timing table
* statement of any remaining fp64 islands and why each remains

## 11. Suggested proof artifact tree

Recommended output locations:

```text
proofs/v020/fp32_analysis/
  GPT_FP32_PLAN.md
  compact_hlo_audit.json
  compact_hlo_audit.md
  forbidden_base_reconstruction_audit.txt
  component_timing_fp64_vs_mixed.json
  nsys_nested_all7_step/
  ncu_acoustic_pgf/
  validation_ladder/
    001_one_substep.json
    002_one_rk.json
    003_20_steps.json
    004_1h.json
    005_3h.json
    006_6h.json
    007_24h.json
    008_72h.json
    009_120h.json
  skill_metrics/
    wind_temp_cloud_24_72_120h.json
    per_domain_d01_d09.md
  nested_all7_maxdom9_final.md
```

## 12. Final strategic recommendation

Proceed with v0.20 only as a compact perturbation-authoritative rewrite. The
older "fp32 field downcast with broad fp64 acoustic island" is useful as a
negative control and fallback, but it does not match the requested objective.

The critical implementation order is:

1. HLO/source audit tools.
2. Compact hot state without total aliases.
3. Explicit fp64 base plumbing.
4. Perturbation-native pressure/EOS.
5. PGF base/perturb decomposition.
6. Acoustic mixed execution with `w` and vertical solve still protected.
7. Nested all-7 stability ladder.
8. One-at-a-time island shrink.

The acceptance policy should be skill-first: no blow-up, protected wind/temp/cloud
skill through 24-120h, and nested all-7 `max_dom=9` fully functional. Do not keep
large fp64 regions for tiny WRFv4 parity differences. Keep fp64 only where it is
provably required for stability, nested boundary behavior, or protected forecast
skill.

If the final speedup is not massive, the correct outcome is still valuable only
if the proof identifies the exact blocker: remaining fp64 live arrays, XLA
promotion, vertical/PGF island cost, launch-bound physics, or nested feedback
constraints. The sprint should be structured to learn that early, before long
forecast validation consumes time.
