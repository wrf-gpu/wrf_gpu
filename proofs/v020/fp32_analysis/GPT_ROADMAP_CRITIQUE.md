# GPT adversarial critique of the v0.20.0 roadmap

Date: 2026-06-21

Analyst: GPT-5 Codex, adversarial critic

Scope: critique of `.agent/decisions/V0200-ROADMAP.md` plus:

* `proofs/v020/fp32_analysis/FINAL_FP32_SPRINT_PLAN.md`
* `proofs/v020/fp32_analysis/OPUS_FP32_PLAN.md`
* `proofs/v020/fp32_analysis/OPUS_P34_PLAN.md`
* `proofs/v020/fp32_analysis/GPT_FP32_PLAN.md`
* `proofs/v020/fp32_analysis/GPT_P34_PLAN.md`

No GPU runs were started. No model source files were edited. This document is a
planning/report artifact only.

## 0. Bottom-line critic verdict

The roadmap's central strategic reframe is mostly sound:

* fp32 by itself, with the current nested all-7 launch/cadence/kernel topology,
  should not be expected to produce a massive nest speedup;
* the all-7 nest is the stability and 24-120h skill gate, not the speed
  demonstrator;
* large wins require structural work, larger stable timestep/cadence changes,
  and/or DRAM-bound scale where fp32 reduces live bytes and HBM traffic;
* the 1 km capability unlock is at least as strategically important as wall time.

But the roadmap still needs hardening before finalization. The current text is
too close to turning several conditional targets into promises:

1. **The phrase "fp32 provably capped on the nest" must be scoped.** It is
   defensible for precision-only changes with unchanged launch/cadence topology.
   It is not a theorem against a massive all-7 improvement from dt reduction,
   acoustic/vertical fusion, or a combined structural + fp32 rewrite.
2. **P4 should not be sold as a cheap win until a CFL/terrain/nesting diagnostic
   gate runs.** Raising dt and cutting acoustic substeps can be the largest lever,
   but it is the fastest way to create a fake-green or unstable 1 km nest.
3. **The combined budget is optimistic.** A realistic pre-fp32 P3+P4 all-7 range
   is more like 1.3-2.2x unless dt can move to 36-54 s and current profiles show
   real residual host/force overhead. 3x is a stretch case, not a base plan.
4. **The fp32 G0 KILL criterion is too speed-centric.** If fp32 unlocks a stable
   1 km grid that fp64 cannot fit, the rewrite can be worth funding even if the
   immediate speedup is only modest.
5. **The acceptance policy is directionally right but not yet canonical.** The
   subplans contain conflicting tolerance bands. The final roadmap needs one
   frozen table, multiple weather cases, no-clamp auditing, restart/output checks,
   conservation budgets, and explicit per-domain reporting.

## 1. Core verdict: sound, but over-scoped if stated as a theorem

### 1.1 What is sound

The evidence supports the core conclusion that naive or conservative mixed fp32
stalls around 1.1x for an exact reason:

* the valid compact-island attempts reduced fp64 tokens but left launch topology
  and dominant kernel walls largely unchanged;
* the all-7 leaves are small, under-occupying, and largely L2-resident;
* fp32 does not remove launch overhead;
* fp32's DRAM benefit is weak when the working set is not DRAM-bound;
* fp64 islands around PGF/EOS/advance_w can keep the transient and HBM footprint
  high even after surface dtype changes.

That is a strong argument against another "flip more leaves to fp32" sprint. It
also justifies making all-7 the correctness/skill benchmark and using larger
DRAM-bound fixtures for speed/capability demonstration.

### 1.2 What is not yet proven

The roadmap should avoid wording that implies "no massive all-7 speedup is
possible from anything involving fp32." The defensible statement is narrower:

```text
fp32 alone, without changing launch topology, acoustic/vertical kernel shape,
or timestep/cadence, is capped on the current nested all-7 benchmark.
```

There are still non-dtype or combined paths to a large all-7 improvement:

* a stable larger timestep reduces model steps, boundary builds, physics calls,
  and launches almost proportionally;
* fused acoustic/vertical kernels can remove the tiny-kernel launch/HBM pattern
  directly;
* compact fp32 can make those fused kernels cheaper by keeping fp64 brackets in
  registers rather than resident full arrays;
* exact-shape leaf fusion or domain-program decomposition may help if current
  profiles still show many tiny underfilled kernels after v0.19.1.

Those paths do not contradict the fp32 cap. They are the structural work the cap
points toward.

### 1.3 The missed-path audit

I do not see a plausible missed path to a **massive nest speedup from dtype-only
fp32**. The likely missed paths are all outside that category:

* dt/acoustic-substep changes;
* acoustic + vertical solve fusion;
* reducing residual boundary/force dispatches;
* capability-driven fp32 at grids that fp64 cannot fit;
* multi-GPU or tiling if single-card VRAM remains the binding constraint.

The roadmap should require a fresh v0.19.1-style current-regime profile before
using "provable" language. The old v0.17 host-bound and v0.18 broken-scan regimes
are useful history, but the final claim must be tied to the current baseline:
warm all-7 wall/fc-hour, kernel launch census, top-kernel table, host ledger,
transfer audit, HLO memory analysis, and per-domain component timing.

## 2. P4 ordering: safe only with a stricter CFL-first ladder

The roadmap puts larger timestep/cutting `n_sound` before the fused kernel. That
can be the right order because P4 is cheaper and reversible, but only if it is
treated as a diagnostic ladder, not as a config tweak.

### 2.1 The unsafe version

The risky ordering is:

```text
cut acoustic_substeps first -> raise root dt -> long skill run
```

That combines two destabilizers before isolating the failure mode. On the all-7
nest, the steep 1 km leaves are where failure will appear first, not d01. A
candidate can look fine for a short d01-dominated segment and then fail through
terrain-following vertical motion, boundary ring growth, qke/PBL instability,
or cloud microphysics feedback on d03-d09.

### 2.2 Safer ordering

Use this ordering instead:

1. **P4.0 diagnostics before any config change.** Measure horizontal Courant,
   vertical Courant, acoustic-step proxy, pressure/mass/geopotential boundary
   ring growth, qke extrema, w spectra, hydrometeor extrema, radiation/PBL call
   counts, and per-domain dt/substep values.
2. **Raise dt with acoustic substeps scaled to keep acoustic dt near baseline.**
   This isolates large-step advective, nesting, physics-cadence, and output-time
   risk without simultaneously reducing acoustic stability margin.
3. **Only after a dt candidate is stable, reduce acoustic substeps at fixed dt.**
   That isolates sound-mode risk.
4. **If fine nests bind, decouple `parent_time_step_ratio` from
   `parent_grid_ratio` under a dedicated nesting gate.** Larger d01/d02 steps can
   still be valuable even if 1 km leaves cannot scale by the same factor.
5. **Only then pursue physics cadence beyond WRF-faithful cleanup.** `cudt`
   honoring on d01 is a correctness cleanup. PBL/microphysics downsampling is a
   fast mode and should be late.

### 2.3 Concrete P4 ladder

Baseline:

```text
d01 dt = 18 s, d02 = 6 s, leaves = 2 s, acoustic_substeps = 10
```

First isolate larger dt with baseline-like acoustic dt:

| Root dt | Leaf dt if ratio=3 | Acoustic substeps for baseline-like acoustic dt | Purpose |
| ---: | ---: | ---: | --- |
| 24 s | 2.667 s | 14 | Low-risk dt probe |
| 27 s | 3 s | 15 | 1.5x ideal fewer-step probe |
| 30 s | 3.333 s | 17 | Mid probe |
| 36 s | 4 s | 20 | 2x ideal fewer-step probe |
| 45 s | 5 s | 25 | High-risk stretch |
| 54 s | 6 s | 30 | WRF heuristic stretch, very high risk |

Then reduce substeps only for candidates that pass the 6h short ladder:

```text
dt 24: n_sound 14 -> 12 -> 10
dt 27: n_sound 15 -> 12 -> 10
dt 30: n_sound 17 -> 15 -> 12 -> 10
dt 36: n_sound 20 -> 15 -> 12 -> 10
dt 45/54: do not reduce before 24h evidence at scaled substeps
```

### 2.4 Stability landmines the ladder must catch

P4 must explicitly monitor these, domain by domain and boundary-ring separately:

* horizontal advective CFL in high-wind terrain channels and gust regions;
* vertical CFL from terrain-induced `w`, especially d03-d09 and sponge/top
  layers;
* acoustic CFL and sound-mode damping as `dt/acoustic_substeps` increases;
* pressure, dry-mass, and geopotential boundary ring growth;
* `ph`/`w` nested boundary forcing phase error;
* parent-child temporal interpolation and output-time alignment;
* `qke` recurrence on d03 1 km and MYNN/PBL instability;
* surface-layer Monin-Obukhov near-neutral iteration sensitivity;
* cloud microphysics stiffness, sedimentation, and hydrometeor positivity;
* low-cloud/radiation/PBL feedback and T2 diurnal phase drift;
* cumulus `cudt` state such as `w0avg`/`nca` if d01 cadence is honored;
* top damping/Rayleigh/sponge reflection;
* restart continuation and history file completeness;
* precipitation/cumulative monotonicity under the carve-out.

## 3. Budget critique

The roadmap's budget is directionally useful, but the middle of the range is too
optimistic if read as expected all-7 speedup from the current v0.19.1 baseline.

| Lever | Roadmap framing | Critic range from current all-7 baseline | Main shortfall mode |
| --- | --- | --- | --- |
| Allocator + root sync | roughly 1.2-1.4x honest nest | likely 1.03-1.15x; 1.2-1.4x only if fresh profile shows real sync/allocator churn | v0.19.1 fused path may already be VRAM-flat and not strongly host-sync-bound |
| Residual boundary/device cascade | moderate | 1.00-1.20x | many "force" events may already be inside fused programs; root-to-d02 fusion may be compile/RSS-limited |
| Command buffers/CUDA graphs | structural host lever | 1.00-1.10x until post-refactor profile proves CPU launch overhead | prior command-buffer tests were no-op/wall-neutral |
| Larger dt + n_sound | 1.4-2.2x plausible | 1.2-1.8x likely; 2.2x if dt 36 and substeps reduced; 3x only stretch dt 54-like | 1 km CFL, boundary ph/w, qke/PBL, cloud skill |
| P3+P4 pre-fp32 combined | 1.7-3x nest | 1.3-2.2x realistic; 2.5-3x stretch | multiplicative assumptions fail if dt cannot rise or host overhead is already small |
| Fused acoustic + vertical | 1.5-3x dycore | plausible for dycore slice; end-to-end all-7 may be 1.1-1.6x unless dycore dominates | physics/boundary/output/fixed costs, custom kernel risk |
| fp32 at scale | 2-3.5x + 1 km capability | plausible if transient actually shrinks; 1.2-2x if fp64 islands/liveness/physics remain dominant | XLA liveness, remaining fp64 vertical/PBL/qke islands, HBM not reduced |

The roadmap should separate three numbers:

1. **Target:** what the sprint is trying to unlock.
2. **Planning expectation:** what is likely enough to justify the work.
3. **Stretch:** what requires several best-case conditions at once.

My recommended wording:

```text
P3+P4 pre-fp32 target: 1.7-3x.
Planning expectation from current v0.19.1 baseline: 1.3-2.2x.
3x is a stretch case requiring stable dt >= 36s, acoustic substep reduction,
and a measured residual host/launch bottleneck.
```

For the fused kernel:

```text
1.5-3x is a dycore/acoustic-slice target, not automatically an all-7 wall target.
The end-to-end claim must report Amdahl share before and after fusion.
```

## 4. fp32 G0 KILL criterion: split speed and capability

The current G0 gate is too binary:

```text
GO if transient majority fp32-able + VRAM lower + >2x trajectory.
KILL if transient still fp64-pinned and ms/step <= 1.2x.
```

That is correct for a **speed rewrite**. It is not sufficient for a **capability
rewrite**. The project endpoint includes a real Canary 1 km practical forecast
system. A stable fp32 mode that fits a valuable grid fp64 cannot fit may justify
funding even with only modest speedup.

Use three G0 outcomes:

### G0-Speed GO

Fund the full speed-oriented compact fp32 rewrite if:

* transient majority is fp32-able;
* peak VRAM is lower than fp64 on the largest both-fit grid;
* dominant large-grid kernels actually get faster;
* warm large-grid acoustic/dycore is on a >2x trajectory, or profiler shows the
  bottleneck moved to a named non-dycore component.

### G0-Capability GO

Fund a narrower capability-oriented rewrite even if speed is modest if:

* fp32 fits a target 1 km/large grid that fp64 OOMs or cannot run with safe
  headroom;
* short stability gates pass on that grid;
* HLO/VRAM proof shows the fit is due to real live-byte reduction, not allocator
  luck or hidden fallback;
* all-7 remains functionally green under the same dtype contract.

This mode can have a different budget and acceptance: the primary deliverable is
"fp64 cannot run this useful grid; fp32 can run it stably."

### G0-KILL

Kill or redirect the fp32 rewrite only if both are true:

* speed path fails: transient remains fp64-pinned and warm large-grid speedup is
  <= 1.2x with no credible structural follow-on;
* capability path fails: fp32 does not materially lower peak VRAM, does not fit
  a meaningful fp64-OOM grid, or fails short stability on the target grid.

If only speed fails, redirect the **speed budget** to P3/P4/multi-GPU while
keeping a capability rewrite on the table.

## 5. Acceptance gates: good skeleton, not yet shipping-safe

The A-E gate structure is the right skeleton:

* compactness before speed;
* component speed before long forecasts;
* stability ladder;
* skill acceptance;
* final profiler/HLO artifact set.

But the final roadmap needs sharper anti-fake-green controls.

### 5.1 Freeze one canonical tolerance table

The subplans conflict:

* OPUS_FP32 uses very tight skill deltas such as 0.1 K / 0.1 m/s and no 120h ACC
  drop;
* GPT_FP32 proposes looser lead-dependent gates such as U10/V10 RMSE increase
  0.25/0.50/0.80 m/s and T2 0.6/1.0/1.5 K at 24/72/120h.

Either choice can be defended, but the roadmap cannot reference both and then
decide after seeing results. The final contract must freeze:

* field list;
* metric;
* lead times;
* per-domain and aggregate thresholds;
* boundary-ring thresholds;
* carve-outs;
* exception process.

My recommendation is a hybrid:

* hard stability bounds exactly as hard gates;
* lead-dependent RMSE/bias bands like GPT's table for operational skill;
* an additional "no material ACC/neighborhood skill drop" test for wind/cloud at
  72-120h;
* thresholds expressed relative to the fp64-GPU-vs-CPU-WRF gap when that gap is
  available, so fp32 cannot add a second full port-sized error.

### 5.2 Require multiple weather cases

One Canary date can fake green. Use at least:

* strong wind / steep-terrain acceleration case;
* marine low-cloud / stratocumulus case;
* moist/cloud/precipitation case;
* hot/dry boundary-layer case;
* if feasible, one winter and one summer regime.

The all-7 `max_dom=9` fixture remains mandatory, but final production acceptance
should not be based on a single synoptic regime.

### 5.3 Add no-clamp and no-fallback audits

A fast build should fail if it passes only by hiding instability. Required:

* no new finite guards, clamps, masks, or floors beyond WRF-like existing
  limiters;
* guard hit counts reported for existing limiters;
* no silent domain fallback to broad fp64;
* no silent disabled physics/cadence changes;
* dtype/island matrix reported per domain;
* all H2D/D2H transfers inside timestep loop reported as hard failures unless
  explicitly approved.

### 5.4 Add conservation, restart, and output checks

The current gates emphasize final wrfout skill. Add:

* dry mass budget;
* water budget and cumulative monotonicity;
* pressure/geopotential boundary-ring growth;
* restart-write then restart-continue equivalence;
* all wrfout times present and readable for d01-d09;
* cold compile/RSS and warm cache-key stability for fused programs;
* host event-list/RAM flatness across repeated output groups.

### 5.5 Reference hierarchy

The final acceptance should use:

1. CPU WRFv4 where available;
2. current fp64 GPU production baseline;
3. observations/AEMET/AIFS/ERA-style references where available for actual
   forecast skill;
4. analytic or savepoint oracles for local components.

JAX-vs-JAX comparisons alone are diagnostic only.

## 6. What is missing across points 1, 3, and 4

### 6.1 Multi-GPU/tiling as an active fallback

The roadmap mentions redirecting to multi-GPU/tiling if fp32 fails. It should
make that a live comparator, not a vague fallback. If G0-speed fails but the
capability target is still valuable, the decision should compare:

* compact fp32 capability rewrite;
* domain/column tiling;
* multi-GPU domain decomposition;
* structural P3/P4 only.

### 6.2 Namelist cadence semantics

`radt` appears to be effectively pinned to 1800 s in parts of the nested
pipeline even when proof inputs vary. `cudt` is accepted but not faithfully
honored for d01 cumulus. `bldt=0` means PBL every step is WRF-faithful for the
benchmark, so PBL downsampling is a fast mode, not cleanup.

The roadmap should require a cadence semantics proof before using cadence speed
numbers.

### 6.3 Boundary package interaction with compact fp32

Current boundary construction can derive child base leaves from total minus
perturbation. That is exactly the pattern the compact fp32 plan must remove from
hot paths. Point 3 and point 1 need a shared base-state contract:

* explicit base/reference leaves available to boundary builders;
* no total-minus-perturb base recovery in mixed hot functions;
* ph/mu/w boundary leaves protected until boundary-ring tests pass.

### 6.4 Custom kernel/Pallas governance

The roadmap mentions a fused acoustic/vertical megakernel. That likely needs an
ADR or sprint contract because it changes backend assumptions, maintenance
burden, and proof obligations. Require:

* XLA/HLO cleanup attempted first;
* savepoint harness before all-7 integration;
* one-column vertical oracle;
* launch/HBM proof;
* fallback path if Pallas/custom kernel is unmaintainable.

### 6.5 Compile, cache, and RSS gates

Fusion work can win warm wall time and still be operationally unusable if it
causes cold megacompiles, cache-key explosion, or host RSS spikes. Add gates for:

* cold compile wall;
* peak host RSS;
* number of compile keys;
* cache reuse across d01-d09;
* warm vs cold reporting separation.

### 6.6 Per-domain and per-component Amdahl accounting

Every speed claim should include:

* wall/fc-hour;
* per-domain time;
* physics vs dycore vs boundary vs output shares;
* launch count;
* graph launch count;
* top 20 kernels by time and count;
* dycore share before claiming a dycore-fusion multiplier.

Without this, a "1.5-3x dycore" claim can be accidentally read as an all-7 wall
claim.

### 6.7 Statistical skill robustness

The roadmap should not rely only on deterministic single-run deltas. Add
neighborhood/object metrics for cloud, lead-dependent scores, and multi-case
reporting. If ensemble evidence is unavailable, state that explicitly and keep
the production claim narrower.

## 7. Direct answers to the roadmap section 7 open questions

### Q1. Is the order right? Is P4 before fused kernel safe?

**Answer:** P4 before fused kernel is acceptable only as a CFL-first stability
ladder. It is not safe as "raise dt and cut `n_sound`" before diagnostics.

Correct order:

1. current-regime profile and CFL diagnostics;
2. allocator/sync/boundary cleanup that is bit-preserving or near-bit-preserving;
3. dt ladder with acoustic substeps scaled to baseline acoustic dt;
4. substep reduction at fixed dt only after dt stability is known;
5. cadence cleanup (`radt` semantics, `cudt`) and only then risky fast-mode PBL
   or microphysics cadence;
6. fused acoustic/vertical savepoint prototype in parallel or after the profile
   shows dycore launch/HBM still dominates.

Stability landmines: horizontal/vertical/acoustic CFL, steep terrain `w`,
boundary ph/w ring modes, parent-child temporal phase, qke/MYNN nonfinite,
surface-layer near-neutral iteration, cloud microphysics stiffness, radiation
diurnal phase, cumulus state cadence, top damping, restart/output alignment, and
cumulative variable monotonicity.

### Q2. Is the 1.7-3x nest budget defensible?

**Answer:** Defensible as a target/stretch range, not as the planning
expectation.

From the current v0.19.1-style all-7 baseline:

* likely P3 cleanup: 1.03-1.20x unless fresh profile shows residual host gaps;
* likely P4 dt/substep: 1.2-1.8x if dt can reach 24-36 s safely;
* combined realistic: 1.3-2.2x;
* 2.5-3x requires stable dt near 36-54 s, reduced acoustic substeps, and real
  residual host/launch savings.

So the roadmap should keep 1.7-3x as an ambition but label the upper half
conditional.

### Q3. Is the fp32 G0 KILL criterion correct?

**Answer:** Not as written. It is correct for killing the **speed-first rewrite**
but too aggressive for killing fp32 entirely.

Rewrite the gate into:

* G0-Speed GO;
* G0-Capability GO;
* G0-KILL only if both speed and capability fail.

If fp32 unlocks a stable 1 km grid that fp64 cannot fit, fund a capability path
even if the immediate large-grid speedup is modest. The project should not throw
away a real 1 km capability unlock just because the first speed probe is only
1.2x.

### Q4. What is missing?

Missing items that should be added before finalization:

* one canonical acceptance threshold table;
* multiple validation weather cases;
* no-clamp/no-fallback audit;
* restart and conservation checks;
* current-regime profile before using "provable" language;
* cadence semantics proof for `radt`, `cudt`, and `bldt`;
* active multi-GPU/tiling fallback comparator;
* compile/RSS/cache-key gates for fusion;
* ADR/sprint contract for Pallas/custom kernels;
* boundary base-state contract shared by fp32 and P3;
* per-domain Amdahl accounting;
* parent-time-step-ratio decoupling experiment if fine nests bind.

## 8. Recommended roadmap edits before manager finalization

### Edit 1: Scope the fp32 cap

Replace broad statements like:

```text
fp32 is provably capped on the nest
```

with:

```text
fp32-only dtype changes with unchanged launch/cadence/kernel topology are capped
on the current all-7 nest. Massive all-7 wins require structural fusion, fewer
model steps, or a larger/capability grid where fp32 changes live bytes and HBM.
```

### Edit 2: Split G0

Add `G0-Speed GO`, `G0-Capability GO`, and `G0-KILL` as defined above. Require
both a largest-both-fit grid and at least one fp32-only or fp64-OOM target for
the capability decision.

### Edit 3: Reorder P4 ladder

Make CFL diagnostics and scaled-acoustic dt probes mandatory before substep
cuts. Add boundary-ring, qke, w-spectrum, cloud, and restart sentinels.

### Edit 4: Downgrade some speed expectations to conditional ranges

Use:

```text
P3+P4 planning expectation: 1.3-2.2x.
P3+P4 stretch: 2.5-3x if dt/substep and host/launch gates all pass.
Fused acoustic/vertical: 1.5-3x dycore slice target; end-to-end wall requires
Amdahl proof.
```

### Edit 5: Freeze acceptance thresholds

Resolve the Opus-vs-GPT tolerance mismatch before implementation. Do not leave
"reasonable margin" as a live choice during validation.

### Edit 6: Add proof artifacts to every close gate

A milestone close should include:

* warm timings, compile excluded;
* profiler artifacts or explicit profiler limitation note;
* HLO dtype/liveness report;
* transfer audit;
* per-domain skill/stability report;
* current dtype/island matrix;
* no-clamp/no-fallback audit;
* restart/output/conservation report;
* unresolved risks and rollback fields.

## 9. Final critic position

The roadmap is strategically right to stop chasing "minimal mixed precision"
and to prioritize structural speed plus fp32 capability. The biggest risk is not
that the team missed an obvious dtype-only 4x nest path. The bigger risk is that
the roadmap overclaims conditional speed budgets, validates P4 on too-short or
too-easy runs, and kills fp32 prematurely if speed disappoints while capability
is still valuable.

Harden those three points and the plan becomes much more defensible:

1. precision-only cap is scoped and proven with current artifacts;
2. P4 is a CFL-first ladder, not a config shortcut;
3. fp32 is judged on both speed and 1 km capability.

Nested all-7 `max_dom=9` remains mandatory at every stage. It should not be the
place where fp32 speed is promised; it should be the place where the project
proves the fast/capability path did not break wind, temperature, cloud, nesting,
or long-run stability.
