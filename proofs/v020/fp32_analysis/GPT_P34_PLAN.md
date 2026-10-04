# GPT point-3/point-4 plan for v0.20.0

Date: 2026-06-21

Analyst: GPT-5 Codex, independent analyst 1 of 2

Scope: planning-only deep analysis of roadmap point 3 and point 4:

* point 3: structural speedup from on-device boundary construction,
  sync-cadence tuning, fused domain cascades, and fused acoustic plus vertical
  implicit solve kernels
* point 4: larger timestep and physics-call cadence, stability-gated

No GPU runs were started. No model source files were edited. This report is the
proof object requested by `BRIEF_P34.md`.

Note on source paths: the brief names
`src/gpuwrf/coupling/boundary_construction.py`; in the current tree the boundary
builder lives at `src/gpuwrf/nesting/boundary_construction.py`.

## 0. Executive verdict

The fp32 reports converged on the important strategic point: the nested all-7
`max_dom=9` benchmark is primarily structural. It is dominated by launch count,
small-grid occupancy, HBM/liveness behavior, orchestration, and codegen shape,
not by a simple fp64 ALU bottleneck. Therefore point 3 and point 4 are the right
near-term levers for wall time.

My ranking:

1. **Point 4 larger timestep is the highest theoretical all-7 speed lever.**
   The all-7 benchmark currently uses `d01=18 s`, `d02=6 s`, and `d03-d09=2 s`
   timesteps. Raising the root step to 27, 36, or 54 seconds reduces the number
   of advances, boundary builds, physics calls, and launches almost
   proportionally if stable. This can beat any kernel micro-optimization. It is
   also the highest physics risk and must be gated through 24-120h wind, temp,
   and cloud skill.
2. **The fused acoustic plus vertical-solve kernel is the highest structural
   kernel lever.** It attacks the tiny dependent kernels and the
   `loop_multiply_fusion` / `input_reduce_fusion` / PCR-tridiagonal pattern
   directly. It also complements the fp32 plan: fp64 brackets can live in
   registers instead of becoming full-array HBM traffic.
3. **On-device boundary builds and domain-cascade fusion are already partly
   implemented and proven valuable historically.** Current code has default-on
   fused cascade support for safe flat one-way subtrees and edge-only boundary
   gathering. The remaining work is to remove residual eager force builds,
   split the megacompile into smaller cached units, and make root-step or
   d02-subtree device programs robust without cold-compile/RSS blowups.
4. **`root_sync_cadence` tuning is low effort and necessary, but not a primary
   speed source once v0.19.1 is in the fused, VRAM-flat regime.** It is a queue
   depth and memory-lifetime policy. It can expose or hide host gaps, but it
   does not remove the GPU kernel work.
5. **CUDA graphs/command buffers should not lead the sprint.** Prior graph
   on/off tests were a no-op for the old leaf program. They become relevant only
   after the step has been reduced to a stable, repeated cascade whose remaining
   bottleneck is CPU launch/replay overhead rather than GPU work.
6. **Physics cadence is real but scheme-specific.** `radt` is honored in the
   operational scan, although the nested pipeline currently targets a fixed
   1800 s cadence. `bldt` and `cudt` are accepted as approximations but the GPU
   port runs PBL/cumulus every dynamics step. Cumulus cadence is an obvious
   correctness and speed cleanup for d01. PBL and microphysics cadence are much
   higher risk because they directly control wind, low cloud, temperature, and
   water.

Expected impact, stated honestly:

* Point 3 from the current v0.19.1-style all-7 baseline: likely **1.1-1.6x**
  incremental if the acoustic/vertical fusion succeeds, with higher upside on
  larger DRAM-bound grids and when combined with compact fp32.
* Point 4 larger timestep: **1.3-2.7x** potential on all-7 depending on whether
  root timestep can move from 18 s to 27/36/54 s without destabilizing 1 km
  nests. This must be proven, not assumed.
* Physics cadence alone: usually **small to moderate** on this benchmark
  (`radt` and `cudt` low single digit percent; PBL cadence could be larger but
  is skill-risky).

The safest v0.20 sequencing is:

1. Establish a fresh warm all-7 structural profile on the v0.19.1 baseline.
2. Implement measurement hardening: NVTX/component ranges, host ledger,
   launch/HLO/memory audits, live-array flatness guard.
3. Tune sync/allocator and remove residual eager boundary builds only where they
   do not reintroduce cold megacompile risk.
4. Prototype acoustic/vertical fusion on savepoints before integrating into
   all-7.
5. In parallel, run a timestep stability ladder: 18 -> 24/27 -> 30/36 -> 45/54,
   with acoustic-substep variants.
6. Only after the short ladder is clean, spend 24-120h forecast-skill budget.

## 1. Current evidence and source facts

### 1.1 Baseline facts from prior proof objects

The relevant historical measurements and plans show two regimes:

* In v0.17, host orchestration was a major wall-clock limiter. The fused
  d02-substep cascade cut dispatches from roughly 47 per root step to roughly 5,
  raised GPU utilization, and measured around 689-702 s/fc-h in the all-7 fast
  mode.
* In v0.18, a later warm profile found a different regime: three recurring
  advance CUDA graphs dominated GPU time, with huge counts of tiny kernels and
  a serial per-step/codegen pathology. In that regime command-buffer off/on was
  a no-op and host fusion alone could not fix the wall.
* In v0.19.1, the closure-cycle VRAM leak was fixed and the all-7 fused
  benchmark measured about 720.9 s/fc-h versus CPU 1020 s/fc-h, finite and with
  all outputs present. This is the practical baseline for point 3/4 planning.

The all-7 topology still matters:

* The proof run reports `advance=15075`, `force=13266`, and `output=81` across
  9 segments, which corresponds to the familiar roughly 5025 advances and
  4422 force/boundary constructions per forecast hour.
* Timesteps in the gate JSON are `d01=18 s`, `d02=6 s`, and `d03-d09=2 s`.
* The namelist has `parent_grid_ratio=3` and `parent_time_step_ratio=3` for
  every child.
* Physics is Thompson `mp_physics=8`, MYNN `bl_pbl_physics=5`, Noah-MP surface,
  and Kain-Fritsch cumulus only on d01 (`cu_physics=1`, children `0`).

### 1.2 Current boundary construction

`src/gpuwrf/nesting/boundary_construction.py` builds a two-time child boundary
package from the just-advanced parent:

```text
old child ring = child current boundary ring
new target     = parent state interpolated to child ring
package        = stack([old, new])
```

The implementation is JAX-native. It reads `parent_state` and `child_state`
arrays and returns a new `State` with updated `*_bdy` leaves. It has a
default-on edge-only path via `GPUWRF_EDGE_ONLY_BOUNDARY`, avoiding the older
full-child-grid-then-slice waste.

Important correctness details:

* It interpolates prognostic perturbations for `p`, `ph`, and `mu`.
* Child base leaves (`PB`, `PHB`, `MUB`) are packed from the child base, not the
  parent base, to avoid terrain/base seams.
* It still computes `child_phb = ph_total - ph_perturbation` and analogous base
  leaves. That is acceptable for the current fp64 path but must be revisited if
  the compact fp32 perturbation state removes totals from the hot carry.

### 1.3 Current domain-tree orchestration

`run_domain_tree_callbacks` has the relevant controls:

* `block_between`
* `root_sync_cadence`
* `fused_cascade`

Its comments correctly state that `block_until_ready` is a queue-depth and
memory-lifetime policy, not a physics-ordering requirement. Parent-to-child
ordering is represented by data dependencies:

```text
parent advance -> boundary package reads parent -> child advance reads package
```

When `root_sync_cadence` is set, per-advance blocking is suppressed and the host
syncs after completed root steps. The v0.19.1 closure fix clears recursive
closure cellvars after integration to prevent carry generations from leaking
across output groups.

The operational fused cascade already exists:

* `_build_fused_cascade_program` jits a parent one-step advance, builds each
  child boundary package inside the jitted program, then advances each child for
  its parent-grid-ratio substeps.
* `_operational_fused_cascade_factory` enables this by default for safe flat
  one-way leaf subtrees.
* It fails closed for root, leaves, non-flat children, feedback, invalid ratios,
  missing weights, and invalid radiation cadences.

This means point 3 is not starting from zero. The next work is to make the
remaining boundary/orchestration work device-resident without reintroducing the
full all-9 megacompile/RSS risk, and to fuse the inner acoustic/vertical kernels
that remain tiny and dependent after domain-level fusion.

### 1.4 Current `_advance_chunk`

`_advance_chunk` defaults to a traced-count `jax.lax.fori_loop`, with a
static-scan diagnostic fallback. It calls `_physics_boundary_step` for each
model step. The per-step body runs:

1. non-timesplit physics at step entry
2. RK3 large-step dynamics
3. acoustic small-step scan
4. post-physics non-dry updates
5. guard/limiter
6. lateral boundary application
7. precision enforcement

`_acoustic_scan` builds an `AcousticCoreState`, computes `calc_coef_w`
coefficients for the vertical solver, and scans
`acoustic_substep_core` for WRF stage-dependent small timesteps:

```text
RK1: 1 acoustic step
RK2: acoustic_substeps / 2
RK3: acoustic_substeps
```

With `acoustic_substeps=10`, each dynamics step runs 16 acoustic substeps. That
is a major launch-generation source on tiny nests.

### 1.5 Current physics cadence

`radt`:

* The operational scan has `radiation_cadence_steps`.
* `run_radiation = step_index % cadence == 0`.
* Held radiation tendencies and held land-surface radiation are carried between
  radiation calls.

Nested pipeline caveat:

* `_radiation_cadence_steps(dt_s)` currently returns a fixed 1800 s target
  divided by each domain timestep.
* The canonical all-7 `run` namelist uses `radt=30`, so that matches.
* The v0.18/v0.19 `run_cadvariant` proof input has `radt=9`, but gate metadata
  still shows 1800 s cadence, for example d01 `radiation_cadence_steps=100` at
  18 s. Early short segments often do not reach radiation, so this mismatch can
  be hidden in speed gates.

`bldt` and `cudt`:

* The scheme catalog and namelist checker state that positive `bldt` and `cudt`
  are accepted as conservative approximations.
* The GPU port runs PBL and cumulus every dynamics step.
* The all-7 namelist has `bldt=0`, so PBL every step is WRF-faithful for this
  case.
* The all-7 namelist has `cudt=5`; d01 has `cu_physics=1`, so the port is likely
  running Kain-Fritsch more often than the requested WRF cadence on d01.

Microphysics:

* Thompson runs every dynamics step. There is no WRF namelist cadence in this
  case analogous to `radt`; reducing microphysics frequency would be an
  algorithmic fast mode, not a WRF-default fix.

## 2. Point 3 strategic analysis

Point 3 should be treated as three different layers, not one generic "fusion"
task.

### 2.1 Layer A: host synchronization and queue depth

This layer controls how far Python dispatches ahead of the GPU:

* `block_between=True`: safe but drains the queue after every advance.
* `root_sync_cadence=1`: sync once per completed root-step cascade.
* `root_sync_cadence=N`: let N root-step cascades queue before waiting.
* segment-level only: maximum overlap, highest memory-retention risk.

Expected reward from current v0.19.1 baseline: **low to moderate**, perhaps
1.00-1.08x, because the fused path is already fast and VRAM-flat. Expected
reward in an unfused/eager host-bound path is much larger; this is why it was
valuable historically.

Risk: larger queue depth can retain more carry generations and scratch buffers.
The v0.19.1 closure fix removes one leak, but it does not make arbitrary queue
depth free. The acceptance gate is flat live arrays and bounded VRAM.

### 2.2 Layer B: on-device boundary package construction and domain cascade

This layer removes Python/JAX dispatch boundaries between:

```text
parent advance
build_child_boundary_package
child advance
```

Current state:

* Boundary construction is JAX-native and edge-only by default.
* Fused cascade already moves boundary construction inside a jitted program for
  safe flat one-way leaf subtrees.
* The root remains excluded by `_fusable_parent`.
* Feedback disables fused cascade.
* Prior full fusion had cold-compile/RSS risk.

Expected reward from current v0.19.1 baseline: **moderate**, perhaps 1.05-1.25x
if residual force builds and dispatches are still material. Higher reward applies
if the run falls back to eager, if root-to-d02 force remains prominent, or if
kernel work is fixed enough that host dispatch is re-exposed.

This is the right place to cut the roughly 4422 force/boundary-builds per
forecast hour, but it has to be done without creating a new 40-60 GB cold
megacompile.

### 2.3 Layer C: acoustic plus vertical implicit-solve kernel fusion

This is the highest-value point-3 kernel lever. It attacks the per-domain
advance graph itself, not just the host recursion around it.

Current acoustic structure:

```text
for each dynamics step:
  RK1: acoustic scan length 1
  RK2: acoustic scan length 5
  RK3: acoustic scan length 10
```

Within each acoustic substep, the model advances pressure-gradient/momentum,
dry mass, theta/geopotential/w, and pressure/rho. `advance_w` includes an
implicit vertical solve. Historical profiles identify many tiny dependent XLA
fusions plus PCR tridiagonal kernels. This is exactly the shape that underfills
the GPU and creates launch/HBM overhead.

Expected reward:

* all-7 current baseline: **1.15-1.6x** if the acoustic/vertical slice is a large
  part of the remaining 720 s/fc-h and the fusion materially reduces launches and
  HBM traffic
* larger DRAM-bound grids: **1.5-2.5x** possible from lower HBM traffic and fewer
  launches
* combined with compact fp32: this is the structural step that can move large
  grids toward the 3-4x story, because fp64 brackets stay in registers instead
  of becoming resident fp64 arrays

This is high effort. It should start as a component savepoint kernel, not as a
whole-model rewrite.

### 2.4 Layer D: CUDA graphs / command buffers

Prior command-buffer experiments were wall-neutral or no-ops for the old leaf
program. That is expected if the program is already captured, GPU-compute-bound,
or dominated by kernels inside CUDA graphs rather than CPU launch overhead.

Command buffers would help after one of these changes:

* the root-step cascade is decomposed into a stable small set of repeated
  programs with remaining CPU launch overhead
* custom kernels reduce GPU work enough that host launch becomes visible again
* boundary and acoustic components are stable-shape and replayable over many
  root steps without new compile keys

They should not be used as a primary design. They are a replay/capture
optimization after the computation has the right shape.

## 3. Point 3 sublever ranking

| Rank | Sublever | Reward | Effort | Risk | Recommendation |
| --- | --- | --- | --- | --- | --- |
| 1 | Fused acoustic + vertical implicit solve | High | High | High | Build as savepoint component first; integrate only after launch/HBM proof. |
| 2 | Split device-resident domain cascade and boundary builds | Medium-high | Medium | Medium-high | Extend existing fused cascade without full all-9 megacompile. |
| 3 | Honor and tune root sync cadence after leak fix | Low-medium | Low | Low-medium | Run A/B sweep; choose largest safe queue depth. |
| 4 | Allocator and live-array flatness hardening | Low-medium | Low | Medium | Needed for stable long runs; speed upside from reducing malloc/free churn. |
| 5 | CUDA graphs / command-buffer replay | Low now, medium later | Medium | Medium | Revisit after components are stable and launch-bound. |
| 6 | Exact-shape sequential leaf bucket fusion | Low-medium | Medium | Medium | Try only after d06/d07 style exact-shape proof; avoid naive `vmap`. |

## 4. Point 3 implementation plan

### P3.0: Measurement and regime gate

Before changing source, the implementation sprint needs a current-regime proof.
Use the v0.19.1 baseline, not stale v0.17/v0.18 assumptions.

Required artifacts:

* warm all-7 `max_dom=9` wall/fc-hour with compile excluded
* event counts: advance, force, feedback, output
* host ledger: advance dispatch, force dispatch, output, explicit sync
* Nsight Systems launch census:
  * kernel launches per root step and per forecast hour
  * graph launches
  * CUDA API sync and alloc/free counts
  * H2D/D2H transfer audit
* HLO audit:
  * scan/while count
  * convert count
  * temp memory
  * dynamic control-flow count
* per-domain/component timing with NVTX or equivalent ranges:
  * d01, d02, each leaf bucket
  * boundary package
  * physics step
  * RK large step
  * acoustic scan
  * `advance_w`/vertical solve
  * pressure refresh

Decision:

* If host ledger sync/force/dispatch is visible, prioritize P3.1-P3.3.
* If GPU graphs/kernels dominate and host sync is near zero, prioritize P3.4.
* If allocator churn is visible, run P3.2 in parallel with memory gates.

### P3.1: Root-sync cadence sweep

Goal: tune queue depth without changing numerics.

Variants:

```text
legacy advance-blocking
root_sync_cadence=1
root_sync_cadence=2
root_sync_cadence=4
root_sync_cadence=8
segment-only sync
```

Gates:

* CPU scheduler identity for events, own-step clocks, and outputs.
* GPU bitwise identity versus `root_sync_cadence=1` for short runs, because
  sync waits are outside traced computation.
* Peak VRAM and live-array count flat over at least 3 output groups.
* No increased D2H/H2D in the timestep loop.
* Warm wall/fc-hour not worse than baseline.

Expected output:

* choose the largest cadence that improves or preserves wall time while keeping
  VRAM flat
* if all cadences are equal, keep `root:1` or the current safe default

### P3.2: Allocator and memory-lifetime A/B

Goal: reduce platform allocator `cudaMalloc/cudaFree` churn without returning to
fragmentation/OOM.

Variants:

```text
XLA_PYTHON_CLIENT_ALLOCATOR=platform
cuda_async
default BFC
```

Gates:

* AC1_FIT / 1 km nest still fits
* all-7 VRAM flat over repeated output groups
* no speed regression
* no live-array growth

This is not a physics change, but it can affect peak memory enough to break long
runs. Treat it as a hardening sprint, not a headline speed claim.

### P3.3: Move remaining boundary/force builds into device programs

Goal: eliminate residual eager boundary builds and reduce force-dispatch count.

Incremental steps:

1. Audit which `force` events are already inside `_build_fused_cascade_program`
   and which remain eager in the v0.19.1 fused path.
2. Add a per-edge jitted boundary builder for any remaining eager edge that
   cannot be folded into the existing cascade safely.
3. Extend fused cascade to root-to-d02 only if cold compile and peak RSS are
   bounded.
4. Split the full all-9 cascade into smaller compile units:
   * d01 root advance/force unit
   * d02 plus leaf-subtree unit
   * optional exact-shape leaf bucket units
5. Keep fail-closed behavior for feedback, non-flat trees, missing weights, or
   unstable compile keys.
6. Preserve edge-only boundary gathering and add an audit proving it did not
   fall back to full-grid gather.

Correctness requirements:

* Boundary package values must be identical to eager boundary construction for
  a short run unless the fused path is explicitly declared a fast mode whose
  acceptance is skill-based.
* Child base leaves must remain child-base owned.
* Nested ph/w boundary forcing stays in the acoustic loop with the existing
  stability rules.
* Compact fp32 mode must not reconstruct base from total-minus-perturbation in
  this boundary path.

Performance gates:

* force/boundary Python dispatch count drops
* graph/kernel launch count drops or host ledger force time drops
* warm all-7 wall improves
* cold compile RSS/time remains bounded

### P3.4: Acoustic plus vertical-solve component fusion

Goal: fuse the tiny dependent acoustic/vertical kernels into fewer, larger
kernels while keeping fp64-sensitive brackets local.

Start with component kernels, not the whole model.

Step plan:

1. Add profiling markers around:
   * `advance_uv_wrf`
   * `advance_mu_t`
   * `advance_w_wrf`
   * `calc_p_rho_step`
   * PCR/Thomas vertical solve
2. Lower HLO for one acoustic substep and one full RK stage. Identify:
   * `loop_multiply_fusion_*`
   * `input_reduce_fusion_*`
   * `pcrGtsvBatchSharedMemKernel*`
   * tiny scalar carry kernels
   * full-grid fp64 temporaries
3. Build a minimal savepoint harness:
   * input: frozen `AcousticCoreState`, coefficients `a/alpha/gamma`, config
   * output: updated acoustic state after one substep
   * oracle: current fp64 JAX path and component savepoints
4. Attempt XLA-level cleanup first:
   * reduce carried state to only evolving leaves
   * close over stage constants
   * test scan carry split only with corrected cache-hit timing
   * avoid prior reverted denominator materialization
5. If XLA cannot reduce launch/HBM count, implement a Pallas/custom kernel for
   the vertical-column part:
   * one block per column or per small column tile
   * vertical dimension in registers/shared memory
   * fp64 brackets only for cancellation-sensitive scalars
   * `w` tridiagonal/PCR/Thomas solve local to the column
   * fp32/compact perturbation storage at kernel boundary when point-1 is active
6. Integrate acoustic substep fusion behind an opt-in mode.
7. Only then test full RK and full all-7.

Do not:

* fuse the whole physics step with acoustic in the first attempt
* materialize denominator arrays that prior probes found slower
* use command buffers as a substitute for changing kernel shape
* keep fp64 brackets as resident full arrays if a register bracket suffices

Correctness gates:

* one acoustic substep
* one RK stage
* one full dynamics step
* 20 steps
* 1h all-7
* then 24h and 72h if stable

Performance gates:

* launches per dynamics step drop
* `pcrGtsvBatchSharedMemKernel` count/time drops or is absorbed into the fused
  vertical kernel
* top tiny kernels by count shrink
* temp memory and HBM proxy bytes drop
* warm wall improves on both all-7 and at least one larger DRAM-bound fixture

### P3.5: CUDA graphs only after refactor

Use command-buffer/cuda-graph capture only when a post-refactor trace shows:

* computation is now a stable repeated cascade
* GPU is not dominated by one multi-second graph or by real compute
* host launch/API time is visible
* capture does not create compile-key or memory lifetime problems

Acceptance:

* graph on/off A/B must show wall improvement
* no numerical difference
* no peak VRAM regression

## 5. Point 3 validation plan

### 5.1 Performance metrics

Record these for every point-3 candidate:

* `s_per_fc_h`, warm, compile excluded
* kernel launches per model step, per root step, per forecast hour
* graph launches per root step
* top 20 kernels by total time and count
* count of kernels under 5 us and under 10 us
* fraction of kernels with grid blocks less than one and two SM waves
* CUDA API time in stream sync, mem alloc/free, graph launch
* H2D/D2H transfer count, with inter-kernel transfer audit
* HLO memory analysis and buffer assignment
* cold compile wall and peak host RSS
* peak VRAM and live-array flatness over repeated output groups

Because hardware counters have been blocked on this workstation before, the plan
must tolerate missing SM occupancy and DRAM bandwidth counters. Use CUDA
timeline duration, grid dimensions, kernel count, and working-set/L2 modeling as
fallback occupancy/HBM proxies. If counters are available, capture them.

### 5.2 Physics/stability validation

Point 3 should be numerically closer to baseline than point 4 because most
changes are structural. Still, fused programs can change FMA/fusion order and
chaotic growth can amplify small differences.

Use the same acceptance policy as point 1:

* Hard fail on NaN/Inf or unphysical blow-up in protected fields.
* Cumulative variables and QVAPOR get the documented carve-outs.
* Wind, temperature, and cloud skill through 24-120h are the production gate.
* Frozen grid-delta manifest remains a diagnostic, not the final veto for a
  fast-mode candidate.

Point-3-specific gates:

* host-sync-only changes should be bitwise identical
* edge-only boundary changes should be bitwise identical to full-grid ring slice
* fused cascade and fused acoustic kernels may not be bitwise, but must pass
  stability and skill gates
* boundary-ring diagnostics must be reported separately for every domain d01-d09

## 6. Point 3 roadblocks and ideas

### 6.1 Cold megacompile and host RSS

The full fused all-7 program has a history of high compile cost and high host
RSS. The fix is not "fuse everything harder"; it is split programs with stable
cache keys:

* root unit
* d02 plus leaves unit
* exact-shape leaf buckets only if proven
* managed precompile outside the critical forecast path

### 6.2 Feedback and non-flat trees

The current fused cascade correctly fails closed when feedback is enabled or the
tree is not a flat leaf subtree. Keep that. For two-way nesting, first build a
separate feedback proof. Do not silently route feedback through a one-way fused
program.

### 6.3 Boundary fp32 interaction

The point-1 compact fp32 plan removes total aliases from the hot carry. Boundary
construction currently depends on total-minus-perturbation to pack child base
leaves. The point-3 implementation must provide explicit base leaves for mixed
mode rather than reintroducing forbidden hot-path total reconstruction.

### 6.4 Vertical solve fusion difficulty

The vertical implicit solve is numerically sensitive and codegen-sensitive.
Likely solutions:

* keep coefficient conditioning fp64 in registers
* store fp32 perturbation inputs/outputs at kernel boundaries
* use shared memory for a column solve
* test Thomas/PCR variants by column oracle
* keep `qke` and steep-terrain sentinels in the validation ladder

### 6.5 Dynamic control flow and cadence

Radiation and physics cadence predicates inside a scan can create dynamic
branches and compile/cache complexity. Where possible, segment forecasts around
cadence events or use held-rate carries with static program shapes.

## 7. Point 4 strategic analysis

Point 4 is different from point 3. It does not merely remove overhead; it changes
the numerical time integration or physics coupling cadence. It can produce the
largest wall-clock win because it reduces the number of model steps.

### 7.1 Larger timestep is the big lever

Current all-7:

```text
d01 dt = 18 s
d02 dt = 6 s
d03-d09 dt = 2 s
acoustic_substeps = 10
RK acoustic substeps per dynamics step = 1 + 5 + 10 = 16
```

Root steps per forecast hour at 18 s: 200.

Candidate root timesteps:

| Root dt | d02 dt | leaf dt | Steps/h | Ideal fewer-step speedup | Risk |
| ---: | ---: | ---: | ---: | ---: | --- |
| 24 s | 8 s | 2.667 s | 150 | 1.33x | Low-medium |
| 27 s | 9 s | 3 s | 133.3 | 1.50x | Medium |
| 30 s | 10 s | 3.333 s | 120 | 1.67x | Medium |
| 36 s | 12 s | 4 s | 100 | 2.00x | Medium-high |
| 45 s | 15 s | 5 s | 80 | 2.50x | High |
| 54 s | 18 s | 6 s | 66.7 | 3.00x | Very high |

The 54 s case corresponds to the common WRF `6*dx(km)` outer-domain heuristic
for 9 km and gives 6 s on 1 km nests, but that does not mean it is stable for
this port, this terrain, this vertical grid, this damping, and this nesting
configuration. It is a candidate, not an assumption.

Expected real speedup is less than ideal because output, compile, initialization,
and some fixed costs do not scale with step count. But on a launch-heavy all-7,
larger dt is still the cleanest path to large wall-time reduction if stable.

### 7.2 Acoustic-substep tradeoff

When increasing `dt_s`, there are three choices:

1. **Keep `acoustic_substeps=10`.**
   * Maximum speed.
   * Increases sound-step size and vertical/acoustic CFL.
   * Highest stability risk.
2. **Scale acoustic substeps with dt.**
   * Example: root dt 36, acoustic substeps 20.
   * Keeps acoustic substep size roughly constant.
   * Reduces physics, boundary, large-step, and output work, but acoustic work
     per forecast hour does not fall as much.
3. **Intermediate mode.**
   * Example: root dt 36, acoustic substeps 12 or 15.
   * Trade speed against acoustic stability.

The implementation should sweep dt and acoustic_substeps together. A larger
outer timestep with the same acoustic substep size may still be worthwhile if
physics/boundary/large-step work is a large fraction of wall time.

### 7.3 Radiation cadence

Radiation cadence is already represented by held tendencies. Increasing `radt`
from 30 to 60 minutes reduces radiation calls by about half. The all-7 measured
share for radiation has historically been small to moderate, so expected speed
gain is likely only low single digits unless radiation is prominent in the fresh
profile.

Risk:

* surface temperature diurnal phase
* low-cloud timing
* SWDOWN/LWDOWN bias
* PBL coupling through held surface radiation

First correctness cleanup:

* make nested pipeline cadence source explicit
* either honor namelist `radt` or record that a fixed 1800 s target is an
  intentional fast-mode/proof-mode choice
* avoid silently comparing a GPU run with a different radiation cadence to CPU
  WRF

### 7.4 Cumulus cadence

The all-7 namelist has `cudt=5` and `cu_physics=1` only on d01. The port warns
that `cudt` is not honored and runs cumulus every dynamics step. Honoring `cudt`
is likely both more WRF-faithful and faster on d01.

Expected speed gain on all-7: small, because d01 cumulus is a small fraction of
the total. Still worth doing because it removes a known approximation.

Implementation must preserve WRF semantics:

* scheme carry such as `w0avg`/`nca` must update only on call steps
* tendencies or state increments must be held/applied in the WRF-consistent way
* do not simply multiply a state delta over a longer interval unless that is the
  scheme's WRF behavior

### 7.5 PBL cadence

The all-7 namelist has `bldt=0`, so PBL every dynamics step is WRF-faithful for
the benchmark. Introducing a nonzero PBL cadence would be an algorithmic fast
mode, not "honoring WRF defaults."

Potential reward is larger than cumulus because MYNN/PBL/surface has been a
large wall-time slice in historical budgets. Risk is also high:

* U10/V10 and low-level wind are protected variables
* PBL controls low cloud and temperature
* qke has a known d03 1 km nonfinite history
* surface flux feedback can destabilize near neutral/strong-terrain columns

Recommendation: do not lead with PBL cadence. Test only after structural and
timestep work, and only with 24-120h wind/temp/cloud gates.

### 7.6 Microphysics cadence

Thompson microphysics normally runs every dynamics step in this configuration.
Downsampling it would be a fast-mode approximation, not WRF-default cleanup.

Expected all-7 speed gain from every-2-step Thompson is probably modest unless
the fresh profile shows Thompson has again become dominant. Cloud skill risk is
high because hydrometeors, cloud fraction, radiation, and precipitation timing
are directly affected.

Recommendation: leave microphysics cadence as a late experiment. If tested,
start with a held-tendency or split slow-process design and gate on cloud objects,
cloud fraction, SWDOWN/LWDOWN, and temperature.

## 8. Point 4 implementation plan

### P4.0: Add timestep/cadence diagnostics first

Before changing timestep:

* record per-domain timestep, parent_time_step_ratio, parent_grid_ratio
* record acoustic substeps and derived acoustic dt per RK stage
* compute horizontal Courant diagnostics for `u`, `v`
* compute vertical Courant diagnostics for `w`
* compute acoustic/sound-step proxy diagnostics
* record max Courant by domain, vertical level, boundary ring, and terrain class
* record physics call counts by scheme and domain:
  * Thompson
  * surface layer
  * Noah-MP
  * MYNN/PBL
  * cumulus
  * radiation
* record whether `radt`, `cudt`, and `bldt` came from namelist or fixed GPU
  defaults

These diagnostics are cheap compared with a failed 24h run.

### P4.1: Timestep ladder, short stability

Run a short ladder before any 24h forecast:

```text
baseline: root dt 18, acoustic_substeps 10
dt 24: substeps 10, 12, 14
dt 27: substeps 10, 15
dt 30: substeps 10, 15, 17
dt 36: substeps 10, 15, 20
dt 45: substeps 15, 20, 25
dt 54: substeps 20, 30
```

Each candidate climbs:

1. one dynamics step
2. 20 root steps
3. 0.1h
4. 1h
5. 3h
6. 6h

Abort on:

* any NaN/Inf outside cumulative/QVAPOR carve-outs
* pressure, mass, or geopotential boundary-ring growth
* horizontal or vertical wind runaway
* qke nonfinite or large spike in d03-d09
* temperature outside physical bounds
* cloud water/ice explosion or collapse
* superlinear divergence growth

Select at most two candidates for 24h:

* fastest candidate that passes the 6h ladder
* conservative candidate with scaled acoustic substeps

### P4.2: Timestep 24-120h skill gate

For selected candidates:

* 24h all-7 `max_dom=9`
* 48h if 24h green
* 72h if 48h green
* 120h only for the final candidate

Production acceptance uses the same policy as the fp32 analysis:

* no blow-up in protected fields
* wind/temp/cloud skill preserved
* cumulative/QVAPOR carve-outs respected
* frozen grid-delta manifest diagnostic only

Use domain-by-domain reporting. The 1 km leaves are the real stability gate.

### P4.3: Radiation cadence cleanup and sweep

First fix the semantics decision:

* if the run should honor namelist `radt`, wire nested pipeline to use
  `radt*60/dt_s` per domain and report it
* if the run intentionally pins 1800 s for proof comparability, label it in the
  proof and do not call it WRF-cadence faithful for namelists with `radt != 30`

Then sweep:

```text
radt 30 min baseline
radt 45 min
radt 60 min
radt 90 min only if 60 min is skill-clean
```

Gates:

* SWDOWN/LWDOWN bias
* T2 diurnal phase
* low-cloud fraction and cloud timing
* PBLH and U10/V10
* no skill degradation at 24-120h

Expected speed gain: low single digit percent unless fresh profile shows
radiation is larger than expected.

### P4.4: Cumulus cadence honoring

Implement WRF-like `cudt` for d01 first.

Plan:

1. Add call-cadence state for cumulus schemes.
2. For KF, update `w0avg`/`nca` only on call steps.
3. Determine WRF-consistent held fields. Do not hold a raw post-state delta if
   WRF holds a tendency or a scheme-specific accumulator instead.
4. Leave children with `cu_physics=0` unchanged.
5. Validate d01 first, then all-7.

Gates:

* d01 wind/temp/cloud skill
* downstream d02 boundary impact
* no unexpected precipitation/cumulative explosion
* all-7 finite and stable

Expected speed gain: small for all-7, but this removes a known approximation and
can reduce d01 physics launches.

### P4.5: PBL cadence fast-mode experiment

Only after P4.1/P4.4:

Candidates:

```text
PBL every step baseline
PBL every 2 steps
PBL every 5 steps or 1 minute equivalent
```

Use a true tendency-hold design where possible, not a stale state overwrite.
Surface layer and land-surface coupling must be handled together because MYNN
reads flux handles written by surface/land code.

Gates:

* U10/V10 skill is primary
* T2 and PBLH
* low-cloud and marine cloud fraction
* qke finite and bounded
* surface fluxes finite and physically plausible

Expected speed gain: potentially 5-20 percent if PBL/surface dominates the
fresh profile, but high risk. This should remain opt-in fast mode unless the
full 24-120h gate is clean.

### P4.6: Microphysics cadence fast-mode experiment

Late experiment only.

Candidates:

```text
Thompson every step baseline
Thompson every 2 steps
adaptive skip only in dry/no-cloud columns
```

The adaptive version is more plausible than a global skip because cloud skill is
protected. It is also more complex and can introduce divergence/control-flow
issues.

Gates:

* cloud object metrics
* cloud fraction
* hydrometeor nonnegative/finite
* SWDOWN/LWDOWN
* T2
* precipitation accumulators finite, monotone, and bounded under carve-out

Expected speed gain: probably modest on all-7 unless Thompson is again a top
profile item. Risk is high.

## 9. Point 4 validation and acceptance gates

Use the concrete point-1 policy. Summary:

### Universal hard fail

Fail on NaN/Inf or unphysical blow-up in:

```text
U, V, W, T, THETA, T2, P, PB, PH, PHB, MU, MUB, PSFC,
QKE, protected cloud fields, boundary ph/w leaves, restart-critical state
```

Pressure, temperature, wind, and qke physical bounds follow the point-1 report.
QVAPOR and cumulative variables have carve-outs from tight pointwise parity, not
from finite/nonnegative/stability checks.

### Protected forecast-skill fields

Wind:

* U10/V10
* lowest-level U/V
* mid-level winds
* boundary-ring winds

Temperature:

* T2
* lowest-model-level temperature
* theta
* boundary-ring temperature

Cloud/radiation:

* QCLOUD/QICE/QSNOW/QGRAUP
* cloud fraction
* column cloud water/ice
* SWDOWN/LWDOWN
* radiative heating tendencies
* PBL/cloud coupling fields

### Timestep-specific gates

Add these to the point-1 gates:

* max horizontal Courant by domain does not exceed the baseline envelope by more
  than the predeclared candidate ratio
* vertical Courant does not develop a growing tail
* boundary-ring Courant and ph/w forcing remain stable
* acoustic-step diagnostics remain bounded
* no systematic time-phase drift in wind/temp/cloud objects that grows faster
  than baseline forecast chaos

### Cadence-specific gates

Radiation:

* T2 diurnal phase
* SWDOWN/LWDOWN bias
* low-cloud response

Cumulus:

* d01 precipitation and cloud fields
* d02 boundary impact
* downstream wind/temp/cloud

PBL:

* U10/V10
* PBLH
* qke
* surface fluxes
* low cloud

Microphysics:

* cloud objects
* hydrometeor fields
* radiation coupling
* precipitation accumulators under carve-out

## 10. Combined roadmap

### Stage 1: Measurement hardening

Deliver:

* current all-7 structural profile
* host ledger
* launch/HLO audit
* cadence/timestep diagnostics
* live-array/VRAM flatness guard

No performance claim without this.

### Stage 2: Safe structural cleanup

Deliver:

* root-sync cadence A/B
* allocator A/B
* residual boundary-build audit
* per-edge or split-cascade boundary device programs where safe

Expected result: small to moderate speedup, better robustness, cleaner proof.

### Stage 3: Timestep ladder

Deliver:

* dt/acoustic-substep short ladder
* pick two candidates for 24h
* reject unstable candidates early

This has the best chance of a large all-7 wall-time win.

### Stage 4: Acoustic/vertical fusion prototype

Deliver:

* savepoint component kernel
* launch/HBM reduction proof
* one-step and 20-step stability

Only integrate into all-7 after component proof.

### Stage 5: Cadence cleanup

Deliver:

* explicit `radt` semantics
* honored `cudt` for d01
* optional PBL/microphysics fast-mode experiments if still needed

### Stage 6: 24-120h acceptance campaign

Deliver:

* nested all-7 `max_dom=9`
* 24h, 72h, and final 120h skill metrics
* performance and profiler artifacts
* explicit statement of which mode is default-safe and which is fast-mode

## 11. How nested all-7 stays functional

Rules:

* Every candidate runs on all domains d01-d09.
* No candidate can be accepted from a single-domain proof.
* Parent/child time ratios remain consistent unless a dedicated nesting ADR
  changes them.
* Boundary update cadence remains equal to parent timestep for live nests.
* `ph`/`w` nested boundary forcing remains protected.
* Feedback-disabled assumptions remain fail-closed.
* History output times must remain aligned and readable for every domain.
* Cache keys and compile units must be stable across all nine domains.
* Long runs must keep VRAM flat across output groups.

For point 3, default-safe changes are those that preserve operation order and
bitwise identity, such as host sync placement. Fused fast paths may be
skill-accepted but must be labeled.

For point 4, all timestep and cadence changes are fast-mode until the 24-120h
skill gate passes. They are not bitwise-preserving and should not be presented
as equivalent to WRFv4.

## 12. Roadblocks needing creativity

Ranked ideas:

1. **Timestep stability at 1 km.**
   * Largest payoff.
   * Needs CFL diagnostics, acoustic-substep sweep, and terrain/boundary
     sentinels.
2. **Acoustic/vertical fused kernel.**
   * Highest structural kernel payoff.
   * Needs custom kernel discipline and savepoint oracles.
3. **Compile-unit decomposition.**
   * Necessary to keep fusion operational.
   * Split programs without losing most host savings.
4. **Correct physics tendency holding.**
   * Required for `cudt`, possible for PBL/radiation.
   * Scheme-specific; state-delta holding is often wrong.
5. **Boundary package in compact fp32 mode.**
   * Must avoid total-minus-perturbation resurrection.
   * Needs explicit base leaves through boundary reconstruction.
6. **Occupancy for tiny leaves.**
   * Exact-shape leaf bucket fusion may help.
   * Naive `vmap` has measured slower behavior; use sequential bucket fusion or
     persistent kernels only after proof.
7. **Profiler limitations.**
   * If hardware counters remain unavailable, rely on timeline/grid/L2 proxy
     artifacts and be explicit about missing counters.

## 13. Final recommendation

Do not spend v0.20 point-3 effort merely toggling CUDA graphs or moving one more
dtype boundary. The structural work should start from the current v0.19.1
fused, VRAM-flat baseline and attack two things:

1. residual host/boundary orchestration without reintroducing megacompile risk
2. the inner acoustic/vertical tiny-kernel graph

For point 4, prioritize a timestep ladder before exotic physics cadence changes.
The all-7 root timestep is conservative enough that a stable increase would be a
larger speed lever than most kernel refactors. But this is also the easiest way
to destroy wind/temp/cloud skill, so it must be handled as an opt-in fast mode
until the 24-120h acceptance campaign passes.

Best near-term combined bet:

```text
safe structural cleanup
+ root dt 24/27/36 stability ladder
+ explicit radt/cudt semantics
+ acoustic/vertical fusion prototype
```

A realistic success story is not one magic knob. It is:

* fewer root steps if stability allows
* fewer boundary/force dispatches
* fewer tiny acoustic/vertical kernels
* compact fp32 storage where point 1 permits it
* no blow-up
* wind/temp/cloud skill preserved through 24-120h
* nested all-7 `max_dom=9` fully functional

