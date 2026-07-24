# wrf_gpu v0.23.4 — nine-nest correctness closure; d03 performance regression measured, root-caused, understood

**Type:** correctness release on top of v0.23.3, with a partial performance fix
and a fully investigated (but not fixed) performance regression for `d03`.
**Status: prepared, pending publication.**

## Release status — read this first

This project cuts a tag only after every known release blocker is either
solved green, proven computationally unsolvable/not-quick-fixable with
evidence, or proven irrelevant. As of this writing:

- **Correctness is fully closed.** All five nine-nest acceptance criteria are
  GREEN (SP2, S2, V10, late-Ni, nine-nest replay), independently reviewed and
  backed by sealed proof objects.
- **Performance investigation is closed; the regression itself is not fixed.**
  Two-domain (`d01`+`d02`) configurations are measured unaffected. Three-domain
  and nine-domain configurations that engage `d03` carry a real, production-
  measured **51.67% regression** versus v0.23.3. Every quick-fix candidate
  (barrier, scan/map, register-pressure/fusion-disable, B1/B2/B3 scheduling,
  roll substitution) was falsified or rejected with evidence, and a Nsight
  Compute hardware measurement of the isolated dominant kernel confirmed it is
  **occupancy/latency-bound, not memory-bandwidth-bound** (DRAM throughput
  20.3% of peak, achieved occupancy 15.6%, register-limited to 2 of 24 possible
  concurrent blocks). This rules out a reduced-precision fix and identifies
  concurrent sibling-domain scheduling as the mechanistically-mapped next
  lever — design-scoped, not yet implemented, with no established delivery
  schedule or proven payoff. **This regression ships as a known, understood
  limitation, not an open question.**

Full evidence trail: `.agent/decisions/VERSION-SPRINT-LEDGER.md`, entries dated
2026-07-20 through 2026-07-24.

## Correctness fixes

The release is one linear correctness chain; later gates include every earlier
fix.

1. **SP2 physics chain (`4d1a32f7`).** The MYNN mass-flux seam and its
   WRF-faithful surface/physics handoff were repaired. The 24-hour real-case
   drift gate wrote 24/24 finite hourly outputs. Its worst binding normalized
   RMSE was 1.4311; endpoint T2/U10/V10/wind-speed normalized RMSE was
   0.6831/0.9537/0.9162/0.9467. No tolerance was weakened.
2. **S2 boundary retention (`1bfee4e9`).** The accepted frozen WRF
   child-boundary bundle was proven to survive production runtime construction.
   The retained path was already correct, so this gate required no numerical
   workaround.
3. **V10 wake displacement (`3b81fb5b`).** The nested GPU path silently ran
   `moist_adv_opt`/`scalar_adv_opt` at `0/0` where the fixture namelist specifies
   `1/1`. Restoring WRF-faithful `sumflux` and post-acoustic-reorder scalar
   transport closed the deterministic step0→step9000 replay. Re-confirmed after
   the late-Ni fix, `V`/`V10` RMSE remained **0.9308318930/1.9509135196**
   (gates ≤1.1318203206/≤2.1128268858), with d03 U10/V10 correlation
   **0.8654/0.8114**.
4. **Thompson late-Ni (`af5a27b9`).** A missing pristine-WRF
   pre-sedimentation ice mass/number balance allowed an unphysical ice terminal
   fall velocity up to 856.956 m/s (WRF-faithful value: 0.63917 m/s), exploding
   ice mass about 7.93×10¹³× at native step 1148.
   `_balance_ice_number` now applies the source-literal balance for mp=8 and
   mp=28 before fall-speed calculation, without a clamp, mask, or
   `nan_to_num`. Eleven new and 27 oracle tests are green. This is **not
   baseline-neutral**: it changes fall speed in 98.1% of ice-bearing cells, so
   prior ice-sensitive trajectories require rebaselining.
5. **Nine-nest acceptance (`484987e1`).** The one-hour, all-physics,
   all-seven-island `max_dom=9` replay completed d01–d09 with their own timestep
   counts and wrote all 27 expected outputs. All 177,497,511 numeric values were
   finite; all nine domain comparators passed all 102 fields under the frozen
   v0.14 tolerance manifest with zero failures. Canonical proof SHA-256:
   `f8c64f85e16f083e2c0c6509c6c9e2d2bcc64ec6da6f313eb08714af638c0916`.

The independent gap critic accepted the closure; its report SHA-256 is
`8546a4329784c95a8899f9d4e9eace4b75520cfe16612ad68905ecbeb4317f59`.

## Validation and resource evidence

- SP2 24-hour run: forecast wall 4,484.88 s (186.87 s per forecast hour), peak
  VRAM 12,405 MiB, peak host RSS 14,580.8 MiB, zero swap; 39/39 closeout tests
  passed.
- Final nine-nest correctness replay: warm model wall 5,744.407 s, peak VRAM
  16,624 MiB, peak host RSS 35,773,432 kB. These describe the accepted stress
  fixture, **not** a cross-version speed claim.
- No fixture, CPU truth, comparator, tolerance manifest, or model output was
  altered to make the final scorer green.

## Scalar-batching dispatch fix — validated, partial

The V10 correction made eight-species moist/number scalar transport genuinely
active on the nested GPU path, where it had previously been a silent no-op.
Under `d03`'s nine substeps per root step, that correct work is expensive.

The dispatch fix batches nested transport species with `jax.vmap`, applies only
below a 524,288-cell threshold, and is restricted to the limited/final-RK-stage
path. An earlier unscoped form caused a **3.52× d02 regression** because d02 is
bandwidth-bound and the stack/unstack traffic cost more than it saved. The final
form passed three independent review rounds and the long-horizon
width1-vs-width2 WRF-fidelity gate (root steps 4/16/67):

- `d01` and `d02` are bit-exact at every checkpoint.
- Every primary-field RMSE is inside its frozen acceptance envelope; the worst,
  d03 vertical velocity `w` at root step 67, is `9.17e-12` of its 0.3 m/s
  envelope.
- **Two-domain performance is effectively parity:** overall ratios
  **0.960–1.018×** versus v0.23.3; d02 is 0.6686075 versus
  0.6647537 s/root-step (+0.58%).

This fix removes a real secondary dispatch cost. It does not remove the
dominant FCT-limiter kernel cost.

## d03/FCT-limiter regression — measured, real, unresolved

The definitive production-faithful measurement used the real
`execute_nested_pipeline` entry point for one forecast hour with segments
67/67/66 and weighted segments 2+3. Both arms were finite and
`PIPELINE_GREEN`, with tight spreads (v0.23.3 3.11%, candidate 0.57%):

- v0.23.3: **1.9668543059 s/root-step**
- correctness-sealed v0.23.4 candidate: **2.9831176877 s/root-step**
- ratio: **1.516694795 — 51.67% slower**

A CPU component profile attributes 79.95% of the added `d03` cost to the final
limited/FCT lane itself. The limited call is 5.237× slower than the unlimited
path on only 1.087× more HLO operations, but 2.188× more estimated bytes
accessed and 1.903× more FLOPs. This is a memory-traffic/lowering inefficiency
inside flux renormalization, not a dispatch- or batching-width problem.

The optimization campaign has not produced a releasable fix:

- `optimization_barrier` was a structural no-op because XLA re-fused across it.
- `lax.map`/`scan` reduced HLO size but introduced a real numerical divergence.
- disabling multi-output fusion reduced registers 255→60 but made runtime
  worse, refuting register pressure as the dominant mechanism.
- B1/B2/B3 scheduling and fusion retests cannot offset the regression; the
  relevant dispatch sequence is already identical on this fixture.
- roll substitution passed a small single-shape check but failed the
  production-scale exact-output gate, with d01 U10 already differing at 5,483
  points (max absolute difference 0.189). It also measured
  `candidate_over_v0233_ratio=1.389`, so it was rejected and not integrated.

A follow-up Nsight Compute hardware measurement against the isolated dominant
kernel (`loop_select_subtract_fusion`) settled the open question from the
investigation above: the kernel is **occupancy/latency-bound, not
memory-bandwidth-bound** (DRAM throughput 20.3% of peak, achieved occupancy
15.6%, dominant warp-stall reason `short_scoreboard`, register-limited to only
2 concurrent blocks of 24 possible). This rules out reduced-precision
intermediates as a fix (they address a bandwidth ceiling this kernel does not
hit) and identifies concurrent sibling-domain scheduling (running the tiny
d03-class leaves alongside other domains' work to hide the occupancy
underfill) as the mechanistically-mapped lever — design-scoped in
`RANK6_SIBLING_SCHEDULING_DESIGN.md`, not implemented or validated, with no
established delivery schedule or proven payoff. The 51.67% production
measurement remains the release authority; this regression is understood down
to the hardware root cause but not yet fixed.

## Practical guidance

- **Single-domain and two-domain configurations are correctness-sealed and have
  no measured performance regression.**
- **Three-domain and nine-domain configurations that engage `d03` are
  correctness-sealed but approximately 1.5× slower per root step than v0.23.3.**
  This is a wall-clock cost, not a correctness risk.

## Carried limits and deferred work

- Nested execution targets a fixed 1,800 s radiation cadence instead of
  honoring arbitrary namelist `radt`; `topo_shading` and `slope_rad` currently
  bind disabled on the nested runtime. The one-hour acceptance fixture remains
  valid, but no longer-horizon radiation-fidelity claim follows from it.
- Broad seasonal/configuration-independent 24–72 hour forecast-skill
  equivalence remains open. This release proves the stated frozen fixtures, not
  universal WRF-scheme coverage.
- The single-domain daily pipeline still drops requested
  `moist_adv_opt`/`scalar_adv_opt` values and runs `0/0`; the accepted nested
  path honors them.
- With `time_step_sound` omitted, the runtime selects 10 acoustic substeps where
  pristine WRF derives 4 on the tracked fixture; residual dry-mass behavior is
  unresolved.
- The Thompson `NSED_MAX=16` static cap and widespread exact-zero carried Ni
  remain separate fidelity debt.
- Shape-only fused-phase leaves and namespace-specific terrain provenance can
  fragment AOT cheap keys and force a safe recompile.

## Upgrade note

v0.23.4 intentionally changes accepted physics results relative to v0.23.3.
Do not use an old ice-sensitive golden trajectory as a byte-identity
expectation; validate or rebaseline it against the v0.23.4 evidence chain.
Compilation caches remain version-keyed and do not cross-load v0.23.3
executables.

## Files and proof trail

Correctness changes include the V10 scalar-transport ordering fix and Thompson
`_balance_ice_number`; performance changes include the size-thresholded
limited-stage species batch. The correctness-sealed source is merged into
`main`; this release-prep branch adds documentation and packaging only.
Canonical per-commit evidence and proof hashes are indexed in
`.agent/decisions/VERSION-SPRINT-LEDGER.md`.
