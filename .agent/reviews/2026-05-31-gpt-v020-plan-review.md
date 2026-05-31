# GPT-5.5 independent v0.2.0 plan review - 2026-05-31

Scope: read-only plan critique of `.agent/decisions/V0.2.0-PLAN.md` on branch
`worker/opus/final-verdict`, grounded in the roadmap, gap inventory, v0.1.0 tag
checklist, and the prior GPT HFX/proof review. I did not change model code. The
only write is this requested review artifact.

## Review Decision

Do not execute the v0.2.0 plan unchanged. The broad ordering is plausible, but the
plan has four material defects:

1. It schedules the CPU-WRF corpus bottleneck too late.
2. It overstates Wave-1 parallel safety.
3. It lets several quality claims close on gates that are too gameable.
4. It omits or under-scopes work needed for the stated "all gap items except native
   init" promise.

Required amendment before dispatch: add a Wave -1 corpus/backfill track, split
several items into "early diagnostic" vs "final closure", move the MYNN/HFX parity
debt earlier, merge boundary-order work into P0-6, and make the release cadence
coarser than "tag every sub-item" unless the tag is externally meaningful.

## Findings

1. **BLOCKER - Full TOST is a calendar bottleneck and must start before Wave 1, not after Wave 4.**

   The plan correctly labels the real n=3 GPU-vs-CPU comparison as an underpowered
   descriptive check. That is honest only if it never appears as an equivalence PASS.
   The issue is the 0.2.0 TOST gate: the plan places "Full seasonal TOST" in Wave 4
   after Noah-MP and multi-GPU, but the corpus is already known to be the long pole.

   Current corpus evidence says only 2 pinned-grid-complete L3 24 h members existed
   at scout time, likely 3 after the live run, while the old Tier-4 path needed at
   least 10. The v0.2 plan now asks for >=15 cases, and ADR-029 says n=15 is still
   underpowered under the provisional 20% sigma assumption; n around 27 is the
   planning target for a 10% RMSE difference. The user-provided estimate of ~68-118
   CPU-h backfill is therefore not a side detail. It is a release-critical dependency.

   Concrete amendment: create **Wave -1: CPU-WRF corpus backfill and TOST manifest**,
   starting immediately and running independently of GPU implementation. Its proof
   object must include selected cases, rejected cases, season label, same-grid
   check, complete-pair mask rules, CPU-WRF runtime/disk status, and frozen ADR-029
   margins. If the resulting corpus is all MAM/late-May, call the gate
   **single-season MAM equivalence**, not seasonal equivalence. If n is 15 and the
   empirical sigma is above the planning value, the release must say underpowered
   even if the point deltas look good.

2. **BLOCKER - P0-6 must own P1-6 boundary-order work and must close before P0-1 final skill.**

   The sequence "P0-6 before P0-1" is right, but the plan currently treats P1-6
   positive-definite/monotonic advection plus boundary order as a later independent
   Wave-3 item. That is wrong. The gap inventory defines P0-6 as map factors,
   terrain terms, lateral specified/nested boundaries, and boundary-aware advection
   order degradation. Those are the same operational surface where P1-6 lives.

   If P1-6 runs later, P0-6 can falsely close while the boundary-order part of real
   nested dynamics is still unresolved. Conversely, if P1-6 edits flux advection or
   boundary degradation after P0-6, it can invalidate P0-6 wind-skill and no-regression
   evidence.

   Concrete amendment: fold P1-6 into P0-6 or make it an immediate P0-6 subgate.
   P0-6 closure should require WRF-savepoint or analytic evidence for map factors,
   terrain slope/diffusion/PGF pieces, specified/nested boundary treatment, and
   boundary-order degradation. Only after that should P0-1 attempt a live parent-child
   skill claim.

3. **BLOCKER - P0-1 live nesting needs P0-4 d01 Kain-Fritsch before its final skill gate.**

   A parent-to-child scheduler can be built before d01 cumulus, but a live d01 -> d02
   -> d03 forecast cannot be claimed WRF-faithful if d01 lacks the Kain-Fritsch scheme
   used by the Canary baseline. The plan currently puts P0-1 before P0-4 and gives
   P0-1 the gate "self-contained nested run reproduces the replay-nest skill, no
   boundary pump." Without d01 KF, any "skill" result is testing a known-wrong parent.

   Concrete amendment: split P0-1 into:
   - **P0-1a nesting scheduler/interpolation/subcycling**: prove parent-to-child
     boundary construction using recorded/controlled parent states, no skill claim.
   - **P0-4 d01 KF**: WRF oracle or instrumented call parity, plus parent-domain
     convective precip sanity.
   - **P0-1b live hierarchy skill**: d01 -> d02 -> d03 live run vs CPU-WRF, after
     P0-6 and P0-4 are closed.

4. **MAJOR - P1-4 MYNN/HFX parity must move earlier than the plan shows.**

   The prior GPT review's G1/G3 findings are not cosmetic. The current HFX repair is
   explicitly not a faithful `module_sf_mynn.F` port, and moisture/PBL regressions
   remain a first-order risk. The v0.2 plan places P1-4 in Wave 3. That means P0-6
   could close its wind-skill proof while surface-layer/MYNN defects are still a
   known confound, and the seasonal TOST could later measure a pile-up of dycore,
   boundary, and MYNN fixes without isolating the known HFX/PBL debt.

   Concrete amendment: add **P1-4a in Wave 1** for the exact prior review debts:
   MYNN land `z_t`/`zolrib` semantics, `PSIH2`/`PSIH10` baselines, prior-step UST
   behavior, and Q2/LH/QFX/MOL/PBLH no-regression over land/water and stable/unstable
   regimes. Full EDMF/cloud can stay later, but no final wind-skill or TOST
   equivalence claim should close before P1-4a closes.

5. **MAJOR - Wave-1 "disjoint file ownership" is overstated.**

   The plan says P0-7, P1-5, P1-8, and P0-5 are disjoint across microphysics,
   precision-config, and I/O. The actual Phase-B ownership contract makes several of
   these shared-core or proof-coupled:

   - `operational_mode.py`, `state.py`, `precision.py`, `grid.py`, `halo.py`, and
     `dynamics/**` are shared-core / manager-merge only.
   - `physics_couplers.py` is a shared file, lane-extensible only by adapter body and
     serialized by the manager.
   - P0-7 conservation and non-masking touches guards, limiters, Thompson fallback,
     precip/water side channels, and whole-pipeline validation.
   - P1-8 precision policy touches `contracts/precision.py`, `State` dtype behavior,
     `operational_mode.force_fp64`, Thompson fp32 controls, restart, and output
     dtype fidelity.
   - P0-5 restart/output completeness touches `wrfout_writer.py`, `daily_pipeline.py`,
     checkpoint/restart, state coverage, and eventually Noah-MP/MYNN/precip carries.

   Unsafe concurrent pairs:
   - **P0-7 with P1-5**: water conservation, precip accumulators, invalid-column
     fallback, and Thompson side channels are the same behavioral surface.
   - **P0-7 with P1-8**: budget tolerances depend on declared precision mode; do not
     freeze budget thresholds while dtype policy changes underneath.
   - **P0-5 with P1-8**: restart and wrfout dtype/metadata semantics depend on the
     precision contract.
   - **P0-5 with P0-3/P1-4/P1-5 final closures**: a "full wrfrst" cannot be final
     until land, PBL, precip accumulators, radiation carry, and diagnostics are final.
   - **P0-6 with P0-7/P1-8**: dycore/boundary changes alter conservation residuals
     and may require precision-specific tolerances.

   Concrete amendment: allow parallel *diagnostic scaffolding* only. Final proof
   closures that freeze tolerances, restart schemas, precision mode, or whole-run
   budgets must serialize through a manager merge and rerun shared end-to-end gates.

6. **MAJOR - Several proof gates are too weak or gameable.**

   Required gate hardening:

   - **P0-6**: "d02/d03 wind-skill improves, no regression vs v0.1.0" is not enough.
     It is relative and can pass while still wrong. Require WRF-source/savepoint or
     analytic evidence for each real-grid operator plus absolute CPU-WRF/persistence
     station gates for U10/V10/T2/Q2/PBLH, guard/limiter engagement counts, and
     no boundary pump on d03.
   - **P0-1**: "reproduces replay-nest skill" is ambiguous. Require live parent-child
     boundary fields compared to CPU-WRF at the boundary strip for U/V/W/T/QV/P/PH/MU,
     interpolation conservation, child subcycling cadence, no in-loop host transfer,
     and d03 skill on the same masks.
   - **P0-4**: "d01 precip sanity" is too soft. Require Kain-Fritsch WRF call/savepoint
     parity for tendencies and convective precip accumulators, or explicitly declare
     a validated physics-contract deviation.
   - **P0-5**: split into output coverage and restart. Restart proof must include
     true model state needed to resume, not just pickle continuity or final-wrfout
     similarity. Final P0-5 must wait until Noah-MP, MYNN, radiation diagnostics, and
     precip accumulators are in their final state.
   - **P0-7**: define dry mass, total water, precip sink/source, heat/enthalpy, and
     limiter/guard accounting before running. "Energy budget closes" is otherwise
     too vague in a nonhydrostatic terrain-following model.
   - **P1-3**: "island diurnal heating vs corpus" must become SWDOWN/GLW/toposhade/
     slope/aspect/lat-lon WRF oracle checks across solar geometry and cloudy/clear
     regimes.
   - **P1-4**: include the exact G1/G3 prior-review items, stable/unstable, land/water,
     HFX/LH/QFX/Q2/MOL/PBLH/U10/V10, plus EDMF/cloud active cases if claiming full MYNN.
   - **P1-5**: require predeclared precip and hydrometeor tolerances, WRF savepoints,
     accumulators, water closure, and invalid-column fallback counts. Do not let
     "within tol" be chosen after seeing the residual.
   - **P1-8**: declare the operational release mode before scoring. If fp32 has no
     speed win, the smart default is fp64 release mode plus a documented non-goal,
     not a risky mixed-precision chase.
   - **P0-3**: "LSM oracle within bounded error" is too vague for Noah-MP. Require
     state-vector coverage, soil/snow/canopy/hydrology option mapping, energy/water
     closure, restart state, and HFX/LH/TSK/T2/Q2/PBLH diurnal comparisons over at
     least a 24 h land-memory cycle.
   - **S1**: require no host-mediated halo exchange inside the timestep loop, identical
     or predeclared-tolerance parity vs single GPU, transfer audit, and strong/weak
     scaling with compile time separated from run time.

7. **MAJOR - Noah-MP and multi-GPU are underestimated and should not be hidden behind optimistic dates.**

   Noah-MP is not a 0.5-1 week "one scheme" unless the scheme is sharply scoped.
   Full prognostic Noah-MP touches surface state, surface layer, radiation surface
   properties, hourly land-refresh removal, restart, output, and the TOST result. Its
   oracle is messy because WRF land state is path-dependent and diurnal. Treat it as
   the highest overrun risk after native init.

   Multi-GPU domain decomposition is also not a routine extension. `contracts/halo.py`
   may have an MPI-shaped interface, but actual `jax.sharding.Mesh`/`shard_map` across
   a scanned timestep with halo collectives is an architecture/performance project.
   It is valuable for scalability, but it is low forecast-quality value relative to
   P0-6, MYNN, radiation, Thompson, and Noah-MP.

   Concrete amendment: make S1 either a separate v0.2.x scalability milestone or an
   explicitly optional 0.2.0 blocker. Do not hold a forecast-quality paper for S1 if
   the quality chain and TOST are ready first.

8. **MAJOR - The plan omits P1-7 gravity-wave drag and P2-2 namelist fail-fast.**

   The roadmap table includes P1-7 gravity-wave drag and says `gwd_opt=1` is active
   in the Canary namelist. The v0.2 execution plan says "the P1 quality items" but
   does not schedule P1-7. Either add a CPU-WRF sensitivity check and implement GWD,
   or explicitly document that it is non-load-bearing for the supported domains and
   not part of v0.2.0.

   Separately, P2-2 namelist compatibility checking is cheap and important for a
   public WRF-compatible release. A fail-loud checker prevents silent unsupported
   WRF options from becoming false fidelity claims. It should be included before
   v0.2.0 even if most P2 breadth is deferred.

9. **MAJOR - P0-5 should be split or it will be redone.**

   Early output coverage is useful, but full `wrfrst` cannot be final until the
   final prognostic state exists. Noah-MP adds land prognostics; MYNN completeness may
   add/require diagnostics and carry state; Thompson parity wires accumulators; RRTMG
   fidelity changes surface/radiation diagnostics. If P0-5 closes in Wave 1 as "full
   wrfout/wrfrst", its proof will be invalidated by later physics work.

   Concrete amendment: split into:
   - **P0-5a**: wrfout variable/metadata coverage inventory, downstream diagnostic
     completeness, and current-state output proof.
   - **P0-5b**: final restart schema and resume proof after P0-3/P1-3/P1-4/P1-5 are
     closed.

10. **MINOR/PROCESS - 0.1.x-per-item public releases create noise and ambiguity.**

   Incremental public tags are useful only if each tag is coherent and externally
   reproducible. Tagging every item creates three risks: reviewers cite the wrong
   base, users land on transient APIs/proofs, and the release branch becomes a log
   of internal milestones rather than usable software versions.

   Concrete amendment: keep `v0.1.0` immutable, use annotated tags, do not move tags,
   and cut a 0.1.x only for externally meaningful bundles with release notes and a
   regenerated proof table. Suggested bundles:
   - `0.1.1`: MYNN/HFX parity debt + moisture/PBL no-regression + precision-mode
     declaration.
   - `0.1.2`: Thompson precip/water + conservation/non-masking budgets.
   - `0.1.3`: P0-6 real-terrain/boundary dycore closure.
   - `0.1.4`: output/diagnostic completeness.
   - Later tags for nesting/Noah-MP/S1 when those are independently usable.

## Recommended Revised Sequence

1. **Wave -1: CPU-WRF corpus/TOST backfill.** Start now; freeze manifest, case
   masks, exclusions, season labels, and ADR-029 margins. Do not wait for GPU work.
2. **Wave 1: low-risk diagnostics plus known proof debt.** P1-4a MYNN/HFX/moisture
   parity, P1-5 Thompson precip/water, P1-8a declare fp64 release mode, P0-7a budget
   instrumentation, P0-5a wrfout/diagnostic inventory. Serialize final gates through
   manager merge.
3. **Wave 2: P0-6 plus P1-6 merged.** Real-terrain/map-factor/boundary dynamics and
   boundary-order degradation close together with WRF/operator evidence and wind gates.
4. **Wave 3: live nesting.** P0-1a scheduler first, P0-4 Kain-Fritsch, then P0-1b
   live d01->d02->d03 skill. Do not claim live hierarchy skill before parent physics
   is WRF-compatible.
5. **Wave 4: radiation and land.** P1-3 RRTMG fidelity and P0-3 prognostic Noah-MP,
   with surface/radiation/land interfaces frozen before final proof.
6. **Wave 5: final I/O/restart and statistics.** P0-5b final wrfrst, rerun
   conservation/precision/restart/transfer audits, then TOST if corpus is sufficient.
7. **Wave 6: S1 multi-GPU.** Keep as v0.2.0 only if the principal explicitly wants
   scalability to block release; otherwise move to v0.2.x.

## Smartest 0.1.x Publication Subset

If the principal later chooses to publish on a 0.1.x base instead of waiting for
full v0.2.0, the best quality/effort subset is:

1. **P1-4a MYNN/HFX/moisture parity debt** - highest known correctness risk per unit
   effort; directly affects T2/Q2/LH/PBLH/U10/V10 and cleans up the prior GPT G1/G3
   open items.
2. **P0-6 real-terrain/map-factor/boundary closure** - largest wind-skill lever, but
   high risk. Publish after it if it lands; do not block a useful 0.1.x forever if it
   does not.
3. **P1-5 Thompson precip/water parity** - low-to-medium effort, meaningful cloud/
   precip credibility, and feeds conservation.
4. **P0-7 conservation/non-masking budgets** - very high publication credibility;
   not a direct skill fix, but it prevents hidden "finite because guarded" failures.
5. **P1-8 precision declaration** - cheap and important. The likely answer should be
   "fp64 operational mode for fidelity; fp32 only experimental until it earns speed."
6. **P1-3 RRTMG topo/slope/lat-lon fidelity** - good T2/diurnal/PBL value; schedule
   after MYNN if time permits.
7. **P0-5a output/diagnostic completeness** - practical user value, but not a science
   blocker unless downstream products need the fields immediately.

Defer for a 0.1.x paper base unless the principal explicitly prioritizes standalone
operation over paper timing: P0-1 live nesting, P0-4 d01 KF, P0-3 Noah-MP, and S1
multi-GPU. They are important, but effort/risk is much higher and their final proof
depends on the earlier quality chain.

## Handoff

- objective: harsh independent critique of the v0.2.0 execution plan, sequencing,
  parallelization safety, proof gates, scope, and release cadence.
- files changed: `.agent/reviews/2026-05-31-gpt-v020-plan-review.md` only.
- commands run: read-only `git status`, `find`, `rg`, `sed`, and `nl` over the
  requested decision docs, prior review, roadmap/gap inventory, Phase-B ownership
  contracts, and relevant source ownership surfaces.
- proof objects produced: this review file only; no model validation or GPU runs.
- unresolved risks: I did not run code, inspect live CPU-WRF backfill status outside
  existing artifacts, or verify whether the in-flight v0.1.0 campaign has since
  produced newer proof files.
- next decision needed: amend the v0.2.0 plan before dispatch, especially Wave -1
  corpus backfill, P1-4a early MYNN/HFX parity, P0-6/P1-6 merge, P0-5 split, and
  whether S1 multi-GPU truly blocks v0.2.0 or moves to v0.2.x.
