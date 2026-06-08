# V0.14 Memory-Fix Roadmap

Date: 2026-06-08
Owner: GPT-5.5 xhigh manager-side analyst
Scope: integrate memory research before the next long validation campaign.

## Decision

The release-critical memory blocker was RRTMG full-column radiation memory. It
is now fixed in the v0.13/v0.14 lineage by bit-identical RRTMG band/optics
chunking plus leading-column tiling.

No remaining unimplemented memory fix should block grid-parity work or the first
post-grid-parity long validation launch. The next long validation branch still
needs a short memory preflight on its exact code lineage because several
remaining risks are static estimates, not measured full-step peaks.

Priority order remains:

1. Grid-cell parity and root-cause fixes.
2. FP32 acoustic / mixed precision, after grid divergence no longer confounds
   dycore diagnosis.
3. Remaining memory work.
4. Powered TOST and longer validation as final gates.

## Evidence Used

- `.agent/reviews/2026-06-08-gpt-memory-refresh.md`
- `.agent/memory/pending/2026-06-08-v013-memory-efficiency.md`
- `.claude/worktrees/gpt-mem-map/.agent/reviews/2026-06-08-gpt-analytic-memory-map.md`
- `.codex/worktrees/v013-memory-refresh/.agent/reviews/2026-06-08-gpt-memory-refresh.md`
- `.agent/reviews/2026-06-08-opus-1km-target-vram-measurement.md`
- `.agent/reviews/2026-06-08-gpt-rrtmg-column-tile.md`
- `.agent/reviews/2026-06-08-gpt-fp32-roi-and-v013-decision.md`
- `.agent/reviews/2026-06-08-gpt-v014-fp32-status-freeze.md`
- `proofs/v013/target_1km_vram_probe.json`
- `proofs/v013/rrtmg_column_tile.json`
- `proofs/v013/rrtmg_column_tile_vram_suite.json`
- `proofs/v013/gpoint_chunk_rrtmg.json`
- `proofs/v013/optics_taumol_chunk.json`
- `proofs/v013/twoway_vram.json`
- `proofs/v0120/nested_oom_fix.json`
- `proofs/v014/fp32_acoustic_probes.json`
- `PROJECT_PLAN.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`

## Long-Validation Gate

Before a long GPU validation campaign runs on a new branch:

1. Confirm the branch contains RRTMG leading-column tiling and the v0.12 nested
   allocator/segmentation controls.
2. Run a short exact-branch memory preflight that exercises the selected
   long-run configuration's radiation, output diagnostics, nesting, GWD, and
   feedback settings.
3. Record peak VRAM, allocator mode, output count, and finiteness proof under
   `proofs/v014/`.
4. Do not merge a broad memory rewrite after the long run without rerunning the
   relevant short gate and any scored validation it invalidates.

## Bit-Identical Layout And Scheduling Fixes

These fixes have no intended physics change. They should prove exact equality
against the previous layout unless the row explicitly says it is only a
measurement or tuning task.

| Rank | Issue | Status | Must fix before long validation? | Expected VRAM gain | Risk | Likely source files | Proof gate |
|---:|---|---|---|---|---|---|---|
| 1 | RRTMG leading-column tiling for SW/LW | Fixed in v0.13 lineage | Yes, already fixed. Required for large 1 km memory headroom. | At `ncol=65536,nlev=48`: LW untiled OOM on 32.11 GiB alloc, tiled peak 5374.84 MiB. SW 10033.10 -> 1619.54 MiB. | Medium runtime risk; correctness proof is strong. | `src/gpuwrf/physics/rrtmg_sw.py`, `src/gpuwrf/physics/rrtmg_lw.py` | `proofs/v013/rrtmg_column_tile.json` all required cases bit-identical; `proofs/v013/rrtmg_column_tile_vram_suite.json` GPU peak suite. Add full-forecast peak profile before a headline memory claim. |
| 2 | RRTMG optics/taumol construction chunking | Fixed | Yes, already fixed. | SW 16729.67 -> 1906.42 MiB (-88.6 percent). LW 17853.85 -> 10068.45 MiB (-43.61 percent). Deep-column OOM then fits. | Low correctness risk; bit identity proven. | `src/gpuwrf/physics/rrtmg_sw.py`, `src/gpuwrf/physics/rrtmg_lw.py` | `proofs/v013/optics_taumol_chunk.json`: SW/LW max rel 0.0, exact bit identity. |
| 3 | RRTMG SW g-point flux-stack chunking | Fixed | Yes, already fixed. | SW peak down 45-57 percent under JIT. Deep unchunked SW OOM then fits. LW path is peak-neutral because upstream taumol/fracs/Planck arrays dominate. | Low correctness risk; honest LW-neutral result. | `src/gpuwrf/physics/rrtmg_sw.py`, `src/gpuwrf/physics/rrtmg_lw.py` | `proofs/v013/gpoint_chunk_rrtmg.json`: exact result equality, energy closure, VRAM rows. |
| 4 | Nested allocator and output-interval segmentation | Fixed in v0.12 and retained | Yes for nested long runs, already fixed. | Exact failing L3 case: BFC default h=2 peak 32019 MiB with creep; platform allocator re-exec peak 15806 MiB, flat, about 16.8 GiB headroom. | Medium performance cost, about 1.33x slower nested path in proof. | `src/gpuwrf/cli.py`, `src/gpuwrf/integration/nested_pipeline.py`, `src/gpuwrf/runtime/domain_tree.py` | `proofs/v0120/nested_oom_fix.json`; repeat exact-branch nested memory preflight after major runtime changes. |
| 5 | Two-way feedback duplicate total/base reconstruction removal | Fixed | No. Keep it. | 9.088 MiB per-feedback-event transient cut, about 11.5 percent of measured parent-field-scaled feedback transient. | Low. | `src/gpuwrf/coupling/boundary_feedback.py`, `src/gpuwrf/runtime/domain_tree.py` | `proofs/v013/twoway_vram.json`: all leaves bit-identical, max abs diff 0.0. |
| 6 | Moisture advection duplicate transport velocity build | Pending | No. Safe cleanup only after grid-parity work unless it is in an adjacent touched file. | Static estimate 0.45-0.65 GiB on 641x321x50 when `moist_adv_opt != 0`; default-inert when off. | Low/medium. Default-off should be byte-identical, but active moisture path must preserve WRF scalar cadence. | `src/gpuwrf/runtime/operational_mode.py`, possibly `src/gpuwrf/dynamics/flux_advection.py` | Default `moist_adv_opt=0` bit identity; active `moist_adv_opt=1/2` conservation, positivity, and `proofs/v013/moisture_advection_wiring.json` rerun; no new transfers. |
| 7 | WDM6 `slmsk` full-column broadcast | Pending | No. | About 0.077 GiB at 641x321x50. | Low, scheme-specific and opt-in. | `src/gpuwrf/coupling/scan_adapters.py` | WDM6 oracle/smoke with exact outputs or predeclared tolerance; default suite unchanged. |
| 8 | Whole-domain column tiling for non-radiation physics | Pending | No, measure first. | 1-3+ GiB per active scheme in static source estimates; larger for two-moment schemes and kernel internals. | Medium/high due per-scheme coupling and output shape details. | `src/gpuwrf/coupling/scan_adapters.py`, `src/gpuwrf/coupling/physics_couplers.py`, `src/gpuwrf/physics/thompson_column.py`, `src/gpuwrf/physics/microphysics_*.py`, PBL/cumulus files per scheme | One scheme at a time. CPU exact-output tile-vs-untiled proof, then short GPU VRAM suite, then real-case smoke for that scheme. |
| 9 | Post-physics non-dry sparse/donated merge | Pending | No. | Static estimate 1.33 GiB output leaves; up to 2.64 GiB if deltas materialize separately. | Medium/high because it changes coupling liveness and donation behavior. | `src/gpuwrf/runtime/operational_mode.py`, `src/gpuwrf/coupling/physics_couplers.py`, `src/gpuwrf/coupling/scan_adapters.py` | Exact default output proof over selected schemes; donation/transfer audit; short coupled GPU run with peak VRAM. |
| 10 | PBL/surface bottom-only prep and duplicate diagnostics reuse | Pending | No. Keep out of first long validation unless it is needed for a correctness fix. | Static estimate 0.3-0.8 GiB. | Medium/high because selected-sfclay diagnostics also affect PBL semantics. | `src/gpuwrf/coupling/scan_adapters.py`, `src/gpuwrf/coupling/physics_couplers.py`, surface-layer adapters, MYNN/MRF/PBL adapters | Surface/PBL WRF oracle for selected pairs; exact default-config no-regression; coupled real-case smoke. |
| 11 | Moisture scalar sequentialization and limiter workspace reduction | Pending | No. | 0.46 GiB for six plain outputs; 1-3+ GiB possible in limited/FCT path. | Medium because limiter order, positivity, and conservation must stay WRF-faithful. | `src/gpuwrf/dynamics/flux_advection.py`, `src/gpuwrf/runtime/operational_mode.py` | Per-species WRF-transcription parity, total water conservation, positivity/monotonicity, active real-case smoke. |
| 12 | Legacy pad-based face-pair helper cleanup | Pending | No. Only do adjacent to acoustic work. | 0.15-0.30 GiB transient per use where present. | Low/medium but dycore-adjacent. | `src/gpuwrf/dynamics/acoustic_wrf.py`, related acoustic helper modules | Focused dycore unit tests plus warm-bubble/Straka CPU gates. |
| 13 | `dry_cqw` full face mask | Pending | No. Too small standalone. | About 0.078 GiB. | Medium because it touches implicit-w masking. | `src/gpuwrf/dynamics/core/advance_w.py` | Exact implicit-w focused gate, flat/terrain rest, idealized dry cases. |
| 14 | State total/perturbation/base alias reduction | Pending, ADR-required | No. Do not mix with validation. | About 0.16-0.32 GiB for one `p/ph/mu` family at 641x321x50; more only through broader schema changes. | High ABI, restart, I/O, boundary, and diagnostics risk. | `src/gpuwrf/contracts/state.py`, init, boundary, restart, wrfout, validation comparators | ADR first; restart roundtrip, wrfout compatibility, boundary and savepoint parity. |
| 15 | RRTMG tile-size tuning and full-forecast profiling | Pending measurement, not a correctness fix | Do not block launch, but run before performance claims. | Unknown runtime/peak tradeoff. Current tile `16384` is proven. | Low correctness risk; performance unknown. | `src/gpuwrf/physics/rrtmg_sw.py`, `src/gpuwrf/physics/rrtmg_lw.py`, run scripts/profilers | Full-forecast profiler artifact, peak VRAM, transfer audit. |

## Dycore Or Physics Semantic Changes

These rows may reduce memory but change numerical formulation, precision policy,
or coupling semantics. They must not be bundled with the bit-identical fixes
above.

| Rank | Issue | Status | Must fix before long validation? | Expected VRAM gain | Risk | Likely source files | Proof gate |
|---:|---|---|---|---|---|---|---|
| S1 | Mixed perturbation-authoritative FP32 acoustic | v0.14 P1, not implementation-ready | No. Must wait until grid divergence root cause is clearer. | Early useful gain can be only 0.08-0.12 GiB. Candidate arithmetic: 754.07 MiB core, 1191.63 MiB core+prep/carry; realistic best-case acoustic peak 1.5-2.3 GiB. | Very high dycore precision/formulation risk. | `src/gpuwrf/dynamics/core/small_step_prep.py`, `small_step_finish.py`, `acoustic.py`, `advance_w.py`, `calc_p_rho.py`, `runtime/operational_mode.py`, boundary/restart/init paths | R0-R8 from `V0140-FP32-ACOUSTIC-ROADMAP.md`: ADR, explicit base-state plumbing, CPU cancellation and one-column probes, WRF savepoint/idealized gates, transfer audit, measured VRAM/profiler, mixed-mode validation. |
| S2 | Acoustic scan carry split / evolving-only carry | Pending; co-design with FP32 or a separate dycore memory sprint | No. Do after grid-parity root cause, or only under a frozen dycore contract. | Static recoverable estimate about 1.56 GiB. | High because prior split attempts were reverted and it touches acoustic substep liveness. | `src/gpuwrf/dynamics/core/acoustic.py`, `src/gpuwrf/runtime/operational_mode.py`, small-step prep/finish state assembly | Default fp64 bit identity, acoustic savepoint parity, warm-bubble/Straka/terrain-rest gates, short GPU smoke with transfer audit. |
| S3 | MYNN BouLac dense `(C,K,K)` rewrite or tiling | Pending; measure first | No, unless empirical map proves it is the next OOM. | One dense array is 3.83 GiB at 641x321x50; plausible 10-25+ GiB if several live. | High PBL semantics risk. | `src/gpuwrf/physics/mynn_pbl.py`, `src/gpuwrf/coupling/physics_couplers.py`, `src/gpuwrf/coupling/scan_adapters.py` | First empirical HLO/RSS proof that arrays materialize. Then MYNN WRF oracle, column/tile exactness, coupled PBL real-case smoke. |
| S4 | Selected surface-layer diagnostics carry into PBL adapters | Pending correctness issue with memory side benefit | No as memory work. Do as a PBL/surface correctness sprint if grid attribution points there. | 0.3-0.8 GiB possible via avoiding duplicate prep; memory is secondary. | High because it changes which surface-layer diagnostics PBL receives. | `src/gpuwrf/coupling/scan_adapters.py`, `src/gpuwrf/coupling/physics_dispatch.py`, `src/gpuwrf/physics/sfclay_*.py`, PBL adapters | WRF surface-driver to pbl-driver contract proof for selected pairs; fail-close unproven pairs; real-case PBL/surface gate. |
| S5 | RRTMG radiation working-set fp32 | Not approved | No. Do not use as a memory shortcut before validation. | Could roughly halve selected radiation work arrays, but no approved precision proof. | High radiation physics and validation risk. | `src/gpuwrf/physics/rrtmg_sw.py`, `src/gpuwrf/physics/rrtmg_lw.py`, radiation couplers | ADR/precision policy update, pristine-WRF radiation oracle, clear/all-sky gates, full forecast radiation validation. |
| S6 | Real multi-GPU column sharding | Future route; fake-mesh only today | No for single-GPU v0.14 validation. | Theoretical ncol-sharding can divide column transients by GPU count, but real throughput and collectives are unmeasured. | High systems risk. | Future sharding/halo substrate, stencil and physics partitioning code | Real multi-GPU hardware proof, partition-invariance, halo transfer audit, throughput and per-watt artifacts. |

## Near-Optimal Or Not Worth Standalone

- Resident `State + OperationalCarry` is about 2.04 GiB at 641x321x50 in the
  target probe. It is not the binding single-GPU issue after RRTMG tiling.
- Base/metric/boundary state is small relative to radiation and physics
  transients.
- NoahMP, Noah classic, and slab carries are mostly 2-D or shallow soil/snow
  columns and are below 0.1 GiB at the target geometry.
- Surface flux output leaves are small; the waste is full-column input prep.
- `rthraten` and `pm1` are necessary WRF-cadence/acoustic memory fields.
- Compile/runtime hygiene and fake-mesh sharding proofs are valuable but are not
  direct single-GPU VRAM fixes.

## Sprint Queue

1. **V014-MEM-0: exact-branch memory preflight.** After grid-parity branch
   stabilizes, run a short memory proof for the intended long validation config.
   Produce `proofs/v014/exact_branch_memory_preflight.json`.
2. **V014-MEM-1: empirical memory map.** Measure MYNN BouLac, non-radiation
   column physics, post-physics merge, and moisture limiter liveness on the
   exact branch. No semantic fixes.
3. **V014-MEM-2: small bit-identical cleanup.** If useful after MEM-1, implement
   moisture velocity reuse and/or WDM6 `slmsk` cleanup with exact default gates.
4. **V014-MEM-3: non-radiation column tiling pilot.** Pick one measured offender,
   preferably a physics scheme with strong oracle coverage, and reuse the RRTMG
   tiling pattern.
5. **V014-FP32-R0/R1/R3: mixed acoustic de-risk.** Keep separate from memory
   cleanup. No GPU mixed forecast until CPU/source gates pass.
6. **V014-MEM-SEM: MYNN/PBL or acoustic semantic memory work.** Dispatch only
   after grid-cell attribution identifies the relevant operator or MEM-1 proves
   it is the next OOM.

## Proof Policy

- Layout fixes require exact bit identity unless a predeclared physical
  tolerance is explicitly justified before the run.
- Semantic changes require WRF fixture, analytic oracle, conservation, or
  ensemble evidence in addition to memory proof.
- GPU memory claims require peak-VRAM artifact and transfer audit.
- No long validation run should be used as the first proof of a new memory
  rewrite.
