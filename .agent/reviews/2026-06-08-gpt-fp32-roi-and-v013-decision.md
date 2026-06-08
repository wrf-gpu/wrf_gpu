# GPT FP32 Acoustic ROI and v0.13 Decision

Date: 2026-06-08
Worker: GPT-5.5 xhigh
Branch: `worker/gpt/v014-fp32-roi`
Scope: report-only ROI and release sequencing review. No source edits. No GPU jobs.

## outcome

Ship v0.13 after the fp64 TOST n=15 gate if it passes. Do not pause the v0.13 tag for FP32 acoustic work.

FP32 acoustic should be v0.14 P1, starting with the already-defined `mixed_perturb_fp32` ADR/base-state contract and CPU proof lane. The current evidence supports feasibility in principle, but it does not show a v0.13-safe implementation, measured mixed-mode VRAM gain, or coupled validation result.

The current memory blocker for v0.13 was radiation, not acoustic precision. That blocker is now materially addressed in the v0.13 lineage by RRTMG g-point chunking, optics/taumol chunking, and leading-column tiling. FP32 acoustic is useful future work, but it is not needed to make the v0.13 TOST-sized path fit.

## decision table

| Candidate | Bucket | Memory gain / evidence | Recommendation |
|---|---:|---|---|
| RRTMG SW g-point band chunking | Already implemented in v0.13 | `proofs/v013/gpoint_chunk_rrtmg.json`: SW peak down 45-57 percent under JIT; deep-column SW OOM then fits. LW flux-stack path is numerically inert but peak-neutral. | Keep. Part of v0.13 memory base. |
| RRTMG SW/LW optics and taumol construction chunking | Already implemented in v0.13 | `proofs/v013/optics_taumol_chunk.json`: SW 16729.67 -> 1906.42 MiB (-88.6 percent), LW 17853.85 -> 10068.45 MiB (-43.61 percent), bit-identical, deep OOM then fits. | Keep. Part of v0.13 memory base. |
| RRTMG leading-column tiling | Already implemented in v0.13 | `proofs/v013/rrtmg_column_tile.json`: all required SW/LW all-sky and clear-sky cases bit-identical. `proofs/v013/rrtmg_column_tile_vram_suite.json`: LW untiled OOM on 32.11 GiB allocation; LW tiled peak 5374.84 MiB. SW 10033.1 -> 1619.54 MiB. | Keep. This removes the reason to hold v0.13 for more memory code before TOST. |
| Nested allocator / segmentation controls from v0.12 memory wave | Already implemented in v0.13 | Existing nested 9/3/1 km and 9/3 km proofs fit after radiation chunking; no current evidence of the old BFC fragmentation failure recurring. | Keep. No new action. |
| Two-way feedback duplicate total/base reconstruction removal | Already implemented in v0.13 | `proofs/v013/twoway_vram.json`: bit-identical; feedback transient -9.088 MiB on measured edge. | Keep. Correct but small. |
| Moisture advection duplicate velocity reuse | Safe to implement in v0.13 | Static evidence from memory refresh: likely 0.45-0.65 GiB duplicate velocity build avoided on large domains, default-inert when `moist_adv_opt=0`. | Safe only if manager wants a small post-TOST cleanup. Do not delay TOST/tag for it. |
| WDM6 `slmsk` full-column broadcast cleanup | Safe to implement in v0.13 | Small, opt-in scheme-specific memory cleanup; not on the default TOST path. | Safe but not release-critical. Prefer v0.14 cleanup unless already in a touched file. |
| RRTMG column tile-size tuning and full-forecast runtime profiling | Safe to implement in v0.13 | Correctness is proven for tiling; runtime cost of `dynamic_update_slice` carries is not yet measured in full forecast context. | Do not block v0.13. Tune after tag or in v0.14 with profiler artifacts. |
| Mixed perturbation-authoritative FP32 acoustic | v0.14 P1 | Feasible in principle per `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`, but current code still uses total-minus-perturbation base recovery in prep/finish and hard fp64 islands. No mixed-mode numerical or GPU proof exists. | Start v0.14 P1 after v0.13 tag: R0/R1 contract and base plumbing, then R2/R3 CPU proofs. |
| Acoustic scan carry split / evolving-only carry | v0.14 P1, co-designed with FP32 | Prior memory refresh estimates about 1.56 GiB recoverable, but prior split attempts were reverted and the acoustic scan is high-risk dycore code. | Revisit only after R0/R1 freezes the precision/base-state contract. |
| MYNN BouLac dense `(C,K,K)` source arrays | v0.14 later | Potentially large if materialized, but not yet confirmed as a TOST/full-step peak and requires PBL oracle/coupled validation. | Measure first; then tile/rewrite with MYNN-specific gates. |
| Post-physics sparse/donated delta merge | v0.14 later | Static estimate: about 1.3 GiB outputs and up to about 2.6 GiB if deltas materialize. Coupling-wide donation and correctness risk. | Defer to a coupled-runtime memory sprint with transfer audit. |
| Whole-domain non-radiation column physics tiling | v0.14 later | Potential 1-3+ GiB per active scheme, but each scheme needs exact-output proof and memory measurement. | Use RRTMG as pattern, one scheme at a time. |
| Moisture species sequentialization / limiter workspace reduction | v0.14 later | Opt-in path today; needs scalar-advection, positivity, and conservation gates. | Defer until moisture advection has stronger validation. |
| PBL/surface bottom-only prep and duplicate diagnostics | v0.14 later | Estimated 0.3-0.8 GiB, but overlaps PBL/surface coupling and known fp64-sensitive fields. | Defer until PBL/surface oracle and coupled gates are in hand. |
| State schema alias reduction for `p/p_total/p_perturbation`, `ph/*`, `mu/*` | v0.14 later | At 641x321x50, removing duplicate total/alias acoustic leaves could save about 0.31 GiB, but changes ABI, restart, I/O, and boundary contracts. | ADR-required. Do not mix into v0.13 or the first FP32 sprint. |
| Small dycore masks/pad cleanups such as procedural `dry_cqw` | Not worth standalone | Individually small and still requires dycore-kernel validation. Some related Wave-A cleanups are already present. | Only do when adjacent to an acoustic rewrite. |
| Global fp32 dtype flip | Not worth doing | Explicitly rejected by the roadmap and by prior fp32 analysis. It reintroduces the known acoustic cancellation failure and threatens qke/PBL stability. | Kill. Do not dispatch. |

## FP32 acoustic ROI

Shape math for the largest current target-grid example, 641x321x50:

| Item | fp64 size | fp32 demotion saves |
|---|---:|---:|
| One mass 3-D field `(nz,ny,nx)` | 78.49 MiB | 39.25 MiB |
| One vertical-face 3-D field `(nz+1,ny,nx)` | 80.06 MiB | 40.03 MiB |
| One surface 2-D field `(ny,nx)` | 1.57 MiB | 0.78 MiB |
| Demote only `p'`, `ph'`, `mu'` | n/a | about 80 MiB |
| Demote `p'`, `ph'`, `mu'`, and `w` | n/a | about 120 MiB |
| Demote current acoustic-locked resident `State` leaves (`w`, `p/*`, `ph/*`, `mu/*`) | n/a | about 280 MiB |
| Remove duplicate total/alias leaves for `p/ph/mu` | n/a | about 320 MiB |
| Demote operational acoustic scratch/save family | n/a | about 400 MiB |
| Demote the large leaves of the full acoustic scan carry | about 3.19 GiB | up to about 1.60 GiB |

Best-case peak gain on the 641x321x50 target is roughly 1.5-2.3 GiB if R2/R6 eventually demote most acoustic scan work and also reduce resident duplicate acoustic state. That is valuable, but it is not comparable to the radiation fixes that removed tens of GiB of transient pressure.

Worst-case useful gain is zero for R0/R1, because contract/base plumbing alone should be default-inert and may even add explicit resident base arrays until duplicates are removed. A first stable mixed mode that keeps many fp64 islands and demotes only perturbation state is only about 0.08-0.12 GiB at the target grid. On the TOST d02 grid (159x66x44), even demoting the large acoustic core leaves would save only about 74 MiB peak, so FP32 acoustic is not a v0.13 TOST fit requirement.

Validation cost is high because this is dycore formulation work, not a dtype knob:

- R0/R1: ADR, precision-mode contract, cache-key isolation, default-off behavior, explicit base-state plumbing, source audit, and fp64-default bit identity.
- R2/R3: perturbation-authoritative state, dtype trace, scalar cancellation probes, one-column recurrence, and WRF savepoint/analytic gates.
- R4-R6: idealized dry cases, boundary/nesting/restart gates, current-module integration, staged GPU runs, transfer audit, and VRAM/profiler artifacts.
- R7/R8: one-at-a-time fp64 island demotion, mixed-mode TOST/AEMET/CPU-WRF validation with predeclared single-precision margins, and documentation claim audit.

That is a v0.14 lane, not a v0.13 tag hold.

## v0.13 pause threshold for FP32

It would be rational to pause v0.13 after fp64 TOST for FP32 only if all of the following already exist before the tag decision:

1. A default-off `mixed_perturb_fp32` implementation with compile-cache isolation and no production CLI behavior change.
2. Default fp64 bit identity over focused acoustic prep/finish tests and a one-step operational carry test.
3. A source and dtype audit proving mixed mode never recovers base or perturbation values by fp32 absolute total-minus-base subtraction inside the timestep loop.
4. Scalar cancellation and one-column acoustic proofs showing the formulation fixes the historical fp32 cancellation failure without tolerance changes, guards, or masking clamps.
5. An explicit list of retained fp64 islands, including base/reference fields, boundary leaves, pressure/EOS refresh, `calc_p_rho` bracket, terrain PGF, and implicit-w pieces.
6. A tiny GPU smoke with transfer audit and measured peak-memory gain large enough to matter for a release objective, not just a small TOST grid.
7. Manager review that the added default-off surface does not invalidate the completed fp64 TOST release candidate.

Current evidence does not satisfy those conditions. It is feasibility analysis plus a roadmap, not a proof object for merging FP32 into v0.13.

## manager recommendation

1. Keep the current v0.13 line focused on fp64 TOST. With RRTMG column tiling now merged and GPU VRAM evidence recorded, memory should not block restarting or completing TOST on the current release-candidate lineage.
2. If fp64 TOST n=15 passes, tag and ship v0.13. Do not wait for FP32 acoustic.
3. Open v0.14 P1 with a narrow R0/R1 contract: default-off `mixed_perturb_fp32`, explicit base-state plumbing, and fp64-default bit identity.
4. Require R3 CPU cancellation/one-column proofs before any real GPU mixed-mode forecast.
5. Treat measured FP32 memory/performance claims as invalid until R6 staged GPU runs produce profiler, transfer-audit, and VRAM artifacts.

## files changed

- `.agent/reviews/2026-06-08-gpt-fp32-roi-and-v013-decision.md`

No source files, proof scripts, configs, skills, or stable memory files were changed.

## commands run

- `sed -n` reads of `PROJECT_CONSTITUTION.md`, `AGENTS.md`, `.agent/sprints/2026-06-08-v014-fp32-acoustic-derisk/sprint-contract.md`, `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`, the three required GPT review files, and `proofs/v013/rrtmg_column_tile_vram_suite.json`.
- `sed -n` reads of local skills: `designing-gpu-state`, `validating-physics`, `profiling-nvidia-gpu`, and `reporting-to-human`.
- `git status --short --branch`
- `git log --oneline --decorate --max-count=16`
- `rg --files .agent/skills`
- `find proofs/v013 proofs/v014 .agent/reviews -maxdepth 2 -type f | sort` for orientation; `proofs/v014` was absent.
- `rg -n ... .agent/reviews .agent/decisions proofs/v013 proofs/v014` for candidate orientation; it was noisy and also hit the absent `proofs/v014` path, so the report relies on the targeted proof/report reads above.
- `jq` summaries for `rrtmg_column_tile.json`, `rrtmg_column_tile_vram_suite.json`, `target_1km_vram_probe.json`, and `twoway_vram.json`.
- `sed -n` inspections of `target_1km_vram_probe.json`, `twoway_vram.json`, `gpoint_chunk_rrtmg.json`, and `optics_taumol_chunk.json`.
- `nl -ba` source inspections of precision, state, operational carry, acoustic core, small-step prep/finish, and operational acoustic assembly.
- A one-off shape-math Python calculation for FP32 memory bounds. It did not read or write repo files.
- `git diff --check`

## proof objects produced

- This review report only.

No numerical proof, GPU run, profiler artifact, or new `proofs/v014/` object was produced.

## unresolved risks

- Full target-grid end-to-end memory after RRTMG column tiling is inferred from the column-tile VRAM suite, not proven by a 641x321x50 full forecast artifact in this lane.
- FP32 acoustic memory gain is shape-math and code-structure ROI, not a measured mixed-mode result.
- Runtime impact of RRTMG column tiling in a full forecast still needs profiling, but it should not block v0.13 correctness release if TOST passes.
- MYNN BouLac and non-radiation physics peaks remain static risks until an empirical memory map catches them in a full operational step.

## next decision needed

Manager should restart or continue fp64 TOST on the current v0.13 release-candidate lineage. If TOST passes, ship v0.13 and dispatch v0.14 P1 FP32 acoustic R0/R1 afterward.

GPT FP32 ROI DONE
