# GPT Memory Refresh - v0.13 Fix Selection

Date: 2026-06-08

Worker: GPT-5.5 xhigh

Worktree branch inspected from: `worker/gpt/v013-memory-refresh`

Important checkout caveat: this worktree is currently at `7fd92fd2`, the stale
`worker/opus/v013-t3-pbl`/`worker/gpt/v013-fp32-refresh` tip. The current
manager base for this reconciliation is `worker/gpt/v013-close-manager` /
`worker/opus/v0120-integration` at `237aceb5`.

## objective

Reconcile the prior static memory map against the current v0.13 manager base and
the named parallel branches, then classify remaining solvable memory issues for
v0.13/TOST sequencing. I did not edit model code and did not use the GPU.

## files changed

- `.agent/reviews/2026-06-08-gpt-memory-refresh.md`

No source files were changed.

## commands run

- `sed -n` inspections of `PROJECT_CONSTITUTION.md`, `AGENTS.md`, the dispatch
  prompt, and the three repo-local skills.
- `sed -n` inspection of the prior report:
  `.claude/worktrees/gpt-mem-map/.agent/reviews/2026-06-08-gpt-analytic-memory-map.md`.
- `git status --short --branch`
- `git branch --list 'worker/*v013*' 'worker/gpt/v013-close-manager'`
- `git log -1 --format=...` and `git merge-base` checks for the named branches.
- `git merge-base --is-ancestor <branch> 237aceb5` for branch freshness.
- `git log --graph --decorate --oneline --max-count=80 --all --date-order --simplify-by-decoration --branches='worker/*v013*' --branches='worker/opus/v0120-integration'`
- `git ls-tree -r --name-only 237aceb5 proofs/v013 .agent/reviews .agent/decisions`
- `git show 237aceb5:<proof/report>` for:
  `target_1km_vram_probe.json`,
  `2026-06-08-opus-1km-target-vram-measurement.md`,
  `optics_taumol_chunk.json`,
  `gpoint_chunk_rrtmg.json`,
  `nested_oom_fix.json`,
  `gwd_nested_24h_gate.json`,
  `twoway_vram.json`,
  `twoway_gwd_9_3km_24h_gate.json`,
  `tost_wrfbdy_fix.md/.json`,
  `tost_rc2_fix.md`,
  `moisture_advection_wiring.json`,
  `pd_moisture.json`.
- `git show worker/gpt/v013-fp32-acoustic:.agent/reviews/2026-06-08-gpt-fp32-acoustic-feasibility.md`
- `git grep` / `git show ... | nl -ba` source inspections for RRTMG column
  tiling, MYNN Boulac dense arrays, duplicate moisture velocity coupling,
  acoustic carry shape, and `dry_cqw`.

## proof objects produced

- This review report only.
- No `proofs/v013/` object was produced because no code was edited and no GPU or
  validation command was run.

## branch/evidence reconciliation

- `worker/gpt/v013-mem-map @7ce31a6b`: based on `237aceb5`; one post-base
  commit adding the prior analytic memory report. Valid as analysis only.
- `worker/gpt/v013-fp32-acoustic @43341c1b`: based on `237aceb5`; analysis only.
  Its own recommendation is v0.14 for production mixed/fp32 acoustic.
- `worker/gpt/v013-fp32-refresh @7fd92fd2`: stale pre-manager branch. Treating it
  as fresh would delete many v0.13 proofs/reports relative to `237aceb5`; ignore.
- `worker/opus/v013-2way-vram @8de39fd9`: ancestor of `237aceb5`; already merged.
  Proof `twoway_vram.json` shows exact bit identity and a small 9.1 MiB feedback
  transient reduction.
- `worker/opus/v013-compile-perf @92fc12f8`: ancestor of `237aceb5`; already
  merged. Compile/runtime hygiene, not a material VRAM fix.
- `worker/opus/v013-skill-closure @25ab8d3e`: ancestor of `237aceb5`; already
  merged. Default-off radiation tendency cadence; not a memory fix.
- `worker/opus/v013-t3-microphysics`, `v013-t3-wdm5`, `v013-t3-cumulus`,
  `v013-t3-radiation`, `v013-t3-gsfc-lw`, and `v013-t3-surface-lsm`: all
  ancestors of `237aceb5`; already merged. Mostly scheme coverage/oracles. They
  can add opt-in physics memory pressure but do not close the current target-grid
  memory blocker.
- `worker/opus/v013-t3-pbl @7fd92fd2`: ancestor/stale, as the prompt warned.
- `worker/opus/v013-rrtmg-coltile` and `worker/opus/v013-empirical-mem-map` both
  point at `237aceb5` locally. I found no committed post-237 RRTMG column-tiling
  fix or empirical memory-map artifact on those refs.

## updated memory ranking

1. **RRTMG full-column radiation transient, especially LW** - still the binding
   v0.13 memory issue if the 641x321x50 1 km target is in scope. Current
   `target_1km_vram_probe.json` projects the target at 89.3 GiB d03 replay /
   89.9 GiB live nest even after band/taumol chunking. One `(ncol,K,16,16)`
   fp64 field is 19.6 GiB at the target. Existing band chunking is proven
   bit-identical but does not tile columns.
2. **MYNN Boulac `(C,K,K)` dense source arrays** - largest non-radiation static
   risk if materialized. Not yet empirically confirmed as a TOST/full-step peak.
3. **Full `AcousticCoreState` scan carry** - about 1.56 GiB recoverable in the
   prior map, but it is broad dycore work and prior split attempts were reverted.
4. **Post-physics non-dry full-grid delta/merge** - about 1.3 GiB outputs and up
   to about 2.6 GiB if deltas materialize. Coupling-wide correctness risk.
5. **Whole-domain column-batched non-radiation physics** - Thompson/WDM/Morrison/
   MYNN/cumulus can cost 1-3+ GiB per active scheme. Column tiling is plausible
   but must be per-scheme validated.
6. **Moisture advection duplicate velocity build and scalar tuple overlap** -
   roughly 0.45-0.65 GiB for the duplicate velocity build plus 0.46+ GiB for six
   moisture outputs. Current moisture advection is opt-in (`moist_adv_opt != 0`)
   and default-inert.
7. **PBL/surface full-column prep for bottom-only surface layer and duplicate
   diagnostics** - about 0.3-0.8 GiB, but changes PBL/surface coupling shape.
8. **Small dycore/schema/mask cleanups** - `dry_cqw` full face mask, old pad-based
   helpers, WDM6 `slmsk` broadcast, and redundant State total/perturbation/base
   leaves. Individually small or ABI-sensitive.

Already resolved in the current manager base:

- RRTMG SW g-point band chunking: merged, bit-identical, SW peak -45% to -57%.
- RRTMG optics/taumol band chunking: merged, bit-identical, SW -88.6%, LW -43.6%
  at the measured anchor; deep-column OOM now fits.
- Nested allocator/segmentation from v0.12: current nested path avoids the old
  long-run BFC fragmentation failure.
- Two-way feedback dedup: merged, bit-identical, small feedback transient cut.

## MUST-FIX-BEFORE-TOST list

1. **Conditional: RRTMG column tiling, if it remains a v0.13 code change.**
   It touches the radiation path used by TOST. The existing 2-domain TOST grids
   are small enough that current evidence says they should fit without it, but a
   later broad radiation-memory merge would change the exact code lineage of the
   scored campaign. Therefore TOST should not be run on pre-column-tiling code
   if the manager still intends to merge RRTMG column tiling into v0.13 before
   tag. Land it with bit-identity proof first, or explicitly defer it.

No other remaining memory item is a clear TOST fit/run blocker. The TOST
`tost_wrfbdy_fix` proof shows the scored L2 path is `max_dom=2`, d01 93x59x44
and d02 159x66x44, live-nested through `wrfbdy_d01`. Existing nested 9/3/1 km
and 9/3 km gates already fit after the merged RRTMG band/taumol chunking.

## SAFE-NOW list

- **Already merged: two-way feedback VRAM dedup.** Exact bit identity,
  `max_abs_diff_over_all_leaves=0.0`, small memory reduction, no TOST validity
  risk.
- **Already merged: RRTMG g-point and optics/taumol band chunking.** Exact
  bit-identity proofs exist. These are part of the current base and are not
  pending.
- **Potential cleanup, not implemented here: pass the existing flux-advection
  `vel` into `_moisture_coupled_tendencies` instead of rebuilding it.** This is
  high-confidence bit-identical for `moist_adv_opt != 0` and default-inert for
  TOST if `moist_adv_opt=0`, but it is not the only safe candidate and it touches
  `runtime/operational_mode.py`; I did not implement it under this report-only
  dispatch.
- **Potential cleanup, not implemented here: WDM6 `slmsk` full-column broadcast.**
  Small, opt-in-scheme-specific, and not TOST-critical.

Because there is not exactly one unique, high-confidence, small, disjoint
SAFE-NOW fix, I made no code change.

## V0.14 list

- **FP32/mixed acoustic production path.** Excluded by this dispatch. The GPT
  feasibility report says it is feasible but should be v0.14 production work.
- **Acoustic carry split / evolving-only scan carry.** Broad dycore work; must
  re-prove acoustic correctness and stability.
- **MYNN Boulac scan/tile rewrite.** Potentially very large memory win, but it
  changes PBL internals and needs WRF/PBL oracle and coupled validation.
- **Post-physics sparse/donated merge.** Coupling-wide behavior and donation
  changes need GPU/correctness validation.
- **Whole-domain non-radiation column physics tiling.** Good pattern after RRTMG,
  but each active scheme needs exact-output proof and memory measurement.
- **Moisture species sequentialization / limiter workspace reduction.** Opt-in
  path today; needs scalar-advection correctness coverage beyond static analysis.
- **PBL/surface bottom-only prep and selected surface-diagnostics threading.**
  This overlaps known PBL/sfclay correctness contracts; not a memory-only edit.
- **State schema alias reduction (`p`, `p_total`, `p_perturbation`, etc.).**
  ABI/contract change requiring ADR and restart/I/O implications.
- **`dry_cqw` procedural mask and small acoustic mask/pad cleanup.** Small memory
  win but still dycore-kernel validation.

## whether TOST can start now, with reasoning

**Not unconditionally.**

The current TOST L2 pipeline should fit/run on the already-merged memory base:
the scored grid is d02 159x66x44, far smaller than the 641x321x50 target that
drives the 89 GiB projection, and the current base already contains the proven
RRTMG band/taumol chunking plus nested allocator controls. I found no additional
non-radiation memory fix that must land for TOST fit.

However, if v0.13 will still include the unlanded RRTMG column-tiling fix, TOST
should wait for that fix and its bit-identity proof. Running TOST now and then
merging a broad radiation-memory rewrite afterward would make the TOST outputs
come from a different radiation implementation than the release candidate. If
the manager explicitly defers RRTMG column tiling to v0.14, then memory no longer
blocks restarting TOST on `237aceb5` plus the already-merged fixes.

## unresolved risks

- No empirical memory-map artifact was found beyond the target 1 km analytic/
  anchor-scaled probe. MYNN Boulac and non-radiation physics peaks remain static
  risks until measured.
- The RRTMG column-tiling branch names currently point at `237aceb5`; I found no
  committed fix to review. There could be uncommitted work in another local
  worktree outside the refs inspected here.
- This worktree is stale relative to `237aceb5`, so this report was produced by
  inspecting refs with `git show`, not by building/running the checked-out tree.
- I did not consume the GPU, per instruction.

## next decision needed, if any

Decide one of these before restarting TOST:

1. **v0.13 includes RRTMG column tiling**: dispatch/finish that sprint first,
   require exact bit-identity and memory proof, then restart TOST on the fixed
   branch.
2. **RRTMG column tiling moves to v0.14**: restart TOST now on the current
   `237aceb5` lineage because no remaining memory fix is TOST-critical.
