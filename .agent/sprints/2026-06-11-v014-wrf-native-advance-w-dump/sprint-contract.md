# Sprint Contract: v0.14 WRF-Native Advance-W Dump

Date: 2026-06-11
Owner: manager
Assignee: Fable high, tmux
Status: READY

## Objective

Build the shortest rigorous WRF-native oracle for the remaining
Switzerland/Gotthard h36 blocker: the single RK1 acoustic substep
`WRF call 21601 -> 21602`, focused inside `advance_w` and the immediate
`calc_p_rho` follow-up.

Endpoint: either prove and fix a local WRF-faithful source defect, or produce a
named, WRF-anchored first mismatching term/state boundary inside
`advance_w_wrf()` / immediate `calc_p_rho_step` that is stronger than
`proofs/v014/switzerland_advance_w_term_split.json`.

## Non-Goals

- Do not run a Switzerland 72h GPU gate.
- Do not work on Canary, TOST, FP32, memory, or performance audits.
- Do not modify `/home/enric/src/canairy_waves`.
- Do not use `ask-hermes`, Telegram, or human notification commands.
- Do not edit stable rules, skills, constitutions, roadmap goal, or unrelated
  release docs.
- Do not accept a JAX-vs-JAX self-compare as closure.

## File Ownership

Owned in this worker worktree:

- `.agent/reviews/2026-06-11-v014-fable-wrf-native-advance-w-dump.md`
- `proofs/v014/wrf_native_advance_w_dump.py`
- `proofs/v014/wrf_native_advance_w_dump.json`
- `proofs/v014/wrf_native_advance_w_dump.md`
- `proofs/v014/wrf_native_advance_w_dump_wrf_patch.diff`
- optional narrowly named helpers under `proofs/v014/` for this sprint only

Production source may be changed only if a local WRF-faithful defect is proven
and the patch is minimal. Likely source scope if needed:

- `src/gpuwrf/dynamics/core/advance_w.py`
- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/dynamics/core/acoustic_wrf.py`
- `src/gpuwrf/dynamics/core/calc_p_rho.py`
- `src/gpuwrf/runtime/operational_mode.py`

WRF source instrumentation must be disposable. Do not modify
`/home/enric/src/wrf_pristine/WRF` in place; it is not a clean scratch tree. If
native instrumentation is required, create a disposable copy/worktree under
`/mnt/data/wrf_gpu_validation/` or the worker scratch area, record the exact
patch as `proofs/v014/wrf_native_advance_w_dump_wrf_patch.diff`, and commit
only the manifest/checksums/comparison summaries, not large WRF output.

## Inputs

Read first:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/skills/building-wrf-oracles/SKILL.md`
- `.agent/skills/validating-physics/SKILL.md`
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
- `.agent/reviews/2026-06-11-v014-gpt-advance-w-term-split.md`
- `proofs/v014/switzerland_advance_w_term_split.py`
- `proofs/v014/switzerland_advance_w_term_split.json`
- `proofs/v014/switzerland_advance_w_phi_discriminator.py`
- `proofs/v014/switzerland_advance_w_phi_discriminator.json`
- `proofs/v014/switzerland_acoustic_substep_blocker.py`
- `proofs/v014/switzerland_acoustic_substep_blocker.json`
- `proofs/v014/switzerland_acoustic_continuation.py`
- `proofs/v014/switzerland_acoustic_continuation.json`
- `proofs/v014/gpt_stage3_wrapper_verifier.md`

Relevant WRF/JAX source surfaces:

- `src/gpuwrf/dynamics/core/advance_w.py`
- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/dynamics/core/acoustic_wrf.py`
- `src/gpuwrf/dynamics/core/calc_p_rho.py`
- WRF `dyn_em/module_small_step_em.F`
- WRF `dyn_em/solve_em.F`

Useful existing instrumentation patterns:

- `proofs/v014/full_pre_rk_savepoint_hook.py`
- `proofs/v014/full_pre_rk_savepoint_hook_wrf_patch.diff`
- `proofs/v014/source_save_boundary_hook.py`
- `proofs/v014/source_save_boundary_hook_wrf_patch.diff`
- `external/wrf_savepoint_patch/README.md`

Current key evidence:

- First remaining h36 `p/ph` error is created in one RK1 acoustic substep:
  `WRF call 21601 -> 21602`.
- Baseline first-stage interior RMSE against WRF call 21602:
  `p=1.1261975184533854`, `ph=0.4352639584631756`,
  `mu=0.02089603774551649`.
- Forcing WRF-call-21602 `mu/muts/muave` into `advance_w` improves `p` by
  only `0.112%` and `ph` by `0.0018%`; post-`advance_mu_t` mass inputs are not
  the primary creator.
- Surface `w`, moist coefficient choice alone, `calc_p_rho` denominator, and
  `smdiv` are not primary.
- Term clues from proof-local variants:
  - zeroing/recomputing `rw_tend` improves `p` by about `32%` but worsens
    `ph` by about `6.84%`;
  - zeroing `ph_tend` improves `ph` by `25.84%` and `p` by `9.07%`.

## Required Work

1. Decide the fastest rigorous method before coding. Prefer reusing existing
   HPG/call dump infrastructure if it can expose the needed arrays; otherwise
   build the minimal disposable WRF instrumentation.
2. Freeze the oracle schema before comparing. Record for each array:
   variable name, WRF source expression, units where known, shape, staggering,
   domain id, timestep/call id, tile/halo policy, precision, and file checksum
   if stored outside git.
3. Produce WRF-native values for call `21601 -> 21602` covering, at minimum:
   - post-`advance_mu_t` inputs: `mu_2`, `muts`, `muave`, `ww`, `t_2`,
     `ph_tend`, `rw_tend`;
   - `advance_w` RHS before and after vertical phi advection;
   - implicit coefficients `a`, `alpha`, `gamma`;
   - Thomas forward RHS / solved `w`;
   - finished `ph`;
   - immediate post-`calc_p_rho` `p`, `al`, `alt`.
4. Compare the WRF-native oracle against the existing Python/JAX harness at the
   same h36 boundary. The result must name the earliest mismatching term/state,
   not merely report final `p/ph` RMSE.
5. If the mismatch identifies a local WRF-faithful source defect, implement the
   smallest production fix and run the focused proof gate. Avoid clamps,
   tolerance relaxation, masking, artificial damping, or host/device transfer
   inside timestep loops.
6. If exact native WRF dumping cannot be completed in this sprint, return a
   `METHOD_LIMIT` only after producing a concrete patch/manifest plan and
   showing why the missing external artifact blocks the comparison.

## Acceptance Criteria

Successful close requires at least one of:

- `FIXED`: source fix implemented and the h36/call-21602 short gate materially
  improves both `p` and `ph` without worsening mass/state invariants.
- `LOCAL_FIX_PROPOSED`: exact source defect and minimal patch identified, but
  manager-controlled GPU/time gate or merge coordination is needed.
- `NARROWED_NO_FIX`: WRF-native oracle comparison identifies the first
  mismatching named `advance_w`/`calc_p_rho` term or state boundary, with the
  next fix target stated.
- `ORACLE_BUILT_NO_FIX`: native dump and comparator are valid, but all exposed
  requested terms match closely enough that the blocker must move to a named
  unexposed term/input; report the missing term explicitly.
- `METHOD_LIMIT`: blocked by an external artifact/build limitation after a
  concrete disposable instrumentation patch and manifest plan are produced.

## Validation Commands

Run, at minimum:

```bash
python -m py_compile proofs/v014/wrf_native_advance_w_dump.py
python -m json.tool proofs/v014/wrf_native_advance_w_dump.json >/tmp/wrf_native_advance_w_dump.validated.json
git diff --check
```

Also run the generated comparison/proof command and record its exact invocation
and exit status in the report. If production source changes are made, run the
shortest relevant focused pytest/proof plus the h36/call-21602 gate.

## Performance Metrics

No performance claim is required. If source changes are made, note any expected
runtime or memory impact and whether they introduce any new host/device transfer
inside timestep loops.

## Proof Object

Commit proof metadata and comparison summaries under `proofs/v014/`. Large WRF
native dumps must stay outside git with paths, sizes, and SHA256 checksums
recorded in `proofs/v014/wrf_native_advance_w_dump.json`.

## Risks

- WRF pristine tree is dirty; using it in place can corrupt evidence.
- Native WRF arrays are Fortran-ordered and staggered; wrong index conversion
  can create false mismatches.
- Tile overlap and halo policy must be explicit.
- The existing Python harness may need proof-local hooks to expose internal
  terms; keep those hooks out of production unless a real source fix is proven.

## Handoff Requirements

Write `.agent/reviews/2026-06-11-v014-fable-wrf-native-advance-w-dump.md` with:

- verdict;
- objective;
- files changed;
- commands run;
- proof objects produced;
- WRF dump schema and data locations/checksums;
- earliest mismatch table;
- source diff summary if any;
- unresolved risks;
- next decision needed, if any.

If source changes are made, commit them on the worker branch. End stdout exactly:

`FABLE WRF_NATIVE_ADVANCE_W_DUMP DONE - see .agent/reviews/2026-06-11-v014-fable-wrf-native-advance-w-dump.md`
