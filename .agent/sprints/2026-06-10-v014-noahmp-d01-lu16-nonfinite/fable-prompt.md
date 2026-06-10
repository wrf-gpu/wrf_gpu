You are Fable high, implementation/debug worker for wrf_gpu2 v0.14.

Read first:
- `/home/enric/src/wrf_gpu2/PROJECT_CONSTITUTION.md`
- `/home/enric/src/wrf_gpu2/AGENTS.md`
- `/home/enric/src/wrf_gpu2/.agent/sprints/2026-06-10-v014-noahmp-d01-lu16-nonfinite/sprint-contract.md`

You are in an isolated worktree:
`/home/enric/src/wrf_gpu2/.claude/worktrees/fable-noahmp-d01-lu16`
on branch `worker/fable/v014-noahmp-d01-lu16`.
First verify `git log -1`. Base should be current v0.14 manager tip
`91cfbb2b` or newer.

Task endpoint:
Close the current v0.14 release blocker: the post-NoahMP-fix exact-branch
Canary L2 nested 1h preflight writes both domains and fits in memory, but exits
`rc=1` because d01 final/output diagnostics are nonfinite. Find and fix the
WRF-faithful root cause, or produce an exact compact proof that the remaining
blocker is outside this sprint and identify the next concrete blocker.

Failure artifact:
`/mnt/data/wrf_gpu_validation/v014_noahmp_l2_preflight_20260610T192315Z`

Inspect:
- `preflight.log`, `preflight.rc`
- `gpu_command.stdout.log`, `gpu_command.stderr.log`
- `proofs/nested_pipeline_run.json`
- `nested_1h_out/wrfout_d01_2026-05-01_19:00:00`
- `nested_1h_out/wrfout_d02_2026-05-01_19:00:00`
- resource CSVs under `resources/`

Known facts:
- `preflight.rc = 1`
- No OOM; peak total VRAM about 10.0 GiB, compute-app peak about 9.2 GiB.
- Both wrfouts written.
- d02 is finite.
- d01 `final_state_finite=false`.
- d01 has exactly 51 nonfinite cells in output fields:
  `T2`, `UST`, `HFX`, `LH`, `TSK`, `TH2`, `LWUPB`, `LWUPT`, `OLR`.
- Bad cells are land:
  `LANDMASK=1`, `XLAND=1`, `LAKEMASK=0`, `SEAICE=0`.
- Sampled bad cells have `LU_INDEX=16`, `IVGTYP=16`, `ISLTYP=1`.
- Initial wrfinput fields are finite there.
- This is likely a Noah-MP category-16 barren/sparsely-vegetated edge in
  forcing/static/phenology/energy/water/surface blend/output overlay, not memory
  tiling, LBC cadence, or d02 live-nest dycore.

Relevant source:
- `src/gpuwrf/integration/nested_pipeline.py`
- `src/gpuwrf/coupling/noahmp_surface_hook.py`
- `src/gpuwrf/physics/noahmp_coupler.py`
- `src/gpuwrf/io/noahmp_land_init.py`
- `src/gpuwrf/physics/noahmp/*.py`
- `src/gpuwrf/runtime/operational_mode.py`

Constraints:
- Do not use Hermes.
- Keep terminal output compact; detailed evidence belongs in proof/report files.
- Do not mask category 16 to water.
- Do not hide NaNs with arbitrary output-only clamps.
- Preserve GPU performance concept: no per-step host transfers or broad
  materialization.
- Prefer CPU/local proof first. Use GPU only for a bounded confirmation if the
  repo GPU lock is free and the fix is ready. Use `scripts/run_gpu_lowprio.sh`
  for any GPU run.
- Commit source/proof/report changes on your branch if you implement a fix.

Acceptance:
- Root cause clearly stated.
- Minimal WRF-faithful fix if local.
- Focused CPU proof/test catches LU16 nonfinite and passes.
- If GPU proof is run, preflight green or next blocker clearly localized.

Deliver:
- `.agent/sprints/2026-06-10-v014-noahmp-d01-lu16-nonfinite/worker-report.md`
- `proofs/v014/noahmp_d01_lu16_nonfinite_closure.md`
- optional JSON/script proof if useful.

When done, send:
`FABLE NOAHMP_D01_LU16_NONFINITE DONE - see .agent/sprints/2026-06-10-v014-noahmp-d01-lu16-nonfinite/worker-report.md`
to tmux `0:2` with repeated delayed Enter presses.
