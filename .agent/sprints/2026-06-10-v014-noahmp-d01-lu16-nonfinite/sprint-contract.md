# Sprint Contract: v0.14 Noah-MP d01 LU16 Nonfinite Closure

Date: 2026-06-10
Manager: Codex
Assignee: Fable high in isolated worktree
Branch: `worker/fable/v014-noahmp-d01-lu16`
Worktree: `/home/enric/src/wrf_gpu2/.claude/worktrees/fable-noahmp-d01-lu16`

## Objective

Close the current v0.14 post-fix GPU preflight blocker: after the nested Noah-MP
activation fix, the exact-branch 1h Canary L2 nested preflight fits in memory and
writes both domains, but exits `rc=1` because d01 final state/output diagnostics
are not finite. Find and fix the WRF-faithful root cause, or produce an exact
proof that the root cause is outside this sprint and identify the next blocker.

## Evidence Already Known

Preflight root:
`/mnt/data/wrf_gpu_validation/v014_noahmp_l2_preflight_20260610T192315Z`

Key files:
- `preflight.log`
- `preflight.rc`
- `gpu_command.stdout.log`
- `gpu_command.stderr.log`
- `proofs/nested_pipeline_run.json`
- `nested_1h_out/wrfout_d01_2026-05-01_19:00:00`
- `nested_1h_out/wrfout_d02_2026-05-01_19:00:00`
- resource CSVs under `resources/`

Observed facts:
- `preflight.rc = 1`
- No OOM marker.
- Peak total VRAM about `10042 MiB`; peak compute-app VRAM about `9161 MiB`.
- Both `wrfout_d01` and `wrfout_d02` were written.
- d02 is finite.
- d01 `final_state_finite=false`.
- d01 has 51 nonfinite cells in these output fields:
  `T2`, `UST`, `HFX`, `LH`, `TSK`, `TH2`, `LWUPB`, `LWUPT`, `OLR`.
- The nonfinite cells are land cells, not water:
  `LANDMASK=1`, `XLAND=1`, `LAKEMASK=0`, `SEAICE=0`.
- All sampled bad cells have `LU_INDEX=16`, `IVGTYP=16`, `ISLTYP=1`.
- Initial `wrfinput_d01` fields are finite at those cells.
- The failure is therefore likely in the Noah-MP land/diagnostic path for
  barren/sparsely vegetated land category 16 or a category-16 parameter/phenology
  edge case, not in memory tiling, LBC cadence, or the d02 live-nest dycore.

Relevant source:
- `src/gpuwrf/integration/nested_pipeline.py`
- `src/gpuwrf/coupling/noahmp_surface_hook.py`
- `src/gpuwrf/physics/noahmp_coupler.py`
- `src/gpuwrf/io/noahmp_land_init.py`
- `src/gpuwrf/physics/noahmp/*.py`
- `src/gpuwrf/runtime/operational_mode.py`

Recent source fix already merged:
- `c2310c5b` activates/seeds Noah-MP in standalone nested pipeline.
- `c6800bfa` adds the h1-h4 Noah-MP land gate scorer.
- `91cfbb2b` updates roadmap/gate status.

## Required Method

1. Verify branch/worktree base with `git log -1`.
2. Reproduce or inspect the existing failure without rerunning GPU first.
3. Localize the first nonfinite source as narrowly as practical:
   Noah-MP forcing, phenology/static parameters, energy/radiation, water/snow,
   surface blend, or output overlay.
4. Implement the smallest WRF-faithful production fix if local and safe.
   Do not paper over the problem by masking land category 16 to water, replacing
   NaNs with arbitrary clamps, or bypassing Noah-MP diagnostics.
5. Add a focused CPU proof/test that catches the failing LU16 category case.
6. If GPU is free and the fix is ready, run only a bounded proof through
   `scripts/run_gpu_lowprio.sh` with resource logging; otherwise leave the exact
   manager command to rerun preflight and H4.

## Acceptance Criteria

- The root cause is documented in a compact proof/report.
- Any source fix is committed on `worker/fable/v014-noahmp-d01-lu16`.
- Focused CPU tests/proofs pass and prove no nonfinite category-16 land
  diagnostics.
- If GPU proof is run, the same exact-branch 1h preflight is green or the next
  blocker is explicitly different and localized.
- No broad roadmap/doc churn except the worker report and proof artifacts.

## Deliverables

- Worker report:
  `.agent/sprints/2026-06-10-v014-noahmp-d01-lu16-nonfinite/worker-report.md`
- Proof artifact:
  `proofs/v014/noahmp_d01_lu16_nonfinite_closure.md`
  and optional JSON/script if useful.
- Commit hash on worker branch, or explicit no-fix blocker report.

## Completion Marker

When done, type into manager tmux window `0:2` with repeated delayed enters:

```bash
tmux send-keys -t 0:2 'FABLE NOAHMP_D01_LU16_NONFINITE DONE - see .agent/sprints/2026-06-10-v014-noahmp-d01-lu16-nonfinite/worker-report.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
