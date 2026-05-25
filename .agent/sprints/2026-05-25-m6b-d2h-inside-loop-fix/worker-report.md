# Worker Report - M6b D2H Inside-Loop Fix

## Objective

Localize and remove warmed inter-kernel D2H transfers from the operational timestep loop, then verify warmed `d2h_inter_kernel == 0`.

## Result

BLOCKED. The warmed residual D2H emitters were localized, but both remaining call sites are in `src/gpuwrf/runtime/operational_mode.py`, which this worker was explicitly told not to modify because an RK1 fix is editing it in parallel. No model-code fix was applied and acceptance was not met.

## Files Changed

- `scripts/m6b_d2h_warmed_recapture.py`: added bisection-only CLI controls for `run_boundary`, `run_physics`, and `acoustic_substeps`.
- `.agent/sprints/2026-05-25-m6b-d2h-inside-loop-fix/proof_bisection_d2h_emitter.txt`: bisection proof and localization.
- `.agent/decisions/ADR-027-d2h-invariant-clarification-PROPOSED.md`: promoted ADR-027 and recorded the operational-mode blocked finding.

No changes were made to `src/gpuwrf/runtime/operational_mode.py`.

## Commands Run

- `python scripts/m6b_d2h_warmed_recapture.py --help`
- `taskset -c 0-3 nsys profile ... python scripts/m6b_d2h_warmed_recapture.py --profile-steps 5 --disable-boundary`
- `python scripts/m6b_d2h_warmed_recapture.py --parse-rep .agent/sprints/2026-05-25-m6b-d2h-inside-loop-fix/proof_bisect_no_boundary.nsys-rep`
- `taskset -c 0-3 nsys profile ... python scripts/m6b_d2h_warmed_recapture.py --profile-steps 5 --acoustic-substeps 1`
- `python scripts/m6b_d2h_warmed_recapture.py --parse-rep .agent/sprints/2026-05-25-m6b-d2h-inside-loop-fix/proof_bisect_acoustic1.nsys-rep`
- `taskset -c 0-3 nsys profile ... python scripts/m6b_d2h_warmed_recapture.py --profile-steps 5 --disable-physics`
- `python scripts/m6b_d2h_warmed_recapture.py --parse-rep .agent/sprints/2026-05-25-m6b-d2h-inside-loop-fix/proof_bisect_no_physics.nsys-rep`

All profiler runs used `taskset -c 0-3`, `OMP_NUM_THREADS=4`, and `GPUWRF_CUDA_PROFILER_RANGE=1`.

## Proof Objects Produced

- `proof_bisect_no_boundary.transfer_summary.json`: `d2h_inter_kernel=20`; boundary disabled does not change the residual.
- `proof_bisect_acoustic1.transfer_summary.json`: `d2h_inter_kernel=20`; acoustic substeps 2 -> 1 does not change the residual.
- `proof_bisect_no_physics.transfer_summary.json`: `d2h_inter_kernel=15`; physics disabled removes only the 1-byte predicate transfer.
- `proof_bisection_d2h_emitter.txt`: decision summary with commands and localized line references.

## Localization

- `src/gpuwrf/runtime/operational_mode.py:353-361`: dynamic `jax.lax.switch` over RK stage index inside the RK scan. This accounts for the 3 x 4 B D2H per outer timestep.
- `src/gpuwrf/runtime/operational_mode.py:374-380`: dynamic radiation-cadence `jax.lax.cond` predicate. This accounts for the 1 x 1 B D2H per outer timestep.

The original top suspects were ruled out:

- `boundary_apply.py`: disabling boundary leaves `d2h_inter_kernel=20`.
- `acoustic_wrf.py`: reducing acoustic substeps leaves `d2h_inter_kernel=20`.

## Unresolved Risks

- Warmed inter-kernel D2H is still nonzero in this worktree.
- `tests/test_m6b_d2h_inside_loop_fix.py` and `proof_d2h_inside_loop_zero.json` were not created because that would incorrectly imply the acceptance gate passed.
- Full regression was not run because the mandatory acceptance fix was blocked before source changes.

## Next Decision Needed

The worker editing `operational_mode.py` needs to fold this localization into its RK1/static-control-flow change: replace the dynamic RK `lax.switch` with statically sequenced RK stages and replace the dynamic radiation predicate with static forecast segmentation or another manager-approved resident schedule. Then rerun the warmed Nsight acceptance and produce `proof_d2h_inside_loop_zero.json`.
