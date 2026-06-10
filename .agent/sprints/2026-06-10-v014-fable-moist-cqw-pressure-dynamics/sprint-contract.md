# Sprint Contract: V0.14 Fable Moist-CQW Pressure Dynamics Closure

Date: 2026-06-10
Owner: manager
Assignee: Fable/Mythos in tmux `0:1` after `/compact`
Status: OPEN

## Objective

Close or formally bound the remaining 3D pressure-state dynamics blocker after
the accepted `PSFC` diagnostic fix. Endpoint: implement a WRF-faithful moist
`cqw` / moist `pg_buoy_w` path in the operational acoustic w-equation and prove
that it materially closes the dry-balanced `P/PH/W` residual without regressing
performance or stability, or return an exact WRF-anchored impossibility/bounds
proof that the lane cannot be closed in v0.14 scope.

This is one whole task, not a sequence of micro-prompts.

## Current State

- PSFC diagnostic fixed and pushed in `a08553dc`.
- GPU h1-h4 validation pushed in `a23f71cc`; proof:
  `proofs/v014/psfc_moist_pressure_gpu_h4_validation.{md,json}`.
- Fixed PSFC h1/h4 RMSE is `57.823/35.487 Pa`; old vapor-light floor is gone.
- Remaining field parity blocker: 3D pressure-state dynamics.
- Fable proof:
  `proofs/v014/psfc_moist_pressure_state_closure.{py,json,md}` and
  `.agent/reviews/2026-06-10-v014-fable-psfc-moist-pressure-closure.md`.
- Key finding: GPU `P+PB(k0)` tracks its own DRY hydrostatic column, while CPU
  WRF tracks the MOIST hydrostatic column. Current operational code uses
  `dry_cqw` / `pg_buoy_w_dry`.

## Non-Goals

- Do not change `PSFC` diagnostic semantics again unless your proof shows the
  accepted fix is wrong.
- No tolerance-only or comparator-only fix.
- No output-only masking of `P/PH/W`.
- No broad FP32 or memory-optimization work in this sprint.
- No long GPU validation campaign; at most short targeted GPU gates after
  manager-visible proof. If you need GPU, first make the CPU proof strong and
  keep the run short.

## File Ownership

Likely production files:

- `src/gpuwrf/dynamics/core/advance_w.py`
- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/contracts/state.py` only if a state/carry field needs explicit
  interface support.

Proof/test files:

- `proofs/v014/moist_cqw_pressure_dynamics_closure.*`
- `tests/test_v014_moist_cqw_pressure_dynamics.py` or focused existing tests.
- `.agent/reviews/2026-06-10-v014-fable-moist-cqw-pressure-dynamics.md`

Do not edit unrelated physics, radiation, memory, or release docs except a
compact report.

## Required Reads

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/managing-sprints/SKILL.md`
- this contract
- `.agent/reviews/2026-06-10-v014-fable-psfc-moist-pressure-closure.md`
- `proofs/v014/psfc_moist_pressure_state_closure.md`
- `proofs/v014/psfc_moist_pressure_gpu_h4_validation.md`
- relevant code matched by:
  `rg "dry_cqw|pg_buoy_w_dry|calc_cq|advance_w_wrf|cqw" -n src/gpuwrf`
- WRF source anchor tree:
  `/home/enric/src/wrf_pristine/WRF`

## Acceptance Criteria

For a production fix:

1. WRF source anchoring for `calc_cq`, `pg_buoy_w`, `cqw/cq1/cq2`, and where
   those fields enter the W acoustic solve.
2. A CPU-only proof that identifies the exact JAX/WRF boundary and quantifies
   pre/post behavior for:
   - `P+PB(k0)` vs own dry/moist hydrostatic column,
   - `P`, `PH`, `W` or `rw_tend` local residual,
   - downstream h1/h4 field comparator impact if feasible.
3. If source is changed, implement the smallest WRF-faithful path:
   - compute/stage real moist `cqw` from qtot in the same convention WRF uses,
   - use the moist `pg_buoy_w` terms instead of the dry specialization where
     WRF does,
   - thread through existing acoustic state/scan without large materialized
     transients or host/device transfers.
4. Preserve GPU-native performance assumptions:
   - no host/device transfer inside timestep loops,
   - no new full-run CPU callback,
   - no unnecessary duplicated large 3D arrays beyond what WRF-faithful dynamics
     requires.
5. Focused tests/proofs pass. Manager will run/accept short GPU h1/h4 or
   equivalent only after CPU proof is strong.

Target signal:

- materially reduce the dry-vs-moist pressure-state residual, especially
  `P+PB(k0)` relative to WRF/CPU,
- do not regress the already-fixed `PSFC`,
- do not destabilize the short Canary GPU h1/h4 run,
- keep `U/V/T/QVAPOR` in the same or better envelope.

If a production fix is not safe:

- return `FORMALLY_BOUNDED`, with exact proof why this cannot be closed in
  v0.14 without unacceptable dycore revalidation risk, and the smallest
  v0.15+ implementation plan.

## Validation Commands

CPU defaults:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/moist_cqw_pressure_dynamics_closure.py
python -m json.tool proofs/v014/moist_cqw_pressure_dynamics_closure.json
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest <focused-tests> -q
python -m compileall -q src tests proofs
git diff --check
```

Short GPU gate, only after CPU proof and if needed:

- coordinate with manager; single GPU job only.
- prefer the same Canary h1/h4 harness used in
  `proofs/v014/psfc_moist_pressure_gpu_h4_validation.md`.

## Proof Object

- `proofs/v014/moist_cqw_pressure_dynamics_closure.py`
- `proofs/v014/moist_cqw_pressure_dynamics_closure.json`
- `proofs/v014/moist_cqw_pressure_dynamics_closure.md`
- `.agent/reviews/2026-06-10-v014-fable-moist-cqw-pressure-dynamics.md`

## Risks

This is a prognostic dycore change. It may affect `P/PH/W`, vertical acoustic
stability, and all field-parity gates. Do not overfit one diagnostic. Prefer
WRF savepoint/oracle evidence before touching production, and keep the patch
small enough to review.

## Handoff Requirements

Report:

- objective
- decision: FIXED / FORMALLY_BOUNDED / BLOCKED
- files changed
- WRF source anchors
- commands run
- proof objects produced
- CPU proof summary and any GPU proof summary
- performance/memory implications
- unresolved risks
- exact next decision

Completion marker to manager pane `0:2`:

```bash
tmux send-keys -t 0:2 'FABLE MOIST_CQW_PRESSURE_DYNAMICS DONE - see .agent/reviews/2026-06-10-v014-fable-moist-cqw-pressure-dynamics.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
