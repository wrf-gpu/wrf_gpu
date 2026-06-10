# Sprint Contract: V0.14 Fable PSFC Moist Pressure-State Closure

Date: 2026-06-10
Owner: manager
Assignee: Fable high in tmux
Status: CLOSED_ACCEPTED_BY_MANAGER

## Objective

Close the fixed-Canary `PSFC` vapor-light pressure-state residual as a real
production/proof task. Endpoint: implement a WRF-faithful pressure-state fix and
prove it, or produce a precise WRF-anchored proof that the remaining residual is
not fixable within v0.14 scope. Treat this as one whole task, not a sequence of
micro-prompts.

## Non-Goals

- No comparator-only tolerance relaxation.
- No output-only `PSFC` clamp unless it is proven to be the exact WRF diagnostic
  path and does not hide inconsistent `P/PH/MU` state.
- No GPU use unless the manager explicitly releases the GPU lock; the fixed
  Canary 72h run is active.
- No broad FP32 or unrelated memory work in this sprint.

## File Ownership

Production files in scope if a real fix is proven:

- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/io/wrfout_writer.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/contracts/state.py`

Proof/report files in scope:

- `proofs/v014/psfc_moist_pressure_state_closure.*`
- `.agent/reviews/2026-06-10-v014-fable-psfc-moist-pressure-closure.md`

Do not edit other production files without first recording why they are in the
actual pressure-state path.

## Inputs

- GPT diagnosis:
  `.agent/reviews/2026-06-10-v014-gpt-psfc-vapor-light-analysis.md`
- LBC root-cause proof:
  `proofs/v014/lbc_cadence_root_cause.md`
- Fixed Canary run root:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_lbcfix_20260610T151455Z`
- h1/h4 comparators in that run root.
- CPU truth:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- WRF source reference under:
  `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF`

## Acceptance Criteria

Required for a production fix:

1. Source-level WRF anchoring for the relevant `PSFC`/`p8w`/moist pressure path.
2. CPU-only h1/h4 budget proof comparing CPU and GPU outputs:
   `PSFC`, `MU`, `MUB`, dry column, vapor column proxy, and
   `P/PH`-extrapolated `PSFC`.
3. Offline ablation showing the expected post-fix `PSFC` RMSE/bias before any
   production patch, and explaining the expected effect on `P/PH/MU`.
4. Production patch if and only if it is WRF-faithful and not a cosmetic clamp.
5. Focused tests/proofs pass on CPU. Manager will schedule a short GPU h1/h4
   validation after review if the active long run must be interrupted or
   relaunched.

Target signal:

- Remove the flat `~210 Pa` vapor-load floor from `PSFC`.
- Do not regress `MU`, `P`, `PH`, `U/V/T`, or `QVAPOR` relative to fixed LBC h1.
- Preserve GPU-resident timestep semantics; no host/device transfer inside
  timestep loops.

## Validation Commands

Use CPU-only commands unless manager grants GPU:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python <proof-script>
python -m json.tool proofs/v014/psfc_moist_pressure_state_closure.json
python -m py_compile $(find src tests proofs -name '*.py' -not -path './cache/*')
git diff --check
```

If production source changes are made, also run the narrow focused tests that
cover the edited modules and record exact commands in the report.

## Performance Metrics

No GPU performance claim is required in this sprint. Any production fix must be
constant-memory per column and must not introduce timestep-loop CPU transfers.

## Proof Object

- `proofs/v014/psfc_moist_pressure_state_closure.md`
- `proofs/v014/psfc_moist_pressure_state_closure.json`
- `.agent/reviews/2026-06-10-v014-fable-psfc-moist-pressure-closure.md`

## Risks

The exact CPU WRF diagnostic path may include a smaller `~14 Pa` formula gap
that is separate from the large vapor-load miss. Do not overfit the proof to
the simple extrapolation if WRF source says otherwise.

## Handoff Requirements

Report:

- objective
- files changed
- commands run
- proof objects produced
- exact WRF source path/function references used
- decision: FIXED / FORMALLY_BOUNDED / BLOCKED
- unresolved risks
- next decision needed

Completion marker to manager pane `0:2`:

`FABLE PSFC_MOIST_PRESSURE_CLOSURE DONE - see .agent/reviews/2026-06-10-v014-fable-psfc-moist-pressure-closure.md`

Use delayed repeated Enters:

```bash
tmux send-keys -t 0:2 'FABLE PSFC_MOIST_PRESSURE_CLOSURE DONE - see .agent/reviews/2026-06-10-v014-fable-psfc-moist-pressure-closure.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
