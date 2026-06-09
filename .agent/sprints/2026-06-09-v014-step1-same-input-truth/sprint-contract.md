# Sprint Contract: V0.14 Step-1 Same-Input Truth And Comparator

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Produce the missing full-domain CPU-WRF d02 step-1
`post_after_all_rk_steps_pre_halo` truth surface, then execute the first strict
same-input WRF-vs-JAX comparison for one complete d02 step if technically
possible.

This is a proof/tooling sprint, not a production source-fix sprint.

## Trigger Evidence

- `proofs/v014/same_input_contract_builder.json`
- `proofs/v014/same_input_contract_builder.py`
- `.agent/sprints/2026-06-09-v014-same-input-contract-builder/manager-closeout.md`
- `proofs/v014/wrf_post_rk_refresh_localization_patch.diff`
- `src/gpuwrf/runtime/operational_mode.py::_rk_scan_step_with_pre_halo_capture`

## Critical Method Rule

Do **not** compare WRF step-1 post-RK/pre-halo truth against the JAX initial
state. That is a false comparison.

The only accepted strict comparison is:

1. Build the same initial d02 `OperationalCarry`/`OperationalNamelist` from
   `wrfinput_d01`/`wrfinput_d02` and the same parent-boundary package.
2. Run one JAX CPU step through `_physics_step_forcing` and
   `_rk_scan_step_with_pre_halo_capture` with step-1 lead/cadence semantics.
3. Compare `result.pre_halo_state` against CPU-WRF d02 step-1
   `post_after_all_rk_steps_pre_halo` truth for every schema field.

If this cannot be done, emit the exact missing JAX wrapper/carry/physics/WRF
field blocker. Do not emit a weak initial-vs-post-step table.

## Non-Goals

- No production `src/gpuwrf/**` edits.
- No production dycore/runtime/physics fix.
- No GPU, no TOST, no Switzerland validation, no FP32 source work, no memory
  source work.
- No Hermes or Telegram.
- No JAX-vs-JAX self-compare, no station-score proxy, no one-cell proof, and no
  mixed-source carry where WRF leaves are spliced into a JAX-produced carry.

## Inputs

- `/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3`
- `/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3/wrfinput_d01`
- `/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3/wrfinput_d02`
- `/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3/wrfbdy_d01`
- `/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3/namelist.input`
- `/mnt/data/wrf_gpu2/v014_source_save_boundary/WRF` preferred disposable-copy
  source because it is already built and contains related hook scaffolding
- `/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF` fallback built source
- `/home/enric/src/wrf_pristine/WRF` fallback built source
- `proofs/v014/same_input_contract_builder.py`
- `proofs/v014/wrf_post_rk_refresh_localization_patch.diff`
- `proofs/v014/same_state_momentum_mass.py` for the existing
  `_physics_step_forcing` + `_rk_scan_step_with_pre_halo_capture` call pattern

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_same_input_truth/**`

## Write Scope

Repository files:

- `proofs/v014/step1_same_input_truth.py`
- `proofs/v014/step1_same_input_truth.json`
- `proofs/v014/step1_same_input_truth.md`
- `proofs/v014/step1_same_input_truth_wrf_patch.diff`
- optional targeted updates to `proofs/v014/same_input_contract_builder.py`
- optional regenerated `proofs/v014/same_input_contract_builder.json`
- optional regenerated `proofs/v014/same_input_contract_builder.md`
- `.agent/reviews/2026-06-09-v014-step1-same-input-truth.md`

External scratch files:

- disposable WRF tree under
  `/mnt/data/wrf_gpu2/v014_step1_same_input_truth/WRF`
- WRF run directory under
  `/mnt/data/wrf_gpu2/v014_step1_same_input_truth/run`
- accepted truth npz:
  `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`
- optional raw WRF dump files under
  `/mnt/data/wrf_gpu2/v014_step1_same_input_truth/raw_truth`

Do not touch:

- production `src/gpuwrf/**`
- unrelated untracked artifacts
- TOST outputs
- global/pristine WRF trees in place; copy before patching

## Required Work

1. Prepare a disposable CPU-WRF tree and run directory.
2. Add an env-gated hook in `dyn_em/solve_em.F` immediately after
   `after_all_rk_steps` and before RK halo includes:
   - enabled by `WRFGPU2_SAME_INPUT_STEP1=1`;
   - domain `grid%id == 2`;
   - `grid%itimestep == 1`;
   - emits full-domain arrays for `T`, `P`, `PB`, `PH`, `PHB`, `MU`, `MUB`,
     `U`, `V`, `W`, `QVAPOR`, `QCLOUD`, `QRAIN`, `QICE`, `QSNOW`, `QGRAUP`.
3. Convert the emitted WRF truth to the accepted npz contract:
   - keys exactly match the 16 schema field names;
   - shapes match JAX logical order:
     `T/P/PB/Q*` `(k,y,x)`, `PH/PHB/W` `(kstag,y,x)`,
     `U` `(k,y,xstag)`, `V` `(k,ystag,x)`, `MU/MUB` `(y,x)`;
   - dtype `float64`;
   - if emitted by tile, overlap duplicates must be exact or fail closed.
4. Build a JAX CPU step-1 comparator from the current same-input loader:
   - construct initial d02 carry and namelist under `JAX_PLATFORMS=cpu`;
   - compute `run_radiation` with the same cadence semantics as production;
   - call `_physics_step_forcing` and `_rk_scan_step_with_pre_halo_capture`;
   - compare `result.pre_halo_state` against the WRF npz.
5. Emit per-field residual metrics:
   - count
   - max_abs
   - RMSE
   - bias
   - p95
   - p99
   - first mismatch index
   - worst mismatch index
6. If any stage cannot run, emit an implementation-ready blocker with exact
   source location, exact missing field/object, and the smallest next patch/tool.

## Verdicts

Emit exactly one final verdict:

- `STEP1_SAME_INPUT_COMPARISON_EXECUTED_FIRST_DIVERGENT_<field>`
- `STEP1_SAME_INPUT_COMPARISON_EXECUTED_CLEAN`
- `STEP1_WRF_TRUTH_READY_JAX_CAPTURE_BLOCKED_<specific>`
- `STEP1_WRF_TRUTH_BLOCKED_<specific>`
- `STEP1_SAME_INPUT_BLOCKED_<specific>`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_same_input_truth.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_same_input_truth.py
python -m json.tool proofs/v014/step1_same_input_truth.json \
  >/tmp/step1_same_input_truth.validated.json
git diff -- src/gpuwrf
```

If `proofs/v014/same_input_contract_builder.py` is updated:

```bash
python -m py_compile proofs/v014/same_input_contract_builder.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/same_input_contract_builder.py
python -m json.tool proofs/v014/same_input_contract_builder.json \
  >/tmp/same_input_contract_builder.step1.validated.json
```

Record all WRF copy/build/run commands, MPI rank count, stdout/stderr/rsl paths,
and the WRF patch diff. CPU-WRF may use MPI and CPU cores; it must not use the
GPU.

## Acceptance Criteria

- Production `src/gpuwrf/**` remains unchanged.
- WRF hook patch is preserved as a diff proof.
- Either the accepted step-1 truth npz exists and a strict WRF-vs-JAX step-1
  comparison executes, or the sprint emits one exact implementation-ready
  blocker.
- JSON validates.
- Review report includes objective, files changed, commands run, proof objects,
  unresolved risks, and next decision.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_SAME_INPUT_TRUTH DONE - see proofs/v014/step1_same_input_truth.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```

If tmux socket access is blocked, still write all artifacts and leave the DONE
marker visible in the worker TUI.
