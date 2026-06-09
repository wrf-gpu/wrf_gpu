# Sprint Contract: V0.14 Step-1 Live-Nest Init Rerun

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Rerun the strict d02 step-1 same-input comparison using the production native
live-nest child base initialization path already present in
`src/gpuwrf/integration/d02_replay.py`, then decide whether the dominant
`MUB/PB/PHB/P` residuals close or a deeper operator/source issue remains.

This is a proof/falsifier sprint first. It is not a broad source-changing sprint.

## Trigger Evidence

- `proofs/v014/step1_same_input_truth.json`
- `proofs/v014/step1_same_input_truth.md`
- `.agent/sprints/2026-06-09-v014-step1-same-input-truth/manager-closeout.md`
- `proofs/v014/live_nest_base_source_fix.json`
- `proofs/v014/live_nest_base_source_fix.md`
- `src/gpuwrf/integration/d02_replay.py::_apply_live_nest_base_init`
- `src/gpuwrf/integration/d02_replay.py::build_replay_case(live_nest_parent=...)`
- `src/gpuwrf/integration/nested_pipeline.py` parent-to-child call pattern

## Method Rule

The previous strict comparison intentionally used a proof-local raw wrfinput
loader and found immediate full-domain divergence. This sprint must test the
already-landed live-nest initialization path, not re-debug the raw loader.

Accepted comparison:

1. Build d01 from native inputs.
2. Build d02 initial state with the live-nest parent initialization path
   (`live_nest_parent` or the same underlying `_apply_live_nest_base_init`
   semantics).
3. Construct the same d02 `OperationalCarry`/`OperationalNamelist` and run one
   CPU JAX step through `_physics_step_forcing` and
   `_rk_scan_step_with_pre_halo_capture`.
4. Compare `result.pre_halo_state` to the existing CPU-WRF truth npz:
   `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`.

Forbidden comparison:

- WRF step-1 truth against the JAX initial state.
- Raw wrfinput d02 initial state without live-nest base initialization as the
  headline result.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No FP32 or mixed-precision source work.
- No memory source work.
- No WRF rebuild or new WRF hook unless the existing truth npz is proven invalid.
- No GPU.
- No Hermes or Telegram.
- No broad dycore, acoustic, radiation, physics, or writer refactor.

## Inputs

- `/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3`
- `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`
- `proofs/v014/step1_same_input_truth.py`
- `proofs/v014/step1_same_input_truth.json`
- `proofs/v014/same_input_contract_builder.py`
- `proofs/v014/live_nest_base_source_fix.py`
- `proofs/v014/live_nest_base_source_fix.json`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/integration/nested_pipeline.py`
- `src/gpuwrf/runtime/operational_mode.py`

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_live_nest_init_rerun/**`

## Write Scope

Repository files:

- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_init_rerun.json`
- `proofs/v014/step1_live_nest_init_rerun.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-init-rerun.md`
- optional targeted updates to `proofs/v014/step1_same_input_truth.py` only if
  they factor reusable comparison helpers without changing its existing verdict
- optional source edits only if the sprint proves the existing live-nest init
  path is defective:
  - `src/gpuwrf/integration/d02_replay.py`
  - `src/gpuwrf/integration/nested_pipeline.py`

Do not touch unrelated source, WRF scratch trees, TOST outputs, or old
untracked artifacts.

## Required Work

1. Build a CPU-only proof path that constructs d01/d02 with live-nest base init.
   Because `build_replay_case` uses GPU-only `State.zeros`, either:
   - call it only if it works under this CPU-only proof setup, or
   - reuse its internal live-nest helpers with proof-local direct constructors
     so CPU-only execution remains possible.
2. Record raw-init versus live-nest-init base-field deltas for d02 initial state
   against CPU-WRF h0 or the step-1 truth where appropriate:
   `HGT`, `PB`, `MUB`, `PHB`, plus `P_TOTAL/MU_TOTAL/PH_TOTAL` if available.
3. Run one JAX CPU step with `_physics_step_forcing` and
   `_rk_scan_step_with_pre_halo_capture`.
4. Compare against the existing step-1 truth npz for all 16 schema fields and
   emit per-field:
   - count
   - max_abs
   - RMSE
   - bias
   - p95
   - p99
   - first mismatch index
   - worst mismatch index
5. Classify the result:
   - If `MUB/PB/PHB` collapse to formula-level residuals and another field is
     first/dominant, name the next operator-localization sprint.
   - If `MUB/PB/PHB` remain large, name the exact missing source/state/loader
     path and whether it is proof-loader wiring or production init.
   - If CPU-only construction is blocked, emit the exact blocker and smallest
     next patch/tool.

## Verdicts

Emit exactly one final verdict:

- `STEP1_LIVE_NEST_INIT_BASE_RESIDUALS_CLOSED_NEXT_<field>`
- `STEP1_LIVE_NEST_INIT_BASE_RESIDUALS_REMAIN_<specific>`
- `STEP1_LIVE_NEST_INIT_BLOCKED_<specific>`
- `STEP1_LIVE_NEST_INIT_COMPARISON_CLEAN`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_live_nest_init_rerun.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_live_nest_init_rerun.py
python -m json.tool proofs/v014/step1_live_nest_init_rerun.json \
  >/tmp/step1_live_nest_init_rerun.validated.json
git diff -- src/gpuwrf
```

If source files are changed, also run:

```bash
python -m py_compile \
  src/gpuwrf/integration/d02_replay.py \
  src/gpuwrf/integration/nested_pipeline.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/live_nest_base_source_fix.py
python -m json.tool proofs/v014/live_nest_base_source_fix.json \
  >/tmp/live_nest_base_source_fix.after_step1_init.validated.json
```

## Acceptance Criteria

- JSON validates.
- CPU-only proof records `jax_default_backend="cpu"` and no visible GPU.
- The existing truth npz is reused; no WRF rebuild is done unless explicitly
  justified.
- Either the strict live-nest-init comparison executes, or the exact blocker is
  named.
- Production `src/gpuwrf/**` remains unchanged unless a concrete live-nest init
  defect is proven and fixed narrowly.
- Review report includes objective, files changed, commands run, proof objects,
  unresolved risks, and next decision.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_LIVE_NEST_INIT_RERUN DONE - see proofs/v014/step1_live_nest_init_rerun.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
