# Sprint Contract: V0.14 WRF Source/Save Boundary for Same-Input RK

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Find and emit the first WRF boundary at `d02` step `6000` where the current-step
source/save-family leaves required by JAX `DryPhysicsTendencies` exist, while
the initial native state for a same-input JAX one-step comparison is still
well-defined. Then either run the strict same-input JAX wrapper or emit the next
exact blocker.

The prior sprint proved full step-entry state is available, but WRF has not yet
computed `*_tendf` / `*_save` at that boundary. This sprint must close that
instrumentation gap, not edit production model code.

## Trigger Evidence

- `proofs/v014/full_pre_rk_savepoint_hook.json`
- `proofs/v014/full_pre_rk_savepoint_hook.md`
- `proofs/v014/same_input_single_rk_parity_full.json`
- `proofs/v014/same_input_single_rk_parity_full.md`
- `.agent/reviews/2026-06-09-v014-full-pre-rk-savepoint-hook.md`
- `.agent/sprints/2026-06-09-v014-full-pre-rk-savepoint-hook/manager-closeout.md`

## Non-Goals

- No production `src/gpuwrf/**` edits.
- No GPU.
- No TOST.
- No Switzerland validation.
- No FP32 or memory work.
- No Hermes or Telegram.
- No source fix. This is an instrumentation and same-input proof sprint.

## Source And Scratch Inputs

Use the already validated lineage:

- `/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF`
- `/mnt/data/wrf_gpu2/v014_post_rk_refresh/run_case3`
- `/mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/full_pre_rk_output/`

Relevant repo references:

- `proofs/v014/full_pre_rk_savepoint_hook.py`
- `proofs/v014/same_input_single_rk_parity_full.py`
- `proofs/v014/full_pre_rk_savepoint_hook_wrf_patch.diff`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/runtime/operational_state.py`
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/contracts/state.py`

Suggested scratch root:

- `/mnt/data/wrf_gpu2/v014_source_save_boundary`

## Write Scope

Repository files:

- `proofs/v014/source_save_boundary_hook.py`
- `proofs/v014/source_save_boundary_hook.json`
- `proofs/v014/source_save_boundary_hook.md`
- `proofs/v014/source_save_boundary_hook_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_sources.py`
- `proofs/v014/same_input_single_rk_parity_sources.json`
- `proofs/v014/same_input_single_rk_parity_sources.md`
- `.agent/reviews/2026-06-09-v014-source-save-boundary.md`

Scratch files:

- `/mnt/data/wrf_gpu2/v014_source_save_boundary/**`

Do not touch:

- production `src/gpuwrf/**`
- unrelated untracked artifacts
- TOST outputs

## Required Work

1. Inspect WRF `dyn_em/solve_em.F` and included routines around
   `first_rk_step_part1`, `first_rk_step_part2`, `rk_tendency`,
   `rk_addtend_dry`, and the first state-changing dry/acoustic update.
2. Identify the earliest valid boundary for same-input comparison:
   - preferred: sources/save-family leaves exist and the native state is still
     exactly the step-entry state from `full_pre_rk_savepoint_hook`;
   - allowed: move the comparison boundary forward, but then emit both the
     native state and source/save leaves at that same later boundary and compare
     JAX from that later state;
   - forbidden: combine step-entry state with later WRF source leaves if WRF has
     already mutated the state.
3. Add an env-gated scratch WRF hook that emits:
   - the native state at the accepted boundary for all needed fields;
   - `ru_tendf`, `rv_tendf`, `rw_tendf`, `ph_tendf`, `t_tendf`, `mu_tendf`,
     `h_diabatic`, `u_save`, `v_save`, `w_save`, `ph_save`, `t_save`;
   - `moist_old` and `scalar_old` or a proof that they are not needed for the
     selected comparison boundary;
   - metadata proving the boundary position relative to first source generation
     and first state mutation.
4. Run CPU-WRF with no GPU for `d02` step `6000`.
5. Produce `proofs/v014/source_save_boundary_hook.*` that inventories records,
   duplicate overlap, boundary placement, command logs, hashes, and whether the
   hook output is sufficient.
6. Implement/run `proofs/v014/same_input_single_rk_parity_sources.py`:
   - construct JAX `State`, `OperationalCarry`, and `DryPhysicsTendencies` from
     WRF-emitted fields only;
   - call the narrowest correct JAX RK entry point, preferably
     `_rk_scan_step_with_pre_halo_capture`;
   - compare halo-valid cells against WRF post-RK/pre-halo truth, or against a
     consistently moved WRF output boundary if the comparison boundary moved;
   - emit per-field max_abs/RMSE and ranked residuals for at least
     `T/P/PB/PH/PHB/MU/MUB/U/V/W` if executed.
7. If strict execution cannot run, emit a precise blocked verdict naming the
   next missing field, boundary ordering conflict, wrapper contract, or patch
   width. Do not produce a weak comparison.

## Verdicts

Emit exactly one final manager-facing verdict:

- `DYNAMICS_CLEAN_SINGLE_STEP_WITH_SOURCES`
- `SAME_INPUT_SINGLE_STEP_WITH_SOURCES_MISMATCH_<dominant_field_or_operator>`
- `SOURCE_SAVE_BOUNDARY_READY_NO_JAX_WRAPPER_<reason>`
- `SOURCE_SAVE_BOUNDARY_ORDERING_CONFLICT_MOVE_COMPARISON_BOUNDARY`
- `SOURCE_SAVE_HOOK_BLOCKED_<reason>`
- `SOURCE_SAVE_PATCH_WIDTH_BLOCKED_<needed>`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/source_save_boundary_hook.py
python -m json.tool proofs/v014/source_save_boundary_hook.json \
  >/tmp/source_save_boundary_hook.validated.json
python -m py_compile proofs/v014/same_input_single_rk_parity_sources.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/same_input_single_rk_parity_sources.py
python -m json.tool proofs/v014/same_input_single_rk_parity_sources.json \
  >/tmp/same_input_single_rk_parity_sources.validated.json
git diff -- src/gpuwrf
```

If WRF build/run is needed, record exact commands and log paths in JSON.

## Acceptance Criteria

- CPU-only and no GPU use.
- JSON artifacts validate.
- `git diff -- src/gpuwrf` is empty.
- The WRF hook output is present and inventoried, or a precise external blocker
  is named with logs.
- The final proof either executes strict same-input JAX with WRF-controlled
  source/save leaves, or blocks with the exact missing boundary/field/ordering
  conflict.
- Review report includes objective, files changed, commands run, proof objects,
  unresolved risks, and next decision.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT SOURCE_SAVE_BOUNDARY DONE - see proofs/v014/same_input_single_rk_parity_sources.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```

If tmux socket access is blocked, still write all artifacts and leave the DONE
marker visible in the worker TUI.
