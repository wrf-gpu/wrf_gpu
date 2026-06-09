# Sprint Contract: V0.14 Full Pre-RK Savepoint Hook + Same-Input Loader

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Create the proof-enabling boundary that the previous sprint found missing:
a full CPU-WRF pre-RK native-state plus RK-fixed tendency/source savepoint at
`d02` step `6000`, and a proof-only JAX loader/wrapper that either runs the
strict same-input one-step comparison or names the next exact blocker.

This sprint is allowed to edit and rebuild a disposable WRF scratch copy. It is
not allowed to edit production `src/gpuwrf/**`.

## Trigger Evidence

- `.agent/reviews/2026-06-09-v014-dynamic-root-cause-opus-critic.md`
- `proofs/v014/dynamic_root_cause_opus_critic.json`
- `proofs/v014/same_input_single_rk_parity.json`
- `proofs/v014/same_input_single_rk_parity.md`
- `.agent/reviews/2026-06-09-v014-same-input-single-rk-parity.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

## Non-Goals

- No production `src/gpuwrf/**` edits.
- No GPU.
- No TOST.
- No Switzerland validation.
- No FP32/mixed precision work.
- No memory source work.
- No Hermes or Telegram.

## Source And Scratch Inputs

Reusable WRF lineage and run directory:

- `/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF`
- `/mnt/data/wrf_gpu2/v014_post_rk_refresh/run_case3`

Existing narrow pre-RK hook:

- `proofs/v014/pre_rk_input_boundary_wrf_patch.diff`
- `/tmp/wrf_gpu2_v014_pre_rk_input_boundary/pre_rk_output/pre_rk_input_d2_step_6000_*.txt`

Existing post-RK/pre-halo WRF truth:

- `/mnt/data/wrf_gpu2/v014_post_rk_refresh/refresh_output/refresh_post_after_all_rk_steps_pre_halo_d2_step_6000_*.txt`

JAX loader references:

- `proofs/v014/same_input_single_rk_parity.py`
- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/runtime/operational_state.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`

Use a new scratch root unless you prove reuse is cleaner:

- `/mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook`

## Write Scope

Repository files:

- `proofs/v014/full_pre_rk_savepoint_hook.py`
- `proofs/v014/full_pre_rk_savepoint_hook.json`
- `proofs/v014/full_pre_rk_savepoint_hook.md`
- `proofs/v014/full_pre_rk_savepoint_hook_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_full.py`
- `proofs/v014/same_input_single_rk_parity_full.json`
- `proofs/v014/same_input_single_rk_parity_full.md`
- `.agent/reviews/2026-06-09-v014-full-pre-rk-savepoint-hook.md`

Scratch files:

- `/mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/**`

Do not touch:

- production `src/gpuwrf/**`
- old unrelated untracked artifacts
- TOST outputs

## Required Work

1. Build or reuse a disposable WRF copy from the existing dmpar lineage.
2. Add an env-gated `dyn_em/solve_em.F` hook immediately after
   `grid%itimestep = grid%itimestep + 1` / `grid%dtbc` and before current-step
   RK work.
3. Emit full native-staggered step-entry state over a patch wide enough for at
   least a small halo-valid score region:
   - mass-grid full levels: `T`, `P`, `PB`, `MU`, `MUB`, `QVAPOR` and active
     moisture/scalar leaves available in this case
   - u-grid full levels: `U`
   - v-grid full levels: `V`
   - w/ph vertical faces: `W`, `PH`, `PHB`
   - history/source state: `T_OLD`, `T_HIST_SRC`, `MU_OLD`, plus full native
     `_1`/`_old` fields needed by the JAX RK carry
   - RK-fixed dry physics/source tendencies consumed by
     `DryPhysicsTendencies`: `ru_tendf`, `rv_tendf`, `rw_tendf`, `ph_tendf`,
     `t_tendf`, `mu_tendf`, `h_diabatic`, `u_save`, `v_save`, `w_save`,
     `ph_save`, `t_save`
   - metadata: domain, step, time, dt, dx/dy, tile bounds, zero-based/native
     index conventions, selected cell, patch bounds, rk order, hashes.
4. Run CPU-WRF with the hook for `d02` step `6000`; use no GPU.
5. Produce `proofs/v014/full_pre_rk_savepoint_hook.*` that inventories the
   emitted fields, shapes, patch bounds, duplicate tile overlaps, command log
   paths, hashes, and whether the hook output is sufficient for a strict
   same-input JAX comparison.
6. Implement a proof-only JAX loader/wrapper in
   `proofs/v014/same_input_single_rk_parity_full.py` if the emitted data is
   sufficient. The wrapper must:
   - construct a JAX `State`, `OperationalCarry`, and `DryPhysicsTendencies`
     from the WRF savepoint without using JAX-generated physics tendencies;
   - run one same-input boundary through the narrowest correct JAX RK entry
     point, preferably `_rk_scan_step_with_pre_halo_capture`;
   - compare only halo-valid cells against WRF `post_after_all_rk_steps_pre_halo`;
   - emit per-field max_abs/RMSE and a ranked residual table for at least
     `T/P/PB/PH/PHB/MU/MUB/U/V/W` if computable.
7. If the comparison cannot run, do not fake it. Emit one blocked verdict naming
   the exact missing WRF field, metadata, JAX wrapper contract, or patch width.

## Verdicts

Emit exactly one top-level verdict for the final comparison proof:

- `DYNAMICS_CLEAN_SINGLE_STEP_FULL_INPUT`
- `SAME_INPUT_SINGLE_STEP_MISMATCH_<dominant_field_or_operator>`
- `FULL_PRE_RK_HOOK_BLOCKED_<reason>`
- `FULL_PRE_RK_JAX_LOADER_BLOCKED_<reason>`
- `FULL_PRE_RK_PATCH_WIDTH_BLOCKED_<needed>`

The WRF-hook proof may have its own hook-level verdict, but the manager-facing
review must state the final comparison status plainly.

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/full_pre_rk_savepoint_hook.py
python -m json.tool proofs/v014/full_pre_rk_savepoint_hook.json \
  >/tmp/full_pre_rk_savepoint_hook.validated.json
python -m py_compile proofs/v014/same_input_single_rk_parity_full.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/same_input_single_rk_parity_full.py
python -m json.tool proofs/v014/same_input_single_rk_parity_full.json \
  >/tmp/same_input_single_rk_parity_full.validated.json
git diff -- src/gpuwrf
```

If WRF build/run is needed, record exact commands and log paths in the JSON.
Recommended build/run pattern:

```bash
mkdir -p /mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook
rsync -a /mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF/ \
  /mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/WRF/
rsync -a /mnt/data/wrf_gpu2/v014_post_rk_refresh/run_case3/ \
  /mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/run_case3/
cd /mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/WRF
timeout 3600 env \
  PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH \
  NETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build \
  PNETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build \
  WRFIO_NCD_LARGE_FILE_SUPPORT=1 \
  tcsh ./compile em_real \
  >/mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/compile_full_pre_rk.log 2>&1
ln -sf /mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/WRF/main/wrf.exe \
  /mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/run_case3/wrf.exe
cd /mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/run_case3
timeout 3600 env \
  PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH \
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  WRFGPU2_FULL_PRE_RK=1 \
  WRFGPU2_FULL_PRE_RK_ROOT=/mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/full_pre_rk_output \
  WRFGPU2_FULL_PRE_RK_GRID=2 \
  WRFGPU2_FULL_PRE_RK_START_STEP=6000 \
  WRFGPU2_FULL_PRE_RK_END_STEP=6000 \
  mpirun --oversubscribe -np 28 ./wrf.exe \
  >/mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/run_case3/full_pre_rk_28rank_stdout.log 2>&1
```

## Acceptance Criteria

- CPU-only and no GPU use.
- JSON artifacts validate.
- `git diff -- src/gpuwrf` is empty.
- WRF hook output is either present and inventoried, or the exact external
  blocker is named with logs.
- No weak same-input claim: either the strict one-step comparison runs with
  controlled WRF inputs, or a precise blocked verdict is emitted.
- Review report includes objective, files changed, commands run, proof objects,
  unresolved risks, and next decision.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT FULL_PRE_RK_SAVEPOINT DONE - see proofs/v014/same_input_single_rk_parity_full.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```

If tmux socket access is blocked, still write all artifacts and leave the DONE
marker visible in the worker TUI.
