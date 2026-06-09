# Sprint Contract: V0.14 Full-Domain Source/Save Wrapper And Truth Surface

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Turn the accepted WRF source/save boundary into a strict same-input single-RK
comparison. Build the missing proof-only wrapper and the WRF-emitted truth
surface needed to call `_rk_scan_step_with_pre_halo_capture` with WRF-controlled
inputs, then compare JAX output against WRF post-RK/pre-halo truth.

If strict execution still cannot run, emit the next exact blocker. Do not make a
weak comparison and do not edit production model code.

## Trigger Evidence

- `proofs/v014/source_save_boundary_hook.json`
- `proofs/v014/source_save_boundary_hook.md`
- `proofs/v014/same_input_single_rk_parity_sources.json`
- `proofs/v014/same_input_single_rk_parity_sources.md`
- `.agent/reviews/2026-06-09-v014-source-save-boundary.md`
- `.agent/sprints/2026-06-09-v014-source-save-boundary/manager-closeout.md`

The accepted boundary is after `first_rk_step_part1`,
`first_rk_step_part2`, and `rk_tendency`, but before `relax_bdy_dry`,
`rk_addtend_dry`, `spec_bdy_dry`, `small_step_prep`, and `advance_uv`.

## Non-Goals

- No production `src/gpuwrf/**` edits.
- No GPU.
- No TOST.
- No Switzerland validation.
- No FP32 or memory source work.
- No Hermes or Telegram.
- No dycore fix. This is an instrumentation/proof-wrapper sprint only.

## Source And Scratch Inputs

Use the validated lineage:

- `/mnt/data/wrf_gpu2/v014_source_save_boundary/WRF`
- `/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3`
- `/mnt/data/wrf_gpu2/v014_source_save_boundary/source_save_output/`
- `/mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/full_pre_rk_output/`
- `/mnt/data/wrf_gpu2/v014_post_rk_refresh/refresh_output/`

Relevant repo references:

- `proofs/v014/source_save_boundary_hook.py`
- `proofs/v014/same_input_single_rk_parity_sources.py`
- `proofs/v014/source_save_boundary_hook_wrf_patch.diff`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/runtime/operational_state.py`
- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/integration/d02_replay.py`

Suggested scratch root:

- `/mnt/data/wrf_gpu2/v014_full_domain_source_wrapper`

## Write Scope

Repository files:

- `proofs/v014/full_domain_source_truth.py`
- `proofs/v014/full_domain_source_truth.json`
- `proofs/v014/full_domain_source_truth.md`
- `proofs/v014/full_domain_source_truth_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_wrapped.py`
- `proofs/v014/same_input_single_rk_parity_wrapped.json`
- `proofs/v014/same_input_single_rk_parity_wrapped.md`
- `.agent/reviews/2026-06-09-v014-full-domain-source-wrapper.md`

Scratch files:

- `/mnt/data/wrf_gpu2/v014_full_domain_source_wrapper/**`

Do not touch:

- production `src/gpuwrf/**`
- unrelated untracked artifacts
- TOST outputs

## Required Work

1. Inventory exactly what `_rk_scan_step_with_pre_halo_capture` reads from
   `State`, `OperationalCarry`, `OperationalNamelist`, `GridSpec`,
   `DycoreMetrics`, `DryPhysicsTendencies`, and boundary/coupling state.
2. Decide the smallest honest comparison surface:
   - preferred: full-domain/full-vertical source/save and post-RK/pre-halo truth;
   - allowed: a wide tile-owned rectangular patch if full-domain text output
     would be wasteful, but it must contain many halo-valid cells and the JSON
     must state why full-domain was not used;
   - forbidden: one-cell comparisons, JAX-vs-JAX self-compares, or mixing a
     JAX-produced checkpoint with WRF source/save leaves.
3. Add a disposable env-gated WRF hook in scratch that emits, at the accepted
   source/save boundary and at the matching post-RK/pre-halo boundary:
   - full native dry state needed to construct `State`;
   - dry source/save leaves needed for `DryPhysicsTendencies`;
   - promoted carry leaves needed by the selected JAX entry point, or an exact
     proof that a dry-only wrapper can omit them safely;
   - boundary/coupling leaves needed by the selected entry point, or an exact
     blocker naming the first missing leaf;
   - current namelist/grid/metric metadata sufficient to build the wrapper.
4. Run CPU-WRF with no GPU for `d02` step `6000` if new scratch output is
   needed.
5. Produce `proofs/v014/full_domain_source_truth.*` that inventories records,
   duplicate overlap/tile ownership, scored cells, hook placement, command logs,
   hashes, and whether the emitted surface is sufficient.
6. Implement/run `proofs/v014/same_input_single_rk_parity_wrapped.py`:
   - construct JAX inputs only from WRF-emitted fields plus static input files;
   - call the narrowest correct JAX RK entry point, preferably
     `_rk_scan_step_with_pre_halo_capture`;
   - compare halo-valid cells against WRF post-RK/pre-halo truth for at least
     `T/P/PB/PH/PHB/MU/MUB/U/V/W`;
   - emit per-field max_abs, RMSE, bias, p95, p99, count, and ranked residuals;
   - keep top-level Markdown short and put detailed tables in JSON/CSV.
7. If strict execution cannot run, emit one precise blocker naming the missing
   wrapper contract, field, boundary ordering conflict, old-field strategy, or
   patch-width limitation.

## Verdicts

Emit exactly one final manager-facing verdict:

- `SAME_INPUT_SINGLE_RK_GRID_CLEAN`
- `SAME_INPUT_SINGLE_RK_GRID_MISMATCH_<dominant_field_or_operator>`
- `FULL_DOMAIN_WRAPPER_BLOCKED_<reason>`
- `FULL_DOMAIN_TRUTH_SURFACE_BLOCKED_<reason>`
- `OLD_FIELD_STRATEGY_BLOCKED_<reason>`
- `PATCH_WIDTH_BLOCKED_<needed>`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/full_domain_source_truth.py
python -m json.tool proofs/v014/full_domain_source_truth.json \
  >/tmp/full_domain_source_truth.validated.json
python -m py_compile proofs/v014/same_input_single_rk_parity_wrapped.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/same_input_single_rk_parity_wrapped.py
python -m json.tool proofs/v014/same_input_single_rk_parity_wrapped.json \
  >/tmp/same_input_single_rk_parity_wrapped.validated.json
git diff -- src/gpuwrf
```

If WRF build/run is needed, record exact commands and log paths in JSON.

## Acceptance Criteria

- CPU-only and no GPU use.
- JSON artifacts validate.
- `git diff -- src/gpuwrf` is empty.
- The final proof either runs strict same-input JAX with WRF-controlled inputs,
  or blocks with one exact next missing field/contract/boundary issue.
- No weak comparison is emitted.
- Review report includes objective, files changed, commands run, proof objects,
  unresolved risks, and next decision.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT FULL_DOMAIN_SOURCE_WRAPPER DONE - see proofs/v014/same_input_single_rk_parity_wrapped.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```

If tmux socket access is blocked, still write all artifacts and leave the DONE
marker visible in the worker TUI.
