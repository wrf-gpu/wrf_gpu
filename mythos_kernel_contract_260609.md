# Mythos Kernel Contract 2026-06-09

## Role

You are the experimental high-capacity kernel/debug model for `wrf_gpu2`
v0.14. The primary manager is intentionally pausing after the current
base-state worker so you can attempt a larger one-pass root-cause fix.

Your task is not station-score tuning and not memory optimization. Your task is
to fix, or exactly root-cause, the current per-cell grid divergence in the GPU
WRF rewrite.

Project goal: a WRF-faithful-enough, GPU-optimized, near compute- and
memory-optimal, scalable GPU rewrite. Do not use shortcuts that make the model
less WRF-faithful, less GPU-native, or dependent on CPU-WRF at runtime.

## Read First

Read in this order:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
5. `.agent/decisions/V0140-VALIDATION-PLAN.md`
6. `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
7. `proofs/v014/step1_jax_start_domain_input_split.md`
8. `proofs/v014/step1_start_domain_perturb_subsurface.md`
9. `proofs/v014/step1_live_nest_perturb_state_init.md`
10. Read the newest base-boundary proof:
    `proofs/v014/step1_base_state_boundary.md`
11. Read memory-manager context only for locks:
    `proofs/v014/memory_manager_260609.md`

Do not use the old global `wrf-gpu-port` skill. The project-local files are
authoritative.

## Critical Coordination

Before editing any production source:

- Confirm no active normal debug worker owns the same source files.
- Do not interrupt or modify the memory manager in `tmux 0:3`.
- Do not start TOST, Switzerland validation, long GPU validation, or memory/FP32
  source work unless the manager explicitly asks.
- Do not send Hermes/Telegram updates.
- Do not edit unrelated dirty/untracked files.

If another worker has written newer proof artifacts, treat those as newer truth
than this contract and update your plan accordingly.

## Current Known Problem

The original symptom was unacceptable per-cell divergence between WRF CPU and
the JAX/GPU rewrite. Station TOST is not sufficient; v0.14 requires field-level
cell comparison to be explained and minimized before long validation resumes.

The current debug lane has localized the Step-1 d02 divergence through many
strict WRF truth surfaces. The latest accepted chain says the remaining active
bug is not broad RK/acoustic/dynamics. It is in live-nest/start-domain
initialization for the `P/MU/W` family:

- WRF source order around `start_domain(nest,.TRUE.)`, hypsometric
  `P/al/alt`, `press_adj`, and W-surface handling is now known.
- Direct formula patching with current JAX inputs was refuted.
- The remaining dominant mismatch is base-state reconstruction feeding fp32
  `AL/ALT` diagnosis, especially `PHB+MUB` and exact WRF `p_surf -> MUB`
  arithmetic before the `AL/ALT` pass.

Key latest accepted metrics from
`proofs/v014/step1_jax_start_domain_input_split.md`:

- Current pressure formula versus WRF P:
  max_abs `3.9458582235092763`, RMSE `0.3832298992869327`.
- Replacing diagnosed ALT with WRF ALT:
  max_abs `0.07605321895971429`, RMSE `0.006830944106223064`.
- FP32 ALT diagnosis with WRF `PHB+MUB`:
  pressure max_abs `0.0859375`, RMSE `0.009877167668418278`.
- WRF fields with fp64 ALT diagnosis:
  pressure max_abs `2.961779549412313`, so dtype/source order matters.
- Best local WRF-order fp32/cp=1004.5 base candidate:
  `P_STATE` max_abs `2.828125`, `MU_STATE` max_abs `0.011962890625`.

Latest accepted base-boundary proof from
`proofs/v014/step1_base_state_boundary.md`:

- Verdict:
  `STEP1_BASE_STATE_BOUNDARY_LOCALIZED_P_SURF_MUB_FP32_SOURCE_ARITHMETIC`.
- Current/proof-local fp32/cp=1004.5 `p_surf` formula still leaves
  `P_STATE=2.828125 Pa` and `MU_STATE=0.011962890625 Pa`.
- Substituting WRF-emitted `MUB` into the same proof-local base/AL/ALT path
  reduces downstream `P_STATE` to `0.40625 Pa` and `MU_STATE` to
  `0.001220703125 Pa`.
- That points to exact WRF surface-pressure/`MUB` arithmetic, not PH state,
  terrain, coefficients, cp alone, PHB integration order, or `press_adj` order.

## Files And Locations

Main repo:

- `/home/enric/src/wrf_gpu2`

Likely source files:

- `src/gpuwrf/integration/d02_replay.py`
- possibly live-nest/init helpers referenced by `d02_replay.py`
- do not broaden into `src/gpuwrf/dynamics/**` unless a proof shows this
  start-domain hypothesis is wrong

WRF source:

- Use paths referenced by the proof scripts and JSON metadata.
- Key source files are typically WRF `dyn_em/start_em.F` and
  `share/module_model_constants.F`.

Important proof scripts/artifacts:

- `proofs/v014/step1_jax_start_domain_input_split.{py,json,md}`
- `proofs/v014/step1_start_domain_perturb_subsurface.{py,json,md}`
- `proofs/v014/step1_start_domain_perturb_subsurface_wrf_patch.diff`
- `proofs/v014/step1_live_nest_perturb_state_init.{py,json,md}`
- `proofs/v014/step1_live_nest_init_rerun.{py,json,md}`
- `proofs/v014/step1_live_nest_theta_semantics.{py,json,md}`
- `proofs/v014/step1_transient_adjust_base_fix.*`
- `proofs/v014/step1_base_state_boundary.{py,json,md}`
- `.agent/reviews/2026-06-09-v014-step1-base-state-boundary.md`

WRF truth roots already produced:

- `/mnt/data/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715/wrf_truth`
- Additional roots are recorded in each proof JSON under `inputs`,
  `wrf_truth_root`, or similar keys.

Memory/FP32 context:

- `proofs/v014/memory_manager_260609.md`
- `.agent/reviews/2026-06-09-v014-memory-manager-260609.md`
- These are lock/context files only. Do not start memory/FP32 source work.

## Allowed Source Scope

Preferred source edit, if proven:

- `src/gpuwrf/integration/d02_replay.py`

Possible adjacent source edits only if proven necessary and GPU-native:

- live-nest/init helper files directly called by `d02_replay.py`

Avoid unless the current hypothesis is refuted by proof:

- `src/gpuwrf/dynamics/**`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/contracts/state.py`
- boundary/restart/wrfout/state ABI files
- memory/FP32 source files

No runtime dependency on CPU-WRF is allowed. Disposable WRF instrumentation for
truth surfaces is allowed under `/mnt/data/wrf_gpu2/` and proof files only.

## Endpoint

Best endpoint:

- Fix the current live-nest/start-domain bug with a narrow GPU-native source
  patch.
- Rerun strict Step-1 proof(s) and show the `P/MU/W` family is within declared
  gates.
- If that reveals another divergence, continue if it is naturally adjacent and
  the proof path is clear.
- Ideal endpoint is a divergence-free or materially bounded Step-1 field
  comparison against WRF CPU truth for the current 16-field schema.

Acceptable endpoint if one-pass fix is not possible:

- Produce an exact root-cause report with the source line/formula/order still
  missing, the next truth surface needed, and why a production patch would still
  be a guess.

Unacceptable endpoints:

- JAX-vs-JAX self-compare only.
- Station-only TOST result.
- Clamps/masks/fudges to make fields look close.
- CPU-WRF runtime call in production.
- Broad dycore rewrite without a strict WRF truth surface proving need.

## Suggested Attack Plan

1. Refresh current state:
   - `git log -1 --oneline`
   - `git status --short`
   - confirm `proofs/v014/step1_base_state_boundary.*` exists and is valid.
2. Mine the base-boundary proof first:
   - current JAX/proof-local `p_surf/MUB` formula leaves `P_STATE` `2.828125`;
   - WRF final `MUB` substitution drops `P_STATE` to `0.40625`;
   - the next missing surface is exact WRF `p_surf_before_mub` or an exactly
     compatible fp32/libm helper.
3. Compare exact WRF `p_surf -> MUB` arithmetic against current
   `d02_replay.py`:
   - dtype (`real`/float32) order;
   - `cp=1004.5` versus `1004.0`;
   - terrain input rounding;
   - assignment order before/after multi-domain reconstitution;
   - `MUB = p_surf - p_top` rounding point;
   - `PB = c3h*MUB + c4h + p_top` evaluation order;
   - PHB integration evaluation order.
4. Patch only after the exact source-order gap is proven.
5. Rerun all relevant proof gates.

## Required Proof Objects

If you patch source, write:

- `proofs/v014/mythos_kernel_fix_260609.py`
- `proofs/v014/mythos_kernel_fix_260609.json`
- `proofs/v014/mythos_kernel_fix_260609.md`
- `.agent/reviews/2026-06-09-v014-mythos-kernel-fix.md`

If you do not patch source, write:

- `proofs/v014/mythos_kernel_analysis_260609.json`
- `proofs/v014/mythos_kernel_analysis_260609.md`
- `.agent/reviews/2026-06-09-v014-mythos-kernel-analysis.md`

The report must include:

- exact files changed;
- commands run;
- WRF truth surfaces used;
- before/after metrics for every changed source path;
- ranked hypotheses and exclusions;
- unresolved risks;
- next decision needed.

## Minimum Validation

Always run:

```bash
python -m py_compile proofs/v014/mythos_kernel_fix_260609.py 2>/dev/null || true
python -m py_compile proofs/v014/mythos_kernel_analysis_260609.py 2>/dev/null || true
python -m json.tool proofs/v014/mythos_kernel_fix_260609.json \
  >/tmp/mythos_kernel_fix_260609.validated.json 2>/dev/null || true
python -m json.tool proofs/v014/mythos_kernel_analysis_260609.json \
  >/tmp/mythos_kernel_analysis_260609.validated.json 2>/dev/null || true
git diff --check
git diff -- src/gpuwrf
```

If you change `src/gpuwrf/integration/d02_replay.py`, also run:

```bash
python -m py_compile src/gpuwrf/integration/d02_replay.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_jax_start_domain_input_split.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_start_domain_perturb_subsurface.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_live_nest_perturb_state_init.py
python -m json.tool proofs/v014/step1_jax_start_domain_input_split.json \
  >/tmp/step1_jax_start_domain_input_split.after_mythos.validated.json
python -m json.tool proofs/v014/step1_start_domain_perturb_subsurface.json \
  >/tmp/step1_start_domain_perturb_subsurface.after_mythos.validated.json
python -m json.tool proofs/v014/step1_live_nest_perturb_state_init.json \
  >/tmp/step1_live_nest_perturb_state_init.after_mythos.validated.json
```

If `proofs/v014/step1_base_state_boundary.py` exists, run it too after any
source or formula change:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_base_state_boundary.py
python -m json.tool proofs/v014/step1_base_state_boundary.json \
  >/tmp/step1_base_state_boundary.after_mythos.validated.json
```

GPU validation is not required for the first Mythos pass unless the manager
explicitly allows it. Do not run TOST.

## Completion Signal

If running in tmux, notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'MYTHOS KERNEL DONE - see proofs/v014/mythos_kernel_fix_260609.md or proofs/v014/mythos_kernel_analysis_260609.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```

The primary manager will then review, validate, decide whether to merge, and
continue v0.14.
