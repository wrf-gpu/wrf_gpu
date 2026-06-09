# Sprint Contract: V0.14 Step-1 JAX Start-Domain Input Split

Date: 2026-06-09 20:05 WEST
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`
Base commit: `66c091fc`

## Objective

Close or precisely localize the current JAX live-nest `start_domain` input gap
identified by the predecessor sprint:
`STEP1_START_DOMAIN_PERTURB_SUBSURFACE_LOCALIZED_CURRENT_JAX_AL_ALT_BASE_INPUT_GAP`.

The predecessor proved WRF source ordering around `start_domain(nest,.TRUE.)`,
`P/al/alt`, `press_adj`, and W-surface handling. It also refuted applying a
`P/MU` production patch with current JAX inputs because:

- current JAX pressure formula vs WRF after-hypsometric `P`:
  max_abs `3.9458582235092763` Pa;
- current JAX `press_adj` formula vs WRF after-press `MU`:
  max_abs `0.047773029698646496` Pa.

This sprint must split the current JAX inputs to that formula: final blended
terrain, base-state surfaces, `PH_STATE`, pre-`press_adj` `MU`, and diagnosed
`AL/ALT`.

## Method Rule

Use the fastest rigorous wall-clock path: proof-only CPU/JAX comparison against
the already-emitted WRF internal truth surfaces. Do not add a new WRF hook
unless the existing surfaces are provably insufficient.

If the manager hypothesis is wrong, do not stop at "blocked": rank alternate
causes, try cheap proof-local falsifiers, and name the next exact truth
surface or source line.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No GPU run.
- No FP32 or mixed-precision source work.
- No memory source work.
- No Hermes/Telegram.
- No broad dycore rewrite.
- No CPU-WRF runtime dependency in production.
- No timestep-loop host/device transfer.

## Inputs

- `proofs/v014/step1_start_domain_perturb_subsurface.py`
- `proofs/v014/step1_start_domain_perturb_subsurface.json`
- `proofs/v014/step1_start_domain_perturb_subsurface.md`
- `proofs/v014/step1_live_nest_perturb_state_init.py`
- `proofs/v014/step1_live_nest_perturb_state_init.json`
- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_jax_loader_tstate.py`
- `src/gpuwrf/integration/d02_replay.py`
- WRF truth root:
  `/mnt/data/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715/wrf_truth`

## File Ownership

Required repo files:

- `proofs/v014/step1_jax_start_domain_input_split.py`
- `proofs/v014/step1_jax_start_domain_input_split.json`
- `proofs/v014/step1_jax_start_domain_input_split.md`
- `.agent/reviews/2026-06-09-v014-step1-jax-start-domain-input-split.md`

Optional production source edit only if exact, narrow, and GPU-native:

- `src/gpuwrf/integration/d02_replay.py`

Do not edit `src/gpuwrf/dynamics/**`, `src/gpuwrf/runtime/**`,
`src/gpuwrf/contracts/**`, FP32 files, memory files, TOST outputs,
Switzerland outputs, or unrelated dirty/untracked artifacts.

## Required Work

1. Verify `66c091fc` is an ancestor and record branch/head.
2. Parse/reuse WRF internal `after_hypsometric_p_al_alt`,
   `before_press_adj`, `after_press_adj`, and `after_w_surface_branch`
   truth surfaces from the predecessor sprint.
3. Reconstruct the current JAX live-nest Step-1 start-domain inputs and compare
   them against WRF internal truth:
   - final blended `HT` and any terrain/fine-terrain delta;
   - `PB`, `MUB`, `PHB`;
   - `PH_STATE` and W-staggered base/perturbation consistency;
   - `MU` immediately before `press_adj`;
   - diagnosed `AL`, `ALT`, and `P_STATE` before/after the pressure formula.
4. Split the remaining `P/MU` residual by input family. The output must say
   whether the dominant cause is terrain/blend, base-state reconstruction,
   `PH/PHB`, `AL/ALT` diagnosis, time-level selection, boundary indexing, dtype
   order, or a missing WRF truth surface.
5. If exact source semantics are proven and a patch is narrow, optionally patch
   `src/gpuwrf/integration/d02_replay.py`, then rerun:
   - this proof;
   - `proofs/v014/step1_start_domain_perturb_subsurface.py`;
   - `proofs/v014/step1_live_nest_perturb_state_init.py`.
6. If exact closure is still not proven, name the exact next surface/source
   boundary and explain why patching would still be a guess.

## Verdicts

Emit exactly one final verdict:

- `STEP1_JAX_START_DOMAIN_INPUT_SPLIT_FIXED_<source_or_formula>`
- `STEP1_JAX_START_DOMAIN_INPUT_SPLIT_READY_FOR_PATCH_<contract>`
- `STEP1_JAX_START_DOMAIN_INPUT_SPLIT_LOCALIZED_<source_or_missing_contract>`
- `STEP1_JAX_START_DOMAIN_INPUT_SPLIT_BLOCKED_<specific_truth_or_source_gap>`
- `STEP1_JAX_START_DOMAIN_INPUT_SPLIT_REFUTED_<hypothesis_and_next_best>`

## Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_jax_start_domain_input_split.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_jax_start_domain_input_split.py
python -m json.tool proofs/v014/step1_jax_start_domain_input_split.json \
  >/tmp/step1_jax_start_domain_input_split.validated.json
git diff -- src/gpuwrf
```

If production source changes:

```bash
python -m py_compile src/gpuwrf/integration/d02_replay.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_start_domain_perturb_subsurface.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_live_nest_perturb_state_init.py
python -m json.tool proofs/v014/step1_start_domain_perturb_subsurface.json \
  >/tmp/step1_start_domain_perturb_subsurface.after_input_split.validated.json
python -m json.tool proofs/v014/step1_live_nest_perturb_state_init.json \
  >/tmp/step1_live_nest_perturb_state_init.after_input_split.validated.json
```

## Acceptance Criteria

- JSON validates and records CPU-only execution.
- The report has a ranked residual split by input family.
- Any source fix has before/after residuals and preserves GPU-native execution:
  no CPU-WRF runtime dependency and no timestep-loop host/device transfer.
- If no source fix is made, the next truth surface/source line is exact enough
  for the next worker to start without rediscovery.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_JAX_START_DOMAIN_INPUT_SPLIT DONE - see proofs/v014/step1_jax_start_domain_input_split.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
