# Sprint Contract: V0.14 Step-1 Start-Domain Perturbation Subsurface

Date: 2026-06-09 19:45 WEST
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`
Base commit: `ee6cbbe1`

## Objective

Close the missing WRF live-nest `start_domain(nest,.TRUE.)` internal truth
surface needed to patch `P_STATE/MU_STATE/W_STATE` initialization safely.

Accepted predecessor:

- Commit `f73542c0`: `v014 close live nest perturb state init`.
- Verdict:
  `STEP1_LIVE_NEST_PERTURB_STATE_LOCALIZED_START_DOMAIN_P_PRESS_ADJ_SET_W_SURFACE_P_AL_ALT_SUBSURFACE_GAP`.
- Proof-local transcriptions reduce residuals:
  - `P_STATE`: `69.96875` -> `3.9458582235092763` Pa.
  - `MU_STATE`: `13.256103515625` -> `0.047773029698646496` Pa.
  - `W_STATE`: `0.7605466246604919` -> `1.2992081932505783e-07` m/s.
- No production patch is safe yet because `P_STATE` still needs the exact WRF
  `al/alt` and pre/post-`press_adj` ordering/surface.

## Method Rule

Use the fastest rigorous wall-clock path: disposable WRF instrumentation plus
CPU-only proof replay. Do not keep iterating broad full forecasts.

The first hypothesis is that one exact WRF `start_domain` internal surface will
close the remaining `P_STATE` and confirm the `MU_STATE` sequencing. If that is
wrong, do not stop at "blocked": report ranked alternate hypotheses, try cheap
proof-local falsifiers, and name the next minimal truth surface.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No GPU run.
- No FP32 or mixed-precision source work.
- No memory source work.
- No Hermes/Telegram.
- No broad dycore rewrite, CPU-WRF runtime dependency in production, or
  timestep-loop host/device transfer.

## Inputs

- `proofs/v014/step1_live_nest_perturb_state_init.py`
- `proofs/v014/step1_live_nest_perturb_state_init.json`
- `proofs/v014/step1_live_nest_perturb_state_init.md`
- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth/**`
- WRF source under `/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF`

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/**`

## File Ownership

Required repo files:

- `proofs/v014/step1_start_domain_perturb_subsurface.py`
- `proofs/v014/step1_start_domain_perturb_subsurface.json`
- `proofs/v014/step1_start_domain_perturb_subsurface.md`
- `proofs/v014/step1_start_domain_perturb_subsurface_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-start-domain-perturb-subsurface.md`

Optional production source edit only if exact and narrow:

- `src/gpuwrf/integration/d02_replay.py`

Do not edit `src/gpuwrf/dynamics/**`, `src/gpuwrf/runtime/**`,
`src/gpuwrf/contracts/**`, FP32 files, memory files, TOST outputs, Switzerland
outputs, or unrelated dirty/untracked artifacts.

## Required Work

1. Verify `ee6cbbe1` is an ancestor and record branch/head.
2. Add a disposable WRF hook or patch against the local WRF copy, not production
   `src/gpuwrf`, to emit d02 step-1 live-nest `start_domain` internal surfaces:
   - after hypsometric `P/al/alt` recompute;
   - immediately before `press_adj`;
   - immediately after `press_adj`;
   - before `first_rk_step_part1_call` if not already reusing accepted truth.
3. Required emitted fields, compact or full-field:
   - `P_STATE`, `MU_STATE`, `W_STATE`, `PH_STATE`;
   - `al`, `alt`, `alb`;
   - `PB`, `MUB`, `PHB`;
   - theta/full theta, QVAPOR;
   - `HT`, `HT_FINE`, and any terrain delta term used by `press_adj`.
4. Compare the WRF internal surfaces to:
   - WRF pre-call truth;
   - current JAX live-nest Step-1 loader values;
   - proof-local formula transcriptions from
     `step1_live_nest_perturb_state_init.py`.
5. If exact source semantics are now proven, optionally apply the smallest
   GPU-native `d02_replay.py` source patch and rerun the strict Step-1 proof.
6. If exact closure is still not proven, name the exact next surface or source
   line and explain why patching would still be a guess.

## Verdicts

Emit exactly one final verdict:

- `STEP1_START_DOMAIN_PERTURB_SUBSURFACE_FIXED_<source_or_formula>`
- `STEP1_START_DOMAIN_PERTURB_SUBSURFACE_READY_FOR_PATCH_<contract>`
- `STEP1_START_DOMAIN_PERTURB_SUBSURFACE_LOCALIZED_<source_or_missing_contract>`
- `STEP1_START_DOMAIN_PERTURB_SUBSURFACE_BLOCKED_<specific_truth_or_source_gap>`
- `STEP1_START_DOMAIN_PERTURB_SUBSURFACE_REFUTED_<hypothesis_and_next_best>`

## Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_start_domain_perturb_subsurface.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_start_domain_perturb_subsurface.py
python -m json.tool proofs/v014/step1_start_domain_perturb_subsurface.json \
  >/tmp/step1_start_domain_perturb_subsurface.validated.json
git diff -- src/gpuwrf
```

If production source changes:

```bash
python -m py_compile src/gpuwrf/integration/d02_replay.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_live_nest_perturb_state_init.py
python -m json.tool proofs/v014/step1_live_nest_perturb_state_init.json \
  >/tmp/step1_live_nest_perturb_state_init.after_start_domain_patch.validated.json
```

## Acceptance Criteria

- JSON validates and records CPU-only execution.
- The WRF patch diff is stored, even if no production source changes.
- The report lists ranked hypotheses, exclusions, and the next fastest rigorous
  path.
- Any production source fix has before/after residuals and preserves GPU-native
  execution: no CPU-WRF runtime dependency and no timestep-loop host/device
  transfer.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_START_DOMAIN_PERTURB_SUBSURFACE DONE - see proofs/v014/step1_start_domain_perturb_subsurface.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
