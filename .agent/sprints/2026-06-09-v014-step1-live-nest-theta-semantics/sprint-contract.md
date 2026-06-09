# Sprint Contract: V0.14 Step-1 Live-Nest Theta Semantics

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Prove and, if proven, port the WRF live-nest `T_STATE`/theta initialization
semantics that follow terrain/base blending.

Trigger evidence:

- `proofs/v014/step1_jax_loader_tstate.json`
- verdict `STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`
- `T_STATE` max_abs versus WRF pre-call remains `5.490173101425171` for raw,
  live, boundary-packaged, carry, and haloed step-entry states
- all `T_STATE` stage transition max_abs values are `0.0`
- live-nest base init closes `PB` from raw max_abs `2627.3828125` to live
  max_abs `0.05357326504599769`, with `PHB/MUB` also closed

Source evidence to prove/falsify:

- WRF `share/mediation_integrate.F` live-nest input-file path blends terrain,
  `mub`, and `phb`, then calls
  `adjust_tempqv(nest%mub, nest%mub_save, nest%c3h, nest%c4h, nest%znw,
  nest%p_top, nest%t_2, nest%p, QVAPOR, use_theta_m, ...)`.
- WRF `dyn_em/nest_init_utils.F::adjust_tempqv` keeps relative humidity fixed
  across the pressure change and updates `th`/`t_2` and `qv`.
- Current JAX `src/gpuwrf/integration/d02_replay.py::_apply_live_nest_base_init`
  updates terrain and `PB/PHB/MUB` only; loader then keeps raw
  `wrfinput_d02` theta.

The sprint must determine whether a faithful port of `adjust_tempqv` closes the
WRF pre-call `T_STATE` residual and, if so, apply the smallest initialization-
only production fix.

## Method Rule

Use the fastest rigorous wall-clock method: proof-local candidate formulas
against the accepted WRF pre-call truth first. Do not run long validation and do
not use the GPU.

Required candidate checks:

1. Implement a proof-local transcription of WRF `adjust_tempqv` using:
   - `save_mub` = raw `wrfinput_d02` `MUB`;
   - `mub` = live-nest recomputed/blended `MUB`;
   - `pp` = raw `wrfinput_d02` perturbation pressure `P`;
   - `qv` = raw `wrfinput_d02` `QVAPOR`;
   - `th` = raw `wrfinput_d02` `T`;
   - `c3h/c4h`, `p_top`, and `use_theta_m` from the real run.
2. Compare candidate `T_STATE` and adjusted `QVAPOR` against WRF pre-call truth.
3. Resolve the state contract explicitly:
   - if WRF `t_2` stores moist theta (`use_theta_m=1`), prove whether
     operational `State.theta` should store WRF `t_2 + 300` or dry theta;
   - compare both candidate views if needed, but do not hide behind a naming
     ambiguity.
4. If a candidate closes `T_STATE`, patch production initialization only, with
   no timestep-loop transfer or hot-loop work.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No FP32 or mixed-precision source work.
- No memory source work beyond noting initialization-only allocation cost.
- No GPU.
- No Hermes or Telegram.
- No broad dycore/physics rewrite.
- No source patch unless the candidate proof closes the WRF pre-call residual.

## Inputs

- `proofs/v014/step1_jax_loader_tstate.py`
- `proofs/v014/step1_jax_loader_tstate.json`
- `proofs/v014/step1_jax_loader_tstate.md`
- `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`
- `proofs/v014/step1_pre_part1_handoff.py`
- `proofs/v014/step1_live_nest_init_rerun.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/nesting/interp.py`
- `src/gpuwrf/contracts/state.py`
- WRF source references:
  - `/home/enric/src/wrf_pristine/WRF/share/mediation_integrate.F`
  - `/home/enric/src/wrf_pristine/WRF/dyn_em/nest_init_utils.F`

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_live_nest_theta_semantics/**`

## Write Scope

Required repo files:

- `proofs/v014/step1_live_nest_theta_semantics.py`
- `proofs/v014/step1_live_nest_theta_semantics.json`
- `proofs/v014/step1_live_nest_theta_semantics.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-semantics.md`

Optional targeted source edits only if the candidate proof closes the residual:

- `src/gpuwrf/integration/d02_replay.py`
- narrowly required helper/tests in `src/gpuwrf/integration/**` only if
  `d02_replay.py` alone cannot express the initialization-only fix cleanly.

Do not touch unrelated source, TOST outputs, Switzerland outputs, FP32 work,
memory source work, or old untracked artifacts.

## Required Work

1. Verify branch/head and that `7ae33eda` is an ancestor.
2. Read the WRF `adjust_tempqv` code and record exact source lines in the proof.
3. Build proof-local candidate arrays under CPU-only JAX/NumPy.
4. Compare raw, current-live, and candidate adjusted fields against WRF pre-call
   truth for:
   - `T_STATE`;
   - `QVAPOR`;
   - `P_STATE`, `PB`, `MUB`, `PHB` for continuity.
5. Classify whether the WRF `adjust_tempqv` candidate closes `T_STATE`.
6. If closing, patch production initialization only and rerun:
   - `proofs/v014/step1_live_nest_theta_semantics.py`;
   - `proofs/v014/step1_jax_loader_tstate.py`;
   - `proofs/v014/step1_pre_part1_handoff.py`;
   - `proofs/v014/step1_part1_physics_state_mutation.py`;
   - `proofs/v014/step1_rk1_source_boundary.py`;
   - `proofs/v014/step1_t_p_operator_localization.py`;
   - `proofs/v014/step1_live_nest_init_rerun.py`;
   plus `python -m py_compile` for edited source.
7. If not closing, name the exact missing WRF call/source field and next proof
   boundary.

## Verdicts

Emit exactly one final verdict:

- `STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_PROVEN_SOURCE_FIX_READY`
- `STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_FIXED`
- `STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_PARTIAL_NEXT_<field_or_source>`
- `STEP1_LIVE_NEST_THETA_BLOCKED_<specific_missing_truth_or_contract>`
- `STEP1_LIVE_NEST_THETA_NO_REMAINING_DIVERGENCE`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_live_nest_theta_semantics.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_live_nest_theta_semantics.py
python -m json.tool proofs/v014/step1_live_nest_theta_semantics.json \
  >/tmp/step1_live_nest_theta_semantics.validated.json
git diff -- src/gpuwrf
```

If production source is edited, run all proof-chain commands listed in Required
Work item 6.

## Acceptance Criteria

- CPU-only proof records `gpu_used=false`.
- WRF source line evidence is included for `adjust_tempqv`.
- Candidate formula result is compared numerically against WRF pre-call truth.
- Any source patch is initialization-only and documented as outside the
  timestep loop.
- No validation campaign or station proxy is used as evidence.
