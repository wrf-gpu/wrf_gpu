# Sprint Contract: V0.14 Pre-RK Input Boundary

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Produce explicit WRF and JAX step-6000 pre-RK input-boundary truth for h10
`d02` over `T/P/PB/MU/MUB`, then decide whether the produced JAX step-5999
prestep carry is already wrong before current-step physics/RK.

This is an evidence sprint, not a production fix sprint.

## Inputs

- `proofs/v014/jax_theta_evolution_localization.json`
- `proofs/v014/jax_h10_prestep_carry.json`
- `proofs/v014/jax_h10_prestep_carry_producer.json`
- `proofs/v014/wrf_same_state_marker_savepoint.json`
- `proofs/v014/wrf_same_state_marker_patch.diff`
- `proofs/v014/wrf_dynamic_term_localization_patch.diff`
- `proofs/v014/wrf_post_rk_refresh_localization_patch.diff`
- `proofs/v014/same_state_savepoint_request.json`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- Checkpoint:
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`
- CPU-WRF scratch/source candidates:
  `/home/enric/src/wrf_pristine/WRF`
  and prior scratch under `/mnt/data/wrf_gpu2/v014_same_state_wrf/` if present

## Write Scope

Repository write scope:

- `proofs/v014/pre_rk_input_boundary.py`
- `proofs/v014/pre_rk_input_boundary.json`
- `proofs/v014/pre_rk_input_boundary.md`
- `proofs/v014/pre_rk_input_boundary_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-pre-rk-input-boundary.md`

External scratch write scope:

- `/mnt/data/wrf_gpu2/v014_pre_rk_input_boundary/**`
- fallback `/tmp/wrf_gpu2_v014_pre_rk_input_boundary/**`

No production `src/` edits. No in-place WRF source edits. No GPU. No TOST. No
Switzerland run. No FP32 source landing.

## Required Work

1. Inspect the previous WRF marker/dynamic/refresh patch diffs and identify the
   earliest source location in the CPU-WRF step-6000 path that can emit explicit
   pre-RK input-boundary values for `T/P/PB/MU/MUB` on the same h10 `d02`
   selected patch.
2. Copy or reuse only a disposable WRF scratch tree. Add env-gated marker hooks
   there and preserve the change as
   `proofs/v014/pre_rk_input_boundary_wrf_patch.diff`.
3. Run the minimal CPU-WRF command needed to emit the h10 `d02` pre-RK boundary.
   Reuse prior built CPU-WRF artifacts if possible. Do not use GPU.
4. Load the produced JAX step-5999 h10 prestep carry checkpoint CPU-only and
   extract same-patch JAX `T/P/PB/MU/MUB` input-boundary candidates with explicit
   offset and base/perturbation conventions.
5. Compare WRF and JAX on the same native indices and record shape, dtype,
   max_abs, RMSE, worst index, and unavailable fields.
6. Include provenance for checkpoint hash/size, WRF executable/source path,
   marker environment, output paths, and exact commands.
7. Produce one of these verdicts:
   - `PRE_RK_INPUT_JAX_PRESTEP_MISMATCH_CONFIRMED` if explicit WRF pre-RK
     `T/P/PB/MU/MUB` differs from JAX prestep carry beyond frozen tolerance;
   - `PRE_RK_INPUT_MATCHES_REFERENCE_MISMATCH_WAS_SURFACE_MISMATCH` if explicit
     pre-RK WRF and JAX match and the previous `T_OLD` comparison was not the
     same surface;
   - `PRE_RK_INPUT_PARTIAL_CONTEXT_MISMATCH` if only a subset of fields differ
     or a field is unavailable but the next source boundary is clear;
   - `PRE_RK_INPUT_BOUNDARY_BLOCKED_<reason>` if the required WRF/JAX boundary
     cannot be emitted or compared. Name the exact missing hook or artifact.
8. State the next decision narrowly:
   - trace JAX checkpoint/prestep carry producer;
   - trace previous-step WRF/JAX update;
   - return to current-step RK/acoustic localization;
   - or open a source-changing fix sprint only if the evidence is sufficient.

## Commands / Validation

At minimum, run:

```bash
python -m py_compile proofs/v014/pre_rk_input_boundary.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/pre_rk_input_boundary.py
python -m json.tool proofs/v014/pre_rk_input_boundary.json \
  >/tmp/pre_rk_input_boundary.validated.json
```

If the WRF run is too expensive or blocked, the proof script must still emit a
valid blocked JSON that points to the exact missing command/artifact/hook.

## Acceptance Criteria

- No production source edits.
- No GPU/TOST/Switzerland/FP32.
- JSON validates and records WRF/JAX boundary provenance.
- The proof compares explicit pre-RK `T/P/PB/MU/MUB`, or explains exactly why
  any field remains unavailable.
- The next decision is specific enough for one source-changing, hook, or
  previous-step attribution sprint.

## Closeout

Close with verdict, files changed, commands run, proof objects, unresolved
risks, and next decision.
