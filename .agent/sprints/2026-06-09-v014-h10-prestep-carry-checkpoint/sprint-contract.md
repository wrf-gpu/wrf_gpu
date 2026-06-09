# Sprint Contract: V0.14 H10 Pre-Step Carry Checkpoint

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Build or locate a CPU-loadable JAX `OperationalCarry` immediately before
`d02` step 6000 / h10, then run
`_rk_scan_step_with_pre_halo_capture` and compare its captured pre-halo state
against Boole's green WRF target:
`post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges`.

This sprint should name the first same-surface JAX-vs-WRF mismatch if the
checkpoint can be built. It must not launch a numerical source fix.

## Inputs

- `proofs/v014/jax_pre_halo_capture.json`
- `proofs/v014/wrf_post_rk_refresh_localization.json`
- `proofs/v014/wrf_post_rk_refresh_localization.md`
- `proofs/v014/same_state_savepoint_request.json`
- `proofs/v014/dynamic_field_attribution.json`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

## Write Scope

Repository write scope:

- `proofs/v014/jax_h10_prestep_carry.py`
- `proofs/v014/jax_h10_prestep_carry.json`
- `proofs/v014/jax_h10_prestep_carry.md`
- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-checkpoint.md`

No production `src/` edits unless the manager opens a separate contract. No WRF
source edits. No TOST. No Switzerland validation. No FP32 source landing. GPU is
not allowed unless the manager explicitly approves a short run after CPU path
assessment; default is CPU-only.

## Required Work

1. Inspect the current replay/operational APIs for any existing checkpoint or
   segmented-carry path that can produce `OperationalCarry` before `d02` step
   6000. Include `runtime/operational_state.py`, `runtime/operational_mode.py`,
   `integration/d02_replay.py`, checkpoint utilities, and prior proof scripts.
2. Prefer a CPU-only proof script that replays/segments from available initial
   state to the target pre-step carry, then calls
   `_rk_scan_step_with_pre_halo_capture`.
3. If a full CPU h10 replay is too expensive, test the smallest resumable
   checkpoint path that can prove the exact missing pieces and emit a compact
   `CHECKPOINT_BLOCKED_<reason>` verdict. Do not compare retained wrfout as a
   substitute for same-surface JAX internals.
4. Compare captured JAX `T/P/PB/U/V/W/PH/MU/MUB` patch values against
   `proofs/v014/wrf_post_rk_refresh_localization.json`.
5. Produce one of:
   - `JAX_MISMATCH_<field_or_operator>` with max_abs/RMSE and first mismatch;
   - `JAX_SURFACE_MATCH_after_all_rk_pre_halo`;
   - `CHECKPOINT_BLOCKED_<reason>` with exact missing input/API and next command.

## Commands / Validation

At minimum, run:

```bash
python -m py_compile proofs/v014/jax_h10_prestep_carry.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/jax_h10_prestep_carry.py
python -m json.tool proofs/v014/jax_h10_prestep_carry.json \
  >/tmp/jax_h10_prestep_carry.validated.json
```

Run additional focused tests only if the proof script introduces helper code.

## Acceptance Criteria

- No production source edits.
- No GPU/TOST/Switzerland/FP32 unless explicitly approved later by the manager.
- JSON validates and records whether a real pre-step carry was built or why it
  could not be built.
- If a comparison runs, it uses the hook-captured JAX pre-halo state and Boole's
  WRF green target, not retained wrfout or JAX-vs-JAX evidence.
- The next decision is explicit: source-fix sprint, narrower checkpoint sprint,
  or escalation after repeated failure.

## Closeout

Close with verdict, files changed, commands run, proof objects, unresolved
risks, and next decision.
