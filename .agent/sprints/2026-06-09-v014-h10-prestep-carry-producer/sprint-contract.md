# Sprint Contract: V0.14 H10 Pre-Step Carry Producer

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Produce a full JAX `OperationalCarry` checkpoint for `d02` completed step 5999,
immediately before step 6000/h10, then rerun the pre-halo hook comparison
against Boole's green WRF target if the checkpoint is produced.

This sprint should use existing carry/checkpoint APIs if possible. It must not
start a numerical source fix.

## Inputs

- `proofs/v014/jax_h10_prestep_carry.json`
- `proofs/v014/jax_pre_halo_capture.json`
- `proofs/v014/wrf_post_rk_refresh_localization.json`
- `proofs/v014/same_state_savepoint_request.json`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

## Write Scope

Repository write scope:

- `proofs/v014/jax_h10_prestep_carry_producer.py`
- `proofs/v014/jax_h10_prestep_carry_producer.json`
- `proofs/v014/jax_h10_prestep_carry_producer.md`
- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-producer.md`

External artifact write scope:

- `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/**`
- `/tmp/wrf_gpu2_v014_h10_prestep_carry/**` as fallback

No production `src/` edits. No WRF source edits. No TOST. No Switzerland
validation. No FP32 source landing.

GPU policy: default CPU-only. A short GPU checkpoint-producer run is acceptable
only if the script records the backend/device, writes a CPU-loadable host
checkpoint, and does not run TOST or validation campaigns. If GPU is used, keep
it to the minimal producer needed for this proof.

## Required Work

1. Inspect the existing producer options:
   - `runtime.checkpoint.write_checkpoint(..., runtime_state=carry)`;
   - `io.restart.write_restart` / carry restart helpers if paired namelist/grid
     can be preserved;
   - operational segmented/chunked forecast private helpers;
   - d02/l2 replay or powered-TOST case setup code for constructing the real
     d02 state/namelist/boundary leaves.
2. Build the shortest producer that can create a CPU-loadable checkpoint at
   completed step 5999 for the same `d02` case/valid time as the WRF truth.
3. If the checkpoint is produced, run:
   `WRFGPU2_H10_PRESTEP_CARRY=<checkpoint> python proofs/v014/jax_h10_prestep_carry.py`
   and include the resulting verdict in this sprint's JSON/MD.
4. If the producer cannot run, emit `PRODUCER_BLOCKED_<reason>` with the exact
   missing source API, input artifact, or command.
5. If the same-surface comparison runs and finds a mismatch, emit
   `JAX_MISMATCH_<field_or_operator>` with compact max_abs/RMSE and first
   mismatch. Do not fix it in this sprint.

## Commands / Validation

At minimum, run:

```bash
python -m py_compile proofs/v014/jax_h10_prestep_carry_producer.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/jax_h10_prestep_carry_producer.py
python -m json.tool proofs/v014/jax_h10_prestep_carry_producer.json \
  >/tmp/jax_h10_prestep_carry_producer.validated.json
```

If GPU is used for the producer, record the exact command, `nvidia-smi` before
and after, peak/observed VRAM if available, and why CPU-only was insufficient.

## Acceptance Criteria

- No production source edits.
- JSON validates and records whether the full carry checkpoint was produced.
- Any produced checkpoint includes full `OperationalCarry`, paired
  `OperationalNamelist`/grid, completed step 5999, and can be loaded by
  `proofs/v014/jax_h10_prestep_carry.py`.
- If a comparison runs, it uses the hook-captured JAX pre-halo state and Boole's
  WRF green target.
- The next decision is explicit: source-fix sprint, narrower producer sprint, or
  escalation after repeated failure.

## Closeout

Close with verdict, files changed, commands run, proof objects, checkpoint path
if produced, unresolved risks, and next decision.
