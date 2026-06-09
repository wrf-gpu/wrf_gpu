# Sprint Contract: V0.14 Prestep Carry Source Trace

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Trace the confirmed h10 `d02` pre-RK input-boundary mismatch back through the
JAX checkpoint/prestep carry producer and previous-step state handoff path.
Decide the next concrete fix target without editing production model source.

The starting fact is no longer "WRF pre-RK truth missing": it exists and proves
`T/P/PB/MU/MUB` are already wrong in the produced JAX step-5999 carry before
current-step physics/RK.

## Inputs

- `proofs/v014/pre_rk_input_boundary.json`
- `proofs/v014/pre_rk_input_boundary.md`
- `proofs/v014/jax_h10_prestep_carry_producer.py`
- `proofs/v014/jax_h10_prestep_carry_producer.json`
- `proofs/v014/jax_h10_prestep_carry.py`
- `proofs/v014/jax_h10_prestep_carry.json`
- `proofs/v014/jax_theta_evolution_localization.json`
- `proofs/v014/jax_t_history_source_attribution.json`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- Checkpoint:
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`

## Write Scope

Repository write scope:

- `proofs/v014/prestep_carry_source_trace.py`
- `proofs/v014/prestep_carry_source_trace.json`
- `proofs/v014/prestep_carry_source_trace.md`
- `.agent/reviews/2026-06-09-v014-prestep-carry-source-trace.md`

External scratch is allowed under:

- `/tmp/wrf_gpu2_v014_prestep_carry_source_trace/**`

No production `src/` edits. No WRF source edits. No GPU. No TOST. No
Switzerland validation. No FP32 source landing.

## Required Work

1. Inspect the checkpoint producer and provenance:
   - exact function that creates the step-5999 carry;
   - whether checkpoint write/read preserves `T/P/PB/MU/MUB` exactly;
   - whether the producer starts from a retained GPU wrfout, restart, live
     replay state, or generated `OperationalCarry`.
2. Build a compact CPU-only proof script that loads:
   - the produced checkpoint;
   - pre-RK WRF truth from `proofs/v014/pre_rk_input_boundary.json`;
   - producer provenance and prior h10 compare artifacts.
3. For each target field `T/P/PB/MU/MUB`, report:
   - checkpoint leaf/source expression used by existing compare scripts;
   - max_abs/RMSE vs WRF pre-RK truth;
   - whether any existing producer-side or retained JAX artifact in the proof
     corpus matches WRF more closely than the checkpoint leaf.
4. Classify the next fix target as exactly one of:
   - `CHECKPOINT_SERIALIZATION_BUG`
   - `PRODUCER_WRITES_BAD_FINAL_CARRY`
   - `PREVIOUS_STEP_FINAL_CARRY_ASSEMBLY_BUG`
   - `PREVIOUS_STEP_BOUNDARY_OR_TENDENCY_PACKAGING_BUG`
   - `EARLIER_INTEGRATION_DIVERGENCE`
   - `TRACE_BLOCKED_<reason>`
5. Do not patch the model. If the trace is blocked, name the exact missing
   artifact/API/hook and the next command needed.

## Commands / Validation

At minimum, run:

```bash
python -m py_compile proofs/v014/prestep_carry_source_trace.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/prestep_carry_source_trace.py
python -m json.tool proofs/v014/prestep_carry_source_trace.json \
  >/tmp/prestep_carry_source_trace.validated.json
```

## Acceptance Criteria

- No production source edits.
- JSON validates.
- The proof uses CPU-WRF pre-RK truth from `pre_rk_input_boundary`, not retained
  wrfout or JAX-vs-JAX self-comparison.
- The proof distinguishes serialization/load identity from bad in-memory carry
  production.
- The next decision is narrow enough to open a source-changing sprint or a
  smaller evidence sprint.

## Closeout

Close with verdict, files changed, commands run, proof objects, unresolved
risks, and next decision.
