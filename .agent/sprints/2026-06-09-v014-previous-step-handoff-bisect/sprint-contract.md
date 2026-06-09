# Sprint Contract: V0.14 Previous-Step Handoff Bisection

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Bisect the live nested replay producer path that writes the bad h10 d02
step-5999 `OperationalCarry`. The sprint must decide whether the first wrong
surface is already present before the final parent/child partial subcycle,
introduced by `_operational_force`, introduced by child `_advance_chunk`, or
blocked behind a missing deeper hook.

The starting fact is `proofs/v014/prestep_carry_source_trace.json` verdict
`PRODUCER_WRITES_BAD_FINAL_CARRY`: checkpoint serialization is exact, and the
persisted carry is bad before current-step physics/RK.

## Inputs

- `proofs/v014/prestep_carry_source_trace.json`
- `proofs/v014/prestep_carry_source_trace.md`
- `proofs/v014/pre_rk_input_boundary.json`
- `proofs/v014/jax_h10_prestep_carry_producer.py`
- `proofs/v014/jax_h10_prestep_carry_producer.json`
- `proofs/v014/jax_h10_prestep_carry.json`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- Checkpoint:
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`

## Write Scope

Repository write scope:

- `proofs/v014/previous_step_handoff_bisect.py`
- `proofs/v014/previous_step_handoff_bisect.json`
- `proofs/v014/previous_step_handoff_bisect.md`
- `.agent/reviews/2026-06-09-v014-previous-step-handoff-bisect.md`

External scratch write scope:

- `/mnt/data/wrf_gpu2/v014_previous_step_handoff_bisect/**`
- fallback `/tmp/wrf_gpu2_v014_previous_step_handoff_bisect/**`

Default rule: no production `src/` edits and no WRF source edits. If existing
private helpers cannot expose the needed surfaces, emit a blocked JSON naming
the exact hook and source file needed; do not add the hook in this sprint.

GPU use is allowed only for this targeted replay/snapshot probe if CPU replay is
not practical. Do not run TOST, Switzerland validation, broad validation
campaigns, or FP32 source work.

Do not use Hermes, Telegram, `ask-hermes`, or any human-notification bridge in
this sprint. If blocked, write the blocked verdict and exact next action into
the JSON/Markdown/review artifacts.

## Required Work

1. Reproduce the h10 producer path as closely as possible:
   - native L2 domain load;
   - `DomainTree.from_domains(..., feedback_enabled=False)`;
   - segment replay through d01/d02 own steps;
   - final partial parent subcycle at parent step 2000 / d02 steps 5998-5999.
2. Capture compact d02 target-leaf snapshots at these surfaces where reachable:
   - after segment replay at d02 completed step 5997, before the final partial
     parent step;
   - before and after parent d01 step 2000 if it affects child forcing;
   - before and after `_operational_force`;
   - after child `_advance_chunk` step 5998;
   - after child `_advance_chunk` step 5999, before checkpoint write;
   - if reachable without source edits, final RK3 pre-halo state and carry
     scratch leaves (`t_2ave`, `t_save`, `mu_save`, `muts`) for the final child
     step.
3. For each snapshot, emit context-sparing statistics over the same WRF pre-RK
   h10 d02 patch for `T/P/PB/MU/MUB` using the same source expressions as
   `prestep_carry_source_trace`.
4. Prove whether the final reproduced d02 step-5999 snapshot matches the
   existing checkpoint target leaves exactly. If it does not, classify the probe
   as non-reproducing and report the first divergence from producer provenance.
5. Classify the next fix target as exactly one of:
   - `BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE`
   - `BAD_AFTER_PARENT_ADVANCE`
   - `BAD_AFTER_OPERATIONAL_FORCE`
   - `BAD_AFTER_CHILD_ADVANCE_STEP_5998`
   - `BAD_AFTER_CHILD_ADVANCE_STEP_5999`
   - `REPRODUCER_MISMATCH_<reason>`
   - `BISECTION_BLOCKED_<reason>`
6. State the next decision narrowly:
   - source-changing fix sprint target;
   - narrower WRF/JAX savepoint hook sprint;
   - or escalation after repeated failed attempts.

## Commands / Validation

At minimum, run:

```bash
python -m py_compile proofs/v014/previous_step_handoff_bisect.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/previous_step_handoff_bisect.py
python -m json.tool proofs/v014/previous_step_handoff_bisect.json \
  >/tmp/previous_step_handoff_bisect.validated.json
```

If GPU replay is required, record the exact command and environment in JSON,
including `CUDA_VISIBLE_DEVICES`, JAX backend, allocator settings, peak VRAM if
available, and why CPU replay was not practical.

## Acceptance Criteria

- JSON validates and top-level output is compact.
- No source fix is landed.
- The proof uses CPU-WRF pre-RK truth from `pre_rk_input_boundary` for the h10
  final comparison.
- The proof distinguishes producer replay mismatch from a real producer-path
  bisection result.
- If blocked, the JSON names the exact missing artifact/API/hook and the next
  command needed.
- The next decision is specific enough to open either a source-changing fix
  sprint or a smaller hook/savepoint sprint.

## Closeout

Close with verdict, files changed, commands run, proof objects, unresolved
risks, GPU use if any, and next decision.
