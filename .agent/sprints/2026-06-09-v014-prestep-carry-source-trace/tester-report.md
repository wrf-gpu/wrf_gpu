# Tester Report

Decision: pass.

Validation performed:

- Compiled the proof script with `python -m py_compile`.
- Reran the proof under `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src`.
- Validated the emitted JSON with `python -m json.tool`.
- Inspected the concise Markdown and review outputs.
- Checked the scoped git status for the four allowed deliverables.

Manager rerun commands:

```bash
python -m py_compile proofs/v014/prestep_carry_source_trace.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/prestep_carry_source_trace.py \
  > /tmp/prestep_carry_source_trace.manager.stdout \
  2> /tmp/prestep_carry_source_trace.manager.stderr
python -m json.tool proofs/v014/prestep_carry_source_trace.json \
  >/tmp/prestep_carry_source_trace.manager.validated.json
```

Rerun result:

- Verdict: `PRODUCER_WRITES_BAD_FINAL_CARRY`.
- Serialization verdict:
  `CHECKPOINT_READ_WRITE_PRESERVES_TARGET_LEAVES`.
- Exact serialization target-leaf checks: `true`.
- Field max_abs vs CPU-WRF pre-RK truth:
  - `T`: `6.218735851548047`
  - `P`: `589.6789731315657`
  - `PB`: `1047.015625`
  - `MU`: `267.01919069732367`
  - `MUB`: `1050.3046875`

Scope check:

The sprint produced only the allowed proof/review artifacts. Existing dirty or
untracked files elsewhere in the worktree were pre-existing and were not
touched for this validation.

Residual risk:

The proof is intentionally not a full-grid validation and not a source fix. Its
value is the narrowed next decision: inspect the previous-step producer handoff
path rather than checkpoint serialization or current-step RK/acoustic.
