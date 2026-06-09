# Worker Report

Summary:

The GPT worker completed the V0.14 prestep carry source trace. It created the
proof script, JSON proof, Markdown summary, and review handoff within the
allowed write scope. No production `src/` files, WRF source files, GPU
validation runs, Switzerland runs, TOST runs, or FP32 source changes were made.

Files changed:

- `proofs/v014/prestep_carry_source_trace.py`
- `proofs/v014/prestep_carry_source_trace.json`
- `proofs/v014/prestep_carry_source_trace.md`
- `.agent/reviews/2026-06-09-v014-prestep-carry-source-trace.md`

Commands run:

- `python -m py_compile proofs/v014/prestep_carry_source_trace.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/prestep_carry_source_trace.py`
- `python -m json.tool proofs/v014/prestep_carry_source_trace.json >/tmp/prestep_carry_source_trace.validated.json`

Proof objects produced:

- `proofs/v014/prestep_carry_source_trace.json`
- `proofs/v014/prestep_carry_source_trace.md`
- `.agent/reviews/2026-06-09-v014-prestep-carry-source-trace.md`
- `/tmp/wrf_gpu2_v014_prestep_carry_source_trace/d02_step5999_roundtrip.pkl`

Accepted verdict:

`PRODUCER_WRITES_BAD_FINAL_CARRY`.

Key finding:

Checkpoint serialization and load/readback preserve the target leaves exactly.
The bad `T/P/PB/MU/MUB` values are already in the live nested replay
`OperationalCarry` that the producer writes with
`write_checkpoint(..., runtime_state=d02_carry)`.

Unresolved risks:

- The proof covers the selected h10 d02 pre-RK mass patch, not the full grid.
- It proves the persisted producer carry is bad, but does not yet split the
  deeper cause between previous-step final carry assembly, parent/child force
  packaging, and earlier integration.
- The original checkpoint was produced in a prior GPU-enabled producer run; the
  trace itself only loads and round-trips it CPU-side.

Next decision needed:

Open a previous-step handoff bisection sprint around the producer path: capture
d02 states/carries near steps 5997, 5998, and 5999 before/after parent force and
child advance, then compare the final step-5999 surfaces against the existing
CPU-WRF pre-RK truth.
