# Review: V0.14 Prestep Carry Source Trace

verdict: `PRODUCER_WRITES_BAD_FINAL_CARRY`

objective: trace the confirmed h10 d02 pre-RK input-boundary mismatch through the JAX checkpoint/prestep carry producer and previous-step handoff path without production source edits.

files changed:
- `proofs/v014/prestep_carry_source_trace.py`
- `proofs/v014/prestep_carry_source_trace.json`
- `proofs/v014/prestep_carry_source_trace.md`
- `.agent/reviews/2026-06-09-v014-prestep-carry-source-trace.md`

commands run:
- `python -m py_compile proofs/v014/prestep_carry_source_trace.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/prestep_carry_source_trace.py`
- `python -m json.tool proofs/v014/prestep_carry_source_trace.json >/tmp/prestep_carry_source_trace.validated.json`

proof objects produced:
- `proofs/v014/prestep_carry_source_trace.json`
- `proofs/v014/prestep_carry_source_trace.md`
- `.agent/reviews/2026-06-09-v014-prestep-carry-source-trace.md`
- `/tmp/wrf_gpu2_v014_prestep_carry_source_trace/d02_step5999_roundtrip.pkl`

unresolved risks:
- This proves the persisted producer carry is bad; it does not split the deeper cause between previous-step final carry assembly, parent/child force packaging, and earlier step integration because no d02 step-5997/5998/5999 in-memory snapshots are present.
- Only the selected pre-RK h10 d02 mass patch was compared.
- The original checkpoint was produced in a prior GPU-enabled producer run; this trace only loads and round-trips it CPU-side.

next decision needed: Open a narrow previous-step handoff bisection sprint: capture CPU-only JAX d02 carries at steps 5997, 5998, and 5999 immediately before/after _operational_force, _advance_chunk, and _carry_from_finished_stage, then compare State theta/p/mu target leaves plus t_2ave/t_save/mu_save/muts against the existing CPU-WRF pre-RK truth. Do not edit current-step RK/acoustic first.
