You are GPT-5.5 xhigh acting as a verifier/debug worker for wrf_gpu2.

Repository: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`

Read and follow:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-prestep-carry-source-trace/sprint-contract.md`
4. Only the source/proof files needed for this sprint.

Task:

Trace the confirmed h10 `d02` pre-RK input-boundary mismatch back through the
JAX checkpoint/prestep carry producer and previous-step state handoff path.

Key facts:

- `proofs/v014/pre_rk_input_boundary.json` verdict is
  `PRE_RK_INPUT_JAX_PRESTEP_MISMATCH_CONFIRMED`.
- CPU-WRF pre-RK truth exists for d02 step 6000 over `T/P/PB/MU/MUB`.
- The produced JAX checkpoint is:
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`.
- All target fields already differ before current-step physics/RK, so do not
  start with current-step RK/acoustic or `small_step_finish`.

Deliver:

- `proofs/v014/prestep_carry_source_trace.py`
- `proofs/v014/prestep_carry_source_trace.json`
- `proofs/v014/prestep_carry_source_trace.md`
- `.agent/reviews/2026-06-09-v014-prestep-carry-source-trace.md`

Rules:

- Evidence sprint only. No production `src/` edits.
- No WRF source edits, no GPU, no TOST, no Switzerland, no FP32.
- CPU-only commands with `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`.
- Keep top-level output compact; do not dump huge arrays.
- If blocked, emit valid JSON with `TRACE_BLOCKED_<reason>` and the exact next
  artifact/API/command needed.

Required validation:

```bash
python -m py_compile proofs/v014/prestep_carry_source_trace.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/prestep_carry_source_trace.py
python -m json.tool proofs/v014/prestep_carry_source_trace.json \
  >/tmp/prestep_carry_source_trace.validated.json
```

When done, print:

`GPT PRESTEP_CARRY_SOURCE_TRACE DONE`
