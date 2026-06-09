You are GPT-5.5 xhigh acting as a verifier/debug worker for wrf_gpu2.

Repository: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`

Read and follow:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-earlier-source-bisect/sprint-contract.md`
4. Only the source/proof files needed for this sprint.

Task:

Bisect the source of the bad h10 d02 `OperationalCarry` before d02 completed
step 5997.

Starting facts:

- `proofs/v014/prestep_carry_source_trace.json` verdict is
  `PRODUCER_WRITES_BAD_FINAL_CARRY`.
- `proofs/v014/previous_step_handoff_bisect.json` verdict is
  `BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE`.
- The final producer-shaped replay exactly matches the existing bad checkpoint.
- The bad state is already present at d02 completed step 5997, before parent
  step 2000, `_operational_force`, and child steps 5998-5999.
- Do not debug `_operational_force`, final child `_advance_chunk`,
  current-step RK/acoustic, FP32, TOST, or Switzerland in this sprint.

Deliver:

- `proofs/v014/earlier_source_bisect.py`
- `proofs/v014/earlier_source_bisect.json`
- `proofs/v014/earlier_source_bisect.md`
- `.agent/reviews/2026-06-09-v014-earlier-source-bisect.md`

Rules:

- Evidence sprint only. No source fix.
- Default to no `src/` edits and no WRF source edits. If a hook is required,
  emit `EARLIER_SOURCE_BLOCKED_<reason>` with the exact hook/source file needed.
- CPU-only first. A short targeted GPU replay is allowed only if CPU replay is
  not practical; record the exact command, backend, allocator, and peak VRAM if
  used.
- No TOST, no Switzerland validation, no broad validation campaign, no FP32.
- No Hermes, no Telegram, no `ask-hermes`. If blocked, record the blocker in
  the sprint artifacts.
- Keep top-level output compact; no large array dumps.

Required validation:

```bash
python -m py_compile proofs/v014/earlier_source_bisect.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/earlier_source_bisect.py
python -m json.tool proofs/v014/earlier_source_bisect.json \
  >/tmp/earlier_source_bisect.validated.json
```

When done, print:

`GPT EARLIER_SOURCE_BISECT DONE`
