You are GPT-5.5 xhigh acting as a verifier/debug worker for wrf_gpu2.

Repository: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`

Read and follow:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-previous-step-handoff-bisect/sprint-contract.md`
4. Only the source/proof files needed for this sprint.

Task:

Bisect the live nested replay producer path that writes the bad h10 d02
step-5999 `OperationalCarry`.

Starting facts:

- `proofs/v014/prestep_carry_source_trace.json` verdict is
  `PRODUCER_WRITES_BAD_FINAL_CARRY`.
- Checkpoint serialization/load/readback preserves `T/P/PB/MU/MUB` exactly.
- The bad target leaves are already in the produced `OperationalCarry`.
- Do not debug current-step RK/acoustic, `small_step_finish`, post-RK refresh,
  history-source remapping, FP32, TOST, or Switzerland in this sprint.

Deliver:

- `proofs/v014/previous_step_handoff_bisect.py`
- `proofs/v014/previous_step_handoff_bisect.json`
- `proofs/v014/previous_step_handoff_bisect.md`
- `.agent/reviews/2026-06-09-v014-previous-step-handoff-bisect.md`

Rules:

- Evidence sprint only. No source fix.
- Default to no `src/` edits and no WRF source edits. If a hook is required,
  emit `BISECTION_BLOCKED_<reason>` with the exact hook/source file needed.
- CPU-only first. A short targeted GPU replay is allowed only if CPU replay is
  not practical; record the exact command, backend, allocator, and peak VRAM if
  used.
- No TOST, no Switzerland validation, no broad validation campaign, no FP32.
- No Hermes, no Telegram, no `ask-hermes`. If blocked, record the blocker in
  the sprint artifacts.
- Keep top-level output compact; no large array dumps.

Required validation:

```bash
python -m py_compile proofs/v014/previous_step_handoff_bisect.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/previous_step_handoff_bisect.py
python -m json.tool proofs/v014/previous_step_handoff_bisect.json \
  >/tmp/previous_step_handoff_bisect.validated.json
```

When done, print:

`GPT PREVIOUS_STEP_HANDOFF_BISECT DONE`
