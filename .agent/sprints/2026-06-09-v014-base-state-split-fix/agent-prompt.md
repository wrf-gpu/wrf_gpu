You are GPT-5.5 xhigh acting as a source-fix worker for wrf_gpu2.

Repository: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`

Read and follow:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-base-state-split-fix/sprint-contract.md`
4. Only the source/proof files needed for this sprint.

Task:

Fix or precisely block the native live-nested d02 base-state split mismatch.

Starting facts:

- `proofs/v014/earlier_source_bisect.json` verdict is
  `BASE_STATE_SPLIT_DEFINITION_MISMATCH`.
- Initial JAX d02 carry matches native `wrfinput_d02` for `PB/MUB`.
- CPU-WRF h0/h1/h10 and h10 pre-RK truth use a stable different `PB/MUB` split.
- The target source is `src/gpuwrf/integration/d02_replay.py::build_replay_case`.
- Do not debug final child advance, `_operational_force`, current-step
  RK/acoustic, FP32, TOST, Switzerland, or memory cleanup in this sprint.

Deliver:

- source patch if local and justified, primarily in
  `src/gpuwrf/integration/d02_replay.py`
- `proofs/v014/base_state_split_fix.py`
- `proofs/v014/base_state_split_fix.json`
- `proofs/v014/base_state_split_fix.md`
- `.agent/reviews/2026-06-09-v014-base-state-split-fix.md`

Rules:

- Source-changing sprint, but keep the patch narrow.
- Do not hide a normal production dependency on CPU-WRF `wrfout` history. If an
  oracle/h0 fallback is used, label it validation-only or fail-closed.
- If the WRF transformation is not local/clear, emit
  `BASE_STATE_SPLIT_FIX_BLOCKED_<reason>` with exact WRF routine/formula/hook
  needed.
- CPU-only first. Targeted GPU replay is allowed only if CPU replay is not
  practical; record backend, allocator, command, and peak VRAM.
- No Hermes, no Telegram, no `ask-hermes`.
- Keep top-level output compact; put field tables in JSON.

Required validation:

```bash
python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/base_state_split_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/base_state_split_fix.py
python -m json.tool proofs/v014/base_state_split_fix.json \
  >/tmp/base_state_split_fix.validated.json
```

When done, print:

`GPT BASE_STATE_SPLIT_FIX DONE`
