You are GPT-5.5 xhigh, debug worker for wrf_gpu2 v0.14.

Read and obey:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-step1-jax-start-domain-input-split/sprint-contract.md`
4. `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

Task:

Close sprint
`.agent/sprints/2026-06-09-v014-step1-jax-start-domain-input-split`.

Current evidence:

- `proofs/v014/step1_start_domain_perturb_subsurface.json`
- Verdict:
  `STEP1_START_DOMAIN_PERTURB_SUBSURFACE_LOCALIZED_CURRENT_JAX_AL_ALT_BASE_INPUT_GAP`
- WRF source ordering is now proven:
  - WRF P from internal ALT vs after-hypsometric P max_abs `0.015625`.
  - WRF `press_adj` vs after-press MU max_abs `4.547473508864641e-13`.
  - WRF after-W branch vs accepted pre-call W max_abs `5.960464477539063e-08`.
- Patch-now with current JAX inputs is refuted:
  - current JAX pressure formula vs WRF after-hypsometric P max_abs
    `3.9458582235092763` Pa.
  - current JAX press_adj formula vs WRF after-press MU max_abs
    `0.047773029698646496` Pa.

Manager hypothesis:

The remaining gap is in current JAX live-nest start-domain input construction
for one of: final blended terrain, `PB/MUB/PHB`, `PH_STATE`, pre-press `MU`,
or diagnosed `AL/ALT`. Split it precisely. Patch only if exact and narrow.

Important:

If this hypothesis is wrong, do not stop at "blocked"; use your context to rank
alternate causes inside this boundary, try cheap proof-local falsifiers, and
report what you excluded.

Constraints:

- No TOST, Switzerland, FP32 source work, memory source work, Hermes, or GPU
  run.
- Prefer CPU-only proof replay against the existing WRF truth surfaces.
- Do not edit unrelated files.
- Production `src/gpuwrf/integration/d02_replay.py` edit is allowed only if the
  exact input/source bug is proven and the fix is narrow/GPU-native.

Required outputs:

- `proofs/v014/step1_jax_start_domain_input_split.py`
- `proofs/v014/step1_jax_start_domain_input_split.json`
- `proofs/v014/step1_jax_start_domain_input_split.md`
- `.agent/reviews/2026-06-09-v014-step1-jax-start-domain-input-split.md`

Required validation:

```bash
python -m py_compile proofs/v014/step1_jax_start_domain_input_split.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_jax_start_domain_input_split.py
python -m json.tool proofs/v014/step1_jax_start_domain_input_split.json \
  >/tmp/step1_jax_start_domain_input_split.validated.json
git diff -- src/gpuwrf
```

If production source changes, also run the source validation commands listed in
the sprint contract.

At the end, print a concise handoff with objective, files changed, commands
run, proof objects, ranked hypotheses/exclusions, unresolved risks, and next
decision. Then notify manager:

```bash
tmux send-keys -t 0:2 'GPT STEP1_JAX_START_DOMAIN_INPUT_SPLIT DONE - see proofs/v014/step1_jax_start_domain_input_split.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
