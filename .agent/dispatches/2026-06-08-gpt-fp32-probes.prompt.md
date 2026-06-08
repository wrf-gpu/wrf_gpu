# GPT-5.5 xhigh Dispatch: FP32 Acoustic Numerical Probes

You are a GPT-5.5 xhigh numerical-methods worker for `/home/enric/src/wrf_gpu2`.

Read in this order:
1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-08-v014-fp32-acoustic-derisk/sprint-contract.md`
4. `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`
5. `.agent/reviews/2026-06-08-gpt-fp32-acoustic-refresh.md`
6. `src/gpuwrf/dynamics/core/calc_p_rho.py`, `small_step_prep.py`, `small_step_finish.py`, `acoustic.py`, `advance_w.py`.

Do not consume the GPU unless the manager explicitly asks later. CPU only.

Task:
1. Create CPU-only proof probes under `proofs/v014/` that demonstrate the numerical mechanism:
   - absolute-total fp32 cancellation,
   - perturbation-form fp32 preservation,
   - one-column/small recurrence sensitivity if feasible.
2. Identify the first fp64 islands that must stay fp64 and the arrays that are plausible fp32 candidates.
3. Quantify memory-savings potential from resident acoustic/carry arrays with formulas and at least one concrete grid size.
4. State whether the evidence supports a v0.13 pull-in or v0.14-only.

Constraints:
- No source code changes outside `proofs/v014/` and report files.
- No post-hoc tolerance widening.
- No JAX-vs-JAX equivalence over production forecasts.

Deliverables:
- `proofs/v014/fp32_acoustic_probes.py`
- `proofs/v014/fp32_acoustic_probes.json`
- `.agent/reviews/2026-06-08-gpt-fp32-probes.md`
- commit your branch only if probes run
- final tmux line: `GPT FP32 PROBES DONE`
