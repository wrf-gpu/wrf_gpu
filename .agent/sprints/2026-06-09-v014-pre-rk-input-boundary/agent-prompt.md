You are GPT-5.5 xhigh acting as a verifier/debug worker for wrf_gpu2.

Repository: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`

Read and follow:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-pre-rk-input-boundary/sprint-contract.md`
4. Only the source/proof files needed for this sprint.

Task:

Produce explicit WRF and JAX step-6000 pre-RK input-boundary truth for h10 `d02`
over `T/P/PB/MU/MUB`, then decide whether the produced JAX step-5999 prestep
carry is already wrong before current-step physics/RK.

This is an evidence sprint, not a production fix sprint. Do not edit production
`src/`, do not edit WRF in place, do not use the GPU, do not run TOST, do not
run Switzerland validation, and do not land FP32 source work.

Key context:

- `proofs/v014/jax_theta_evolution_localization.json` verdict is
  `THETA_MISMATCH_PRESTEP_OR_INPUT`.
- The next decision from that proof is:
  open a WRF/JAX input-boundary emitter or hook sprint for explicit step-6000
  pre-RK `T/P/PB/MU/MUB` before any source-changing fix.
- The produced JAX h10 prestep carry is:
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`

Deliver:

- `proofs/v014/pre_rk_input_boundary.py`
- `proofs/v014/pre_rk_input_boundary.json`
- `proofs/v014/pre_rk_input_boundary.md`
- `proofs/v014/pre_rk_input_boundary_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-pre-rk-input-boundary.md`

Required validation:

```bash
python -m py_compile proofs/v014/pre_rk_input_boundary.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/pre_rk_input_boundary.py
python -m json.tool proofs/v014/pre_rk_input_boundary.json \
  >/tmp/pre_rk_input_boundary.validated.json
```

If the WRF run is blocked, emit valid blocked JSON and name the exact missing
hook/artifact/command.

When done, print:

`GPT PRE_RK_INPUT_BOUNDARY DONE`
