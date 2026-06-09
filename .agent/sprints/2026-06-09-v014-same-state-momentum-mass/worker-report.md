# Worker Report

Summary:

GPT produced a CPU-only same-state comparison at the nearest named WRF target
surface, `post_after_all_rk_steps_pre_halo`, for d02 step 6000 / h10. The proof
uses the existing WRF text-surface oracle and a proof-local JAX wrapper over the
selected h10 checkpoint/cells. It does not edit production `src/` code and does
not use the GPU.

Files Changed:

- `proofs/v014/same_state_momentum_mass.py`
- `proofs/v014/same_state_momentum_mass.json`
- `proofs/v014/same_state_momentum_mass.md`
- `.agent/reviews/2026-06-09-v014-same-state-momentum-mass.md`

Commands Run:

- `python -m py_compile proofs/v014/same_state_momentum_mass.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_state_momentum_mass.py`
- `python -m json.tool proofs/v014/same_state_momentum_mass.json >/tmp/same_state_momentum_mass.validated.json`

Proof Objects:

- `proofs/v014/same_state_momentum_mass.json`
- `proofs/v014/same_state_momentum_mass.md`
- `.agent/reviews/2026-06-09-v014-same-state-momentum-mass.md`

Result:

Verdict is `JAX_MISMATCH_U_post_after_all_rk_steps_pre_halo`. The first failing
field in sprint order is `U`, max_abs `6.292358893898424`, RMSE
`2.032497018496295`, worst native key `[4, 13]`, JAX
`-4.735481996086533` vs WRF `1.55687689781189`.

Risks:

- The h10 checkpoint predates the live-nest base-source partial fix, so
  `PB/MUB/PHB` residuals need a regenerated carry before attribution.
- The dynamic `U/V/W/T/MU` residuals are still actionable because the nearest
  named momentum/mass surface already fails before halo exchange or output
  writing can explain the symptom.
- The comparison covers the selected h10 patch and K1/KSTAG01 layers, not full
  domain/full column.

Handoff:

Regenerate the h10 step-5999 carry on current code, rerun this same proof, then
instrument one layer earlier inside final RK U/V large-step/acoustic update,
mass coupling, and theta-pressure source assembly.
