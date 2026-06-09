# Worker Report: V0.14 Step-1 Start-Domain Perturbation Subsurface

Summary: produced disposable WRF `start_domain(nest,.TRUE.)` internal truth
surfaces and localized the remaining `P/MU/W` gap to current JAX
`AL/ALT/base/PH` start-domain inputs; no production source edit was made.

objective: close the missing WRF live-nest `start_domain` truth surface needed
to patch Step-1 `P_STATE/MU_STATE/W_STATE` initialization safely.

files changed:

- `proofs/v014/step1_start_domain_perturb_subsurface.py`
- `proofs/v014/step1_start_domain_perturb_subsurface.json`
- `proofs/v014/step1_start_domain_perturb_subsurface.md`
- `proofs/v014/step1_start_domain_perturb_subsurface_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-start-domain-perturb-subsurface.md`

commands run:

- WRF scratch compile under the `wrf-build` env after documenting the missing
  `/bin/csh` and missing-env failures.
- CPU-only WRF replay with `mpirun --map-by :OVERSUBSCRIBE -np 28 ./wrf.exe`.
- `python -m py_compile proofs/v014/step1_start_domain_perturb_subsurface.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_start_domain_perturb_subsurface.py`
- `python -m json.tool proofs/v014/step1_start_domain_perturb_subsurface.json >/tmp/step1_start_domain_perturb_subsurface.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:

- `proofs/v014/step1_start_domain_perturb_subsurface.json`
- `proofs/v014/step1_start_domain_perturb_subsurface.md`
- `proofs/v014/step1_start_domain_perturb_subsurface_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-start-domain-perturb-subsurface.md`

result:

- Verdict:
  `STEP1_START_DOMAIN_PERTURB_SUBSURFACE_LOCALIZED_CURRENT_JAX_AL_ALT_BASE_INPUT_GAP`.
- WRF internal P-from-ALT formula/order closes: max_abs `0.015625` Pa.
- WRF `press_adj` formula closes: max_abs `4.547473508864641e-13` Pa.
- WRF after-W branch versus accepted pre-call W closes: max_abs
  `5.960464477539063e-08`.
- Current JAX pressure formula remains too far from WRF after-hypsometric P:
  max_abs `3.9458582235092763` Pa.
- Current JAX press_adj formula remains above the MU material gate:
  max_abs `0.047773029698646496` Pa.

unresolved risks:

- No production source patch was applied.
- WRF source ordering is now proven, but current JAX `AL/ALT/base/PH` inputs
  still need a smaller split before a safe `P/MU` patch.

next decision needed:

Open the next narrow split for current JAX live-nest start-domain inputs:
compare final blended `HT`, `PB/MUB/PHB`, `PH_STATE`, `MU` before `press_adj`,
and diagnosed `AL/ALT` against WRF internal `after_hypsometric` truth.
