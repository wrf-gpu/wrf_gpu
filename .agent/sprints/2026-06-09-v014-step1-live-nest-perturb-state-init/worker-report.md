# Worker Report: V0.14 Step-1 Live-Nest Perturbation-State Init

Summary: localized live-nest `P/MU/W` perturbation-state initialization to WRF
`start_domain(nest,.TRUE.)`; no production source edit.

objective: close or precisely localize the live-nest `raw_child_state ->
live_child_state` perturbation-state mismatch for `P_STATE/MU_STATE/W_STATE`.

files changed:

- `proofs/v014/step1_live_nest_perturb_state_init.py`
- `proofs/v014/step1_live_nest_perturb_state_init.json`
- `proofs/v014/step1_live_nest_perturb_state_init.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-perturb-state-init.md`

commands run:

- `python -m py_compile proofs/v014/step1_live_nest_perturb_state_init.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_perturb_state_init.py`
- `python -m json.tool proofs/v014/step1_live_nest_perturb_state_init.json >/tmp/step1_live_nest_perturb_state_init.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:

- `proofs/v014/step1_live_nest_perturb_state_init.json`
- `proofs/v014/step1_live_nest_perturb_state_init.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-perturb-state-init.md`

result:

- Verdict:
  `STEP1_LIVE_NEST_PERTURB_STATE_LOCALIZED_START_DOMAIN_P_PRESS_ADJ_SET_W_SURFACE_P_AL_ALT_SUBSURFACE_GAP`.
- `P_STATE`: `69.96875` -> `3.9458582235092763` Pa.
- `MU_STATE`: `13.256103515625` -> `0.047773029698646496` Pa.
- `W_STATE`: `0.7605466246604919` -> `1.2992081932505783e-07` m/s.

unresolved risks:

- No production source edit was applied.
- `P_STATE` still requires an internal WRF `start_domain` `al/alt` and
  pre/post-`press_adj` truth surface before patching.

next decision needed:

Open the narrow WRF `start_domain(nest,.TRUE.)` savepoint/source sprint.
