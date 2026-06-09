# Review: V0.14 Step-1 Live-Nest Perturbation-State Init

Verdict: `STEP1_LIVE_NEST_PERTURB_STATE_LOCALIZED_START_DOMAIN_P_PRESS_ADJ_SET_W_SURFACE_P_AL_ALT_SUBSURFACE_GAP`.

objective: close or precisely localize the live-nest `raw_child_state -> live_child_state` perturbation-state mismatch for `P_STATE/MU_STATE/W_STATE`.

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
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_live_nest_perturb_state_init.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_live_nest_perturb_state_init.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-live-nest-perturb-state-init.md`
- reused WRF pre-call truth root `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`

ranked hypotheses/exclusions:
- rank 1: SUPPORTED_LOCALIZED - Missing WRF start_domain perturbation-state initialization after live-nest base/theta/QV correction.
- rank 2: REMAINING_GAP - Exact P/MU closure needs internal start_domain pre/post-press_adj and al/alt truth surfaces or stricter Fortran evaluation order.
- rank 3: LOWER_RANKED - Parent interpolation/blending alone explains P/MU/W.
- excluded: WRF after_step_increment -> before_first_rk_step_part1_call is exact for P/MU/W/PH in reused pre-part1 truth.
- excluded: Prior proof showed WRF before_first_rk_step_part1_call -> after_first_rk_step_part1 is exact for P/MU/W/PH.
- excluded: Prior proof showed JAX raw/live/boundary/carry/halo all retain the same raw P/MU/W residuals.
- excluded: Boundary package, initial carry, halo application, _physics_step_forcing, first_rk_step_part1, phy_prep, and acoustic refresh are not the first cause for this boundary.
- excluded: W_STATE is not an unknown physics tendency: raw surface W is zero and WRF default use_input_w=.false. forces set_w_surface.

unresolved risks:
- The proof-local P_STATE formula is localized but not exact enough for a production patch.
- A safe source edit needs WRF start_domain sub-surfaces for al/alt and pre/post press_adj MU.
- No full Step-1 rerun was attempted because no production source changed.

next decision: Open one narrow WRF savepoint/source sprint at start_domain live-nest perturbation init, then patch d02_replay only if P_STATE closes with the exact al/alt/pre-press-MU contract.
