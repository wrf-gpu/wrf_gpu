# Review: V0.14 Step-1 P/PH/MU Boundary Localization

Verdict: `STEP1_P_PH_MU_LOCALIZED_FIRST_RK_STEP_PART1_P_STATE`.

objective: localize or narrowly fix the remaining d02 Step-1 strict same-input divergence after production live-nest theta/QV initialization closure.

files changed:
- `proofs/v014/step1_p_ph_mu_boundary_localization.py`
- `proofs/v014/step1_p_ph_mu_boundary_localization.json`
- `proofs/v014/step1_p_ph_mu_boundary_localization.md`
- `.agent/reviews/2026-06-09-v014-step1-p-ph-mu-boundary-localization.md`

commands run:
- `python -m py_compile proofs/v014/step1_p_ph_mu_boundary_localization.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_p_ph_mu_boundary_localization.py`
- `python -m json.tool proofs/v014/step1_p_ph_mu_boundary_localization.json >/tmp/step1_p_ph_mu_boundary_localization.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_p_ph_mu_boundary_localization.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_p_ph_mu_boundary_localization.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-p-ph-mu-boundary-localization.md`

unresolved risks:
- Existing WRF truth does not emit raw p/ph/mu boundary-package leaves, so this localizes before boundary application but does not prove package construction equality.
- Existing WRF truth does not split post-acoustic/pre-refresh from final calc_p_rho_phi pressure refresh, so no narrow pressure-refresh or acoustic-finish source fix was applied.
- U has no early substage source surface in the reused truth; its earliest checked material residual is the final post-RK/pre-halo comparison.

next decision: If fixing rather than further localizing, emit one WRF scratch surface inside first_rk_step_part1 around phy_prep/calc_p_rho_phi state writes for P/MU/W, or emit a post-acoustic/pre-refresh pressure surface if the manager wants to split the downstream final P residual before editing source.
