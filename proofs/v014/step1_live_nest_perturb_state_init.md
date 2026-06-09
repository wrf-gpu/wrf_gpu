# V0.14 Step-1 Live-Nest Perturbation-State Init

Verdict: `STEP1_LIVE_NEST_PERTURB_STATE_LOCALIZED_START_DOMAIN_P_PRESS_ADJ_SET_W_SURFACE_P_AL_ALT_SUBSURFACE_GAP`.

## Result

- CPU backend: `cpu`; GPU used: `False`.
- Required ancestor `131b27cd` present: `True`.
- Manager hypothesis is supported but not fully patch-ready: WRF does recompute/adjust `P/MU/W` before the first part1 call, while current JAX keeps raw `wrfinput_d02` perturbation leaves.
- No production source edit was applied because `P_STATE` still needs an internal `start_domain` `al/alt` or pre-press-MU surface before an exact GPU-native patch.

## Candidate Table

| Field | Raw JAX max_abs | Candidate/source | Candidate max_abs | Notes |
|---|---:|---|---:|---|
| `P_STATE` | 69.96875 | `start_domain` hypsometric pressure recompute | 3.9458582235092763 | WRF-exact-input FP32 falsifier: 0.3828125 Pa |
| `MU_STATE` | 13.256103515625 | `press_adj` terrain-delta correction | 0.047773029698646496 | Not patch-ready: p95 4.775559752943084e-05, p99 0.002854282405910437; needs pre/post-press_adj truth |
| `W_STATE` | 0.7605466246604919 | `set_w_surface(fill_w_flag=.true.)` | 1.2992081932505783e-07 | Closed proof-locally |

## Ranked Hypotheses

- 1. Missing WRF start_domain perturbation-state initialization after live-nest base/theta/QV correction. Status: `SUPPORTED_LOCALIZED`. P start_domain recompute reduces max_abs from 69.96875 to 3.945858; MU press_adj reduces 13.2561 to 0.04777; W set_w_surface reduces 0.76055 to 1.3e-7.
- 2. Exact P/MU closure needs internal start_domain pre/post-press_adj and al/alt truth surfaces or stricter Fortran evaluation order. Status: `REMAINING_GAP`. Even WRF exact PB/PHB/PH/T plus FP32 formula leaves P max_abs 0.3828125 Pa; MU is near-closed but still not exact enough to prove the source sequencing without pre/post press_adj truth. A source patch would still be a guess.
- 3. Parent interpolation/blending alone explains P/MU/W. Status: `LOWER_RANKED`. Base/theta/QV are already close; W closes through set_w_surface and MU/P both point into start_domain, not a new parent interpolation surface.

## Exclusions

- WRF after_step_increment -> before_first_rk_step_part1_call is exact for P/MU/W/PH in reused pre-part1 truth.
- Prior proof showed WRF before_first_rk_step_part1_call -> after_first_rk_step_part1 is exact for P/MU/W/PH.
- Prior proof showed JAX raw/live/boundary/carry/halo all retain the same raw P/MU/W residuals.
- Boundary package, initial carry, halo application, _physics_step_forcing, first_rk_step_part1, phy_prep, and acoustic refresh are not the first cause for this boundary.
- W_STATE is not an unknown physics tendency: raw surface W is zero and WRF default use_input_w=.false. forces set_w_surface.

## Next Surface

Emit WRF start_domain live-nest surfaces after the hypsometric P/al/alt recompute and immediately before/after press_adj, including P_STATE, MU_STATE, al, alt, alb, PH_STATE, PB, MUB, PHB, theta, qv, HT, and HT_FINE. That is the smallest surface needed before a GPU-native source patch.

Detailed metrics are in `proofs/v014/step1_live_nest_perturb_state_init.json`.
