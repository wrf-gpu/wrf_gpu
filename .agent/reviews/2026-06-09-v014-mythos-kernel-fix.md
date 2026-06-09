# Review: V0.14 Mythos Kernel Fix (live-nest start_domain P/MU/W)

Verdict: `MYTHOS_KERNEL_FIX_START_DOMAIN_P_MU_W_CLOSED_FP32_LIBM_SINT_BLEND_BIT_EXACT`.

objective: one-pass root-cause and fix of the Step-1 live-nest/start-domain `P/MU/W` grid divergence.

files changed:
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/nesting/interp.py`
- `proofs/v014/mythos_kernel_fix_260609.{py,json,md}`
- `.agent/reviews/2026-06-09-v014-mythos-kernel-fix.md`
- regenerated: `proofs/v014/step1_jax_start_domain_input_split.{json,md}`,
  `step1_start_domain_perturb_subsurface.{json,md}`, `step1_live_nest_perturb_state_init.{json,md}`,
  `step1_base_state_boundary.{json,md}` (+ their review files)

commands run:
- `python -m py_compile src/gpuwrf/integration/d02_replay.py src/gpuwrf/nesting/interp.py proofs/v014/mythos_kernel_fix_260609.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/mythos_kernel_fix_260609.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_jax_start_domain_input_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_start_domain_perturb_subsurface.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_perturb_state_init.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_base_state_boundary.py`
- `python -m json.tool` on every regenerated JSON; `git diff --check`; `git diff -- src/gpuwrf`

WRF truth surfaces used:
- `/mnt/data/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715/wrf_truth` (`after_hypsometric_p_al_alt`, `after_press_adj`, `after_w_surface_branch`)
- `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz` (strict Step-1 post-RK/pre-halo 16-field truth)

before/after (max_abs vs WRF internal start_domain truth):
- `P_STATE`: 69.96875 -> 0.0390625 (gate 1.0, pass=True)
- `MU_STATE`: 13.256103515625 -> 4.547473508864641e-13 (gate 0.01, pass=True)
- `W_STATE`: 0.7605466842651367 -> 5.551115123125783e-17 (gate 0.001, pass=True)
- `MUB`/`PB`: bit-exact 0.0 after patch (were 0.05/0.054); `PHB`: 4.547473508864641e-13 (was 0.108); `HT`: 4.547473508864641e-13 (was 2.682e-05)

ranked hypotheses:
- rank 1: PROVEN_BIT_EXACT - The remaining p_surf->MUB gap was float32 libm provenance: WRF calls scalar glibc expf, and gfortran compiles (...)**0.5 to a glibc powf(x,0.5) call (1 ulp from sqrtf on rare inputs).
- rank 2: PROVEN_AND_CLOSED - The residual production P gap after exact-libm base was the float64 SINT/blend terrain (<= 1 fp32 ulp HT error amplified ~50x through the AL/ALT layer-thickness division).
- rank 3: CONFIRMED_BY_PREDECESSORS_AND_CLOSED_IN_PRODUCTION - Raw wrfinput perturbation P/MU/W leaves miss three WRF start_domain mutations (hypsometric P rederivation, press_adj MU, set_w_surface W).
- rank 4: REFUTED_AS_PRIMARY_KEPT_AS_FALLBACK - float64-rounded transcendentals are a sufficient production formula.

unresolved risks:
- Bit-exactness binds to the host glibc float32 libm (same libm the WRF truth build linked). On a
  non-glibc host the helper falls back to float64-rounded float32 (P_STATE residual then ~2.3 Pa
  worst-case, still ~1e-5 relative); the provider is recorded in the init metadata.
- The ctypes scalar libm loops cost roughly a second per nest domain at init (one-time, host-side;
  d03-scale domains tens of seconds). A vectorized binding is a later cleanup, not a correctness need.
- The proof surrogate constructors in older step1 proofs do not call the new perturbation init;
  their `raw` rows intentionally still show the pre-patch residuals.
- The strict Step-1 16-field one-RK-step residual (see proof JSON `strict_step1_16field_with_patched_init`)
  is now the authoritative remaining divergence; it is a dynamics-side question, outside this contract's
  allowed source scope.

next decision needed: Init-time P/MU/W is closed to WRF start_domain truth (bit-exact base, P 0.039 Pa). The strict Step-1 16-field one-RK-step comparison is now the authoritative post-init divergence frontier, and the attribution probe shows it is REAL one-step dynamics state divergence in the PH/MU/P (acoustic/mass/vertical) lane, not init and not pressure-diagnosis semantics. Next sprint: (1) freeze one-step namelist parity (acoustic substep count, epssm, damping) between the proof surrogate and the WRF case3 namelist; (2) rerun the existing RK1 substage comparator chain (/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/wrf_truth is still valid WRF-side) against JAX substages built from the NOW-CLOSED init state. No dycore source edit is justified before that localization.
