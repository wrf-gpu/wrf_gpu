# Review: V0.14 Step-1 Start-Domain Perturbation Subsurface

Verdict: `STEP1_START_DOMAIN_PERTURB_SUBSURFACE_LOCALIZED_CURRENT_JAX_AL_ALT_BASE_INPUT_GAP`.

objective: close the WRF live-nest `start_domain(nest,.TRUE.)` internal truth surface for Step-1 `P_STATE/MU_STATE/W_STATE` initialization.

files changed:
- `proofs/v014/step1_start_domain_perturb_subsurface.py`
- `proofs/v014/step1_start_domain_perturb_subsurface.json`
- `proofs/v014/step1_start_domain_perturb_subsurface.md`
- `proofs/v014/step1_start_domain_perturb_subsurface_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-start-domain-perturb-subsurface.md`

commands run:
- `date +%Y%m%d_%H%M%S`
- `rm -rf /mnt/data/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715`
- `cp --reflink=auto -a /mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF work_clean_20260609_194715/WRF`
- `cp --reflink=auto -a /mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/run work_clean_20260609_194715/run`
- `apply_patch work_clean_20260609_194715/WRF/dyn_em/start_em.F`
- `diff -u backup/start_em.F.before_start_domain_perturb_subsurface WRF/dyn_em/start_em.F > proofs/v014/step1_start_domain_perturb_subsurface_wrf_patch.diff`
- `./compile em_real (failed: /bin/csh missing, exit 126)`
- `PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH /home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin/tcsh ./compile em_real (first env-missing check failed in log, then rerun with PATH/NETCDF/PNETCDF)`
- `PATH=wrf-build/bin:$PATH NETCDF=wrf-build PNETCDF=wrf-build tcsh ./compile em_real`
- `mpirun -np 28 ./wrf.exe (failed: insufficient slots)`
- `mpirun --map-by :OVERSUBSCRIBE -np 28 ./wrf.exe`
- `python -m py_compile proofs/v014/step1_start_domain_perturb_subsurface.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_start_domain_perturb_subsurface.py`
- `python -m json.tool proofs/v014/step1_start_domain_perturb_subsurface.json >/tmp/step1_start_domain_perturb_subsurface.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_start_domain_perturb_subsurface.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_start_domain_perturb_subsurface.md`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_start_domain_perturb_subsurface_wrf_patch.diff`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-start-domain-perturb-subsurface.md`
- `/mnt/data/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715/wrf_truth`

ranked hypotheses/exclusions:
- rank 1: SUPPORTED_BY_INTERNAL_SURFACES - WRF live-nest start_domain recomputes P/al/alt, then press_adj updates MU, then set_w_surface updates W.
- rank 2: REFUTED_FOR_P_IF_THRESHOLD_EXCEEDED - A narrow production patch using current JAX inputs is exact enough now.
- rank 3: SUPPORTED - Remaining P gap is in current JAX start_domain input surfaces, not source ordering.
- excluded: The prefilled scratch-root WRF/run was ignored; all trusted WRF files came from the timestamped clean workdir.
- excluded: The hook is gated on grid%press_adj=.TRUE. and grid id 2, so ordinary non-live-nest d02 start_domain calls are excluded.
- excluded: WRF after_hypsometric P_STATE is continuous with accepted pre-call P_STATE; press_adj and W branch do not mutate P.
- excluded: WRF after_press_adj MU_STATE is continuous with accepted pre-call MU_STATE; remaining current-JAX MU gap is formula/input, not later solve_em.
- excluded: WRF after_w_surface_branch W_STATE is continuous with accepted pre-call W_STATE; W is not an acoustic or physics tendency source here.
- excluded: Boundary package, carry, halo, first_rk_step_part1, phy_prep, and acoustic refresh remain excluded by predecessor proofs.

unresolved risks:
- No production source patch was applied in this sprint.
- WRF source ordering is now proven, but current JAX AL/ALT/base/PH inputs still need a smaller split before a safe P/MU patch.
- The WRF truth is text savepoint data outside git; the repo commits only script, diff, JSON, and report metadata/checksums.

next decision: Split current JAX live-nest start_domain input construction for AL/ALT: compare final blended HT, PB/MUB/PHB, PH_STATE, MU before press_adj, and diagnosed AL/ALT against the WRF internal after_hypsometric surface. Patch P/MU only after that current-input gap is below the material gate.
