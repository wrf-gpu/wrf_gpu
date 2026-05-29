# F7I Worker Report — kill the residual vertical mode (WRF ground truth)

**Status: `F7I_PARTIAL`** (AC1 reframed+evidenced; AC2/AC3 FAIL; AC4 build-deferred
with definitive JAX-side clearance of the implicit solve; AC5 PASS). No
WRF-grounded fix landed because no tested WRF-grounded fix removes the mode and
the WRF binary arbiter is build-blocked. STOP per hard rule 2.

## Objective
Settle the residual warm-bubble vertical blow-up with WRF ground truth: build
WRF `em_quarter_ss`, dump center-column implicit-solve savepoints, diff vs JAX,
land the WRF-correct off-centering/coefficient fix, pass Skamarock + Straka.

## Headline finding (the framing was wrong)
The "residual 2Δz acoustic mode in the implicit w/ph solve" is **REFUTED**. The
growing structure is a **deep buoyancy-driven vertical standing mode** (≈12-level
wavelength, NOT an adjacent-level checkerboard) that grows exponentially
(e-folding ≈ 29 s, fixed modal structure, amplitude-growing) and is driven by the
once-per-RK-stage `pg_buoy_w` large-step forcing — **not** by the implicit
acoustic solve. Two independent lines of evidence clear the implicit solve;
the residual is in the large-step vertical momentum/buoyancy balance.

## Files changed
- **NO production code changed** (git tracked non-proof diff is empty).
- NEW diagnostic scripts (never weaken invariants): `scripts/f7i_center_column_w_trace.py`,
  `f7i_epssm_sensitivity.py`, `f7i_term_ablation.py`, `f7i_rwtend_profile.py`,
  `f7i_net_vert_forcing.py`, `f7i_wadv_fix_probe.py`.
- NEW proofs under `proofs/f7i/`.
- Separate WRF ideal build tree `/home/enric/src/wrf_ideal_f7i/WRF` (rsync copy;
  canonical Gen2 em_real tree + its sha256 UNTOUCHED) + conda env `wrfbuild`
  (gfortran 14.3 + netcdf-fortran + tcsh). ~75% built; blocked on GPU-fork modules.

## Commands run (all `PYTHONPATH=src taskset -c 0-3`, cuda:0, fp64)
- `python -u scripts/f7i_center_column_w_trace.py --steps 2000 --stride 100`
- `python -u scripts/f7i_epssm_sensitivity.py`  (epssm 0.1..1.0 sweep)
- `python -u scripts/f7i_term_ablation.py`  (pg_buoy / theta_adv ablation)
- `python -u scripts/f7i_rwtend_profile.py --steps {0,30,100,300,600}`
- `python -u scripts/f7i_net_vert_forcing.py --steps 100`
- `python -u scripts/f7i_wadv_fix_probe.py`  (advect_w-into-rw_tend fix test)
- `run_warm_bubble_case` + `run_density_current_case` → verdicts (FAIL)
- `pytest tests/test_m4_acoustic.py test_m4_dycore_step.py test_m4_tier2_invariants.py` → 10 passed (x2)
- WRF build: rsync copy; conda `wrfbuild` env; `./configure` (gfortran serial);
  6 `./compile em_quarter_ss` passes (registry race, PGI-flag strip,
  GPU-phase import patch) → blocked on ~13 remaining `*_gpu_phase*` modules.

## Proof objects (`proofs/f7i/`)
- `center_column_w_trace.json` — 2Δz before-state trace (the mode is a deep
  standing wave, sign_alt const 0.075, exp growth e-fold 29 s).
- `epssm_sensitivity.json` — epssm 0.1→1.0 barely changes growth (NOT the implicit solve).
- `term_ablation.json` — zeroing `pg_buoy_w` removes the entire mode (driver pinpoint).
- `rwtend_profile_{0,30,100,300}.json` — IC balanced (rw_tend≈0); smooth -115 downward
  by t3s; multi-lobe oscillation by t30s against smooth θ′/grid_p.
- `net_vert_forcing_100.json` — full-p == work-p `rw_tend` at t10s (F7H distinction moot).
- `wadv_fix_probe.json` — adding `advect_w` to `rw_tend` does NOT fix (detonates <500s).
- `wrf_em_quarter_ss_savepoints.json` — build-deferral note + resolution path.
- `wrf_vs_jax_implicit_w.json` — implicit-solve clearance + residual localization.
- `2dz_fix.md` — full root-cause writeup with WRF file:line.
- `skamarock_warm_bubble.json` + `skamarock_bubble_verdict.md` (FAIL).
- `straka_density_current.json` + `straka_density_current_verdict.md` (FAIL).
- `regression_recheck.json` — AC5 no-regression evidence.
- (folded in) `gpt-coefaudit-findings.md` — parallel GPT audit, converges:
  injected 2Δz mode is DAMPED (ratio 0.996) by the implicit solve.

## Acceptance gates
- **AC1 (2Δz mode gone): REFRAMED.** It is not a 2Δz mode and not in the implicit
  solve (epssm-independent + injected-mode probe damps it). Before-trace delivered;
  no after-trace because no fix landed.
- **AC2 (warm bubble): FAIL.** Bubble RISES (centroid +~170 m), max|w|@100s ≈ 4.3
  (10× below pre-F7H), but the buoyancy-driven mode grows exp and detonates ~180 s
  (< 500 s gate).
- **AC3 (Straka): FAIL.** Same mode; non-finite before 900 s.
- **AC4 (WRF ground truth): DEFERRED (build infeasible in bounded effort).** The
  canonical Gen2 tree is an NVHPC/OpenACC GPU fork; CPU-building it under gfortran
  requires unwinding ~18 GPU-phase modules + PGI build rules + refactored
  interfaces. 6 attempts cleared the registry race, PGI flags, and 5 advect-GPU
  modules; ~13 diffusion/small_step GPU modules + a `save`-symbol linkage remain.
  Delivered instead the two-way JAX-side clearance of the implicit solve + the
  pg_buoy_w driver localization. WRF binary remains the definitive arbiter.
- **AC5 (no regression): PASS.** 10/10 m4 tests (x2); no production code changed;
  no clamps/caps/epssm-fudge/diffusion landed (all sweeps diagnostic-only).

## Root cause (evidence-first, triangulated with the parallel GPT audit)
The implicit w/ph acoustic solve + `calc_coef_w` are WRF-faithful and
2Δz-STABLE (epssm-independent growth; injected `(-1)^k` mode is damped 0.996/substep;
line-by-line source match). The exponential mode is a **vertical
momentum/buoyancy BALANCE discretization** problem in the large-step path:
the `pg_buoy_w` vertical PGF (`g·rdn·Δ(grid%p)`) and the in-solver
`c2a·alt·t_2ave` buoyancy do not discretely telescope into a saturating
response, so the warm bubble's vertical response grows instead of saturating.

## WRF-faithfulness gaps found (real; none alone fixes the mode)
1. `advect_w` missing from `rw_tend` (WRF module_em.F:1011-1059 vs JAX
   `_acoustic_core_state_from_prep`). Tested: does NOT fix.
2. `rhs_ph`/`ph_tend` never computed (init 0; `accumulate_ph_tend` is a stub).
   NOT tested as a fix (substantial operator; not added speculatively).
3. Diffusion/top-damp mismatch vs WRF em_quarter_ss (`diff_opt=2 km_opt=2
   khdif=kvdif=500 damp_opt=2` vs JAX `const_nu=0 damp_opt=3`). Tested:
   diffusion delays detonation to ~180s+ but ALL nu∈{75,200,500} detonate <500s.

## Unresolved risk / next decision needed
The definitive arbiter (WRF em_quarter_ss center-column `rw_tend`/`c2a·alt·t_2ave`
savepoint diff) is build-blocked. **Manager decision:** (A) reinstall the NVHPC
SDK that built the canonical tree and build em_quarter_ss with the GPU toolchain;
(B) obtain pristine upstream WRFv4 and build serial with the ready `wrfbuild`
gfortran env; or (C) finish stubbing the Gen2 GPU-phase modules for a CPU link.
Then diff the center-column large-step vertical balance and land the WRF-correct
fix (candidates: advect_w into rw_tend, rhs_ph ph_tend, km_opt=2 diffusion —
each verified against the binary, not guessed). The two committed F7D/F7H fixes
(grid%p refresh, full-p pg_buoy_w) are correct and regression-safe; keep them.

F7I_PARTIAL
