# GPT Coefficient/Off-Centering Audit

Objective: independently audit whether the F7I warm-bubble vertical blow-up is caused by a missing/wrong `epssm` off-centering or tridiagonal coefficient error in the JAX implicit `w/ph` solve.

## Probe Evidence

Environment for probes: `/tmp/wrf_gpu2_coefaudit`, `PYTHONPATH=src taskset -c 0-3`, `CUDA_VISIBLE_DEVICES=0`, JAX fp64, device `cuda:0`.

Runtime namelist check:

- Warm bubble uses `dt_s=0.1`, `acoustic_substeps=10`, `epssm=0.1`, `top_lid=True`, `w_damping=1`, `damp_opt=3`, `dampcoef=0.2`, `zdamp=3000`.
- WRF Registry default is `epssm=.1` (`Registry.EM_COMMON:2870`); WRF `em_quarter_ss/namelist.input:80-81` also uses `epssm=0.1`, `time_step_sound=6`. Gen2 d02 backup uses `epssm=0.5`, but that is not the quarter-ss default.

Warm-bubble center-column sweep, current code:

| epssm | t=100s max\|w\| | t=125s max\|w\| | t=150s max\|w\| | t=150s strict `(-1)^k` projection |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 4.386 | 9.104 | 17.273 | 0.0375 |
| 0.5 | 4.305 | 8.890 | 16.777 | 0.0185 |
| 1.0 | 4.208 | 8.636 | 16.199 | 0.0077 |

Raising `epssm` damps the strict alternating projection, but it does not remove the growing vertical response. Baseline becomes non-finite at 180 s. The center profile at 170 s is a large vertical standing oscillation, not a pure adjacent-level checkerboard: around the active layer it is approximately `+10.4,+11.9,+10.2,+5.23,-2.49,-12.1,-21.4,-26.6,-25.2,-17.3,-5.07,+7.77,+17.6,+22.2,+17.4`, with strict `(-1)^k` projection only `0.046` against max `26.55`.

Direct injected 2Δz-mode probe inside `advance_w_wrf`:

- Input: pure interior alternating `w(k)=(-1)^k`, zero `rw_tend`, zero `ww`, zero `ph_tend`, no buoyancy, IC warm-bubble coefficients.
- Result: the implicit operator does not amplify the 2Δz mode. One RK1 substep gives amplitude ratios:
  - `eps_coef=0.0`, `eps_adv=0.0`: `0.99647`
  - `eps_coef=0.1`, `eps_adv=0.1`: `0.99612`
  - `eps_coef=0.5`, `eps_adv=0.5`: `0.99472`
  - `eps_coef=1.0`, `eps_adv=1.0`: `0.99298`
  - mixed coefficient/advance eps probes also stayed damping, not amplifying.

Read-only stabilizer A/B:

- Base: non-finite at 180 s.
- `const_nu_m2_s=500` monkeypatch: finite at 180 s, max center `28.81`.
- `diff_6th_opt=2`, `diff_6th_factor=0.12` monkeypatch: finite at 180 s, max center `30.45`.
- This is not a fix proof, but it says the failure window is sensitive to missing/changed large-step stabilizers while `epssm` alone is not decisive.

## Source Comparison

I do not find an `epssm` threading or sign bug in the JAX implicit vertical solve.

WRF `calc_coef_w`:

- WRF `module_small_step_em.F:624`: `cof=(.5*dts*g*(1.+epssm))**2`.
- JAX `src/gpuwrf/dynamics/acoustic_wrf.py:648`: `cof = (0.5 * dt * gravity * (1.0 + epssm)) ** 2`.
- WRF lower/top/interior coefficients are at `module_small_step_em.F:625-649`.
- JAX matching rows are `src/gpuwrf/dynamics/acoustic_wrf.py:654-686`.
- Signed `rdn/rdnw` are preserved in the idealized metrics; no abs-metric substitution was observed.

WRF `advance_w` off-centering:

- WRF applies `(1+epssm)`/`(1-epssm)` to `t_2ave` at `module_small_step_em.F:1341-1344`; JAX matches at `src/gpuwrf/dynamics/core/advance_w.py:230-237`.
- WRF uses old-time `0.5*g*(1-epssm)*w` in the `rhs` at `module_small_step_em.F:1345`; JAX matches at `advance_w.py:239-245`.
- WRF applies the mixed new/old geopotential pressure coupling at `module_small_step_em.F:1477-1485`; JAX matches at `advance_w.py:305-336`.
- WRF final geopotential update uses `0.5*dts*g*(1+epssm)*w` at `module_small_step_em.F:1581-1584`; JAX matches at `advance_w.py:424-432`.
- JAX threads `epssm` into coefficient assembly at `src/gpuwrf/dynamics/core/acoustic.py:962-970` and into `advance_w_wrf` at `acoustic.py:972-982` and `acoustic.py:554-604`.

## Most Likely Cause

The evidence does not support the prompt hypothesis that the residual is admitted by missing/wrong `epssm` off-centering in the implicit `w/ph` solve. The most likely supported cause is that the observed 180 s failure is being driven upstream or around the implicit solve by incomplete/mismatched WRF large-step stabilizer physics for the idealized warm bubble, then the implicit solve carries that growing vertical oscillator.

The concrete mismatch I can cite:

- JAX warm-bubble setup leaves warm-bubble diffusion off and sets top damping to the `advance_w` `damp_opt=3` path: `src/gpuwrf/ic_generators/idealized.py:565-595`.
- WRF `em_quarter_ss` namelist uses `diff_opt=2`, `km_opt=2`, `damp_opt=2`, `khdif=500`, `kvdif=500`, `epssm=0.1`, `time_step_sound=6`: `WRF/test/em_quarter_ss/namelist.input:71-81`.
- WRF applies `damp_opt=2` through large-step `rk_rayleigh_damp`, not the `advance_w` `damp_opt=3` branch: `WRF/dyn_em/module_em.F:1621-1631`; source comments for that damper are at `module_big_step_utilities_em.F:6139-6142`.

This conclusion is bounded: I did not build/run an instrumented WRF `em_quarter_ss` binary in this audit, so the WRF-vs-JAX savepoint diff remains the definitive arbiter.

## Concrete Fix

Do not change `advance_w_wrf` `epssm` signs or `calc_coef_w_wrf_coefficients` based on this audit. First align the warm-bubble driver with WRF ground truth:

1. Add/source the missing WRF `em_quarter_ss` large-step stabilizer configuration before declaring the implicit solve guilty: `diff_opt=2`, `km_opt=2`, `khdif/kvdif=500`, `damp_opt=2`, and `time_step_sound=6` semantics, or document an intentional deviation.
2. Implement or explicitly defer WRF `damp_opt=2` `rk_rayleigh_damp` and the WRF diffusion path used by quarter_ss. The current `damp_opt=3` implementation is a different small-step top damper.
3. Then rerun the WRF savepoint comparison requested by the sprint: center-column `calc_coef_w`, `w`, `ph`, `p`, `t_2ave`, `muave`, `epssm` for the same IC/steps.

## Falsifiable Check

The fix is validated only if:

- WRF-savepoint diff shows JAX `a/alpha/gamma` and `advance_w` intermediates match WRF within tolerance at the center column.
- Warm bubble remains finite to 500 s with no growing adjacent-level `(-1)^k` projection in center-column `w(k)`.
- The run still uses WRF-correct `epssm=0.1` for `em_quarter_ss` unless the compared WRF namelist says otherwise; no epssm tuning beyond the WRF namelist.

## Handoff

- objective: audit the implicit vertical coefficient/off-centering path for the residual warm-bubble vertical mode.
- files changed: `/home/enric/src/wrf_gpu2/.agent/sprints/2026-05-29-f7i-2dz-mode-wrf-truth/gpt-coefaudit-findings.md` only.
- commands run: source reads via `sed`/`nl`/`rg`; JAX warm-bubble sweeps for `epssm=0.1/0.5/1.0`; direct injected 2Δz `advance_w_wrf` probe; stabilizer A/B with `const_nu_m2_s=500` and `diff_6th_opt=2`.
- proof objects produced: this findings file with embedded numeric probe results.
- unresolved risks: no WRF `em_quarter_ss` binary savepoint was generated; stabilizer A/B is localization evidence, not a correctness proof.
- next decision needed: whether F7I should implement WRF quarter_ss stabilizers first or proceed directly to WRF binary savepoint instrumentation.

F7I_AUDIT_COMPLETE
