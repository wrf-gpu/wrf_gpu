# F7N Worker Report — Straka touchdown per-substep diff + close

**Status: `F7N_COMPLETE`. The dry dynamical core is DONE.** Straka density current
**PASSES 6/6** at 900 s and the Skamarock warm bubble **STILL PASSES 6/6** (no
regression). The decisive per-acoustic-substep WRF ground-truth diff at the
cold-pool touchdown column localized the residual to one sign error in the
vertical momentum advection; the WRF-faithful fix closes it.

## Objective

Build a per-acoustic-substep WRF `em_grav2d_x` (Straka) touchdown-column savepoint
diff, identify the first operator where JAX diverges from WRF at the cold-pool
touchdown, fix it WRF-faithfully, and close the dynamical core.

## What was done

1. **Instrumented pristine WRF v4.7.1** `dyn_em/solve_em.F` (new `WRFGPU2_TOUCHDOWN`
   block after the in-loop `calc_p_rho`) to dump per acoustic substep, for
   `itimestep 170..205`, the touchdown column `i=256` (= `nxc` = domain center)
   and neighbours `i=255,257`, `j=2`: `w_2, ph_2, p, rw_tend, ph_tend, ww (omega),
   u_2` (faces i and i+1), `v_2, t_2save, muave, muts, mut`. Incremental recompile
   (serial, configure 32+nesting 0), re-ran 6 model min →
   `/mnt/data/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_dump.txt` (360 records).
2. **Built the matching JAX per-substep dump** (`scripts/f7n_jax_touchdown_dump.py`,
   operational RK3/acoustic cadence, Python substep loop) at the JAX center mass
   index → `..._touchdown_substeps.json`.
3. **Diffed** (`scripts/f7n_touchdown_diff.py`) → `proofs/f7n/touchdown_substep_diff.json`.
   Found the JAX center-column **`ww` (omega) develops a 2Δz vertical sawtooth**
   (z≈2200–3700 m) that WRF does not, traced to a **growing 2Δz vertical mode in
   `u`** in the descent layer: u 2nd-diff 0.04(t120)→4.2(t170)→137(t190)→NaN,
   max|u| 17→47→135→523 m/s, max|w| 21→44→407, front frozen 2350–2450 m. This is
   exactly the F7M "descending w not converted to outflow" signature, now resolved.
4. **Bisection:** disabling the vertical momentum advection removed the mode
   (u 2Δz 26.6→0.7 at t=180) → the vertical momentum advection was *generating*
   it (anti-dissipative), not under-damping.
5. **Fixed** the root cause (below). Re-ran Straka to 900 s + warm bubble; both PASS.

## Root cause + fix

- **The close:** `flux_advection._vertical_flux_div_3` (the `advect_u`/`advect_v`
  3rd-order vertical flux) applied the upwind correction with the **opposite sign**
  to WRF. WRF `advect_u` (`module_advect_em.F:1474-1480`) assembles
  `vflux = vel*flux3(u, -vel)` with `flux3` (`:202-204`) =
  `flux4 + sign(-vel)*corr` → `vel*flux4 - |vel|*corr` (DISSIPATIVE). JAX used
  `sign(+rom)` → `+|rom|*corr` (ANTI-dissipative), pumping the 2Δz-in-z mode that
  the corrector is meant to damp. Fixed to `flux3 = flux4 + sign(-rom_k)*corr`,
  `vflux = rom_k*flux3`. The scalar (`advect_scalar_flux`) and `w`
  (`_vertical_flux_div_w`) vertical-flux paths were already sign-correct.
- **Secondary (mass conservation):** once Straka ran to 900 s the dry-mass drifted
  3.4e-8 (>1e-8 gate), traced (diffusion-off A/B → 0 drift) to the
  non-conservative `mass*K∇²` const-K diffusion (F7L). Replaced with the
  WRF flux-divergence form `d/dx_j(mass*K*d./dx_j)`
  (`conservative_constant_k_diffusion_tendency`; WRF
  `module_diffusion_em.F:2999-3018`). Drift → 2.25e-9; θ′min improved to −9.97 K.
  Bubble is inviscid (ν=0) so this is a no-op there.

## Files changed

- **M** `src/gpuwrf/dynamics/flux_advection.py` — WRF sign on `_vertical_flux_div_3`.
- **M** `src/gpuwrf/dynamics/explicit_diffusion.py` — add
  `conservative_constant_k_diffusion_tendency`.
- **M** `src/gpuwrf/runtime/operational_mode.py` — wire conservative diffusion (u/v/w/θ).
- **M** `proofs/f7/DYCORE_STATUS.md` — marked CLOSED.
- **NEW** `proofs/f7n/` touchdown_substep_diff.json (key artifact), touchdown_fix.md,
  straka_density_current_diagnostics.json+verdict.md, skamarock_bubble_*,
  regression_recheck.json, plots/.
- **NEW** `scripts/f7n_jax_touchdown_dump.py`, `f7n_touchdown_diff.py`, `f7n_official_run.py`.
- **NEW** WRF instrumentation `/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F`
  (`WRFGPU2_TOUCHDOWN`), `recompile_grav2d_incremental.sh`; ground truth
  `/mnt/data/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_dump.txt`.

## Commands run (CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src taskset -c 0-3, cuda:0, fp64)

- WRF: `recompile_grav2d_incremental.sh`, `run_grav2d.sh`.
- JAX: `scripts/f7n_jax_touchdown_dump.py --start 170 --end 205`,
  `scripts/f7n_touchdown_diff.py`, `scripts/f7n_official_run.py both`,
  bisection probes (novadv / strongvdiff / diffusion-off A/B).
- `pytest tests/test_m4_acoustic.py test_m4_dycore_step.py test_m4_tier2_invariants.py` → **10 passed**.

## Acceptance gates

- **AC1 (Straka PASS): PASS 6/6** — finite to 900 s; front 14.15 km; θ′min −9.97 K;
  max|w| 14.57; 4 rotors; mass drift 2.25e-9.
- **AC2 (warm bubble PASS 6/6): PASS** — thermal_rise 1924.3 m, max|w| 11.68,
  θ′max 1.92, h-drift 1.8e-12, mass drift 0 (identical to F7K/L/M; no regression).
- **AC3 (touchdown per-substep parity): DELIVERED** —
  `proofs/f7n/touchdown_substep_diff.json`; post-fix JAX center-column ww
  low-level bounded (~0) matching WRF, pre-fix runaway gone.
- **AC4 (no regression): PASS** — m4 10/10; flat-rest machine-zero; conservation;
  all prior F7 operators intact; no masking clamps; only WRF-faithful sign +
  flux-form fixes.

## Unresolved risk / next decision

None blocking. The coupled center-column `w` in the touchdown diff runs
larger-magnitude in JAX (~−3000) than WRF (~−1000..−2500) due to the half-cell
grid registration (WRF bubble center on mass point i=256; JAX center on a cell
face) and the vertical grid (JAX 60 uniform 100 m vs WRF 65 stretched); both are
bounded and the operational physics gate (max|w|@900s = 14.57) matches WRF
(~14–15). Next decision for the manager: GPT-5.5 pre-close critique → merge the
`f7d` chain → Phase B. No performance work was done (out of scope).

F7N_COMPLETE
