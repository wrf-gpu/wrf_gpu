# WRF-faithful reference for the d03 Ni fix (pristine WRF v4, <USER_HOME>/src/wrf_pristine)

Extracted so the fix MATCHES real WRF (faithful, NOT a masking clamp). Two possible roots — the onset-order
diagnostic (step ~720) decides which; this gives the faithful fix for EACH.

## A. Microphysics root (Ni) — exact WRF bounds in `phys/module_mp_thompson.F`
Restore whichever the port DROPPED (these are literally what WRF does):
- **ni floor:** `R2 = 1.E-6` (line 184) — ni never below 1e-6 when cloud ice present.
- **ice particle mass floor:** `xm0i = 1.E-12` (line 223) — nucleated crystals init at 1e-12 kg.
- **ni HARD ceiling:** `if (xni.gt.999.E3) niten = (999.E3-ni1d*rho)*odts*orho` (lines 3054-3055) — ni ≤ 999e3 m⁻³,
  enforced AFTER all microphysics tendencies summed.
- **diameter-bounded ni limits** (lines 3041-3049): if mean ice diameter < 5µm → clamp ni to 999e3; if > 300µm →
  recompute ni to stay consistent.
- **sublimation number clamp:** `pni_ide = MAX(-ni*odts, pni_ide)` (line 2658) — the deposition/sublimation
  number tendency (the `deposition/xmi` division at the heart of the suspected blow-up) is clamped so it can
  NEVER remove more ice number than exists. THIS is the WRF guard on the unguarded port division.
- **positive-definite advection on ni:** WRF treats ni as a scalar/tracer with `moist_adv_opt=1 /
  scalar_adv_opt=1` (default ON) → monotonic flux limiter (`dyn_em/module_advect_em.F:10392-10444`) keeps ni in
  [qmin,qmax] of neighbors → no negative ni, no overshoot. If the port runs ni without the positive-definite
  limiter, that is a prime suspect.

## B. Dynamics root (mass-pump / vertical) — WRF defaults in Registry + dyn_em
- **epssm = 0.1** (Registry.EM_COMMON:2870) off-centering in the implicit vertical solve (NOT 0.0 explicit).
- **w_damping = 0 by default**, but when on: damp when `vert_cfl > w_damp_on` (=1.0 non-IEVA / 2.0 IEVA),
  term `-sign(w)*w_alpha*(vert_cfl-w_crit_cfl)*(...)`, `w_alpha=0.3` (big_step_utilities_em.F:2687).
- **Rayleigh damping:** zdamp=5000m, dampcoef=0.2 (top-5km sin² layer) — off by default on nests.
- **divergence damping:** `smdiv=0.1` (small-step), `emdiv=0.01` (external mode).
- mass(mu): WRF stores all prognostics as ρ·q and divides by column mass `(c1h*MUT+c2h)` — no bare divide-by-mu;
  the structure ties divisions to MUT so it can't go to zero unguarded.

## Verdict (Explore, to confirm with onset-order)
Most plausible faithful gaps in a port that NaNs ni at 72min on a 1km nest:
1. **Missing positive-definite/monotonic advection limiter on ni** → negative/overshoot ni → NaN (top suspect).
2. **Missing the `pni_ide = MAX(-ni*odts, pni_ide)` sublimation clamp** → unbounded `deposition/xmi`.
3. **Missing the 999e3 ni ceiling + R2 floor** post-microphysics.
4. (if dynamics-root) insufficient w_damping/epssm on the marginally-stable 1km nest → CFL blow-up → ρ/p NaN → ni NaN.

Any fix = restore the WRF behavior above. NO invented clamp beyond what WRF does (project rule: no masking).
