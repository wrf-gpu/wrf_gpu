# Dry Dynamical Core — Status (single source of truth for the F7 rewrite)

**Last updated: 2026-05-29 (Sprint U, operationalize+harden). Branch `worker/opus/f7d-pressure-mass-fix`.**
This file exists so future agents do NOT waste tokens re-investigating already-cleared components. Update it when the dycore status changes.

## ✅ DRY DYCORE OPERATIONAL-READY FOR PHASE B (Sprint U, 2026-05-29)
Sprint U closed the 4 P0 + 3 P1 GPT pre-close findings. The dry dycore is now
**operationally unified, WRF-validated, and CI-gated** — the operational/real-case
path uses the SAME validated F7 operators as the idealized gates.

- **P0-1 operational unification**: `daily_pipeline._build_real_case` now builds the
  real-case namelist with the F7 operators (flux advection incl. F7N sign fix, fp64,
  diff_6th_opt=2, WRF Rayleigh+w damping, open top). `run_forecast_operational`
  matches the idealized harness **BITWISE** over 50 warm-bubble steps (theta/w
  linf=0.0 → same dycore), and the full warm bubble PASSES 6/6 through the
  operational entry point. Real Canary d02 (44×66×159) builds + runs finite.
  (`proofs/sprintU/operational_path_unification.md`, `real_case_smoke.json`)
- **P0-2 WRF deformation momentum diffusion**: `wrf_deformation_momentum_tendency`
  wired for u/v/w (theta keeps the conservative scalar flux-divergence). Analytic
  oracle: du matches FD to round-off, dw to ~1% + 2nd-order convergence. Straka
  PASSES 6/6 WITH the deformation operator (mass drift 1.4e-16).
  (`proofs/sprintU/momentum_diffusion_deformation.md`, `straka_deformation_gate.md`)
- **P0-3 canonical-WRF Straka parity**: reran under canonical em_grav2d_x controls
  (dt=1, 6 acoustic substeps==time_step_sound, damp_opt=0, nz=64); array-level
  WRF-vs-JAX through touchdown PASSES — worst max|w| rel diff 0.119 (5% at the 240s
  touchdown peak, 0% at 300s), front within 400m, finite through 360s. The dycore
  DECELERATES like WRF instead of the old runaway → NaN.
  (`proofs/sprintU/straka_canonical_parity.{json,md}`)
- **P0-4/P0-5 CI close-gate**: `tests/idealized/test_dycore_close_gate.py` +
  `close_gate` marker assert `verdict==PASS` (not `in {PASS,FAIL}`) and archive the
  proof JSON; existing idealized tests tightened to assert PASS. Both PASS (411s).
  (`proofs/sprintU/ci_close_gate.md`, `proofs/sprintU/close_gate/`)
- **P1-5 advect_w open-top face**: WRF top-face flux + lid pickup
  (`module_advect_em.F:6014-6028`) wired behind `top_lid`; rigid-lid idealized path
  byte-unchanged (top tendency stays 0). 4 unit tests.
  (`proofs/sprintU/advect_w_topface.md`)
- **P1-6 guards-off proof**: warm bubble PASSES 6/6 fully guards-off; real Canary
  dycore finite guards-off. The theta limiter is a safety net, NOT load-bearing.
  (`proofs/sprintU/guards_off_operational_proof.json`)

**Honest remaining scope (Phase-B gates, NOT closed):** 3D terrain slope (zx/zy)
diffusion cross-coordinate terms, map factors, lateral specified/nested boundaries,
moist/scalar coupling through the RK bundle, and per-cell (vs time-series) WRF field
parity. The operational ADVECTION/DIFFUSION/DAMPING/PRECISION operators are unified
and validated; terrain/map-factor/boundary coupling remains for Phase B.

## ✅ DRY DYNAMICAL CORE CLOSED (F7N, 2026-05-29)
Both idealized gates PASS against published references + WRF ground truth:
- **Skamarock warm bubble: PASS 6/6** (thermal_rise 1924 m, max|w| 11.68, θ′max 1.92, mass drift 0).
- **Straka density current: PASS 6/6** (finite to 900 s; front **14.15 km**; θ′min **−9.97 K**; max|w| **14.57**; **4 rotors**; mass drift **2.25e-9**).
- m4 **10/10**; flat-rest machine-zero; no masking clamps.

**F7N root cause + fix (the close):** the per-acoustic-substep WRF `em_grav2d_x`
touchdown-column diff (`proofs/f7n/touchdown_substep_diff.json`,
`touchdown_fix.md`) localized the Straka detonation to a **growing 2Δz vertical
mode in `u`** in the cold-pool descent layer (z≈2000–4000 m) that pumped omega
(`ww`) and ran w→NaN. Cause: `flux_advection._vertical_flux_div_3` (the
`advect_u`/`advect_v` 3rd-order vertical flux) applied the upwind correction with
the **opposite sign** to WRF (`module_advect_em.F:1474-1480, :202-204` →
`vflux = vel*flux4 - |vel|*corr`), making it **anti-dissipative**. Fixed to the WRF
sign. Secondary: replaced the non-conservative `mass*K∇²` const-K diffusion with
the WRF flux-divergence form (`conservative_constant_k_diffusion_tendency`,
`module_diffusion_em.F:2999-3018`) → mass drift 3.4e-8 → 2.25e-9. Bisection proof:
disabling vertical momentum advection removed the mode (it was *generating* it,
not under-damping). The scalar/`w` vertical-flux paths were already sign-correct.

## Why the F7 rewrite exists
The pre-reset "dycore done, bitwise WRF parity at 100 steps" was a **JAX-vs-JAX self-compare tautology** (discredited 2026-05-28) — the operational dycore was actually missing ~7 WRF operators and produced fast-but-wrong forecasts. The F7.A–J chain is the honest rebuild, validated against **published idealized-case references** (Skamarock warm bubble, Straka density current) and now against **pristine WRF v4.7.1 ground-truth savepoints**.

## VERIFIED-CORRECT — do NOT re-investigate
| Component | How verified |
|---|---|
| Acoustic small-step core + WRF cadence (advance_uv→advance_mu_t→advance_w→calc_p_rho per substep) | F7.A; 12-step transaction audit |
| Implicit w/ph solve + `calc_coef_w` + `epssm` off-centering | F7I: line-by-line WRF match, injected 2Δz mode damped (0.996/substep), epssm-independent. The "2Δz mode" framing was REFUTED. |
| Flux-form WS5/3 advection | F7.B: 5th-order convergence proof |
| MUT/MUTS total-mass semantics (`mut=MUB+MU_cur`, `muts=mut+mu_work`) | F7D, GPT-verified |
| grid%p refresh from finished ph'/θ + full-perturbation grid%p for pg_buoy_w | F7H: cut warm-bubble max|w| 10× |
| `calc_p_rho_phi` geopotential term (`rdnw·Δph'` + full-θ EOS) | F7F (was dropped → dead bubble) |
| `rhs_ph`/`ph_tend` (full 4-term WRF, was a stub=0) + `advect_w` fold | F7J: **killed the exponential vertical standing mode**; warm bubble now rises coherently, finite past 600s |
| Persistent coupled work-theta across acoustic substeps (couple-once / advance-N / decouple-once) | F7K: was re-coupled+decoupled EVERY substep → theta advanced only 1/N_sound of correct (warm bubble rose 213 m not ~2000 m). Fixed: advance `theta_coupled_work`. **Skamarock warm bubble now PASSES 6/6 (thermal_rise 1925 m).** |
| Flat-rest exactly stable (machine-0); mass conserved to 0 over 300+ steps | continuous regression gate |

## OPEN RESIDUAL — F7M localized it to the COLD-POOL TOUCHDOWN with WRF ground truth (Straka still detonates)
**F7M built pristine WRF v4.7.1 `em_grav2d_x` (the Straka case) ground truth and
diffed it against JAX.** Decisive result (`proofs/f7m/wrf_vs_jax_straka_front.json`,
`proofs/m9/wrf_em_grav2d_x_front_savepoints.json`):
**JAX and WRF AGREE to ~3% through 180 s** (both peak ~21 m/s central downdraft at
z~2050 m; w@1100m −17.4 vs −17.5) — buoyancy/descent/acoustic/advection are correct
up to touchdown. **At touchdown (180→200 s) they diverge:** WRF's central downdraft
DECELERATES (21→19→18→15 at 180/240/300/360 s) as the cold air spreads along the
rigid floor (front 2650→5750 m); JAX's central downdraft ACCELERATES (21.1→29.5→NaN)
while its front crawls (2350→2650 m). The runaway is a **smooth central downdraft**
(NOT 2Δx, NOT the front, NOT the top — top w<0.13) at x=0, z~1100–2050 m: the cold
pool reaches the surface but JAX **fails to convert vertical motion into horizontal
outflow** (GPT probe: u_outflow 25–36 m/s while front crawls ~5 m/s) → trapped
descending air → w→NaN ~220–240 s.
**F7M ruled OUT by ground truth:** advection FORM (flux-form WRF `advect_u/v/w`
implemented+wired — ~4% diff, trace byte-identical, still NaN); diffusion
MAGNITUDE/STRUCTURE (deformation-tensor const-K, factor-2 diagonal + du/dz↔dw/dx
cross terms, implemented, ~2–3× stronger — trace byte-identical, still NaN); CFL;
time discretization; top/Rayleigh damping; lower-BC w; scalar limiter.
**→ residual is the TOUCHDOWN horizontal-spreading coupling, NOT advection/diffusion.**
Next instrument: per-acoustic-substep WRF savepoint diff at the touchdown column
(center, z<1500m, t=180–200s) to resolve omega/ww continuity vs advance_uv acoustic
PGF vs surface mass coupling. **F7M_PARTIAL** (warm bubble still PASS 6/6; m4 10/10).
KEPT: WRF-faithful flux-form momentum advection. Operator available (not wired):
`constant_k_deformation_momentum_tendency`.

### (superseded) F7L residual notes

**F7L found+fixed a genuine missing operator** but Straka is NOT fully closed.
The F7.B constant-K (ν=75) diffusion was wired only on u, v, θ, but WRF's
`diff_opt=2` const-K path diffuses **u, v, w AND θ** (`module_diffusion_em.F:2864-3113`
calls `horizontal_diffusion_w_2` at :2999; `:4004-4458` `vertical_diffusion_2`
likewise; Straka et al. 1993 define ν=75 on u, w, θ). **F7L fix**
(`operational_mode.py` `_augment_large_step_tendencies`, committed): add
`mass_f*K*∇²w` in the `nu>0` block. Decisive A/B: nu0 NaN at 240 s; nu75+w-diff
**finite past 240 s** (max|w|=23.9 at 240, ramp flattening) — but it **still
detonates between 240–300 s** at the cold-pool touchdown. max|w|=23.9 m/s at 240 s
EXCEEDS the canonical Straka ν=75 reference (~12–18) while the gust front is only
~2.65 km from center (reference head is further along): **excess vertical velocity
+ sluggish lateral spreading** → a residual operator/coupling defect at the
descending sharp cold front (candidate: gust-front horizontal-PGF→cold-pool-outflow
conversion or descending-front lower-boundary w handling), NOT mere under-diffusion.
Acoustic CFL was never the issue (c·dts/dx≈0.035); emdiv=0.01/smdiv=0.1/w_damping=1/
damp_opt=3 already active+WRF-correct. **F7L_PARTIAL** (no ad-hoc clamps per the
hard rule). Warm bubble unaffected (ν=0 ⇒ block skipped; still **PASS 6/6**).
See `proofs/f7l/straka_diffusion_fix.md` + `straka_wdiff_compare.json`.

**Next step (F7M):** target the descending-cold-front residual — multi-angle:
(1) gust-front horizontal-PGF / cold-pool u-outflow vs WRF at the sharp ground
front; (2) descending-front w lower-BC; (3) cold-pool front-speed deficit
(u too weak ⇒ air sinks not spreads). NOT more diffusion (ν=75 is the spec).

## WRF ground truth (the arbiter — USE IT)
Pristine WRF **v4.7.1** (same version as Gen2) built at `/home/enric/src/wrf_pristine/WRF` (gfortran serial, conda env `wrfbuild`; `csh -f ./compile`). Center-column (i=20,j=20) per-acoustic-substep `em_quarter_ss` savepoints at:
- `/mnt/data/wrf_gpu2/wrf_truth/em_quarter_ss_center_savepoints.json` (persistent)
- `proofs/m9/wrf_em_quarter_ss_savepoints.json`
Fields per (step,rk,substep)×41 levels: `w_2, ph_2, p, rw_tend, ph_tend, t_2save, muave, muts, mut` + `a, alpha, gamma, cqw, c2a`; scalars `dts_rk=4.0, epssm=0.1`.
**Do NOT try to CPU-build the canonical Gen2 WRF tree — it's an NVHPC/OpenACC GPU fork that won't build under gfortran (F7I burned 6 attempts).** See [[project-wrf-ground-truth-build-2026-05-29]].

## Honest gate for "dycore CLOSED"
Skamarock warm bubble (rises ≥500 m, bounded w) + Straka density current (front ≈15 km at 900 s, min θ′≈−9..−10 K) both PASS, ideally confirmed by the WRF center-column savepoint diff. Then: GPT-5.5 pre-close critique → merge `f7d` chain → Phase B.
