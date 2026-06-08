# Dry Dynamical Core — Status (single source of truth for the F7 rewrite)

**Last updated: 2026-06-03 (v0.4.0 close). Earlier: 2026-06-01 v0.1.0 refresh; Sprint U 2026-05-29.**
This file exists so future agents do NOT waste tokens re-investigating already-cleared components. Update it when the dycore status changes.

## ✅ v0.4.0 CLOSE (2026-06-03) — 4 WRF-faithful dycore fixes consolidated; standalone wind bias is a DOCUMENTED forecast-skill item, NOT a fidelity bug

See [`.agent/decisions/V0.4.0-CLOSE.md`](../../.agent/decisions/V0.4.0-CLOSE.md) and the proof
object [`proofs/v040/v040_close_proof.json`](../v040/v040_close_proof.json).

Four WRF-faithful dycore fixes were consolidated onto the v0.6.0 integration trunk (`350a7c6`),
each savepoint-PASS vs the WRF oracle, each leaving the idealized/periodic path **bit-identical**
(`max_abs=0.0`) and the v0.1.0/v0.2.0 replay path **unchanged**:

1. **Dry-mass continuity for specified/nested LBCs** (BC-conditional `advance_mu_t` bounds;
   `mu_continuity_savepoint_parity.json` PASS) — collapses the standalone PSFC−/U10+ *drift*.
2. **PGF `al`(alb/muts) + `dpn`(cfn/cfn1)** (`calc_p_rho_phi` + top_lid `dpn` faces;
   `boundary_transport_savepoint_parity.json` PASS) — adjudication: bias **NOT** in large-step terms.
3. **MYNN `s_aw` stability floor on `kmdz`** even with `bl_mynn_edmf_mom=0` (a REAL WRF omission,
   `module_bl_mynnedmf.F:3990-3997`; `r4_saw_floor_savepoint_parity.json` PASS).
4. **Split-explicit `php`-freeze** in the advance_uv 4th PGF term (`calc_php` is `INTENT(IN)`;
   `r5_php_freeze_savepoint_parity.json` PASS).

**The single open item** is a domain-uniform near-surface **westerly excess** in the 24 h
standalone forecast (+1.2 m/s 20260429 / +0.75 m/s 20260521; T2 correct, stable/finite). After 10
debug rounds it is **ruled out vs unmodified WRF against every faithful ported operator and scheme**
(incl. a decisive, independent CPU-WRF Kain-Fritsch cu0-vs-cu1 oracle) — it is **dynamical, not a
fidelity bug**. Expand-dates falsification is DATA_BLOCKED (purged met_em). Carried as
[`.agent/tasks/V0.4.0-WIND-BIAS-CARRYOVER.md`](../../.agent/tasks/V0.4.0-WIND-BIAS-CARRYOVER.md).
**Do NOT** re-investigate the ruled-out angles, loosen tolerances, add a wind clamp, or claim the
bias is fixed.

## ✅ v0.1.0 RELEASE STATUS (2026-06-01) — dycore CLOSED (idealized) + validated on real cases via the operational path

For v0.1.0, the dycore is validated at two levels, both traceable to
[`proofs/PROOF_TABLE.md`](../PROOF_TABLE.md) and the binding contract
[`publish/VERIFICATION.md`](../../publish/VERIFICATION.md):

- **Idealized gates PASS (proof rows 1/2):** Skamarock warm bubble **6/6** and Straka density
  current **6/6** vs the published references + pristine WRF v4.7.1 ground truth. The OPEN-RESIDUAL
  notes below (F7M/F7L "Straka still detonates") are **SUPERSEDED** — the residual was the
  vertical-momentum-advection sign error and a non-conservative constant-K diffusion, both fixed at
  F7N (see the "DRY DYNAMICAL CORE CLOSED (F7N)" section). Straka now PASSES 6/6.
- **Real-case validation via the OPERATIONAL path (proof rows 4/5/7):** the production dycore runs
  the operational `small_step_prep → _rk_scan_step` stepper inside the full coupled forecast and is
  finite/stable over complete runs: d02 3 km 3-case **D02_VALIDATED** (72 h, beats persistence on
  winds), d03 1 km 24 h **D03_1KM_VALIDATED** (T2 RMSE 1.92 K ≤ 3.0 K, beats persistence), and
  conservation **guards-off finite + fp64** on real d02.

### Row 3 (savepoint operator parity) = FAIL — comparator-harness gap, NOT a production-dycore defect

The `scripts/verify/savepoint_parity.sh` row is the one **FAIL** in the proof table, and it is
honestly a **fixture/harness gap, not a dycore defect**:

- The original `theta=None` threading bug in the comparator is genuinely **FIXED**
  (`_seed_coupled_work_theta` now seeds the persistent coupled-work theta WRF-faithfully:
  `theta_work = mass_muts*theta_1 - mass_mut*theta`).
- Fixing it **exposed** a deeper gap: the savepoint oracle is an **hourly `wrfout` history state**,
  not a true per-RK / restart-complete WRF savepoint. The validation-only `coupled_timestep_core`
  path (`rk_stage_core → acoustic_scan_core`; **zero operational callers**) is therefore fed a bare
  `AcousticLoopState.from_mapping` lacking the ~30 `small_step_prep`-derived leaves
  (c2a/alt/al/phb/ph_1/cf*/c1f/c2f/rdn/ht/pm1/rw_tend_pg_buoy …); `calc_p_rho`/`advance_w` then emit
  non-finite `p`/`ph` and the comparator step blows up across all 3 tiers at step 1.
- The comparator asserts `FAIL_COMPARATOR_HARNESS_GAP` (`is_production_dycore_defect=False`) — **not
  masked, not a manufactured pass.** This was **independently confirmed by two models (Opus +
  GPT-5.5).** The production dycore is independently proven by rows 1/2/7 + the d02/d03 real-case
  runs that exercise the operational `small_step_prep → _rk_scan_step` path.
- **v0.2.0 follow-up:** regenerate true per-step / restart-complete WRF savepoints — or route the
  comparator through the operational `small_step_prep → _rk_scan_step` stepper — so row 3 exercises
  a numerically-stable composition. Tracked in `.agent/decisions/V0.2.0-PLAN.md`.

Everything below is the F7-era engineering record (retained for provenance).

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
- **P0-2 WRF deformation momentum diffusion (2D one-row u/w subcase ONLY;
  full 3D u/v/w DEFERRED to Phase B — honest scope, GPT confirm-close)**:
  `wrf_deformation_momentum_tendency` implements the flat-slab **one-row u/w**
  deformation reduction (`du`/`dw` from `defor11/33/13`); it does NOT implement
  the full WRF deformation tensor (`defor22/12/23`, `horizontal_diffusion_v_2`,
  `horizontal_diffusion_u/w_2` multi-row). When deformation mode is enabled, the
  runtime explicitly keeps **v** on the scalar flux-divergence form
  (`operational_mode.py:1206-1216`). Analytic oracle: du matches FD to round-off,
  dw to ~1% + 2nd-order convergence (constant-density u/w closed forms only).
  Straka (2D x-z) PASSES 6/6 WITH the one-row operator (mass drift 1.4e-16).
  **This is sufficient and correct for the 2D Straka density-current gate and is
  NOT on the operational real-case critical path** — see the P0-2 deferral entry
  below.
  (`proofs/sprintU/momentum_diffusion_deformation.md`, `straka_deformation_gate.md`;
  GPT confirm-close `.agent/sprints/2026-05-29-sprintU-operationalize-dycore/gpt-confirm-close-findings.md`)
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

**P0-2 deferral — full 3D u/v/w deformation tensor is NOT on the operational
critical path (Sprint U, GPT-conditionally-blessed deferral, documented honestly):**
The operational real-case path (`daily_pipeline._build_real_case`) runs WRF's
`diff_6th_opt=2` (6th-order numerical filter, `module_big_step_utilities_em.F:6504-6920`)
for momentum/scalar dissipation — it does **NOT** enable `km_opt` deformation
diffusion (`use_deformation_momentum_diffusion=False` in the real-case namelist).
Therefore the full WRF deformation tensor (`defor11/22/33/12/13/23`,
`horizontal_diffusion_u/v/w_2`, `cal_deform_and_div`,
`module_diffusion_em.F:3323/3503-3508`) is **NOT** exercised by the operational
forecast. The implemented **one-row u/w** deformation operator is sufficient for
the 2D Straka x-z density-current gate (the only place deformation diffusion is
turned on). The full 3D u/v/w deformation tensor (incl. v / D22 / D12 / D23, the
multi-row terms, and terrain-slope coupling) is **DEFERRED to Phase B** (terrain /
3D coupling), where it becomes load-bearing. No overclaim: P0-2 is closed only for
the 2D one-row u/w subcase; the stated full u/v/w WRF deformation remediation is a
Phase-B item.

**Honest remaining scope (Phase-B gates, NOT closed):** full 3D u/v/w deformation
tensor (see P0-2 deferral above), 3D terrain slope (zx/zy) diffusion
cross-coordinate terms, map factors, lateral specified/nested boundaries,
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

## (SUPERSEDED by F7N — Straka now PASSES 6/6) OPEN RESIDUAL — F7M localized it to the COLD-POOL TOUCHDOWN with WRF ground truth
**NOTE (2026-06-01): This residual is CLOSED.** F7N traced the Straka detonation to the
vertical-momentum-advection upwind-correction **sign error** (`_vertical_flux_div_3`, opposite sign
to WRF) plus a non-conservative const-K diffusion; both were fixed and Straka now PASSES 6/6 (see
the "DRY DYNAMICAL CORE CLOSED (F7N)" section above). The investigation notes below are retained for
provenance only.

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
Pristine WRF **v4.7.1** (same version as Gen2) built at `~/src/wrf_pristine/WRF` (gfortran serial, conda env `wrfbuild`; `csh -f ./compile`). Center-column (i=20,j=20) per-acoustic-substep `em_quarter_ss` savepoints at:
- `/mnt/data/wrf_gpu2/wrf_truth/em_quarter_ss_center_savepoints.json` (persistent)
- `proofs/m9/wrf_em_quarter_ss_savepoints.json`
Fields per (step,rk,substep)×41 levels: `w_2, ph_2, p, rw_tend, ph_tend, t_2save, muave, muts, mut` + `a, alpha, gamma, cqw, c2a`; scalars `dts_rk=4.0, epssm=0.1`.
**Do NOT try to CPU-build the canonical Gen2 WRF tree — it's an NVHPC/OpenACC GPU fork that won't build under gfortran (F7I burned 6 attempts).** See [[project-wrf-ground-truth-build-2026-05-29]].

## Honest gate for "dycore CLOSED" — MET (idealized) for v0.1.0
Skamarock warm bubble (rises ≥500 m, bounded w) + Straka density current (front ≈15 km at 900 s,
min θ′≈−9..−10 K) **both PASS 6/6** (F7N). For v0.1.0 the production dycore is additionally
validated on real Canary cases via the operational `small_step_prep → _rk_scan_step` path
(D02_VALIDATED / D03_1KM_VALIDATED / guards-off fp64 conservation — proof rows 4/5/7). The
remaining per-operator savepoint comparator (row 3) is a fixture/harness gap, not a dycore defect
(see the v0.1.0 banner at the top), and full-3D-deformation / terrain / map-factor / boundary
dynamics closure are Phase-B / v0.2.0 items, not v0.1.0 dycore blockers.
