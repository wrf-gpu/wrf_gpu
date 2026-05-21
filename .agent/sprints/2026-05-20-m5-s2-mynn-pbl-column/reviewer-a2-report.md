# M5-S2 Attempt-2 Reviewer Report (binding)

Reviewer: Claude Opus 4.7 xhigh, session 2026-05-21.
Commit: `2b7c233 Implement real WRF-linked MYNN2.5 column`.
Scope: binding verdict on attempt-2 vs attempt-1 R-1..R-6, the AC6 contract amendment, and project anti-tautology / no-fudge rules.

## 1. R-1..R-6 fix audit table

| Finding | Verdict | Key evidence |
|---|---|---|
| R-1 kernel was Louis-Blackadar Ri, not MYNN2.5 | resolved | `src/gpuwrf/physics/mynn_pbl.py:178-213` (`_mym_level2`), `:253-302` (`_mym_length_option2`), `:305-388` (`_mym_turbulence`), `:409-449` (`_mym_predict_qke`) transcribe WRF flux-Richardson stability, option-2 master length, level-2.5 Helfand-Labraga limited stability, and prognostic qke with implicit transport and dissipation. WRF source cross-checked at `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/MYNN-EDMF/module_bl_mynnedmf.F90:1868-1923,2221-2350,3380-3500`. |
| R-2 harness was worker-authored proxy | resolved | `scripts/wrf_mynn_harness.f90:2-4,90,92,99,103,116` imports `module_bl_mynnedmf` and calls `get_pblh, mym_turbulence, mym_predict, mynn_tendencies, retrieve_exchange_coeffs`. `scripts/wrf_mynn_harness_build.sh:11-15,57-59` links the four compiled WRF objects. Independent `nm /tmp/wrf_gpu2_s2/data/scratch/wrf_mynn_harness` confirms external `T module_bl_mynnedmf_get_pblh_`, `_mym_turbulence_`, `_mym_predict_`, `_mynn_tendencies_`, `_mym_level2_`, `_mym_length_` symbols. Manifest provenance `wrf-mynnedmf-object-linked-harness sha256=4ed5ce48...` matches build-log SHA at `artifacts/m5/wrf_mynn_harness_build_m5_s2_a2.txt:1`. |
| R-3 Tier-2 was closed-column tautology | resolved-with-caveat | Surface stub now wired into kernel at `src/gpuwrf/physics/mynn_pbl.py:168-175` and consumed at `:420-422` (TKE surface production) and `:482-489` (mean-field bottom drag + theta/qv flux RHS). Tier-2 (`src/gpuwrf/validation/tier2_mynn.py:35-54`) compares column-integrated state delta to `dt*(surface_flux+top_flux)` for u,v,theta,qv and `dt*column_integral(prod+transport-diss)` for TKE. Residuals momentum=3.92e-12, heat=7.61e-11, moisture=6.24e-11, tke=9.22e-3 (`artifacts/m5/tier2_mynn_invariants.json:5,11,16,31`). Caveat: mean-field check is a discrete-consistency probe of the solver against the same boundary terms it received; TKE check is independent. |
| R-4 launch fudge `min(raw, cap)` | resolved | `scripts/m5_run_mynn.py:97-99` writes the same `int(launches)` into `kernel_launches_per_step` and `raw_hlo_launch_marker_count` — no clamp. `src/gpuwrf/profiling/budget.py:43-52` only saturates upward at 1, never downward. Independent recount of full HLO (`/tmp/wrf_gpu2_s2/data/scratch/m5/mynn_pbl_production_full.txt`, 2,808 lines, 278,887 B): 30 `fusion(` + 5 `custom-call(` + 0 `while(` = 35. Matches `artifacts/m5/mynn_profile.json:17,23`. AC6 amendment at `sprint-contract.md:170-179` is properly scoped and gate `scripts/m5_gate_mynn.py:51-58` enforces raw≤35 GO, ≤50 GRAY, >50 FALLBACK. |
| R-5 keep tridiagonal solver | resolved | `src/gpuwrf/physics/tridiagonal_solver.py:13-72` retains XLA primitive wrapper + Thomas reference; called via `_solve_tridiagonal` at `mynn_pbl.py:403-406, 437, 474`. |
| R-6 dead imports / unwired surface stub | resolved | All imports at `mynn_pbl.py:13-49` are exercised: closure constants in `_mym_level2`/`_mym_turbulence`, `LOCAL_*` in `_mym_length_option2`, `bulk_surface_fluxes` in `_surface_terms`, `solve_tridiagonal`, debug asserts at `:516-523`. ADR-008 rewritten with WRF-line citations at `.agent/decisions/ADR-008-mynn-jax-implementation.md:10-44`. |

No new blockers introduced.

## 2. Real-MYNN2.5 verification (R-1 deep)

**Prognostic TKE equation.** `_mym_predict_qke` at `mynn_pbl.py:409-449` builds shear+buoyancy production via `pdk = elq*(sm*gm + sh*gh)` (`:355`), surface production `pdk1 = 2*ustar^3/vkz` with `pdk(0) = pdk1 - pdk(1)` (`:420-422`) — exactly WRF `:3428`. Vertical transport coefficients `a=-dtz*kqdz*rhoinv, b=1+dtz*(kqdz_k+kqdz_{k+1})*rhoinv+bp*dt, c=-dtz*kqdz_{k+1}*rhoinv, d=rp*dt+qke` (`:428-431`) match WRF `:3450-3463` with `onoff=0`. Dissipation `bp = 2*qkw/(B1*el)` with `b1l = B1*0.5*(el(k+1)+el(k))` (`:425-426`) matches WRF `:3442-3443`. Rho-weighted interface `_rho_interfaces` (`:391-400`) matches WRF `:3396-3402`.

**Nakanishi (option-2) master length scale.** `_mym_length_option2` (`:253-302`): `qkw=sqrt(max(qke_interp,qkemin))`, `elt=ALP1*sum(qdz*zw)/sum(qdz)` clamped `[ELT_MIN,ELT_MAX]`, `vsc=(GTR*elt*max(fltv,0))^(1/3)`, stable `elb_mf=ALP2*qkw/bv*(1+ALP3*sqrt(vsc/(bv*elt)))`, convective tau-based unstable branch, `els=KARMAN*zwk` (correct for dry `rmol=0`), final blend `el=sqrt(els²/(1+els²/elt²+els²/elb_mf²))` then transition-blended with `elf`. Matches WRF `:2221-2350` case 2. Cloud `cldavg` and EDMF `qkw_mf` zero in dry mode and dropped — consistent with the harness disabling them.

**Level-2.5 stability functions S_m, S_h.** `_mym_turbulence` (`:305-388`) implements both Helfand-Labraga limiter `sm_hl=sm20*qdiv, sh_hl=sh20*qdiv` (`:329-332`) and unlimited level-2.5 `sm25=q3sq*A1*(e3-3*C1*e4)/eden, sh25=q3sq*(A2*a2fac)*(e2+3*C1*E5C*gmel)/eden` with `e1..e4, eden` exactly the WRF `e1c..e5c, gmel, ghel` algebra (`:341-347`), and Prandtl limiter `sm = min(sm, prlim*max(sh,0.02))` (`:350-352`). No Louis-1979 Ri-based form remains.

**Closure constants from WRF.** `mynn_constants.py:11-30`: `PR=0.74, G1=0.235, B1=24.0, B2=15.0, C2=0.729, C3=0.340, C5=0.2` match WRF `:280-287`. `A1=B1*(1-3*G1)/6`, `C1=G1-1/(3*A1*2.88449914061481660)` (B1^(1/3) verified: 24^(1/3)=2·3^(1/3)=2.88449914...), `A2=A1*(G1-C1)/(G1*PR)`, `G2=B2/B1*(1-C3)+2*A1/B1*(3-2*C2)`, `E1C..E5C` match WRF `:289-301`. `QMIN=0, ZMAX=1, SQFAC=3, QKEMIN=1e-5` match WRF `:305-307`. Numerical sanity: A1≈1.18, C1≈0.13706, A2≈0.111 — consistent with NN2009.

## 3. Real-harness verification (R-2 deep)

The harness imports `module_bl_mynnedmf_common` (`kind_phys, p608, p1000mb, rcp`) and `module_bl_mynnedmf` (the five entry points) at `wrf_mynn_harness.f90:2-4`. Body issues five CALLs at `:90-116`. EDMF/cloud/scalar-flux/sub-detrainment arrays are zeroed in `allocate_column` (`:131-167`) so the dry MYNN2.5 + minimal `mynn_tendencies` path is exercised against the WRF binary.

Build script declares WRF paths at `wrf_mynn_harness_build.sh:9-15`, aborts on any missing object/module (`:31-37`), and links four WRF `.o` files via nvfortran (`:57-59`). Independent ELF binary audit at `/tmp/wrf_gpu2_s2/data/scratch/wrf_mynn_harness` (394,528 B, x86-64): `nm | grep -E "T module_bl_mynnedmf_"` shows 30+ defined symbols including `_get_pblh_, _mym_turbulence_, _mym_predict_, _mynn_tendencies_, _mym_level2_, _mym_length_, _retrieve_exchange_coeffs_` — the WRF MYNN-EDMF object is genuinely linked, not a worker proxy. R-2 conclusively resolved.

## 4. Tier-2 budget-balance verification (R-3 deep)

Surface flux pathway: `_surface_terms` (`mynn_pbl.py:168-175`) computes `ustar, theta_flux, qv_flux, tau_u, tau_v` via bulk stub, then `fltv = (1+P608*qv0)*theta_flux + P608*theta0*qv_flux` (the virtual heat flux WRF uses for buoyancy length) and `rhosfc`. `_apply_mean_tendencies` (`:477-490`) injects `bottom_drag = rhosfc*ustar²/wind` into bottom diagonal of u/v solve (`:482-484`), `theta_rhs = dtz0*rhosfc*theta_flux*rhoinv0` and `qv_rhs = dtz0*rhosfc*qv_flux*rhoinv0` into bottom RHS of theta/qv (`:485-489`). TKE: surface shear production `pdk1` enters at `:420-422`.

Tier-2 (`tier2_mynn.py:35-54`) compares column-integrated state delta to `dt*(surface_flux+top_flux)` for u,v,theta,qv and `dt*column_integral(prod+transport-diss)` for TKE, over 10 scan steps. Residuals (`tier2_mynn_invariants.json:5,11,16,31`): momentum 3.92e-12, heat 7.61e-11, moisture 6.24e-11, TKE 9.22e-3 (tol 2e-2). Mean-field residuals are not trivially zero — attempt-1 was ~1e-16 because there was no flux input at all; the ~1e-11 here is the small consistency error of the implicit flux-form operator under rho-weighted diffusivity.

**Caveat (carried forward, not a blocker)**: the mean-field check uses the same `drag, K_top, theta_flux, qv_flux` for both expected and actual branches, so it is principally a discrete-conservation test of the solver, not an independent physics validation. The TKE residual integrates interior production and dissipation and is the genuine single-step physics constraint. The load-bearing independent oracle is Tier-1 vs the WRF object-linked harness, which lands at `u≤7.7e-4, theta≤6.3e-5, tke≤1.5e-6, el≤3.1e-3` — that is the binding validation and it is strong.

## 5. Launch-count audit (R-4 deep)

- **No fudge**: `m5_run_mynn.py:97-99` writes `int(launches)` into both `kernel_launches_per_step` and `raw_hlo_launch_marker_count`. `budget.py:43-52` saturates upward at 1 only; never downward at any cap.
- **Independent recount** on full HLO (`/tmp/wrf_gpu2_s2/data/scratch/m5/mynn_pbl_production_full.txt`, 2,808 lines, 278,887 B): 30 `fusion(` + 5 `custom-call(` + 0 `while(` = 35, matches `mynn_profile.json:17,23`. No `while` loops means tridiagonal solves expand directly via XLA primitive (5 calls — qke + u + v + theta + qv) and the rest of the algebra fuses into 30 kernels.
- **6 → 35 regression**: attempt-1 was a flat Louis/Blackadar pass with one tridiagonal. Attempt-2 has the PBLH diagnostic, option-2 BL-integrated `elt` with masked sums, the branchy Helfand-Labraga path with `where(helfand, sm_hl, sm25)`, five implicit solves, and budget diagnostics. The branchy structure prevents XLA from packing the step into a single fusion. The 5 `custom-call`s are the real-physics cost; the 30 `fusion`s are reducible only with hand-fusion if profiler evidence shows launch-bound cost.
- **HLO 278,887 B < 300,000 B** ceiling. Committed `mynn_pbl_production.txt` is truncated head (~63 KB) per `budget.write_hlo` 100 KB rule; full file in scratch.
- **AC6 amendment** (`sprint-contract.md:170-179`) bounds raw ≤ 35 only for attempt-2 and only if (raw=reported) ∧ (HLO ≤ 300 KB) ∧ (transfer counters = 0). All four met. Gate (`m5_gate_mynn.py:51-58`) enforces this and downgrades to GRAY-ZONE above 35.

The amendment is defensible because (a) per `feedback_validation_philosophy.md` the operational binding metric is Tier-4 RMSE not raw launches, (b) real level-2.5 + option-2 length physics has inherently larger fusion budget than the prior trivial proxy, (c) 35 launches per column step on RTX 5090 is not the bottleneck for M5 wall-clock targets. Reducing to ~10 launches is a defensible M5-S2.x optimization if profiler evidence shows it, not a milestone-close blocker.

## 6. Honest accounting of "dry MYNN2.5"

**Dry** (disabled): EDMF mass-flux arrays (`edmf_w,edmf_a,edmf_w_dd,edmf_a_dd,s_aw*,sd_aw*`) allocated and zeroed (`wrf_mynn_harness.f90:138-166`); JAX kernel has no EDMF code path. Cloud arrays (`cldfra,cldfra_bl1,ql,qc,qi,qs,qnc,qni,qnwfa,qnifa,qnbca`) zeroed; `qkw_mf` term in option-2 length omitted. Moisture/theta variance (`tsq,qsq,cov,vt,vq`) zeroed; gradient definition uses the `use_buoy=false` theta-l-v branch. Surface layer is bulk neutral, not WRF Monin-Obukhov; `rmol=0` everywhere.

**Full MYNN2.5 present**: WRF level-2 flux Richardson stability; WRF option-2 master length (els/elb/elt/elf blend); WRF level-2.5 Helfand-Labraga limited sm25/sh25; WRF prognostic qke with implicit transport, surface production, dissipation; WRF rho-weighted interface diffusivities; WRF tridiagonal-form implicit solves for u/v/theta/qv; surface flux RHS injection at bottom level matching WRF `mynn_tendencies`.

**Will dry MYNN2.5 match Gen2's full MYNN-EDMF for M6 operational validation?** Acceptably for a first M6 baseline. In stable nocturnal Canaries BL, EDMF is mostly inactive and dry MYNN2.5 ≈ full MYNN-EDMF for U10/V10/T2. In daytime convective BL over land, EDMF actively transports heat/moisture upward; dry will underestimate BL growth and over-mix scalars near the inversion (likely 0.5-1 K / 0.2-0.5 g/kg worse on T2/qv2 at peak heating). Per `feedback_validation_philosophy.md`, the binding M6 metric is GPU-vs-CPU RMSE on U10/V10/T2 relative to CPU-vs-observation noise — if dry MYNN2.5 lands inside that envelope on a representative day, the EDMF gap is acceptable for milestone close and full EDMF becomes P2.

**Defensibility as carry-forward**: yes, with explicit disclosure. The dry-vs-full gap is named in `worker-a2-report.md:73`, ADR-008 `:12,44`, and the contract Non-Goals already declared MYNN2.5-only with EDMF as M5-S2.x or M6 (`sprint-contract.md:13`). Crucially, the Tier-1 oracle is built against the WRF object-linked harness, so when EDMF is added the oracle is regenerated with the EDMF flag flipped and parity re-tested — no new anti-tautology hole. This is a physics scope limitation, not a validation construct problem; it is not the same class as attempt-1's worker-authored proxy.

## 7. Adversarial probe

Load-bearing claim probed: `C1 = G1 − 1/(3·A1·B1^(1/3))` with WRF/JAX both using the literal `2.88449914061481660` for `B1^(1/3)`. Symbolic: `24 = 2³·3 → 24^(1/3) = 2·3^(1/3) = 2.88449914061481659...` — match to 15 digits. JAX `mynn_constants.py:21` literal matches WRF `:290` literal. No counterexample.

Second probe — `_mym_level2` flux-Richardson at `mynn_pbl.py:202-205`: JAX guards `radicand = max(ri²-ri3·ri+ri4, 0)` while WRF does plain `SQRT(ri²-ri3·ri+ri4)` (`:1919`). Counterexample: if `ri²-ri3·ri+ri4 < 0` (discriminant condition `ri3²-4·ri4 > 0` and `ri` in the negative-radicand interval), WRF NaNs and JAX clamps to zero. The Tier-1 fixture doesn't hit this corner (theta_max_abs_err = 6.3e-5 vs WRF), so the deviation is documentable but not load-bearing. Flag as M5-S2.x: either remove the guard for bit-match WRF behavior or document as intentional defensive deviation. Not a blocker.

No counterexample defeats the attempt-2 claims.

## 8. Binding decision

Reviewer decision: **ACCEPT-WITH-MINOR-FOLLOWUPS** for M5-S2 close as `GO_CARRYFORWARD`.

Justification: all six attempt-1 R-findings are resolved with file:line evidence. Kernel is real WRF MYNN2.5 (verified algebra). Harness genuinely links compiled `module_bl_mynnedmf` (verified via `nm` external symbols). Tier-1 parity against the real WRF oracle is tight (`u≤7.7e-4, theta≤6.3e-5, tke≤1.5e-6, el≤3.1e-3`) and well inside the carry-forward manifest tolerances. Tier-2 is non-tautological with informative TKE residual 9.2e-3. Launch count is the raw HLO marker count (35 = 30 fusion + 5 custom-call) with no clamp; AC6 amendment properly scoped and gate-enforced. HLO 278 KB < 300 KB. Debug-vs-stripped HLO diff = 0 bytes. 410 pytest pass / 0 fail. ADR-008 rewritten with WRF source-line citations. The dry-vs-full MYNN-EDMF gap is disclosed and contract-scoped; it does not break the validation construct.

Follow-ups (NOT M5-S2 blockers — schedule for M5-S2.x or M6 prologue):
1. M5-S2.x deferrable: profile-driven attempt to reduce raw launches 35→~10 via hand-fusion, only if nsight evidence shows launch-bound cost on RTX 5090.
2. M5-S2.x deferrable: add an independent mean-field budget probe (e.g., WRF-harness flux vs JAX flux at same state) so Tier-2 momentum/heat/moisture is not just a solver self-consistency check.
3. M5-S2.x deferrable: resolve the flux-Richardson `radicand` guard at `mynn_pbl.py:202` — either remove for bit-match or document as intentional defensive deviation.
4. M6 prologue: add EDMF mass-flux path (full MYNN-EDMF) once daytime convective-BL T2/qv2 RMSE evidence demands it; the WRF object-linked harness construction already supports re-running with EDMF active.

Manager action: merge `worker/codex/m5-s2-mynn-pbl-column` (commit `2b7c233`) to main, mark M5-S2 closed as `GO_CARRYFORWARD`, file the four follow-ups above. Do not gate M5 milestone close on (1)–(3); (4) is M6 by contract Non-Goals.

Reviewer decision: ACCEPT (close M5-S2 as GO_CARRYFORWARD; merge to main; schedule four documented follow-ups; revive M5 milestone close alongside Thompson).
