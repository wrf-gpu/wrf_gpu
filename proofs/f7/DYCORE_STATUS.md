# Dry Dynamical Core — Status (single source of truth for the F7 rewrite)

**Last updated: 2026-05-29 (~12:00). Branch `worker/opus/f7d-pressure-mass-fix` (unmerged until idealized cases pass).**
This file exists so future agents do NOT waste tokens re-investigating already-cleared components. Update it when the dycore status changes.

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
| Flat-rest exactly stable (machine-0); mass conserved to 0 over 300+ steps | continuous regression gate |

## OPEN RESIDUAL (one localized issue, as of F7J)
The exponential runaway is GONE. Remaining: the warm bubble **rises but under-translates** (`thermal_rise`≈213 m vs ≥500 m target in 500 s); Straka still NaNs at ~240 s (was 30 s). F7J localizes this to **prognostic-w ↔ continuity-omega (ww) ↔ scalar vertical transport consistency (deformation vs translation)** — distinct from the implicit solve and the geopotential RHS.

**Next step is GROUND-TRUTH-DRIVEN, not a guess:** diff JAX center-column per-substep fields against the WRF savepoints (below). See [[project-dycore-rewrite-status-2026-05-29]] in memory.

## WRF ground truth (the arbiter — USE IT)
Pristine WRF **v4.7.1** (same version as Gen2) built at `/home/enric/src/wrf_pristine/WRF` (gfortran serial, conda env `wrfbuild`; `csh -f ./compile`). Center-column (i=20,j=20) per-acoustic-substep `em_quarter_ss` savepoints at:
- `/mnt/data/wrf_gpu2/wrf_truth/em_quarter_ss_center_savepoints.json` (persistent)
- `proofs/m9/wrf_em_quarter_ss_savepoints.json`
Fields per (step,rk,substep)×41 levels: `w_2, ph_2, p, rw_tend, ph_tend, t_2save, muave, muts, mut` + `a, alpha, gamma, cqw, c2a`; scalars `dts_rk=4.0, epssm=0.1`.
**Do NOT try to CPU-build the canonical Gen2 WRF tree — it's an NVHPC/OpenACC GPU fork that won't build under gfortran (F7I burned 6 attempts).** See [[project-wrf-ground-truth-build-2026-05-29]].

## Honest gate for "dycore CLOSED"
Skamarock warm bubble (rises ≥500 m, bounded w) + Straka density current (front ≈15 km at 900 s, min θ′≈−9..−10 K) both PASS, ideally confirmed by the WRF center-column savepoint diff. Then: GPT-5.5 pre-close critique → merge `f7d` chain → Phase B.
