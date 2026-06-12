# V0.14 Switzerland venting ROOT CAUSE + FIX — theta-limiter ceiling (500 K) heat pump

Branch `fable-midlevel-momentum` (base `worker/gpt/v013-close-manager` @ 0cd8af7d).

## Verdict

The h37 depth-8 **−26.5 Pa/cell/h venting excess is NOT a momentum-term defect**.
It is the hydrostatic response to a steady **spurious interior heating of
~+0.5 K/h** manufactured by the *theta safety-net limiter*
(`operational_mode._positive_definite_theta_increment_limiter`, envelope
`[0, 500] K`): for this domain (p_top = 5000 Pa, e_vert = 45) the REAL top-level
(k43) potential temperature reaches **507.5 K, > 500 K over 62–98 % of the
domain from h33 on**. The guard, believed non-load-bearing since the 2026-06-01
fix, fired EVERY step:

1. interior k43 theta is crushed to ≤ 500 K (GPU h37 k43 frac>500 = 0.10 vs CPU
   0.69 from the identical h36 reinit state);
2. `apply_lateral_boundaries` runs AFTER the guard and restores the wrfbdy
   >500 K boundary band, so every step re-clips the ring (+3..7 K × ~10³ cells);
3. the limiter's mass-conserving redistribution spreads the clipped theta·mass
   over the whole domain ∝ headroom (500−θ) → a steady, smooth, domain-wide
   tropospheric warming (+0.30 k00 … +0.61 k06 … +0.48 k15-20 … 0 k30, −0.12
   k40 after ONE hour) — the measured profile matches the headroom shape;
4. heated columns expand, mid/upper levels diverge, dry mass exports through
   the depth-8 surface = the venting; the low-level over-strong inflow /
   mid-level outflow vertical U dipole is the secondary (heat-low) circulation.

## Evidence chain (all from the identical-state h36 reinit)

| probe | result |
|---|---|
| forecast budget (`switzerland_midlevel_momentum_budget.py`) | +0.3..+0.6 K interior warm bias at k00-k25 after 1 h, growing ~linearly to h38; net column heating ≈ several 100 W/m² — beyond ANY physics scale |
| subsidence bookkeeping | explaining the warming adiabatically needs ~40× the observed mass export → warming is the CAUSE, not the effect, of the venting |
| radiation-off run (`gpu_output_norad`) | warm bias + u-dipole unchanged → radiation exonerated |
| mp=0 run (existing `gpu_output_nomp2`) | unchanged → microphysics exonerated |
| ALL-physics-off run (`gpu_output_nophys`) | k15-25 warm bias identical (+0.49/+0.52/+0.28 at k15/20/25) → 100 % dycore |
| QKE profile | dead above k12 → MYNN cannot touch k15-25 |
| operator oracle (`switzerland_theta_advection_operator_oracle.py`) | JAX flux-form theta advection + calc_ww_cp omega vs WRF-literal numpy at the same h36 state: mean diff ~1e-19 K/s, rom max diff 1.4e-14 → advection operators bit-faithful |
| k43 theta census | CPU h36 frac>500 = 0.616 (max 507.5); GPU h37 interior crushed to ≤500 (frac 0.10 = boundary band only); continuous-run warm-bias takeoff at h33-h35 EXACTLY tracks the arrival of >500 K air (h01: 0 %, h33: 20 %, h36: 62 %, h48: 98 %) — explains the day-1 immunity and day-2/3 runaway of the 72 h run |
| five refuted momentum lanes (prior rounds) | consistent: every momentum/acoustic/boundary fix left the excess at −26.5 because none of them touched the heat source |

## Fix (WRF-faithful: WRF has NO theta clamp)

`src/gpuwrf/runtime/operational_mode.py`: `_THETA_LIMITER_MAX_K` 500.0 → 1000.0
(theta(1 hPa) ≈ 870 K; unreachable for any p_top this port runs, so the guard
returns to a genuine NaN/blow-up trap — `limited_mask` all-False on physical
states, redistribution residual identically 0). One-line constant change; the
limiter mechanism itself is untouched. `tests/savepoint/test_dycore_limiter.py`
updated to the corrected envelope (3/3 pass).

## Binding-metric confirmation (PASS)

`switzerland_guardfix_venting_budget.{py,json}` — production-config 3 h reinit
run (`gpu_output_guardfix`, physics ON, guards ON, ONLY the ceiling changed) vs
CPU truth:

| metric | baseline (phys_tendf) | guardfix |
|---|---:|---:|
| depth-8 excess outflux h37 (Pa/cell/h) | **−26.54** | **+8.54** |
| depth-8 excess h37→h38 window | −26.9 (speccad) / −21.5 | +14.91 |
| depth-8 excess h36→h38 cumulative per h | ~−21.5..−26.9 | **+6.37** |
| CPU own residual (metric noise floor) | ±5.2 | — |
| interior warm bias h37 (k0/k6/k15/k20, K) | +0.30/+0.61/+0.48/+0.47 | −0.23/+0.01/−0.03/+0.01 |
| interior warm bias h38 (K, max over k) | +1.28 | ±0.21 |
| k43 frac(theta>500K) h37 (CPU 0.693) | 0.100 (crushed) | 0.667 (free) |
| mu bias h38 (Pa) | −83.4 | +19.2 |
| u/v/t/w rmse h38 | 1.17/0.83/0.85/0.140 | 0.86/0.68/0.49/0.105 |
| u-dipole bands h38 (k00/k01-07/k10-24/k27-33, m/s) | +0.30/+0.45..1.02/−0.65/+0.25 | +0.12/+0.19/−0.08/+0.18 |
| stability | — | h39 finite, u rmse 0.96, w rmse 0.12, no growth pathology |

The venting excess collapses from −26.5 to +6..+15 (sign REVERSED, magnitude at
the metric's CPU-residual noise scale); the spurious heating, the vertical
U dipole, and the top-level theta crush all collapse together. The residual
small positive excess (GPU retains slightly more mass) is a separate,
order-smaller lane (mu bias +19 Pa @h38 vs −83 baseline).

Limiter identity proof at the fixed ceiling: on the REAL h36 state the guard
limits **0 cells** (max |Δ| = 0.0) vs **10,100 cells** (= 61.6 % of k43) that
the old 500 K ceiling clipped EVERY step.

Blast radius note: Canary d02 (May run) k43 theta reaches 502.8 K with 13.9 %
of the domain >500 K — the same latent pump was active on the Canary domains
whenever >500 K stratospheric air was present (candidate KI-9 mu/psfc-drift
contributor). The fix benefits all domains.
