# case3 wind residual — findings (worker/opus/wind-residual)

Base `73574c0` (contains the merged boundary fix d9846a3 + revalidation a70e4cd).
Reuses corpus CPU-WRF only; NO new WRF runs. File ownership respected: only
`surface_layer.py` touched (T2 bracket guard + honest comments); dycore/acoustic/
thompson/boundary_apply untouched.

## STEP 1 — case3 V10/U10: REAL model deficiency, NOT a regime/metric limit

Decisive evidence (`proofs/wind/case3_regime_diagnostic.py/.json`). case3
(init 2026-05-21 18z) has MULTIPLE independent CPU-WRF forecasts of the SAME init
on the SAME d02 (66x159) grid:
  - L3 run1 (133443Z): 0..24 h (the persistence-baseline truth)
  - L2   run (133443Z): 0..19 h (9->3 km parent, genuinely independent — verified
    max|diff| 0.61 m/s vs L3 at +6h, NOT a duplicate)
  - L3 run2 (072630Z): bit-identical to L3 run1 (same nest output) -> discarded

CPU-WRF self-spread (= irreducible forecast uncertainty) vs persistence error,
at the leads where L2 overlaps (3..19 h):

  | field | CPU-WRF L2-vs-L3 self-spread | persistence error | ratio |
  |-------|------------------------------|-------------------|-------|
  | V10   | 0.002 – 0.020 m/s            | 2.2 – 2.8 m/s     | ~0.001–0.009 |
  | U10   | 0.002 – 0.016 m/s            | 1.7 – 2.4 m/s     | ~0.001–0.007 |

For comparison, the PROVEN-skillful case2 V10 self-spread/persistence ratio is
~0.005–0.06. case3 is in the same (or tighter) skillful regime.

Regime is NOT calm: case3 init V10 mean = -5.4 m/s, std 2.1 (strong southerly);
the field evolves (change-from-init RMSE 1.6 -> 3.1 m/s over 24 h). CPU-WRF tracks
that evolution to ~0.02 m/s regardless of parent resolution.

=> case3 V10/U10 are STRONGLY PREDICTABLE. The GPU losing to persistence is a
   REAL model deficiency, not a regime/metric artifact.

Caveat (honest): case3 L2 and L3 share the same d01 lateral forcing, so the tiny
self-spread partly reflects shared LBC making d02 near-deterministic. That does
NOT overturn the conclusion: it still proves the V10 field is highly constrained
and reproducible, i.e. a faithful model SHOULD reproduce it — so a persistence
loss is a real deficiency.

Lead-coverage caveat (GPT-5.5 sidecar, proofs/wind/gpt_sidecar_verdict.md): the L2
self-spread only directly overlaps the L3 truth to lead 19 h, while the failing
score is at 24 h. Addressed by the FULL hourly trend (1..19 h,
case3_regime_diagnostic.json): the V10 self-spread/persistence ratio is flat-tiny
across all 19 h (0.0002 -> peak 0.011 at lead 11 -> 0.0023 at lead 19, NO growth
trend), while persistence error holds 2.2-2.8 m/s. Extrapolating a non-growing,
~0.005 ratio 5 h further to 24 h is well-justified; the GPU's 24 h V10 skill
-0.099 is nowhere near that floor. (No second independent 24 h CPU-WRF case3
forecast exists on disk — the L2 d02 history stops at +19 h, L2rerun has no d02.)

GPT-5.5 sidecar VERDICT (independent): "No sound, low-regression surface_layer.py
lever. Honest close: case3 residual is dominated by prognostic k0 wind deficiency
outside surface-diagnostic scope; mynn_pbl.py remains a possible B2-owned suspect
until budgeted out." — agrees with this analysis.

## STEP 2 — WHERE the deficiency is: the PROGNOSTIC lowest-level wind (dycore/PBL), NOT the surface diagnostic

GPU localization @ 24 h (`proofs/wind/gpu_wind_localize_case3.json/.npz`), over
water (93% of the 10494-cell domain), vs CPU-WRF truth:

  | quantity                       | CPU-WRF | GPU   | GPU bias |
  |--------------------------------|---------|-------|----------|
  | lowest-level u0 (k=0, mass pt) | -0.69   | +1.12 | +1.81 (wrong SIGN) |
  | lowest-level v0 (k=0)          | -7.11   | -4.85 | +2.26 (too weak)   |
  | lowest-level wspd (k=0)        |  7.90   |  6.11 | -1.79  |
  | 10 m wspd diagnostic           |  7.37   |  5.35 | -2.02  |
  | 10 m / k0 ratio                |  0.934  | 0.871 | -0.06  |
  | u* (friction velocity)         |  0.261  | 0.255 | -0.006 (MATCHES) |

Key facts:
1. The GPU lowest-level PROGNOSTIC wind VECTOR is wrong over water in BOTH
   components (u0 wrong sign +1.81, v0 too weak +2.26). The 10 m diagnostic ratio
   (a 7% scalar) cannot fix a vector that is wrong in direction AND magnitude.
2. u* MATCHES WRF (0.255 vs 0.261). So our surface DRAG is faithful; the weak
   wind is NOT caused by excess surface drag (refutes the z0/Charnock lever — a
   z0 change is neither needed nor the cause; u* is already right).
3. Therefore the deficit is in the lowest-level momentum field itself: dycore/PBL
   advection of the synoptic flow + the residual boundary plume advecting into the
   ocean interior (the WIND_SKILL_ROOT_CAUSE.md "deep-interior residual"). This is
   OUTSIDE surface_layer.py ownership.

### The one in-scope lever (diagnostic ratio) is NOT viable — proven by counterfactual

WRF's marine 10 m ratio is 0.934 (median 0.939), tightly ABOVE the neutral-log
ratio 0.896; a stable stability-correction can only go BELOW neutral, so WRF's
MYNN diagnoses the marine layer as near-neutral while our sfclayrev diagnoses it
as stable (zol +0.25) and over-suppresses to 0.871. Real, but worth only ~0.5 m/s.

Counterfactual rescore applying a corrected ratio to the GPU's OWN u0/v0:

  | ratio       | V10 skill | U10 skill |
  |-------------|-----------|-----------|
  | 0.871 (now) | -0.099    | -0.001    |
  | 0.896 (neutral) | -0.081 | -0.013   |
  | 0.934 (WRF marine) | -0.055 | -0.045 |
  | 1.000 (max) | -0.020    | -0.105    |

Raising the ratio HELPS V10 (positive bias on negative truth) but HURTS U10
(positive bias on near-zero truth) — a scalar ratio bump TRADES the two
components and STILL cannot beat persistence (V10 stays a loss even at ratio=1.0).
This confirms: a surface-diagnostic change cannot close case3; the residual is the
prognostic wind, outside surface ownership. (And case2 water V10 is already
balanced at -0.18, so raising the shared ratio would regress case2.)

Roughness-provenance refinement (GPT-5.5 sidecar caveat #3, verified): the ratio
gap is at least partly ROUGHNESS, not stability. WRF MYNN dynamically updates the
ocean z0 (COARE) to ~2.0e-5 m at this wind; our static water znt=0.00285 is ~145x
larger, which alone drops the NEUTRAL-log ratio from 0.933 (z0=2e-5) to 0.896
(z0=2.85e-3) — most of the 0.871->0.933 gap. So "our sfclayrev is more stable" is
an over-attribution; roughness provenance explains much of it. Importantly, our
water u* STILL MATCHES WRF (0.255 vs 0.261) despite the 145x-larger z0, because
the u*=0.5*ust_old+0.5*k*wspd/psix update compensates — so the static z0 is NOT
producing excess drag in practice, and a z0/Charnock change is an unpredictable,
multi-field perturbation that would also hit the already-balanced case2 wind gain.
Not a clean, low-regression lever (a known surface limitation, not a quick fix).

mynn_pbl.py not exonerated (sidecar caveat #2): the k0 wind is prognostic AFTER
dycore + MYNN PBL momentum mixing. It is outside surface_layer.py, but MYNN PBL
(in my ownership) is NOT ruled out without a MYNN-off / PBL-tendency budget /
drag-sensitivity proof. That isolation (dycore vs PBL momentum) is the correct
NEXT proof — and it is a larger, instrumented sprint, not a quick surface tweak.

CONCLUSION STEP 2: case3 wind residual = a PROGNOSTIC lowest-level wind deficiency
(dycore + MYNN-PBL momentum + residual boundary plume), NOT closeable from the
surface diagnostic. The surface scheme is faithful (u* matches WRF; T2/U10/V10
within WRF Fortran-oracle bands). No sound, low-regression surface_layer.py wind
lever exists; the diagnostic ratio trades V10 for U10 and cannot beat persistence.

## STEP 3 — T2 slip (case2 -0.097): a domain-wide PROGNOSTIC warm bias, faithful diagnostic

The boundary fix moved case2 T2 skill +0.073 -> -0.097 (commit d9846a3). The slip
is a warm bias: case2 +1.08–1.18 K, case3 +0.84 K (water). It is DOMAIN-WIDE and
~uniform (case3: frame +0.99, deep interior +0.84, deep box +0.83), NOT
concentrated near the boundary — so it is a systematic prognostic near-surface
warm bias, not a near-boundary-advection artifact.

The T2 DIAGNOSTIC itself is faithful: the REAL WRF Fortran surface-layer oracle
parity (`proofs/wind/surface_mynn_parity_wrf_bracketguard.json`,
`/mnt/data/wrf_gpu2/physics_oracle/surface_mynn`, sf_sfclay_physics=5, one WRF
step) gives T2 RMSE 0.041 K (band 1.5), U10 0.007, V10 0.022 — all PASS. So given
WRF's inputs our T2 reproduces WRF to <0.04 K. The +0.8 K operational bias comes
from our prognostic near-surface state being warmer than WRF (radiation/surface
energy/dycore), upstream of the surface diagnostic and outside surface ownership.
RMSE 1.0–1.3 K is physically fine (~CPU-WRF's own spread). A diagnostic-side
"correction" would be a masking clamp (forbidden).

### Faithful in-scope improvement added (no-op on the validated oracle)

Added WRF MYNN's 2-m theta BRACKET GUARD (module_sf_mynn.F:1140-1144), missing
from our port: th2 must be bracketed by thgb (surface) and thx (lowest level);
falls back to thgb + 2*(thx-thgb)/za when an ill-conditioned psit2/psit pushes it
out. This is WRF reference code (not a clamp). It is a NO-OP on the WRF oracle
columns (T2 RMSE delta 1e-18, U10/V10 deltas ~1e-18 — exact parity preserved) and
only fires on physically-impossible columns; it does not move the prognostic warm
bias. Kept as a fidelity hardening; it does NOT reduce the T2 slip (the slip is
prognostic and outside surface scope).

## STEP 4 — core re-validation (RESULTS)

- WRF Fortran surface-layer oracle parity (the strongest surface check;
  /mnt/data/wrf_gpu2/physics_oracle/surface_mynn, sf_sfclay_physics=5, one WRF
  step): ALL gated fields PASS, exact parity preserved vs the pre-change baseline
  (t2 RMSE 0.0406 K, u10 0.0073, v10 0.0224, hfx 17.37, lh 10.68, ust 0.026,
  psim 0.216, psih 0.244 — all RMSE deltas ~1e-18). The bracket guard is a no-op
  on the validated oracle columns. (proofs/wind/surface_mynn_parity_wrf_bracketguard.json)
- idealized warm-bubble + Straka density current (dry core; physics OFF so the
  surface layer is not even called -> regression guard for the edit/build):
  BOTH PASS, values bit-identical to the pre-change baseline (warm bubble
  theta_prime_max 1.9201, max|w| 11.68, thermal_rise 1924 m, mass drift 0.0;
  Straka theta_prime_min -9.97, front 14150 m, rotor 4, mass drift 2.25e-9).
  (proofs/f2/*verdict.md, proofs/wind/idealized_bracketguard.log: 2 passed 437 s)
- case3 24 h coupled GPU forecast re-run WITH the bracket guard: U10/V10/T2 RMSE,
  bias and skill are BYTE-IDENTICAL to the pre-change baseline (all deltas 0.0:
  U10 2.3919, V10 3.3882, T2 1.1451; skill U10 -0.0012, V10 -0.0989, T2 -0.0042).
  Confirms the guard is a true no-op on the operational forecast (never fires on
  physical columns) AND that the T2 slip is NOT a diagnostic bracket violation
  (T2 RMSE unchanged) -> the T2 slip is prognostic, outside surface scope.
  CORE INTACT, no stability/skill regression.
  (proofs/wind/gpu_wind_localize_case3_bracketguard.json)

## VERDICT

- case3 V10/U10: REAL deficiency (not regime/metric) — PROVEN via CPU-WRF
  self-spread << persistence error on a strong, evolving southerly regime.
- It is a PROGNOSTIC lowest-level-wind deficiency (dycore/PBL + residual boundary
  plume), NOT surface-layer-fixable: u* already matches WRF; the only in-scope
  lever (10 m diagnostic ratio) trades V10 for U10 and cannot beat persistence.
- T2 slip: domain-wide prognostic warm bias; diagnostic is WRF-faithful
  (<0.04 K oracle parity); not reducible from the surface layer without a clamp.
- Surface change made: WRF-faithful 2-m theta bracket guard (no-op on oracle,
  exact parity preserved) + corrected/honest 10 m-diagnostic comments.
