# F7L Worker Report — Straka density-current diffusion close

**Status: `F7L_PARTIAL` (strong).** Found and fixed a **genuine missing
WRF operator** — the constant-K (ν=75) diffusion was applied only to u, v, θ
but WRF's `diff_opt=2` const-K path diffuses **u, v, w AND θ**. Adding the
WRF-faithful `K∇²w` term **moved the Straka detonation from 240 s (the F7K
failure) to between 240–300 s** (the cold-pool touchdown), proving the diagnosis,
but it is **not sufficient** to reach 900 s. The **Skamarock warm bubble still
PASSES 6/6** (inviscid, unaffected) and m4 regression is 10/10. Per the F7L hard
rule (only benchmark ν=75 + WRF damping/CFL, no ad-hoc clamps), Straka still
detonates → mark **PARTIAL** with the trace + best diagnosis and STOP.

## Objective
Close the Straka density current WRF-faithfully (ν=75 const-K + WRF
damping/CFL); confirm the warm bubble stays PASS; finish the dry dynamical core.

## Root cause found + fixed (the real advance)
F7K's Straka detonation at 240 s (max|w| 7→15→21→NaN) is a **vertical-velocity
grid-scale runaway**. The F7.B `constant_k_diffusion_tendency` was wired in the
large-step dry tendency (`operational_mode.py` `_augment_large_step_tendencies`,
`nu>0` block) for **u, v, θ only**. WRF's constant-K path diffuses **w too**:
- `dyn_em/module_diffusion_em.F:2864-3113` `horizontal_diffusion_2` calls
  `horizontal_diffusion_w_2` (:3519, invoked :2999) and `_s` for θ (:3711).
- `:4004-4458` `vertical_diffusion_2` calls `vertical_diffusion_w_2` (:4688) and
  `_s` (:4789).
Straka et al. (1993) define the reference with ν=75 on u, **w**, θ. The const-K
form was verified WRF-faithful in F7.B (coupled `g·(H3(k+1)−H3(k))/dnw` with
`H3=−Kρ·Δvar·rdz` reduces to `μ·K·∂²var/∂z²` on the flat hydrostatic grid =
`mass_f·K∇²`).

## The fix (1 line family, WRF-faithful, no masking) — COMMITTED
`src/gpuwrf/runtime/operational_mode.py`, in the `nu > 0.0` block:
```python
w_t = w_t + mass_f * constant_k_diffusion_tendency(haloed.w, k_m2_s=nu, dx_m=dx, dy_m=dy, dz_m=dz)
```
Gated on `const_nu_m2_s > 0`; warm bubble (ν=0) skips the entire block and is
byte-for-byte unaffected. The w-diffusion folds into the stage `rw_tend`
(`operational_mode.py:716`, `rw_tend_stage += tendencies.w`) so it is applied
every acoustic substep, exactly as the buoyancy is.

## Decisive A/B evidence (`proofs/f7l/straka_wdiff_compare.json`)
dx=dz=100m, dt=0.1s, 10 acoustic substeps; emdiv=0.01, smdiv=0.1, w_damping=1,
damp_opt=3 dampcoef=0.2 zdamp=3000, top_lid — all WRF defaults, all active.

```
t(s)   nu=0                       nu=75 on u,v,w,θ (F7L)
180    maxw 21.3 (=F7K)           maxw 21.2
200    maxw 23.5                  maxw 22.6
240    NaN (DETONATE = F7K)       maxw 23.9   FINITE   <-- fix crosses 240s
300                               NaN (detonate ~touchdown)
```

## Why still PARTIAL (honest diagnosis)
max|w|=23.9 m/s at 240 s is **above** the canonical Straka ν=75 reference
(~12–18 m/s) while the gust front is only ~2.65 km from center (reference head is
further along). This **excess w + sluggish lateral spreading** (cold air sinking/
oscillating rather than spreading), with detonation exactly at the cold-pool
touchdown (~270–300 s), points to a residual **operator/coupling defect at the
descending sharp cold front**, not mere under-diffusion (ν=75 is the spec, and
the acoustic CFL is a trivial 0.035 — never the issue). Most likely candidates:
gust-front horizontal-PGF → cold-pool u-outflow conversion (u too weak ⇒ sinks
not spreads), or the descending-front w lower-boundary handling.

## Files changed
- **M** `src/gpuwrf/runtime/operational_mode.py` (+w-diffusion line + WRF
  citation comment; commit `f405418`).
- **M** `proofs/f7/DYCORE_STATUS.md` (residual section updated, honest PARTIAL).
- **NEW** `scripts/f7l_straka_probe.py`, `scripts/f7l_straka_batch.py`,
  `scripts/f7l_official_run.py`.
- **NEW** `proofs/f7l/`: straka_diffusion_fix.md, straka_wdiff_compare.json,
  straka_maxw_trace_900probe.txt, straka_density_current_{diagnostics.json,verdict.md}
  (+plot), skamarock_bubble_{diagnostics.json,verdict.md} (+plots),
  regression_recheck.json.

## Commands run (CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src taskset -c 0-3, cuda:0, fp64)
- `f7l_straka_batch.py --configs nu0,nu75 --end 280` → A/B that pinpointed w.
- `f7l_straka_probe.py --end 900` → detonation now 240–300 s (was 240 s).
- `f7l_official_run.py straka` → FAIL (non-finite by 900 s) — official AC1.
- `f7l_official_run.py bubble` → **PASS 6/6** (AC2 no-regression).
- `pytest tests/test_m4_acoustic.py test_m4_dycore_step.py test_m4_tier2_invariants.py`
  → **10 passed** (AC3 no-regression).

## Acceptance gates
- **AC1 (Straka PASS): FAIL.** Non-finite by 900 s; detonates 240–300 s at
  cold-pool touchdown. WRF-faithful stabilizers applied (ν=75 on u,v,w,θ + WRF
  emdiv/smdiv/w_damping/damp_opt/CFL); no masking clamps. F7L advanced it from
  240 s → ~270–300 s but did not close it.
- **AC2 (warm bubble STILL PASS 6/6): PASS.** Identical to F7K (thermal_rise
  1925 m, max|w| 11.72, θ′max 1.92, drift 0, mass drift 0). Inviscid, unaffected.
- **AC3 (no regression): PASS.** m4 10/10; flat-rest/conservation intact; all
  prior F7 operators untouched; only addition is benchmark ν=75 w-diffusion +
  already-present WRF damping, documented with WRF file:line. No clamps/caps.

## Unresolved risk / next decision
The dry dynamical core is **one localized residual from done**: the warm bubble
(the clean buoyancy+transport test) PASSES 6/6, so the F7 buoyancy/advection/
acoustic path is sound. The remaining defect is **specific to the stiff
descending cold front** (Straka touchdown). Recommended F7M: parallel multi-angle
hunt — (1) gust-front horizontal-PGF / cold-pool u-outflow vs WRF; (2)
descending-front w lower-BC; (3) front-speed deficit (u-spreading) — NOT more
diffusion. Decision for the manager: dispatch F7M as a focused front-residual
bug-hunt (Opus + Codex parallel angles).

F7L_PARTIAL
