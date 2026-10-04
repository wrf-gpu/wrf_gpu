# d03 Ni non-finite — ROOT CAUSE FOUND (manager, 2026-06-26, from code; no new GPU run needed)

## Decisive result: Ni-ROOT (Thompson ice-number), NOT a dynamics mass-pump
The finite checker `runtime/finite_state_guard.py:first_nonfinite_state_location` iterates
`PROGNOSTIC_STATE_FIELDS` IN ORDER and returns the FIRST non-finite field. The order is:
`u,v,w,theta,qv,p_total,p,p_perturbation,ph_total,ph,ph_perturbation,mu_total,mu,mu_perturbation`
(indices 0–13, ALL dynamics) → `qc,qr,qi,qs,qg` (14–18, ice/water **mass**) → **`Ni` at index 19**.
The N1 failure reported **`field=Ni`**. Therefore at step 720 **every dynamics field AND every moisture-MASS
field (incl. ice mass qi) was FINITE** while ice-NUMBER Ni was the first non-finite. Since non-finiteness
propagates (NaN/Inf persists), the dynamics never went non-finite up to the failure — **the origin is the
Thompson ice-NUMBER (Ni) path, not the acoustic/vertical dynamics, not even the ice mass.**
→ The v0.21 "mass-pump → ph detonation → Ni downstream" hypothesis is REFUTED for this case. It's microphysics.

## The dropped WRF guard (the fix)
Port's MAIN Ni microphysics update `physics/thompson_column.py:1214-1221`:
```
Ni = jnp.maximum(0.0, state.Ni + ... + where(ice_deposition<0, ice_deposition/max(xmi,XM0I), 0) - ...)
```
- It bounds Ni BELOW (`maximum(0.0, …)`) but has **NO UPPER ceiling**. WRF caps ice number at **999e3** AFTER
  summing all microphysics tendencies (`module_mp_thompson.F:3054-3055`:
  `if (xni.gt.999.E3) niten = (999.E3-ni1d*rho)*odts*orho`). The port applies 999e3 only at init/saturation
  (line 752) and the missing/small-diameter branches (445/450) — **NOT on this per-step tendency update.**
- The WRF sublimation number clamp `pni_ide = MAX(-ni*odts, pni_ide)` (bounds the `deposition/xmi` number
  removal to ≤ existing number) — verify whether the port's `where(ice_deposition<0, …/max(xmi,XM0I), 0)` term
  is correspondingly clamped (it relies only on the final `maximum(0.0, …)`).
Mechanism: with no upper ceiling, Ni can grow unbounded over the steep-terrain d03 column across ~720 steps →
eventual float64 overflow → Inf → NaN. WRF's 999e3 ceiling prevents exactly this runaway.

## Fix (WRF-faithful, NOT a masking clamp)
Restore WRF's behavior on the port's main Ni update (1214-1221):
1. Apply the **999e3/rho upper ceiling** to Ni AFTER the tendency sum (mirroring WRF 3054-3055), alongside the
   existing `maximum(0.0, …)` floor (and the R2=1e-6 floor where WRF uses it).
2. Confirm/restore the **sublimation-term clamp** so the `ice_deposition/xmi` number removal cannot exceed the
   existing Ni (WRF `pni_ide = MAX(-ni*odts, pni_ide)`).
Property: WRF only caps when Ni would exceed 999e3 → on already-finite cases (9-nest canary) the cap NEVER
fires → **bit-identical default** (the no-regression gate proves this). On Mont-Blanc/d03 it prevents the
runaway → finite.

## Why this also unlocks K2-nested
A bounded Ni removes the nest's blow-up at default dt AND at the K2 higher dt (the cap is dt-agnostic) → the
nest can run the K2 dt-ladder (N2–N4) → the ~1.8× nested speedup becomes testable. This is the v0.21.1 AND-gate.

## Note on process
GPT 0:3 spent ~45 min stuck reproducing the v0.21 AOT-cache *identity* for the onset trace (unnecessary — the
bug reproduces under plain JIT). The root was answerable from code (field order + the existing N1 result), so
the manager resolved it directly. Fix applied via manager implementer; 0:3's onset-trace is now moot.
