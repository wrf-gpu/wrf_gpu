# case3 V10 momentum-budget — ROOT CAUSE: missing Coriolis force in the dycore

Branch `worker/opus/v10-momentum`, base `59915f2`. Reuses corpus CPU-WRF ONLY
(no new WRF runs). File ownership: only `mynn_pbl.py` + `boundary_apply.py` were
in-scope for a fix; **neither was modified** because the cause is the DYCORE CORE
(read-only here), so this is `V10_DEFERRED_DYCORE`.

## VERDICT

**case3's below-persistence V10 over water is caused by the COMPLETE ABSENCE of
the Coriolis force from the GPU dycore's momentum tendency.** It is NOT the MYNN
PBL momentum mixing and NOT the residual lateral-boundary plume. The fix lives in
the acoustic/RK dycore (the sacrosanct core) and is DEFERRED to a supervised
dycore sprint. Stability/correctness of the hard-won dycore is not risked for this
-0.099 residual.

## EVIDENCE — three independent, mutually-consistent proofs

### 1. The deficiency is the DEEP INTERIOR, not the boundary (rules out the plume)

Boundary-frame skill decomposition of the existing case3 24 h fields
(`proofs/wind/gpu_wind_localize_case3_fields.npz`, skill = 1 − GPU_RMSE/pers_RMSE,
>0 beats persistence):

| region (V10, water)        | skill   |
|----------------------------|---------|
| water (all)                | −0.132  |
| water EXCLUDING 5-cell frame | **−0.204** (WORSE) |
| water deep-box (r20-46,c30-120) | −0.141 |
| water 5-cell FRAME ONLY    | **+0.390** (the frame scores BEST) |

Excluding the boundary frame makes V10 *worse*; the frame is where the GPU does
best. The merged in-loop normal-momentum boundary protection
(`apply_normal_bdy_work`, strength 20) is WORKING on case3 — the deficiency is the
free interior over water, which has no boundary forcing and (we show below) no
Coriolis. This is the OPPOSITE signature to case2 (`WIND_SKILL_ROOT_CAUSE.md`,
where the normal-momentum boundary plume WAS the cause).

### 2. The error is the WHOLE LOWER COLUMN, wrong-direction (dycore, not PBL)

Standard 24 h case3 forecast, water-mean u/v profile vs CPU-WRF destaggered U/V
(`proofs/wind/v10_momentum_budget_standard.json`, run wall 916.7 s, finite):

```
 k   gpu_v / wrf_v     gpu_u / wrf_u     gpu_wspd / wrf_wspd
 0   -4.85 / -7.11     +1.12 / -0.69      6.10 / 7.90
 1   -6.34 / -7.59     +1.38 / -0.79      7.68 / 8.40
 2   -6.89 / -7.82     +1.40 / -0.88      8.18 / 8.60
 3   -7.51 / -7.92     +1.28 / -1.01      8.58 / 8.63
 4   -8.12 / -7.97     +0.54 / -1.39      8.82 / 8.60
 5   -8.19 / -7.52     -0.42 / -2.02      8.70 / 8.16
```

- **u (zonal) is WRONG SIGN through k0..k4**: GPU positive (+1.1..+1.4) where WRF
  is negative (−0.7..−1.4). The sign error is UNIFORM up the column, not a
  near-surface artifact — it only converges to the WRF sign at ~k5.
- **v (meridional) is uniformly TOO WEAK through k0..k3** (GPU −4.85..−7.51 vs WRF
  −7.11..−7.92), crossing over only above k4.

Per the GPT-5.5 sidecar's own discriminator
(`proofs/wind/v10_momentum_sidecar_verdict.md`): "Dycore-favored: u/v errors have
the same sign through k0..k5." This is exactly that case. Vertical PBL mixing is
direction-preserving (mixes toward the column mean); it CANNOT manufacture a
whole-column wind-vector DIRECTION error. A rotated, weakened mean wind vector is
the signature of a missing rotational (Coriolis) force that should balance the
pressure-gradient force in geostrophic equilibrium.

### 3. MYNN momentum is 1000x too small to matter and is NOT pushing the error

Single-step MYNN k0 momentum increment over water (post-dycore final state),
split per the sidecar falsifier #4
(`mynn_k0_increment_single_step_water`, m/s per 10 s step):

| variant          | du0       | dv0       |
|------------------|-----------|-----------|
| FULL (drag+diff) | −1.22e-3  | +3.58e-3  |
| drag-only (dfm=0)| −2.87e-3  | +1.41e-2  |
| diffusion-only   | +1.63e-3  | −1.03e-2  |

The error vector to correct is (u TOO POSITIVE → need du0<0, v TOO WEAK → need
dv0<0). The FULL MYNN increment is du0=−1.2e-3 (correctly tiny-negative on u) and
dv0=+3.6e-3 (the WRONG way on v, i.e. marginally worsening the weak southerly).
Crucially the magnitude is ~1e-3 m/s/step against a ~2 m/s column error — three
orders of magnitude too small to be the driver. The drag and diffusion terms
nearly cancel on v (+1.4e-2 vs −1.0e-2), so neither is a usable 2 m/s lever; the
sidecar pre-approved NO empirical MYNN tuning for a −0.099 residual.

### 4. FALSIFIER — MYNN-momentum-OFF makes the wind vector WORSE (proves MYNN helps, not harms)

Counterfactual 24 h run zeroing ONLY the PBL u/v increment (separate process, no
stale jit cache; `proofs/wind/v10_momentum_budget_mynn_off.json`, finite):

| metric (water)   | standard | MYNN-mom-OFF | delta |
|------------------|----------|--------------|-------|
| V10 skill        | −0.132   | **−0.185**   | WORSE |
| U10 skill        | −0.003   | **−1.063**   | CATASTROPHIC |
| k0 u (vs WRF −0.69) | +1.12 | **+2.80**    | wrong-sign DOUBLES |
| k0 v (vs WRF −7.11) | −4.85 | −7.25        | speed closer, but… |

Turning MYNN momentum off makes the wind VECTOR much worse: the wrong-sign u nearly
triples and U10 skill collapses to −1.06. The sidecar's MYNN-fixable criterion
("MYNN-off improves vector/component skill, not just V10 speed") FAILS decisively.
MYNN's surface friction was the only sink partially OPPOSING the unbalanced
(ageostrophic, no-Coriolis) dycore wind; removing it lets the dycore error run
unchecked. This confirms MYNN momentum is NOT the cause and is, if anything,
mitigating. The whole-column wrong-sign u persists in BOTH runs → the error is the
dycore, upstream of and independent of the PBL.

## ROOT CAUSE (precise) — Coriolis force absent from the entire dycore

Static-code proof: `grep -rniE "corioli|curvatur|f_cor|fcori|geostroph|7\.292|
earth.?omeg"` over `src/` returns **zero** real hits. `DycoreMetrics`
(`src/gpuwrf/contracts/grid.py`) has NO Coriolis-parameter leaf and NO per-cell
latitude; the real-case builder never reads the `F`/`XLAT` fields that the wrfout
files DO contain.

The operational RK momentum tendency is assembled in
`src/gpuwrf/runtime/operational_mode.py::_augment_large_step_tendencies`
(≈ lines 1208-1333): flux-form advection + 6th-order diffusion + constant-K
diffusion + `large_step_horizontal_pgf` (the PGF, faithfully ported from WRF
`module_em.F` `horizontal_pressure_gradient`), then `rk_addtend_dry`.

Verified against pristine WRF v4 (`~/src/wrf_pristine/WRF`). In
`dyn_em/module_em.F::rk_tendency` the call sequence is:
- `module_em.F:717`  `CALL horizontal_pressure_gradient` (PGF — ported)
- `module_em.F:749/761` `CALL perturbation_coriolis` / `CALL coriolis` (**MISSING**)
- `module_em.F:773`  `CALL curvature` (**MISSING**)

The `coriolis` body (`module_big_step_utilities_em.F:3640`) adds, per cell:
```
ru_tend += (msfux/msfuy)*0.5*(f(i,j)+f(i-1,j)) *0.25*(rv → u-face)        [:3726]
rv_tend -= (msfvy/msfvx)*0.5*(f(i,j)+f(i,j-1)) *0.25*(ru → v-face)        [:3800]
```
i.e. +f·v into `ru_tend` and −f·u into `rv_tend` (plus the `e=2Ω cosφ` cosine-
Coriolis and `cosα/sinα` map-rotation terms). The port reproduced the PGF call but
NOT the Coriolis/curvature calls that follow it. So at every RK stage
`ru_tend`/`rv_tend` (consumed by `advance_uv` in the acoustic loop,
`u += dts*ru_tend`) carry PGF but ZERO rotation. `f`, `e`, `cosa`, `sina` exist in
WRF and in the wrfout `F`/`XLAT` fields; none is read or stored by the port.

Why this produces exactly the observed defect:
- With PGF but no Coriolis, the interior flow cannot reach geostrophic balance;
  the PGF accelerates the wind ageostrophically and the mean vector drifts in
  DIRECTION and is held off its balanced magnitude → whole-column wrong-sign u,
  weak v.
- At the Canary latitude (~28.5 °N) f = 2Ω sin φ ≈ 6.96e-5 s⁻¹, so the inertial
  period 2π/f ≈ 25 h. Over a 24 h forecast the omitted rotation is a leading-order
  (≈ one inertial revolution) effect — precisely the lead at which case3 fails.
- The idealized gates (warm bubble, Straka) are non-rotating 2-D density currents
  (f-plane-irrelevant) and never exercise Coriolis, which is why the dycore passed
  every idealized gate while silently missing this term.
- u* matches WRF (0.255 vs 0.261): surface drag is faithful; the deficit is purely
  the upstream prognostic momentum balance, consistent with a missing body force
  (not a surface or mixing error).

## DEFERRED FIX PROPOSAL (for a supervised dycore sprint — NOT done here)

1. Add a Coriolis-parameter leaf `f` (2-D, mass points; plus `e` and `sin/cos α`
   for full map-projection curvature if pursued) to `DycoreMetrics`, populated from
   the wrfout `F` field (and `XLAT` for analytic f = 2Ω sin(XLAT) cross-check) in
   the real-case builder. Idealized cases set f = 0 (preserving the passing gates
   bit-identically).
2. In `_augment_large_step_tendencies`, immediately AFTER `large_step_horizontal_pgf`,
   add the WRF `coriolis` contribution to the coupled large-step tendencies:
   `ru_tend += mu * f * v_at_u`, `rv_tend += − mu * f * u_at_v` (with the C-grid
   averaging of v→u-face and u→v-face exactly as `module_big_step_utilities_em.F`
   does; include the `curvature` term only if map-factor curvature is needed at d02
   scale — likely second-order vs Coriolis here).
3. Gates: (a) idealized warm-bubble + Straka must stay bit-identical (f=0 path);
   (b) re-run this profile harness — the lower-column u sign must flip negative to
   match WRF and v must strengthen; (c) re-score case3 V10/U10 water skill (target
   > 0, beating persistence) and confirm case2 + T2 do not regress; (d) 24 h coupled
   stability must hold.

## What was tried in-scope and correctly NOT changed

- `mynn_pbl.py`: momentum increment is ~1e-3 m/s/step and partly correctly-signed;
  the sidecar pre-approved no empirical MYNN lever for this residual. Untouched.
- `boundary_apply.py`: the boundary frame already scores BEST on case3; the plume
  is not the cause here. Untouched.
- The dycore/acoustic core: the cause, but SACROSANCT. Diagnosed + deferred.
