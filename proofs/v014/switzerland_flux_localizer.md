# V0.14 Switzerland Venting — Empirical Flux Localization + Root Cause

Branch `opus-flux-localizer` (base `worker/gpt/v013-close-manager` @ 6fe8e163).
CPU-only proofs; no source change shipped (one candidate fix proven INERT and
reverted). Endpoint per the venting-residual review's named next step #1.

## Binding metric reproduced

`switzerland_flux_localizer.py` rebuilds the EXACT depth-8 `budget_between`
outflux from the wrfout U/V/MU faces and decomposes it per-face / per-level.
CPU net outflux h36->h37 = +74.51 Pa/cell/h; GPU (phys_tendf) net **excess
+26.55** (≡ the binding -26.5). awd_fix_open is +26.62 (invariant, as expected).

## Localization (face / level / field + magnitude)

Per-face excess (Pa/cell/h, h37): **W -1.6, E -12.1, S +20.4, N +19.8** — a
meridional-dominant pattern, but the real structure is VERTICAL, not by-face:

| band | per-level excess sum (Pa/cell/h) |
|---|---:|
| k00-k09 (boundary layer) | **-246** (GPU imports too much low-level mass) |
| k10-k29 (mid-troposphere) | **+227** |
| k30-k43 (upper) | **+46** |

A coherent vertical DIPOLE. The corresponding normal-wind bias is the same
dipole and is **domain-wide / deep-interior** (depth-20, far from any boundary):

| band | deep-interior U bias (GPU-CPU, m/s) |
|---|---:|
| k00-k09 | **+0.34 .. +0.45** (low-level westerly TOO STRONG) |
| k15-k33 | -0.04 .. -0.16 (too weak aloft) |

Spatial cut (u-bias vs distance from W boundary, central rows): **~0 at the
forced boundary (i0-i3), grows to +2.3 m/s at i8-i32, decays toward domain
center.** => NOT a lateral-boundary-forcing/relaxation error (those peak AT the
boundary). The relaxation zone holds i1-i4; the free interior then develops the
dipole. The lateral-boundary lane is FALSIFIED as the venting driver.

Wind→flux sensitivity check: 1 m/s low-level (k0-9) W inflow = 172 Pa/cell/h of
mass import => the +0.34 m/s low-level bias quantitatively reproduces the -246
low-band. The dipole grows in time (low -246→-438, mid +227→+415 h37→h38).

## Root cause (proven, `switzerland_ustar_drag_root.py`)

The low-level westerly bias = too little SURFACE MOMENTUM DRAG. The JAX
revised-surface-layer (sfclayrev) `ustar` delivered to the MYNN momentum bottom
BC is only **61 % of the WRF h36 UST** (mean 0.380 vs 0.624). Since
`bottom_drag = rhosfc*ust^2/wspd` (WRF module_bl_mynnedmf.F:4011), the GPU
surface momentum drag is **~37 % of WRF's**. Consequence at k0 (real chain,
real ustar): the MYNN momentum source `corr(rublten, u) = +0.65` and mean
+0.0030 m/s/s — it ACCELERATES the low-level wind instead of decelerating it,
because the weak drag loses to the upward turbulent diffusion from k1.

Falsifiable knob: scaling the MYNN-input ustar back to the WRF magnitude
(x1.64) flips k0 to `corr = -0.71`, mean -0.0010 — the CORRECT decelerating
drag sign. This is the lever.

## Candidate local fix TESTED and FALSIFIED (not shipped)

WRF builds a different k0 diagonal for momentum than for scalars: the momentum
row (4010-4017) EXCLUDES `kmdz(kts)` from the diagonal (surface stress carried
by the explicit drag only), while the scalar row (4131-4132) keeps `khdz(kts)`.
The JAX `_diffusion_solve_with_surface`/`_with_mf` added `kdz(kts)` to the
diagonal for momentum too — a real WRF-faithfulness discrepancy. Implemented a
`momentum_bottom=True` flag to drop it. Result: **INERT** — `dfm(kts)=kdz(kts)=0`
by MYNN construction at this state, so the production rublten is byte-identical
on/off (verified fresh-process A/B). Reverted; not the lever.

## Recommendation (next step, unfixed by design — surface-layer lane)

Build a `sfclayrev` ustar/CD oracle at the h36 strong-flow state vs WRF UST
(corr is already 0.92, only the MAGNITUDE is ~61 %). Fix the JAX revised
surface-layer ustar under strong-flow / warm-TKE conditions in
`src/gpuwrf/physics/surface_layer.py` (likely the CD / z0 / stability-function
or the `ust = sqrt(CD)*wspd` closure). Raising ustar to WRF restores
bottom_drag, flips the k0 momentum sink to the correct sign, removes the
low-level westerly bias, and collapses the depth-8 venting. This is a surface-
layer faithfulness fix, NOT a MYNN or dynamics fix.

## Files

* `switzerland_flux_localizer.{py,json}` — per-face/per-level flux oracle + wind
  bias (the localization; reproduces +26.55).
* `switzerland_ustar_drag_root.{py,json}` — ustar deficit + k0 drag-sign flip
  test (the root cause; x1.64 ustar => correct sign).
* No `src/` change (candidate momentum_bottom fix reverted as inert).
