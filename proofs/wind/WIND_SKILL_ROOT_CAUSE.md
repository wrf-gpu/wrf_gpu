# Near-surface wind skill gap — root cause (case2 0509 18z L2, d02)

**Verdict: the U10/V10 below-persistence gap is a REAL model deficiency (not a
regime limit). The dominant, fully-localized cause is a lateral-boundary defect
in the NORMAL wind component (U at the W/E boundaries, V at the S/N boundaries):
the normal velocity explodes to ±7–17 m/s inside the 5-cell relaxation zone
within ~90 dynamics steps (0.25 h), then advects into the 93%-ocean interior.**

All numbers below are reproducible from the committed proofs in this directory.

---

## 1. Is the regime or the model the limit? → THE MODEL.

`proofs/wind/cpu_regime_diagnostic.py` (CPU-only, no GPU, uses only existing
corpus CPU-WRF history):

- **CPU-WRF is highly skillful on V10.** Two independent CPU-WRF forecasts of the
  same case (L2 9→3 km and L3 higher-res) agree at d02 to **0.13 m/s RMSE on V10
  at 24 h**, while persistence RMSE is **2.05 m/s**. The V10 field genuinely
  evolves (its 24 h change-from-init is 1.2–1.5× its own spatial std) and CPU-WRF
  tracks that evolution tightly. So a faithful model SHOULD beat persistence here.
- **Our GPU does not** — V10 RMSE 3.28 (skill −0.59) at 24 h, worse than holding
  the init constant. This is a model deficiency, not a metric/regime artifact.

## 2. The 10 m diagnostic is NOT the cause (refutes the earlier agy theory).

The earlier fix (`surface_layer.py:558-562`) gated a neutral-log 10 m wind for
`7 < za < 13 m`. **The actual lowest mass-level height in this config is
za ≈ 25.5 m** (lowest layer ≈ 51 m thick; verified from PH/PHB/HGT), so that
branch never fires. For za > 13 m **both** the CPU-WRF comparator
(`module_sf_mynn.F:1127-1131`) **and** our `sfclayrev` use the SAME
stability-corrected ratio `U10 = U1D·PSIX10/PSIX`. The V10 spike is present in the
*prognostic* lowest-level wind `v0` itself (row-1 v0 ≈ −12 m/s), not introduced by
the diagnostic (ratio10 ≈ 0.92, correct). The CPU-WRF comparator namelist is
`sf_sfclay_physics=5, bl_pbl_physics=5` (MYNN sfc+PBL); we port `sfclayrev`+MYNN —
a real scheme mismatch, but it is **not** the wind-skill driver here.

## 3. Spatial localization → the NORMAL component at its perpendicular boundary.

`proofs/wind/gpu_wind_localize.py` dumps full GPU fields + surface internals and
decomposes the error by land/water/coast and by boundary frame.

24 h, row/column means (GPU vs CPU-WRF truth), m/s:

| Boundary | component | role at bdy | GPU edge rows | WRF (flat) |
|---|---|---|---|---|
| S (rows 0–4)   | **V10** | **normal**     | −7.0, −10.9, −7.5, −4.8, −3.9 | ≈ −1.3 |
| N (rows −5..−1)| **V10** | **normal**     | −3.1, −2.4, +1.4, **+7.4**, +4.8 | ≈ −1.4 |
| W (cols 0–4)   | **U10** | **normal**     | −3.2, **−7.0**, −2.9, +1.0, +2.5 | ≈ +2.8 |
| E (cols −5..−1)| **U10** | **normal**     | +1.4, +2.3, +5.0, **+9.6**, +6.8 | ≈ +0.9 |
| S/N            | U10     | tangential     | matches WRF (no spike) | — |
| W/E            | V10     | tangential     | matches WRF (no spike) | — |

The **normal** wind component blows up in the relaxation zone at the boundary it
is perpendicular to; the **tangential** component is clean. The spec edge
(b_dist=0) IS correctly pinned (V row 0 = the interpolated `v_bdy` value, e.g.
−1.33 at 24 h, matching WRF). The corruption is the relaxation rows 1–4.

Error decomposition (skill = 1 − GPU_RMSE/persistence_RMSE):

| lead | field | full skill | excl 5-cell normal-bdy frame |
|---|---|---|---|
| 24h | V10 | −0.59 | −0.29 |
| 24h | U10 | −0.20 | **+0.006** |
| 48h | V10 | −1.14 | −0.65 |
| 48h | U10 | −0.33 | −0.14 |

Removing the boundary frame recovers ~half the V10 RMSE excess and makes U10 tie
persistence at 24 h. A deep-interior box (rows 20–46, cols 30–90) still shows
V10 skill −0.31 / U10 −0.22 → a residual interior error remains (boundary plume +
sfclayrev-vs-MYNN momentum-profile difference), but the boundary spike is the
single largest, sharpest, most fixable feature.

## 4. Temporal + isolation → it is the dycore/acoustic coupling at the boundary.

- At **lead 0.25 h (90 steps)** the staggered `v` row 1 is already −17 m/s (spec
  row 0 still correctly pinned at −3.8). The blow-up is fast, not a slow drift.
  (`proofs/wind/short_lead_vbdy.log`.)
- The boundary relaxation **in isolation is stable**: applying
  `apply_lateral_boundaries` 90× to a consistent frozen state leaves `v` exactly
  unchanged (`proofs/wind/relax_stability.log`). So the relaxation stencil is not
  unconditionally unstable.

Therefore the energy is injected by the **acoustic/RK dycore advancing the
normal-momentum boundary zone**, which a single end-of-step decoupled relaxation
(weight 0.1·residual/step) cannot remove. WRF avoids this: `dyn_em/solve_em.F`
applies the boundary to the *mass-coupled* tendencies `ru_tend`/`rv_tend` INSIDE
the RK loop and uses `spec_bdyupdate`/`advance_uv` with `spec_zone` so the normal
momentum in the spec zone is set from the boundary tendency, NOT integrated by the
acoustic solver. Our port (`coupling/boundary_apply.py`) relaxes the *decoupled*
U/V once per full step and never protects the spec-zone normal momentum inside the
acoustic substep loop (`dynamics/core/acoustic_wrf.py` / `advance_uv`).

## 4b. A value-level boundary band-aid is structurally INCAPABLE of fixing it.

`proofs/wind/normal_bdy_fix_probe.py` monkey-patches a STRONG end-of-step pin
(weight 0.7) of the normal component's entire spec+relax zone toward the
interpolated boundary value, then runs 0.5 h. Result
(`proofs/wind/normal_bdy_fix_probe_05h.log`): the spike is **unchanged** —
`v` row 1 = −17 m/s, `u` col 1 = −10 m/s, `u|max| = 61 m/s. The run stays finite
but the boundary zone is still wrong. This is expected: the M9 diagnostic is taken
right after a dynamics step, and the acoustic/RK loop regenerates the normal-wind
boundary spike WITHIN the step regardless of how hard the previous step's end was
pinned. **Therefore the correction must live INSIDE the acoustic substep loop**
(WRF's `spec_bdyupdate`/`advance_uv` with `spec_zone`), not in the
`boundary_apply.py` coupling layer. This falsifies the cheapest candidate fix and
narrows the follow-up to the dycore.

Note: idealized warm-bubble/Straka runs use `run_boundary=False`
(`ic_generators/idealized.py:589`) and never call `apply_lateral_boundaries`, so
the boundary path cannot affect the idealized dycore gates — the fix risk is
confined to the real-case acoustic-loop change.

## 5. Fix direction (for the follow-up; outside this agent's file ownership)

The fix is in the dycore/boundary coupling, NOT in the surface layer (a surface
z0/Charnock fix would make the already-too-fast wind faster — confirmed wrong
lever). Concretely:

1. **Protect the spec zone inside the acoustic substep loop** — apply WRF's
   `spec_bdyupdate` to the normal momentum (U at W/E, V at S/N) at each acoustic
   substep so the outermost `spec_zone` rows/cols are driven by the boundary
   tendency rather than advanced by the acoustic solver
   (`dynamics/core/acoustic_wrf.py` + `advance_uv`). Section 4b proved a
   coupling-layer (post-step) pin does NOT work — the loop regenerates the spike
   within the step.
2. Apply the relaxation tendency to the mass-coupled `ru/rv` tendency inside the
   RK loop (matching `relax_bdytend` placement), instead of as a single decoupled
   end-of-step value nudge.
3. Re-run `proofs/wind/gpu_wind_localize.py` and `proofs/m19/persistence_baseline.py`;
   target V10/U10 skill > 0 (beat persistence) AND re-run the idealized dycore
   gates (warm bubble + Straka) and the 24 h coupled stability — stability must
   hold.

### Owned-file experiments already ruled out
- **Surface scheme (sfclayrev vs MYNN, `surface_layer.py`)**: confirmed NOT the
  lever. The over-water z0 is a static 2.85e-3 (CM=0 at init → surrogate default;
  WRF Charnock/COARE ocean z0 is ~1e-4–3e-4, so our drag is ~2× too large), but
  the GPU 10 m wind over water is already TOO FAST (+1.9 m/s vs CPU-WRF). A
  Charnock z0 fix would REDUCE drag and make the wind faster → wrong direction.
  The too-fast ocean wind is driven by the boundary spike advecting inward and a
  momentum-profile difference, not by surface drag.
- The 10 m diagnostic gate (`surface_layer.py:558-562`) is inert at this grid
  (za ≈ 25.5 m, not 7–13 m); both schemes use `PSIX10/PSIX` here.

## Reproduce

```
# CPU-only regime test (no GPU):
JAX_PLATFORMS=cpu PYTHONPATH=src OMP_NUM_THREADS=2 taskset -c 0-3 \
  python proofs/wind/cpu_regime_diagnostic.py

# GPU localization (real forecast, ~25 min for 24h+48h):
PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 OMP_NUM_THREADS=2 taskset -c 0-3 \
  python proofs/wind/gpu_wind_localize.py --leads 24 48
```
