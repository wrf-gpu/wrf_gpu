# Phase-B Lane B4 proofs — static fields + lateral boundaries

Pinned case: `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z`,
domain `d02` (WRF grid 159×66×44, nested under d01, `dt=6s`).
All validation is against REAL WRF NetCDF (`wrfinput_d02`, `wrfout_d02_*`,
`wrfbdy_d01`), no JAX-vs-JAX self-compares. Run on the RTX 5090 under
`taskset -c 0-3 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 JAX_ENABLE_X64=true`.

## Artifacts

| File | What it proves | Status |
|------|----------------|--------|
| `static_field_parity.json` | Every prescribed State leaf sourced from WRF (XLAND, LU_INDEX, LAKEMASK, TSK→t_skin, SMOIS[0]→soil_moisture, HGT→GridSpec terrain) is **bitwise-equal** to the WRF ground truth. | PASS |
| `metrics_consistency.json` | Loader fix: dycore terrain slopes now sourced from the same wrfout snapshot as the state PHB; PGF terrain-slope error vs PHB dropped from 0.054 (228 m terrain mismatch) to 2.8e-8. | PASS |
| `boundary_application_validation.json` | WRF spec+relaxation reproduced: spec-zone exact (all 4 sides, diff=0); relaxation bitwise-matches an independent NumPy re-derivation of WRF `relax_bdytend` (diff=0); relaxation drives the boundary strip toward WRF hour-1 truth (theta RMSE 0.68→0.05 K); boundary strip is NOT the dominant first-hour error in the true WRF solution. | PASS |
| `fp64_verification.json` | Boundary apply preserves float64 end-to-end under `force_fp64`; on the operational perf path it never downcasts an fp64-locked prognostic nor reads its forcing leaf in fp32. | PASS |
| `b2_b4_edge_seam.json` | Gate-1 decision #4 seam: MYNN's periodic `jnp.roll` mass→face inverse corrupts ONLY the outer staggered faces (u x∈{0,nx}, v y∈{0,ny}); B4's spec zone hard-sets exactly those faces to the WRF value after physics, fully closing the seam. | PASS |

## Reproduce

```
taskset -c 0-3 env OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 JAX_ENABLE_X64=true \
  python3 scripts/b4_static_field_parity.py
  python3 scripts/b4_metrics_terrain_consistency.py
  python3 scripts/b4_boundary_application_validation.py
  python3 scripts/b4_fp64_verification.py
  python3 scripts/b4_b2_edge_seam.py
```

## WRF source of truth for the boundary scheme

- `share/module_bc.F` — `relax_bdytend` (relaxation-zone residual + Laplacian
  smoothing stencil) and `spec_bdytend` (outer specified zone).
- `dyn_em/module_bc_em.F` — `relax_bdy_dry` driver and `lbc_fcx_gcx` (the
  `fcx`/`gcx` weight taper, `specified`/`nested` branches identical).
- `namelist.input`: `spec_bdy_width=5, spec_zone=1, relax_zone=4, spec_exp=0`.

## Loader fixes made (B4-owned files)

1. `integration/d02_replay.py` `build_replay_case`: source the dycore metrics
   (terrain slopes) from the t=0 `wrfout_<domain>` snapshot rather than
   `wrfinput_<domain>`, so the PGF terrain is consistent with the state's
   base-state geopotential PHB. Only HGT differs between the two files (parent→
   nest terrain blending at init); all map factors / eta coefficients are
   bitwise-identical. See `metrics_consistency.json`.
2. `coupling/boundary_apply.py`: rewrote the relaxation to the exact WRF
   `relax_bdytend` stencil (correct residual Laplacian using the boundary-width
   strips at `b_dist±1`, WRF corner trimming, all four sides evaluated from the
   same input field), keeping the spec-zone hard-set and legacy 4-D leaf
   support. Bitwise-validated vs an independent NumPy re-derivation.

## Documented departure from bit-exact WRF (honest)

WRF relaxes the *mass-coupled* variables (`ru=c1·mu·u`, `t=mu·theta`,
mass-weighted `ph`/`w`). The Gen2 replay forces with *decoupled* wrfout
side-history, so B4 relaxes the decoupled fields against decoupled boundary
leaves — an O(mu') approximation, correct for a side-history replay. Quantified
by the relaxation-toward-truth check (0.05 K residual after relaxation).

## Residual risk / not covered

- A full 1-hour coupled forecast "boundary vs interior RMSE" gate cannot be
  cleanly attributed to B4 today because the coupled dycore (F7) + physics
  (B1/B2/B3) hit the sanitiser guards within ~30 steps on this real case
  (theta/w/u pinned at clamp limits). The B4-scoped evidence (boundary operator
  fidelity vs WRF hour-1, WRF-intrinsic boundary≤interior change) is provided
  instead; the end-to-end gate should be re-run once the dycore/physics lanes
  land.
- B2 hardening (replace the periodic `jnp.roll` mass→face inverse with edge
  extrapolation) is recommended but non-blocking; the seam is closed by B4's
  spec-zone hard-set as long as the frozen `physics→boundary` order holds.
