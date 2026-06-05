# v0.11.0 Sloped Diffusion Status

Verdict: **PASS**.

Implemented WRF map-factor and terrain-slope deformation terms in `src/gpuwrf/dynamics/explicit_diffusion.py` for the owned diffusion helpers.

What changed:

- `horizontal_deformation_2d` now accepts WRF map factors plus `zx`/`zy`/vertical metric coefficients and computes D11/D22/D12 with the terrain-following cross-coordinate terms.
- `deformation_components_3d` adds D33/D13/D23 for the same real-terrain stencil.
- `wrf_terrain_deformation_momentum_tendency` adds the constant-K / variable-K deformation stress divergence for u/v/w, including WRF slope corrections in horizontal stress divergence and vertical D13/D23/D33 stresses.
- `smag2d_horizontal_km` now uses `mlen_h=sqrt((dx/msftx)*(dy/msfty))` and includes WRF's `diff_opt=2` slope-reduction branch when requested.

Parity proof:

```bash
taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python proofs/v0110/slopediff_parity.py
```

Result: PASS. Proof object: `proofs/v0110/slopediff_parity.json`.

The parity fixture is a periodic sloped-terrain synthetic case with non-unit `msftx/msfty/msfux/msfuy/msfvx/msfvy` and nonzero `zx`/`zy`. JAX fp64 output is compared against an independent NumPy transcription of WRF `dyn_em/module_diffusion_em.F`:

- `cal_deform_and_div`: D11/D22/D12/D33/D13/D23 all passed; max residual 0.
- `smag2d_km`: `diff_opt=1` map-factor K and `diff_opt=2` slope reduction passed; max residual 0.
- constant-K momentum stress divergence: `ru_tendf`, `rv_tendf`, `rw_tendf` passed; max residual `3.78e-15` vs tolerance `2e-10`.

Regression commands:

```bash
taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true pytest -q tests/dynamics/test_diffopt1_smagorinsky.py tests/dynamics/test_deformation_momentum_diffusion.py tests/dynamics/test_diffopt1_smagorinsky_integration.py
```

Result: `19 passed in 5.13s`.

```bash
/tmp/wrf_gpu_run.sh env PYTHONPATH=src JAX_ENABLE_X64=true pytest -q tests/idealized/test_density_current.py tests/idealized/test_warm_bubble.py
```

Result: `2 passed in 248.12s`.

d02 sanity:

```bash
/tmp/wrf_gpu_run.sh env GPUWRF_CANAIRY_ROOT=/mnt/data/canairy_meteo PYTHONPATH=src JAX_ENABLE_X64=true python proofs/v0110/recompile_diag.py --domain d02 --chunks 1 --steps 1 --cadence 180 --out proofs/v0110/slopediff_d02_sanity.json
```

Result: PASS. Proof object: `proofs/v0110/slopediff_d02_sanity.json`; all recorded state and diagnostic hashes are finite.

Scope notes:

- Runtime wiring is not changed in this lane per file ownership; existing callers retain flat defaults unless real metrics are passed.
- No WRF executable savepoint for this exact sloped diffusion operator was present in the repo. The parity proof is therefore WRF-source formula parity, not a Fortran-run binary savepoint compare.
