# B2 — Surface layer + MYNN PBL proof objects

Lane B2 (surface layer + MYNN PBL) for the JAX WRF GPU port. The surface layer is
a faithful rebuild of the WRF **revised** surface scheme
(`module_sf_sfclayrev.F` -> `physics_mmm/sf_sfclayrev.F90` `sf_sfclayrev_run`),
replacing the FAILED M12 MM5-`sfclay` attempt. The MYNN PBL kernel
(`physics/mynn_pbl.py`) now consumes the surface layer's fluxes directly through
the FROZEN surface->MYNN coupling instead of an internal neutral-bulk stub.

## Proof objects

| file | what it proves | how to run | result |
|------|----------------|------------|--------|
| `surface_mynn_parity_wrf.{py,json}` | **REAL WRF parity** vs the surface-layer oracle at `/mnt/data/wrf_gpu2/physics_oracle/surface_mynn/` (real Canary L3 run, `itimestep=1`). NOT a self-compare. | `PYTHONPATH=src python proofs/b2/surface_mynn_parity_wrf.py` | **PASS** (operational gate) |
| `surface_layer_oracle.{py,json}` | Independent algebraic invariants of `sf_sfclayrev_run` (CB05 table-vs-full, `zolri` residual, MO sign/regime consistency). No WRF dependency. | `PYTHONPATH=src python proofs/b2/surface_layer_oracle.py` | PASS |
| `coupled_smoke.{py,json}` | surface_adapter -> mynn_adapter through `State`: finite, **fp64 under force_fp64 (precision-defeat guard)**, sane diagnostic bands, non-periodic edge reconstruction. | `PYTHONPATH=src python proofs/b2/coupled_smoke.py` | PASS |
| `surface_mynn_parity.{py,json}` | Frozen HDF5-savepoint-v1 schema parity harness (loader = `phase_b_savepoint.load_phase_b_savepoint`). | `PYTHONPATH=src python proofs/b2/surface_mynn_parity.py` | PENDING-ORACLE (factory wrote raw `.f64`, not `.h5`) |

Run with `taskset -c 0-3 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.15`.

## Real-WRF parity result (surface_mynn_parity_wrf.json)

Operational M9 diagnostics (B2-owned, coupler_interface.md §4), RMSE over 5487
active columns vs real WRF, all within operational bands:

| field | RMSE | band | field | RMSE | band |
|-------|------|------|-------|------|------|
| T2    | 0.041 K  | 1.5 K  | HFX | 17.4 W/m² | 30 |
| U10   | 0.0073   | 1.5    | LH  | 10.7 W/m² | 30 |
| V10   | 0.022    | 1.5    | ustar | 0.026 | 0.05 |
| MOL   | 0.045 K  | 2.0    | PSIM/PSIH | 0.22/0.24 | 0.5 |

## KEY FINDING (manager decision needed)

The contract specified porting `module_sf_sfclayrev.F`. The WRF-oracle factory run
actually used **`sf_sfclay_physics=5` = the MYNN surface layer (`sf_mynn.F90`)**,
not sfclayrev (`manifest.json` -> `physics_options`). The two schemes share the
Monin-Obukhov similarity core, so the ported sfclayrev matches the real oracle on
**every operational diagnostic** within bands. They differ only in scheme-specific
*internal* diagnostics, which are reported but NOT gated:

- `br`  — sf_mynn clamps to [-2,2]/[-4,4]; sfclayrev leaves it unbounded.
- `zol` — sf_mynn defines `zol = za·k·g·mol/(θ·max(ust²,1e-4))` clamped [-20,20]
  via a *different* relation + the `zolrib`/`li_etal_2010` solver; sfclayrev uses
  the `zolri` Richardson solve. (We clamp to [-20,20] to match the reported band.)
- `qsfc` — small definitional offset (psfc vs lowest-level p handling carry).

If exact internal-diagnostic parity with `sf_mynn` is required, B2 should port
`physics_mmm/sf_mynn.F90` instead. For the operational forecast skill target
(T2/U10/V10/HFX/LH), the current sfclayrev port already meets the gate.

## Non-periodic C-grid edge handling (Gate-1 decision #4)

`physics_couplers._mass_to_u_face`/`_mass_to_v_face` were rewritten from the prior
PERIODIC `jnp.roll` reconstruction to non-periodic: interior faces are centred
averages; the two domain-edge faces use zero-gradient extrapolation (no
cross-domain wrap). **B4 seam:** the edge faces inside the relaxation/specified
zone are owned by `apply_lateral_boundaries` (runs AFTER mynn_adapter), which
overwrites them with wrfbdy values. MYNN provides only a finite interior-consistent
guess that B4 corrects at the wall.
