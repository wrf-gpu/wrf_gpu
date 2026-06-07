# WRF `namelist.input` compatibility

**Bring your existing WRF `namelist.input`.** The GPU port reads a standard
Fortran WRF v4 ARW `namelist.input` (all the usual groups: `&time_control`,
`&domains`, `&physics`, `&dynamics`, `&bdy_control`, ...) and, *before any
expensive JAX import or compile*, validates the physics/dynamics scheme
selections fail-closed. You never get a silent wrong answer from selecting a
scheme the port has not implemented. This is the **no-silent-gaps** contract:
every WRF v4 option is either implemented, or loudly fail-closed with a named
reason, or a documented out-of-scope decision.

## The support catalog: four honest statuses

A machine-readable catalog (`src/gpuwrf/io/scheme_catalog.py`) classifies
*every* WRF v4 code of the gated namelist parameters — `mp_physics`,
`cu_physics`, `bl_pbl_physics`, `sf_sfclay_physics`, `sf_surface_physics`,
`ra_lw_physics`, `ra_sw_physics`, the dynamics options `diff_opt`, `km_opt`,
`damp_opt`, `diff_6th_opt`, `rk_order`, `w_damping`, and `sf_urban_physics` —
against the **full WRF v4 enumeration** (`src/gpuwrf/io/wrf_scheme_catalog.py`,
transcribed from `WRF/run/README.namelist`). Each selection resolves to exactly
one status:

1. **`implemented` -> runs.** Operationally GPU-scan-wired and consumed
   normally.

2. **`reference_only` -> accepted at the namelist layer, fail-closed in the
   operational scan.** A recognized WRF scheme with a parity-proven (savepoint /
   isolated / analytic-oracle) adapter that is *not yet* threaded into the
   operational GPU scan. The namelist validator accepts it (so you can run a
   single-column / reference comparison), and the operational forecast scan
   refuses it loudly with a named reason — never a silent wrong result. Today:
   MYJ PBL (`bl_pbl_physics=2`) + Janjic Eta surface (`sf_sfclay_physics=2`),
   New Tiedtke (`cu_physics=16`), classic RRTM LW (`ra_lw_physics=1`), Dudhia SW
   (`ra_sw_physics=1`).

3. **`recognized_fail_closed` -> fail closed, specific message.** A valid WRF v4
   option the port does not implement. Example:

   ```
   physics.mp_physics=28 (aerosol-aware Thompson (water/ice-friendly)):
   recognized WRF v4 microphysics scheme, NOT YET IMPLEMENTED in the GPU port.
   Supported mp_physics values: 0, 1, 2, 3, 4, 6, 8, 10, 16. ...
   ```

   (A value that is not a WRF option at all — e.g. `mp_physics=99` — fails closed
   with a `not a recognized WRF v4 ...` message.)

4. **`out_of_scope` -> fail closed, named scope decision.** A WRF capability the
   port deliberately does not port (see the out-of-scope list below). Example:

   ```
   chem.chem_opt=401 (WRF-Chem coupled chemistry/aerosols): OUT OF SCOPE:
   ... -- a documented out-of-scope decision for this GPU port, NOT silently
   ignored. Action: Run a meteorology-only configuration (chem_opt=0); ...
   ```

The validator **never silently accepts** an unimplemented scheme or an
out-of-scope feature. Multi-domain columns (`mp_physics = 8, 28`) and the
Fortran repeat-count syntax (`mp_physics = 3*8` = three domains of `8`) are both
handled.

## Implemented-scheme matrix

`[impl]` = operationally wired in the GPU scan; `[ref]` = `reference_only`:
accepted at the namelist layer as a savepoint-parity / isolated / analytic-
oracle reference path, but the operational GPU scan fail-closes it — see the
per-option `Action:` text in the error for the exact pairing/wiring caveat.
(Codes not listed for a parameter are `recognized_fail_closed`: valid WRF
options the port does not implement.)

| Parameter            | Selectable values                                                                 |
|----------------------|-----------------------------------------------------------------------------------|
| `mp_physics`         | 0 off; 1 Kessler; 2 Purdue-Lin; 3 WSM3; 4 WSM5; 6 WSM6; **8 Thompson [impl]**; 10 Morrison; 16 WDM6 |
| `cu_physics`         | 0 off; 1 Kain-Fritsch; 2 BMJ; 6 Tiedtke; 3 Grell-Freitas; 16 New Tiedtke [ref]    |
| `bl_pbl_physics`     | 0 off; **1 YSU [impl]**; **5 MYNN [impl]**; **7 ACM2 [impl]**; 8 BouLac; 2 MYJ [ref] |
| `sf_sfclay_physics`  | 0 off; 1 revised-MM5; **5 MYNN-SL [impl]**; 7 Pleim-Xiu; 2 Janjic Eta [ref]        |
| `sf_surface_physics` | 0 off; 2 Noah classic; **4 Noah-MP [impl]**                                        |
| `ra_sw_physics`      | 0 off; **4 RRTMG [impl, operational]**; 1 Dudhia [ref, isolated savepoint]         |
| `ra_lw_physics`      | 0 off; **4 RRTMG [impl, operational]**; 1 classic RRTM [ref, isolated savepoint]   |

Dynamics: `rk_order=3`; `diff_opt` 0/1/2; `km_opt` 0/1/4; `diff_6th_opt` 0/2;
`damp_opt` 0/3; `w_damping` 0/1; `sf_urban_physics=0`.

Mandatory WRF pairing enforced: MYJ PBL (`bl_pbl_physics=2`) <-> Janjic Eta
surface layer (`sf_sfclay_physics=2`) must be selected together.

## Out-of-scope features (documented decisions, fail closed)

These WRF capabilities are deliberately **not ported**. Selecting a truthy
(non-zero / `.true.`) value for the corresponding switch fails closed with a
scope decision and the disable/alternative recipe — it is never silently
ignored:

| Feature                                    | Switch(es)                          | Disable with |
|--------------------------------------------|-------------------------------------|--------------|
| WRF-Chem coupled chemistry/aerosols        | `chem_opt`                          | `chem_opt=0` |
| WRF-Fire (SFIRE) wildfire spread           | `ifire`                             | `ifire=0`    |
| WRF-Hydro hydrological coupling            | `wrf_hydro`                         | disable coupler |
| FDDA analysis/obs nudging (4DVAR-adjacent) | `grid_fdda`, `obs_nudge_opt`        | `=0`         |
| Stochastic physics                         | `sppt`, `skebs`, `spp`, `rand_perturb`, `stoch_force_opt` | `=0` |
| Moving / vortex-following nests            | `vortex_interval`, `num_moves`      | static nest only |
| Multi-layer urban canopy (UCM/BEP/BEM)     | `sf_urban_physics` 1/2/3            | `sf_urban_physics=0` |
| Coupled ocean mixed-layer / 3-D ocean      | `sf_ocean_physics`                  | `sf_ocean_physics=0` |
| Time-varying SST lower-boundary update     | `sst_update`                        | `sst_update=0` |

## Bring-your-WRF-namelist transition recipe

To run an existing real-data WRF `namelist.input` on the port today:

* Physics suite: pick from the implemented matrix above (the operational default
  is `mp_physics=8` Thompson, `cu_physics=1` KF, `bl_pbl_physics=5` MYNN,
  `sf_sfclay_physics=5` MYNN-SL, `sf_surface_physics=4` Noah-MP + `use_noahmp`,
  `ra_lw_physics=4` / `ra_sw_physics=4` RRTMG).
* Turbulence/diffusion: WRF's recommended real-data defaults `diff_opt=1`,
  `km_opt=4` (2-D Smagorinsky) **run as-is** (see the honesty note below). If you
  use a 3-D closure (`km_opt=2/3/5`), switch to **constant-K: `diff_opt=2`,
  `km_opt=1`** (or `diff_opt=0`).
* Turn off any out-of-scope switch in the table above (`chem_opt=0`,
  `grid_fdda=0`, `sppt=0`, `sf_urban_physics=0`, `sf_ocean_physics=0`, ...).

## Known compatibility limitations (honest)

* **2-D Smagorinsky vs 3-D closures.** WRF's recommended real-data turbulence
  settings `diff_opt=1` (2nd-order on coordinate surfaces) + `km_opt=4`
  (horizontal Smagorinsky) **are implemented and operationally wired** (WRF
  `smag2d_km`, parity in `proofs/v090/diffopt1_smagorinsky_parity.json`), as is
  the constant-K path `diff_opt=2`/`km_opt=1`. The Smagorinsky parity scope is
  the documented idealized-slab reduction (unit map factors, flat-eta
  `zx=zy=0`); the slope-correction branch is gated on `diff_opt==2`. The 3-D
  closures `km_opt=2` (3-D TKE), `km_opt=3` (3-D Smagorinsky) and `km_opt=5`
  (SMS-3DTKE) are **not** implemented and fail closed — transition recipe:
  **switch to constant-K `diff_opt=2 / km_opt=1`** (vertical mixing then comes
  from the PBL scheme).
* The scheme catalog reflects **WRF v4** option codes
  (`WRF/run/README.namelist`, audited 2026-06-04); re-audit on a WRF version
  bump using the per-entry `source_lines` references in
  `wrf_scheme_catalog.py`.
* The validator checks *selected* options, not namelist completeness. It does
  not (yet) validate non-physics consistency (timestep CFL, domain ratios, IO
  intervals); those remain the user's responsibility / a downstream concern.

## Where this lives

* `src/gpuwrf/io/wrf_scheme_catalog.py` — full WRF v4 code->name enumeration.
* `src/gpuwrf/io/scheme_catalog.py` — the machine-readable four-status support
  catalog (`implemented` / `reference_only` / `recognized_fail_closed` /
  `out_of_scope`), the public honesty contract. `classify_scheme()`,
  `classify_feature_switch()`, `iter_full_catalog()`, `status_counts()`, and
  `assert_catalog_consistent()` (the anti-over-claim invariants vs the frozen
  accept-matrix and the operational scan-wired set).
* `src/gpuwrf/io/namelist_check.py` — `validate_namelist()`, the public
  fail-closed entrypoint (scheme support + out-of-scope features), called by the
  `gpuwrf run` CLI before any heavy import. `validate_supported_namelist()`
  remains the scheme-only checker it composes.
* `src/gpuwrf/contracts/physics_registry.py` — the authoritative
  implemented/accepted matrix (`ACCEPTED_*`).
* `tests/test_namelist_check.py`, `tests/test_scheme_catalog_fail_closed.py` —
  the compatibility test suites.
