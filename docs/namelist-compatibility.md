# WRF `namelist.input` compatibility

**Bring your existing WRF `namelist.input`.** The GPU port reads a standard
Fortran WRF v4 ARW `namelist.input` (all the usual groups: `&time_control`,
`&domains`, `&physics`, `&dynamics`, `&bdy_control`, ...) and, *before any
expensive JAX import or compile*, validates the physics/dynamics scheme
selections fail-closed. You never get a silent wrong answer from selecting a
scheme the port has not implemented.

## Three outcomes per scheme selection

For every gated namelist parameter (`mp_physics`, `cu_physics`,
`bl_pbl_physics`, `sf_sfclay_physics`, `sf_surface_physics`, `ra_lw_physics`,
`ra_sw_physics`, and the gated dynamics options `diff_opt`, `km_opt`,
`damp_opt`, `diff_6th_opt`, `rk_order`, `w_damping`, `sf_urban_physics`), the
selected value is classified against the **full WRF v4 enumeration** (see
`src/gpuwrf/io/wrf_scheme_catalog.py`, transcribed from
`WRF/run/README.namelist`):

1. **Implemented / accepted -> runs.** The selection is in the port's accepted
   matrix below and is consumed normally.

2. **Recognized WRF v4 scheme, NOT YET IMPLEMENTED -> fail closed, specific
   message.** Example:

   ```
   physics.mp_physics=28 (aerosol-aware Thompson (water/ice-friendly)):
   recognized WRF v4 microphysics scheme, NOT YET IMPLEMENTED in the GPU port.
   Supported mp_physics values: 0, 1, 2, 3, 4, 6, 8, 10, 16. ...
   ```

3. **Not a valid WRF v4 option -> fail closed, "not recognized" message.**
   Example:

   ```
   physics.mp_physics=99 is not a recognized WRF v4 microphysics option.
   Supported mp_physics values: 0, 1, 2, 3, 4, 6, 8, 10, 16. ...
   ```

The validator **never silently accepts** an unimplemented scheme. Multi-domain
columns (`mp_physics = 8, 28`) and the Fortran repeat-count syntax
(`mp_physics = 3*8` meaning three domains of `8`) are both handled.

## Implemented-scheme matrix

`[impl]` = operationally wired in the GPU scan; `[ref]` = accepted as a
savepoint-parity / reference path (selectable, but the operational GPU scan
fail-closes it — see the per-option `Action:` text in the error for the exact
pairing/wiring caveat).

| Parameter            | Selectable values                                                                 |
|----------------------|-----------------------------------------------------------------------------------|
| `mp_physics`         | 0 off; 1 Kessler; 2 Purdue-Lin; 3 WSM3; 4 WSM5; 6 WSM6; **8 Thompson [impl]**; 10 Morrison; 16 WDM6 |
| `cu_physics`         | 0 off; 1 Kain-Fritsch; 2 BMJ; 6 Tiedtke; 3 Grell-Freitas [ref]; 16 New Tiedtke [ref] |
| `bl_pbl_physics`     | 0 off; **1 YSU [impl]**; **5 MYNN [impl]**; **7 ACM2 [impl]**; 8 BouLac; 2 MYJ [ref] |
| `sf_sfclay_physics`  | 0 off; 1 revised-MM5; **5 MYNN-SL [impl]**; 7 Pleim-Xiu; 2 Janjic Eta [ref]        |
| `sf_surface_physics` | 0 off; 2 Noah classic; **4 Noah-MP [impl]**                                        |
| `ra_sw_physics`      | 0 off; **4 RRTMG [impl, operational]**; 1 Dudhia [ref, isolated savepoint]         |
| `ra_lw_physics`      | 0 off; **4 RRTMG [impl, operational]**; 1 classic RRTM [ref, isolated savepoint]   |

Dynamics: `rk_order=3`; `diff_opt` 0/2 with `km_opt` 0/1 (constant-K);
`diff_6th_opt` 0/2; `damp_opt` 0/3; `w_damping` 0/1; `sf_urban_physics=0`.

Mandatory WRF pairing enforced: MYJ PBL (`bl_pbl_physics=2`) <-> Janjic Eta
surface layer (`sf_sfclay_physics=2`) must be selected together.

## Known compatibility limitations (honest)

* **Real-data WRF diffusion defaults fail closed.** WRF's recommended real-data
  turbulence settings are `diff_opt=1` (2nd-order on coordinate surfaces) and
  `km_opt=4` (horizontal Smagorinsky). The port only implements the constant-K
  path (`diff_opt=2 / km_opt=1`), so `diff_opt=1` / `km_opt=2..5` correctly
  fail closed with the "recognized WRF scheme, NOT YET IMPLEMENTED" message.
  To run an existing real-data namelist today, switch to `diff_opt=2, km_opt=1`
  (or `diff_opt=0`).
* The scheme catalog reflects **WRF v4** option codes
  (`WRF/run/README.namelist`, audited 2026-06-04); re-audit on a WRF version
  bump using the per-entry `source_lines` references in
  `wrf_scheme_catalog.py`.
* The validator checks *selected* options, not namelist completeness. It does
  not (yet) validate non-physics consistency (timestep CFL, domain ratios, IO
  intervals); those remain the user's responsibility / a downstream concern.

## Where this lives

* `src/gpuwrf/io/wrf_scheme_catalog.py` — full WRF v4 code->name enumeration.
* `src/gpuwrf/io/namelist_check.py` — `validate_supported_namelist()`, the
  three-outcome fail-closed checker (called by the `gpuwrf run` CLI before any
  heavy import).
* `src/gpuwrf/contracts/physics_registry.py` — the authoritative
  implemented/accepted matrix (`ACCEPTED_*`).
* `tests/test_namelist_check.py` — the compatibility test suite.
