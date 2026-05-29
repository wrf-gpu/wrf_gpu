# Phase B Savepoint Schema (FROZEN — Gate-1)

Status: FROZEN at Gate-1. This is the schema the WRF-oracle factory (running in
parallel) WRITES to and the physics lanes B1/B2/B3 READ from.

It reuses and extends the existing, committed dycore savepoint machinery rather
than inventing a new format:
- Schema objects: `src/gpuwrf/validation/savepoint_schema.py`
  (`VariableMetadata` `:113`, `SavepointMetadata` `:162`, `Savepoint` `:268`).
- HDF5 reader/writer + checksum: `src/gpuwrf/validation/savepoint_io.py`
  (`write_savepoint` `:43`, payload SHA-256 `_payload_digest` `:25`).
- Tolerance ladder: `src/gpuwrf/validation/tolerance_ladder.json`
  (loaded/validated by `load_tolerance_ladder` `savepoint_schema.py:290`).
- Phase-B loader/validator added by Gate-1:
  `src/gpuwrf/validation/phase_b_savepoint.py`.

------------------------------------------------------------------------------
## 1. File format and integrity
------------------------------------------------------------------------------

- Container: HDF5, one savepoint per file. `file_format = "hdf5-savepoint-v1"`
  (`savepoint_schema.py:58`).
- Root attrs: `metadata_json` (canonical sorted JSON of `SavepointMetadata`) and
  `payload_sha256` (`savepoint_io.py:13-15`). The checksum covers the canonical
  metadata + every array's name/dtype/shape/C-order bytes
  (`_payload_digest`, `savepoint_io.py:25`). The Phase-B loader RE-VERIFIES the
  checksum on read and refuses a tampered/partial file.
- Arrays live under the `fields` group, gzip-4 + shuffle for arrays >= 16 elements
  (`savepoint_io.py:38`).

## 2. Per-variable metadata (`VariableMetadata`, savepoint_schema.py:113)

| key        | meaning                                                          |
|------------|------------------------------------------------------------------|
| name       | variable name (WRF or State field name)                          |
| dtype      | numpy dtype string; must match array exactly (`Savepoint.validate` :282) |
| shape      | exact array shape (validated)                                    |
| stagger    | one of `VALID_STAGGERS` = {mass,u,v,w,eta-half,eta-full,scalar} (`:59`) |
| units      | non-empty SI/WRF units string                                    |
| provenance | non-empty source string (WRF routine / wrfout var / derivation)  |
| role       | `input` \| `expected` \| `diagnostic` (`:123`)                   |

C-grid stagger convention matches `State`: `u`=u-face, `v`=v-face, `w`/geopotential
=`eta-full` (w-stagger), mass fields=`mass`, surface 2-D=`mass`.

## 3. Run / operator metadata (`SavepointMetadata`, savepoint_schema.py:162)

Required, all validated in `__post_init__` (`:187`):
`run_id`, `wrf_version`, `wrf_commit`, `namelist_hash`, `source_path`,
`domain_index>=1`, `tier`, `operator` (in `VALID_OPERATORS`), `boundary`
(in `VALID_BOUNDARIES`), `dt_seconds>0`, `rk_stage_index>=0`,
`acoustic_substep_index>=0`, `map_factors`, `vertical_grid`, `variables`.
`sanitizer_mode` is locked to `"off"` (`:205`) — no masking/clamping in oracle data.

**source-run identifier:** the tuple `(run_id, wrf_version, wrf_commit,
namelist_hash, source_path, domain_index)` uniquely identifies the WRF run a
savepoint came from. The Phase-B loader exposes it as `Savepoint.source_run_id`.

### 3.1 Phase-B operator/boundary names

The committed `VALID_OPERATORS`/`VALID_BOUNDARIES` cover the dycore. The WRF-oracle
factory for physics will need physics operator/boundary names (e.g.
`mp_gt_driver`, `sfclay`, `mynn`, `radiation_driver`). These are added in
`phase_b_savepoint.py` as `PHASE_B_OPERATORS` / `PHASE_B_BOUNDARIES` and the
loader accepts the union. The exact WRF savepoint boundary list per scheme is in
`oracle_manifest.json`. (We do not edit the frozen dycore sets in
`savepoint_schema.py`; the Phase-B loader widens acceptance.)

------------------------------------------------------------------------------
## 4. Tolerance ladders — tight transcription vs operational RMSE (FROZEN)
------------------------------------------------------------------------------

Two distinct tolerance regimes, per the project validation philosophy
(`feedback_validation_philosophy`):

- **Tier-1 transcription tolerance (tight):** applied at an *operator-boundary*
  savepoint with WRF supplying the EXACT inputs. Per-field abs/rel/ulp from the
  ladder; catches transcription bugs. Does NOT bind milestone close.
- **Tier-4 operational RMSE tolerance (loose):** applied at the operational
  divergence-map fields (SWDOWN/GLW/HFX/LH/PBLH/TSK/T2/U10/V10/PSFC and 3-D
  U/V/W/theta/qv) over a full forecast vs WRF wrfout. This is the operational
  gate.

The machine-readable ladder is `tolerance_ladder.json`. Each field entry has
`units,dtype,abs,rel,ulp,accumulation_exception` (validated `:300-305`); the
"perturbation must be >= 10x tolerance" rule is enforced (`:297`). Gate-1 adds the
Phase-B physics + operational-diagnostic field entries (see `phase_b_savepoint.py`
`PHASE_B_TOLERANCES` and the appended entries in `tolerance_ladder.json`).

Frozen Phase-B tolerance bands (rationale: physics is process-split and
chaotic-amplifying, so transcription tol is tight at the operator boundary but the
operational RMSE band is field-physical, not bitwise):

| Field class                | transcription abs | transcription rel | operational RMSE band      |
|----------------------------|-------------------|-------------------|----------------------------|
| qv,qc,qr,qi,qs,qg (kg/kg)  | 1e-9              | 1e-6              | 1e-4 (column-integrated)   |
| Ni,Nr,Ns,Ng (m^-3)         | 1e-3 (abs count)  | 1e-4              | 10% relative               |
| theta heating (K)          | 1e-6              | 1e-7              | T2 RMSE band (below)       |
| ustar (m/s)                | 1e-7              | 1e-7              | 0.05 m/s                   |
| theta_flux,qv_flux,tau_*   | 1e-7              | 1e-6              | flux-derived (HFX/LH below)|
| SWDOWN,GLW (W/m^2)         | 1e-4              | 1e-5              | 20 W/m^2                   |
| HFX,LH (W/m^2)             | 1e-4              | 1e-5              | 30 W/m^2                   |
| PBLH (m)                   | 1e-2              | 1e-4              | 150 m                      |
| T2 (K)                     | 1e-5              | 1e-6              | 1.5 K                      |
| U10,V10 (m/s)              | 1e-5              | 1e-6              | 1.5 m/s                    |
| PSFC (Pa)                  | 1e-3              | 1e-7              | 50 Pa                      |
| TSK (K)                    | 1e-5              | 1e-6              | bitwise (data-replayed)    |

These bands are the FROZEN starting point; a lane may TIGHTEN (never loosen) a
transcription tol with evidence, via the manager.

------------------------------------------------------------------------------
## 5. Inactive-vs-missing handling (matches the Gate-1 harness fix)
------------------------------------------------------------------------------

A WRF oracle savepoint may legitimately show **zero** for a scheme's output on a
column where the scheme is physically inactive (dry column for microphysics,
night column for SW). A comparison/validator MUST NOT treat a zero-delta output as
a failure when the *input forcing* was below the scheme's activation floor — this
is the same "physically-inactive != missing operator" rule fixed in the diagnostic
harness (`src/gpuwrf/diagnostics/comprehensive_harness.py`, `_physical_opportunity`
+ the `INACTIVE_PHYSICAL` verdict). The Phase-B loader exposes
`activation_floor_for(field)` so lane validators apply the same rule.
