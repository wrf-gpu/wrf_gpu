# Sprint Contract — v0.3.0 S1 (forcing decode / normalize)

Owner: **Opus MAX or GPT** (interchange). Base: the reconciled v0.3.0 base (see S0
risk). CPU-only ingest path (decode is offline I/O); interp itself is S3.

## Objective
Decode the raw AIFS GRIB2 forcing into the per-field SOURCE arrays (on the native
AIFS 0.25° global grid) that S3's interp consumes, mapped to the met_em variable set
the Canary namelist needs, with units and provenance recorded. Produce the
"forcing decode" half of the metgrid-equivalent pipeline: everything UP TO horizontal
interpolation.

## Non-Goals
- NO horizontal interpolation to the WRF grid (that is S3).
- NO static-geog fields (LANDUSEF/SOILC*/terrain/mapfac — that is S2).
- NO use of the surface-only `aifs_single*.nc` files (wrong product; see Inputs).

## File Ownership (DISJOINT)
- `src/gpuwrf/init/aifs_grib.py` — eccodes/cfgrib GRIB2 reader keyed by
  `Vtable.AIFS_PURE` triplets (discipline/category/parameter + level type/level),
  NOT by shortName (the soil-moisture shortName decodes as `unknown`).
- `src/gpuwrf/init/forcing_decode.py` — assembles the decoded fields into a
  `DecodedForcing` container: 13 isobaric levels of TT/UU/VV/SPECHUMD/GHT on the
  AIFS grid + the surface/2m/10m fields + 2 soil layers; performs the
  surface-level (k=0) assembly and the dewpoint→specific-humidity conversion.
- Tests: `tests/init/test_aifs_grib.py`, `tests/init/test_forcing_decode.py`.

## Inputs
- Raw forcing: `/mnt/data/canairy_meteo/runs/wps_cases/<case>/ungrib/step_NNN.grib2`
  (the TRUE 3D source; 0.25° global 1440×721; 13 isobaric levels 50–1000 hPa of
  t/u/v/q/gh + surface + 2 soil layers). 6-hourly steps 000…072.
- Variable map: `/home/enric/src/canairy_meteo/Gen2/configs/Vtable.AIFS_PURE`
  (the authoritative GRIB→met_em map; RECON.md §3 reproduces it).
- Schema: `gpuwrf.init.metgrid_schema` (field names/units/levels).
- Surface-level + dewpoint→q recipe: confirm against WPS
  `metgrid/src/process_domain_module.F` and the WRF `module_initialize_real.F`
  moisture handling.

## Acceptance Criteria
- Decodes all 16 Vtable.AIFS_PURE entries from a real `step_NNN.grib2` into named,
  unit-correct arrays (RECON.md §3 table) on the AIFS 0.25° grid.
- 3D fields land on the 13 isobaric levels in met_em order (1000→50 hPa).
- Surface (k=0) level assembled per the metgrid recipe: TT[0]←2t, UU[0]←10u,
  VV[0]←10v, SPECHUMD[0]←q(2d,PSFC), GHT[0]←orog, PRES[0]←PSFC. Recipe documented
  with file:line refs to the WPS/real source.
- Dewpoint→specific-humidity conversion matches WPS's formula within 1e-6 kg/kg on a
  hand-checked column (oracle: compare SPECHUMD surface level vs the metgrid output's
  k=0 SPECHUMD for one case/point AFTER S3 interp — recorded as an S1 sub-check, the
  full gate is S4).
- Soil ST/SM decoded by the explicit Vtable depth triplets (0–1000→*000010,
  1000–4000→*010040), NOT by shortName.
- Units normalized to the schema units (GHT in m: geopotential-height already m;
  confirm gh is geopotential height not geopotential m²/s²).
- Provenance dict per field: source path, GRIB sha, level type, Vtable line.

## Predeclared per-variable check (vs the un-interpolated source, sanity-only)
S1 cannot be graded against met_em directly (that needs S3 interp). S1's own gate is
**round-trip + physical-range**: decoded TT ∈ [180,330] K, PSFC ∈ [50000,105000] Pa,
SPECHUMD ∈ [0,0.03], |UU|,|VV| ≤ 120 m/s, ST ∈ [220,330] K, SM ∈ [0,1]; and a
re-encode→decode round-trip is bit-stable. The against-oracle parity is S4.

## Validation Commands
```
JAX_PLATFORM_NAME=cpu taskset -c 0-3 python3 -m pytest tests/init/test_aifs_grib.py \
  tests/init/test_forcing_decode.py -q
```

## Proof Object
`proofs/v030/s1_forcing_decode_report.json`: per-field decoded ranges, level lists,
the surface-assembly + dewpoint→q recipe with source refs, provenance, and the
round-trip check result.

## Risks
- ECMWF soil-moisture GRIB2 param decodes as `unknown` — MUST use the Vtable triplet.
- gh vs geopotential confusion (factor g). Verify against met_em GHT magnitudes
  (RECON: ~167 m surface, ~20786 m at 50 hPa).
- Dewpoint→q formula choice (Bolton vs Tetens vs WMO) must match what WPS used.

## Handoff Requirements
objective, files, commands, proof, the exact surface-assembly + dewpoint→q recipe
(with WPS source refs) so S3/S4 know what to expect, unresolved risks.
