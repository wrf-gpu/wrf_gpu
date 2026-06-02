# Sprint Contract — v0.3.0 S4 (WPS parity harness + oracle)

Owner: **GPT** (a sibling GPT lane is already scaffolding this — this contract
formalizes the interface). CPU-only. Owns `proofs/v030/parity/**` EXCLUSIVELY.

## Objective
Build the comparator that grades the native metgrid-equivalent artifact against the
real WPS `met_em.*` oracle, per-variable, at the predeclared schema tolerances, over
the 13 cases × {d01,d02,d03}. Record provenance/units/interp/missing per field.
Provide the met_em format spec + the AIFS→met_em variable map as harness data. This
is the ≥10-case parity GATE evaluator (the gate DECISION is S5).

## Non-Goals
- NO ingest code (S1/S2/S3). The harness consumes a `MetEmArtifact` / met_em-format
  NetCDF and the oracle; it does not produce them.
- NO change to the frozen schema or the per-field tolerances (those are S0-frozen;
  a tolerance change needs manager sign-off + RECON.md note).

## File Ownership (DISJOINT — `proofs/v030/parity/` is S4's alone)
- `proofs/v030/parity/compare_metem.py` — per-variable comparator: aligns native vs
  oracle by name/dims/stagger; computes max-abs, max-rel, RMSE, and the masked-region
  variants (RECON.md §5 masking); PASS/FAIL at schema `parity_tol`/`rel_tol`.
- `proofs/v030/parity/run_parity_campaign.py` — runs the comparator over the 13
  cases × {d01,d02,d03} × timestamps; emits the aggregate report + a per-case CSV.
- `proofs/v030/parity/metem_format_spec.md` — the met_em format spec (dims, vars,
  attrs, FLAG_*; can reference RECON.md §2).
- `proofs/v030/parity/aifs_to_metem_map.md` — the AIFS→met_em variable map (from
  RECON.md §3 + Vtable.AIFS_PURE).
- Optional: `proofs/v030/parity/regen_oracle.sh` — re-run `metgrid.exe` for a case
  (the WPS binary is on disk: `.../WPS/install_gen2_dmpar/bin/metgrid.exe`).

## Inputs
- Oracle: `/mnt/data/canairy_meteo/runs/wps_cases/<case>/l3/met_em.d0{1,2,3}.*.nc`
  (13 cases; RECON.md §2).
- Native artifact: the met_em-format NetCDF from S3's `metem_writer`.
- Schema (tolerances/masking/units): `gpuwrf.init.metgrid_schema`.
- Recon: `proofs/v030/RECON.md`, `recon_inventory.json`.

## Acceptance Criteria
- Comparator handles every present schema field; reports max-abs, max-rel, RMSE, and
  (for masked fields) the metric on the relevant land/water subset only.
- PASS/FAIL per (field, domain, case, timestamp) at the predeclared schema
  tolerances; the campaign aggregates to a per-field PASS-rate.
- Categorical/flag fields (LU_INDEX, LANDMASK, SCT_DOM, SCB_DOM, LANDSEA) graded as
  EXACT (parity_tol=0).
- Provenance/units/interp/missing recorded per field in the report.
- Harness is runnable standalone (does not import S1/S2/S3 internals beyond the
  artifact + schema).

## Predeclared per-variable tolerances (the S4 gate — from the FROZEN schema)
| field | abs tol | rel tol | mask |
|---|---|---|---|
| TT | 0.20 K | — | none |
| UU, VV | 0.25 m/s | — | none |
| GHT | 2.0 m | 1e-4 | none |
| SPECHUMD | 1e-4 kg/kg | 1e-2 | none |
| PRES | 5.0 Pa | 1e-5 | none |
| PSFC, PMSL | 10 Pa | 1e-5 | none |
| SKINTEMP | 0.30 K | — | both |
| ST*, ST000010, ST010040 | 0.30 K | — | water |
| SM*, SM000010, SM010040 | 0.02 | — | water |
| SOILHGT | 1.0 m | — | none |
| DEWPT | 0.50 K | — | none |
| XLAT*/XLONG*/CLAT/CLONG | 1e-4 deg | — | none |
| MAPFAC_* | — | 1e-5 | none |
| F, E | 1e-9 s⁻¹ | 1e-6 | none |
| HGT_M, SOILTEMP | 0.5 | — | none |
| LU_INDEX, LANDMASK, SCT_DOM, SCB_DOM, LANDSEA | EXACT | — | — |
| LANDUSEF, SOILCTOP/CBOT | 1e-4 | — | — |
| GREENFRAC | 1e-3; ALBEDO12M 0.5; LAI12M 1e-2; SNOALB 0.5 | — | — |
(Full authoritative list = `gpuwrf.init.metgrid_schema.metem_field_specs()`.)

## Validation Commands
```
JAX_PLATFORM_NAME=cpu taskset -c 0-3 python3 proofs/v030/parity/compare_metem.py \
  --native <artifact.nc> --oracle <met_em....nc>
JAX_PLATFORM_NAME=cpu taskset -c 0-3 python3 proofs/v030/parity/run_parity_campaign.py
```

## Proof Object
`proofs/v030/parity/parity_report.json` (+ per-case CSV): per-(field,domain,case)
PASS/FAIL + error stats; the aggregate per-field PASS-rate that S5's gate reads.

## Risks
- Stagger/dim alignment between native and oracle (UU on west_east_stag etc.) — a
  silent transpose makes everything "fail"; assert dim names match first.
- Masked metrics MUST use the oracle's LANDSEA, not the native one, to grade the
  water-masked soil fields fairly near coasts.
- Float32 storage rounding (~1e-7 rel) is below all tolerances — do not mistake it
  for error.

## Handoff Requirements
objective, files, commands, the parity_report schema, the per-field PASS-rates,
unresolved risks. The aggregate report is the input to the S5 ≥10-case gate.
