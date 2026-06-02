# Sprint Contract — v0.3.0 S5 (integration + GATE)

Owner: **manager-merge** (Opus, after S1–S4 land). CPU-only. The milestone close.

## Objective
Wire S1 (forcing decode) + S2 (static geog) + S3 (interp/assemble/write) into one
offline `build_metem_equivalent(case, domain, valid_time) -> met_em-format NetCDF`
pipeline; run the S4 comparator over ≥10 cases × {d01,d02,d03}; enforce the
≥10-case met_em-parity GATE; add the P2-2 namelist checker that rejects unsupported
options on the ingest path; assert NO CPU-WRF wrfinput/wrfbdy/wrfout is consumed.

## Non-Goals
- NO new interp/decode/geog logic (those are S1–S3).
- NO native real.exe-equiv init (that is v0.4.0).
- NO live nesting (v0.5.0).

## File Ownership (DISJOINT)
- `src/gpuwrf/init/build_metem.py` — the integrated offline pipeline entry point
  (calls aifs_grib/forcing_decode + static_geog + metgrid_assemble + metem_writer).
- `src/gpuwrf/init/namelist_v030_check.py` — P2-2 ingest-path namelist validator
  (reuse/extend `gpuwrf.io.namelist_check`): reject options the ingest does not
  support (e.g. fields AIFS cannot supply: SST/SEAICE/SNOW-dependent options unless
  a fallback is declared; non-Lambert projection; >2 soil layers).
- Tests: `tests/init/test_build_metem.py`, `tests/init/test_namelist_v030_check.py`.

## Inputs
- S1/S2/S3 modules (merged); S4 comparator (`proofs/v030/parity/`); the 13-case
  oracle; the frozen schema; the WPS namelist (`namelist.wps`, RECON.md §1).

## Acceptance Criteria (the GATE — falsifiable)
1. **≥10-case met_em parity PASS**: the S4 campaign over ≥10 of the 13 cases ×
   {d01,d02,d03} reports every MANDATORY schema field within its predeclared
   tolerance at ≥ a stated PASS-rate (target: 100% of mandatory fields PASS on
   ≥10 cases; any field below tol is a documented, manager-accepted exception with
   root cause, NOT an amended-away tolerance).
2. **No CPU-WRF artifact consumed**: a static + runtime audit proves
   `build_metem` reads only AIFS GRIB + geo_em (+ WPS METGRID.TBL/Vtable as static
   config); it imports NO `wrfinput`/`wrfbdy`/`wrfout`/Gen2-wrfout loader. Proof: an
   import/IO audit listing every file the pipeline opens for one full case.
3. **P2-2 checker** rejects an unsupported namelist (test: a namelist requesting a
   field AIFS cannot supply fails loudly with a clear message).
4. **Structural**: produced NetCDF re-opens identically to a real met_em file
   (dims/attrs/FLAG_*), and `real.exe` (the v0.4.0 consumer, on disk) ACCEPTS it on
   at least one case as a smoke (real.exe runs to wrfinput without a metgrid-format
   error) — a forward-compat check, not a full v0.4.0 gate.

## Validation Commands
```
JAX_PLATFORM_NAME=cpu taskset -c 0-3 python3 -m pytest tests/init/ -q
JAX_PLATFORM_NAME=cpu taskset -c 0-3 python3 proofs/v030/parity/run_parity_campaign.py
JAX_PLATFORM_NAME=cpu taskset -c 0-3 python3 src/gpuwrf/init/build_metem.py --audit-io \
  --case 20260531_18z_72h --domain d02
python3 scripts/close_sprint.py .agent/sprints/2026-06-02-v030-s5-integration-gate
```

## Proof Object
`proofs/v030/s5_gate.json`: the ≥10-case aggregate parity verdict (per-field PASS),
the IO audit (files opened, proving no CPU-WRF artifact), the P2-2 rejection test
result, and the real.exe accept-smoke result. Plus a one-page `proofs/v030/GATE.md`
verdict for the manager/principal.

## Risks
- Parity may pass on the bulk grid but fail at coastal/masked points (Tenerife soil) —
  the gate must look at the masked-region metric, not just the global one.
- The real.exe accept-smoke needs a built real.exe + the correct namelist.input; if
  the build is not reproducible here, downgrade item 4 to a structural-only check and
  flag for v0.4.0 S1.
- Tolerance honesty: any field that cannot meet its predeclared tol must be reported
  as a tracked gap with root cause, NOT silently relaxed (PROJECT_CONSTITUTION /
  no-slop).

## Handoff Requirements
the GATE verdict, the IO-audit proof, the list of any tracked parity gaps with root
cause, the v0.4.0-readiness note (does the artifact feed real.exe cleanly?).
