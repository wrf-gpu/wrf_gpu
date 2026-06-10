# V0.14 Field-Parity Release Gate

Date: 2026-06-10 13:45 WEST
Owner: manager

## Decision

The v0.14 release and paper gate is no longer powered TOST. The required gate is:

1. **Switzerland/Gotthard 72h CPU-WRF vs GPU-JAX field-parity/stability**.
2. **Canary 72h CPU-WRF vs GPU-JAX field-parity/stability**.
3. **Grid-Delta Atlas** over every paired lead, grid cell, and common numeric
   `wrfout` field, including release-ready stability plots.

Powered TOST is retained as optional secondary station sanity evidence. It can
support the release, but it cannot block or override the all-field parity result.

## Why This Replaces TOST

TOST is useful for station-level weather-skill sanity, but it is not the most
direct proof that the GPU model is WRF-close. Station skill depends on
observation representativeness, verification settings, initial conditions,
terrain mismatch, and short-sample noise. Direct `wrfout` comparison tests what
we actually need to prove for v0.14: all written model fields remain finite,
bounded, and close to CPU-WRF across space and lead time.

For release and paper credibility, a 72h or longer all-cell stability envelope
with plots is stronger than 15 short station comparisons.

## Switzerland CPU Baseline Status

The existing 24h CPU baselines cannot be honestly resumed:

- `restart = .false.` in the Switzerland `namelist.input`.
- No `wrfrst_d0*` files are present in the 24h CPU run roots.
- Existing `wrfbdy_d01` files contain only 0h through 21h boundary times, so
  they do not support a 72h continuation.

Therefore the 72h Switzerland truth must be rebuilt from the same GFS/WPS/WRF
case definition rather than "continued" from the 24h output. The first release
gate should use the 129x129/128-mass-point grid because it matches the accepted
v0.14 24h CPU rerun and lowers first-pass GPU OOM risk. A 151x151 larger
benchmark remains useful after the 128-mass 72h gate is green.

## Canary Domain Decision

Use **Canary L2 d02 72h** as the mandatory v0.14 Canary field-parity gate.

Evidence:

- `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output` already contains 15
  complete d02 CPU-WRF 72h cases with 73 hourly frames each.
- The live-nested L2 d02 GPU path is the path already used by the current
  powered-TOST runner and by the short 1h field falsifier.
- d02 is the 3 km operational target and has enough spatial scale for drift
  diagnostics.
- Retained d03 truth is currently 24h-oriented, not 72h; d03 72h would be a
  heavier 9/3/1 km campaign and should be a secondary or v0.15 gate after the
  two required 72h gates are green.

The Canary d03/1 km path remains important, but it is not the fastest rigorous
wall-clock route to the v0.14 all-field 72h stability claim.

## Required Artifacts

For each required 72h region:

- CPU and GPU run roots with immutable path and command provenance.
- Resource CSVs:
  - GPU runs: `*_gpu_usage.csv`, `*_process_usage.csv`, `*_system_memory.csv`.
  - CPU runs: `*_process_usage.csv`, `*_system_memory.csv` with `--no-gpu`.
- Field comparison JSON/Markdown over every common numeric `wrfout` field.
- Grid-Delta Atlas summary, compact plots, and README/paper-ready dashboard.
- Explicit pass/fail manifest using the accepted tolerance classes.

## Start Signal

Start the long GPU gates only after:

- the short 1h Canary field falsifier has not exposed renewed radical field
  drift or schema failure;
- exact-branch memory preflight is green on the final candidate branch;
- the matching CPU truth exists and is finite for the selected 72h case;
- the GPU run is launched through `scripts/run_gpu_lowprio.sh` with resource
  CSV logging.

TOST may run only after the field gates are already in motion or complete, and
only as secondary evidence.
