# V0.14 Field-Parity Release Gate

Date: 2026-06-10 22:14 WEST
Owner: manager

Update 2026-06-10 22:14 WEST: the missing nested Noah-MP source wiring is fixed
and pushed (`c2310c5b`), with CPU activation/carry proof
`NOAHMP_NESTED_ACTIVATION_CPU_PROVEN`. The follow-up d01 LU16/sand nonfinite
blocker is also closed: `22a2cc0c` fixes Noah-MP WATER 1-based soil/veg
category indexing; `aff7d124`/`5a708074` record closure proof and GPU
confirmation. GPU preflight
`/mnt/data/wrf_gpu_validation/v014_noahmp_l2_preflight_fix_20260610T205333Z`
is `rc=0`, `PASS_SHORT_GPU_PREFLIGHT` / `PIPELINE_GREEN`,
all_domains_finite=true. Long GPU gates now wait on the post-fix h1-h4 land
gate, because soil-water evolution changed for all soil categories.

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

The existing 24h CPU baselines could not be honestly resumed:

- `restart = .false.` in the Switzerland `namelist.input`.
- No `wrfrst_d0*` files are present in the 24h CPU run roots.
- Existing `wrfbdy_d01` files contain only 0h through 21h boundary times, so
  they do not support a 72h continuation.

Therefore the 72h Switzerland truth was rebuilt from the same GFS/WPS/WRF case
definition rather than "continued" from the 24h output. The selected CPU truth
is now complete:

- run root:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z`
- CPU truth:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`
- result: `rc=0`, 73 `wrfout_d01_*` frames, final frame finite PASS
- timing: 2906.3 s total wall, 2887.6 s WRF mainloop, 24 dmpar MPI ranks
- resource proof:
  `proofs/v014/switzerland_cpu72_reference_resource_summary.md`

The first release gate uses the 129x129/128-mass-point grid because it matches
the accepted v0.14 24h CPU rerun and lowers first-pass GPU OOM risk. A 151x151
larger benchmark remains useful after the 128-mass 72h gate is green.

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

The selected mandatory Canary case is:

- run id: `20260501_18z_l2_72h_20260519T173026Z`
- CPU truth:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- retained input/run dir:
  `/mnt/data/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z`
- domain: `d02`, 72h, 73 hourly frames, `159 x 66 x 44` mass grid,
  `DX=DY=3000 m`, `USE_THETA_M=1`
- inventory proof: `proofs/v014/canary_cpu_truth_inventory.md`

This case is preferred because it is already the current h1 field-falsifier case,
so the short-run blocker and the final 72h gate test the same input/provenance
chain. Do not start a fresh Canary CPU-WRF baseline unless this retained truth is
shown to be unusable.

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
- the standalone nested pipeline activates the selected land-surface model
  honestly. The `sf_surface_physics=4`/Noah-MP source wiring is fixed in
  `c2310c5b`, and the d01 LU16/sand nonfinite blocker is fixed in `22a2cc0c`.
  The h1-h4 land gate must now be green/bounded before any release-green 72h
  GPU gate;
- exact-branch memory preflight is green on the final candidate branch
  (`/mnt/data/wrf_gpu_validation/v014_noahmp_l2_preflight_fix_20260610T205333Z`,
  `rc=0`, peak total VRAM `9783 MiB`);
- the matching CPU truth exists and is finite for the selected 72h case;
- the GPU run is launched through `scripts/run_gpu_lowprio.sh` with resource
  CSV logging.

TOST may run only after the field gates are already in motion or complete, and
only as secondary evidence.
