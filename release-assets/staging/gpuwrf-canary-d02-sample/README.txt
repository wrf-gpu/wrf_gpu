gpuwrf-canary-d02-sample
========================

Public sample case for the gpuwrf (JAX GPU port of WRF v4 ARW) README
runnability gate. One self-contained Canary Islands d02 (3 km nest) case
from the CPU-WRF / Gen2 operational backfill, packaged so a clean-room agent
can run ONE forecast hour through the GPU port and dimension-compare the
generated wrfout against the CPU-WRF reference.

Provenance
----------
CPU-WRF / Gen2 backfill case 20260429_18z_l2_72h_20260524T204451Z.
Domain d02: west_east=159, south_north=66, bottom_top=44 (e_we=160, e_sn=67,
e_vert=45 staggered). Initial valid time 2026-04-29_18:00:00 UTC.

Contents
--------
  namelist.input                       GPU-port namelist (validated fail-closed).
                                        Faithful copy of the CPU-WRF namelist with
                                        ONE documented deviation: diff_opt 1->0 and
                                        km_opt 4->0 (the GPU operational path runs
                                        neither; it uses diff_6th_opt=2 instead, kept
                                        identical). See the header in the file.
  namelist.input.cpu-wrf-original      The verbatim CPU-WRF namelist that produced
                                        the reference wrfout (provenance only).
  wrfout_d02_2026-04-29_18:00:00       CPU-WRF d02 history t=0  (initial conditions).
  wrfout_d02_2026-04-29_19:00:00       CPU-WRF d02 history t=1h (boundary + 1h ref).
  wrfout_d02_2026-04-29_20:00:00       CPU-WRF d02 history t=2h (boundary margin).
  wrfinput_d02                         d02 wrfinput (Gen2Run inventory / metrics).
  wrfbdy_d01                           d01 lateral boundary file (bdy-width decode).

How to run the gate (requires a GPU)
------------------------------------
Extract this tarball to a directory DIR, install the package
(pip install -e . from the cloned tag), then:

  gpuwrf run \
      --namelist        DIR/namelist.input \
      --input-dir       DIR \
      --output-dir      runs/canary_d02_sample \
      --domain          d02 \
      --hours           1 \
      --compare-cpu-dir DIR

The CLI:
  1. validates namelist.input fail-closed (all schemes supported) BEFORE any
     JAX import;
  2. loads d02 IC + hourly replay boundaries from the wrfout series, advances
     1 forecast hour through the JAX GPU port, writes a generated
     wrfout_d02_2026-04-29_19:00:00 under --output-dir;
  3. dimension-compares that generated wrfout against the CPU-WRF reference of
     the same basename in --compare-cpu-dir and writes
     <output-dir>/proofs/dimension_compare.json (status PASS/FAIL).

A PASS means every NetCDF dimension (Time, west_east, south_north, bottom_top,
the staggered variants, soil/land dims, ...) matches the CPU-WRF reference
exactly. Value/RMSE comparison is a separate, deeper check beyond this gate.

Notes
-----
- This is a REPLAY-nest sample: d02 lateral boundaries are taken from the CPU
  d02 hourly wrfout series (which is why the wrfout files are part of the
  input, not just the reference).
- The sample namelist's run_hours=72 is informational; the gate advances only
  --hours 1.
