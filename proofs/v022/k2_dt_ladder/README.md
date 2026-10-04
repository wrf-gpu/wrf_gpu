# K2 — dt / n_sound CFL-first ladder (v0.22, lossless lever)

Measurement + recommended config. NOT a default flip, NOT bit-identity.

## What this is
A CFL-first sweep of the model timestep (`dt_s`) and acoustic substep count
(`n_sound` = `acoustic_substeps`) on the FAST Switzerland 128x128 single-domain
case, then a single best-safe-rung validation on the 3-dom Canary. The levers
are **config-level only** (no core dycore edit); a larger dt with fewer acoustic
substeps is the same math on a coarser step, so results differ from baseline —
acceptance is **CFL-safe + finite + bounded + conserving + within a strict
operational band vs the R1 baseline**, and we report the s/step speedup.

## The max-Courant readout (the evidence this milestone adds)
There was NO trustworthy CFL diagnostic in the codebase (only a mass-continuity
stability fraction in `dynamics/acoustic_wrf.py`). `courant_diag.py` adds the
per-frame advective Courant readout from the wrfout state:

    Cx = max|U|*dt/dx,  Cy = max|V|*dt/dy,  Cz = max|W|*dt/dz   (dz from PH+PHB)

with `C_total = Cx+Cy+Cz` as the combined RK3 advective-stability metric. The
gate is `C_total < 1.0` (conservative; RK3 multidim advective limit ~1.4).

## Ladder (scaled to the case EFFECTIVE baseline dt=10s; config default)
| Rung | dt_s | n_sound |
|------|------|---------|
| R1   | 10   | 10      |  baseline
| R2   | 12   | 9       |
| R3   | 15   | 8       |
| R4   | 18   | 7       |

The 3-dom-Canary validation scales the Canary root dt by the SAME factor as the
best safe Switzerland rung (`scale_namelist.py` edits only root `time_step`; the
child dts follow via the parent_grid_ratio chain).

## Files
- `courant_diag.py`         — max-Courant from a wrfout (the new CFL readout)
- `run_rung.py`             — one rung on the daily pipeline (dt_s/n_sound override
                              + radt held constant in seconds); finite/bounded/mass
- `run_switzerland_sweep.sh`— all 4 rungs under ONE GPU lock (serial; queues
                              behind the corpus / sibling openers, never collides)
- `compare_to_r1.py`        — operational-tolerance RMSE vs R1 (per field bands)
- `pick_best_rung.py`       — best safe rung = largest dt passing ALL gates
- `scale_namelist.py`       — scale root time_step for the Canary validation
- `run_canary_validate.sh`  — 3-dom Canary R1c vs best-rung (own GPU lock)
- `canary_courant_report.py`— per-domain Courant + compare for the Canary
- `chain_after_sweep.sh`    — sweep -> pick -> Canary, autonomous

## Cost guard
Sweep = Switzerland 128 ONLY. Validate ONLY the best safe rung on the 3-dom
Canary. NEVER run the ladder on the all-7 9-nest (each step ~40min).

## Results
See `best_rung.json` (sweep verdict) and `canary_runs/canary_validation.json`
(3-dom validation). FILLED IN by the runs.
