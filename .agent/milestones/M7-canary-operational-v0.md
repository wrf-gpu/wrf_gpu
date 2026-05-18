# M7 - Canary Operational v0

Goal: produce a useful Canary daily-run path.

Deliverables:

- 3 km pipeline
- 1 km readiness and memory audit
- I/O and restart plan
- WRF baseline comparison

Acceptance gates:

- **IC/BC mapping proof object** driven by AIFS (per `PROJECT_PLAN.md §11.6`): source dataset, update cadence, boundary-field variables, interpolation policy, restart interaction, one Canary day driven from real AIFS IC/BC
- **`wrf{input,bdy,out,rst}` I/O compatibility matrix** (or explicit deviation document for every intentional difference)
- **restart-continuity test**: N-step → checkpoint → restart → N-step compare within Tier-1 tolerance
- end-to-end 3 km daily pipeline repeatable
- WRF baseline vs. GPU forecast on at least one full Canary day, on the surface/land/SST/static-geog setup frozen by M3/M5
- forecast-vs-observation verification using the M6-selected verification toolchain (METplus or alternative); T2, wind, precip BIAS/RMSE plus one neighbourhood or object-based precip score
- **full Tier-4 ensemble** sized per the cost model approved at M6 closeout
- 1 km memory audit and operational gaps documented
- wall-clock evidence vs. CPU operational baseline

See `.agent/milestones/ROADMAP.md` M7 for full proof-object list.
