# Worker Report

## Summary
Summary:

Replaced the v0.14 gate in the project documentation: powered TOST is now
secondary station sanity evidence only. The mandatory v0.14 release/paper gate
is Switzerland/Gotthard 72h field-parity/stability plus Canary L2 d02 72h
field-parity/stability, both scored by Grid-Delta Atlas.

Checked Switzerland resume feasibility. The existing 24h CPU roots cannot be
continued honestly because they have `restart=.false.`, no `wrfrst_d0*`, and
`wrfbdy_d01` contains only 0-21h boundary times. Started a fresh detached 72h
Switzerland CPU baseline builder/runner.

## Files Changed
- `.agent/decisions/V0140-FIELD-PARITY-RELEASE-GATE.md`
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`
- `.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`
- `.agent/decisions/V0140-STEP1-TOLERANCE-POLICY.md`
- `.agent/decisions/PAPER-STRATEGIC-FRAMING.md`
- `PROJECT_PLAN.md`
- Sprint files under `.agent/sprints/2026-06-10-v014-field-parity-release-gate/`

## Commands Run
- `find /mnt/data/wrf_gpu_switzerland* ... wrfrst/wrfbdy/wrfout`
- Python NetCDF inspection of Switzerland `wrfbdy_d01` boundary times.
- Inventory counts for Canary L2 d02 72h and L3 d03 retained truth.
- Detached Switzerland 72h CPU baseline launcher with resource monitor.

## Proof Objects
- Decision: `.agent/decisions/V0140-FIELD-PARITY-RELEASE-GATE.md`
- Running CPU baseline root:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z`
- Failed first launcher root, retained transparently:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122814Z`

## Risks
- The Switzerland 72h run is still in progress; the commit records the gate and
  launch path, not a completed CPU truth result.
- The first launcher failed before expensive work because `f000` was formatted
  incorrectly as `f010`; the corrected launcher is now downloading `f027+`.
- Canary d03 remains important, but current retained 72h truth supports d02.

## Handoff
Monitor:

```bash
tail -f /mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/switzerland_72h_cpu.log
cat /mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/runinfo
cat /mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/switzerland_72h_cpu.rc
tail -f /mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/resources/switzerland_72h_cpu_system_memory.csv
```

After CPU truth is complete, launch matched Switzerland GPU 72h through
`scripts/run_gpu_lowprio.sh` with resource CSVs, then build the Atlas.
