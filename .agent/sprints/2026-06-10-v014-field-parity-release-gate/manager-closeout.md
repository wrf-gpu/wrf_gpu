# Manager Closeout

## Outcome
Accepted. v0.14 is now field-parity-first by explicit principal decision:

- Required gate 1: Switzerland/Gotthard 72h CPU-WRF vs GPU-JAX field
  parity/stability.
- Required gate 2: Canary L2 d02 72h CPU-WRF vs GPU-JAX field parity/stability.
- Required artifact: Grid-Delta Atlas with compact release/paper plots.
- Powered TOST: optional secondary station sanity evidence, not a tag gate.

The Switzerland 24h CPU run cannot be resumed; a fresh 72h CPU truth build/run
has been launched.

## Proof Objects
- `.agent/decisions/V0140-FIELD-PARITY-RELEASE-GATE.md`
- Running baseline:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z`
- Failed formatting-attempt:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122814Z`

## Merge Decision
Merge Decision:

Commit the documentation/governance change now so it survives compaction and
worker handoff. Do not wait for the long CPU baseline to complete before
recording the gate change.

## Scope Changes
TOST is demoted from v0.14 gate to optional secondary sanity evidence. Canary
d02, not d03, is selected for the mandatory 72h Canary gate because complete
retained CPU truth exists now and d03 is currently 24h-retained.

## Lessons
Long validation launchers need a tiny dry-run sanity check on generated
forecast-hour filenames. The failed `f010` attempt cost seconds, not hours,
because it was fail-closed before WPS/WRF started.

## Next Sprint
Monitor the Switzerland 72h CPU baseline to completion; then run exact-branch
memory preflight and launch Switzerland 72h GPU with resource CSV logging. In
parallel, finish the current 1h Canary field falsifier and use its result to
select the first L2 d02 72h GPU case.
