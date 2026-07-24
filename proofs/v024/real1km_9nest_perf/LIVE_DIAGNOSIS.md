# real1km 9-Nest Live Performance Diagnosis

## Verdict

The hourly penalty is not history-file writing, allocator fragmentation, or CPU
core overlap. The deployed per-domain path reconstructs the operational advance
factory at every root-history segment and discards its local AOT callable memo.
It therefore deserializes all nine executables again every root hour.

At the ten-segment snapshot the log contained 90 real `path=...xlaexec` loads,
not nine. The active blobs total 2,009,494,602 bytes (1.871 GiB) per reload.

## Direct hour-10 observation

- Previous d09 group completed: 21:46:14.735 WEST.
- Root d01 history write completed: 21:46:45.538.
- d01 through d09 AOT loads completed: 21:47:26.247 to 21:51:39.290,
  a serialized 253.0 s interval.
- Next d03 through d09 output completed: 21:53:28.061 to 21:53:41.913,
  only 13.9 s of file-write span.
- Complete d09-to-d09 hourly-transition interval: 447.2 s.

During the sampled reload interval the RTX 5090 averaged 0.8% SM utilization
and was at 0% SM for 81.7% of samples. The Python process averaged 98.7% CPU,
almost entirely user time. Immediately after reload, GPU SM averaged 57.0%; an
ordinary compute interval averaged 63.0%. The evidence distinguishes serialized
host deserialization from GPU compute and NetCDF output.

The GPU process is restricted to CPUs 12-15 while the parallel 12-rank CPU WRF
is restricted to CPUs 0-11. The service reached 28.04 GB RAM and 11.72 GB swap;
that memory churn can amplify reload time, but it is not the initiating cause.

## Next gate

The bounded fix is a prepared B=1 operational runtime created once before the
segmented loop and reused for all segments. The controlled three-hour A/B must
prove 27 to 9 loads, exact output identity, at least 20% wall improvement,
post-hour gaps at most 180 s and at least 25% better, no transfer/QA regression,
and no more than 512 MiB additional VRAM.

No live source was changed. No GPU A/B is authorized until the active filler is
terminal, product `20260709_00z` is published and verified, and both arms plus QA
fit safely before the next 18z GPU window. A warm fused-runtime experiment with
a compile watchdog remains a separate follow-up for the 1.4-1.5x target.

Machine-readable evidence and raw artifact paths are in `live_diagnosis.json`.
