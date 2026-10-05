# v0.3.2 best-case scaling — measured anchors and inferred potential

The frozen VAL32 matched PROD receipt and accepted 24-hour N=4 WN3 sweep
now supply the CPU-rendered figures. Source hashes and geometry are retained in
[evidence/v032/best_case_anchors.json](evidence/v032/best_case_anchors.json).
The published v0.3.1 six-hour results remain historical context, not replacement
anchors for the new release.

Required caption: **B200 measured on earlier versions; H100/B200/B300
extrapolated to v0.3.2, best case, not measured.** The CPU and RTX reference
rates are measured anchors; an N-member × 72-hour completion time is inferred
even for those devices. No 72-hour VAL32 timing/fidelity claim is implied.

## Inputs and source checks

The collector requires:

- `integrate/VAL32/FREEZE_REV`, a full commit SHA.
- WN3 24-hour N=4 summary, all four warm cases successful, admitted concurrency
  four, 24 hourly segments, actual per-PID VRAM and host high-water memory.
- The first WN3 case's complete run proof with actual mass-grid shapes and
  `dt_s` for all nests.
- PROD 24-hour S1 receipt and its geometry/device run proof.

All measured source revisions must equal FREEZE_REV. Six-hour, failed, cold or
CPU results are refused as final anchors. Source hashes are retained in the
normalized input. Geometry work counts **cell-updates**, weighting each domain
by its timestep; adding grid cells without accounting for the 54/18/6-second
nest cadence would produce the wrong throughput.

```bash
JAX_PLATFORMS=cpu python scripts/release_plots/v032_best_case_scaling.py collect \
  --freeze-rev /path/to/VAL32/FREEZE_REV \
  --sweep-summary /path/to/wn3/summary.json \
  --sweep-proof /path/to/first_case/wrfout/proofs/nested_pipeline_run.json \
  --prod-receipt /path/to/VAL32/extra24h/run/receipt.json \
  --prod-proof /path/to/PROD/proofs/nested_pipeline_run.json \
  --out /path/to/v032_best_case_anchors.json
```

## Ensemble figure

The common ensemble size defaults to **32 members × 72 hours** and can be
changed on the command line. Using a common number avoids comparing different
ensemble workloads merely because GPU capacities differ.

The measured aggregate steady rate is the producer's all-four-running window
(4.857 s/case-hour), divided by actual case-hours within that window. The
collector supports mean per-case h2–h24 rates only for an older producer schema
without this explicit statistic; it is not used for these final figures. Fixed seconds per case are inferred as:

```text
fixed_seconds_per_case = (24h_whole_aggregate_rate − steady_rate) × 24
long_run_rate = steady_rate / hardware_factor + fixed_seconds_per_case / 72
ensemble_minutes = members × 72 × long_run_rate / 60
```

The fixed proxy includes start-up/load/first-hour and output effects; it is not
a separately instrumented host-only interval. **Run length matters because the
fixed start-up cost per case is amortized over more forecast hours.** The chart
shows steady case-forecast-hours/wall-hour and the inferred 72-hour duration.
Cold compilation is excluded. Adequate host cores, RAM and output bandwidth are
assumed; a partially filled FIFO tail can run below saturation and is not
modelled by the best-case completion time.

H100/B200/B300 device rates use bandwidth ratio × historical B200/5090 efficiency.
The central efficiency is 3.53/(8/1.792)=0.79072; scenario edges use the historical
small-grid ratio 2.69 and the theoretical bandwidth cap. These are assumptions
transferred from old, different-workload B200 measurements, not confidence
intervals or current hardware measurements. The fixed seconds are unscaled.
The [historical scaling method](SCALING_METHOD.md) retains the older B200
measurement provenance and hardware assumptions; those runs are not VAL32 runs.

VRAM packing uses 90% of physical-capacity scenarios, measured warm per-case
allocated memory plus a 1 GiB portability allowance. Host RAM is shown at the
estimated resident capacity, before reserve. B200's 192 GB physical scenario
uses a 172.8 GB packing budget; actual advertised/driver-usable capacity must be
checked. A packing estimate is not an admission receipt.

## Giant-domain figure

The second chart estimates a **single 44-level, square, 1 km domain** on each
device. Gross bytes/cell are measured per-case allocated VRAM divided by the
sum of WN3 mass-grid cells. This includes CUDA contexts, allocation pools,
compiled modules and scratch; it is not a measurement of state-only bytes.

The default capacity scenario leaves 10% device headroom and a further 1 GiB
global reserve, then adds a 10% growth factor to gross bytes/cell. Square side
length is the integer square root of the allowed horizontal cell count. It is
an estimate, not a tested largest domain: scratch scaling, compiler layouts,
tiling and allocation fragmentation can change the limit. R1 linear memory
scaling is not proved by one measured size.

An explicit **6-second timestep** is used for the 1 km scenario (configurable).
PROD steady cell-updates/s provide the GPU work-rate anchor, scaled by the same
bandwidth/efficiency assumption. CPU-WRF twelve-core work-rate scales from the
PROD reference, 92.3 s/forecast-hour. CPU and GPU times are plotted at the
**same estimated domain size for each device**, and every giant-domain result
is [I]. Changes in physics mix, geometry, cadence, stability or parallelism can
invalidate linear work-rate transfer. No generated giant WRF input or forecast
run is implied.

```bash
JAX_PLATFORMS=cpu python scripts/release_plots/v032_best_case_scaling.py render \
  --anchors /path/to/v032_best_case_anchors.json \
  --members 32 --hours 72 --levels 44 --giant-dt 6 \
  --out-dir /path/to/figures
```

Outputs are separate large light/dark ensemble and giant-domain PNGs plus
`v032_best_case_scaling.json`, preserving every scenario value and assumption.
The CPU-throughput reference is 123.4 s/case-hour for WN3 3×4 cores; its 72-hour
ensemble time uses that rate and is [I], not a new twelve-core run.

## Safe preparation before measurements

```bash
JAX_PLATFORMS=cpu python scripts/release_plots/v032_best_case_scaling.py draft \
  --out-dir /path/to/layout_preview
```

The draft produces “WAITING FOR FROZEN VAL32 RECEIPTS” layouts with no invented
benchmark values. Formula/provenance controls use explicitly fabricated unit
inputs; those checks are not physics or GPU-performance evidence. The final figures use source freeze 3e194296c and actual receipts; draft mode
continues to show no values and cannot publish a pending template.
