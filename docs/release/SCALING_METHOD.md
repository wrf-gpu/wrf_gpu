# H100/B200/B300 scaling scenarios for v0.3.1

All data-centre GPU results are **[I] inferred**, not measurements of v0.3.1.
**B200 was measured on earlier wrf_gpu versions; H100/B200/B300 values are
extrapolated to v0.3.1, not measured.** The scenario band is not a confidence interval.

## Anchors and clocks

The [retained WN3 sweep](evidence/v031/parallel_sweep_selected.json) measured
three-nest Tenerife cases for six forecast hours on RTX 5090. N=4 whole-run
aggregate time is 7.774 s/case-forecast-hour [M], first case process start to
last phase history file divided by N×6. The mean steady h2–h6 diagnostic is
4.6076 s/case-hour; the difference, 3.1664 s/case-hour, is the fixed-cost proxy
used here. It includes start-up/load/first-hour effects and is not a separately
instrumented host-only interval.

Original CPU-WRF's 3×4-core reference is 123.4 s/case-hour [M]. Its power basis is
**owner-reported approximately 200 W for the twelve-core CPU-WRF run**; the
measurement boundary was not specified as package power. CPU energy of
6.86 Wh (24.68 kJ)/case-hour is a power-times-clock inference [I].

RTX 5090 board energy comes from the [timestamped samples](evidence/v031/energy_samples.json),
integrated by the E158 trapezoid rule inside that forecast window [M]. The
observed N=4 board cost is 0.710 Wh (2.557 kJ)/case-hour. Unlogged endpoint tails total
1.587 seconds, maximum sample gap two seconds. Local sampler timestamps are
converted from Atlantic/Canary to UTC for the run date.

The GPU host share is **200×4/12 = 66.67 W [I]**, not a separate measurement.
Adding it for the full measured window gives 0.854 Wh (3.075 kJ)/case-hour [I] for RTX 5090.
The board segment is measured; the host segment and every combined-energy
comparison are inferred. This is not whole-node/PSU energy accounting.

## Historical transfer and central model

[B200_EXTRAPOLATION.md](B200_EXTRAPOLATION.md) records the older, fp64-heavy,
single-domain measurements: B200/5090 ratios 2.69–3.61, throughput-ceiling ratio
3.53. Those tests used different physics/work; transferring their ratio is an
assumption. The central bandwidth efficiency is 3.53/(8/1.792) = 0.79072.
The lower efficiency is 2.69/(8/1.792) = 0.60256. This **assumes** bandwidth
scaling for device-related work; it does not establish that the current whole
forecast is bandwidth-bound.

For a device with bandwidth ratio `rho`, the central time is:

```text
t_central = 3.1664 + 4.6076 / (rho × 0.79072)
throughput = 3600 / t_central
```

The fixed proxy remains unscaled. For B200/B300, rho=8/1.792, so the central
time is 4.472 s/case-hour, throughput about 805 case-forecast-hours/wall-hour,
or 27.6× the stated CPU reference [I]. H100 central throughput is about 573,
19.6× CPU [I]. These are conditional scenarios, not measured forecasts.

The lower-throughput edge keeps the same fixed proxy and uses efficiency 0.60256.
The optimistic upper edge scales the **whole** 7.774-second anchor by the
historical central efficiency; for B200/B300 it is about 1,635 case-hours/wall-hour,
56× CPU [I]. The former optimistic central bar therefore appears only as the
upper scenario. The theoretical pure-bandwidth cap is not the displayed upper edge.

## Power and capacity scenarios

Data-centre board power is inferred as 0.625×TDP centrally, with a 0.56–0.69
load-fraction band inherited from the RTX/old B200 observations. It is not a
v0.3.1 power measurement. Add the same explicit 66.67 W host-share assumption
and multiply by each corresponding scenario time. Lower energy combines the
low-power/optimistic-time edges; higher energy combines high-power/conservative-time
edges. Central board-plus-host costs are H100 0.880 Wh (3.17 kJ), B200 0.859 Wh (3.09 kJ) and B300 1.170 Wh
(4.21 kJ)/case-hour [I]. Larger capacity does not imply less energy per case-hour.

| Input scenario | Capacity | Bandwidth | Power ceiling used |
|---|---:|---:|---:|
| RTX 5090 anchor | 32 GB | 1.792 TB/s | 575 W |
| H100 SXM | 80 GB | 3.35 TB/s | 700 W |
| B200 physical-capacity scenario | 192 GB | 8 TB/s | 1,000 W |
| B300 physical-capacity scenario | 288 GB | 8 TB/s | 1,400 W |

The B200/B300 physical scenarios are inherited hardware inputs, not claims of
driver-reported usable bytes. NVIDIA's [DGX B200 guide](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html)
lists 1,440 GB across eight GPUs (180 GB each). Here **90% of the 192 GB physical
scenario, 172.8 GB**, is used for central B200 packing, conservatively below
that advertised usable figure. Verify actual NVML memory on a real system.
Other specification sources are [H100](https://www.nvidia.com/en-us/data-center/h100/)
and [RTX 5090](https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/rtx-5090/).

Observed warm allocation is 5,354 MiB = 5.2285 GiB/case [M]. Packing adds
**1 GiB per case** for portability and uses decimal GB capacity consistently.
The central 90% budgets yield H100 10, B200 25, B300 38 cases [I]. The upper
capacity scenario keeps the growth margin and uses 100% of physical capacity;
it is not an actual admission result. Host high-water memory is approximately
6.1 GB/case, so 38 cases need about 230 GB host RAM before reserves. Host cores,
warm loading, allocator differences and scheduling can bind before HBM capacity.

## Outputs and reproduction

The two separate light/dark panels are
`img/v031_scaling_throughput{,_dark}.png` and
`img/v031_scaling_energy{,_dark}.png`. They retain the mandatory inference
caption. [scaling_extrapolation.json](img/scaling_extrapolation.json) stores
every plotted value and the assumptions. The generator reads the retained
observed energy rather than inventing endpoint energy from a mean wattage.

```bash
JAX_PLATFORMS=cpu python scripts/scaling/scaling_extrapolation.py \
  --out-dir docs/release/img
```

The calculation is conditional on transferring an old hardware ratio to a new
kernel mix. Host dispatch, launch latency, power behavior and installation
overhead can differ. No H100/B200/B300 execution or cluster benchmark is claimed.
