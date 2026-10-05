# wrf_gpu

**WRF-compatible forecasts on one NVIDIA GPU.**

`wrf_gpu` is a GPU rewrite of WRF v4's ARW dynamics and physics, validated against
the original Fortran. It reads WRF inputs, advances the physical equations with
JAX/XLA/Pallas, and writes WRF-compatible history. The aim is practical regional
forecasting: shorter turnaround, more independent forecasts per workstation,
and less energy per forecast hour.

This is an independent implementation, not WRF itself, and is not affiliated
with or endorsed by UCAR/NCAR. The current release is **v0.3.2**. Its fast paths
are enabled by default; the measurements below use that model rather than an
experimental flag collection.

## Speed and energy first

On an **RTX 5090 (sm_120, CUDA 13)** with a **Ryzen 9 9950X host**, a cached
two-domain Canary 9/3 km forecast completes 24 simulated hours at **3.305 seconds
per forecast hour**: a **27.9× ratio to the twelve-core CPU-WRF reference** [M].
Four independent three-domain Tenerife forecasts sharing the card reach
**5.507 seconds per case-forecast-hour**, or **22.4× CPU-WRF's three-by-four-core
throughput** [M], in the measured 24-hour sweep. Its all-four-running steady
window reaches **4.857 s/case-hour, 25.4×** [M].

The GPU timings include start-up and output. The CPU reference clocks cover
their main loops. These are ratios of explicitly stated clocks, not identical
timing windows. Other NVIDIA GPUs and other hosts have not been tested on this
release. The absolute Canary result was measured under host contention and is
conservative; the accepted matched A/B ratio is **0.901, 9.9% less whole-run
wall time** against v0.3.1 [M]. It is not a quiet-host absolute benchmark.
Details and source hashes are in the [measurement report](docs/release/V0.3.2.md).

| Canary 9/3 km, 24 h | CPU-WRF, 12 cores | v0.3.1, matched host | v0.3.2, matched host |
|---|---:|---:|---:|
| Seconds per forecast hour [M] | 92.3 | 3.669 | **3.305** |

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/release/img/v032_speed_dark.png"><img src="docs/release/img/v032_speed.png" alt="Canary forecast seconds per forecast hour: CPU-WRF and the matched v0.3.1/v0.3.2 pair under host contention"></picture>

The matched complete GPU process took **79.319 seconds for 24 forecast hours**,
against **88.060 seconds** for its v0.3.1 reference on the same contended host.
The v0.3.2 steady-segment median is **2.625 s/hour** and initialization/pre-step
work **15.008 seconds**, both diagnostics included in the whole run [M]. The
first unpaired S1 attempt had 11.9% A/A variation and is excluded from speed
claims. Historical v0.3.1's quieter 3.604 s/hour remains a separately scoped
measurement; it is not used to manufacture the matched percentage.

Energy matters as much as speed. For the 24-hour Tenerife sweep, timestamped
board-power samples give **0.678 Wh (2.441 kJ) per case-forecast-hour** at N=4 [M].
Adding an explicit host-share scenario gives **0.780 Wh (2.808 kJ)** [I], versus
**6.86 Wh (24.68 kJ)** [I] for the CPU reference. That is approximately **8.8×
less energy on this accounting basis**, not a measured whole-machine efficiency
claim. The next figure makes the measured and assumed parts visible.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/release/img/v032_energy_dark.png"><img src="docs/release/img/v032_energy.png" alt="24-hour Tenerife energy: CPU reference versus measured GPU board energy plus inferred host share"></picture>

The CPU basis is **approximately 200 W, owner-reported power for the twelve-core
CPU-WRF run**. CPU energy multiplies that power by its 123.4 s/case-hour
reference [I]. GPU board energy integrates actual timestamped `nvidia-smi`
samples with a trapezoid rule [M]. The added host share is **4/12 of 200 W,
about 67 W per GPU**, an assumption [I]. GPU sampling gaps and endpoint tails
are disclosed; no DRAM, storage, PSU or whole-node measurement is invented.
The [energy method and receipts](docs/release/V0.3.2.md#energy-accounting)
state the windows. The shipped four-core Swiss example has no CPU energy claim.

## What it is good for — and what it is not

Use a supported regional WRF configuration with existing `wrfinput`, `wrfbdy`
and `namelist.input` files. Keep WRF preprocessing and downstream analysis,
and run time integration on one GPU. Original-CPU comparisons cover maritime
terrain, convection-permitting nests and winter Alpine snow.

Independent cases are another useful match. Different initial dates, ensemble
members or parameter studies can share the card through separate processes.
The launcher checks both GPU and host memory and queues remaining work. More
concurrency improves aggregate throughput here, but can increase the elapsed
time of an individual case: N=4 is a batching result, not a promise that each
forecast finishes four times sooner.

It is **not universal WRF option parity**, an MPI replacement or a neural weather
surrogate. Multi-GPU domain decomposition is unimplemented. An isolated module
or oracle does not establish operational support: preflight refuses unsupported
selections with a named reason.

## Scope and physics menu at a glance

The v0.3.2 release validation uses **Thompson microphysics, RRTMG radiation, MYNN
turbulence/surface layer and Noah-MP land physics**. Kain–Fritsch is used on
the appropriate outer Canary/Tenerife domains; the Swiss examples disable
cumulus. Those choices define the headline evidence, rather than every possible
combination in the wider scheme catalog.

| Area | Current scope |
|---|---|
| Implemented and measured in this release | WRF real-data inputs; ARW dynamics; the physics combination above; fixed one-way nests; complete history; memory-aware independent-case launcher |
| Implemented, with a historical v0.3.1 regression gate | Nested checkpoint/resume; 2,262 variables byte-identical across six control/resume files [M] |
| Recognized, but refused operationally where unwired | Reference-only schemes and unsupported options; diagnostics explain the selected scheme and reason |
| Out of scope or unimplemented | MPI/multi-GPU domain decomposition, moving/vortex-following nests, coupled chemistry/fire/hydrology, FDDA/4DVAR and stochastic physics |

The wider catalog includes Kessler/Lin/WSM/Morrison/aerosol-aware Thompson,
KF/BMJ/Grell–Freitas/Tiedtke and several boundary-layer/surface-layer pairs.
Catalog support does not extend these measurements to every pairing.
Reference-only examples include CAM-UW/NSSL. The [physics guide](https://wrf-gpu.github.io/wrf_gpu/physics.html)
and [machine-readable catalog](src/gpuwrf/io/scheme_catalog.py) give exact status;
operational preflight is authoritative. Recognized cadence approximations are
reported separately from scheme support.

The dedicated Noah-MP glacier runtime remains unported. Glacier initialization
and missing-value history conventions are corrected, but agreement on a whole
domain does not establish each glacier process. Urban, ocean and other advanced
options should be checked against their individual support status, not inferred
from the presence of similarly named source files.

## Quickstart: run the bundled Alpine case

Use **Python 3.11+**, CUDA 13-compatible NVIDIA drivers and WRF v4 runtime tables.
The tested workstation is RTX 5090/9950X; other recent NVIDIA GPUs should be
usable in principle but are untested [I]. Install the CUDA JAX build before
running the model:

```bash
git clone https://github.com/wrf-gpu/wrf_gpu.git
cd wrf_gpu
pip install "jax[cuda13]==0.10.*"
pip install -e .
export GPUWRF_WRF_ROOT=/path/to/WRF   # contains run/ and its runtime tables

python -m gpuwrf.cli run --input-dir examples/switzerland_d01 \
  --output-dir runs/switzerland --domain d01 --hours 24 \
  --scratch-dir runs/switzerland_scratch
```

The [bundled example](examples/switzerland_d01/README.md) is a small 3 km Alpine
cutout initialized on 2023-01-15 00Z. Its input files ship with the repository.
It selects Thompson/RRTMG/Noah-MP/MYNN, with cumulus off. The example page explains
the expected files, its fresh v0.3.1 benchmark and original four-core CPU-WRF
comparison, including its own strict-integrity findings. Its measured example results are
explicitly v0.3.1; it has not been re-benchmarked on v0.3.2.

**The first run compiles.** It can spend minutes before the first integration
output; a new geometry is a new compilation workload. Later runs reuse cached
executables. Run a new version or domain geometry alone once first, so its compiled programs and memory plan are cached.
Keep the case inputs and runtime tables consistent. v0.3.2 repairs auxiliary
cache reuse after installation relocation; a new version or geometry still
needs compilation.

History uses WRF NetCDF names/dimensions and familiar NetCDF/xarray tools.
A matching header establishes compatibility, not physics fidelity. For a new
case, retain the original CPU reference, input hashes and cadence, and use a
fresh output directory rather than mixing histories.

For multiple fixed nests, use `--domains-from-namelist` so the domain count comes
from the case. For operational setup, follow the [User's Guide](https://wrf-gpu.github.io/wrf_gpu/)
or [AI-assisted operator instructions](AI_OPERATOR.md). `python -m gpuwrf.cli
run --help` lists the supported controls. The normal recipe uses the public CLI;
no development lock or force override is required in a public installation.

## Several independent forecasts on one GPU

```bash
scripts/run_parallel_cases.sh --out-root runs/batch CASE_A CASE_B CASE_C -- \
  --domains-from-namelist --hours 24
```

The launcher admits cases when both the GPU and host budgets allow them,
records the admission decisions and queues the rest. Cold compilation needs
more host memory than a warm forecast. The launcher recognizes the required cached
executables and supports the fused three-nest C-auto memory plan; a guessed
steady-state memory budget is not used to admit several cold compiles at once.

| WN3 Tenerife, three nests, four concurrent 24-hour runs [M] | Whole run | All-four-running steady window |
|---|---:|---:|
| Aggregate s/case-forecast-hour | **5.507** | **4.857** |
| CPU-WRF 3×4-core throughput ratio | **22.4×** | **25.4×** |

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/release/img/v032_parallel_dark.png"><img src="docs/release/img/v032_parallel.png" alt="Measured four-case 24-hour Tenerife whole-run and steady rates"></picture>

These are whole-run figures including each case's start-up, so longer runs amortize it further; output was not compressed (the v0.3.0 figures used compression).
The whole aggregate clock is the first process start to the phase's last history
file divided by 4×24 forecast hours. The steady rate uses the measured window
where all four cases were advancing. The v0.3.1 six-hour N=1/3/4 results remain
[historical context](docs/release/V0.3.1.md#parallel-throughput-m); the change in
run length prevents treating the whole-run difference as a pure code speedup.

Warm per-case GPU allocation was **5,032 MiB**, with reported host high-water
memory around **5.6 GB**. Every case exited successfully, and the shared 0227
case retained all 75 file hashes against cold/validation outputs [M], a GPU
regression control. Those resources are observed values, not universal sizing
constants. Actual admission still checks CUDA/desktop use, host headroom and
the resolved geometry. [Sweep receipts](docs/release/evidence/v032/sweep24_summary.json)
retain the clocks, memory samples and decisions.

## Agreement with WRF v4, checked on the same grid

Fidelity comes from **original CPU-WRF**, not comparing the rewrite with itself.
GPU and Fortran forecasts use matched inputs, paired by domain and valid time.
Common numeric fields are checked against predeclared variable-specific limits.
The release gate is the full **24-hour window**, including paired lead zero.
Bounds are frozen before scoring.

| Original CPU-WRF comparison [M] | Domains | Frames/domain | Frozen-bound failures |
|---|---|---:|---:|
| Canary 9/3 km | d01/d02 | 24-hour gate | 0 / 0 |
| Tenerife 0227, 9/3/1 km | d01/d02/d03 | 25 | 0 / 0 / 0 |
| Swiss 2025-02-18, 3/1 km (v0.3.1) | d01/d02 | 25 | 0 / 0 |
| Swiss 2023-01-15, 3/1 km (v0.3.1) | d01/d02 | 25 | 0 / 0 |

Worst hourly whole-domain RMSE through 24 hours gives a more useful impression
than a pass label alone. The Canary case has T2 **0.066/0.059 K**, U10
**0.073/0.141 m/s**, V10 **0.126/0.157 m/s** and PSFC **1.55/1.55 Pa** on
d01/d02 ([VAL31b 24-hour scores](docs/release/evidence/v031/prod24_d6_selected.json), not the earlier six-hour window). Tenerife's v0.3.2 1 km domain has T2 **0.144 K**, U10/V10 **0.337/0.391 m/s**
and PSFC **8.66 Pa**. Swiss January has T2 **0.255/0.441 K**, U10/V10 up to
**0.487/0.498 m/s**, PSFC **3.93/7.14 Pa** and rainfall **0.310/0.416 mm**
on d01/d02 [M]. These are RMSE values, not maximum single-cell errors.

January's mean interior 10 m wind-speed bias is **−0.017/−0.028 m/s**. Its
**direct RC3 strict output-integrity check passes all 25 frames in both domains**,
with no degenerate fields, missing CPU variables or attribute gaps. The glacier
writer control checks all 50 CPU frames, with zero violations [M]. The Swiss evidence is explicitly v0.3.1, not a new v0.3.2 run. The
[current validation evidence](docs/release/V0.3.2.md) states which build was
scored and which fields are proven unchanged by the writer-only final update.

The v0.3.2 two-domain Canary output is **bitwise versus v0.3.1** on all 48
24-hour history files. Three-nest Tenerife keeps d01 bitwise; children d02/d03
differ at round-off level and remain within the frozen D6 limits. On d03,
worst PSFC RMSE is **8.66 Pa versus v0.3.1's 4.49 Pa**, below the **120 Pa**
limit. It is not an all-nest bitwise claim; actual new error values, raw
differences and registered integrity classes stay published.

Whole-forecast agreement is tolerance-based. Component checks use pristine WRF
savepoints and real columns for snow/water coupling and boundary logic. GPU
restart/byte controls establish regression identity, not independent physics
validation; undefined WRF history values are not physical predictions.

## Honest limitations and what is not claimed

Strict integrity is a second check alongside the numerical limits. Raw findings
on other cases remain visible: **R2** negligible ice/snow traces, **R4** canopy-water
onset/decay timing and **R5** canopy-ice/ground-snow threshold timing. Registered
exceptions retain the original field/frame reports; they do not become an
unqualified clean-output claim. The glacier runtime limitation also remains,
despite corrected initialization and writer conventions.

Longer-lead evidence is explicitly dated. The six-case 72-hour Tenerife and
162-hour Canary studies were run on **v0.3.0**, not repeated on v0.3.2. Canary
stayed within its limits in that older study. Two Tenerife 1 km cases exceeded
the 24-hour-designed bounds: 0227 U10 from h44–72, reaching about 3.7 m/s,
RAINNC at h67–72 and W at h69–70; 0120 U10 at h58, 1.58 m/s. The other four
twins and d01/d02 were clean over 72 hours. Their
[exact frame failures](docs/release/VALIDATION.md) stay published. Without a
perturbed CPU ensemble, those comparisons do not establish a predictability
limit or absolve every error as chaos.

The release does not claim every physics pairing validated, station skill equal
to CPU-WRF in every environment, a measured H100/B200/B300 forecast, a new
multi-day throughput result, whole-system energy savings, or an optimal GPU
implementation in every geometry. Station verification is a separate ALISIOS
activity. CPU tests exercise supporting behavior, while real GPU runs establish
GPU timing and execution. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for the carried
limitations and [methods](docs/release/METHODS.md) for the clock definitions.

## How it works — and where the time goes

WRF's ARW equations remain recognizable: third-order Runge–Kutta dynamics,
time-split acoustic integration, staggered-grid state and column physics.
JAX stages programs for fixed geometry and XLA compiles them; Pallas/native
GPU kernels handle hot stencils, vertical recurrences and column operations.
The Python interfaces retain the case/grid/state/I/O structure while the
expensive execution is rewritten for the device.

Precision is a performance decision with a fidelity gate. **WRF REAL work and
carry use fp32 fast kernels by default; WRF DOUBLE islands retain double
precision.** On the tested consumer GPU, fp64 arithmetic throughput is about
**1/64 of fp32**, inferred from [NVIDIA’s documented SM core counts](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#compute-capability-12-0) [I]. Moving an entire legacy array program onto a card therefore
does not automatically make it fast. The rewrite targets arithmetic precision,
kernel structure and memory traffic together; it does not simply disable
double precision globally or loosen comparison bounds.

The hot path combines larger device operations, keeps forecast carry on the
device and avoids repeatedly materializing intermediates. Column dependencies
need deliberate layouts and recurrences; radiation has different work from an
ordinary dynamics root. Stable executable reuse and buffer lifetime matter
alongside arithmetic. Init, load, history writing and process completion still
cost time, which is why the headline uses a complete forecast clock.

On the historical v0.3.1 Canary profiling program, ordinary-root device time is **30.393 ms** and
radiation-root device time **78.093 ms** [M]. A root represents one outer step
and its nested work; these are profiler diagnostics, not seconds per forecast
hour. Eight conservative kInput flags are bound to their actual PROD runtime:
maximum **0.049 ms**, ordinary-root sum **0.213 ms**, maximum stack **32 B**.
The flags are disclosed rather than called absent. The source/executable/options
binding accompanies the [v0.3.1 runtime evidence](docs/release/V0.3.1.md); these
profile numbers are not transferred to the optimized v0.3.2 program as new measurements.

That historical ordinary kernel-plus-copy **mean is 33.109 ms/root**; the figures
above are kernel-only medians and should not be mixed. The latest accepted
[endpoint evidence](docs/release/S4_ENDPOINT_RC3.md) preserves all measured time:

| Largest ordinary-root families | Device ms/root [M] |
|---|---:|
| Acoustics | 9.835 |
| MYNN | 4.028 |
| Advection | 3.304 |
| Carry/stage glue | 2.499 |
| Positive-definite scalar transport | 2.076 |
| Full kernel + copy mean | **33.109** |

Even deleting all d02 advection, glue and scalar-transport work would cap that
scoped device speedup at **1.27× [I]**. A further 2× needs changes across several
families: a common stable layout and cooperative spatial/column tiles keeping
acoustic coefficients, face fluxes and stage intermediates on chip are a
structural route to investigate. Such a gain is unproven. The traffic model
actually overcounts the measured total; it is not a calibrated bandwidth floor
or a claim of “90% hardware utilization.”

A roofline compares arithmetic capacity with memory bandwidth, but neither
number alone predicts a full forecast. Launch latency, dependent vertical
work, cache state, layouts, output and host scheduling can limit progress
before a theoretical roof is reached. The N=4 sweep reported median SM
utilization near 99% [M]; utilization is evidence of activity, not a proof that
every kernel is optimal. Performance changes still need measured profiles and
the same physical work, with correctness riding alongside.

## Scaling direction: larger GPUs and independent batches

JAX is the foundation for a scalable implementation, and data-centre GPUs are
a development target. Today one forecast runs on one GPU. Independent cases
offer a route to larger batches; distributed domains and a validated multi-GPU
forecast driver remain future work.

The new figures show **best-case potential [I]** for long ensembles and giant
single domains, anchored on the measured v0.3.2 24-hour sweep and PROD steady
rate. A common **32-member × 72-hour ensemble** makes the workload comparable:
the conditional completion estimates are about **195 minutes on RTX 5090**, 134
on H100 and **61 minutes on B200/B300** [I]. Even CPU/RTX durations are inferred
from measured rates; no 72-hour v0.3.2 benchmark is claimed.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/release/img/v032_best_case_ensemble_dark.png"><img src="docs/release/img/v032_best_case_ensemble.png" alt="Best-case inferred 32-member 72-hour ensembles, measured rate anchors and data-centre extrapolations"></picture>

Run length matters because the fixed start-up cost per case is amortized over
more forecast hours. The 24-hour anchor gives a **15.6-second fixed-cost proxy
per case**; it stays unscaled and is divided by 72 in the ensemble scenario.
Device-related steady work scales with bandwidth and a historical efficiency
assumption. These are warm, saturated, well-provisioned cases; cold compilation
and FIFO-tail under-utilization are not priced as a measured completion result.
Resident capacities are H100 **11**, B200 **27** and B300 **40** cases, requiring
roughly **64/156/231 GB host RAM** before reserve [I].

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/release/img/v032_best_case_giant_domain_dark.png"><img src="docs/release/img/v032_best_case_giant_domain.png" alt="Largest estimated single 44-level one-kilometre domain per GPU and matched-size CPU/GPU times, all inferred"></picture>

The giant-domain scenario estimates about **488×488 at 1 km on RTX 5090**,
781×781 on H100, **1,215×1,215 on B200** and **1,490×1,490 on B300**, each with
44 levels [I]. GPU/CPU times are compared at the same domain size for each
card. Gross allocated bytes per cell include pools, context and scratch; the
linear capacity/work assumption does not prove a domain fits or passes physics
validation. The explicit timestep is **6 seconds**. Different layouts, physics,
stability and output needs can change both the limit and the speed.

**B200 measured on earlier versions; H100/B200/B300 extrapolated to v0.3.2,
best case, not measured.** Historical B200/5090 efficiency transfers to this
new kernel mix by assumption; bands are not confidence intervals. The
[best-case scaling method](docs/release/V032_BEST_CASE_SCALING.md) and
[actual normalized anchors](docs/release/evidence/v032/best_case_anchors.json)
record the model, reserves, timestep and source hashes. The v0.3.1 shorter-run
[scenarios](docs/release/SCALING_METHOD.md) remain a conservative, differently
scoped reference; no multi-GPU forecast or data-centre run is claimed.

## What's new, and what comes next

v0.3.2 enables a stable nested-domain layout pin and the measured momentum
fusion, and reduces startup through lazy dormant-scheme initialization and
path-independent Pallas cache configuration. Relocating the measured install
reuses auxiliary programs instead of recompiling them. The final two-domain
identity and three-nest Tier-P results above govern the composed release, not
an assumption that component gains simply multiply.

The v0.3.1 open-top, per-record boundary, single edge-wind, snow/water and writer
corrections remain in the model. No new physics compromise is exchanged for
speed. [Release notes](release_notes/RELEASE_NOTES_v0.3.2.md) give the final
source, matched timing, cache checks and honest limitations.

The next performance work targets measured residuals: nested-domain layout
variation and opportunities to reduce working memory and launch overhead.
The new release pin passed its registered device and identity gates; layout
and buffer-lifetime opportunities still need their own measurements. EOS metadata pruning
remains off by default because removing copies did not produce a measured
standalone wall/VRAM gain. These are research directions, not booked speedups.
Further physics/options, glacier runtime support and longer independent
validation need their own gates; multi-GPU execution is a longer-term direction.

| Version | Short history |
|---|---|
| v0.3.2 | Matched whole-run −9.9%; N=4 24 h throughput 22.4× CPU; stable layout, startup/cache and momentum work |
| v0.3.1 | WRF-order/snow/writer corrections; 3.604 s/hour quiet Canary and six-hour N=4 15.9× throughput, preserved as historical results |
| v0.3.0 | fp32 hot kernels became the default; six 24-hour Tenerife gates, longer-lead studies and a parallel sweep, preserved as historical results |
| v0.23.4 and earlier | Earlier implementation, capability/oracle work and different performance characteristics; not the current precision or speed baseline |

## Get involved and read further

If the rewrite interests you, start with [CONTRIBUTING.md](CONTRIBUTING.md),
the [release methods](docs/release/METHODS.md), the
[validation report](docs/release/V0.3.2.md) and the [User's Guide](https://wrf-gpu.github.io/wrf_gpu/).
Include the namelist, input provenance, version, output cadence and a
reproducible comparison in reports.
[Questions and issues](https://github.com/wrf-gpu/wrf_gpu/issues) are welcome.
Documentation, setup, fidelity, performance and independent validation
contributions are welcome; speedups need scientific and timing evidence.

## License

wrf_gpu's own code is released under the [MIT License](LICENSE). Some files keep
upstream terms: AER's RRTMG/RRTM code and data may not be sold; NCAR MMM physics
translations carry NCAR's BSD 3-Clause notice; WRF-derived material carries the
UCAR public-domain notice. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
wrf_gpu is not affiliated with UCAR/NCAR; WRF® is a registered trademark of UCAR.
