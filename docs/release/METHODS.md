# Architecture and measurement methods

This page records v0.3.0 measurement definitions and historical parallel/energy evidence.
For current v0.3.1 timings and validation scopes, see [V0.3.1.md](V0.3.1.md).

The GPU timestep keeps WRF state on the GPU and uses fused fp32 kernels for dynamics,
acoustics and the supported column physics. WRF DOUBLE islands remain in double
precision. JAX/XLA supplies compilation and reusable executables; Pallas controls
selected GPU kernels. Radiation, cumulus and acoustic substeps retain WRF cadence.
Full history and the validated fast paths are the release defaults.

## Timing

PROD uses a cached 24 h run from process creation through the last original output
write, divided by 24 simulated hours. Initialization, first-segment and steady-segment
costs are disclosed separately. CPU-WRF's 12-rank reference is 92.3 s per simulated
hour of its main loop. This older WN2 case illustrates numerics and performance;
Tenerife WN3 cases govern the operational release validation.

The WN3 throughput headline uses the fully admitted six-hour N sweep, including
startup, history output, lossless compression and launcher completion. Divide its
whole launcher wall by N × 6 case-hours. The CPU reference runs three cases on four
cores each: 123.4 s/case-hour on twelve cores. The separately labelled 72 h production
row divides the full six-case batch wall by 432 case-hours, including both FIFO
waves and compression. It is an operational bulk rate, with concurrent CPU post-processing;
the controlled N sweep remains the benchmark headline.
First-publication clocks are retained for interval analysis and are not used as the
whole-launcher endpoint. The first overlapped N=1 predecessor arm is superseded by
a clean repeat, with both receipts kept.

Admission checks GPU memory and host RAM. The benchmark uses measured warm per-case
host high-water ×1.1 and an 8 GB reserve; the original cold-compile high-water is
preserved in the receipt. Cold compilation is reported separately from cached runs.

The measured v0.3.0 three-nest Tenerife runs used demand allocation because
the fused d02/d03 executable was not covered by C-auto planning. v0.3.1 repairs
that planning and warm admission. The older throughput/energy figures retain
their original allocator/work; the v0.3.1 six-hour sweep is recorded in [V0.3.1.md](V0.3.1.md).

The development-throughput comparison uses the measured LW9 WN3 baseline,
which still reported software version 0.23.4. It is not a benchmark of the
previous public tag. Both baseline and new source revisions accompany the ratio;
the public-release version is stated separately in the README.

## Energy and memory

The energy basis is approximately 200 W: **owner-reported power for the twelve-core CPU-WRF run**; its boundary is not recorded as package power. Multiplying
that power by the CPU reference wall gives component energy per case-hour. The Swiss
four-core CPU energy is unmeasured; the twelve-core value is not reused for it.
GPU energy integrates timestamped nvidia-smi board-power samples at one-second cadence;
unlogged endpoint tails and sampling gaps are disclosed. GPU host CPU, other system
components and supply losses are outside this board-versus-reported-CPU-load comparison.

Per-process GPU memory peaks and total board usage are different quantities. CUDA
pool reservations and desktop usage are disclosed where relevant; they are not
presented as the model's working set. Source, input hashes, resolved settings and
actual CUDA device proofs accompany every measured forecast.

## B200 scenarios

[B200_EXTRAPOLATION.md](B200_EXTRAPOLATION.md) identifies the newest old-version
measurements, method and uncertainty. Every modern B200 number is labelled
"extrapolated from B200 runs on an old version". No new B200 or cluster run is claimed.
