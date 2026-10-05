# v0.3.1 — WRF-order corrections, snow fidelity and faster cached forecasts

**v0.3.1, 2026-10-05.** All registered release gates pass. The final Swiss
2023-01-15 comparison covers all 25 hourly frames in both domains against the
complete original CPU-WRF reference, with zero D6 failures and a direct RC3
strict output-integrity pass.

This release corrects the systematic dry-mass offset, several boundary and snow
process differences, and history-field conventions. It also enables measured
acoustic/Noah-MP speedups and repairs warm-case admission and three-nest memory
planning. It preserves the public plain-CLI recipe; no development lock or
`--force-gpu-run` is part of the default user instructions.

## Physics and initial state

- **Open-top boundary condition.** The nested pipeline had forced a rigid lid
  even when the case used WRF's default open top. It now honors `top_lid` from
  the namelist and defaults to `.false.`. The namelist reader also preserves
  Fortran null entries and repeat syntax. The Swiss dry-mass bias fell from
  approximately +54 Pa to +0.37 Pa; a pristine CPU-WRF rigid-lid experiment
  reproduced 93% of the old signal. This was a coordinate/boundary-condition
  defect, rather than a reason to clamp or subtract a pressure offset.
- **Boundary records use their own dry mass.** Root `wrfbdy` temperature, wind,
  height and moisture records are decoupled with the mass of each record,
  rather than the initial mass for the entire forecast. The previous Swiss
  outer-strip discrepancies, roughly 0.31 K and 41 m²/s², fell to REAL4 rounding.
- **Normal-wind relaxation happens once.** An extra blend in the root's
  acoustic substeps duplicated WRF's relaxation tendency on the inner edge
  rows. Removing it restores the WRF update order without suppressing the
  prescribed boundary or altering the relaxation-zone width.
- **Andreas snow surface layer.** MYNN's heat and moisture roughness lengths
  over land snow depths of at least 0.1 m use WRF's Andreas formulation, with
  distinct `ZT` and `ZQ`, and the snow depth at entry to the Noah call. Momentum
  roughness and the state/restart schema are unchanged. On the real Swiss
  February snow cells, the 24-lead mean wind RMSE fell from 0.141 to 0.062 m/s
  on d01 and 0.192 to 0.084 m/s on d02; T2 RMSE fell from 0.143 to 0.101 K and
  0.200 to 0.113 K. These compare the release candidates before and after this fix.
- **Nested lead-zero state.** Child initialization follows WRF's terrain-blend
  and rebalance path. The initial history frame exposes the initialized nested
  state, rather than an unadjusted input snapshot.
- **Thin snow films and ground precipitation (NF12).** The WRF end-of-step
  reset removes vanishing snow films. `QSNOWXY` and `QRAINXY` now report ground
  snow/rain rates rather than column sums of atmospheric mixing ratios. Bounds
  and deletion checks come from pristine WRF, including threshold-edge columns.
- **Warm cloud ice (NF13).** Above 0 °C, Thompson cloud ice follows WRF's
  melting path instead of also entering cold-block sublimation/deposition and
  ice autoconversion. Pristine-driver trace columns and deletion controls
  verify both the process selection and latent-heat response.

## History output

Water-cell `TSLB` follows WRF: the top layer reports prescribed SST and lower
layers retain their WRF values, without exposing dummy ocean-soil evolution.
Initial land statics and phase-dependent Swiss missing values are corrected.

Glacier-landuse history now emits WRF's undefined-value conventions for 32
vegetation fields and `Q2V`. The normal sentinel is float32(−1e36); `Q2V` uses
−1. Lead zero remains unchanged. In the writer-only final candidate, all 50
Swiss January files were checked against the previous candidate: no dimensions,
global attributes, non-target cells or non-target variables changed. The changed
fields also match the complete original CPU reference on all 50 frames. This
proves the writer change alongside the independent 24-hour numerical comparison;
it does not establish dedicated glacier-runtime parity.

## Performance and parallel operation

The measured cached Canary 9/3 km run on RTX 5090 takes **86.507 s for 24 forecast
hours, or 3.604 s/forecast hour** [M], through process completion with initialization
and output included. The v0.3.0 comparison is 4.126 s/hour: **12.6% less wall time**,
or 14.5% higher throughput. Against the unchanged original CPU-WRF 12-core main-loop
reference, 92.3 s/hour, the rate ratio is **25.6×**. These clocks differ; they are
stated rather than presented as identical timing windows.

The short warm proxy is 3.088 s/hour, with repeated arms at 3.101 and 3.075
(0.84% difference). It is a diagnostic stepping/output proxy, not the whole-run
headline. The 24-hour steady-segment median is 2.791 s/hour and pre-step initialization
is approximately 17.1 s. The original candidate's noisy 21% repeat is rejected
and is not used in release speed claims.

Defaults now include the blocked acoustic vertical-velocity recurrence (A1-SL)
and Noah-MP layer-list/column kernels (#14). Their component/device measurements
compose into a 3.9% reduction in ordinary-root device time. End-to-end results
above determine the release speed; component gains are not multiplied together.

The parallel launcher recognizes warm required executables using their binary
and metadata receipts. It reserves host headroom for cold compilation, records
the resolved per-domain controls in the run proof, and supports the fused
d02/d03 executable in a three-nest C-auto memory plan. The real capped WN3 test
retained all 75 original history-file hashes. Run a new version or geometry alone
first.

The measured v0.3.1 WN3 three-nest Tenerife sweep uses **6 h runs on one RTX
5090** [M], with the shipped launcher, fused C-auto and one retained autotune pin:

| Concurrent cases | Whole-run aggregate s/case-forecast-hour | CPU-WRF throughput ratio |
|---|---:|---:|
| N=1 | 13.67 | 9.0× |
| N=3 | 8.20 | 15.0× |
| N=4 | 7.77 | 15.9× |

The CPU-WRF 3×4-core throughput reference is 123.4 s/case-hour. The GPU aggregate
clock runs from the first case process start to the phase's last history file,
divided by N×6 forecast hours. **These six-hour runs include per-case init/load
and omit `--compress`; v0.3.0's FINAL-b R4 used compression.** No 24-hour or
energy extrapolation is made.

Warm per-case VRAM is 5,354 MiB (5.23 GiB); reported warm host VmHWM is about
6.1 GB per case. All cases returned rc 0; N=3 and N=4 were admitted together
at phase start. The host MemAvailable floor was 50.4 GB. The separate launcher-wall
metric is 14.28/8.46/7.99 s/case-hour at N=1/3/4, versus FINAL-b R4's
16.63/9.63/9.68: reductions of approximately 14%/12%/17.5%, with the compression
difference disclosed above. The shared 0227 case retained all 21 file hashes
across concurrency levels; this is a GPU regression check, not CPU fidelity.
The v0.3.0 production-batch figures remain dated historical measurements.

## Validation and remaining limits

Completed numerical gates compare with original CPU-WRF: Canary d01/d02 at
6 and 24 hours, Tenerife 0227 d01/d02/d03 at 24 hours, and Swiss 2025-02-18
d01/d02 at 24 hours, all with zero frozen-bound failures. The restart regression
check retains 2,262 variables exactly. The writer-only final candidate preserves
six Canary whole-file hashes and all Swiss non-target values, so the earlier
candidate's unchanged numerical fields carry over with explicit byte evidence.

The **Swiss 2023-01-15 full CPU gate passes** on d01/d02, all 25 frames including
lead zero: zero D6 failures. Worst whole-domain RMSE (d01/d02) is T2
0.255/0.441 K, U10 0.236/0.487 m/s, V10 0.249/0.498 m/s, PSFC 3.933/7.140 Pa
and RAINNC 0.310/0.416 mm. Mean interior 10 m wind-speed bias over leads 1–24
is −0.017/−0.028 m/s. Direct RC3 strict integrity passes 25/25 frames in each
domain, with zero degenerate fields, missing variables or attribute gaps. The
50-frame glacier writer identity check also passes against the complete CPU
reference. The November and June Swiss CPU references are further studies, not completed
release evidence. The six-case 72-hour and Canary 162-hour studies are v0.3.0
results and have not been repeated for v0.3.1.

Strict integrity keeps raw failures and registered disclosures: **R2** trace
ice/snow, **R4** canopy-water onset/decay timing, and **R5** canopy-ice/ground-snow
threshold traces. These are explicit field/frame exceptions, not a blanket
output-integrity pass. The dedicated Noah-MP glacier physics runtime is still
unported.

XLA's nested-domain layout can vary with build/settings. A proposed manual pin
failed its full value-preservation/performance gates and is not shipped. Three
small Pallas auxiliary programs can recompile for a different installation path;
main forecast programs still reuse their path-independent AOT keys. EOS metadata
pruning remains **off by default** because its removed copies did not yield a
measured standalone wall/VRAM gain. Other GPU hardware and multi-GPU execution
remain untested/unimplemented as described in [known issues](../KNOWN_ISSUES.md).

See [the evidence report](../docs/release/V0.3.1.md) for source hashes, exact
clocks and completed scopes. MIT applies to the project's
own code; third-party files retain their [upstream terms](../THIRD_PARTY_NOTICES.md).
