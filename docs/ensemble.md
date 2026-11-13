# Ensemble products and prepared workflows

`gpuwrf ens` is an opt-in, offline ensemble tool. Gridded statistics and station
verification use NumPy, the existing NetCDF reader lock and the merged
`gpuwrf verify` extraction/observation adapters. They run on CPU and never step
the forecast model. CLI products, verification and preparation are tested with
JAX imports blocked. The launch action delegates memory admission, multi-GPU
placement, checkpoint stores and retries to `scripts/run_parallel_cases.sh`.
The ensemble tools are tested on the CPU; running many members on one GPU uses
the same forecast program as single runs. Real multi-GPU scaling has not been
measured. CPU tests use explicitly fake members and analytic fields; those
tests are not WRF forecast fidelity evidence.

## Inventory and gridded products

Write an inventory with unique member IDs and explicit files. Relative file
paths resolve relative to the inventory JSON. List one domain and one forecast
initialization per member. Files may each hold one or several output times.
Duplicate paths, repeated inventories, missing fields, masked data, NaN and Inf
are refused. Equal values from distinct real members remain valid.

```json
{"members": [
  {"id": "control", "files": ["control/wrfout_d01_2026-01-01_00:00:00"]},
  {"id": "physics-1", "files": ["physics-1/wrfout_d01_2026-01-01_00:00:00"]}
]}
```

```bash
JAX_PLATFORMS=cpu gpuwrf ens products --members members.json --variable T2 \
  --thresholds 273.15 303.15 --quantiles 0.1 0.5 0.9 --output ensemble_T2.nc
```

The new NetCDF contains `mean`, `spread`, `quantile`, `probability`, `Times`,
latitude/longitude, and threshold/quantile coordinates. Data are float64 offline
statistics; this does not alter model fp32 carry. Every member has weight 1/M,
with M >= 2 fixed across the output. Spread is the sample standard deviation
(ddof=1), quantiles use linear interpolation (NumPy method `linear`), and
probability means **strict exceedance**, `count(X > threshold) / M`. A value
equal to the threshold does not count. Thresholds must increase strictly and
use the input variable's units: `T2` is K, `RAINNC` is the stored cumulative mm,
not an interval rainfall product. Quantile levels must increase and lie in [0,1].

The field must be a 2D or 3D scalar on the unstaggered WRF mass grid. Every frame
must match dimensions, units, projection metadata and exact latitude/longitude;
3D products also require matching `ZNU`. The command does no regridding,
destaggering, time interpolation, rain differencing or circular processing.
Grid wind components and known direction fields are refused; prepare an
earth-frame scalar wind-speed diagnostic first, or use station verification.

By default all member valid-time sets must match. For lagged members, explicitly
pass `--alignment intersection`. This intersects actual UTC `Times` (never
forecast lead indices), records each issue time and excluded frame count, and
refuses an empty intersection. Member count never silently drops at an edge.
The output stores input SHA256s and member provenance. Existing outputs are
refused; failed validation never publishes a partial final NetCDF.

Only one valid-time slice from every member is held at once. Memory therefore
scales with M times one grid plus the current statistics, not forecast length.
There are no GPU kernels, transfers, or compile cost in offline statistics.
Reading/hashing inputs and writing NetCDF have CPU/disk cost.

## Station ensemble verification

Install the optional plotting extra for standard standalone SVG figures:

```bash
pip install 'gpuwrf[ens]'
JAX_PLATFORMS=cpu gpuwrf ens verify --members members.json \
  --stations stations.csv --obs observations.csv --variable t2 \
  --thresholds 0 30 --output-dir ensemble-verification
```

Inventory entries can alternatively contain one canonical station forecast CSV
per member. Observation/station schemas, quality control, timezone validation,
bilinear/nearest sampling, land-aware stencils, earth-frame vector rotation,
lapse-rate and neutral-log wind-height options are those of
[station verification](station_verification.md). CSV predictions are already
corrected by their producer. There are no implicit correction defaults or live
observation requests. Only local exported observations are read.

The scalar variables are `t2` (degC), `wind` and `gust` (m/s), `precip_rate`
(mm/h). Linear wind-direction statistics are refused. Gust requires a declared
field and window for wrfout extraction. Precipitation uses complete total
accumulation through the existing verifier; its reset/bucket refusals apply.
Complete disjoint observation/member rate intervals are integrated onto the
**first inventory member's** precipitation windows using the shared interval
adapter. Missing coverage is unavailable. Instantaneous fields use exact
timestamps. Every score uses the intersection of observations and **all**
members on station, valid time, variable and interval start. Pair counts are
checked against `gpuwrf verify` and source/common row counts are reported.
Sampling is at history times, not every model timestep.

The report directory contains `ensemble.json`, `paired.csv`,
`reliability.svg`, `rank_histogram.svg` and `spread_skill.svg`. Per-sample mean,
spread, quantiles, exceedance probabilities, empirical/fair CRPS and observations
are retained in CSV. JSON also includes each member's existing deterministic
verification scores on common support.

CSV level columns use round-trip float identifiers: `quantile_<level>` and
`probability_gt_<threshold>` (for example `probability_gt_1.0000001`). Their
suffixes preserve the exact requested float levels, so close distinct levels
keep separate columns and values. Integer-valued levels include `.0`.

For member values x_i and observation y, empirical CRPS is
`sum_i |x_i-y| / M - sum_ij |x_i-x_j| / (2 M²)`.
The fair variant replaces M² by M(M-1). Both have the variable's units; lower
is better. The correction is unbiased for an underlying distribution only when
members are iid draws. Physics and lagged members are correlated and generally
not exchangeable; the tool reports both estimators without claiming iid
calibration. The implementation sorts members and evaluates the same pair sum
in O(M log M) time without allocating an M-by-M distance matrix. Independent
literal energy-sum tests cover both estimators.

Ranks range from 0 (observation below all members) to M. With k tied members,
one sample contributes 1/(k+1) to each of its admissible ranks: the deterministic
expectation of uniform random tie breaking. Counts can therefore be fractional,
and always sum to the number of pairs. Reliability uses strict `>` events in
the same units as the selected variable and bins probabilities in [0,0.1), …,
[0.9,1]. It reports bin counts, mean forecast probability and observed frequency;
empty bins use JSON null and are omitted from the plot. Brier score is the
mean squared probability-minus-event error.

Spread–skill reports `sqrt(mean(sample variance))` and the RMSE of the ensemble
mean on the same pairs, plus their ratio (null for zero RMSE). The scatter shows
sample spread versus absolute mean error. These are descriptive diagnostics,
not confidence intervals, independent station-hours or a release verdict.

Formula references: [scoringrules CRPS estimators](https://scoringrules.readthedocs.io/en/latest/crps_estimators.html)
and [xskillscore reliability](https://xskillscore.readthedocs.io/en/stable/api/xskillscore.reliability.html).

## Prepare and launch members

Prepared WRF inputs must already exist. This tool neither downloads forcing nor
runs WPS/real.exe. It hashes inputs once, symlinks the shared wrfinput/wrfbdy and
optional top-level forcing/table files, copies a private namelist per member,
and writes member manifests and `plan.json`. The merged `gpuwrf manifest` API
records source-case input/source/environment provenance once per original case
in `source_manifests/`; physics members share that record and separately record
their selected copied namelist/configuration. The source-case record describes
preparation, while the scheduler records each actual run's environment. Preserve the original inputs while
the plan is in use. Source histories, restart outputs and logs are not linked.
Each member gets its own run/output/checkpoint directory from the scheduler.
Input and manifest SHA256s are rechecked before launch, hashing a shared file
once across members. Changed inputs or namelists refuse launch.

```bash
# One case, distinct namelists; only &physics may differ.
JAX_PLATFORMS=cpu gpuwrf ens prepare --mode physics --case prepared/control \
  --namelists physics/control.input physics/alternative.input \
  --hours 24 --output-dir physics-ensemble --gpus 0,1 --max-per-gpu 2

# Distinct consecutive initializations with uniform spacing and overlapping windows.
JAX_PLATFORMS=cpu gpuwrf ens prepare --mode lagged \
  --cases prepared/00z prepared/06z prepared/12z --hours 24 --output-dir lagged

# Independent dates, no common valid-time window required or pooled skill claim.
JAX_PLATFORMS=cpu gpuwrf ens prepare --mode batch \
  --cases prepared/day1 prepared/day2 --hours 24 --output-dir batch
```

Every wrfinput initialization must agree with its namelist. Delayed nest starts
are refused; root dt must divide the requested hours exactly (dt=54 s needs
multiples of 3 h). Root boundary Times must be ordered and cover the full run.
Preparation does not certify a selected physics scheme: the model's existing
namelist checks and the scheme's validation status still apply. Lagged
plans record the common forecast window; products still use actual emitted
valid times through `--alignment intersection`.

Launch requires a source checkout, which contains the scheduler
(`scripts/run_parallel_cases.sh`):

```bash
gpuwrf ens launch --plan physics-ensemble/plan.json
```

The scheduler preserves the caller's CUDA/JAX settings, assigns one GPU per
member, admits cases by VRAM and host RAM, and records placement and
measurements in its run records. `--checkpoint-interval-steps` and
`--retry-failed` on preparation are passed to its existing isolation/recovery
logic. For offline work set `JAX_PLATFORMS=cpu`; for a GPU launch, run the
command in your normal GPU environment. The CPU tests with fake members make no
hardware throughput or fidelity claim.

### Several members on one GPU (measured with v0.5.0 development code)

Members on one GPU run as separate processes over one warm compile cache
(`scripts/parallel_cases.py`, `--max-parallel N`); the scheduler admits them by
VRAM and host RAM. On an RTX 5090 (32 GB) with three different dates of a
nine-nest Canary Islands case (3 h each, warm cache, no new compiles):

| Members at once | Wall for the three members | Throughput vs one at a time | GPU memory peak | Host RSS peak |
|---|---:|---:|---:|---:|
| 1 | 387 s | 1.00× | 11.2 GiB | 8.4 GiB |
| 2 | 309 s | 1.25× | 19.8 GiB | 17.7 GiB |
| 3 | 242 s | 1.60× | 28.3 GiB | 24.2 GiB |

The `wrfout` files of every member are byte-identical to the one-at-a-time runs.
Each nine-nest member used about 8.6 GiB of GPU memory and 9.7 GiB of host
memory when warm (a cold compile needs 20–22 GiB of host memory), so three
members fit on 32 GB only with little else on the card. Expect more than 1.6×
only for smaller grids.

After a run, create the product inventory from the chosen domain's actual
wrfout files under each `runs/member-*/wrfout`. Batch runs are independent cases;
only combine them into ensemble products if valid times/grid/meaning actually
match. The scheduler's per-member run records and this tool's input manifests
provide complementary provenance.

## Stochastic and multiphysics presets

Two presets generate the member namelists for you from one prepared case. Each
member is a copy of the case's `namelist.input` with surgical edits: only the
preset's lines change, every other byte is kept, and the re-parsed member must
differ from the source only in `&stoch` (stochastic) or `&physics` (+`&stoch`
with `--schemes`) (multiphysics). Inputs are shared exactly as in `--mode physics`;
the generated namelists are kept in `preset_namelists/` and each
`member_manifest.json` records the member's preset settings.

```bash
# 10 members with WRF SKEBS + SPPT (default schemes); nens = 1..10.
JAX_PLATFORMS=cpu gpuwrf ens prepare --mode stochastic --case prepared/control \
  --members 10 --lengthscale-km 15 --hours 24 --output-dir stoch-ensemble

# SKEBS + SPPT + SPP (MYNN) members, numbered from nens 21.
JAX_PLATFORMS=cpu gpuwrf ens prepare --mode stochastic --case prepared/control \
  --members 10 --schemes skebs,sppt,spp --lengthscale-km 15 --nens-start 21 \
  --hours 24 --output-dir stoch-spp

# The first 7 rows of the qualified physics table, or explicit rows plus SKEBS.
JAX_PLATFORMS=cpu gpuwrf ens prepare --mode multiphysics --case prepared/control \
  --members 7 --hours 24 --output-dir physics-ensemble
JAX_PLATFORMS=cpu gpuwrf ens prepare --mode multiphysics --case prepared/control \
  --physics-rows P00,P02,P04,P07 --schemes skebs --hours 24 --output-dir physics-skebs
```

**Stochastic members** follow WRF's ensemble convention: all members keep the
same `iseed_*` values and get a distinct `nens` (`--nens-start`, then +1 per
member). WRF's `rand_seed` mixes `iseed`, `nens` and the start date into each
pattern's random stream, so members are reproducible and different. Preparation
computes every member's seed words with the same WRF/GNU Fortran recipe the
model uses and refuses the plan if any two streams (across members and schemes)
would coincide; the words are written to the member manifest. The switches are
written for every domain (`skebs = 1, 1, ...`); nests receive the parent pattern
as in WRF. Existing `&stoch` tuning keys of the case (amplitudes, time scales)
are kept; a case that already sets stochastic switches or `nens` is refused.

Valid schemes are `skebs`, `sppt` and `spp` (default `skebs,sppt`). The model's
own pre-run stochastic admission is applied to every member before anything is
written: SPPT needs PBL 0/5 and cumulus 0/1, SPP with GSL GWDO (`gwd_opt=3`) is
refused. The preset additionally refuses SPP without MYNN (`bl_pbl_physics=5`):
in WRF 4.7.1 the convection and land-surface SPP consumers are upstream no-ops,
so those members would be unperturbed. Every active pattern is also run through
the model's own d01 spectral setup, which refuses a Gaussian length scale the
grid cannot resolve. Original CPU-WRF 4.7.1 does not refuse: on the 126 km Swiss
domain the 700 km SPP-PBL and 500 km RAND defaults gave NaN spectral
coefficients and a constant clipped pattern (`PATTERN_SPP_PBL` = 0.3 everywhere).
On that grid the limit lies between 200 and 250 km; the 9 km Canary Islands outer grids
resolve all defaults. Use `--lengthscale-km` to set every active Gaussian
pattern (WRF defaults 150 km SPPT, 150/700/50 km SPP conv/PBL/LSM). See
[stochastic.md](stochastic.md) for what each scheme does and how it is
qualified.

**Multiphysics members** take rows of this table. Every row is a complete
`&physics` edit of a composite that passed the frozen D6 manifest against
original CPU-WRF over 24 h on GPU in the v0.4.0 final validation (Swiss d01,
3 km, Noah-MP, otherwise the default namelist; v-prep case id given). Rows marked FAIL in the kernel-check column
are D6-PASS but carry the disclosed flags of the compiled-kernel resource check. A case with a
land model other than Noah-MP is accepted and every member is labelled
`UNQUALIFIED combination` in its manifest.

| row | mp | cu | pbl | sfclay | lw | sw | extra | evidence | kernel check |
|---|---|---|---|---|---|---|---|---|---|
| P00 | 8 Thompson | 0 | 5 MYNN | 5 | 4 RRTMG | 4 RRTMG | | c00 | PASS |
| P01 | 3 WSM3 | 3 Grell-Freitas | 3 GFS | 3 | 4 | 1 Dudhia | | c04 | PASS |
| P02 | 8 | 10 KF-CuP | 5 | 5 | 4 | 4 | | c30 | PASS |
| P03 | 8 | 11 multiscale KF | 5 | 5 | 4 | 4 | | c31 | PASS |
| P04 | 8 | 0 | 9 CAM UW | 5 | 3 CAM | 3 CAM | | c28 | PASS |
| P05 | 24 WSM7 | 93 Grell-Devenyi | 5 | 5 | 4 | 4 | use_mp_re=1 | c17 | PASS |
| P06 | 50 P3 | 0 | 5 | 5 | 4 | 4 | use_mp_re=1 | c21 | PASS |
| P07 | 1 Kessler | 1 KF | 1 YSU | 1 | 1 RRTM | 1 Dudhia | | c02 | FAIL |
| P08 | 4 WSM5 | 5 Grell-3D | 7 ACM2 | 1 | 4 | 4 | | c05 | FAIL |
| P09 | 6 WSM6 | 6 Tiedtke | 8 BouLac | 1 | 4 | 4 | | c06 | FAIL |
| P10 | 2 Lin | 2 BMJ | 2 MYJ | 2 | 4 | 2 Goddard | | c03 | FAIL |
| P11 | 9 Milbrandt | 16 New Tiedtke | 11 Shin-Hong | 1 | 4 | 4 | use_mp_re=1 | c07 | FAIL |

Without `--physics-rows`, `--members N` takes the first N rows (rows passing the kernel check
first). A row qualified on the Swiss case is not automatically qualified for a
different region or resolution, and cumulus rows at convection-permitting grid
spacing are a modelling choice for you to make. `--schemes skebs` adds SKEBS with
distinct `nens` to every row (Berner et al. 2011 combined both); SPPT/SPP with
rows that do not satisfy their physics requirements are refused.

Neither preset claims calibrated spread or probabilistic skill: verify the
members with `gpuwrf ens verify` against stations. Berner et al. (2011)
compared exactly these two representations of model uncertainty in a WRF
mesoscale ensemble and found the stochastic and multiphysics ensembles both
better than a control without model-error representation, with the combination
best on several measures. Reference: Berner, J., S.-Y. Ha, J. P. Hacker,
A. Fournier and C. Snyder, 2011: Model uncertainty in a mesoscale ensemble
prediction system: stochastic versus multiphysics representations. *Mon. Wea.
Rev.*, **139**, 1972–1995. SKEBS: Shutts, G., 2005: A kinetic energy backscatter
algorithm for use in ensemble prediction systems. *Q. J. R. Meteorol. Soc.*,
**131**, 3079–3102.
