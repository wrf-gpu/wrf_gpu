# Tune WRF for your region

Optimizer runs stop after every iteration/rung for external review. See
[wave reports, exact resume, registry and budgets](optimizer-waves.md).

Start with a prepared WRF case and a question you can check against observations:
for example, whether afternoon winds are too strong at exposed sites or nights
are too cold in a valley. Keep the original case as your control. The steps below
lead into the physics-selection and continuous-parameter tools on this page.

## From a case to a checked improvement

1. **Check and run your control.** Use the same WPS/real.exe inputs for GPU and
   original CPU-WRF. The shipped Swiss case is a small rehearsal:

   ```bash
   gpuwrf doctor examples/switzerland_d01
   gpuwrf run --input-dir examples/switzerland_d01 \
     --output-dir runs/region-control --domain d01 --hours 3
   ```

   This forecast uses the GPU and compiles on its first run. For your own region,
   follow [case preparation](onboarding.md) and keep input hashes, namelist,
   software version and output cadence with the results.
   For several complete namelists, the [precompile queue](precompile.md) plans
   ordinary GPU priming runs on CPU; deviceless GPU compilation remains
   unqualified.
2. **Measure the error.** Prepare [station metadata and observations](station_verification.md),
   then compare both models on the same stations, times and precipitation windows:

   ```bash
   JAX_PLATFORMS=cpu gpuwrf verify --stations stations.csv --obs observations.csv \
     --model runs/region-control/wrfout_d01_* \
     --compare runs/cpu-control/wrfout_d01_* --output-dir runs/region-scores
   ```

   The CSVs must describe real sites inside your grid; the repository does not
   ship observations for this rehearsal. Check station height, wind height,
   coastal sampling and pair counts before interpreting a bias.
3. **Diagnose before tuning.** Compare day/night, land/water and altitude bands.
   Use [radiation diagnostics](radiation_diagnostics.md) for cloud/heating
   questions and the [support matrix](release/SUPPORT_MATRIX_v0.4.3.md) for
   [terrain-radiation behavior](topographic_radiation.md). The radiation page lists the original
   CPU-WRF comparison fields. [MYNN budgets](mynn-budgets.md) (experimental, GPU checks
   pending) write WRF's TKE budget terms and EDMF plume profiles to history for
   turbulence and plume questions.
4. **Change one cause at a time.** Choose among the released physics tuples
   below, or adjust the existing [surface parameters](surface_params.md). The
   [nine new soil/canopy/snow controls](surface_params.md#noah-mp-soil-canopy-and-snow-parameters-research-candidate)
   are excluded from v0.4.4; older roughness controls remain available.
   For a cheap Swiss preparation example:

   ```bash
   JAX_PLATFORMS=cpu gpuwrf surface-params basis \
     --input-dir examples/switzerland_d01 --output runs/swiss-basis.npz
   JAX_PLATFORMS=cpu gpuwrf surface-params expand --basis runs/swiss-basis.npz \
     --output runs/swiss-roughness.npz --set z0mvt=1.1
   gpuwrf run --input-dir examples/switzerland_d01 --domain d01 --hours 3 \
     --output-dir runs/region-roughness --surface-controls runs/swiss-roughness.npz
   ```

   [Thompson parameters](thompson-parameters.md) are research candidates for
   changing coefficients between runs without recompiling. Diffusion/damping
   parameter overrides (including `c_s`) are refused in v0.4.4. They may come in
   a later release after GPU qualification; no date or commitment. MYNN overrides are experimental, with component/short
   two-domain GPU root PASS only; three-nest D6, stations and 24–72-hour drift
   remain unqualified ([family status](runtime-parameters.md)). Track P controls are experimental. Editing a diagnosed surface field in
   wrfinput may have no lasting effect because Noah-MP recomputes it; see
   [what is recomputed](initial_overrides.md#what-noah-mp-recomputes).
   [Initial-field overrides](initial_overrides.md) edit checked private input
   copies; their host/file scope is qualified, while edited forecast gates
   remain open. The [porosity helper](soil_porosity.md) checks effective soil
   caps without changing the soil or qualifying a parameter scheme.
   [Static-field patches](static_patch.md) (`gpuwrf patch`, v0.5.0 development) change
   land-use class, soil, roughness or terrain (via geo_em and real.exe) with
   consistency checks across fields and nests; land-use and terrain variants each
   need their own compile.
5. **Separate initialization from physics.** [Land cycling](cycling.md) carries
   soil/snow state; [atmospheric warm start](atmospheric-warm-start.md) (experimental,
   GPU checks pending) also carries the atmosphere, using WRF's cycling initialization.
   A file warm start reinitializes model internals; checkpoint/resume continues them. Keep paired cold/cycled runs to
   measure spin-up rather than attributing it to a parameter change. For a
   sensitivity test from held model state, use
   [checkpoint branches and short windows](checkpoint-branching.md); their
   host admission is qualified and GPU branch gates remain pending.
6. **Check uncertainty and keep dates out of training.** [Wind perturbations](ensemble.md),
   [ensemble products](ensemble.md) and [experimental data assimilation](da.md)
   answer different questions. Use multiple weather regimes and later held-out
   dates; never use those dates to select the winner. A Swiss twin experiment
   checks the tools, not forecast skill in your region.
7. **Re-run the winning setup on held-out dates.** Compare raw forecasts with
   the unchanged control and observations using identical sampling. Keep RMSE,
   bias, rain-event scores and support counts, including regressions. Only adopt
   a change when those independent results support it. Station MOS calibration
   below corrects products after the forecast; it does not tune the model.

These are gpuwrf workflow extensions around WRF physics. Offline planning,
verification and file preparation have CPU/IO cost and add no timestep kernels.
Every candidate forecast still pays its stepping, initialization, output and
possibly compilation cost. The measured optimizer costs below are examples,
not estimates for an arbitrary region or scheme combination.

## Physics selection

`gpuwrf optimize` picks the best of several **released** physics configurations for a
region by running them as GPU forecasts on past dates and scoring them against station
observations. `gpuwrf calibrate` fits station bias correction or simple linear model
output statistics (MOS). Planning, scoring and calibration run on the CPU with
NumPy/pandas and the station verifier; they add no model kernels or compilation.
Model execution goes through the existing parallel-case launcher. The workflow has been run
end to end on the CPU and on an RTX 5090 (see [Running members in parallel](#running-members-in-parallel-on-one-gpu)).
The result is a held-out diagnostic for your region, not a release promotion of a scheme.
**Continuous parameters** (experimental, v0.4.1): the same command can also tune surface parameters (Noah-MP table
values, the topo_wind CTOPO factor) inside one released configuration by ensemble Kalman inversion, with one
compiled program for every member — see [Continuous parameters](#continuous-parameters-experimental).

## Which physics configurations are released

A configuration is the exact seven-option tuple `mp_physics`, `bl_pbl_physics`,
`sf_sfclay_physics`, `sf_surface_physics`, `ra_lw_physics`, `ra_sw_physics`,
`cu_physics` (one value per domain) for one domain count. **Only tuples that passed the
frozen bounded D6 comparison against original CPU-WRF in the v0.4.0 FINAL validation
are released for optimize**: the 66 FINAL composites plus the release-default runs
(three-nest Tenerife, two-nest Canary Islands, and Swiss). Single options are never released on their own,
because pairing an option with others it was not measured with is a new combination.
Everything else is refused, including operator admission files.

```bash
JAX_PLATFORMS=cpu gpuwrf optimize released                 # all 47 tuples, their runs and E41
JAX_PLATFORMS=cpu gpuwrf optimize released --max-dom 1 --e41 PASS --json
```

The list ships with the package (`gpuwrf/optimize/released_v0.4.0.json`, generated by
`proofs/w41_optimize/build_released.py` from `proofs/r_docs040/RELEASE_CHECKPOINT.json`;
`--check` rebuilds it byte for byte). It applies to gpuwrf 0.4.x only. For v0.4.0 it holds
47 configurations: 44 single-domain, 2 two-domain and 1 three-domain. Each entry lists every
run that qualified it, with geography, gate hours, D6 and **E41** (the compiled-kernel
resource gate). E41 FAIL means the program has known register/stack or fusion-size
findings; it ran and passed D6, but it may be slower or less robust. `"e41": "PASS"` in a
study (or `--e41 PASS`) keeps only tuples with at least one E41 PASS run. Gate hours are
listed too: Goddard radiation 5/5 (c45) was accepted on a 6 h gate only. Options that passed D6
on an earlier source but are not in the FINAL checkpoint (CAM radiation c28, Zhang-McFarlane
c29) are not released for optimize.

For an ALISIOS-style d01-only Canary run, the default tuple with Kain-Fritsch on d01
(`8,5,5,4,4,4,1`, ID `p-c8964de962e8`) is released with E41 PASS (Madrid and Noah-MP
irrigation composites); the same tuple without cumulus (`p-727b941dbde2`) is qualified,
among others, by c60, a d01-only run of the Atlantic root grid with `sst_update`.

The qualifying runs used their own namelists (mostly the 3 km Swiss case). Applying a
tuple to your case combines it with your dynamics, cadence and land settings. The plan
therefore lists, per configuration and case, the namelist settings in which your case
differs from the **closest** qualifying run (`context`). That is disclosure, not a new
qualification: your study's own scores and checks remain the evidence for your region.

## Study

Prepared WRF cases must already exist. The tool does not download observations, run
WPS, change the original namelists or edit model code. Supply a canonical station
metadata CSV and a local observation CSV as defined in
[station verification](station_verification.md). Choose one initialization per UTC date,
at least two earlier training dates and two later held-out dates. The declared date must
match the namelist and wrfinput initialization. Requested hours must contain integral
root time steps and have complete boundary coverage (dt=54 s requires multiples of 3 h).
All cases share one d01 grid, so each configuration compiles one program (per boundary record
count in v0.4.0, see Planning).

Example (paths resolve relative to the JSON; eight configurations, two rungs):

```json
{
  "schema": "gpuwrf.optimize.study.v1",
  "region": "canary-d01",
  "objective": "T2",
  "hours": 24,
  "budget_case_hours": 624,
  "test_start": "2026-04-01",
  "stations": "stations.csv",
  "observations": "observations.csv",
  "configurations": ["c36", "c00", "c30", "c31", "c62", "c64", "c43", "c45"],
  "e41": "PASS",
  "rungs": [{"train_dates": 2, "keep": 4}, {"train_dates": 4, "keep": 1}],
  "cases": [
    {"id": "d0110", "date": "2026-01-10", "input_dir": "cases/0110", "domain": "d01"},
    {"id": "d0201", "date": "2026-02-01", "input_dir": "cases/0201", "domain": "d01"},
    {"id": "d0220", "date": "2026-02-20", "input_dir": "cases/0220", "domain": "d01"},
    {"id": "d0310", "date": "2026-03-10", "input_dir": "cases/0310", "domain": "d01"},
    {"id": "d0405", "date": "2026-04-05", "input_dir": "cases/0405", "domain": "d01"},
    {"id": "d0420", "date": "2026-04-20", "input_dir": "cases/0420", "domain": "d01"}
  ],
  "sampling": {"method": "bilinear", "land_aware": true, "wind_frame": "grid"}
}
```

`configurations` names released entries by ID (`p-...`) or by a qualifying run (`c36`,
`F20-WN3-0227`), or is `"all"` (every released tuple with the case domain count, after the
`e41` filter). Configurations that the operational namelist binding refuses for your case
(for example RUC without six soil layers) are listed under `rejected`. The older
`admissions` files (`gpuwrf.optimize.admission.v1`, bound to model source, version,
domain count and GPU evidence) are still accepted, but only for released tuples: an
admission file can add evidence, never widen the release. An optional `search_space` (all seven
keys, lists of options) intersects the candidates; at most 4096 combinations.

**Rungs.** Successive halving over dates: rung *r* scores the surviving configurations on
the first `train_dates` training dates and keeps the best `keep`. `train_dates` must
increase and the last rung uses every training date and keeps exactly one configuration. The winner then runs once on all held-out
dates. Without `rungs`, the old automatic schedule applies (two dates, doubling, best half).
The example costs (2 + 8×2 + 4×2) × 24 = 624 case-hours: 16 members in rung 1, 8 new members
in rung 2 (the 4 survivors on 2 more dates), 2 held-out members. Already evaluated dates are
reused, later dates never decide which configuration wins, and ties break by stable ID.

## Planning

```bash
JAX_PLATFORMS=cpu gpuwrf optimize plan --study study.json --output plan.json
```

The plan lists candidates (release evidence, context, warnings), `rejected`, `rounds`,
`planned_case_hours` and:

- `program_groups`: the boundary/lower-boundary record counts per case. In v0.4.0 the
  record count of `wrfbdy_d01`/`wrflowinp_d0N` is part of the compiled program shape, so
  each distinct count is a separate cold compile per configuration
  (`cold_programs_upper_bound`). Generate every case's inputs over the same window length.
  Once the step programs read only the active record window (`record_count_independent`
  true, v0.4.1), one program per configuration serves every case.
- `resources`: the per-member GPU memory estimate and how many members fit (below).

## Running members in parallel on one GPU

Each (configuration, date) pair is one member, i.e. one `gpuwrf run` process. A rung
submits **all** its new members, of every surviving configuration, to the parallel-case
launcher; the launcher starts as many as GPU memory and host RAM admit and queues the rest.
Members are sent in two waves: first one member per configuration whose program is not
compiled yet (cold), then the remaining dates, which load that program instead of compiling
the same program again in parallel. Nested studies are sized by the launcher's C-auto memory
plans. Single-domain members have no C-auto plan, so the optimizer passes a fixed per-member
pool (`--pool-gib`, default from the plan).

**Measured on the RTX 5090** (release-default physics, cu 0 and KF cu 1; 3 h members):

| member | VRAM per process | host RAM peak | wall |
|---|---|---|---|
| Tenerife d01 grid 120×70×44, 2 in parallel, cold compile, 1.0 GiB pool | 1,712 / 1,744 MiB | 6.4 GB | 631 / 638 s |
| Swiss d01 42×42×44, warm from the v0.4.0 bundle, 1.5 GiB pool | 2,214 MiB (pool-bound) | 2.8 GB | 41 / 49 s |
| Swiss d01, KF, cold compile into the bundle layer, 1.5 GiB pool | 2,216 MiB (pool-bound) | 6.0 GB | 466 / 491 s |

For the Tenerife d01 grid the plan therefore uses a 1.25 GiB pool (the 1.0 GiB that sufficed
plus 25 % for physics that was not measured), i.e. 2.25 GiB per member with the CUDA
context: **12 d01-only members next to a 4 GiB desktop** on a 32 GB card (31.35 GiB
visible). Other grids use a formula scaled from the bundle's measured step-program
footprint (685 MiB arguments+outputs+temporaries and 123 MiB code for the Tenerife d01 program;
the three-nest live peak is 1.41× its footprints) with a 1.5 physics margin:

```
pool = ceil_0.25( (max(0.25 GiB, 685 MiB × columns·levels / (120·70·44)) × 1.41 × 1.5 + 123 MiB) × 1.2 + 0.5 GiB )
need = pool + 1 GiB (CUDA context and modules outside the pool)
```

**Host RAM usually binds first.** A cold member needs about 6–6.5 GB while compiling, a
warm one about 3 GB. Until members are measured, the launcher assumes 16 GB per member;
after the first batch the optimizer passes 1.1 × the largest measured peak
(`--host-gb-per-member` overrides it), and warm waves use the warm peaks.

```bash
gpuwrf optimize run --study study.json --output tournament --chunk-minutes 45
# later, for example in the next free GPU window:
gpuwrf optimize run --study study.json --output tournament --chunk-minutes 45 --resume
```

**Resumable chunks.** Completed members are recorded in `tournament/ledger.json` (wrfout
SHA256, launcher record: rc, wall, host peak, per-process VRAM peak, how many members the
launcher actually ran at once). With `--chunk-minutes` every batch is sized to the chunk time
that is left when it starts: members per wave = the slowest measured member wall (else
`--member-wall-s`, default `hours×60 + 900` s), and members at once = what the launcher will
admit, never just `--max-parallel`. Until members are measured the optimizer assumes **one at a
time**; afterwards it applies the launcher's own host and VRAM rules to the measured peaks and
to the free VRAM the launcher saw, and a batch in which the launcher ran fewer members at once
than it had caps later batches. When nothing fits the rest of the chunk, the command exits with
status 3 and `progress.json` says `PAUSED`. If a single member is estimated longer than the whole
chunk, the first batch of an invocation runs that one member alone and records `over_chunk` in
the ledger; choose a chunk longer than one member. `--resume` continues in the same directory
with the identical study, reuses every recorded member (its history files must be unchanged)
and salvages members that finished inside a batch that was interrupted. Start a resume only
after the previous run has exited.

**Inputs are bound by content.** The study records the SHA256 of every input file a member uses
(`wrfinput_*`, `wrfbdy_*`, `wrflowinp_*`, `wrffdda_*`, `*.TBL`, the namelist). A resume refuses the
study when any of them changed, appeared or disappeared, and a run stops if one changes while
members run. Hashing reads each file once per command.

**Records per rung.** `tournament/rungs/rung-NN.json` holds the rung's dates, scores,
survivors, the members it used (batch, wall, host and VRAM peaks, AOT cache state) and the
program groups; `result.json` is written only when the held-out evaluation is complete.

**Release bundle.** `--bundle <release bundle directory>` (for example the v0.4.0 bundle) runs
every member with the sealed bundle's source checkout and caches inside a private copy-on-write layer
(`tournament/bundle_layer`, via `scripts/release/bundle_consumer.sh`): programs the bundle
already holds load warm, new programs compile into the layer and are reused by later rungs,
and the bundle itself is never written. Do not combine `--bundle` with other `GPUWRF_*`
model flags.

The compute budget is integer case-hours, not a guaranteed wall time or GPU energy cap.
Every dispatched member is charged, including failed or interrupted attempts and their
retries, so leave headroom above `planned_case_hours` for retries; a launch that would exceed
the budget is refused before it starts. Compile, initialization, IO and launcher waits add
wall time. For CPU rehearsals set
`JAX_PLATFORMS=cpu`: CPU-only members need no GPU lock and never create a CUDA client.

## Continuous parameters (experimental)

A study with schema `gpuwrf.optimize.continuous.v1` keeps ONE released (or admitted) physics configuration and
calibrates continuous multipliers of the [Track P surface controls](surface_params.md) — Noah-MP roughness, albedo,
emissivity, LAI/SAI, … per vegetation class — and the topo_wind drag factor `ctopo`. Every member is an ordinary
forecast whose input directory holds its own control file; the values are runtime inputs of the compiled step
program, so **every member, iteration and evaluation run uses the same executable**: the first member compiles it,
all others load it (the result's `program` block checks this from every member's run proof and launcher receipt).

```json
{"schema": "gpuwrf.optimize.continuous.v1", "region": "my-valley", "hours": 24, "budget_case_hours": 1300,
 "test_start": "2025-07-01", "configuration": "c00",
 "cases": [{"id": "d0610", "date": "2025-06-10", "input_dir": "cases/2025-06-10"}, "...", {"id": "d0702", "...": "held out"}],
 "basis": "basis.npz", "stations": "stations.csv",
 "parameters": [{"name": "z0_forest", "control": "z0mvt", "classes": [1, 2, 3, 4, 5], "prior_sd_log": 0.4, "bounds": [0.3, 3.0]},
                {"name": "z0_crop", "control": "z0mvt", "classes": [12, 14]}],
 "eki": {"members": 12, "iterations": 3, "seed": 1, "tikhonov": true},
 "scorer": {"kind": "stations", "variables": ["t2", "wind"], "sigma": {"t2": 1.0, "wind": 1.2},
            "observations": "observations.csv"},
 "sampling": {"method": "bilinear", "land_aware": true}}
```

- `configuration` (a released ID or run name, see `gpuwrf optimize released`) or `admission` (one receipt, as in
  discrete studies). The cases' own physics are replaced by it, exactly as for discrete members.
- `basis`: `gpuwrf surface-params basis --input-dir <one case>` output. One basis serves cases of one static
  geography: every case must have the same land-use/soil classes, SHDMAX and terrain inputs as the basis case, and
  the basis must equal the Noah-MP tables the members will run (both checked; rebuild the basis otherwise).
- `parameters`: log-multiplier parameters (prior mean 0 = table value, `prior_sd_log` default 0.5, `bounds`
  `low <= 1 <= high`, default [0.2, 5]). Bounds must keep every scaled cell physically valid (z0mvt <= 5 m, EG <= 1,
  leaf reflectance + transmittance <= 1, CTOPO' >= 0, ...): bounds that reach beyond are refused when the study is
  loaded (the message names the feasible factor range); the default [0.2, 5] is intersected with it and recorded in
  `parameters[].bounds` / `feasible_factors`. The EKI clips every member to the bounds, so no member is refused
  mid-study. Two parameters may not scale the same cells of one control. `ctopo` takes no classes; its factor scales
  the topo_wind correction, CTOPO' = 1 + f (CTOPO − 1) (needs `topo_wind` 1/2 with YSU or Shin-Hong; not with
  two-way feedback; a ctopo-only study runs exactly the default program of the configuration). Valley/ramp cells
  with CTOPO < 1 cap the factor at 1/(1 − CTOPO): a domain with CTOPO = 0 cells allows at most f = 1 (e.g.
  `"bounds": [0.2, 1]`; the default prior becomes [0.2, 1]).
- `eki`: Tikhonov-regularised ensemble Kalman inversion (host float64), tempered: `iterations` updates with
  dt = 1/iterations, so the data are used once in total. Each iteration runs `members` forecasts per training date.
- `scorer`: `stations` (canonical station CSV, as for discrete studies; `sigma` = observation + representation error
  per variable, `t2`/`wind`) or `twin` (OSSE: `truth` multipliers per parameter, `noise_seed`,
  `tolerance_log`: truth members run first, their histories at the stations plus seeded Gaussian noise become the
  observations; EKI never sees the truth values; the result says RECOVERED or NOT_RECOVERED).
- Not yet supported (v0.4.1): accumulated precipitation (`precip_rate`). Its observations cover an interval
  (`interval_start`..`valid_time`) that the continuous collector does not carry yet, so a study naming it is
  refused at load instead of comparing mismatched intervals. Discrete studies keep their `precip` objective.
- After the iterations the posterior mean and the prior (multiplier 1) run on every training and held-out date;
  `result.json` reports both (normalised misfit, RMSE and bias per variable), the estimate, the final ensemble,
  the misfit per iteration and the control files of prior and posterior (usable with `gpuwrf run --surface-controls`).

Budget: `(members × iterations + sensitivity probes) × training dates + 2 × all dates`
(+ all dates for a twin), multiplied by forecast hours. The shared prior/control is counted once;
two interior parameters add four sensitivity probes per training date. See
[identifiability before EKI](#identifiability-before-eki) for rank checks and frozen directions.
`plan`, `run`, `--resume`, `--chunk-minutes`, `--max-parallel`, `--bundle` and the admission rules work as for
discrete studies (members of one program: one cold member, then warm batches). `score` is discrete-only.
ALISIOS data roots are refused for every input, the study file included: its real path is checked before the
CLI reads it to pick discrete or continuous mode (stage a copy once with `alisios_guard.facade_stage`).

Status (v0.4.1): experimental. Checked on the CPU with an analytic forward model (EKI recovers a two-class
roughness truth within 1 % in log, one program, pause/interrupt/resume reproduce the uninterrupted study) and as a
GPU twin experiment on the Swiss d01 case (RTX 5090, `proofs/w41_optcont/`): 10 members × 2 iterations of 3 h
forecasts recovered roughness multipliers 1.6 (forest) and 0.7 (open shrubland) as 1.63 and 0.71; the posterior
cut the normalised misfit from 1.86 to 1.04 on the training date and from 1.88 to 1.11 on the (relabelled)
held-out date; all 26 forecasts used one compiled program (one compile of 380 s, then 23–48 s per member, four at
a time), 11.7 GPU minutes in total. The held-out date reuses the training date's initial state, so this shows the
estimator and the plumbing work, not skill. Not done: a calibration against real stations (a twin result does not
say that real stations constrain the parameters) and a `ctopo` forecast on the GPU (its program identity is
checked on the CPU). Hold out dates and station blocks before trusting an improvement, and tune only what the
stations constrain.

## Scoring existing forecasts

```bash
JAX_PLATFORMS=cpu gpuwrf optimize score --study study.json \
  --forecasts forecasts.json --output selection.json
```

Existing forecast inventory:

```json
{"schema": "gpuwrf.optimize.forecasts.v1", "forecasts": {
  "p-<ID from plan>": {
    "jan10": ["control/jan10/wrfout_d01_2026-01-10_00:00:00", "control/jan10/wrfout_d01_2026-01-10_01:00:00"],
    "jan20": ["control/jan20/stations.csv"]
  }
}}
```

List complete selected-domain history files or one canonical station forecast CSV per
case. Populate every required training arm and the eventual winner's held-out arms.
Missing arms fail rather than shortening a round. Offline labels associate supplied files
with configurations; the tool records file hashes but does not attest that externally
supplied forecasts ran the labelled physics. The `run` workflow prepares and records its
actual private namelists instead.

## Objectives, support and uncertainty

For MSG3 cloud occurrence, both discrete selection and continuous EKI use the
[cloud objective and footprint contract](optimize_clouds.md). This is a separate
single-objective scorer with an admitted facade truth copy and frozen daylight
scope; it does not use station CSVs.

Objectives: `T2` (degC, temperature errors also K), `WS10` (scalar speed m/s),
`precip` (total precipitation rate mm/h) and `solar` (instantaneous downward
shortwave `SWDOWN`, W/m²). `all` combines all four and requires explicit positive
`scales`, e.g. `{"t2":1,"wind":1,"precip_rate":1,"solar":100}` in those units.
A single objective defaults to scale 1 in its physical units. The loss is
the mean of `station-balanced RMSE / scale` across selected variables. Each
station has equal weight within a variable; its available errors receive equal
weight. All candidates in a round use the exact same station/time/variable/rain
interval intersection, including case ID. Counts and per-variable RMSEs are
reported. At least two stations and support on every requested case are required.

WRF history extraction reuses `verify`: bilinear/nearest land-aware sampling,
optional explicit lapse-rate and wind-height correction, scalar speed sampled
before vector norm, and complete total rainfall differencing. Missing intervals
are unavailable, accumulation resets/buckets are refused. Observed rain rates
integrate onto the forecast's complete intervals. Solar is an isolated extension
using the same sampler; canonical CSV variable `solar` means instantaneous W/m²,
not hourly mean irradiance or PV output. UTC/offset timestamps, unique keys,
finite numbers and registered stations are mandatory. No NaN can disappear in
an aggregation. Solar-only history should include a regular surface diagnostic
such as T2, as required by the shared verifier.

Initialization rows are read for rainfall differencing but excluded from skill.
Both issue dates and actual scored intervals respect `test_start`: training
valid times precede it, held-out interval starts are at/after it. The held-out
result includes a reproducible 400-replicate UTC date-block bootstrap 95%
interval (seed 0). All stations of a date are resampled together. This interval
describes the selected configuration on supplied test dates; correlated days,
few dates, station selection and repeated studies can understate uncertainty.
It is not proof of improvement against CPU-WRF or a release check.

## Station calibration

```bash
JAX_PLATFORMS=cpu gpuwrf calibrate fit --stations stations.csv --obs past-obs.csv \
  --forecasts past-forecasts.csv --method bias --output station-model.json
JAX_PLATFORMS=cpu gpuwrf calibrate apply --model station-model.json \
  --forecasts future-forecasts.csv --output corrected-stations.csv
```

Historical CSVs contain one selected forecast per station/time/variable/interval,
across dates. The exporter must choose initialization and lead consistently;
this first version does not fit separate lead/season/regime models. Temperature,
scalar speed, rain rate and solar are supported; vector wind/direction is not.

Per station/variable, `bias` predicts `x + mean(y-x)`.
`mos` fits `a+b*x`, where `b=sum((x-mean(x))*(y-mean(y)))/sum((x-mean(x))²)` and
`a=mean(y)-b*mean(x)`. These are
[ordinary least-squares coefficients](https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd431.htm).
MOS refuses constant predictors. At least three earlier training dates per
scored group are needed. Expanding chronological cross-validation fits only
dates before each next validation date, excludes intervals crossing its start,
and records every fold's coefficients, support, raw/corrected RMSE and bias.
It never scores a fitted row as a held-out row. Final deployment coefficients
refit all historical pairs after the validation scores are computed.
Every exported station/variable must have at least one chronological held-out
pair. Fitting refuses the complete model if a group lacks that evidence, even
when other stations have validation scores. Application also checks per-group
held-out support, refusing groups without that support in previously saved models.

Speed, rainfall rate and solar predictions apply the documented statistical
constraint `max(0,a+b*x)`; affected counts are reported. This is a station product
constraint, never a model-state repair. Temperature is unconstrained. Application
requires every valid time and interval start to follow the last fitted valid
time. Unknown station/variable groups fail rather than borrowing coefficients.
The output is canonical scalar CSV plus a provenance sidecar; raw vector wind
components are omitted because they would disagree with corrected scalar speed.
All training/model/forecast files carry SHA256 identities. Past correction skill
does not guarantee future improvement; examine the held-out scores.

Memory scales with extracted station rows and the candidates surviving a round,
not retained full forecast grids. Grid readers stream frames; input hashing and
NetCDF reads have CPU/disk cost. Bootstrap computations operate on station rows.
Actual execution memory/compile cost is that of the admitted physics and existing
batch launcher; this feature makes no new GPU speed or memory claim.

## Joint objectives and reference/control screening

Use [`gpuwrf optimize assess`](optimizer_multiobjective.md) to score aligned
cloud, wind, T2 and RH exports and produce a JSON/Markdown impact matrix with
paired point/region tests against both a supplied reference forecast and the
control. Its admissibility flag must accompany the joint score in wave selection.

## Cloud parameters in continuous EKI

Continuous studies also accept scalar `control` names of the form
`thompson_parameters.nt_c` or `mynn_parameters.alp1`. Multipliers act on the
WRF default in the table below. They are positive log-space EKI parameters;
`bounds` are multiplier limits containing 1. Runtime-only studies omit `basis`.
Surface and runtime controls may be combined; surface controls still need their
usual basis. Module constants require `classes: "all"` and the same value on
every run domain (the default domain selection for runtime parameters).

| Control suffix | WRF default | Admitted scalar range | Unit | WRF provenance |
|---|---:|---:|---|---|
| `thompson_parameters.nt_c` | 100e6 | 2..1999e6 | m⁻³ | module_mp_thompson:88; fixed mp8 droplet number |
| `thompson_parameters.tno` | 5 | 2.5..10 | m⁻³ | :188/2625; Cooper ice-number prefactor |
| `thompson_parameters.ato` | 0.304 | 0.152..0.608 | K⁻¹ | :189/2625; Cooper temperature exponent |
| `thompson_parameters.av_c` | 31694600 | 15847300..63389200 | m⁻¹ s⁻¹ | :163/3661; cloud-droplet fall speed |
| `mynn_parameters.sqfac` | 3 | finite REAL32 ≥0 | 1 | module_bl_mynnedmf:305/3037; Kq/Km |
| `mynn_parameters.alp1` | 0.23 | finite REAL32 ≥0 | 1 | :1761; CASE(1) mixing length |
| `mynn_parameters.cphm_unst` | 16 | finite REAL32 ≥0 | 1 | :275/7724; unstable phi_m |
| `mynn_parameters.am_st` | 6.1 | finite REAL32 ≥0 | 1 | :7712; stable phi_m |
| `mynn_parameters.entrainment` | 0.33 | finite REAL32 ≥0 | m s⁻¹ | :6136; EDMF entrainment numerator |

All nine use the **same compiled program when only values change**, with fixed
selected names, shapes, dtypes, geometry, flags and physics. Enabling a family
or changing its selected names requires another key and may cold-compile once.
The initial EKI multiplier range is 0.5..2; explicit ranges must remain inside
the scalar admission envelope. Log-space bounds contain positive multipliers;
expanded operands are rounded to REAL32 by the public binder. MYNN overrides are
**experimental**: short component/two-domain evidence does not establish
three-nest, 24–72 h drift or station skill. See [runtime parameter scope](runtime-parameters.md)
and [Thompson admission](thompson-parameters.md).

For example, replace the parameter list of a continuous study with:

```json
{
  "parameters": [
    {"name": "droplets", "control": "thompson_parameters.nt_c", "bounds": [0.5, 2]},
    {"name": "length", "control": "mynn_parameters.alp1", "bounds": [0.5, 2]}
  ],
  "cloud_arm": {"icloud": 3, "icloud_bl": 1, "bl_mynn_cloudpdf": 2}
}
```

Keep the usual cases, stations, scorer, configuration, EKI and budget fields.
`gpuwrf optimize plan --study STUDY.json --output PLAN.json` checks the complete
study on the CPU without a model run. Each forecast member receives its own
SHA-bound `runtime_parameters.json`, passed through the ensemble launcher as
`--runtime-parameters`. The runtime binder remains the authority for native-path
requirements. A normal model restart requires unchanged parameter values.
`result.json` exports SHA-bound prior/posterior parameter-file identities under
`runtime_parameters`; pass that file to `gpuwrf run --runtime-parameters`.

`cloud_arm` is **discrete**, one static selection per study. Run separate studies
for the other arms, each with its own output directory and forecast budget:
`icloud=0/1/2/3` selects clear/cloud-default/cal_cldfra2/cal_cldfra3 respectively.
`cal_cldfra3` is `icloud=3`, not a second switch. These arms use different keys
and may each require a cold program; values are never EKI interpolation targets.
The optimizer applies the existing RRTMG/microphysics/cumulus admission matrix.
The optimizer's admitted MYNN PDF arm is CASE(2); `bl_mynn_cloudpdf=2` is forwarded
through main's bound namelist field, overriding an inherited variant. Existing bound `icloud_bl=0/1`
and `bl_mynn_edmf=0/1` branches are exposed as static arms, subject to the existing
MYNN admission. Entrainment is refused with EDMF off. PDF CASE(0/1) remain refused
pending the separate MYNN-options lane; no new kernel is implemented here.
Explicit MYNN arms require PBL 5. A requested `icloud` arm that WRF would reset
to 1 under `cu_physics=11` is refused, so an ignored selection is not reported
as a separate physical experiment.

| Discrete knob | Optimizer admission | Unit / provenance | Compile behavior |
|---|---|---|---|
| `icloud` | 0, 1, 2, 3; existing scheme restrictions apply | integer WRF radiation_driver selection, :1309–1508 | Static namelist key per selection |
| `icloud_bl` | 0, 1 | integer WRF radiation_driver MYNN SGS-cloud merge, check_a_mundo:1802–1820 | Static namelist key per selection |
| `bl_mynn_edmf` | 0, 1; MYNN only | integer WRF Registry MYNN mass-flux selection | Static namelist key per selection |
| `bl_mynn_cloudpdf` | 2 | integer WRF mym_condensation CASE(2), module_bl_mynnedmf:3425 onward | Sole current kernel variant; explicit 2 records provenance; 0/1 need new programs |

The cloud scorer and WN3 constraints use the ALISIOS format contract: MSG3
`times/cloud/lat/lon`, codes 0/1/255, exact UTC matching and an explicit day mask;
centre-assignment footprints in EPSG:6933 weighted by `DX*DY/MAPFAC_M²`; WN3
references are canonical Parquet with role, statistic, interval and availability.
These inputs belong to the scorer, not the parameter files. The FORMAT_ONLY
package is parser evidence only. Real inputs require manager admission and
`alisios_guard.facade_stage`; an unavailable reference cannot establish success.
The existing continuous station scorer remains t2/wind until the cloud-scorer
and multiobjective integration is supplied by their lanes.

### Identifiability before EKI

Every continuous study starts with **at most two active parameters**, including
surface controls. Screen families in separate studies; never search all nine
soil controls together. Before the first ensemble update, the optimizer runs
the prior/control and bounded ± log perturbations on training dates. Each
family's null branch uses the shared control with all selected multipliers at
1 (WRF defaults). This keeps the family enabled and does not assert numerical
identity to the program without an override file. Discrete physics arms remain
separate studies.

The default `identifiability` settings are `step_log: 0.1`, `rank_rtol: 1e-6`,
`collinearity_cos: 0.99`. They describe a numerical sensitivity screen, not a
regional-skill gate. Explicit settings are recorded in the plan and report.
Perturbations stay within the declared bounds; at a bound of 1 the control
replaces the unavailable side. Probe forecasts are charged to the plan's
case-hour budget, in addition to EKI/evaluation. Two interior parameters add
four probes per training date; the prior/control was already in the budget.

`identifiability.json` records the response matrix divided by the admitted
observation errors, singular values, rank, near-collinear columns, bounds and
prior, the admitted physics/nest tuple, and the actual probe receipts (wall,
RSS, VRAM and warm/cold state). It also records elapsed preflight time for the
invocation that completed it; receipt totals persist on resume. Missing compile
breakdowns remain unknown. A completed report is checked before reuse on resume.

Unobservable singular directions stay at prior=0 before and after every EKI
update. If projection leaves the parameter box, the vector shrinks toward the
prior, preserving the frozen null space. Rank zero skips EKI forecasts and
keeps the control. This is local identification evidence at the prior, not proof
of identification under wrong forcing or a different scheme. The real-station
twin and forcing/schema contrasts need their separately admitted inputs.

Raw `msg3:` pixel rows are refused at study admission, before case/model work.
Use the cloud scorer's admitted superobservations by region/correlation length
as the default operator. The guard recognizes `ENSEMBLE_SPACE_UPDATE`: backend
capacity alone does not admit native vectors while the ALISIOS correlation
length/operator contract is unset. Covariance labeling alone does not bypass
this guard. A diagonal observation-error vector is not a
correlation model. Pixel hit/FSS scores remain process checks, not extra
independent observations. The current station scorer accepts t2/wind only.
Superobservations also need the admitted ALISIOS correlation length in the cloud
operator; a region label alone does not establish admission.

### Missing kernel controls

Estimates below are inferred lane-days including CPU oracle/wiring work, with
GPU qualification scheduled separately. They are scoped work estimates, not
performance claims. Every listed control is currently refused by the optimizer.
The pending MYNN scalars below have unit 1 and no admitted runtime range. The
eight PDF constants come from `mym_condensation` CASE(2), WRF
`module_bl_mynnedmf.F:3425–3434`; `cf_thresh` is the DMP shallow-cloud overwrite
threshold at :5741. Their fixed source defaults are shown for provenance.

| Missing control | Remaining work / estimated effort |
|---|---|
| `bl_mynn_cloudpdf=0` | Separate MYNN-options lane; implement/wire CASE(0); 2–3 days shared with CASE(1) |
| `bl_mynn_cloudpdf=1` | Separate MYNN-options lane; implement/wire CASE(1); 2–3 days shared with CASE(0) |
| PDF `qlim_sfc` (0.007) | New traced variance floor in condensation and native path; 0.5–1 day shared PDF scalar bundle |
| PDF `qlim_pbl` (0.020) | New traced variance floor in condensation and native path; 0.5–1 day shared PDF scalar bundle |
| PDF `qlim_trp` (0.025) | New traced variance floor in condensation and native path; 0.5–1 day shared PDF scalar bundle |
| PDF `clim_sfc` (0.010) | New traced cloud-fraction variance floor; 0.5–1 day shared PDF scalar bundle |
| PDF `clim_pbl` (0.025) | New traced cloud-fraction variance floor; 0.5–1 day shared PDF scalar bundle |
| PDF `clim_trp` (0.030) | New traced cloud-fraction variance floor; 0.5–1 day shared PDF scalar bundle |
| PDF `rhcrit` (0.83) | Replace baked threshold and derived constants with traced dependencies; 0.5–1 day shared PDF scalar bundle |
| PDF `rhmax` (1.10) | Trace upper RH threshold through legacy/native consumers; 0.5–1 day shared PDF scalar bundle |
| DMP `cf_thresh` (0.5) | Trace condensate decision threshold through consumers; 0.5–1 day shared cloud scalar bundle |
| Thompson `av_i` | Existing low-level CPU work; runtime GPU/E41/cost admission still missing; 0.5–1 day qualification |
| Thompson `av_r` | Existing low-level CPU work; runtime GPU/E41/cost admission still missing; 0.5–1 day qualification |
| Thompson `autoconversion_threshold` | Existing low-level CPU work; runtime GPU/E41/cost admission still missing; 0.5–1 day qualification |
| Thompson `autoconversion_rate` | Existing low-level CPU work; runtime GPU/E41/cost admission still missing; 0.5–1 day qualification |

`ccn_conc` has no Thompson mp8 consumer and is not an alias for `nt_c`.
Diffusion/`c_s` parameter overrides remain refused until their GPU gate closes.
This optimizer change adds host-side JSON and copies a few scalars per member;
it adds no timestep arithmetic, kernel launches, device synchronization or model
arrays. CPU key/HLO evidence covers scalar plumbing, including full constant
payloads; existing family GPU evidence carries only at its documented scope.
New whole-forecast speed, peak VRAM and RSS are unmeasured.
