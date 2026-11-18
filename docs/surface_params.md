# Surface parameters as traced inputs (Track P) — experimental

`gpuwrf run --surface-controls CONTROLS.npz` replaces selected Noah-MP surface parameters with per-cell value
fields **without recompiling**: the fields are runtime inputs of the compiled step program, so every parameter
set of one grid reuses the same candidate executable for that control signature (same AOT key).
Without the option, this tree uses its candidate-default path. Its scoped evidence is CPU-only;
the private research bundle uses cold layers. This does not establish equivalence to the released
v0.4.6 GPU executable, frozen identities or paid cache (see Cache scope below).

Status: experimental, opt-in, Noah-MP (`sf_surface_physics = 4`) only. Checked so far:

The historical released-program verification below covers the original Track-P
controls; it does not qualify this candidate tree. The new `z0bare` control has
separate CPU evidence and pending GPU qualification, described below.

- every control equals running the released code on an MPTABLE edited the same way (byte-exact, every control);
- two different parameter sets use one compiled program (same cache key on CPU and on the GPU: a second Swiss
  24 h run with other values loaded the first run's executable and compiled nothing);
- Swiss 24 h on the GPU with unchanged (prior) values passes the frozen field tolerances against original
  CPU-WRF (T2 RMSE 0.085 K, 10 m wind 0.04 m/s);
- forecast-level check against original CPU-WRF: a Swiss 24 h run with eight edited parameters moves the GPU
  forecast like the same MPTABLE edit moves CPU-WRF, compared hour by hour over land. LAI responds identically;
  median pattern correlations are 0.97–1.00 for emissivity, albedo, 10 m wind and friction velocity, 0.85–0.90 for
  sensible/latent heat flux and 0.49–0.59 for skin and 2 m temperature. The median hourly response amplitude
  (GPU RMS / CPU RMS) is 0.986–1.009 for every field, but single hours vary more (2 m temperature 0.92–1.07, skin
  temperature 0.89–1.10, latent heat 0.93–1.13). The leftover GPU–CPU response difference has about the same size
  as the ordinary difference between a GPU and a CPU-WRF forecast of the unedited case, which is consistent with
  it being run-to-run noise, but this is an inference, not a proof of its origin;
- resuming a run with different, added or missing controls is refused.

Not done in v0.4.1: calibration against real stations. The EKI tools below are a scaffold, checked on a synthetic
twin experiment (OSSE) only; no parameter set has been estimated from real observations yet. The guarded
real-station runs and the roughness pilot are planned in `proofs/w41_trackp/P1_GPU_PILOT_PLAN.md`.

## What can be controlled

| Control | Replaces (Noah-MP consumer) | WRF source | Needs |
| --- | --- | --- | --- |
| `z0bare` (EXPERIMENTAL) | ground-tile roughness → Z0MG → ZNT/UST | ENERGY local `Z0 = 0.002 m` | spatial extension, not WRF-faithful; real GPU qualification pending |
| `z0mvt` | vegetation roughness → Z0M/Z0WRF → ZNT → surface layer | MPTABLE `Z0MVT(VEGTYP)` | – |
| `eg` | ground emissivity → EMG → FIRA/TRAD/EMISS | MPTABLE `EG` (soil) | – |
| `albsat`, `albdry` | saturated/dry soil albedo (vis, nir) → GROUNDALB → SAV/SAG, ALBEDO | MPTABLE `ALBSAT/ALBDRY(SOILCOLOR)` | – |
| `rhol`, `rhos` | leaf/stem reflectance (vis, nir) → two-stream | MPTABLE `RHOL/RHOS(VEGTYP)` | ρ + τ ≤ 1 |
| `laim`, `saim` | monthly LAI/SAI (12 months) → phenology → ELAI/ESAI | MPTABLE `LAIM/SAIM(VEGTYP,month)` | `dveg` 1, 3 or 4 |
| `shdmax` | annual-max green fraction, FVEG = SHDMAX | wrfinput `VEGMAX/100` | `dveg = 4` |
| `vcmx25`, `mp` | Ball–Berry photosynthesis / stomatal slope → transpiration | MPTABLE `VCMX25`, `MP` | `opt_crs = 1` |
| `ctopo` (host control) | topo_wind drag factor → YSU/Shin-Hong momentum surface drag | start_em `CTOPO` | `topo_wind` 1/2, `bl_pbl_physics` 1/11, no two-way feedback onto the domain |

`ctopo` is a *host* control: CTOPO is already a runtime input of the step program, so the control replaces its
value when the run starts and adds nothing to the program — a file with only `ctopo` runs exactly the default
program of the case (same lowered program text for any value, checked on the CPU). It takes a domain-wide or
per-cell factor f of the correction, CTOPO' = 1 + f (CTOPO − 1) rounded once to REAL, so cells without a correction
keep CTOPO = 1 and f = 1 reproduces the start_em field byte for byte; negative results are refused. Not supported
in v0.4.1 (refused): two-way feedback onto a topo_wind domain, B > 1 lanes, checkpoint/resume (the restart digest
covers the traced controls only).

Each field has the shape the consumer reads ((ny, nx), (2, ny, nx) for vis/nir, (12, ny, nx) for months) and
the consumer's dtype. A control whose consumer is not active under the chosen Noah-MP options is refused, as are
non-finite values, values outside physical bounds and unknown names. The research extension below adds
soil hydraulics, canopy heights, Jarvis resistance and snow energy parameters. Land-use category and terrain
remain discrete/geometry inputs rather than surface controls.

Semantics: a field built from multipliers is `round_once(table value × multiplier)` per cell — for a per-category
multiplier this is byte-identical to running the released code on an MPTABLE whose row was edited the same way
(tested for every control). Glacier and lake cells do not run Noah-MP SFLX, so controls there are inert.

## Experimental bare-ground roughness (`z0bare`)

`z0bare` replaces ENERGY's ground-tile `Z0` in metres, including the ground below
vegetation. Original CPU-WRF uses the single constant `0.002` at every land cell;
a land-use mask is an experimental spatial extension and is **not WRF-faithful**.
It leaves the snow contribution intact: `Z0MG = z0bare*(1-FSNO) + Z0SNO*FSNO`.
Urban overrides, displacement and the existing `Z0H = Z0MG` logic are unchanged.
Values must be finite and in `[0.0001, 0.5]` m and are validated on the host before
device allocation. With no control file the default program has no new array or
kernel. ON adds one traced float32 `(ny,nx)` field (4 bytes/cell) per controlled
domain; values share a program signature, but enabling the control changes that
signature. CPU proofs do not qualify GPU fusion, cost or default-value identity.

The float64 basis is exactly `0.002`; a `{VEGTYP: factor}` map changes only those
categories before one rounding to the consumer dtype. `static.parameters.isbarren`
identifies barren land (class 16 for MODIS; check this for other land-use tables).
For the requested MODIS pilot, use 0.02 m on barren cells in d03–d09 only:

```bash
gpuwrf surface-params expand --basis basis.npz --output bare.npz \
    --class d03:z0bare:16=10 --class d04:z0bare:16=10 \
    --class d05:z0bare:16=10 --class d06:z0bare:16=10 \
    --class d07:z0bare:16=10 --class d08:z0bare:16=10 \
    --class d09:z0bare:16=10
```

`float32(float64(0.002)*10)` equals `float32(0.02)` byte for byte. Omit d01/d02
entries to retain their default programs. Explicit value fields of 0.002 express
the ON-default comparison. Checkpoint digests cover the field bytes, and B>1
members must use the same names/shapes/dtypes with values carried per lane.
Batched z0bare forecast coverage is unverified; the research pilot is serial B=1.
The older Track-P table-edit and GPU evidence above does not qualify `z0bare`.

CPU development evidence: both the no-file comparison against v0.4.6 and the
explicit 0.002 m file comparison preserve all 355 carry leaves byte for byte
after one real PROD two-domain root step. The no-file programs have identical
full HLO protos using an AST-identical, source-aligned v0.4.6 control. Four mask,
ignored-control and snow-term mutants trigger genuine assertions. A private
original-WRF `Z0 = 0.02` build matches the existing ENERGY tolerances on two real
barren daytime/nighttime columns (34 field checks; the established QSFC exclusion
is retained). These are CPU regression and scoped column checks, not GPU or
forecast qualification. In these CPU programs, enabling the native REAL field adds 4 input
bytes per cell and 8 temporary bytes per domain, with unchanged output size.

## Workflow

```bash
# 1. once per case: the float64 table basis of every control (CPU is fine)
JAX_PLATFORMS=cpu gpuwrf surface-params basis --input-dir CASE --output basis.npz
# 2. any number of parameter sets (NumPy only, seconds); entries without a prefix are d01
gpuwrf surface-params expand --basis basis.npz --output rough.npz \
    --class z0mvt:2=1.25 --class z0mvt:5=1.25 --set albdry=0.95
# 3. run (no recompile across parameter sets of one grid); without domain options run is d01 only
gpuwrf run --input-dir CASE --output-dir OUT --surface-controls rough.npz
```

Per-domain entries are optional (`d01`, `d02`, …); domains without an entry use this tree's
candidate-default path without surface-control overrides, with the CPU/cold-layer evidence limits above.
For a nested case, prefix entries with their domain (`--set d02:albdry=0.95`) and run all domains with
`--domains-from-namelist` (or `--max-dom N`): a control for a domain outside the run is refused.
`basis` refuses ALISIOS data before reading anything: the case directory (directory links followed, files named in
the namelist), and every runtime root the load reads (WRF tables via `GPUWRF_WRF_ROOT`, asset and compile caches,
paths named in the environment) must not resolve into an ALISIOS data root, and any other open of such a path
during the load is refused. Copy such inputs once under `alisios_guard` and pass the copy.
The control file carries a canonical manifest and per-domain content digests; altered files are refused.

<a id="noah-mp-soil-canopy-and-snow-parameters-research-candidate"></a>

## New soil, canopy and snow controls: not available in v0.4.4

The nine research controls `smcmax`, `dksat`, `bexp`, `psisat`, `hvt`, `hvb`,
`z0sno`, `snow_emis` and `rsmin` are refused in v0.4.4 before partial
application. The joint LSM default-value comparison failed on the actual GPU
root workload. Jarvis `rsmin` was byte-identical in its tested numerical
comparison, but the approved release policy excludes all nine names.

The existing controls in the table above, including `z0mvt`, are a separate,
older interface and remain available. Their recorded qualification does not
extend to the excluded family. A later release would require successful GPU
qualification; there is no date or commitment.

The [LSM development report](../proofs/w44_lsmparams/LSMPARAMS.md) retains
original-WRF column and earlier CPU round-off evidence. Those results are
historical scope, not permission to use the refused controls in v0.4.4.
See the [current support matrix](release/SUPPORT_MATRIX_v0.4.4.md) and
[release checks](../release_notes/RELEASE_NOTES_v0.4.4.md).

## Restart

The content digest of each domain's controls is part of the nested restart identity (and of the single-domain
checkpoint payload). Resuming with different, missing or added controls is refused.

## Batched members (B > 1)

The batched-ensemble path (`GPUWRF_BATCH_ENSEMBLE=B`, one vmapped program for B lanes of one grid) accepts one
control file per lane (`NestedPipelineConfig.batch_surface_controls`); every lane must carry the same control
names (one program signature). Lanes may share an input directory (members of one case) or use different dates.

## Ensemble Kalman inversion scaffold (synthetic tests only in v0.4.1)

`gpuwrf surface-params eki init|update` manages a small parameter ensemble in log-multiplier space
(Tikhonov-regularised EKI, host float64, tempering by `--dt`), writing one control file per member. Members are
forecasts with `--surface-controls member_NNN.controls.npz`. `eki collect --member-dir … --stations S.csv --obs
O.csv --sigma t2=1.0 --sigma wind=1.5` samples every member's wrfouts at the stations with the `gpuwrf verify`
operator (one set of extraction options, earth-relative winds by default) and keeps the station/time/variable keys
that all members and the observations share, in one fixed order → `predictions.npy` (members × obs) and `obs.npz`
(`y`, `sigma`) for `eki update`. Inputs whose real path (symlinks followed) lies under an ALISIOS data root —
observations, station metadata or member wrfouts — are read only inside ONE `alisios_guard` session that admits all
of them and stays open until the last member is read (`--alisios-issue-days/--alisios-valid-days`); without the guard
such inputs are refused before anything is opened. Purely local inputs need no guard.
Parameter spec:

```json
{"parameters": [
  {"name": "z0:forest", "control": "z0mvt", "classes": [1, 2, 3, 4, 5], "prior_sd_log": 0.4, "bounds": [0.3, 3.0]},
  {"name": "z0:shrub",  "control": "z0mvt", "classes": [6, 7, 8, 9, 10], "prior_sd_log": 0.4}],
 "tikhonov": true}
```

Status: the update and the station operator are tested on synthetic data and a twin experiment; a run against
real ALISIOS stations (guarded, ALLOW days only, observations before 2026-07-01) is planned, not done.
Limits: updates live in the span of the initial ensemble (≤ members − 1 directions), spread is not a posterior,
repeated hourly observations are correlated (temper or inflate `sigma`). Tune only what stations constrain;
hold out dates and station blocks before believing any improvement (see the design critique,
`.agent/decisions/evidence/DESIGN-SURFACE-PARAM-ESTIMATION-CRITIQUE-astra.md`).
`gpuwrf surface-params osse` runs an offline Noah-MP twin experiment (synthetic truth, CPU, minutes) that
recovers a two-group roughness multiplier with one compiled, member-vmapped land program.
`gpuwrf optimize` runs the whole loop with real forecast members (members, iterations, resume, held-out prior vs
posterior, one program for every member): see
[Continuous parameters](regional_optimization.md#continuous-parameters-experimental).

Proposed support-matrix row (v0.4.1): *Track P surface controls — experimental, opt-in; table-edit oracle
(byte-exact), one compiled program per grid (CPU + GPU), Swiss 24 h GPU forecast response matches original CPU-WRF
on the edited MPTABLE (median hourly response ratios 0.986–1.009); restart refuses mismatched controls. EKI calibration: scaffold + synthetic twin (OSSE)
only — not done against real stations in v0.4.1.*

## Optional carried ZNT history

`gpuwrf run --history-znt ...` adds `ZNT` (m) to history frames in both the
control and no-control arms. This output option defaults off and is independent
of surface controls. The API equivalent is `WRFOUT_ZNT=1` on daily/nested output.
It writes the existing carried `State.roughness_m`, without a physics re-solve:
after a Noah-MP surface step this is current `z0wrf` on land and the current
surface-layer roughness on water. Lead-zero frames contain the initialized
roughness, before the first surface step. Snow retains the normal blend.

A host-only adapter outside the traced AOT source closure appends ZNT to the
unpublished NetCDF temporary before atomic publication or checkpoint journaling.
The base writer and model program remain unchanged; OFF delegates directly to
the original writer. ON copies one float32 2-D field at each output frame and
writes that field once; it can require an additional output-boundary device-to-host
read, but adds no timestep kernels or model-state leaves. For a declared variable subset, ZNT is written only when that subset explicitly
names ZNT; this also holds when one shared asynchronous payload feeds full and
subset streams. An excluded subset makes no ZNT publication/append. The option
is currently refused for B>1 before model work; the pilot is B=1. Real GPU output
cost has not been measured.

CPU output checks preserve all 375 existing variables byte for byte on a real
Swiss history frame; ON adds ZNT as variable 376. Current land and water
provenance, asynchronous publication, journal composition and no-replace failure
handling are tested. Fresh CPU step retraces under OFF/ON give the same full HLO
proto (apart from replay module ID) and AOT key; the traced-source fingerprint
also remains unchanged. GPU output wall time has not been measured.

For a B/ZB pilot, use `--history-znt` in both arms; add `--surface-controls` only
to ZB. The two NSRT input namelists currently request 72 hours of hourly history;
pass the intended `--hours 42` override for the proposed pilot. The ordinary full
history stream contains U/V on all 44 mass levels and PH/PHB on all 45 interfaces,
at every history event. The compact training subset omits UST, so use full
history for the pilot's joint 3-D/surface comparison. Initialize-time ZNT and
conditional no-snow static maps are distinct from evolved forecast ZNT.


The history-ZNT selection is fixed for a writer/run and bound in the checkpoint's
output policy. An ordinary resume that changes it in either direction is refused
before model initialization or history recovery. An absent legacy policy value
means OFF. Full history still adds ZNT when enabled; a named subset must include
ZNT explicitly. Batched z0bare forecasting, GPU output costs and E41 qualification
remain unverified/open.

Cache scope: the output-selector comparisons above are within the candidate
source. The earlier z0bare ENERGY source change alters the traced-source
fingerprint relative to released v0.4.6, so its source-keyed AOT entries are not
reusable. The manager accepted cold layers for the private research bundle.
CPU carry/full-constant-HLO and source-aligned-proto checks do not establish
identity of the released GPU executable or paid cache. The real-GPU no-file
comparison and readiness/cost qualification remain separate gates.

FIX1 CPU transport checks exercise the actual checkpoint store, native resume
path and history-journal recovery with both policy values, refusing OFF→ON and
ON→OFF before initialization. Shared asynchronous payloads respect each write's
subset, and B>1 is refused. A real two-domain PROD CPU run writes ZNT at lead0
and after one root step (54 s, d01/d02 own steps 1/3); every value matches the
current carry, with post-step land values matching current Noah-MP roughness.
This short CPU model/IO integration is not a forecast-fidelity or GPU-cost gate.

## Grid-locked static control files (z0bare Nightly path)

`scripts/z0bare_grid_controls.py build --input-dir <case> --out <file.npz> --evidence <json>` writes the
z0bare control file for one 9-nest grid (0.02 m on IVGTYP 16 in d03–d09, float32 0.002 elsewhere, no control
in d01/d02) and embeds a grid lock (`gpuwrf.surface_params.grid_lock`): a fingerprint of the wrfinput geometry,
land classification, LANDMASK and HGT per locked domain. `gpuwrf run --surface-controls` re-derives the
fingerprint from the run's own wrfinput files and refuses on any difference, on a locked domain missing from the
run or from the file, and on an altered lock. A grid change therefore stops the run instead of silently running
OFF or applying the control to the wrong cells. `verify --controls <file> --input-dir <case>` performs the same
check offline (rc 2 on refusal). Files without a lock keep the previous behaviour. EXPERIMENTAL, not WRF-faithful.

The lock is enforced at the native runtime boundary, not only by the CLI: `execute_nested_pipeline` admits
`surface_controls` and every `batch_surface_controls` entry against the case directories it actually resolves
(B=1 input dir, `batch_input_dirs`, or `GPUWRF_BATCH_INPUT_DIRS` lanes) before any initialization, and
`_load_domains` / the B>1 lane binder admit again against the directory they load (memoized: one fingerprint per
distinct lock, case dir, run domains and wrfinput file identity). `bind_domain_controls` refuses a locked file
unless it is given the `grid_admission` receipt from `grid_lock.admit_control_source` for that file's lock.
