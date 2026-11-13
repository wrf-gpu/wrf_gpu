# WRF online diagnostics

**Status in v0.4.0: Experimental.** It runs, but has not yet passed a 24-hour comparison with the original CPU WRF.

`&physics prec_acc_dt = 60.,` enables the three WRF interval accumulations in
the native forecast pipeline. This implementation is checked against an
extracted original WRF REAL operator; it has not yet been compared with original
CPU-WRF history output in a full forecast. It currently supports Thompson
`mp_physics=8` with `cu_physics=0` or WRF-coupled KF (`cu_physics=1`). Other
pairings and the CPU-history replay pipeline refuse the option.

`&time_control nwp_diagnostics=1` enables WRF NWP maxima and mean fields;
`&physics do_radar_ref=1` enables Thompson radar reflectivity. These are
opt-in features, checked on the CPU against the original WRF code. NWP currently
supports complete specified/nested
nonperiodic domains with at least three levels and a 3×3 horizontal grid.
Radar and NWP require `mp_physics=8`, WRF-order microphysics, fixed positive
time steps and the native pipeline. Of the AFWA package only the severe-weather
`WSPD10MAX` is available (see below).
Like WRF's `check_a_mundo`, namelist binding forces `do_radar_ref=1` whenever
NWP diagnostics are enabled. Direct Python callers must set both selectors.

```fortran
&time_control
 nwp_diagnostics = 1,
 history_interval = 60, 60,
/
&physics
 prec_acc_dt = 60., 60.,
 do_radar_ref = 1,
/
```

| History field (WRF spelling) | Units | Definition |
| --- | --- | --- |
| `PREC_ACC_C` | mm | Convective precipitation in the current interval |
| `PREC_ACC_NC` | mm | Grid-scale precipitation, including all frozen phases |
| `SNOW_ACC_NC` | mm water equivalent | Grid-scale snow and ice; convective precipitation is added when the in-step T2 is below 273.15 K |

The interval is in minutes. WRF tests the **step-entry** REAL clock with
`MOD(curr_secs2, 60 * prec_acc_dt) == 0`, clears the buckets, then adds the
current step's amounts. History reads the existing buckets without resetting
them. An interval that does not divide the model time step does not trigger a
reset merely because its boundary was crossed. This reproduces WRF's exact
test. Set the interval and time step accordingly.

The producer uses the instantaneous Thompson rain/snow/graupel/ice amounts and
the actual KF rain increment. Differencing rounded lifetime precipitation
totals can lose small amounts and is deliberately avoided. Thompson excludes
the specified/nested boundary zone just as its driver does. WRF's source clamps
the two rain buckets to zero and clamps the snow bucket only in its cold-T2
branch; the port retains that operation order.

Three fp32 surface arrays live in `OperationalCarry.precipitation_buckets` and
are saved exactly in both restart formats. Missing active buckets, older
bucket schema, wrong precision or wrong surface dimensions are refused. Disabled runs have no bucket arrays. History output supports the
usual full and subset streams, rejects non-finite bucket values, and performs
no bucket recomputation. There are no bucket host reads inside the timestep.

With the option off, the default forecast program is unchanged. When it is on,
resident storage is `12 * ny * nx` bytes per domain, plus compiled temporary
storage. GPU kernel launches, peak GPU memory and whole-forecast timing have not
been measured.

## NWP and radar fields

| Field | Units | WRF sampling |
| --- | --- | --- |
| `WSPD10MAX` | m s-1 | Maximum in-step 10 m wind speed |
| `W_UP_MAX`, `W_DN_MAX` | m s-1 | Updraft maximum and negative downdraft minimum where interface pressure >400 hPa |
| `UP_HELI_MAX` | m2 s-2 | Pre-RK 2–5 km updraft helicity with WRF's smoothing and physical-edge ownership |
| `W_MEAN` | m s-1 | Mean W over eta interfaces in [0.5,0.8], accumulated and normalized at WRF's step count |
| `GRPL_MAX` | kg m-2 | Maximum column-integrated graupel |
| `HAIL_MAXK1`, `HAIL_MAX2D` | m | Maximum diagnosed hail diameter at the bottom level / in the column |
| `REFD_MAX` | dbZ | Maximum bottom-level reflectivity; its WRF initial/reset value is zero |
| `REFL_10CM` | dBZ | Thompson 10 cm radar reflectivity at the microphysics exit |

NWP resets after RK when `MOD(itimestep-1,idump)==0`, where
`idump=INT(history_interval*60./dt)`. Helicity runs before RK, including on a
reset step; its maximum is consequently cleared later on that step, as in
WRF. `UH` is restart state but is not added to the history stream.

Radar uses a full column on the history-alarm step, bottom-only on ordinary
NWP steps, and otherwise holds its previous values. History alarms detect
whether the current step reaches/crosses an output time. These alarm semantics
differ from NWP's integer step counter when dt does not divide the interval.
With radar enabled and NWP disabled, ARW still allocates `REFD_MAX` but leaves
it at zero; history preserves that behavior without another device carry.
`REFD` is a distinct AFWA/RIP radar product, and `afwa_radar_opt=1` is refused.
The specified/nested microphysics exclusion zone retains its previous radar
values. Clear reflectivity is floored at -35 dBZ inside computed columns.

The radar producer preserves raw microphysics vapor before the driver's
negative-vapor repair and its final balanced private graupel number. Reading
post-step history or the public `State.Ng` cannot recover this boundary.
Thompson's NWP hail caller instead diagnoses its own graupel-number
distribution from mass: WRF does not pass that private Ng to NWP.

Pristine `module_diag_nwp.F` computes W extrema below 400 hPa, averages W over
eta interfaces in [0.5, 0.8], and samples bottom-level reflectivity for
`REFD_MAX`. `T2MIN`/`T2MAX` belong to the separate climate diagnostics package.
`UP_HELI_MAX` is calculated by `cal_helicity` in the dynamics diagnostics.

NWP adds ten REAL surface carry fields (40 bytes per horizontal cell), and
radar adds one REAL volume (4 bytes per mass cell). The radar context adds two
temporary REAL volumes and extends the lifetime of existing MP outputs.
There are no per-step host reads. CPU operator costs, which do not describe
GPU cost, are recorded in the qualification records: 42×42 bottom radar 3.26 ms,
full-column 133.37 ms. The bottom-only specialization reduced the measured
40-column CPU cost by 25.48× while retaining exact bottom oracle values.
Both restart formats preserve every held leaf and refuse missing active
groups, wrong precision/shapes, or incompatible saved cadence.
The native CLI's synchronized checkpoints apply these same guards before
publication and restoration, binding the selections and cadences per domain.
History metadata, including descriptions and coordinates, matches the
original CPU-WRF output exactly.

## AFWA severe 10 m maximum wind (`WSPD10MAX`)

**Status: Experimental (v0.5.0 item 0.g).** Checked bit for bit against the original WRF code on real
columns, and GPU-checked on the two-domain 9/3 km Canary Islands case (12 November 2026, 3 hours, three on/off
pairs; cost below); not yet compared with a full CPU-WRF forecast. Switching it on makes a different compiled
program: other fields then differ from an AFWA-off run (in that check up to 0.87 K in 2 m temperature and
0.78 m/s in 10 m wind after 3 hours), so compare AFWA-on runs only with an AFWA-on twin.

WRF 4.7.1 has no `AFWA_GUST` field. The AFWA severe-weather package instead keeps
`WSPD10MAX`, the maximum 10 m wind speed since the last history write, and raises it
where precipitation is heavy (`phys/module_diag_afwa.F`, "surface wind gust
potential"). It is a model wind maximum, **not an observed gust**; please do not label it as a gust.

> **What `WSPD10MAX` is (agreed with ALISIOS, 2026-11-11):** the maximum of the model-time-step 10 m wind speed
> since the last history output — not a measured 3-second gust. Above 50 mm/h step precipitation it is blended
> towards the 1 km AGL wind (step 2 below); this is WRF's AFWA behaviour and is reported, not corrected.
> The history cadence defines the window of the maximum, so use hourly history output on d03 and finer nests
> (`history_interval = 60` there). A gust estimate of your own (for example `U10 + k·UST`) needs `U10`, `V10` and
> `UST` from the same frames.

```fortran
&afwa
 afwa_diag_opt   = 1, 1,
 afwa_severe_opt = 1, 1,
/
```

Both switches are per domain and both must be 1, as in WRF (the AFWA driver only runs
under `afwa_diag_opt=1`). Every own step, after microphysics:

1. `wind = sqrt(U10² + V10²)` with this step's surface-layer winds.
2. Step rate `prate = (RAINCV + RAINNCV) / dt * 3600` (mm/h). If `prate > 50`, WRF
   interpolates the wind to 1 km above ground from the first model level at or above
   1 km and blends it into the 10 m wind:
   `factor = max((150 - prate)/100, 0)`, `blend = U10*factor + U1km*(1-factor)`; the larger
   of `wind` and `|blend|` is kept. WRF divides by the level height above ground instead of
   the layer depth in this interpolation; the port reproduces that literally. Winds aloft are
   the step-start winds. Heights are those WRF's driver sees: the heights after the dynamics step and
   before microphysics inside the microphysics zone, and the step-start heights on the boundary ring,
   where WRF does not refresh them.
3. `WSPD10MAX = max(WSPD10MAX, wind)`.

The maximum is reset on step 1 and on the step that **starts** at the first step time at or
after each history time (WRF's history alarm). The value written at a history time therefore
covers the steps since the previous history write. With `dt=54 s` and hourly history the
reset falls on step 68, not on step 67 as for `nwp_diagnostics`. The reset uses the same exact
step schedule as the history writer. Set `history_interval` explicitly for every domain with AFWA on.
The other history alarm controls (`history_interval_d/h/m/s`, `history_begin*`, `history_end*`) move
WRF's alarm and are refused. A time step whose decimal value is not a short exact fraction of the
history interval is also refused. An example is `54/81 s = 0.666… s`, a fourth nest level at ratio 3.

| Field | Units | Description (Registry) |
| --- | --- | --- |
| `WSPD10MAX` | m s-1 | WIND SPD MAX 10 M |

Supported: Thompson `mp_physics=8` with `cu_physics=0` or KF `cu_physics=1`, MYNN surface layer
`sf_sfclay_physics=5` with Noah-MP `sf_surface_physics=4` (these produce `U10`/`V10` every step), `bldt=0`,
fixed time step, native pipeline with one member (B=1). Omitted per-domain `&afwa` entries are off, as in
WRF; they are not copied from the last value given. Refused: other surface/land suites, `bldt>0`,
batched runs (B>1), all other AFWA products (`afwa_ptype/vil/radar/icing/vis/cloud/therm/turb/buoy_opt`),
`afwa_diag_opt=1` without `afwa_severe_opt=1` (AFWA_MSLP etc. are not implemented),
`afwa_bad_data_check=1`, `nwp_diagnostics=1` together with AFWA (both own `WSPD10MAX` but reset at
different steps), CPU-history replay and adaptive time steps. The other AFWA severe fields
(`W_UP_MAX`, `AFWA_HAIL`, `AFWA_CAPE`, ...) are not produced.

Cost: one REAL surface plane per domain (`4*ny*nx` bytes, 125 KB on a 267×117 nest) and one GPU kernel per own
step that reads five surface planes. Only tiles with a column above 50 mm/h read the 3-D winds and geopotential up
to 1 km. While AFWA is on, the step keeps two extra values alive until the end of the step: the step-start winds
and one pre-microphysics geopotential field (`4*(nz+1)*ny*nx` bytes, 5.6 MB on the 267×117×44 nest). Measured on
that case (two warm on/off pairs, 12 November 2026): the GPU allocator's peak memory grew by 283,512 bytes
(0.27 MiB) in both pairs, well below that upper bound.
Measured on the lowered PROD programs (CPU, native CUDA lowering; independent critic rerun of this version):
each step program gets +1 Triton kernel and +2 LayoutConstraints. The HLO instruction count grows by +100 on d01
(56643 to 56743) and by +97 on d02 (49140 to 49237); these are the reset clock, the precipitation total and the
boundary-ring height merge. The nest force-down keeps its kernels and gains one parameter.
Measured whole-run cost on the 9/3 km case (3 hours, release check of 12 November 2026): warm seconds per
forecast hour with AFWA on vs off 0.998 (median of two warm pairs) and the peak-memory change above. After
optimization the domain-1 step program carries one input-transpose fusion of 664 HLO instructions (rule 500),
accepted and disclosed. Not measured: the GPU time of the AFWA kernel or of that fusion on their own, the
optimized fusion count against the off program, and the compile time.
With the option off the three PROD programs are byte-identical to the previous release tree (full-constant HLO,
all kernel bodies) and history files are byte-identical.
The plane is saved exactly in both restart formats and in native checkpoints. A restart is refused if the plane
is missing or mismatched, if the `U10`/`V10` history operands are missing or not in the surface-history carry
(`HistoryDiagnostics`), or if the history cadence differs from the saved one.
Oracle and generator: `proofs/v50_gust/`.

## WRF-Solar surface interpolation

`&physics swint_opt=1` enables the original WRF REAL surface interpolation in
the native full-carry pipeline, with RRTMG SW/LW=4, Noah-MP=4 and fixed time
steps. Solar, terrain shading, slope, urban and lake variants remain refused;
FARMS `swint_opt=2` and CPU-history replay are also refused. The runtime requires
the supplied WRF XLAT/XLONG and the existing WRF REAL solar geometry, enabled
by the default native profile (`GPUWRF_SOLAR_JULIAN_WRF=1`).

| Field | Units | Meaning |
| --- | --- | --- |
| `SWDOWN` | W m-2 | Interpolated total downward shortwave irradiance |
| `SWDDIR` | W m-2 | Interpolated horizontal direct irradiance |
| `SWDDNI` | W m-2 | Interpolated direct normal irradiance |
| `SWDDIF` | W m-2 | Total minus horizontal direct irradiance |
| `GSW` | W m-2 | Interpolated absorbed shortwave flux |

The seven parameters refresh only on actual radiation calls. Each own step
uses the current solar cosine and current albedo to interpolate the five
surface fields, including on radiation steps. The fit uses the same solve's
direct flux and WRF's REAL `GSW/(1-ALBEDO)` reconstruction. Atmospheric heating,
the global radiation cosine passed to Noah-MP and the profile `SWDNB` remain
held at their radiation cadence. WRF's negative diffuse values are retained.
History writes these carried surface fields without another radiation solve.

The carry contains twelve REAL surface arrays, or `48*ny*nx` logical bytes
when enabled and zero array leaves when disabled. Both standalone restart
formats and native synchronized checkpoints preserve all twelve leaves and
validate their schema, precision, surface shape, selections and clocks.
The nominal radiation interval and integer cadence are stored independently:
`radt=10`, `dt=18` uses a 600-second nominal interval and a 33-step cadence.

The interpolation operator is checked on the CPU against the original WRF code.
The isolated 42×42 CPU operator measured
49.77 µs on held steps and 73.88 µs on radiation steps, with 44,676 temporary
bytes. Those numbers exclude solar geometry, RRTMG, Noah-MP and whole-forecast
overhead; GPU launches, peak memory and compile time have not been measured.

Reproducible oracle/probe tools and the small operator fixture are in
`proofs/u2_diag/`. The original CPU-WRF Swiss run used a private, checksum-verified
copy of the executable with its own output directory; no reference inputs are
modified.
