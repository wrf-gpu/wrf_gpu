# Known issues — v0.5.1

**v0.5.1: the new options are off by default; their limits are listed first. The v0.5.0 items and older items below still apply.**

## v0.5.1: limits of the new options

- **`z0bare` bare-ground roughness** is experimental and not WRF-faithful by design: WRF uses one fixed ground roughness
  (0.002 m) and has no such control. It ran on the GPU with the v0.5.1 release build in the nine-nest qualification (two cases,
  72 hours); a benefit at stations is not shown. Switching it on compiles its own program, so compare
  `z0bare` runs with a twin from the same program (see the noise item below). Batched members (B > 1) with `z0bare` are
  unverified.
- **Grid-locked control files:** the lock is checked on CPU (a changed land-use cell or a 1 m terrain change is refused).
  Control files without a lock (for example from v0.5.0 research runs) are still accepted, but nothing checks their grid;
  rebuild them with `scripts/z0bare_grid_controls.py build`.
- **`--history-znt`** runs one member at a time (B=1); a checkpoint resumes only with the same `--history-znt` setting.
- **Accumulated rain between two compiles:** on the 72-hour Tenerife check, the 1 km `RAINNC` difference to CPU-WRF at
  hours 67–72 was +3.8 to +19.1 % larger than with v0.5.0, although both versions run the same lowered programs (same start
  frame, different compiled binaries). This was observed in one pair of compiles on one case and is **not** an upper bound:
  no same-build twin was run. Compare rain between builds only against a same-build twin. All six rows stay inside the frozen
  floor rule ([details](release_notes/RELEASE_NOTES_v0.5.1.md#accumulated-rain-on-the-1-km-nest-two-compiles-of-the-same-program)).
- **Nine-nest qualification:** undecided; the operated nightly stays on v0.3.3. Dew point at elevation and the winter case are not
  cleared for serving; a transfer test against station measurements is pending ([details](release_notes/RELEASE_NOTES_v0.5.1.md#nine-nest-qualification)).
- **72-hour Tenerife check:** six floor-limited rows (d03 accumulated rain at hours 67–72, as in v0.5.0), 0 real failures, and
  the known d03 graupel trace at hour 72, cause still open (see the v0.4.5 item below).

## v0.5.0: limits of the new options

- **Static-field patches (`gpuwrf patch`)** are checked on CPU against real cases; a patched forecast has not been
  compared with CPU-WRF. Categorical patches (land-use class, soil class, land/sea flip) and terrain patches change
  compile constants, so each variant gets new compile keys and one cold compile. This is confirmed on the GPU for one
  land-use patch on the 1 km nest (3 hours, finite output; domain-1 and domain-2 keys unchanged, nest keys new); for
  soil class, land/sea and terrain it is derived from the code. The `z0` control (`gpuwrf patch z0`) and soil-state or
  vegetation-fraction edits are traced inputs and reuse the compiled program; a `z0` file made by `gpuwrf patch z0` has not
  run on the GPU.
- **AFWA `WSPD10MAX`** matches the original WRF code bit for bit on real columns and passed its GPU check (warm cost
  on/off 0.998), but has not been compared with a full CPU-WRF forecast. Its program carries one larger input-transpose
  fusion (664 HLO instructions), accepted and disclosed. Switching it on makes a new compiled program: other fields then
  differ from an AFWA-off run (in the 3-hour check up to 0.87 K in 2 m temperature and 0.78 m/s in 10 m wind), so compare
  AFWA-on runs only with an AFWA-on twin. It is a model wind maximum, not an observed gust. Only Thompson (with or without KF),
  MYNN surface layer and Noah-MP, `bldt=0`, one member are supported; the other AFWA products are refused.
- **Stochastic physics in nests:** `rand_perturb=1` has no CPU-WRF reference (WRF 4.7.1 GNU dmpar crashes before
  its first history). Nests that open later than domain 1, nests with another number of levels, two-way feedback
  and batched members are refused with stochastic physics on. The stochastic-on program has one larger
  input-transpose fusion (600 optimized HLO instructions, rule 500), accepted and disclosed; its runtime share is
  unmeasured.
- **Branch series** supports one-way nesting only; the parallel launcher refuses branches; seeds cannot vary
  between branches.
- **In-run extract:** batched members and branches are refused; a 15-minute d03 cadence that differs from the
  history interval is not built; GPU wall-clock and host-memory cost were not measured separately.
- **Shared compile store (experimental, off by default):** runs from two builds are not byte-identical even when the forecast
  programs are shared, because about 4,100 small start-up and output programs recompile and the GPU autotuner
  picks differently. Compare runs of one build or measure a same-build noise band.
- **85 vertical levels** are a research setting: the Thompson column kernel needs 600 B of stack (registers spill)
  above 64 levels, and a step costs 1.7× the 45-level step at equal time step.
- Combinations of the new options (for example stochastic physics + AFWA + extract in one run) are not GPU-tested
  together; each option was GPU-checked on its own.
- **Cold compile time:** measured on the same basis (empty cache, same host, same day) v0.5.0 and v0.4.6 need the same
  cold compile time (624.42 s and 623.17 s for the WN3 three-hour proxy, ratio 1.002). A WN3 run that starts from an empty
  cache compiles about 150 s more than one that follows the Swiss and two-domain runs, because those already compiled many
  of its small start-up programs and autotuning results; compare cold times only from the same starting cache.
- **72-hour Tenerife check:** six floor-limited rows (d03 accumulated rain at hours 67–72; v0.4.6 had five), 0 real
  failures, and the known d03 graupel trace at hour 72 (next item).

## v0.4.5: final-frame d03 graupel trace remains open

On WN3 0227 at 72 h, GPU QGRAUP is zero and CPU-WRF retains a trace
(maximum about 6.58e-12 kg/kg). The sedimentation fix is active in the GPU
program. Pristine WRF on the tested reconstructed RC t72 columns also gives
zero. These are reconstructed hourly inputs, not actual pre-microphysics
savepoints. The trajectory and the process responsible between 71 and 72 h
remain OPEN; a coupled “trajectory trace, not a port defect” conclusion has
not been established. See v45-graupel report commit `340d8d8b56`.
Closing that cause is a proposed v0.4.6 item. The raw integrity failure remains
visible. The [manager decision](docs/release/evidence/v045/MANAGER_DECISION_V045_QGRAUP_20261029.md)
explicitly accepts this disclosed trace disposition for v0.4.5. The
[trajectory report](docs/release/evidence/v045/RC_TRAJECTORY_REPORT.md) retains
the evidence and open cause; no automatic waiver or tolerance change is added.

## v0.4.5: qualification limits

- Diffusion/damping overrides, including `c_s`, and the nine new LSM controls
  stay **REFUSED**. CPU implementation is not GPU qualification.
- MYNN cloud PDF modes 0/1 are CPU-qualified development options; coupled GPU
  qualification is pending. Default mode 2 is unchanged.
- RW3 now uses the full original CPU-twin floor through the regular frozen A3
  rule. Five checks are FLOOR-LIMITED, zero REAL FAIL; raw D6 remains FAIL.
  This covers d03 RAINNC at 69–72 h and W at 69 h; it is not raw error removal.
- The fresh two-pair speed/memory cohort passes: stepping N/O 0.9965,
  S=0.0181 and upper bound 1.0146; whole-run N/O 1.0072, VRAM 0.9735,
  RSS 0.9995. The spread does not resolve a speed change. Old density-rejected
  arms are not counted. O72 did not run: no same-session 72-hour old-release timing.
- The measured cold-minus-warm compile proxy is 489.9 s versus 473.420 s,
  **+3.48%**. Fresh-empty versus seeded autotune state differs.
- Compact-AOT optimized-HLO metadata can be UNKNOWN; hard-resource checks
  and available metadata are reported separately. No UNKNOWN-to-PASS conversion.

[Candidate release notes](release_notes/RELEASE_NOTES_v0.4.5.md).
Previous release records and their unresolved limits follow.

# Known issues — v0.4.3, v0.4.2, v0.4.1 (released 18 October 2026) and v0.4.0 (released 17 October 2026)

**v0.4.3 (released 21 October 2026).** v0.4.3 keeps the forecast model of v0.4.2 (all 21 compiled programs and every release-check output file are identical); its
new items are listed first. Everything from "v0.4.2: heavy-rain cells on the 1 km Tenerife domain" on
is the v0.4.2, v0.4.1 and v0.4.0 record and still applies. Per-option status:
[v0.4.3 support matrix](docs/release/SUPPORT_MATRIX_v0.4.3.md).

## v0.4.4 planned scope: parameter overrides

Diffusion/damping parameter overrides, including `c_s`, are **not shipped in
v0.4.4** after failing GPU qualification. They may come in a later release
after GPU qualification; no date or commitment.

MYNN coefficient overrides ship in v0.4.4 as **experimental**. Component and
short two-domain GPU root checks passed: default bytes identical, the same
cached programs with no new native blobs, and compiled-resource checks passed.
Three-nest D6, station comparisons and 24–72-hour drift are not qualified.
The shipped default forecast and existing static namelist support are unaffected.
See [runtime parameter family status and evidence](docs/runtime-parameters.md).

## v0.4.3: nine-domain Tenerife forecast differs from CPU-WRF on the 1 km leaves

**Open, cause not found.** On the first end-to-end nine-domain GPU forecast (issued 18 October 2026,
12 UTC, seven 1 km domains next to each other inside one 3 km domain), several 1 km domains and the
3 km domain exceed the frozen limits against CPU-WRF within 24 hours (3 km domain from hour 17, two
1 km domains from hour 10). The 1 km domain with the same grid as the three-nest Tenerife setup does
not. The study found no defect in the GPU model (no engine bug localised); the cause is
**inconclusive**. A defect limited to convection is not ruled out. The frozen-limit failures stay failures.
Accumulated rain at 72 hours differs in both directions (one 1 km domain 15 % less than CPU-WRF,
another 14 % more). A GPU check of neighbouring
nests that share one compiled program (the Canary Islands outer domain with three copies of its inner
domain, one copy disturbed, over an ordinary and a radiation step) left the parent and the undisturbed
copies byte-identical to a run without the disturbance, so a disturbance does not leak between such
neighbours. It covered nests of equal size and one parent step; nests of different sizes were checked
only on the CPU. Until this is resolved, nine-domain results are not qualified;
the three-nest qualification in the [v0.4.3 notes](release_notes/RELEASE_NOTES_v0.4.3.md) does not
cover them.

## v0.4.3: heavy-rain cells — the operators checked so far agree with WRF

The heavy-rain cell deficit on the 1 km Tenerife domain (below) is still open. New in v0.4.3: on
real convective inputs from the nine-domain case, the microphysics (Thompson), boundary layer and
surface layer (MYNN), scalar advection, sixth-order filter, horizontal diffusion and the acoustic
step were each called once with the same inputs as original WRF 4.7.1, on the CPU and with the
released GPU kernels. All agree within their frozen tolerances (Thompson: surface rain 52.0068 against
52.0055 mm on 586 columns). These are single calls with given inputs. How the model assembles those
inputs during a run, the boundary-layer state after many steps, and the coupling over a full time
step are not covered by this check and remain open.

## v0.4.3: zero output where CPU-WRF has trace amounts

**Open, cause not found; no engine bug localised.** In the three-nest qualification, four of eight
72-hour forecasts fail the file-integrity part of the check (on the 62 fields with frozen D6 limits all
three domains of these four forecasts pass): at a
few hours the GPU field is exactly zero where CPU-WRF holds a trace amount above the frozen allowance.

| Forecast | Where (valid time, UTC) | CPU-WRF |
| --- | --- | --- |
| 2 May 2026 | 9 km domain: cloud ice and ice number on 3 May 07:00 (93 and 101 cells), graupel on 4 May 11:00–12:00 and 5 May 04:00 (up to 82 cells), Noah-MP rain on 5 May 08:00 (2 cells); 1 km domain: accumulated rain on 4 May 07:00 | ice up to 1.6e-8 kg/kg, graupel up to 5.6e-6 kg/kg; rain 1.9e-6 mm |
| 20 February 2026 | 1 km domain: cloud water on 22 February 22:00 (GPU zero over the whole domain) | one cell, 8.9e-8 kg/kg |
| 18 February 2024 (ERA5) | 9 km domain: first light rain on 19 February 06:00; 1 km domain: cloud water on 19 February 16:00 (GPU zero over the whole domain) and Noah-MP rain on 21 February 10:00 (2 cells); 3 km domain: Noah-MP rain on 21 February 15:00 (1 cell) | rain 0.0059 mm; cloud water up to 9.8e-6 kg/kg; Noah-MP rain up to 1.3e-6 |
| 2 March 2024 (ERA5) | 3 km domain: canopy water and wet fraction on 4 March 14:00–15:00 (up to 6 cells), after much less light rain on the GPU in those cells in that hour (CPU-WRF 0.055 and 0.117 mm, GPU 0.001 and 0.005 mm) | canopy water up to 4.9e-4 mm |

The microphysics removes only amounts below 1e-12 kg/kg, as WRF does, and GPU history files keep
many values far below the flagged ones. The reverse case, CPU-WRF exactly zero where the GPU has a
trace, occurs about as often in the same forecasts. Whether these are differences in when and where
small amounts first appear is not established. The integrity failures stand; the limits are not changed.

## v0.4.3: coastal 10 m wind near El Médano

**No evidence of a systematic GPU-vs-CPU bias.** A user reported weaker coastal wind than measured near
El Médano. There is no evidence of a systematic GPU-vs-CPU coastal wind bias in the F20 cases: over three Tenerife cases (24 h) the mean
GPU minus CPU-WRF difference of 10 m wind speed is at most 0.007 m/s on every coastal, sea and town mask, while
single cells differ by up to 3.9 m/s. The formal 72-hour CPU-WRF noise floor and a matched v0.4.2 72-hour check
are pending. Against station
data the GPU and CPU-WRF statistics agree to 0.01 m/s, and the low bias depends on which model cell
is compared with the station (sea cell +1.0 m/s, town cell −0.8 m/s); that deficit is not a model bug. Changing the number of
domains from three to nine leaves the three shared domains byte-identical on the GPU in one check
(the W9N case of 16 March 2026, hours 0–3; later hours and other dates are not checked); original CPU-WRF changes them at round-off level. Compare only runs with the same number of
domains.

## v0.4.3: the 31 May 2025 case stopped at hour 58

**Not a model bug.** In a user's batch this 72-hour three-nest case was stopped by the batch
script's wall-time limit at hour 58 (d03), while the disk was nearly saturated by other writes. On a
quiet disk the same case ran 72 hours in 564 s with all 73 output times per domain. That rerun is a
diagnosis, not a 72-hour qualification of this case.

## v0.4.3: a host suspend stops batch scripts, not the model

When the computer hibernates during a run, the GPU is suspended; a batch script that queries the GPU
can time out and stop the case. v0.4.3 records the suspended time in the run receipt
(`host_suspend`, see the release notes). Whether a run that continues across a suspend writes
exactly the same files as an uninterrupted run is not tested; its wall times are not valid.

## v0.4.3: 10 m wind after hour 40

**Held.** In all eight qualification forecasts, the bounded 10 m and 3-D wind fields pass their D6 limits from
hour 41 to 72 on all three domains, except one value limited by the CPU-WRF noise floor (27 February, 1 km vertical
wind at hour 69). The operational user keeps wind after hour 40 held
until they qualify it; on the ERA5 cold-low case the 1 km 10 m wind differs from CPU-WRF 2.5 times as much as two
CPU-WRF runs differ from each other, still within the limit.

## v0.4.2: heavy-rain cells on the 1 km Tenerife domain

**Open, found during the v0.4.1 follow-up; affects v0.4.0, v0.4.1 and v0.4.2 equally.** On the
72-hour Tenerife forecast issued 20 April 2026 (three domains, default physics with slope
radiation), the 1 km domain has fewer cells with heavy accumulated rain than CPU-WRF on days 2
and 3. Two CPU-WRF runs (the second differs only in its MPI decomposition) and four GPU runs
(v0.4.0 and v0.4.1, two each) were compared with a rule fixed before the second CPU-WRF run:

| 1 km domain | CPU-WRF runs | GPU runs | Verdict |
| --- | --- | --- | --- |
| Domain-mean rain, 24 / 48 / 72 h | 2.14–2.15 / 3.30–3.51 / 4.32–4.51 mm | 2.04–2.11 / 3.15–3.29 / 4.12–4.24 mm | Not systematic |
| Cells ≥ 10 mm, 24 h | 467, 522 | 375–506 | Not systematic |
| Cells ≥ 10 mm, 48 h | 1144, 1090 | 915–980 | **Systematic** |
| Cells ≥ 10 mm, 72 h | 1454, 1377 | 1217–1271 | **Systematic** |

"Systematic" means the GPU mean differs from the CPU-WRF mean by more than the two spreads
together and every GPU run lies below both CPU-WRF runs. The deficit is about 12–15 % (comparing
means). Cells with at least 25 mm at 72 h: CPU-WRF 106 and 83, GPU 69–97; largest totals CPU-WRF
62.8 and 59.9 mm, GPU 40.4–58.2 mm. On the 3 km domain the totals agree (72 h domain mean CPU-WRF
0.309–0.327 mm, GPU 0.325–0.329 mm). The whole ground precipitation is liquid in this case, in
both models. One day only: the cause is open and will be investigated in the Thompson
microphysics and sedimentation path under heavy rain, confirmed on a second rain case.

## v0.4.2: the 1 mm accumulated-rain limit and CPU-WRF itself

On the same forecast, the two CPU-WRF runs differ by 1.20, 2.23, 3.31 and 4.20 mm
(root-mean-square accumulated rain, 1 km domain) at 17, 24, 48 and 72 hours, slightly more than
a v0.4.0 GPU run against CPU-WRF (1.15, 2.13, 3.16 and 4.00 mm). Two GPU runs of the same code
differ by 0.85–0.92 times as much as a GPU run differs from CPU-WRF. So original CPU-WRF already exceeds the frozen 1 mm rain limit against itself
from hour 17 on this day: there the limit measures the growth of tiny differences in convective
rain, not the faithfulness of the port. The limits stay frozen; such cases are reported with
the CPU-WRF-against-CPU-WRF difference next to them. From hour 48 every GPU run is closer to the
second CPU-WRF run than to the first.

## v0.4.2: WDM5/WDM6/WDM7 rain fall-out sub-step cap

Pre-existing since the port, unchanged in v0.4.2. The port limits rain and rain-number
fall-out to 64 sub-steps per microphysics call; WRF has no limit. The cap can only bind when
the fall speed of the largest drops times the microphysics time step exceeds about 63.5 lowest
layer depths (for a 120 s step: layers thinner than about 22 m; 180 s: 34 m) in heavy rain. In
such a column the port moves rain down more slowly than WRF. On the Swiss Alps WDM6 case WRF
needs at most 5 sub-steps (6 in the worst case on that grid), so the cap never binds there. Since
v0.4.2 the loop stops at WRF's own count, so removing the cap is cheap; it changes results only
in the affected columns and is planned for a later release.

## v0.4.2: two CPU tests that also fail on v0.4.1

`tests/test_v040_urban_slucm_wiring.py::test_catalog_and_validation_accept_slucm` checks for a
phrase in the catalogue note of the BEP/BEM urban options that the note no longer contains, and
`tests/a1_da/test_member_driver.py::test_process_aot_reuse_shares_only_blob_loaded_executables`
inspects an internal function closure that changed. Both fail identically on the v0.4.1 release
and on v0.4.2; the model code they cover is unchanged. All other CPU tests of the release
candidate pass or are skipped with a stated reason.

## v0.4.2 opt-in items: limits

- **`icloud=0`**: one 24-hour comparison with CPU-WRF (Tenerife three domains, 27 February 2026).
  Needs RRTMG longwave and shortwave; same cumulus and microphysics restrictions as `icloud=2/3`.
  Run both members of a cloud/no-cloud pair from one compile cache or bundle.
- **`GPUWRF_SWDDIR_V040_SPLIT=1`**: reproduces v0.4.0, not WRF. Refused with `aer_opt=1/3`. Its
  compiled shortwave kernel is checked byte-identical to v0.4.0 in machine code, but no GPU
  forecast was run with the switch for this release; it compiles its own programs.
- **Boundary layers inside nests** (YSU, Shin-Hong, ACM2, EEPS): one forecast date, one domain
  set-up, revised MM5 surface layer only; the compiled-code quality scan of these nested programs
  was not run. KEPS and the other boundary layers stay validated on single domains only.
- **WDM5 and WDM7 speed** after the shared WDM6 change was not measured.
- **72-hour release check, trace graupel:** on the 1 km Tenerife domain (27 February 2026) v0.4.2 has no graupel at hour 72, where CPU-WRF keeps 3 cells with at most 6.6e-12 kg/kg (none above 1e-9). A second CPU-WRF run trips the same file check at the same hour, and up to hour 70 v0.4.2 carries comparable trace graupel (154 against 217 cells); recorded as an exception, not a defect.
- **Compiled bundles:** settings not in a compiled bundle (for example `GPUWRF_SWDDIR_V040_SPLIT=1`, other physics or other domains) compile once on first use. The code-quality scan of the bundle programs is incomplete, as in v0.4.1 (compact programs carry no fusion sizes); per-thread stack and coverage pass.

**v0.4.1 (18 October 2026).** All v0.4.1 additions are opt-in. The new items found or
added in v0.4.1 are listed first; everything from "Current frozen default" on is the v0.4.0
record and still applies. Per-option status: [v0.4.1 support matrix](docs/release/SUPPORT_MATRIX_v0.4.1.md).

## v0.4.1 direct-beam surface shortwave (SWDDIR)

**Fixed in v0.4.1; affects v0.4.0 and every earlier release** (the **default** physics, since the
RRTMG shortwave port). Found during the v0.4.1 aerosol work.
Up to v0.4.0 the RRTMG shortwave path wrote the direct-beam surface flux `SWDDIR` (and `SWDDIF`,
derived from it) from the delta-scaled cloud optical depth. WRF writes the unscaled product
(`ztdbt_nodel`, `phys/module_ra_rrtmg_sw.F` 8577–8600 and 8737). A pristine `RRTMG_SWRAD` oracle on
186 real Tenerife columns (27 February 2026) gives v0.4.0 `SWDDIR` differences of 31–129 W/m² on
cloudy daytime columns and 0.27 W/m² on clear columns. The horizontal total surface shortwave
(`SWDOWN`/`GSW`) and the atmospheric heating rates are not affected. **Forecast impact:** with
`slope_rad=1` or `topo_shading=1` the direct/diffuse split sets the slope-adjusted surface
shortwave, so on cloudy sloping terrain the surface energy budget, and through it the forecast,
can differ from WRF; `swint_opt` also uses the split. An estimate on 81 real cloudy Tenerife
columns with a slope sweep gives errors of the slope-adjusted shortwave up to 31–102 W/m² on
10–30° slopes, depending on season and slope direction [estimate, not a paired run]. The Canary
Islands and Tenerife comparisons of v0.4.0, which run with slope radiation, passed their field
limits with this difference present.

**v0.4.1:** `SWDDIR` agrees with the pristine oracle within 0.32 W/m² on 293 real Tenerife and
Swiss columns. The fix changes the default program; together with the boundary-record change it
passed 24-hour comparisons with the original CPU-WRF on the Swiss, Canary Islands and Tenerife
cases. Forecasts with slope radiation can therefore change against v0.4.0. The clear-sky
`SWDDIRC`/`SWDDNIC` outputs are not written.

**v0.4.2:** the full WRF code path from RRTMG to the slope-adjusted surface flux was traced again
and matches; a 72-hour Tenerife case where the fixed run looked worse on cloudy slopes was traced
to where the clouds formed in that run, and a repeat run did not show it (details in the
[v0.4.2 release notes](release_notes/RELEASE_NOTES_v0.4.2.md#the-shortwave-fix-of-v041-checked-again)).
The fix stays the default; `GPUWRF_SWDDIR_V040_SPLIT=1` restores the v0.4.0 split for
continuity comparisons.

## v0.3.0–v0.3.4: nested programs from a compile cache shared across dates

In v0.3.0 to v0.3.4 the fused program of an inner domain and its nests contained the Noah-MP
deep-soil temperature `TMN` as a compiled constant, but its cache key ignored it. A compile cache
shared between forecast dates could therefore load another date's nested program without notice
(`GPUWRF_AOT_VERIFY` is off by default). v0.4.0 keys each date separately (it recompiles instead);
v0.4.1 passes `TMN` at run time, so the question does not arise.

Measured on v0.3.3: `TMN` is the only consumed date-dependent constant; it differs between dates
only over water (`real.exe` sets it to the day's SST), where it feeds a placeholder soil column that
is masked out of every atmosphere-facing output. Land `TMN` was identical in all 90 checked pairs of
ten Tenerife three-domain forecasts and all 48 pairs of four nine-domain Canary forecasts. On the
Tenerife three-domain grid, a 24 h forecast issued on 7 July 2026 that loaded the 27 February
program wrote all 72 history files byte-identical to a fresh compile. The nine-domain result is
inferred from the identical land values and code path, not run end to end. Restart checkpoints from
such a run carry the other date's water soil temperatures (never in wrfout).

For a grid whose land `TMN` changes with the date, or to be safe with any v0.3.x shared cache, set
`GPUWRF_AOT_VERIFY=1` (a mismatching nested program is quarantined and recompiled, about 6 minutes on
v0.3.3), or upgrade. [Impact record](proofs/w41_v033impact/IMPACT.json).

## v0.4.1 opt-in features: limits

- **Data assimilation (`gpuwrf da`)**: outer domain (d01) only; dry column mass not analysed; no
  cloud or radiance assimilation; `supersaturation: keep` only. Forecast skill is not established.
  The optional ensemble settings are overdispersive in their registered twin experiment (prior
  spread 2.59/2.32/2.53 × the actual error at 06/09/12 UTC; required 0.5–2.0); with default
  settings the spread decays by 28–76 % per 6 h. The GPU cycle loop has run zero-update arms only;
  the analysis step runs on the CPU. [Guide](docs/da.md), [ensemble options](docs/da_ensemble.md).
- **`topo_wind`**: acts only with YSU or Shin-Hong, as in WRF. `topo_wind=1` needs `VAR_SSO` in
  `wrfinput` (WPS `varsso` data). `topo_wind=2` is validated on one 9 km domain only; two-way
  feedback of the correction fields is CPU-tested only. The `topo_wind=1` YSU program exceeds the
  per-thread stack limit (400–496 B); runtime cost not measured.
- **`aer_opt`**: RRTMG shortwave only; `aer_opt=3` needs `mp_physics=28` on a single domain;
  `aer_opt=2` is refused. `aer_opt=3` inherits three oversized fusions of the Thompson mp28 code;
  the `aer_opt=1` code-quality scan is incomplete (measured on a re-compile).
- **`icloud=2`**: one 24-hour comparison with CPU-WRF (Tenerife three domains); other
  configurations are not compared. `icloud=0` was not bound in v0.4.1 (it is in v0.4.2).
- **Surface-parameter controls (`--surface-controls`)**: Noah-MP only; no calibration against real
  stations yet (the EKI tools are checked on a synthetic twin only); batched members are API-only
  and a batched lane is not bitwise equal to its solo run.
- **`gpuwrf optimize`**: a held-out diagnostic, not a skill result; host memory (6–6.5 GB per
  member that compiles) is the first limit.
- **Stochastic physics**: validated on one single-domain case; nested domains, GPU restart and
  ensemble spread/skill are not checked. WRF's default length scales give NaN patterns on small
  domains ([upstream issues](docs/UPSTREAM_WRF_BUGS.md)).
- **FLG radiation (`ra_lw/ra_sw_physics=7`, experimental, CPU-only)**: the 17 strict original-WRF
  band-bound tests are xfailed (strict), pre-existing since v0.4.0; each cites the per-band disposition
  in [FG93](proofs/s2_flggfdl/receipts/FG93_flg_compiler_disposition.json) (max band error
  0.0014 W/m²; 10 bands outside the native-compiler envelope, not independently source-bounded). They
  are known limitations, not passes.
- **WDM5/WDM7 (`mp_physics=14/26`)**: the v0.4.1 compiler barrier leaves residual stack use above
  the limit (536/400 B and 368 B); WDM7's hail mixing ratio is not covered by the in-step negative
  value census.

## v0.4.0 record

**Default path frozen — 2026-10-16 11:59:18Z.** Main `5c15881ede`
(`403240f257` adds proof only) carries the caller-audit repairs and device-only
C5′ MYNN default. Release identities are Swiss **3356e249** and PROD d01
**196aed9f**. F20 plus C5′ original-CPU 24h default gates are closed; the
FINAL composite checkpoint is **66/66 accepted-scope scientific D6 PASS (65 full24h + c45 accepted6h)** at 00:26Z on 17 October.
Source code and tags are published. Large microphysics lookup tables, including the Thompson tables of the default
physics, are downloaded automatically on first use from the
[v0.4.0 GitHub release](https://github.com/wrf-gpu/wrf_gpu/releases/tag/v0.4.0) and checked against pinned SHA-256 sums.
See [runtime tables](docs/runtime-tables.md).
Historical f672a573/6fc715a2 receipts remain pre-F20 baseline evidence.

## Current frozen default and composite checkpoint

Swiss **3356e249** / PROD d01 **196aed9f** are frozen on5c15881ede; full
hashes and source bindings are in the [support matrix](docs/release/SUPPORT_MATRIX_v0.4.0.md#frozen-defaults-and-results-so-far).
F20 covers five24h default cases. C5′ reruns Swiss/PROD/WN3-0227 with150
paired domain-frames, bounded D6/inventory/NF0 and actual E41 PASS3/3.
Host transfers/synchronization/callback counts equal F20; matched nsys −1.4%
is not S1 or ABAB. [Default closure](proofs/s2_integrate/batch62/gate_WN3_closed/GATE_CLOSURE.json).

[66/66 accepted-scope scientific D6 PASS](docs/release/FINAL_COMPOSITES_v0.4.0.md)
are recorded at00:26Z on17October with their measured source. All66 scientific scopes pass:65 full24h plus c45 accepted6h. The [class-A/default fix list](docs/release/FIXES_v0.4.0.md)
records batches38–62. The raw original failures retained below are historical.

## c11 resolved compiler stall and residual resources

The former mp16 WDM6/cu96 SAS-family/bl17 KEPS compile on5c15881ede reached
its60-minute cap without a stepping executable; that INCOMPLETE/no-model-result
receipt remains historical. The BYTE fusion repair merged in batch65; actual
changed-source **b823c8ca87 full24h now passes25frames, complete inventory,
zero bounded misses and NF0**. The measured process wall is900.005s and sampled
process VRAM5386MiB [M], with once/second sampling that can miss brief peaks.
This is not S1 or a new whole-run speedup claim.

Residual actual E41 remains **FAIL**: maximum STACK512B and kInput1353
optimized HLO instructions. Exact sites and SHA bindings are retained in the
[E41 inventory](proofs/r_docs040/E41_OPTIN_DISCLOSURE.json). The old failed
compile's host RSS/cgroup measurements are not a current minimum-RAM claim;
the successful run used a48GiB host-memory cap, not a measured requirement.
[Historical triage and final qualification](proofs/r_docs040/RELEASE_CHECKPOINT.json).

## v0.4.0 draft qualification limits

These limits apply to the v0.4.0 release and the exact source-specific
receipts below.
The [support matrix](docs/release/SUPPORT_MATRIX_v0.4.0.md),
[historical frozen snapshot](proofs/r_docs040/EVIDENCE_SNAPSHOT.json) and
[post-merge reconciliation](proofs/r_docs040/RECONCILIATION.json) distinguish
bounded D6 results, field coverage, E41 resources and items that are not yet validated.
The v0.3.4 statements retained below are dated historical records.

### E41 and incomplete forecast coverage

The manager's 2026-10-16 04:36Z decision keeps E41 release-blocking for the
default WN3, PROD and Swiss F20 programs. Opt-in composites require strict
original-CPU-WRF D6, complete inventory and zero nonfinite output values (NF0).
An opt-in resource failure is disclosed as **E41 FAIL (perf/robustness), D6 PASS**
only when those scientific gates pass. Raw E41 verdicts remain FAIL; incomplete
or failed D6 is never promoted. The frozen C5′ successor has
closed the default E41 gate; this opt-in disposition did not waive it. STACK, non-leaf callee and kInput thresholds
remain unchanged. Exact executable properties and unknown runtime impact remain in
the [complete provisional E41 ledger](docs/release/E41_LEDGER_v0.4.0.md);
source attribution for older named executables remains in the
[supplement](docs/release/E41_OPT_IN_v0.4.md). This is not a waiver for
release defaults or an isolated spill-cost measurement.

Goddard radiation 5/5 has an accepted six-hour scope: seven frames, complete
346-field inventory, zero bounded misses/nonfinite output and E41 PASS [M].
The 24-hour attempt has only h0–23; h24 was cut by the run cap. The tile-2048
repair supersedes the old slowdown report. It is not a full 24-hour PASS.
[Accepted receipt and source](proofs/r_docs040/EVIDENCE_SNAPSHOT.json).

### Source-bound opt-in E41 disclosure (00:26Z final)

This table is generated from v-prep's final `FINAL_E41_DISCLOSURES.json`
at `2026-10-17T00:30:45.643194+00:00`: **32 FINAL resource FAILs**, with
33 PASS and c89 UNKNOWN counted separately. The [portable exact-content
projection](proofs/r_docs040/FINAL_E41_DISCLOSURES.json) retains every actual STACK site, kInput fusion,
non-leaf/recompute callee, ELF/JIT/AOT origin and receipt SHA. All rows below
are measured registered FINAL sources (F); the three historical-only failures
in the earlier provisional inventory do not enter the final counts. Raw
E41 FAIL stays FAIL despite scientific D6 PASS. Missing scan data is UNKNOWN,
never zero. STACK is bytes; kInput is optimized HLO; callees are SASS. Callee
columns give site count and independent maxima; exact sites remain in JSON.
Options are MP/CU/PBL/SFCLAY/LSM/LW/SW.

| Case / source scope | Options | Raw D6 / scientific gate | Max STACK B | kInput >500 (HLO) | Non-leaf/recompute sites (max SASS / max nested calls) |
| --- | --- | --- | --- | --- | --- |
| c02 / `04cb09f816` F | 1/1/1/1/4/1/1 | PASS / PASS [E41 disclosed] | 16 | 549, 570, 586, 632, 659, 661 | — |
| c03 / `04cb09f816` F | 2/2/2/2/4/4/2 | PASS / PASS [E41 disclosed] | 2456 | 540, 598 | — |
| c05 / `04cb09f816` F | 4/5/7/1/4/4/4 | PASS / PASS [E41 disclosed] | 856 | 785, 954 | 1 site; 14418 / 72 |
| c06 / `db78f9bd8c` F | 6/6/8/1/4/4/4 | PASS / PASS [E41 disclosed] | 40 | 536, 598, 722 | — |
| c07 / `04cb09f816` F | 9/16/11/1/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 538, 615, 622, 643, 679 | — |
| c08 / `04cb09f816` F | 10/14/12/1/4/4/4 | PASS / PASS [E41 disclosed] | 408 | 714, 904, 973, 1320, 1600 | — |
| c09 / `04cb09f816` F | 13/94/99/1/4/4/4 | PASS / PASS [E41 disclosed] | 424 | 626, 1383 | — |
| c10 / `04cb09f816` F | 14/95/16/1/4/4/4 | PASS / PASS [E41 disclosed] | 536 | 1463 | — |
| c11 / `b823c8ca87` F | 16/96/17/1/4/4/4 | PASS / PASS [E41 disclosed] | 512 | 1353 | — |
| c12 / `04cb09f816` F | 17/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 1336 | 562, 660, 724, 835, 1041, 1064, 1128, 1172 | — |
| c13 / `04cb09f816` F | 18/5/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 681, 973, 1405, 2096 | — |
| c14 / `04cb09f816` F | 19/99/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 900, 1034, 1096 | — |
| c15 / `04cb09f816` F | 21/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 208 | 660, 1235, 1287 | — |
| c16 / `04cb09f816` F | 22/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 640, 641, 649, 856, 944, 1060 | — |
| c18 / `5c15881ede` F | 26/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 904 | 618, 1211 | — |
| c19 / `5c15881ede` F | 28/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 741, 1439 | — |
| c20 / `5c15881ede` F | 40/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 820, 832, 2227 | — |
| c26 / `2ed8c04c26` F | 8/0/4/4/4/4/4 | PASS / PASS [E41 disclosed] | 7232 | — | — |
| c27 / `2ed8c04c26` F | 8/0/10/10/4/4/4 | PASS / PASS [E41 disclosed] | 32 | 529, 531, 543 | — |
| c40 / `36fcfaa64d` F | 8/0/1/1/8/4/1 | PASS / PASS [E41 disclosed] | 384 | 1369 | — |
| c42 / `5c15881ede` F | 8/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 572 | — |
| c46 / `5c15881ede` F | 8/1/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 8 | 502, 569 | — |
| c55 / `5c15881ede` F | 8/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 80 | 503, 645, 820, 956 | — |
| c56 / `04cb09f816` F | 8/0/1/1/4/4/4 | PASS / PASS [E41 disclosed] | 968 | 516, 544, 566, 586, 687, 759, 937, 1015, 1179, 2477 | — |
| c57 / `2ed8c04c26` F | 8/0/8/1/4/4/4 | PASS / PASS [E41 disclosed] | 968 | 509, 578, 643, 708, 712, 759, 797, 830, 992, 1015, 1253, 1707, 2566 | 15 sites; 9612 / 2 |
| c61 / `04cb09f816` F | 97/0/0/91/4/4/4 | PASS / PASS [E41 disclosed] | 328 | — | — |
| c65 / `5c15881ede` F | 38/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 496 | 638, 644, 1089, 1134, 1314 | — |
| c83 / `5c15881ede` F | 8/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 887 | — |
| c85 / `04cb09f816` F | 8/0/1/1/4/4/4 | PASS / PASS [E41 disclosed] | 384 | — | — |
| c87 / `04cb09f816` F | 55/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 5520 | — | — |
| c88 / `5c15881ede` F | 56/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 3624 | 894 | — |
| c90 / `5c15881ede` F | 8/1/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 8 | 502, 569 | — |

For c03, STACK2456B is pre-existing since the f4fd predecessor; final c03 also
has kInput540/598 HLO. For c05, STACK856B, kInput954/785 HLO and a14,418-SASS
non-leaf callee with72 nested calls are a recomputation signature. This does
not measure its runtime share; c05 runtime attribution is a v0.4.1 follow-up.
Earlier c05 raw D6/resource failures remain preserved.

The producer separately lists c89 as resource UNKNOWN despite D6/inventory/NF0 PASS under
its attempt-selection scope. Missing flags are UNKNOWN, never zero; historical
reuse/successor receipts in the older ledger retain their exact-source validity.
Neither list approves an unmeasured final executable. CAM-MG's separate offline
native STACK1240B scan must not be replaced by an XLA glue-only PASS.

The JSON retains the historical five-case F20 packet at sourcee7e37390cc:
bounded D6 PASS but raw E41 FAIL. That old failure remains failed. The frozen
C5′ successor now closes current default D6/E413/3 and transfer gates; its
identity/default qualification is in the release checkpoint. The old
`default_F20` field is not a current default blocker. F2 remains separate and
these profiling/component results make no S1 claim.

### GPU refusals and pending affected runs

CAM-MG `mp_physics=11` is GPU-admitted through the merged stock-Triton 3.6.x
route (batch45), with explicit compiler/device preflight. Its original c89
GPU attempt failed frozen RAINNC D6, terminated with only15/25 hourly frames
and lacked DLF2. A later source9a40424b0 run has25frames, zero bounded misses
and NF0, but raw D6 remains INCOMPLETE because DLF2 was missing. The merged
DLF2 writer preserves WRF's initialized +0 while shallow convection is refused;
registered-source c89 now passes full24h D6/inventory/NF0; native E41
remains UNKNOWN while the separate offline STACK1240B scan is FAIL.
[Current receipt](proofs/r_docs040/RELEASE_CHECKPOINT.json).
[CAM-MG detail](docs/CAMMGMP.md), [raw receipt/decision](proofs/r_docs040/EVIDENCE_SNAPSHOT.json).
The localized defect is in jaxlib/XLA's bundled Triton pipeline: a minimal
reproducer lost live input loads and wrote zeros, while stock Triton retained
them and passed the small actual-GPU activity gate. This does not by itself
qualify the full native CAM-MG forecast. The retained localization packet is
[proofs/t_rainnc/LOCALIZATION.json](proofs/t_rainnc/LOCALIZATION.json).


`km_opt=3` is GPU-refused by the integrated pre-JAX/pre-device guard.
The Swiss GPU forecast became nonfinite in U at
step200 (one hour); attempted original-CPU Swiss/Madrid references crashed.
Both classes of failure remain preserved; column evidence is not a successful
coupled reference. [Original failed pair](proofs/v040/gpuqual/km3/D6.json),
[decision](proofs/r_docs040/EVIDENCE_SNAPSHOT.json).

The c05 WSM5/ACM2/Grell composite on de374a3c0 retains RAINNC RMSE1.108mm,
above the frozen1mm bound at h16–24 [M]. The paid ACM2 condensate-caller successor5131ddffb now has a full24h
original-CPU bounded D6/coverage PASS over25frames, with E41FAIL and F2OPEN.
The earlier failed trajectory remains failed. The merged final-composition
source04cb09f816 c05 also passes strict24h D6, full inventory and NF0, while
E41 remains FAIL. [Composed-source receipt](proofs/r_docs040/RECONCILIATION.json),
[earlier successor receipt](proofs/r_docs040/EVIDENCE_SNAPSHOT.json).
Jensen c87 and final-source NTU c88 now pass full24h D6/inventory/NF0 with
E41FAIL. Older QHAIL-incomplete output remains historical. The merged writers do not relabel old receipts.
[Current registered-source table](docs/release/FINAL_COMPOSITES_v0.4.0.md).

Surface mosaic (`sf_surface_mosaic=1`) is refused: Noah-classic needs missing
per-tile binding/carry/restart/aggregation infrastructure, while original WRF
ignores this switch for Noah-MP. Valid crop/irrigation/groundwater maps do not
qualify mosaic. [Admission rule](src/gpuwrf/io/noahmp_options.py).

Where advanced single-domain-only options remain admitted in nests, report
**nested not validated**. Actual nested/pairing refusals still apply. Neither
an original single-domain D6 result nor a finite nest smoke provides arbitrary
nested fidelity. [Scope](docs/release/SUPPORT_MATRIX_v0.4.0.md).

### Intermediate negative moisture and number values

Original WRF reproduces the magnitude and predominantly interior location
classes of large post-RK negatives in matched Swiss default/WDM6 runs. These
are not purely roundoff. This accepted narrow known issue does not justify
every GPU amplitude or repair. Residual GPU/original extremum ratios are
1.140410 for default QI, 1.394381 for default QG, 1.004246 for WDM QS and
1.164901 for WDM Nr [M]. WDM snow/graupel visit-count differences also remain
OPEN; counts are repeated phase-field visits, not unique events. No extra
clip or new acceptance band was introduced.
[Original-WRF comparison and disposition](proofs/t_f2neg/REPORT.md).

The final hail quartet retains negative-event justification as OPEN. Its
observed nonfinite counters are zero, but qh is absent from the counted field
inventory (Nh is present); this is not full hail pre-mask clearance.
[Final-source quartet](proofs/v040/gpuqual_qhmass/QUALIFICATION.json).
NTU's separate case-scoped source-diffusion exception is described below;
neither exception is global F2 clearance.

### Original MYNN undefined initialization and P3 precision

Pristine WRF 4.7.1 passes local `rmol` into MYNN cold initialization before its
first assignment. A Valgrind/native-PC source witness locates the mixing-length
read. The port already uses zero. Some original MYNN/P3 references therefore
depend materially on an undefined value. Designated comparisons use unchanged
original bodies compiled under the documented ZERO convention; raw-original
results and initialization spread remain preserved. Blanket compiler
initialization can affect other undefined sites too. Frozen D6 bounds do not
change. [Upstream source proof](proofs/t_p3fix/UPSTREAM_WRF_BUGS.md),
[original/ZERO reference disposition](proofs/t_p3fix/DISPOSITION.md).

The retained P3 native column replay has 240 strict deviations; eight values
remain outside the independently reproduced original 44-level FMA envelope
(six NR, one QC, one BIR) [M]. They are accepted as a disclosed small deviation,
not relabelled as a strict column pass. MP52's compiler repair produced an original-CPU 24-hour bounded D6 PASS and
actual E41 PASS at its measured source. Newer P3 50–53 receipts also close
full history inventory; the earlier missing-field reports remain historical.
Final composed-source approval is **TBD-VALIDATION**. [P3 numerical disposition](proofs/t_p3fix/DISPOSITION.md),
[MP52 successor](proofs/t_p3compile/QUALIFICATION.json),
[later radar retention](proofs/t_writer_p3_radar/QUALIFICATION.json).

### WDM5 and Jensen compiled residuals

WDM5 Nn/Nc miss the canonical two-percent number band in the theta-handoff
columns; original REAL4 versus REAL8 itself differs by 9.8852928e7 #/kg [M].
The retained fp64 island is precision-sensitive. This is not a number-gate
PASS or a new tolerance. [Two-call original-kind evidence](proofs/t_thetafix/README.md).

Jensen's existing compiled CPU column implementation has 335 values outside
its independent enclosure across 17 finite columns [M]; those remain small
experimental deviations, not strict enclosure passes.
[Original column qualification](proofs/s2_mpmore/jensen55/STATUS55.json).
Its expanded ice diagnostics preserve previous operational outputs,
but 165 values across 17 original columns exceed the unchanged input-ULP/FMA
enclosure, with maximum relative excess 1.43e-5 [M]. The small diagnostic
deviation is explicitly disclosed; full coupled GPU qualification remains
**TBD-VALIDATION**. [Definitions and independent disposition](docs/physics/ice_diagnostics.md).

### Exclusions and other open work

- MP15 and `km_opt=3` are column-qualified experimental paths, excluded from
  coupled coverage because original CPU-WRF crashes on the attempted cases.
  They are not successful reference forecasts. Crop/tile/map-selected
  irrigation branches without geographically valid NCAR maps lack 3-D
  validation. [Reference/exclusion receipts](proofs/r_docs040/EVIDENCE_SNAPSHOT.json).
- Class-D producers and WRF-constant fields are merged (batch40/41), with
  admitted initialized/held branches and active refusals listed below. CPU
  producer proofs do not qualify final GPU trajectories or repair old missing
  inventories. [Composition](proofs/s2_integrate/batch40/QUALIFICATION.json),
  [independent caller review](proofs/rv_t_classd/composition_review/VERDICT_COMPOSITION.md).
- The earlier Shin-Hong diagnostic deviation has a later caller/kernel
  correction with source44b7b226b c07 full24h D6/coverage PASS and E41FAIL;
  final-source identity and the disclosed native quantum residual remain separate.
  History `TKE_PBL` was written as 0 for Shin-Hong, MYJ, QNSE, BouLac, CAM-UW
  and GBM (original WRF writes each scheme's shared TKE array; unbounded
  REPORT_ONLY field, so D6 never flagged it), and Shin-Hong `EL_PBL` stayed 0
  for `shinhong_tke_diag=1`. v0.4.1 writes the held arrays (host-only) and
  seeds MYJ/BouLac/CAM-UW with their original cold-start TKE instead of the
  MYNN profile (initial data change for MYJ/BouLac). `EL_PBL` of MYJ, BouLac
  and GBM is still not written (needs a step-program change). GPU 24 h re-runs
  (MYJ, BouLac, Shin-Hong `shinhong_tke_diag=1`) and a 6 h CAM-UW run pass D6
  against original CPU-WRF; lead-0 `TKE_PBL` is byte-identical.
  [Measurement and fix](proofs/w41_shinhongtke/README.md).
  GBM mixing diagnostics retain small strict residuals. Their investigation
  is separate from finite output or bounded forecast comparisons.
  [CAM-MG/PBL caller audit](proofs/s2_cammg/README.md),
  [GBM repair qualification](proofs/t_c02nan/NAN_REPAIRS_QUALIFICATION.json).
- MP28's startup emission-height correction follows the measured mismatch
  between WRF interface height and the port's mass-averaged height. Its
  CPU producer evidence does not establish a new full GPU trajectory PASS.
  [Measured mismatch](proofs/v040/gpuqual_mp28_followup/summary.md),
  [subsequent fix](proofs/t_writer_mp28_emission/QUALIFICATION.json).
- Final composed-candidate gates, station campaigns, full CPU-suite closure
  and real multi-GPU scaling remain **TBD-VALIDATION**. Heavy tests have a
  measured separate CI tier; the test/source/attempt list is bound as
  `heavy_tests` in the [snapshot](proofs/r_docs040/EVIDENCE_SNAPSHOT.json).
  Historical v0.3.3 coastal-wind/rain/cirrus deviations below remain disclosed.

## v0.3.4 historical release status

**v0.3.4 (fix release).** wrf_gpu ≤ 0.3.3 silently ran the default physics suite when `namelist.input` selected another microphysics, PBL, surface-layer or radiation scheme or other damping/filter values. Re-run such forecasts; the binding fix and full key list are described below. Experimental opt-in schemes added in v0.3.4 are matched to unmodified-WRF oracles on CPU and most were GPU-smoked (three finite steps); none is a validated forecast configuration. CAM radiation (`ra_lw/ra_sw_physics=3`) still fails the static kernel stack check (2,144 B spill; fix planned for 0.4.0). Per-scheme gaps: [support matrix](docs/release/SUPPORT_MATRIX_v0.3.4.md). The default program is unchanged, so the v0.3.3 deviations below still apply.

Open v0.3.3 follow-ups (not addressed in v0.3.4): coastal 10 m wind-speed excess; nine x-major loop-layout invariants on unpinned paths (E41 hardening); the amplitude of WRF-consistent negative-moisture visits versus original WRF is unproven; the ≤4,096-column stability firewall cost should become a layout-level fix.

## CAM-MG GPU precipitation

`mp_physics=11` (CAM-MG) runs on GPU again. Earlier v0.4.0 builds refused it
because XLA's bundled Triton (both the in-graph route and the PJRT plugin
compiler) miscompiled the native column and returned zero `RAINNCV`/`SNOWNCV`.
The column is now compiled to PTX by **stock Triton 3.6** and launched through
jaxlib's `triton_kernel_call`; other kernels keep XLA's route. GPU mp11 runs
therefore need `pip install "triton==3.6.*"`; other Triton versions, a lowering
without a visible CUDA device, or a missing Triton install are refused with an
explicit error rather than silently compiled. The native kernel uses 255
registers and a 1240-byte stack frame (E41 STACK flag, disclosed for this
opt-in scheme); the first mp11 call of each fresh process assembles the 5.2 MB
PTX in-process (about 26 s per grid shape; not cached across processes).
Coupled 24 h GPU qualification against CPU-WRF is pending.

<a id="v040-history-fields-without-a-producer"></a>
## v0.4.0 history producers and WRF-constant fields

The nine CPU-WRF history fields previously omitted here are now written. Every
admitted configuration writes the value WRF computes; no field is "not computed"
in an admitted branch. Only the WRF branches named in the last column remain
unported and are **refused at load**, never run silently. Original CPU-WRF proofs
cover every cell of all 25 frames ([proofs/t_classd](proofs/t_classd/)).

| Fields | Configuration | Value written (WRF source) | Refused branch (fields it would change) |
| --- | --- | --- | --- |
| `QGR_URB4D`, `TGR_URB4D`, `DRAINGR_URB3D` | BEP+BEM `sf_urban_physics=3`; c57 | Without green roofs: held `QGR_URB3D` = Noah-MP top-layer `SMOIS` (`urban_var_init`), reset to 0 on BEP/BEM columns every call; `TGR`/`DRAINGR` +0. With `GR_FLAG=1` (v0.4.1): WRF's per-call green-roof sums from the ported soil/vegetation model ([proofs/w41_writergaps](proofs/w41_writergaps/README.md)) | `GR_FRAC_ROOF>0` with `GR_FLAG=0`, `GR_FLAG=1` without a class fraction, `GR_TYPE` other than 1/2 (all three fields) |
| `EP_PV_URB3D` | BEP+BEM; c57 | +0 without PV; with `PV_FRAC_ROOF>0` (v0.4.1): WRF's panel production sum (`hsfluxpv`) | none |
| `QNBCA` | Thompson aerosol-aware `mp_physics=28/38`; c19/c65 | +0: `mp_gt_driver` zeroes black carbon unless `wif_input_opt=2`; nonzero `QNBCA` input is refused | `wif_input_opt=2` black-carbon aerosol (`QNBCA`) |
| `TAOD5502D` | `mp_physics=28/38`; c19/c65 | +0 unless `aer_opt=3` with RRTMG SW; under `aer_opt=3` (v0.4.1) the computed column AOD ([aerosol guide](docs/AEROSOL.md)) | none |
| `MSKF_REFL_10CM` | Morrison aerosol `mp_physics=40`; c20 | +0 with `do_radar_ref=0` (WRF assigns it only under `do_radar_ref=1`). With `do_radar_ref=1` it equals the computed `REFL_10CM`: its MSKF cumulus hydrometeor arrays exist only for `aercu_opt>0` and are unallocated in WRF otherwise ([upstream note](proofs/t_classd/UPSTREAM_WRF_BUGS.md)) | `aercu_opt>0`, MSKF + CESM aerosol (`MSKF_REFL_10CM`) |

One-moment NSSL (`mp_physics=19/21`) `REFL_10CM` is computed with WRF's one-moment
`radardd02` branch (byte-exact against pristine WRF REAL4 column output). Morrison
`mp_physics=10/40` with `do_radar_ref=1` (and `nwp_diagnostics=0`) now computes
`REFL_10CM` with WRF's `refl10cm_hm` at history times (pristine REAL4 column oracle:
byte-exact on all 13,376 levels, melting layers included).

The Morrison history reconstruction has84 column-frames at h10–14 differing
by up to15.45dBZ. Missing exact MP-exit operands is an attribution hypothesis,
not a demonstrated cause or coupled PASS. The pristine oracle on the
reconstructed operands equals the port byte for byte. Preserve this distinction
from the routine-boundary oracle and final trajectory gate.
[Class-D evidence and limits](proofs/t_classd/README.md).

Jensen55 now retains its actual pre-sedimentation `REFL_10CM` every MP call;
it is not reconstructed at output time. Its routine-boundary CPU oracle is
byte-exact on5,632 recorded dBZ values, with the disclosed one-ULP input-recipe
qualification. NTU56 now writes prognostic `QHAIL`; original CPU c88 has no
REFL_10CM field. Jensen ice diagnostics are implemented with the residuals
disclosed above. CAM-MG `DLF2` is WRF's initialized +0 only in the admitted
shallow-off branch. NSSL/Jensen `REFD_MAX` is initialized +0 only under the
documented inactive diagnostic controls. These repairs do not relabel old
INCOMPLETE outputs or establish affected-source GPU cost/fidelity.
[Jensen/NTU review](proofs/rv_t_classd/jensen_review/VERDICT_JENSEN.md),
[DLF2 source/CPU proof](proofs/t_cammgws/README.md#dlf2-history-field-mp11-2026-10-16),
[exact inactive branches](src/gpuwrf/io/producer_history.py).

For Noah-MP (`sf_surface_physics=4`) with `grid_sfdda=2`, `HFX_BOTH` and
`QFX_BOTH` retain WRF's initialized zeros: their flux calculation belongs to
classic Noah. The [c78 original-CPU proof](proofs/t_writer_fields/FASDAS_NOAHMP_CPU_ALL_FRAMES.json)
covers every cell in all 25 frames. Composite c53 has only one CPU reference
frame and remains **INCOMPLETE**; no 24-hour qualification is inferred for it.

## NTU56 development: case-scoped F2 source-diffusion undershoots

The opt-in experimental NTU scheme (`mp_physics=56`) has post-RK ice mass
and ice number undershoots in the real Swiss two-step CPU continuation.
The complete legacy-path census records 93 qi and 127 Ni visits, with minima
−1.3227e−10 kg/kg and −954.597/kg, respectively, about 1.5% of the affected
column maxima. These exceed the tested 1e−12 relative round-off criterion;
**they are not round-off and are absent from the matched default control.**

Pristine WRF's complete sixth-order horizontal diffusion routine reproduces
the negative class on the captured full-grid operands. The captured settings
are `diff_6th_opt=2`, `diff_6th_factor=0.12`, `diff_6th_slopeopt=0`: the source
uses the RK1 dt/3 tendency, frozen for the final dt update. No NTU source-term
sign, units or clipping discrepancy was found. No equation, diffusion
coefficient, input, tolerance or clip was changed to suppress the events.

The manager accepted this **case-scoped source-diffusion F2 known issue** on
2026-10-14 at 22:44:27Z for v0.4.0; a full original RK-chain comparison is
not required for this disposition. Its scope is the captured single-domain,
44-level Swiss CPU continuation at dt=18 s, CCNTY=2, FAST0; selecting the
existing native PD transport path retains the class. This is not global F2
clearance or complete original-CPU/GPU forecast qualification. The later old-source
c88 GPU run has25frames with zero bounded misses and NF0, but raw D6 remains
INCOMPLETE for QHAIL and E41 FAIL (STACK3496B). The QHAIL writer and NTU
performance repairs are now merged; final affected-source D6, resources and
performance/VRAM qualification remain pending.
[Final-source integration](proofs/s2_integrate/batch57/QUALIFICATION.json).
See the [qualification and sealed replay inputs](proofs/s2_ntu/README.md)
and [measured F2 ledger](proofs/s2_ntu/QUALIFICATION.json).

## Noah-MP glacier points in shipped versions through v0.3.4

Glacier initialization and undefined history fields were handled, but the
timestep incorrectly ran natural-soil Noah-MP at `IVGTYP == ISICE` points.
WRF calls the separate `NOAHMP_GLACIER` model there. The shipped Swiss example
contains 22 affected points; all 54 domains of the 18 WN3 validation cases and
both PROD domains contain zero. Earlier Swiss forecast receipts do not qualify
the missing glacier timestep.

The development fix adds glacier energy, ice phase change, snow layers and
water balance for `opt_gla=1/2`, with the previous-call `QSNOWXY` retained in
restart state. It intentionally changes the Swiss program. Glacier CPU oracle
and short coupled qualification are separate from the final Swiss 24-hour
comparison against CPU-WRF. The new c00 source-specific 24-hour comparison
now passes bounded D6, complete coverage and E41; final-RC carry-forward remains
TBD-VALIDATION ([receipt](proofs/r_docs040/EVIDENCE_SNAPSHOT.json)). `opt_snf=5` on a glacier
domain is refused because the pristine glacier precipitation partition has
no branch for that choice.

The development glacier history writes `QRAINXY=0` at ice points, matching
WRF's defined per-point hold on the qualified cold-initialized path. The
pristine driver loads that point's `QRAINXY` before the glacier branch and
writes it back unchanged; `NOAHMP_INIT` initializes it to zero. Arbitrary
nonzero externally seeded `QRAINXY` is outside this qualification. The glacier
water budget uses its own rain partition.

Development fix in v0.4.0: WDM5/6 (`mp_physics=14/16`, including shipped WDM6)
omitted cloud/CCN number (`Nc/Nn`) transport, and Morrison (`mp_physics=10`)
omitted snow/graupel number (`Ns/Ng`) transport. WDM rain number (`Nr`) could
already use the materialized Ni/Nr path; the fix transports Nr exactly once
and removes inactive WDM Ni. The fix transports those scalars with WRF's
scalar advection option, reads `QNSNOW/QNGRAUPEL/QNCLOUD/QNCCN` from inputs,
and applies WRF's `flow_dep_bdy_qnn` with bound `ccn_conc` inflow. WDM7 (`26`)
uses the same repaired number path plus its existing hail transport and a
documented fp64 microphysics island. Missing prognostic numbers in active
restarts are refused. These schemes are restricted to single domains until
parent-to-child forcing is qualified. CPU operator evidence is separate from
GPU/coupled-forecast qualification. Later scheme-specific status is recorded in
the [v0.4.0 matrix](docs/release/SUPPORT_MATRIX_v0.4.0.md); the default
Thompson release configuration is unchanged.

## Non-Thompson precipitation and WDM caller defects

These defects are confirmed in **v0.3.0–v0.3.4 and pre-fix v0.4.0 development
builds**, whenever the affected adapter actually ran. In v0.3.3 and earlier,
the CLI often ignored non-default physics selections and ran Thompson instead;
those selections alone do not establish exposure. API/custom-driver runs and
v0.3.4 CLI runs must be assessed by the scheme that actually executed. WSM7,
WDM7 and Morrison-aerosol were refused by the v0.3.4 CLI but became runnable in
v0.4.0 development.

- **Frozen precipitation counted twice in `RAINNC`.** The generic adapter
  stored WRF's all-phase `RAINNCV` as liquid rain, then the writer added the
  overlapping frozen channels again. This affects Purdue–Lin (`mp_physics=2`),
  WSM3/5/6 (`3/4/6`), Morrison (`10`), SBU-YLin (`13`), WDM5/6 (`14/16`),
  WSM7/WDM7 (`24/26`), Morrison-aerosol (`40`) and Goddard (`97`) when frozen
  fallout occurs. Kessler has no frozen increment; Thompson (`8`) and
  Thompson-aerosol (`28`) use separate adapters and are unaffected by this
  bookkeeping defect.
- **Shipped WDM6 (`mp_physics=16`) and WDM7 (`26`) thermodynamic coupling was
  wrong.** Both passed prognostic moist theta to a scheme expecting dry theta
  and omitted the conversion back to moist theta after microphysics. WDM6
  also passed water mask `0` instead of WRF's `2`, selecting the land
  autoconversion threshold over water. The development fix restores the WRF
  dry-theta preparation, moist-theta finish and WDM6 water mask.
- **WSM5/WSM6 (`mp_physics=4/6`) and WDM5/6/7 (`14/16/26`) radiation
  coupling and caller operands were not WRF's.** With RRTMG and `use_mp_re=1`
  (the WRF default) WRF `physics_init` sets `has_reqc/i/s=1`, so RRTMG reads
  the scheme's effective radii and treats snow as its own optical species. The
  port never activated this for these schemes: RRTMG used fallback radii and
  folded snow into the ice path, making snow layers too opaque in the longwave
  (Swiss hour-1 OLR bias c05/c06 −3.5/−3.2, WDM c10/c11/c18 −3.2/−3.1/−4.0
  W m⁻² against original CPU-WRF; radius-coupled schemes ≈0). The WSM/WDM
  callers also passed the Thompson-formula density (R 287.04) instead of the
  post-RK dry `1/(al+alb)` and `DELZ` with g 9.80665 instead of WRF's 9.81.
  The development fix (lane t-wsmdz) activates the WSM5/6 and WDM5/6/7 radii
  and passes WRF's post-RK `DEN`/`DELZ` (WSM3 included); 24-hour GPU
  re-qualification of the affected composites is pending.
- **NSSL (`mp_physics=17/18/19/21/22`), Ferrier (`5/15`) and Goddard 4-ice
  (`7`) microphysics received the wrong air density.** WRF's microphysics
  caller passes the post-RK dry density `1/(al+alb)` that `moist_physics_prep_em`
  forms immediately before `microphysics_driver`; the port passed the earlier
  `phy_prep` total `(1+qv)/ALT`, about 0.3–1.5 % too large in moist layers (NSSL
  also clamped `DZ` at 1 m, which WRF does not). Recomputing original CPU-WRF
  NSSL `REFL_10CM` (composites c14/c15) from its own history reproduces WRF to
  7.6e-6 dBZ only with the dry density (0.043 dBZ off with the old one). The
  development fix (lane t-classd) passes WRF's post-RK `RHO`/`DZ8W` for all
  three families; 24-hour GPU re-qualification of c12–c16 and c62–c64 is pending.
- **WSM3 (`mp_physics=3`) reached RRTMG with its cold ice/snow staged as
  liquid.** WSM3 stores ice in `qc` and snow in `qr`; WRF's RRTMG LW/SW drivers
  restage them as ice/snow below 273.15 K (`module_ra_rrtmg_lw.F:12075-12083`,
  `module_ra_rrtmg_sw.F:10676-10684`) and, with the RRTMG LW/SW pair and
  `use_mp_re=1`, read the WSM3 radii. The port did neither (Swiss c04, RRTMG LW
  + Dudhia SW, hour-1 OLR bias −12.9 W m⁻²). The development fix adds the phase
  view and the WSM3 radius selection; the c04 hour-1 OLR bias on the original
  CPU-WRF state drops to −3.1 W m⁻², the same residual the other radius-less
  RRTMG schemes show (RRTMG `has_req=0` path, separate item).

The v0.4.0 development fix has independent CPU oracle and one-hour CPU-WRF
comparison evidence. Later receipts establish source-specific full 24-hour
bounded D6/coverage for WDM6 and the corrected WSM7/WDM7 quartet; Morrison-aerosol
coverage remains incomplete and F2 negative-event justification stays OPEN.
See the [current matrix](docs/release/SUPPORT_MATRIX_v0.4.0.md) and
[quartet receipt](proofs/v040/gpuqual_qhmass/QUALIFICATION.json). Substantial negative
moisture/number candidates are not justified as blanket roundoff; this CPU
defect fix does not establish GPU forecast qualification or close F2. See the
[CPU composition evidence](proofs/t_mprain/COMPOSITION.json) and
[re-qualification plan](proofs/t_mprain/REQUALIFICATION.json).

**Accumulations in checkpoints written by older affected versions are not
repaired or migrated.** Resuming preserves the existing erroneous totals;
correcting earlier accumulated precipitation requires a fresh forecast.

## v0.3.4 restart schema gaps

In v0.3.4, the NetCDF restart API refuses a populated experimental CAM-UW PBL
carry (`bl_pbl_physics=9`) because its inventory lacks that group. The older
State-only pickle checkpoint reader also refuses current default checkpoints
with a `missing non-additive leaves` error. These are loud errors, not silent
loss of state; the CLI's authenticated full-carry checkpoint store is separate
and its release-default continuation was verified byte for byte.

The development fix adds the CAM-UW NetCDF group and allows inactive optional
State fields in the older reader. Held CAM-UW carry, MYNN WSPD and effective
radii are preserved exactly when populated. Active configurations with missing
held state are refused (E78), including Morrison-aerosol radii independently of
the Thompson radiation flag. CAM-UW files predating the new group/schema marker
can resume only with CAM-UW inactive; regenerate active checkpoints. A legacy
NetCDF file without saved PBL selection requires the caller's resolved inactive
namelist (`read_wrfrst_carry(path, namelist=...)`); an unknown selection is refused.

## v0.3.3 disclosed deviations (unchanged in v0.3.4)

All eight original-CPU scoring readers are closed. The manager accepts the disclosed deviations under the Owner rule; raw failed checks remain failed.

- 0115 strict24 accumulated rain RMSE 1.042–1.045 mm versus 1 mm; fresh same-IC replica .845–.886 mm passes, demonstrating trajectory sensitivity without replacing the failed primary.
- 0115 late V10: primary h44 1.692 m/s; fresh replica h42 1.725/h44 1.623 versus1.5. Frozen A3 REAL FAIL rows retained. Coastal station wind-speed RMSE +1.8%; T2/RH2/WD improve.
- 0408 trace graupel: CPU2.46e-10kg/kg/GPU0 in5boundary cells. 0115 cirrus lifetime: primaryd02h60 QICE2.28e-7kg/kg/QNICE2.4e4/kg in50columns; freshd03h63 QICE2.14e-8/QNICE4.5e4 in112columns, GPU0. Both raw integrity annexFAILs disclosed.
- Unclipped WRF-order source/transport may briefly produce negative moisture: qc−3.84e-6,qv−4.20e-6kg/kg;634857censusvisits,0nonfinite/newclass. This is not a claim of pure round-off.
- Swiss≤4096-column stability firewall:+12%kernels/~8%ordinary-root time, notwholewall; larger WN/Mon programs unchanged.

[All final plots](docs/release/evidence/v033/final_w3/index.html) · [Raw reports, scope and ledger](docs/release/TECHNICAL_VALIDATION_APPENDIX_v0.3.3.md#final-eight-arm-results).

## ≤ v0.3.4: `moist_adv_opt = 0` or an omitted `moist_adv_opt` ran without moisture transport (fixed after v0.3.4, lane o2-moistadv)

In wrf_gpu **0.3.0 – 0.3.4** (and the internal 0.23.4+ builds) the RK step transported water vapour and every
condensate (`QVAPOR`, `QCLOUD`, `QRAIN`, `QICE`, `QSNOW`, `QGRAUP`, plus scheme extras) **only when
`moist_adv_opt` was 1 or 2**. WRF's option 0 (`ORIGINAL`) is the ordinary 5th/3rd-order flux transport without
the positive-definite limiter; the port treated it as *no* dycore moisture transport. Both CLI drivers also bound
an **omitted** `moist_adv_opt` / `scalar_adv_opt` as 0, while the WRF Registry default is 1.

**Who is affected:** CLI runs whose `namelist.input` set `moist_adv_opt = 0` **or did not set `moist_adv_opt` at
all**. In 0.3.0 – 0.3.2 (and in 0.3.3 / 0.3.4 with `GPUWRF_FAST_DEFAULTS=0`) such runs completed with moisture
moved only by physics and lateral boundaries — re-run them. With the 0.3.3 / 0.3.4 release defaults the same
namelists stopped after model load with `NotImplementedError: GPUWRF_PHYS_TEND_RK_WRF: physics sc_tend for
non-advected [...]` — no completed forecast (an initial lead-zero frame may exist). A namelist that omitted
`scalar_adv_opt` (or set 0) ran Thompson `QNICE`/`QNRAIN` with the plain stencil instead of WRF's default
positive-definite one (on nests with both keys 0 also from stage-start operands). Namelists that set both keys to
1 or 2 — the release namelists (Swiss, PROD, WN3), the WRF `em_real` template and the shipped examples — are **not
affected by that transport fix**; its historical pre-glacier Swiss identity
was `2bf12cb8`. The pre-F20 candidate baseline has Swiss `f672a573` and PROD d01
`6fc715a2`, with separate glacier/source qualification
([identity and receipts](proofs/r_docs040/EVIDENCE_SNAPSHOT.json)).

**Fix:** option 0 now runs WRF's plain transport on every RK stage (pristine `advect_scalar` parity on real
Swiss operands, `tests/test_o2_moistadv_original.py`), an omitted key binds the Registry default 1 in both
drivers, and `cu_physics = 6/16` no longer needs `moist_adv_opt ≠ 0`. Omitted-key audit of every bound option:
[`docs/namelist-compatibility.md`](docs/namelist-compatibility.md#cli-binding-v034). Besides the already-documented
substitutions (omitted scheme selectors run the release suite with a warning, `sf_surface_physics` and the replay
driver's fixed harness values, omitted `diff_opt`/`km_opt`), the only other mismatch was an omitted `radt` (WRF:
radiation every step; the port: 30 min), which now prints a warning — set `radt` explicitly.

## ≤ v0.3.3: `gpuwrf run` silently ignored non-default physics options (fixed in v0.3.4)

In wrf_gpu **0.3.3 and earlier**, both CLI drivers (the native single-root / live-nested
driver and the CPU-history replay driver) **never read** `mp_physics`, `bl_pbl_physics`,
`sf_sfclay_physics`, `ra_lw_physics` or `ra_sw_physics` from `namelist.input`, nor the
`&dynamics` knobs `epssm`, `damp_opt`, `zdamp`, `dampcoef`, `w_damping`, `diff_6th_opt` and
`diff_6th_factor` (the replay driver also ignored `cu_physics`, `radt`, `cudt`, `top_lid`).
The namelist check accepted the requested scheme and the forecast then **ran
Thompson / MYNN / MYNN surface layer / RRTMG** with epssm 0.5, damp_opt 3, zdamp 5000,
dampcoef 0.2, w_damping 1 and the 6th-order filter (2, 0.12). The validator also refused
`icloud_bl=1`, the value that actually runs, and accepted `icloud_bl=0` and
`bl_mynn_mixlength=2`, which never ran.

**Who is affected:** only runs whose namelist selected something else. The release
configuration (Thompson / MYNN / Noah-MP / RRTMG / KF with those dynamics values) asked for
exactly what ran and is unchanged; its traced program is byte-identical in v0.3.4. **Re-run
any ≤ 0.3.3 forecast that selected a different microphysics, PBL, surface-layer or radiation
scheme or different damping/filter values:** its output came from the default suite.

**v0.3.4:** every scheme and dynamics option is bound per domain. Before any compute,
`gpuwrf run` **refuses** any explicit `&physics` / `&dynamics` / `&noah_mp` value the
selected driver would not run. That includes values that differ from the WRF Registry
default of a key the port does not bind, per-domain selections WRF itself rejects or
overrides (WRF runs the innermost domain's `mp_physics` everywhere), and land options the
native driver cannot build (it supports only 0 and 4). The CPU-history replay driver warns
loudly that its fixed 10 s step, 10 sound steps and hourly CPU land replay replace
`time_step`, `time_step_sound`, `sf_surface_physics` and `sst_update`. An omitted `&dynamics`
knob now runs the WRF Registry default (`epssm` 0.1, `w_damping` 0, `diff_6th_opt` 0; ≤ 0.3.3
hard-wired 0.5 / 1 / 2).

**Exposed by the binding (v0.3.4 CLI probe on the real Swiss case, release defaults):** most
non-Thompson microphysics (`mp=1/2/3/4/6/10/13/14/16/97`), PBL `bl=7/8/12` and GSFC shortwave
`ra_sw=2` could not even be traced under the REAL32 release carry; ≤ 0.3.3 hid this by running
the default scheme. They now run as an explicit fp64 island (scheme computed in fp64, stored
REAL): CPU-traced, not GPU- or forecast-qualified. `mp=24` (WSM7), `26` (WDM7), `28`
(aerosol-aware Thompson) and `40` (Morrison-aerosol) are refused: the release root boundary has no
`qh` / `nwfa,nifa` / `Ns,Ng,Nc` record. Constant-K diffusion (`diff_opt=2`, `km_opt=1`) ran as *no*
explicit diffusion through the CLI and is now refused, with every other `(diff_opt, km_opt)` pair
that has no operational path. The 3-D TKE (`km_opt=2`) and SMS-3DTKE (`km_opt=5`) closures are
unqualified v0.22 scaffolds that produce NaN under the release REAL carry and are refused in v0.3.4;
`diff_opt=1/km_opt=4` (release) and `diff_opt=2/km_opt=3` (3-D Smagorinsky) run.
Per-code CLI support matrix:
[`docs/namelist-compatibility.md`](docs/namelist-compatibility.md#cli-binding-v034).

## Historical issue records (source-dated)

# Known issues and validation limits — v0.3.3

## v0.3.3 CF/b6 wave — historical release HOLD

[Exact current eight-arm evidence](docs/release/CURRENT_W3_EVIDENCE.md) retains
strict0115 d03 RAINNC h21–24 FAIL, hard72 0408 d02 RAINNC h39–72 FAIL and two
0408 QGRAUP REAL FAILs. Current original-CPU/station evidence has0227 8/8 CLEAR
and0115 WS10 A/B DAMAGE+REPLICATED, SYSTEMATIC (2/16); other14 CLEAR.
Monica72/Swiss24 D6 and all-frame integrity annex pass, without clearing the
other cases. Replica CANICE/SNOWENERGY raw flags remain disclosed separately.
Six arms executed CF; Monica/Swiss executed b6. No publication or fullE41 PASS.

## Historical W2 record — source-dated

Reviewed W2 source/defaults candidate is `e83ec1fa7` (`src/` tree
`fb50726fee1e`). Its external gates remain open; source MERGE is not release
acceptance. Original 69d8a9a7d evidence is retained as **pre-W2**.

- **W0115 D6+floor annex: FAIL.** Of 89 raw d03 rows, 57 are floor-limited
  and 32 remain REAL: U10 h38–52 (15), V10 h37–58 (17). Raw wind maxima
  2.38/3.04 m/s; 54 RAINNC h19–72 rows are floor-limited. The strict
  24-hour window is unchanged. Y0 is a 219/219-byte-identical pinned replay,
  not an independent realization; distinct ZR/ZP checks are pending.
- **W0115 station WS10: OPEN.** Non-data-chosen whole-run RMSE excess
  +0.027 m/s versus S 0.018; CI [0.015,0.042]. Late h48–72 is CLEAR.
  T2/RH2*/WD10 are CLEAR; data-chosen scopes are separately disclosed.
- **W0408:** raw D6 PASS, all 73 frames/domain. Two d02 QGRAUP exceptions
  have identical cells/values to pre-W2, so the manager's boundary-trace
  judgement applies. Canopy trace CPU 1.65e-6 mm/GPU zero lies at WRF's
  1e-6 zeroing threshold; interception is active and snow-burial key inert.
- **0227 afternoon WS10: NOT REPRODUCED.** Fresh XC +0.122 versus S
  0.452; realization-class disposition, no named code cause.
- **W2 Monica/Swiss:** Monica diagnostic available with E41 HARD FAIL;
  W2 Swiss cancelled. Old identity/benchmarks stay on
  69d8a9a7d. The new [W2 bands](docs/release/evidence/v033/w2/index.html)
  cover 657 paired domain-hours. This source-dated candidate was held.
- **W2 Monica E41: HARD FAIL [M].** Two-domain fused d01 contains three
  STACK results 664/520/648 B, above 256 B. Their small measured runtime
  share is not a waiver. W3 must compose a reviewed fix and revalidate
  every executable; its attribution and source are pending.

- **W2 0227 B/T2: one-realization class [M].** R32-65 is 8/8 CLEAR, but
  gate A is DAMAGE on IC R (+0.0148 vs S 0.0109); IC P is −0.0038,
  NOT REPLICATED → CLEAR under the frozen pair rule. One realization per
  IC; W3 requires a fresh reading, not an inherited clearance.

## Archived pre-W2 findings

- **BD92(c): KNOWN RESIDUAL**, not zero/fixed. Binding d01 diagnostic maximum
  0.0136 K; the non-binding 0227 d03 h67 −0.0596 K departure remains. Clear-night
  cooling aloft is a v0.3.4 roadmap item.
- **0227 model-side cloud/wind defect: fixed in the final default set [M].**
  MYNN step-one/two, F1'', R32 and whole-run gates pass. Observational R32-60 is DAMAGE [M] in A/WD10, B/T2 and B/RH2*;
  the station defect remains OPEN and tag/push is held.
- **MYNN predictor floor: OPEN.** WRF applies plume `s_aw` floors in TKE
  prediction; the held port omits that path. The upper variance-face extent
  also differs. Default-off fixes are under pristine-oracle review.
- **SWDOWN history on slopes: OPEN.** With slope_rad=1 the held port writes
  slope-normal SWNORM as SWDOWN; WRF restores horizontal flux after the LSM.
  This output-only convention fix is being validated; forcing is unchanged.
- **72 h D6:** 0227 d03 retains five raw failures. Original CPU-WRF's IC twin
  breaches the same fields/leads with larger errors; frozen floor annex accepts
  them. Other final cases are raw clean. Strict 24 h limits are unchanged.
- **Integrity:** all frames are scanned. Trace annex and two manager judgements
  explain recorded flags without changing caps: frozen-dew canopy placement
  (CPU twin reproduces one of two cells) and parent-boundary graupel traces.
  These are disclosed exceptions, not blanket clean-output parity.
- **Native code:** hard stack coverage passes; conservative CALL/kInput flags
  remain, runtime-bound. Compact AOT kInput metadata is UNKNOWN, not zero.
- **Scope:** the development Noah-MP glacier runtime is CPU oracle and short
  coupled-smoke qualified; GPU and Swiss 24-hour qualification remain pending.
  Universal physics-option parity and MPI/multi-GPU domain decomposition
  remain unimplemented/unvalidated.
  Original CPU-WRF restart compatibility is not claimed; GPU checkpoint/resume
  and WRF-style history are separate capabilities.
- Existing speed/energy and data-centre extrapolations remain explicitly
  v0.3.2 evidence. No new whole-machine energy or four-core Swiss CPU energy
  measurement is claimed.

[Current validation and exact annexes](docs/release/V0.3.3.md),
[per-hour statistical bands](docs/release/evidence/v033/final33v/index.html),
and the [verbatim candidate history](docs/release/evidence/v033/CANDIDATE_HISTORY.txt)
keep source scopes and old failures visible. Candidate history is not the final
release verdict.

## W2 source and publication dependencies

The [W2 source inventory](docs/release/V033_W2_FIXES.md) covers urban soil
(overrides/FRZX order), SWDOWN history, MYNN predictor floors, CANWATER
ELAI/ESAI and the **land-only** post-LSM Q2 cap. The old RC k0 warm PBL bias
masked the urban cold term; preserving that compensator is not a fix.
A/WD10 is NOT REPLICATED (roadmap); replicated B/T2/B/RH2* remain the active
fix target in the archived reading. Eight W2 keys are now reviewed defaults;
full release acceptance remains separate. Approved
[history cleanup](docs/release/HISTORY_CLEANUP.md) requires sealed maps,
full-object scans and verified remote refs; credential rotation is unconfirmed.


W2 qualification update (2026-10-07 10:18Z, source471): the distinct ZR realization leaves two REAL V10 rows at h42/h44; ZP has zero REAL rows. Frozen Z class is PROPERTY, IC-R-specific, and the original W0115 FAIL is retained. R32-67 v3 confirms SYSTEMATIC station WS10 damage in A/B/C (13 CLEAR, 3 SYSTEMATIC, 0 OPEN); B′ remains CLEAR. Earlier OPEN statements above are dated readings, superseded by this result. W3 must apply the affected original-CPU and station gates on its final source; no W2 waiver or inherited clearance.

## Noah-MP nondefault controls: compiled precision and qualification

The u2-land v0.4 development controls are opt-in and CPU original-source qualified; Whole-executable GPU resource/E41, performance and full coupled-forecast D6 remain pending. CPU deviceless component resource scans and short coupled integration/restart smokes are recorded separately. Literal REAL4 operation-order gates match pristine WRF byte for byte. Compiled differences are compared separately with independently frozen input-ULP envelopes, without widening nominal tolerances. The following small excesses are disclosed; final D6 decides forecast acceptance.

| Boundary | Compiled envelope excess | Practical scale |
|---|---|---|
| Existing Jordan snow default | 3/765 values, maximum5.96e-8 | New BATS/SNF2–5 choices all within; old default program retained |
| BTR/RSF surface choices | 64/480 outside | Maximum raw RSURF difference0.00390625s/m at large resistance; at most7ULP |
| Monthly phenology | 9/288 outside, maximum5.96e-8 excess | Input-LAI7–9 and dynamic2/5/6 gates all within |
| RAD1 geometry | 2/432 outside, maximum2.032e-43 | Subnormal WGAP only |
| SFCDIF2 canopy drag | 9/5400 outside, maximum7.45e-9 | All Jarvis values within |
| Dynamic carbon trajectories | 29/1920 outside, maximum1.526e-5 excess | Maximum raw pool difference0.00024414g/m2 |
| Equilibrium water table | 6/192 outside, maximum4.547e-13 | RUNSUB in mm/s |
| Prognostic groundwater alone | 98/1440 outside, maximum1.4305e-6 excess | Maximum raw WT difference0.00048828125mm |
| Complete prognostic soil water | 20/240 outside, maximum2.6226e-6 excess | ZWT/QIN/RUNSUB |
| Saxton–Rawls pedotransfer | 28/576 outside, maximum4.768e-7 excess | Maximum raw coefficient difference1.907e-6 |
| VIC/XAJ/dynamic VIC surface | 12/480 outside, maximum5.684e-14 excess | VIC/XAJ all within |
| Full runoff/tile soil solver | 46/2496 outside, maximum8.2784e-7 excess | Maximum raw difference1.5423e-6; active Hooghoudt both permeability modes all within |

Irrigation, SST skin, thermal/freezing, Liu crop and crop phenology gates are all within their independent compiled bounds. Receipts and source/input hashes are under `proofs/u2_land` and the lane's retained `LAND*_precision.json` artifacts. These are component and caller results, not GPU-vs-GPU fidelity claims.

Dynamic VIC has a separate material upstream initialization sensitivity: original unassigned YD reads change52/480 outputs under NaN versus zero compiler initialization, maximum0.0026633055. The port mirrors the binding zero policy at those exact sites and labels run8 experimental. See [upstream defects](proofs/u2_land/UPSTREAM_WRF_BUGS.md). Liu crop requires valid active crop calendar/season-GDD geography; crop fractions in a crop-disabled wrfinput do not establish that those maps exist. Missing active maps are refused. The approved independent glacier model supports opt_gla1/2; glacier SNF5 is undefined in WRF and refused. MMF run5 is a separate lane, and GECROS2 retains main's scientific refusal.

Native RAD1 uses REAL4 source operations (source jaxpr fp64 outputs0). CUDA libdevice tanf/cosf emits6 fp64 conversion/multiply instructions in its generic slow range-reduction branch; finite canopy angles are bounded below π/2, so that branch is unreachable in the qualified regime. The original trig expressions are retained. Component resources:72registers,STACK32B,LOCAL0; runtime GPU execution remains pending. Other new native phases haveSTACK0/LOCAL0; RUN1 preserves the explicit WRF S_NODE DOUBLE island.
