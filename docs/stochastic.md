# Stochastic physics and input perturbations

**Status: Experimental.** In v0.4.0 the stochastic physics options (`&stoch`) ran but had not been compared with the original CPU WRF over 24 hours. Since then, single-domain 24-hour runs with SKEBS, SPPT and SPP match original CPU WRF runs with the same namelist and seeds (Swiss case, [GPU check](#gpu-qualification-swiss-d01-24-h)). In v0.5.0 (checked 11 November 2026) a 24-hour three-nest Tenerife run (9/3/1 km) with SKEBS, SPPT and SPP on all domains also matches original CPU WRF on every domain ([nested GPU check](#gpu-qualification-tenerife-three-nest-24-h-v050)); `rand_perturb=1` has no CPU-WRF reference. `gpuwrf perturb` runs on the CPU, offline.

`gpuwrf perturb` prepares offline ensemble member inputs. It supports
**spatially uniform fractional wind perturbations** of U and V and
**seeded boundary-layer temperature/moisture noise** (see below). Pressure and
spatially varying balanced perturbations are not implemented.

```bash
gpuwrf perturb --case inputs/case --output inputs/wind-members \
  --members 5 --seed 451 --wind-std 0.05
gpuwrf ens prepare --mode batch --cases inputs/wind-members/member-* \
  --hours 24 --output-dir ensembles/wind
```

The command creates new member case directories containing private copies of
`wrfinput_dNN` and `wrfbdy_dNN`. Fixed `namelist.input` and `wrflowinp_dNN` inputs
are linked to the source case; retain the source case while using the members.
It refuses overwrites, missing/packed/nonfinite selected fields and incomplete
boundary wind records. `--ic-only` explicitly leaves lateral boundaries unchanged
and permits a case without `wrfbdy_d01`. Source inputs are never edited.

For N members the recipe draws N standard normal values using NumPy PCG64 with
the unsigned 64-bit seed. It subtracts their ensemble mean, then rescales their
sample standard deviation (N−1 denominator) to `--wind-std`. A member's multiplier
is one plus its rescaled value. The example uses a **dimensionless 5% sample
spread**. Before REAL32 rounding, the finite ensemble has mean multiplier one;
normalization makes the members dependent. A recipe producing nonpositive multipliers is refused.
The manifest stores the full-precision factors and the REAL32 factors actually
used, so exact recipes can be inspected and replayed independently of RNG version.

One multiplier scales U and V throughout every nest, level and boundary record
of a member. Both boundary values (`U_BXS`, etc.) and time tendencies
(`U_BTXS`, etc., including the final record) receive the same multiplier. WRF's
mass/map-factor coupling is linear in wind with mass fixed, so this operation
commutes with coupling and with time differentiation, up to REAL32 rounding.
Wind units remain m/s and coupled boundary units remain their original WRF units.
All other fields, attributes, dimensions and file formats are retained.

This is an **unbalanced input recipe**: it preserves wind direction and the
coherent fractional IC/BC change, but provides no geostrophic balance, calibrated
ensemble spread, forecast skill guarantee or spatial covariance model. It does
not use WRF's RNG and does not enable stochastic physics during a forecast.

`perturb_manifest.json` records source/prepared SHA256 hashes, exact multipliers,
seed, sampling definition and changed-cell counts. Prepared cases feed the existing
`ens prepare batch` scheduler; this command itself never runs a forecast or uses
JAX/GPU. Storage grows by the copied IC/BC files per member. Peak array workspace
is proportional to the largest selected wind variable; there is no stacked
member array. Hashes detect input edits during preparation.

### Seeded boundary-layer temperature and moisture noise

Original WRF has no such option; this recipe is a gpuwrf addition and runs only on
the input files. It gives otherwise identical runs a small, reproducible spread
(repeat runs of the same inputs are bitwise identical, so without it there is no
noise floor). The recipe lives in a small JSON file:

```json
{"schema": "gpuwrf.perturb.thermo.v1", "seed": 20260227,
 "theta_std_k": 0.1, "qv_rel_std": 0.01, "full_m": 500.0, "top_m": 1500.0}
```

```bash
gpuwrf perturb --case inputs/case --output inputs/thermo-members \
  --members 3 --thermo-config thermo.json
```

For each member and nest, NumPy PCG64 seeded with `[seed, member, nest]` draws
independent standard-normal values on the mass grid (clipped at 3 standard
deviations, `clip_sigma`). Potential temperature gets `theta_std_k` kelvin times
that value, water vapour is multiplied by one plus `qv_rel_std` times a second
value. The full amplitude applies up to `full_m` above ground, then fades linearly
to zero at `top_m`; the outer `spec_bdy_width` rows of every nest are left alone.
Heights come from each file's own geopotential and terrain. WRF reads the moist
potential temperature `THM`, gpuwrf reads the dry `T`; both are updated
consistently, and every untouched cell keeps its exact bytes. `domains` (optional)
limits the edit to some nests; the others are linked unchanged. Lateral and lower
boundary files are linked, never edited. The command refuses inputs that contain
NetCDF fill or missing values, refuses to publish a non-finite `THM`, and refuses a
recipe that changes no value (for example a layer below the first model level).

The change is not balanced: dry mass, geopotential, pressure and winds stay as
they were. Keep the amplitudes small. Because only input files change, the model
program is the default one, and original CPU WRF can run the very same member
inputs for a matched reference. `perturb_manifest.json` records the recipe file,
the per-file change counts and all input/output hashes. On the three-nest
Tenerife case two members took 2 seconds and 0.26 GB of memory and changed about
60,000, 248,000 and 77,000 values per nest; the existing input checks
(`validate_wrfinput`, with the balance check waived) pass, with a highest
relative humidity of 0.998 on d02.

## Opt-in WRF stochastic physics

The default is OFF. Select WRF controls in `&stoch`, for example:

```fortran
&stoch
 skebs = 1,
 sppt = 1,
 spp_pbl = 1,
 rand_perturb = 1,
 nens = 7,
/
```

`skebs=1` (also the original `stoch_force_opt=1` alias) generates wind and
potential-temperature backscatter tendencies. `sppt=1` multiplies the combined
WRF physics tendencies for U, V, potential temperature and water vapor, using
the capped dimensionless pattern. Other moisture species are unchanged.
`rand_perturb=1` generates and exports WRF's random diagnostic field; the
upstream driver applies no separate model tendency from it.

`spp=1` enables all three SPP selectors. Independent `spp_conv`, `spp_pbl` and
`spp_lsm` arrays are also accepted, including child-domain selections. WRF
allocates all three fields when any SPP selector is enabled; the selectors
separately control generation and consumption. Scalar Fortran array assignments
set the first domain entry; write vectors for independent domain selections.
The original `spp` master overrides all three selector arrays globally.

MYNN SPP changes momentum/heat diffusivities, the cloud-PDF humidity numerator,
plume entrainment before its limiter, and MYNN surface-layer local momentum and
thermal roughness. It consumes the previous held pattern, then the first RK
part2 updates patterns for the current tendencies and child forcing. Perturbed
roughness is local, so it does not repeatedly multiply the stored surface ZNT.
WRF 4.7.1's cumulus driver explicitly resets its local SPP controls to zero;
RUC copies the LSM pattern but does not use it in the soil arithmetic. These two
selectors therefore generate diagnostic patterns with the same upstream physical
no-op. No generic tendency multiplier is substituted for either parameterization.

The native profile requires physics and WRF RK source coupling. SPPT currently
requires PBL 0/5 and cumulus 0/1; GWDO requires MYNN. SPP with GSL GWDO=3 is
refused. Values other than 0/1, unknown controls, nonfinite parameters and
unsupported multi-perturbation/boundary extensions fail closed before execution.
Registry INTEGER controls require signed INTEGER4 values, including vertical
modes and spectral bounds; fractions, quoted numbers and logical values are
refused. Custom `kminforct`, `lminforct`, `kmaxforct`, `lmaxforct` bounds are
SKEBS-only. WRF requires their defaults whenever SPPT, RAND or any SPP selector
is active in any domain, including an independent child selector.
`hrrr_cycling=.true.` is required for the restart profile. Fourier setup needs
at least 12 points on each full horizontal stochastic extent. All source defaults
come from `Registry/registry.stoch`; the LSM length scale is 50 km. Small domains
may need explicitly smaller Gaussian length scales to resolve their spectrum.

## Reproducibility and qualification

The stochastic physics is checked on the CPU against the original WRF code and,
for 24-hour single-domain forecasts, on the GPU against original CPU WRF runs with
the same namelist and seeds (see [GPU check](#gpu-qualification-swiss-d01-24-h)).
Since v0.5.0 a 24-hour three-nest Tenerife run has the same GPU check (see
[nested GPU check](#gpu-qualification-tenerife-three-nest-24-h-v050)); `rand_perturb=1` cannot be checked that way because
original WRF crashes with it.
The RNG profile matches GNU Fortran 14.3's eight-word `RANDOM_SEED` state and
xoshiro256** REAL4 stream, including original WRF signed INTEGER4 overflow,
seed/member/date mixing and Gaussian rejection. WRF RNG is compiler-defined;
another compiler's stream is not promised. Seeds `iseed_skebs`, `iseed_sppt`,
`iseed_rand_pert`, `iseed_spp_conv`, `iseed_spp_pbl`, `iseed_spp_lsm`, `nens`
and the start date determine the stream. There is no host RNG in stepping.

The full carry stores REAL32 spectral coefficients, held fields, normalization
plans, INTEGER4 seeds and pre-cap event counters. Integer64 operations implement
the RNG recurrence; no floating-point64 stochastic carry is introduced. The
FFT follows original unnormalized FFTPACK5 ordering. The vertical imaginary
assignment deliberately reads the already-updated real coefficient, as WRF does.

Original CPU-WRF three-step histories and restart seeds check all eight fields
with vertical structure off/on. Seven independent 1,000-step original-WRF
trajectories check amplitude and spectra, including 5,600 exact seed words.
MYNN signed land/water/snow surface, active plume-limiter, real drizzle-cloud and
diffusivity-height regimes have separate checks against the original source. SKEBS/SPPT
C-grid tendency adapters use the original source expressions and split mass,
including WRF's initialized high-side mass ghost. These component checks and
short CPU coupled runs do not establish full forecast drift or station skill;
the 24 h GPU runs against CPU-WRF are described below.

Children receive Registry-staggered parent SINT fields through the existing
nest forcing path. They hold no Fourier plans and advance no independent RNG.
As in WRF, only domain 1 makes patterns; every nest receives its parent's
fields after each parent step. SKEBS and SPPT act on every nest as soon as any
domain switches them on; a nest caps the SPPT pattern with its own
`gridpt_stddev_sppt` and `stddev_cutoff_sppt`, and applies SPP in MYNN only when
`spp` or its own `spp_pbl` entry is on. Write per-domain `&stoch` values as full
vectors: a single value sets domain 1 only. When WRF opens a nest of domain 1 it
re-initializes domain 1's stochastic setup once more; for `rand_perturb` this
draws ten extra random planes per such nest, which gpuwrf reproduces (checked
against the original code). gpuwrf counts only the nests the run actually opens, so a
domain-1-only run from a three-nest namelist keeps the single-domain stream. Note that
original WRF 4.7.1 (GNU, dmpar) crashes with `rand_perturb=1` before its first
history, so there is no CPU-WRF reference for that switch; the nested reference runs
without it. A nest's first history (`--emit-initial-history`) comes, as in WRF, after
the parent's first step and force-down, so its pattern fields are already nonzero
while domain 1's are still zero. Nests that open later than domain 1, nests with a
different number of levels, two-way feedback (`feedback=1`) and batched members (B>1)
are refused with stochastic physics on. Nests write the SPP spectral coefficient
fields as zeros and their seed arrays as zeros, like WRF.
Both NetCDF and pickle restarts preserve every stochastic leaf and the next
step exactly. Restarts record the stochastic controls (`nens`, seeds, scales) per
domain and refuse to resume with different ones or with a domain's stochastic carry
missing. Active checkpoints missing the carry fail closed; old OFF checkpoints
remain usable. Compiled programs are shared between ensemble members: `nens`, seeds
and scales do not change the compile-cache key. Active history adds eight native pattern fields and seven `STOCH_CLIP_*`
pre-cap counters, and fills the six native seed arrays from the live streams.
The existing OFF output schema is retained.

With stochastic physics off, there is no extra device state or step work; the
default forecast programs are checked to be unchanged with
`proofs/s1_port/release_identity.py`.
With it on, the model uses FFTs, fused pointwise forcing and an exact parallel
form of WRF's serial random stream. Every polar Gaussian proposal consumes
exactly two uniform numbers, so parallel lanes can start at known positions of
the stream (GF(2) jump-ahead of the xoshiro256** state, with polynomials
precomputed on the host), draw their proposals with the same arithmetic as the
serial kernel, and keep the first accepted values in order. The next seed is the
state after the last proposal used. The result is bit for bit the same as the
single-thread serial kernel on CPU and GPU (random values and seed words); an
exact serial fallback covers the case of too few proposals, which has never been
observed. For one model step (seven chained pattern draws) on an RTX 5090 this
takes 0.36 ms instead of 4.85 ms on the 43x43 Swiss grid and 0.49 ms instead of
21.4 ms on the 71x121 Canary Islands 9 km grid. There are no per-draw launches and no
host round trips per time step. Vertical-constant fields are stored with one
level; child domains keep only fields.

Reproduce the component checks with `tests/test_stochastic_*.py`. Original arithmetic
fixtures and compiler/source hashes are tracked in `tests/fixtures/u2_stoch_real4.*`;
build recipes and the actual CPU-WRF reference staging are in `proofs/u2_stoch/`.
Temporary originals are copied privately, preserving executable `.text` bytes.

## Stochastic ensembles

`gpuwrf ens prepare --mode stochastic` writes N member namelists from one case
with SKEBS and SPPT (optionally SPP), identical `iseed_*` and `nens` = 1..N,
following WRF's ensemble convention, and records each member's seed words.
`--mode multiphysics` uses a table of physics combinations that passed the
v0.4.0 GPU-vs-CPU-WRF D6 gate, optionally with SKEBS on top. Both refuse
combinations the model or WRF would not perturb (SPPT outside PBL 0/5 and
cumulus 0/1, SPP without MYNN) and Gaussian length scales that the model's
spectral setup cannot resolve on the case grid. See
[ensemble.md](ensemble.md#stochastic-and-multiphysics-presets) for usage, the
physics table and the Berner et al. (2011) comparison of the two approaches.
A plan for that comparison on the Tenerife three-nest cases is in
`proofs/w41_stoch/BERNER_PLAN_V050.md`.

## GPU qualification (Swiss d01, 24 h)

Each arm runs the default Swiss case (Thompson, MYNN, Noah-MP, RRTMG, no
cumulus; 43x43x45 at 3 km, dt 18 s) for 24 h on an RTX 5090 with only an
`&stoch` group added, against an original serial CPU-WRF 4.7.1 run of the
**identical namelist and seeds**. Gates: the frozen D6 manifest over every
paired hourly frame (`proofs/v040/final_validation/score.py`: all original
history fields present on GPU, raw finite audit of both arms), every stochastic
history field and seed array against CPU-WRF (`proofs/w41_stoch/pattern_compare.py`,
pre-registered bound `8e-6 + 64 eps` of the field maximum, seeds exact) and the
compiled-kernel resource check. Gaussian length scales are 15 km (see the length-scale note).

| run | `&stoch` | D6 vs CPU-WRF (fields x hourly frames) | stochastic fields vs CPU-WRF (worst error / field max), seeds | kernel check | seconds per forecast hour |
|---|---|---|---|---|---|
| OFF (reference run of the same case, 2026-10-16) | none | PASS, 362 x 25 | none written | PASS | 0.91 |
| SKEBS | `skebs=1` | PASS, 365 x 25 | 3 fields, 5.2e-7; seeds exact | PASS | 1.00 |
| SPPT | `sppt=1` | PASS, 363 x 25 | 1 field, 6.2e-7; seeds exact | PASS | 0.98 |
| SPP | `spp=1` (convection, PBL, land) | PASS, 374 x 25 | 12 fields, 1.9e-6; seeds exact | PASS | 1.01 |
| all | SKEBS + SPPT + SPP + `rand_perturb`, vertical structure on, `nens=7` | PASS, 379 x 25 | 17 fields, 1.6e-6; seeds exact | PASS | 1.12 |

The field counts are the full set of history fields compared in every frame.
The frozen D6 limits apply to 62 of them (temperatures, winds, pressure,
moisture, precipitation and similar); the others are reported without a
limit, and the stochastic fields and seeds are checked separately (column 4).
Every field that CPU-WRF writes is present in every GPU frame and no frame
holds a NaN or infinity. The only guard repairs are the hydrometeor
non-negativity repairs that the OFF run also makes, with similar counts
(172,848-176,737 against 174,717 over 24 h); they belong to the default
program and remain an open item (negative moisture visits). The
GPU files carry seven extra `STOCH_CLIP_*` counters (port diagnostics, not
written by WRF). Before this qualification the GPU history lacked SPP's
`FIELD_CONV`, `FIELD_PBL` and `FIELD_SF`, which CPU-WRF writes as zero when
SPP is on without Grell cumulus; they are now written. With all schemes on, the
earlier single-thread random stream took 1.99 s per forecast hour; the
parallel stream gives 1.12 s with identical patterns and seeds.

Times are the median of hours 2-24 of cold single-process runs on a shared
machine, not benchmarks; the OFF run is from another session and source tree,
so the differences are indicative only. Records: `proofs/w41_stoch/GPU_QUALIFICATION.md`.

## GPU qualification (Tenerife three-nest, 24 h, v0.5.0)

Checked on 11 November 2026 on an RTX 5090 with the v0.5.0 stochastic code: the WN3 Tenerife case of
27 February 2026, 18 UTC (three nests, 9/3/1 km), 24 hours, `skebs`, `sppt` and `spp` on all three domains,
`nens=1`, `rand_perturb=0`, explicit seeds, against an original CPU-WRF 4.7.1 run of the **identical namelist
and seeds**.

| check | result |
|---|---|
| D6 vs CPU-WRF, every hourly frame | PASS on d01, d02 and d03, 25 paired frames each; no missing or non-finite field |
| 13 stochastic history fields per domain vs CPU-WRF | worst error 3.5e-6 (d01), 1.9e-6 (d02), 1.8e-6 (d03) of the field maximum, bound 1.6e-5; seed arrays exact in all 25 frames |
| second member (`nens=2`), same compiled programs | seeds advance and every enabled pattern differs from member 1; no new compiled program |
| seeded boundary-layer noise member | seeds equal to member 1, initial temperature change at most 0.3 K; no new compiled program |
| compiled-resource check | stack check PASS; one larger input-transpose fusion (600 optimized HLO instructions, rule 500) in the nest program, accepted and disclosed ([release notes](../release_notes/RELEASE_NOTES_v0.5.0.md#compiled-resource-disclosures-opt-in-programs)) |

The first (cold) run took 814 s including compilation, with a GPU memory peak of 10,670 MiB and 12.0 GiB of
host memory. Not measured: the runtime share of the larger fusion. Not covered: `rand_perturb=1` (original WRF
crashes), nests that open later than domain 1, nests with another number of levels, two-way feedback and batched
members (all refused with stochastic physics on). Result record SHA-256
`543b49b9ca5007bd350f5eded890736ee3eac76aaca3413585b14dd3b81de333`.
