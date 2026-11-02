# Cloud objectives in regional optimization

Cloud scoring runs on the CPU and reads existing WRF history files. It adds no
forecast kernels, device transfers, allocations or compilation. It needs
`pip install 'gpuwrf[optimize]'` for SciPy and pyproj. A score diagnoses supplied
forecasts; it does not establish that tuning improves real forecasts.

The input contract follows the ALISIOS MSG3 format handoff of 2026-10-28.
Read real ALISIOS data only from a private copy produced by
`alisios_guard.facade_stage` after manager admission. Protected roots and
symlinks into them are refused before payload reads. The structure package's
`FORMAT_ONLY` example is for parser checks only, never truth for fitting or
scientific scores. No ALISIOS data ships with this feature.

## Truth and daylight

Configure `format: "alisios.msg3.v1"` and `truth: "<facade-returned-path>"`.
For multiple training/test dates or a 72-hour case, `truth` also accepts a list
of native facade-returned paths. Every source is bound independently; identical
D18 endpoint duplicates are counted once. Conflicting endpoint values or a
changed native grid/order fail. No new merged source file is needed.
The NPZ has exactly four arrays, read with `allow_pickle=False`:

| Key | Type/shape | Meaning |
| --- | --- | --- |
| `times` | little-endian int64 `(slots,)` | Actual UTC epoch seconds, increasing 15-minute slots |
| `cloud` | uint8 `(slots,pixels)` | 0 clear, 1 cloudy, 255 unavailable |
| `lat`, `lon` | little-endian float32 `(pixels,)` | Corrected native centres, same flat index as cloud |

The D18 filename denotes an 18Z-to-18Z window, not a UTC calendar day. A 24-hour
file has 97 endpoint-inclusive slots. Use the recorded epoch times. Matching
is exact: there is no nearest-time sampling, temporal interpolation or
carry-forward. Missing a frozen requested daytime model frame fails, preserving
the paired denominator. Unmatched unused/night slots are counted. Initialization and forecast times
beyond the declared horizon are excluded/refused; training valid times must
precede `test_start`.

The NPZ includes night and has no day flag. A frozen, externally supplied
`daylight` list is mandatory. Each entry names an exact native time and either
`"all"` pixels or explicit native flat indices. Omitted slot/pixel combinations
are excluded. Bind these slots to the agreed solar-elevation contract before
starting the study; the scorer never infers daylight from a local hour or a
clear code. For example, in a **synthetic rehearsal**:

```json
{
  "format": "alisios.msg3.v1",
  "truth": "staged/msg.npz",
  "daylight": [
    {"time": "2026-01-11T10:00:00Z", "pixel_ids": [10863, 11052]},
    {"time": "2026-01-11T11:00:00Z", "pixel_ids": [10863, 11052]}
  ],
  "threshold_g_m2": 10,
  "pixel_threshold": 0.5,
  "species": ["QCLOUD", "QICE"],
  "fss_radii_km": [0, 5, 10],
  "max_distance_m": 2263.2503,
  "min_common_area": 0.9,
  "superobs": {"correlation_length_km": 10, "contract": "SYNTHETIC_REHEARSAL_ONLY"},
  "model_region": {"land_only": true, "max_height_m": 1500},
  "regions": {"north": [10863, 11052]}
}
```

These example slots/indices illustrate the configuration and authorize no
scientific use. The real pilot needs its own admitted, frozen scope.

## Model and footprint operator

For each WRF mass cell, CWP in g/m² is
`1000 * sum((QCLOUD + QICE) * rho_dry * dz)` over levels. Additional condensate
species are opt-in. WRF mixing ratios are per kg dry air: dry density is
`p / (287*Tair*(1 + QVAPOR/(287/461.6)))`, with `p=P+PB`,
`Tair=(T+300)*(p/100000)^R_D_OVER_CP`, with WRF's shared
`R_D_OVER_CP = R_d/cp`, `R_d=287` and `cp=7*R_d/2=1004.5`, and interface heights
`(PH+PHB)/9.81`. Native signed condensate is integrated without clipping.
Positive pressure, density, temperature and layer thickness, correct dimensions
and recognized units are required. Nonfinite candidate/reference values fail
before any averaging, including under a night/255 mask.

Threshold native CWP at **>=10 g/m²**, then area-average the cell occurrences
onto the fixed footprint. This order follows the delivered ALISIOS 4V operator.
The footprint's continuous output is a spatial cloudy-area fraction in [0,1],
**not a cloud probability**. It is compared with binary MSG3 occurrence by
equal-pixel mean squared residual. EKI uses these fractions directly, with an
explicit representation/instrument error `sigma` in fraction units.

Each selected model-cell centre is assigned to its nearest corrected MSG3
centre in **EPSG:6933** within `max_distance_m`; exact ties choose the lowest
native index. Weights are `DX*DY/MAPFAC_M²`. `model_region` optionally restricts
land/height; freeze it with the study. This approximates pixel footprints by
assigned model area. **Exact satellite polygons and view geometry beyond
corrected geolocation are unavailable; cloud-height parallax is not applied.**
No native row/column layout is invented from flat indices.

Zero-assignment pixels are unavailable. The common valid area in a slot must
cover at least `min_common_area` (>=0.9) of the assigned model area of that
slot's frozen daylight pixels; otherwise that slot is excluded and counted.
Area outside the assignment-distance limit is disclosed through selected versus
assigned area. All ensemble members/candidates must have identical scored
slot/pixel support; missing model support cannot improve a candidate by dropping
errors. Cell assignment, physical areas and native pixel order also have a
shared SHA256 binding, so identical pixel support with a changed footprint fails.
All-empty daytime support fails.

## Diagnostics and integration

Discrete pixel occurrence uses an inclusive footprint-fraction threshold
(`pixel_threshold`, default0.5). Pixel and region reports give hits, false
alarms, misses, correct negatives, frequency bias, CSI and ETS. Undefined
event ratios are JSON null, with their counts retained.

FSS uses great-circle neighbourhood radii in kilometres on the **flat native
centres**, area-weighted fractions over valid neighbours, and area-weighted
valid centres. Squared terms pool over slots before division. Radius0 is the
pixel-scale comparison. Clear/clear has FSS1; regions without support have null.
The displacement diagnostic is the mean great-circle distance between
area-weighted spherical cloudy-pixel centroids over defined slots. When either
cloud set is empty or has no defined centroid, displacement is null and its
defined-slot count is reported. A centroid can hide compensating object shifts;
use it together with FSS and categorical errors.

For a discrete `gpuwrf.optimize.study.v1` study, set `objective: "cloud"` and
`cloud: { ...the config above... }`. Stations/observation CSVs are unnecessary.
`scales` defaults to `{"cloud":1}`; if supplied, loss is MSE divided by scale².
Existing `optimize plan`, `score` inventories and `run` use this collector.
Date-separated halving selects the winner before reading held-out scores.

`channel_diagnostics` keeps three distinct channels per matched slot: prognostic
CWP, PBL/subgrid `QC_BL`/`QI_BL`/`CLDFRA_BL`, and actual held RRTMG
`QC_RAD`/`QI_RAD`/`QS_RAD`/`QG_RAD`/`CLDFRA` inputs. They share the frozen
footprints, common valid area, regions and daylight masks. Regional output gives
area-mean CWP, the fixed CWP10 process fraction and layer-mean condensate/fraction
profiles. Missing fields explicitly make a channel UNAVAILABLE; prognostic
condensate never fills a missing active-radiation tap. Nonfinite available taps
fail even at night. The active condensate integral uses history-time density/dz
and is labelled accordingly: the actual radiation-call time and call-time CWP
are not recorded by existing history and remain unavailable. Held input profiles
retain their solver-input meaning. Enabling these existing taps still requires
the radiation diagnostic's own ON/OFF consumer qualification.

The CWP10 process fraction always remains10 g/m². An alternative
`threshold_g_m2` is an explicitly pre-registered observation-operator choice
bound to study identity; it is not an EKI parameter or a post-hoc selection knob.

For `gpuwrf.optimize.continuous.v1`, replace the station scorer with
`"scorer": {"kind":"cloud", "sigma":0.2, "cloud":{...}}` and omit `stations`.
The default EKI observation operator is **superobservations**. Supply a frozen,
externally admitted `superobs.correlation_length_km` and its `contract` reference;
the scorer does not invent a scientific correlation length or grant admission.
The example10 km is synthetic. Blocks are fixed squares of that side length in
EPSG:6933 with origin(0,0), partitioned by disjoint registered regions and native
time. Pixels outside named regions form a separate partition. Overlapping
regions are refused for this operator. Prediction and binary truth are averaged
with the same assigned model-area weights. A block's truth is therefore a
cloudy-area fraction. Each block is one residual; supplied sigma is **not**
divided by square-root pixel count. Block keys, counts and areas are reported.
Spatial aggregation does not remove temporal/regime correlation: the admitted
error/weighting contract must still account for it.

The ensemble-space capability removes a **memory limit only**. It does not
admit correlated native MSG pixels as independent scalar-sigma observations.
Native fitting (`observation_operator: "native"`) is currently refused before
truth/model payloads, at driver construction before member prefetch, and by the
direct collector, pending the admitted ALISIOS
correlation contract **and its bound operator**. No native covariance/ordering
contract has been delivered; the scorer does not guess a Gamma or accept a bare
length/reference label as a native covariance. This matches the cloud-parameter
optimizer's P1.5 guard. Native pixel diagnostics remain available.

The default remains `observation_operator: "superobservations"`, which requires
and actually applies its admitted correlation length. On older EKI backends the
memory guard separately refuses more than 2048 total observations across all
training cases (each float64 matrix is then at most 32 MiB).
An admitted scope that exceeds that legacy cap needs the new update; do not change the
measurement contract after looking at scores. Native pixel counts, hits, FSS,
displacement and the fixed CWP10 process channel remain diagnostics/controls,
independent of the coarser EKI residuals.
The existing control basis, parameter bounds, EKI and ensemble launcher remain
in use. The result retains prior/posterior cloud diagnostics by case alongside
normalized residuals. Mixed cloud/station constraints and wave admission are
separate optimizer features, not a property of this single-objective scorer.

Source identities bind the truth bytes and study configuration. The scorer
streams WRF columns per frame but retains footprint fractions/CLM masks for the
selected slots. Memory scales with `slots*pixels`, the native model frame, and
sparse neighbourhood edges, rather than all 3D forecast fields. FSS graphs omit
unassigned native-box pixels. Multiple EKI members additionally retain their
fraction vectors.

Measured CPU workload: 72 reads of one archived 120×70×44 WRF frame plus
8400-pixel FSS and a synthetic 176-pixel superobservation scope took **3.39 s
for the whole script**, **2.075 s inside the scoring phases**, and **345.2 MiB
peak RSS**. The superobservation phase reduced 12672 native samples to 576
observations in 0.041 s. This repeats warm cached weather and uses an invented
50 km correlation length; it is a component cost proxy for an hourly 72-hour
case. Actual full 72-hour real-MSG scoring wall time remains unmeasured. It
does not establish a forecast or optimizer speedup. Source-bound receipts,
scope and checks are in `proofs/v45_cloudscore/GATE.json`.

The current admission rule is recorded in
`proofs/v45_cloudscore/CORRELATION_GATE.json`. Earlier scorer and rebase receipts
remain immutable history. The rebase receipt's native-positive check established
memory capacity only and is superseded for statistical admission by P1.5.
Superobservations remain the default; there is no automatic native fallback.
