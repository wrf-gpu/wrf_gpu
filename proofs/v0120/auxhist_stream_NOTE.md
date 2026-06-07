# auxhist secondary history output stream — proof note

**Sprint:** auxhist secondary history output stream
**Branch:** `worker/opus/auxhist-stream` (off `d617029`)
**Status:** PASS — off by default, zero trunk risk.

## What it does

Adds a single configurable WRF auxiliary-history (`auxhist`) output stream: a
SECOND NetCDF stream written at its own interval (independent of the hourly main
`wrfout` cadence), carrying a configurable variable subset — the classic
high-frequency surface-diagnostic stream (e.g. 15-min `U10/V10/T2/Q2/PSFC/...`).

WRF namelist semantics honoured (`gpuwrf.io.auxhist_stream.AuxhistStreamConfig`):

- `auxhist{N}_outname`    → `outname` pattern; default `auxhist{N}_d<domain>_<date>`
  with WRF `<domain>` (2-digit index) / `<date>` (`YYYY-MM-DD_HH:MM:SS`) tokens.
- `auxhist{N}_interval`   → `interval_minutes`.
- `frames_per_auxhist{N}` → `frames_per_file`.
- `io_form_auxhist{N}`    → `io_form` (2 == NetCDF; the only form this port writes).

## How (stream-generic reuse, additive only)

- `wrfout_writer.write_prepared_wrfout(prepared, *, variable_subset=None,
  target_override=None)`: when `variable_subset is None` (default) EVERY prepared
  field is written exactly as before → main stream byte-for-byte unchanged. A
  subset restricts the emitted vars; `Times`/`XTIME` + global attrs are always
  written, so the auxhist file is a genuine schema-valid WRF history file.
- `AsyncWrfoutWriter.submit_subset(...)`: the auxhist frame reuses the SAME
  host-materialized payload as the main hour (no extra device→host pull) and is
  serialized on the same background writer thread.
- `daily_pipeline`: optional `DailyPipelineConfig.auxhist` (default `None`). When
  set, the forecast hour is advanced in `gcd(60, interval)`-minute sub-segments so
  every auxhist frame is a GENUINE sub-hour GPU snapshot (never interpolated). With
  `auxhist=None`, sub-steps collapse to 1 full-hour advance → existing behaviour.

## Proof

- Test: `tests/test_auxhist_stream.py` (7 tests, all PASS — sync + async paths).
- Artifact: `proofs/v0120/auxhist_stream_proof.json` (`status: PASS`).

Falsifiable checks (all green):

| check | result |
|---|---|
| off by default → no second stream | PASS |
| main wrfout field values identical off-vs-on | PASS (byte-equal) |
| second stream files distinct from main stream | PASS |
| 15-min interval over 1 h → 4 frames at :15/:30/:45/:00 | PASS |
| each auxhist file carries ONLY the requested subset (+Times/XTIME) | PASS |
| frames carry distinct sub-hour state (T2 +2.5 K/15 min) | PASS |
| WRF-named filenames `auxhist1_d02_<date>` | PASS |
| main wrfout keeps full field set when auxhist subset is small | PASS |

## Files changed

- `src/gpuwrf/io/auxhist_stream.py` (new) — stream config + WRF naming/cadence.
- `src/gpuwrf/io/wrfout_writer.py` — `write_prepared_wrfout` subset/target hooks.
- `src/gpuwrf/io/async_wrfout.py` — `submit_subset` + `_WriteJob` for both streams.
- `src/gpuwrf/io/__init__.py` — export the new config/helpers.
- `src/gpuwrf/integration/daily_pipeline.py` — `auxhist` config + cadence wiring.
- `tests/test_auxhist_stream.py` (new) — proof.

## Risks / scope

- Sub-hour stepping requires the auxhist interval × dt to be an integer step count
  (`run_forecast_operational` raises otherwise) — true for 15/20/30/60-min at
  dt=10 s. Intervals coprime to 60 (e.g. 7 min) step at gcd(60, m); a `gcd`-vs-dt
  misalignment would surface as a clear `_steps_for_hours` error, not silent
  fabrication.
- One auxhist stream is supported (the sprint scope). Multiple simultaneous
  streams would be a small generalization of the same config list.
- Non-blocking: off by default, no main-stream change → must NOT gate the v0.12.0
  tag. Recommended merge target: **v0.13** (clean, additive; can also ride v0.12.0
  if a secondary stream is wanted at release since it is risk-free when unused).
