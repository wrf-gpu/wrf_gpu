# V0.14 Lat/Lon Writer Payload Review

Date: 2026-06-09
Worker: GPT-5.5 xhigh / Codex
Branch: `worker/gpt/v013-close-manager`

## Objective

Fix the writer-only `XLAT`/`XLONG` fallback by routing real WRF lat/lon host
payloads into wrfout output without changing dycore, physics, runtime state, or
JIT-visible `GridSpec`/`State`/`OperationalNamelist` leaves.

## Verdict

`PASS` for the contracted CPU-only writer proof.

The writer now selects real `XLAT`/`XLONG` and staggered variants from host-only
diagnostics when supplied, and falls back to the projection fallback only when no
real pair is supplied. This is writer-only output plumbing and does not change
model numerics.

## Files Changed

- `src/gpuwrf/io/wrfout_writer.py`
- `src/gpuwrf/integration/daily_pipeline.py`
- `src/gpuwrf/integration/nested_pipeline.py`
- `proofs/v014/latlon_writer_payload.py`
- `proofs/v014/latlon_writer_payload.json`
- `proofs/v014/latlon_writer_payload.md`
- `.agent/reviews/2026-06-09-v014-latlon-writer-payload.md`

## Commands Run

- `python -m py_compile src/gpuwrf/io/wrfout_writer.py src/gpuwrf/integration/daily_pipeline.py src/gpuwrf/integration/nested_pipeline.py proofs/v014/latlon_writer_payload.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 python proofs/v014/latlon_writer_payload.py`
- `python -m json.tool proofs/v014/latlon_writer_payload.json >/tmp/latlon_writer_payload.validated.json`
- `python -m py_compile proofs/v014/latlon_writer_payload.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_m7_netcdf_writer.py tests/test_m7_daily_pipeline.py tests/test_auxhist_stream.py tests/test_auxhist_multistream.py`

## Proof Objects

- `proofs/v014/latlon_writer_payload.py`
- `proofs/v014/latlon_writer_payload.json`
- `proofs/v014/latlon_writer_payload.md`
- Temporary emitted NetCDF proof files:
  - `/tmp/v014_latlon_writer_payload/wrfout_d02_2026-05-01_19:00:00.fallback`
  - `/tmp/v014_latlon_writer_payload/wrfout_d02_2026-05-01_19:00:00.real_latlon`

## Key Evidence

- `XLAT` emitted with real payload vs GPU-native wrfinput: RMSE `0.0`, max_abs `0.0`.
- `XLONG` emitted with real payload vs GPU-native wrfinput: RMSE `0.0`, max_abs `0.0`.
- Staggered `XLAT_U`/`XLONG_U`/`XLAT_V`/`XLONG_V` are also exact vs GPU-native wrfinput.
- Fallback remains active when payload is absent:
  - `XLAT` fallback max_abs vs wrfinput: `0.027322769165039062`
  - `XLONG` fallback max_abs vs wrfinput: `0.02660655975341797`
- `C1/C2/C3/C4`, `DN/DNW/RDN/RDNW`, and `MAPFAC_*` fields are exact between the with-payload and no-payload prepared paths.

## Unresolved Risks

- No fresh GPU smoke was run in this sprint by contract, so retained GPU wrfout files still show the old fallback until a new run writes fresh output.
- Remaining static/base exclusions are unchanged: `PHB`, `HGT`, `PB`, `MUB`.
- The proof imports the CPU JAX stack to load existing metric helpers and emitted XLA CPU AOT feature warnings, but the command exited 0 and used `JAX_PLATFORMS=cpu`.

## Next Decision

CPU-only proof is sufficient for the writer selection fix. A fresh GPU h1 live-nested smoke is needed only if the manager wants the next wrfout comparator run to clear the retained on-disk `XLAT`/`XLONG` fallback artifact.
