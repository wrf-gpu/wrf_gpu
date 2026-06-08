# Sprint Contract: V0.14 Lat/Lon Writer Payload

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Close the remaining writer-only `XLAT`/`XLONG` fallback in fresh GPU wrfout
files without touching dycore, physics, runtime state, or JIT-visible grid
contracts.

`proofs/v014/base_state_writer_attribution.*` classifies `XLAT` and `XLONG` as
`writer_fallback`: CPU `wrfinput`/`wrfout` and GPU native `wrfinput` are exact,
but the GPU writer currently emits a synthetic projection fallback because the
runtime `State` lacks lat/lon arrays and `GridSpec` intentionally does not carry
them. This sprint must route the real WRF lat/lon payload to the writer as
host-only output metadata.

## Inputs

- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `proofs/v014/base_state_writer_attribution.json`
- `proofs/v014/base_state_writer_attribution.md`
- `proofs/v014/post_static_writer_grid_compare.json`
- fresh h1 GPU smoke output:
  `/tmp/v014_post_static_writer_smoke/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
- CPU truth:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- run-root inputs:
  `/tmp/v0120_merged_run_root/20260501_18z_l2_72h_20260519T173026Z`

## Write Scope

Allowed source files:

- `src/gpuwrf/io/wrfout_writer.py`
- `src/gpuwrf/integration/daily_pipeline.py`
- `src/gpuwrf/integration/nested_pipeline.py`

Allowed proof/review files:

- `proofs/v014/latlon_writer_payload.py`
- `proofs/v014/latlon_writer_payload.json`
- `proofs/v014/latlon_writer_payload.md`
- `.agent/reviews/2026-06-09-v014-latlon-writer-payload.md`

Do not edit `src/gpuwrf/contracts/grid.py`, `src/gpuwrf/contracts/state.py`,
`src/gpuwrf/runtime/operational_mode.py`, dycore files, physics files, WRF
source, or any active same-state marker artifacts.

## Implementation Constraints

- No new JIT-visible `State`, `GridSpec`, or `OperationalNamelist` array leaves.
- No device transfer inside timestep loops.
- Prefer loading static lat/lon once at case/domain setup and passing it to
  `write_wrfout_netcdf`/`prepare_wrfout_payload` via existing host output
  metadata (`diagnostics`) or a similarly host-only writer argument.
- The writer may still use its projection fallback only when real lat/lon is not
  supplied. Synthetic/unit tests without real WRF statics must continue to work.
- Include staggered fields if available: `XLAT_U`, `XLONG_U`, `XLAT_V`,
  `XLONG_V`. At minimum, mass-grid `XLAT`/`XLONG` must be exact for the L2 d02
  proof case.
- Keep the Markdown summary short; put tables and file paths in JSON.

## Required Proof

The proof must be CPU-only unless the manager explicitly approves a short GPU
writer smoke later. It must:

1. Demonstrate the writer selects real lat/lon payloads when supplied and falls
   back only when they are absent.
2. Compare emitted or prepared `XLAT`/`XLONG` against the CPU/GPU native
   `wrfinput_d02` payload with RMSE, bias, p99_abs, max_abs, finite coverage,
   and worst cell.
3. Check that unrelated formerly fixed static metric fields remain untouched by
   the proof path (`C1/C2/C3/C4`, `DN/DNW/RDN/RDNW`, `MAPFAC_*`).
4. State whether this changes model numerics. Expected answer: no, writer-only
   output payload.

Suggested commands:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 \
  python proofs/v014/latlon_writer_payload.py
python -m json.tool proofs/v014/latlon_writer_payload.json \
  >/tmp/latlon_writer_payload.validated.json
python -m py_compile proofs/v014/latlon_writer_payload.py
```

## Acceptance Criteria

- Source change is limited to host-only writer plumbing in the allowed files.
- JSON validates and names exact input/output files compared.
- Mass-grid `XLAT` and `XLONG` are exact or within NetCDF serialization
  roundoff against the native WRF input payload for the L2 d02 proof case.
- No dycore, physics, runtime-state, or GPU memory claim is made.
- The closeout names whether a fresh GPU h1 smoke is needed before the next
  comparator run, or whether CPU-only writer proof is sufficient.

## Closeout

Close with:

- files changed;
- exact commands run;
- proof objects produced;
- `XLAT`/`XLONG` verdict;
- any remaining static/base exclusions (`PHB`, `HGT`, `PB`, `MUB`);
- next manager decision.
