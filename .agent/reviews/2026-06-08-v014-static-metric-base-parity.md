# V0.14 Static Metric/Base-State Parity Review

## Objective

Implement and run the V0.14 Static Metric/Base-State Parity probe for Case 3 d02, then isolate whether the reported static/base mismatches originate in CPU WRF artifacts, GPU/native init, runtime metrics, writer payload selection, or forecast-step dynamics.

## Verdict

Source touched: yes.

The vertical C/DN/RDN and MAPFAC mismatches in the retained GPU h1 wrfout are pre-fix writer payload mismatches caused by stale `GridSpec.metrics`. The raw `Gen2Run.grid(...).as_grid_spec()` path creates `GridSpec.metrics` with `DycoreMetrics.flat`; the writer reads `grid.metrics`, while runtime namelist metrics are loaded from WRF input. After the local patch, the current synthetic writer payload uses loaded WRF metrics and matches wrfinput for those fields. The retained GPU h1 artifact predates the patch, so it still demonstrates the old emitted-static payload.

The local source patch is limited to `src/gpuwrf/integration/d02_replay.py`: load `metrics_source = run.history_files(domain)[0]`, call `load_wrfinput_metrics(metrics_source)`, and attach it to the frozen grid with `dataclasses.replace(grid, metrics=metrics)` immediately after `GridSpec` creation. This preserves existing `case.metrics` / `namelist.metrics` runtime behavior and fixes shared grid metadata consumed by the writer without editing `contracts/grid.py`, dycore, radiation, or `wrfout_writer.py`.

## Field Origins

- `C2H/C2F/C4H/C4F/RDN`: pre-fix writer payload from stale flat `GridSpec.metrics`; fixed in current patched synthetic writer payload. Not CPU wrfinput, not GPU native-init, not runtime-consumed dynamics.
- `MAPFAC_M` and related MAPFAC fields: same stale `GridSpec.metrics` writer path; current patched grid metrics match wrfinput.
- `XLAT/XLONG`: writer payload fallback, separate from metrics. The runtime state does not carry lat/lon arrays, so writer-generated projection coordinates match retained GPU h1 and differ from wrfinput.
- `HGT`: CPU wrfinput-vs-CPU-wrfout terrain convention. Current writer/retained GPU follows wrfinput; CPU wrfout h0/h1 differ from wrfinput.
- `PHB`: dominated by CPU wrfinput-vs-CPU-wrfout convention; current zero-step writer follows wrfinput and retained GPU h1 is near wrfinput.
- `PB/MUB`: CPU wrfinput-vs-CPU-wrfout convention plus a retained-GPU-h1 component. With no retained GPU h0, the GPU-side component cannot be split between forecast-step dynamics and h1 writer base-field reconstruction.

## Commands Run

- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m py_compile proofs/v014/static_metric_base_parity.py src/gpuwrf/integration/d02_replay.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 python proofs/v014/static_metric_base_parity.py`
- `python -m json.tool proofs/v014/static_metric_base_parity.json >/tmp/static_metric_base_parity.validated.json`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 python proofs/v014/grid_cell_envelope.py`
- `python -m json.tool proofs/v014/grid_cell_envelope.json >/tmp/grid_cell_envelope.validated.json`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python - <<'PY' ... import gpuwrf.integration.d02_replay ... PY`

All required commands exited 0. The static probe emitted nonfatal XLA CPU AOT feature warnings. No GPU job was launched and TOST was not resumed.

## Proof Objects

- `proofs/v014/static_metric_base_parity.py`
- `proofs/v014/static_metric_base_parity.json`
- `proofs/v014/static_metric_base_parity.md`
- `proofs/v014/grid_cell_envelope.json`
- `proofs/v014/grid_cell_envelope.md`
- `/tmp/static_metric_base_parity.validated.json`
- `/tmp/grid_cell_envelope.validated.json`

## Key Evidence

- `wrfinput_vs_loaded_namelist_metrics`: `EXACT=33`, `WITHIN_TOL=5`, `MISSING=11`, `DIFF=0`.
- `wrfinput_vs_raw_grid_metrics_without_attach`: `DIFF=29`; C2H/C2F max `95000 Pa`, RDN max `161.674`.
- `wrfinput_vs_current_grid_metrics`: `EXACT=33`, `WITHIN_TOL=5`, `MISSING=11`, `DIFF=0`.
- `wrfinput_vs_current_synthetic_writer_payload`: `DIFF=6`, all lat/lon fields; vertical/map metrics fixed in the current writer payload.
- `prefix_synthetic_writer_payload_vs_retained_gpu_h1`: `EXACT=46`, `DIFF=3`, proving the retained GPU h1 mostly matches the pre-fix writer payload.
- `current_synthetic_writer_payload_vs_retained_gpu_h1`: `DIFF=26`, expected because retained GPU h1 predates the patch.

## Grid Envelope Note

`proofs/v014/grid_cell_envelope.py` was rerun CPU-only after the source patch, but it compares retained old GPU wrfouts. It therefore cannot reflect the patched writer payload without a fresh GPU/writer artifact. The rerun still reports static/grid mismatch count `31`; this must not be claimed as a post-fix improvement or failure.

## Unresolved Risks

- No retained GPU h0 wrfout exists, so `PB/MUB` cannot be split between forecast-step dynamics and h1 writer base-field reconstruction.
- `XLAT/XLONG` writer fallback remains unresolved and was not fixed by the metric plumbing patch.
- A fresh GPU/writer smoke would be needed to prove emitted wrfout static metrics are fixed on disk.

## Next Decision Needed

Manager decision: whether to authorize a fresh low-priority GPU writer smoke with `scripts/run_gpu_lowprio.sh` to produce a post-fix h0/h1 artifact, or leave this as a CPU-only proof plus source patch until the next scheduled GPU validation.
