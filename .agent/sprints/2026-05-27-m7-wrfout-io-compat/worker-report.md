# Worker Report — M7 wrfout I/O Compatibility Matrix

Summary: COMPAT_MATRIX_READY. Built a CPU-only structural audit for one representative Gen2 d02 wrfout and the current `write_wrfout_gpu` payload. The reference file loaded successfully. The GPU writer is not drop-in WRF-compatible today: it writes a compact `.npz` proof container, not NetCDF, and the matrix records 13 documented WRF-variable deviations, 7 GPU-only payload keys, and 362 CPU WRF variables missing from GPU output. Downstream-critical rows are 5 documented deviations and 21 missing GPU fields.

## Files Changed

- `scripts/m7_wrfout_io_compat_audit.py`
- `tests/test_m7_wrfout_io_compat.py`
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/cpu_wrfout_reference_inventory.json`
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/gpu_wrfout_writer_inventory.json`
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/compat_matrix.md`
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/io_endpoint_audit.md`
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/explicit_deviations.md`
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/worker-report.md`

## Commands Run

Command:

```bash
taskset -c 0-3 python -m py_compile scripts/m7_wrfout_io_compat_audit.py
```

Output:

```text
<no stdout/stderr>
```

Command:

```bash
taskset -c 0-3 pytest -q tests/test_m7_wrfout_io_compat.py
```

Output:

```text
...                                                                      [100%]
3 passed in 0.05s
```

Command:

```bash
taskset -c 0-3 python scripts/m7_wrfout_io_compat_audit.py --reference /mnt/data/canairy_meteo/runs/wrf_l3/20260525_18z_l3_24h_20260526T221207Z/wrfout_d02_2026-05-25_18:00:00 --output-dir .agent/sprints/2026-05-27-m7-wrfout-io-compat
```

Output:

```text
M7 wrfout I/O compatibility audit complete
cpu_inventory: .agent/sprints/2026-05-27-m7-wrfout-io-compat/cpu_wrfout_reference_inventory.json
gpu_inventory: .agent/sprints/2026-05-27-m7-wrfout-io-compat/gpu_wrfout_writer_inventory.json
compat_matrix: .agent/sprints/2026-05-27-m7-wrfout-io-compat/compat_matrix.md
endpoint_audit: .agent/sprints/2026-05-27-m7-wrfout-io-compat/io_endpoint_audit.md
explicit_deviations: .agent/sprints/2026-05-27-m7-wrfout-io-compat/explicit_deviations.md
```

## Proof Objects

- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/cpu_wrfout_reference_inventory.json`
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/gpu_wrfout_writer_inventory.json`
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/compat_matrix.md`
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/io_endpoint_audit.md`
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/explicit_deviations.md`

## Risks

- The current GPU output is not NetCDF and lacks WRF filenames, dimensions, `Times`/`XTIME`, global attrs, and variable attrs.
- Downstream Gen2 consumers still need missing fields including `XLAT`, `XLONG`, `PSFC`, `RAINC`, `RAINNC`, `SWDOWN`, `GLW`, `PBLH`, `HFX`, `LH`, `CLDFRA`, and terrain/static fields.
- `P`, `PH`, and `MU` are especially risky because the writer labels total-state aliases as WRF perturbation variables while omitting `PB`, `PHB`, and `MUB`.
- `build_replay_case` uses d02 hourly wrfout side-history for boundaries; native `wrfbdy` is not part of the current forecast path. No `wrfrst` endpoint was found.

## Handoff

- objective: Produce the M7 wrfout I/O compatibility matrix and endpoint audit without model code changes or GPU runtime.
- files changed: listed above, all within contract ownership.
- commands run: listed above, all CPU-pinned with `taskset -c 0-3`.
- proof objects produced: listed above.
- unresolved risks: GPU output is not drop-in Gen2 wrfout-compatible; downstream-critical fields and restart/boundary endpoints remain implementation decisions.
- next decision needed: Decide whether the next sprint implements a real NetCDF WRF-compatible writer or a documented adapter layer for Gen2 consumers.
