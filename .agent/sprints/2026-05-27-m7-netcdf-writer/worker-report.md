# Worker Report - M7 NetCDF WRF-Compatible Writer

Summary: WRITER_READY. Implemented an additive CPU-only NetCDF wrfout writer for the 41-variable minimum subset selected from the M7 I/O compatibility audit. The old `.npz` writer path in `src/gpuwrf/coupling/driver.py` was not modified. The new writer emits WRF-standard dimensions, `Times`, `XTIME`, global attributes, per-variable metadata, downstream-critical fields, staggered coordinate fields, and WRF base/perturbation pairs for `P/PB`, `PH/PHB`, and `MU/MUB`.

## Files Changed

- `src/gpuwrf/io/wrfout_writer.py`
- `src/gpuwrf/io/__init__.py`
- `tests/test_m7_netcdf_writer.py`
- `scripts/m7_netcdf_writer_smoke.py`
- `.agent/sprints/2026-05-27-m7-netcdf-writer/minimum_variable_list.md`
- `.agent/sprints/2026-05-27-m7-netcdf-writer/total_to_perturbation_mapping.md`
- `.agent/sprints/2026-05-27-m7-netcdf-writer/roundtrip_proof.json`
- `.agent/sprints/2026-05-27-m7-netcdf-writer/compat_matrix_v2.md`
- `.agent/sprints/2026-05-27-m7-netcdf-writer/worker-report.md`

## Commands Run

```text
taskset -c 0-3 python -m pytest tests/test_m7_netcdf_writer.py -q
....                                                                     [100%]
4 passed in 0.42s
```

```text
taskset -c 0-3 python scripts/m7_netcdf_writer_smoke.py --output-dir .agent/sprints/2026-05-27-m7-netcdf-writer
M7 NetCDF writer smoke complete
roundtrip_proof: .agent/sprints/2026-05-27-m7-netcdf-writer/roundtrip_proof.json
compat_matrix_v2: .agent/sprints/2026-05-27-m7-netcdf-writer/compat_matrix_v2.md
candidate_file: /tmp/wrf_gpu2_ncwriter_m7_netcdf_writer/wrfout_d02_2026-05-25_18:00:00
pass: True
```

```text
taskset -c 0-3 python scripts/validate_agentos.py
{
  "errors": [],
  "ok": true,
  "required_files_checked": 31,
  "skills_checked": 13
}
```

```text
taskset -c 0-3 python -m pytest tests/test_m7_wrfout_io_compat.py tests/test_m7_netcdf_writer.py -q
.......                                                                  [100%]
7 passed in 0.41s
```

## Proof Objects

- `.agent/sprints/2026-05-27-m7-netcdf-writer/minimum_variable_list.md`
- `.agent/sprints/2026-05-27-m7-netcdf-writer/total_to_perturbation_mapping.md`
- `.agent/sprints/2026-05-27-m7-netcdf-writer/roundtrip_proof.json`
- `.agent/sprints/2026-05-27-m7-netcdf-writer/compat_matrix_v2.md`
- `tests/test_m7_netcdf_writer.py`

## Risks

- The writer can derive or zero-fill some diagnostics when the state object does not provide explicit fields. That is acceptable for this schema sprint and synthetic proof, but the daily-pipeline integration must provide real `SWDOWN`, `GLW`, `PBLH`, precipitation, cloud, and flux diagnostics before making forecast-quality claims.
- This sprint did not wire the writer into the operational driver. AC7 explicitly left GPU integration to a later sprint.
- The compatibility proof uses one Gen2 reference wrfout, per contract. It does not claim coverage over every historical WRF output variant.

## Handoff

Objective: produce a real WRF-compatible NetCDF writer for the minimum downstream-critical wrfout subset.
Branch: `worker/gpt/m7-netcdf-writer`.
Proof status: `roundtrip_proof.json` reports `"pass": true`; `compat_matrix_v2.md` reports 0 downstream-critical missing fields, 0 AC1 dimension mismatches, 0 dtype mismatches, and 0 metadata mismatches.
Next decision needed: manager/tester should decide whether the next sprint wires `write_wrfout_netcdf` into the daily-pipeline output path or first adds stricter no-placeholder checks for operational fields.
