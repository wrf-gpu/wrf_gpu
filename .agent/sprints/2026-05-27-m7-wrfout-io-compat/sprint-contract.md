# Sprint Contract — M7 wrf{input,bdy,out,rst} I/O Compatibility Matrix

**Sprint ID**: `2026-05-27-m7-wrfout-io-compat`
**Created**: 2026-05-27 (autonomous overnight loop, parallel to 1km memory audit)
**Status**: READY
**Predecessor**: `.agent/decisions/M7-PERF-MEASUREMENT-CLOSEOUT.md` (M7 perf step closed)

## Objective

Per M7 acceptance gate #2 (`.agent/milestones/M7-canary-operational-v0.md`), produce a complete I/O compatibility matrix between the GPU forecast's input/output formats and the Gen2 CPU WRF reference. The matrix names every supported variable, schema-level difference, and explicit deviation. Downstream consumers (the Gen2 post-processing pipeline, AEMET station verification, daily delivery) must be able to read GPU output the same way they read CPU output, or the deviations must be explicit and documented.

This sprint is **schema-comparison only**, no model code changes, no GPU runtime, no test forecasts. Pure structural audit. Reads existing Gen2 wrfout files + the GPU's `write_wrfout_gpu` path (already implemented at `src/gpuwrf/coupling/driver.py:1139`).

## Acceptance

- **AC1 — Reference Gen2 wrfout inventory**: pick a single representative wrfout from the Gen2 backfill (e.g. `/mnt/data/canairy_meteo/runs/wrf_l3/20260520_18z_l3_24h_*/wrfout_d02_*`). Use `netCDF4` or `xarray` to enumerate every variable: name, dimensions, dtype, attributes, units. Emit `.agent/sprints/2026-05-27-m7-wrfout-io-compat/cpu_wrfout_reference_inventory.json`.

- **AC2 — GPU wrfout inventory (synthetic, no GPU runtime)**: read the `write_wrfout_gpu` function in `src/gpuwrf/coupling/driver.py:1133-1264` and statically enumerate which State fields it writes, with what netCDF dim/dtype/attribute mapping. Do NOT execute the function on the GPU. Emit `.agent/sprints/2026-05-27-m7-wrfout-io-compat/gpu_wrfout_writer_inventory.json`.

- **AC3 — Compatibility matrix**: produce `.agent/sprints/2026-05-27-m7-wrfout-io-compat/compat_matrix.md` table with rows = WRF variable names (union of both sides), columns = (CPU has, GPU writes, dim agreement, dtype agreement, units agreement, classification). Classifications: `MATCH`, `DEVIATION_DOCUMENTED`, `DEVIATION_UNDOCUMENTED`, `MISSING_GPU`, `MISSING_CPU`, `EXTRA_GPU`. Aim: every variable consumed by Gen2 post-processing must be MATCH or DEVIATION_DOCUMENTED.

- **AC4 — wrfinput / wrfbdy / wrfrst footprint check**: the GPU forecast currently consumes `wrfinput_d02` + `wrfout_d01` (for boundary tendencies) via `gpuwrf.integration.d02_replay.build_replay_case` and produces `wrfout` only. Check whether wrfbdy + wrfrst (restart) formats need to be either consumed or produced for M7's daily-pipeline gate. Emit `.agent/sprints/2026-05-27-m7-wrfout-io-compat/io_endpoint_audit.md` covering each of the four endpoints.

- **AC5 — Deviation report**: write `.agent/sprints/2026-05-27-m7-wrfout-io-compat/explicit_deviations.md` listing every intentional schema difference between GPU output and Gen2 WRF. For each, provide: (a) what's different, (b) why (perf, simplification, scope), (c) whether downstream consumers care, (d) action required (none / document / re-implement). This is the "explicit deviation document for every intentional difference" called out in `.agent/milestones/M7-canary-operational-v0.md`.

- **AC6 — Worker report** with verdict `COMPAT_MATRIX_READY` or `BLOCKED_REFERENCE_MISSING` (if no Gen2 wrfout is loadable).

## Files Worker May Modify

- `scripts/m7_wrfout_io_compat_audit.py` (NEW — inventory + comparison script)
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/**`
- `tests/test_m7_wrfout_io_compat.py` (NEW, OPTIONAL — pin the inventory schema, not the actual file content)

## Files Worker Must Not Modify

- `src/gpuwrf/**` — audit only, no code change
- governance files
- `/mnt/data/canairy_meteo/**` — Gen2 data is read-only

## Hard Rules

1. **No model code changes.** Schema audit only.
2. **No GPU runtime.** This sprint must not allocate GPU memory or run any JAX kernels. Use plain Python + netCDF4/xarray. This is enforced so it can run in parallel with the 1km memory audit (worker/gpt/m7-1km-memory-audit) without GPU contention.
3. **CPU pinning**: `taskset -c 0-3`.
4. **Do not interfere with tmux `0:1`** (nightly WRF on cores 4-31).
5. **No remote push.** Local commit on `worker/gpt/m7-wrfout-io-compat` only.
6. **One reference wrfout only for AC1** — do not bulk-iterate the 34-day backfill. Pick one representative file from a recent successful 3km l3 run and document the choice.

## Proof Objects

- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/cpu_wrfout_reference_inventory.json` (AC1)
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/gpu_wrfout_writer_inventory.json` (AC2)
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/compat_matrix.md` (AC3 — the main deliverable)
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/io_endpoint_audit.md` (AC4)
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/explicit_deviations.md` (AC5)
- `.agent/sprints/2026-05-27-m7-wrfout-io-compat/worker-report.md` (AC6)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 2-4 h
- Branch: `worker/gpt/m7-wrfout-io-compat`
- Worktree: `/tmp/wrf_gpu2_iocompat`
- GPU usage: NONE (parallel-safe with 1km audit which DOES use GPU)
