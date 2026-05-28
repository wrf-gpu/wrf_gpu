# F1 Phase 1 Instrumentation Assessment

Outcome: fallback path selected.

Evidence:

- `find /home/enric/src -type f -name wrf.exe -print` found only legacy deprecated WRF binaries under `/home/enric/src/wrf_gpu_DEPRECATED_archived_on_github_nric_wrf_gpu/builds/`.
- `find /home/enric/src -type f -name ideal.exe -print` found no `ideal.exe`.
- `external/wrf_savepoint_patch/HOOK_INVENTORY.md` states all wrapper hook bodies are empty and active emission is still Python-orchestrated.
- `bash external/wrf_savepoint_patch/build.sh` failed immediately because `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` no longer exists in this workstation layout.
- Direct edits to `/home/enric/src/wrf_gpu/` are prohibited by the sprint contract, and this worktree is the only manager repo write target.

Fallback oracle used:

- Source start: `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z/wrfout_d02_2026-05-22_00:00:00`
- Source end: `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z/wrfout_d02_2026-05-22_01:00:00`
- Fixture: `tests/savepoint/fixtures/wrf_b6_100step/column/`

Limitation:

This is a real CPU WRF history-output fallback, not a per-RK-stage or per-acoustic-substep Fortran savepoint stream. It is good enough to retire the JAX-vs-JAX tautology and force an honest failure, but not good enough to prove bitwise WRF dycore parity.
