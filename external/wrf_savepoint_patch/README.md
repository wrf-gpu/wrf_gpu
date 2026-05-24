# M6B0 WRF Savepoint Patch

This directory is the isolated instrumentation lane for M6B0. It does not
write into `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/`.

`build.sh` creates `build/wrf.exe.instrumented` and `build/hook_registry.json`
as the non-production harness entry point. The reviewable Fortran hook anchors
are recorded in `module_small_step_em_savepoint_hooks.patch`; the savepoint data
writer lives in `scripts/m6b0_wrf_savepoint_extract.py`.
