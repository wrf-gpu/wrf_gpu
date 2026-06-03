# Grell-Freitas Oracle Factory

`build_and_run.sh` compiles the unmodified WRF Grell-Freitas sources from
`/home/enric/src/wrf_pristine/WRF/phys` with a standalone single-column driver.
It writes small JSON savepoints under `proofs/v060/savepoints`.

This is a real WRF-module oracle, not a full `wrf.exe` run. The parity report
therefore sets `full_wrf_exe_run=false`.
