# B36 gate harness (host-sync)

Pristine-WRF REAL4 gate for the nest moist/number boundary records (B36) on real operands:
PROD d02 (h01, CPU-WRF h12) and WN3 0227 d03 (Rain01, CPU-WRF h18), 7 Thompson species,
producer (forcedown value/rate, pristine Fortran + NumPy REAL4 literal) and consumer
(relax_bdy_scalar + spec_bdytend, 3 leads incl. the lead-0 cancellation), deletion mutations.

GPU (under scripts/with_gpu_lock.sh, JAX_PALLAS_USE_MOSAIC_GPU=false):

    python tools/host_sync/b36/gpu_probe.py --source <snapshot> --out <lane dir>/oracle.json --skip-restart

Expected on main after B36: producer 28/28, consumer 84/84, every active species deletion-sensitive.
Fortran oracle libraries are built/reused under <USER_HOME>/wrf_gpu2_lanes/host-sync/B36/oracle.
Inputs are read-only lane pickles + CPU-WRF wrfout files (paths in gate.py).
