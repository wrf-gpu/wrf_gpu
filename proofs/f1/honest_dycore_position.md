# Honest Dycore Position — F1

Verdict: `F1_PARTIAL`.

## What Changed

The old M6B6 comparator no longer compares JAX output to JAX output read back from the same directory. `emit_jax_savepoints` writes candidate HDF5 only; `compare_jax_vs_wrf` loads a separate NetCDF real-WRF reference fixture from `tests/savepoint/fixtures/wrf_b6_100step/`.

## Current Result

Against the F1 real-WRF history fallback, current HEAD passes through **0** compared steps. The first compared step fails:

- step: `1`
- stage: `history_interp_full_timestep`
- field: `mu`
- inferred operator class: `dycore_coupled_step`
- max abs delta: `392.4362662760416 Pa`
- mean abs delta: `104.66851663630047 Pa`
- tolerance: `3e-6 Pa`
- location: `[1, 2]`

Proof: `proofs/f1/m6b6_real_wrf_comparison.json`.

## What This Does Not Prove

This fallback cannot localize the exact WRF RK stage, acoustic substep, or JAX function. It uses linear interpolation between real hourly CPU WRF `wrfout` files because true Fortran savepoint emission was not available in this worktree. Therefore, the exact answer to "bitwise WRF parity at how many RK/acoustic stages?" remains: **not measured**.

## M11.3 Direction

Not measured. This sprint ran current HEAD against the new real-WRF fallback. It did not rerun a pre-M11.3 baseline, and the previous speedup proof remains blocked by nonfinite model state after forecast hour 1.

## Speed / GPU Status

No dycore or runtime model source was edited, so no speed path was intentionally changed. GPU comparison could not be run because `nvidia-smi` cannot communicate with the NVIDIA driver in this environment. AC5 is therefore not satisfied.

## Next Sprint Target

Implement real Fortran hook-body emission for `module_em.F` and `module_small_step_em.F`, preferably on a B6 ideal case with `ideal.exe`. Emit actual end-of-RK-stage, acoustic-substep, and full-timestep savepoints, then rerun this comparator against those files and add a baseline-vs-current comparison.
