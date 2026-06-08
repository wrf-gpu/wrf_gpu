# GWDO oracle (pristine WRF `bl_gwdo_run`)

`oracle_driver.F90` drives the pristine WRF v4 orographic gravity-wave-drag
kernel `bl_gwdo_run` (`phys/physics_mmm/bl_gwdo.F90`) on a controlled 3-column
batch and dumps inputs + outputs to `oracle_out.txt`. The JAX port
(`src/gpuwrf/physics/gwd_gwdo.py`) is then run on the IDENTICAL inputs by
`proofs/gwd/compare_oracle.py`, which writes `proofs/gwd/gwdo_oracle_gate.json`.

## Build + run (reproduce the oracle)

The pristine WRF `kind_phys` is single precision (`selected_real_kind(6)`), so
the oracle runs fp32. The pristine `.mod` files are GFORTRAN module version 15
(incompatible with the conda gfortran 14.3.0 here), so recompile the two
self-contained sources from the pristine tree:

```bash
export PATH=~/miniconda3/envs/wrfbuild/bin:$PATH
cp ~/src/wrf_pristine/WRF/phys/ccpp_kind_types.F            ccpp_kind_types.F90
cp ~/src/wrf_pristine/WRF/phys/physics_mmm/bl_gwdo.F90      bl_gwdo.F90
gfortran -O2 -ffree-line-length-none -c ccpp_kind_types.F90 -o ccpp_kind_types.o
gfortran -O2 -ffree-line-length-none -c bl_gwdo.F90        -o bl_gwdo.o
gfortran -O2 -ffree-line-length-none oracle_driver.F90 ccpp_kind_types.o bl_gwdo.o -o oracle_driver
./oracle_driver           # writes oracle_out.txt (errflg=0)
```

## Compare the JAX port

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python proofs/gwd/compare_oracle.py proofs/gwd/oracle/oracle_out.txt
```

## Result (2026-06-07)

VERDICT: PASS. The JAX port matches the pristine fp32 oracle to within fp32
round-off on all three columns:

| col | terrain          | rublten rel err | dusfcg (JAX / WRF)      |
|-----|------------------|-----------------|-------------------------|
| 1   | flat (var=0)     | exact 0         | 0 / 0 (ldrag short-cut) |
| 2   | westerly mtn     | 6.1e-6          | 17.95801 / 17.95805     |
| 3   | SW-flow mtn      | 1.5e-5          | 9.40927 / 9.40930       |

The JAX kernel runs fp64 internally; the residuals are the oracle's fp32
quantization, i.e. the port is bit-faithful within fp32.
