#!/usr/bin/env bash
set -o pipefail
H=<USER_HOME>/wrf_gpu2_lanes/o1-nssl/cleanup_dev/adv; cd $H || exit 1
python3 gen_inputs.py || exit 2
source <USER_HOME>/miniconda3/etc/profile.d/conda.sh; conda activate wrfbuild 2>/dev/null
for MODE in fp32 fp64; do
  PROMOTE=""; [ $MODE = fp64 ] && PROMOTE="-fdefault-real-8 -fdefault-double-8"
  FF="-O2 -ffree-form -ffree-line-length-none -ffpe-summary=none -cpp $PROMOTE"
  B=$H/build_$MODE; rm -rf $B; mkdir -p $B; cd $B || exit 3
  cp <USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_nssl_2mom.F module_mp_nssl_2mom.F
  # visibility only: make the four routines callable from the harness
  sed -n '192p' module_mp_nssl_2mom.F | grep -q '^  public nssl_2mom_init$' || { echo ANCHOR; exit 4; }
  sed -i '192a\  public calcnfromq, smallvalues, radardd02, calc_eff_radius' module_mp_nssl_2mom.F
  cp <USER_HOME>/src/wrf_gpu2_wt/o1-nssl/proofs/v034/oracle/nssl2mom/nssl2mom_stub_modules.f90 $H/cleanup_adv.F90 $H/adv_in.txt .
  : > namelist.input
  taskset -c 25 nice -n 19 gfortran $FF -c nssl2mom_stub_modules.f90 || exit 5
  taskset -c 25 nice -n 19 gfortran $FF -c module_mp_nssl_2mom.F || exit 6
  taskset -c 25 nice -n 19 gfortran $FF -c cleanup_adv.F90 || exit 7
  taskset -c 25 nice -n 19 gfortran $FF -o adv *.o || exit 8
  taskset -c 25 ./adv > out.txt || exit 9
  cd $H
done
echo ADV_DONE
