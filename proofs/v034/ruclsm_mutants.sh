#!/bin/bash
# Mutant check for physics/ruclsm.py: each mutant must make tests/test_v034_ruclsm.py fail
# with an AssertionError (E202), run on a scratch copy of src (core 9, sequential).
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
W=<USER_HOME>/src/wrf_gpu2_wt/o1-ruc; M=<USER_HOME>/wrf_gpu2_lanes/o1-ruc/mutants
run() {
  name=$1; old=$2; new=$3; sel=$4
  rm -rf $M/$name && mkdir -p $M/$name && cp -r $W/src $M/$name/src && ln -s $W/data $M/$name/data  # E128
  python3 - "$M/$name/src/gpuwrf/physics/ruclsm.py" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read(); assert s.count(old) == 1, (old, s.count(old)); open(p, "w").write(s.replace(old, new))
PY
  # pyproject pythonpath=["src"] would shadow PYTHONPATH (E93): override the ini option itself
  (cd $W && PYTHONPATH=$M/$name/src JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 RUC_EXPECT_SRC=$M/$name/src \
     taskset -c 9 nice -n 19 timeout 1800 \
     python -m pytest -q -s -p no:cacheprovider -o pythonpath=$M/$name/src \
     --basetemp <USER_HOME>/wrf_gpu2_lanes/o1-ruc/pytest/$name \
     tests/test_v034_ruclsm.py -k "$sel" > $M/$name.log 2>&1)
  rc=$?; used=$(grep -c "RUC_SRC_OK" $M/$name.log)
  echo "$name rc=$rc src_ok=$used failed=$(grep -c '^FAILED' $M/$name.log) asserts=$(grep -c 'AssertionError' $M/$name.log) errors=$(grep -c '^ERROR' $M/$name.log) $(tail -1 $M/$name.log)"
  rm -rf $M/$name/src
}
run M1_no_melt_second_pass 'p2 = energy_pass(snoh_m, beta_m, melt, snwe_m)' 'p2 = p1' 'fp64 and (10 or 25)'
run M2_sfcevp_single_add '    sfcevp_l = sfcevp_l + o["eeta"] * delt  # WRF adds qfx*dt a second time (:1116)' '' 'fp64 and (1 or 16)'
run M3_thin_snow_ilnb2 'ilnb = W(one_l, 1, W(two_l, 2, 0))' 'ilnb = W(one_l, 1, 2)' 'fp64 and (7 or 12 or 14)'
run M4_no_graupel_density 'rhonewsn_f * snowrat + rhonewgr * grauprat + rhonewsn_f * icerat' 'rhonewsn_f * snowrat + rhonewsn_f * grauprat + rhonewsn_f * icerat' 'fp64 and 11'
run M5_freeze_keep_riw 'liq_k = jnp.maximum(0.0, soilmois - ice_k * 900.0 * 1.0e-3)' 'liq_k = jnp.maximum(0.0, soilmois - ice_k * riw)' 'fp64 and (6 or 7 or 9 or 13 or 23)'
# rv-ruc mutants (E154): survive the 35-regime set by construction, must die on regimes 37-45
run M6_rv_bottommelt_cap '               jnp.minimum(smeltg, 5.8e-9), smeltg)' '               jnp.minimum(smeltg, 6.2e-9), smeltg)' 'fp64 and (42 or 43 or 44 or 45)'
run M7_rv_conv_warm_liquid 'W(cold, 0.0, jnp.maximum(0.0, conv)))' 'W(cold, 0.0, 0.5 * jnp.maximum(0.0, conv)))' 'fp64 and (37 or 40 or 41)'
