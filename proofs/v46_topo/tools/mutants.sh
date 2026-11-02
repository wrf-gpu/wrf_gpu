#!/usr/bin/env bash
# v46-topo TS02: deletion-sensitivity mutants for test_shadow_ray_loop_structure.py (CPU, symlink-farm trees).
set -u
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
T=<DATA_ROOT>/wrf_gpu2/worktrees/v45-ekialg; M=<USER_HOME>/wrf_gpu2_lanes/v46-topo/mut
TEST=tests/v025/w44_topo/test_shadow_ray_loop_structure.py
rm -rf -- "$M"; mkdir -p "$M"
mk() {  # name file python-edit
  local d=$M/$1; mkdir -p "$d/tests/v025/w44_topo"; cp -as "$T/src" "$d/src"; cp "$T/pyproject.toml" "$d/"
  cp "$T/tests/conftest.py" "$d/tests/"; mkdir -p "$d/scripts"; cp -as "$T/scripts/v025" "$d/scripts/v025"; cp "$T/tests/v025/conftest.py" "$d/tests/v025/"; cp "$T/$TEST" "$d/$TEST"
  rm "$d/$2"; python3 -c "import sys;p,o,n,_=sys.argv[1:];s=open(p).read();assert s.count(o)==1,o;open(sys.argv[4],'w').write(s.replace(o,n))" "$T/$2" "$3" "$4" "$d/$2"
}
mk M1_ref_python_unroll src/gpuwrf/physics/topographic_shading.py \
 "    mask, height, _ = jax.lax.fori_loop(1, steps + 1, scan, (imported, height, daylight))" \
 "    carry = (imported, height, daylight)
    for _step in range(1, steps + 1):
        carry = scan(_step, carry)
    mask, height, _ = carry"
mk M2_ref_scan_unroll src/gpuwrf/physics/topographic_shading.py \
 "jax.lax.fori_loop(1, steps + 1, scan, (imported, height, daylight))" \
 "jax.lax.fori_loop(1, steps + 1, scan, (imported, height, daylight), unroll=True)"
mk M3_native_python_unroll src/gpuwrf/kernels/topographic_shadow.py \
 "    _, mask, height, _ = jax.lax.while_loop(keep, scan,
        (jnp.int32(1), imported, height, valid & daylight))" \
 "    carry = (jnp.int32(1), imported, height, valid & daylight)
    for _step in range(steps):
        carry = scan(carry)
    _, mask, height, _ = carry"
mk M4_prod_calls_legacy src/gpuwrf/coupling/physics_couplers.py \
 "            previous_height_m=old_height,
        )" \
 "            previous_height_m=old_height,
        )
        _legacy = _wrf_topographic_shadow_mask"
for d in "$M"/M*; do
  ( cd "$d" && JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 taskset -c 12,28 nice -n 19 timeout 600 \
    <USER_HOME>/miniconda3/bin/python3 -m pytest -q -p no:cacheprovider --basetemp <USER_HOME>/wrf_gpu2_lanes/v46-topo/pytest_mut "$TEST" > "$d.log" 2>&1 ); rc=$?
  echo "$(basename "$d") rc=$rc $(grep -cE '^E +(AssertionError|assert)' "$d.log") assertion-lines; errors=$(grep -c ' ERROR ' "$d.log"); $(tail -1 "$d.log")"
done
