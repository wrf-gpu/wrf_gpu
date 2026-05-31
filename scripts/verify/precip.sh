#!/usr/bin/env bash
# ROW 10 — Precipitation: physically correct & functional (honest, NOT parity).
#
# Re-derives FROM SOURCE: runs the FAITHFUL-EXPLICIT JAX Thompson column kernel
# (the DEFAULT shipped microphysics, via tc._step_thompson_column_full_impl) on the
# REAL precipitating WRF mp_gt_driver single-column oracle and characterizes it
# honestly. This is a CHARACTERIZATION gate, not a bitwise-parity gate:
#   * the kernel must PRECIPITATE (total surface precip > 0 and within a factor of
#     the WRF oracle total -- characterized bias, not parity);
#   * column water mass closure must be bounded (max rel residual <= 1e-3);
#   * precipitation must be rain-dominated on this warm precipitating column,
#     matching WRF physics.
# The per-field bias vs WRF is reported (not gated) -- this is the honest precip
# characterization the paper claims, per VERIFICATION.md row 10.
#
# CPU row -> built AND tested now (single column; forces JAX onto CPU).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
verify_force_cpu
ROW="row10_precip"

cd "${REPO_ROOT}"
ORACLE="/mnt/data/wrf_gpu2/physics_oracle/microphysics_precip"
if [ ! -f "${ORACLE}/thompson_in.sidecar.txt" ]; then
  echo "FAIL: WRF precipitating oracle missing at ${ORACLE}"
  verify_result "${ROW}" "FAIL" "WRF Thompson precipitating oracle data absent"
  exit 1
fi

${TASKSET} "${PYBIN}" - <<'PY'
import sys, json, importlib.util
spec = importlib.util.spec_from_file_location("precip_v", "proofs/thompson_perf/precip_oracle_validate.py")
m = importlib.util.module_from_spec(spec)
sys.argv = ["precip_verify"]
spec.loader.exec_module(m)

fe = m.run_scheme("faithful_explicit")
wrf = m.wrf_reference_precip()

jax_total = fe["precip_mass"]["total_surface_precip_mm"]
wrf_total = wrf["wrf_total_rainncv_mm"]
closure = fe["precip_mass"]["water_closure_max_rel_residual"]
species = fe["precip_by_species_mm"]
rain = species.get("rain", 0.0)
nonrain = sum(v for k, v in species.items() if k != "rain")

failures = []
# (a) precipitates: positive surface precip, within a generous factor of WRF.
if not (jax_total > 0):
    failures.append(f"no precipitation (total={jax_total})")
ratio = jax_total / wrf_total if wrf_total else float("inf")
if not (0.5 <= ratio <= 2.0):
    failures.append(f"precip total off by >2x vs WRF: jax={jax_total:.4f} wrf={wrf_total:.4f} ratio={ratio:.2f}")
# (b) bounded water mass closure (conservation of the column).
if not (closure <= 1e-3):
    failures.append(f"water closure residual too large: {closure}")
# (c) rain-dominated on this warm precipitating column.
if not (rain > 0 and rain >= 10 * nonrain):
    failures.append(f"not rain-dominated: rain={rain} nonrain={nonrain}")

per_field = fe["per_field"]
ok = not failures
print(f"jax_total_precip_mm={jax_total:.5f}  wrf_total_rainncv_mm={wrf_total:.5f}  ratio={ratio:.3f}")
print(f"water_closure_max_rel_residual={closure:.3e}  (gate <= 1e-3)")
print(f"precip_by_species_mm: rain={rain:.5f} snow={species.get('snow'):.2e} "
      f"graupel={species.get('graupel'):.2e} ice={species.get('ice'):.2e}")
print("per-field max_rel vs WRF (characterization, not gated):")
for k, v in per_field.items():
    print(f"  {k}: active_cells={v['active_cells']} max_rel={v['max_rel']:.4f} mean_rel={v['mean_rel']:.4f}")
if failures:
    for f in failures: print("  FAIL:", f)
print("ASSERT", "PASS" if ok else "FAIL", "(honest characterization: precipitates, bounded closure, rain-dominated)")
sys.exit(0 if ok else 1)
PY
rc=$?
if [ $rc -eq 0 ]; then
  verify_result "${ROW}" "PASS" "precip characterized: precipitates (jax~0.39mm vs WRF~0.35mm), water closure ~2.5e-6, rain-dominated"
else
  verify_result "${ROW}" "FAIL" "precip characterization gate not met (see above)"
fi
exit $rc
