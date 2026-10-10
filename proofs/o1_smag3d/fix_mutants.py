import subprocess, sys, os
from pathlib import Path
ROOT = Path("<USER_HOME>/src/wrf_gpu2_wt/o1-smag3d")
MUTANTS = [
    ("msfuy_fold_wrong_component", "src/gpuwrf/runtime/les3d_km3.py",
     "u_t=out.ru_tendf / msfuy,", "u_t=out.ru_tendf / a(metrics.msfux)[None],", "folds_like"),
    ("msfvx_fold_dropped", "src/gpuwrf/runtime/les3d_km3.py",
     "v_t=out.rv_tendf * msfvx_inv,", "v_t=out.rv_tendf,", "folds_like"),
    ("moist_sc_else_arm_deleted", "src/gpuwrf/runtime/operational_mode.py",
     "else (dict(rk1_les3d_km3.scalar_sc) if root_scalar_hdiff_active else {})", "else {}", "reaches_the_rk"),
]
env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
out = {}
for name, rel, old, new, sel in MUTANTS:
    path = ROOT / rel
    orig = path.read_text()
    assert orig.count(old) == 1, name
    try:
        path.write_text(orig.replace(old, new))
        r = subprocess.run(["taskset", "-c", "29", "nice", "-n", "19", sys.executable, "-m", "pytest", "-q", "-x",
                            "tests/dynamics/test_les3d_smagorinsky.py", "-k", sel, "--basetemp",
                            "<USER_HOME>/wrf_gpu2_lanes/o1-smag3d/pytest_mut", "-p", "no:cacheprovider"],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
        tail = r.stdout.strip().splitlines()[-1]
        killed = r.returncode != 0 and "AssertionError" in r.stdout and " error" not in tail
        out[name] = (killed, tail)
        print(name, "KILLED" if killed else "SURVIVED/ERROR", tail, flush=True)
    finally:
        path.write_text(orig)
assert all(k for k, _ in out.values())
