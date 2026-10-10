"""Deletion-sensitivity check: each mutant must make >=1 test FAIL by assertion/expected-error (E39/E202).

Run from the worktree root with no real-case arm running (mutates src/, restores via git checkout).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TEST = "tests/v025/restart/test_boundary_coverage.py"
MUTANTS = {
    "K1_cli_coverage_check_removed": ("src/gpuwrf/cli.py",
        "            require_wrfbdy_coverage(input_dir, effective_hours, domain=\"d01\")\n", "            pass\n"),
    "K2_driver_coverage_check_removed": ("src/gpuwrf/integration/nested_pipeline.py",
        "        require_wrfbdy_coverage(input_dir, config.hours, domain=names[0])\n", "        pass\n"),
    "K3_ignore_wrf_boundary_metadata": ("src/gpuwrf/io/wrfbdy_coverage.py",
        '"coverage_seconds": model_s if wrf_s is None else min(model_s, wrf_s),', '"coverage_seconds": model_s,'),
    "K4_ignore_leaf_span": ("src/gpuwrf/io/wrfbdy_coverage.py",
        '"coverage_seconds": model_s if wrf_s is None else min(model_s, wrf_s),', '"coverage_seconds": model_s if wrf_s is None else wrf_s,'),
    "K5_clamp_tolerated_one_record": ("src/gpuwrf/io/wrfbdy_coverage.py",
        'if float(hours) * 3600.0 > info["coverage_seconds"] + 1.0e-6:', 'if float(hours) * 3600.0 > info["coverage_seconds"] + 3600.0 + 1.0e-6:'),
}
env = {**os.environ, "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0"}
results = {}
for name, (path, old, new) in MUTANTS.items():
    original = Path(path).read_bytes()
    source = original.decode()
    assert source.count(old) == 1, (name, source.count(old))
    Path(path).write_text(source.replace(old, new))
    try:
        proc = subprocess.run(["taskset", "-c", "8", "nice", "-n", "19", sys.executable, "-m", "pytest", TEST, "-q",
                               "-p", "no:cacheprovider", "--basetemp", f"<USER_HOME>/wrf_gpu2_lanes/o1-restart/pytest/mutc_{name}"],
                              capture_output=True, text=True, env=env, timeout=1200)
    finally:
        Path(path).write_bytes(original)  # restore exact bytes (never git checkout: it drops uncommitted work)
    out = proc.stdout
    failed = re.findall(r"^FAILED (\S+)", out, re.M)
    errors = re.findall(r"^ERROR (\S+)", out, re.M)
    genuine = ("AssertionError" in out or "assert " in out or "DID NOT RAISE" in out or "Regex pattern did not match" in out
               or "ValueError" in out or "FileExistsError" in out)
    results[name] = {"rc": proc.returncode, "failed": failed, "setup_errors": errors,
                     "killed": bool(failed) and not errors and genuine, "summary": out.strip().splitlines()[-1]}
    print(name, json.dumps(results[name]), flush=True)
assert subprocess.run(["git", "diff", "--quiet", "--", "src"]).returncode == 0, "source not restored"
Path("<USER_HOME>/wrf_gpu2_lanes/o1-restart/mutants_coverage.json").write_text(json.dumps(results, indent=1) + "\n")
print("ALL_KILLED", all(r["killed"] for r in results.values()))
