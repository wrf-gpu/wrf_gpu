"""Deletion-sensitivity check: each mutant must make >=1 test FAIL by assertion/expected-error (E39/E202).

Run from the worktree root with no real-case arm running (mutates src/, restores via git checkout).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TEST = "tests/v025/restart/test_single_domain_restart.py"
MUTANTS = {
    "M1_cli_single_domain_refusal_restored": ("src/gpuwrf/cli.py",
        "    if restart_requested and max_dom == 1 and not native_single:",
        "    if restart_requested and max_dom == 1:"),
    "M2_hours_back_in_identity": ("src/gpuwrf/runtime/restart_store.py",
        '        "run_start": run_start.isoformat(),\n',
        '        "run_start": run_start.isoformat(), "hours": config.hours,\n'),
    "M3_end_time_check_removed": ("src/gpuwrf/integration/nested_pipeline.py",
        '            restart_report["end_time"] = _restart_end_time(restart_snapshot["driver_state"], config)\n',
        ""),
    "M4_forecast_hours_not_recorded": ("src/gpuwrf/integration/nested_pipeline.py",
        '                        "forecast_hours": config.hours,\n', ""),
    "M5_shortening_allowed": ("src/gpuwrf/integration/nested_pipeline.py",
        "    if requested < saved:\n", "    if False:\n"),
    "M6_extension_without_flag": ("src/gpuwrf/integration/nested_pipeline.py",
        "    if requested > saved and not config.extend_end_time:\n", "    if False:\n"),
    "M7_journal_recover_noop": ("src/gpuwrf/runtime/restart_store.py",
        "        removed = []\n        for path, device, inode in pending:\n",
        "        removed = []\n        pending = []\n        for path, device, inode in pending:\n"),
}
env = {**os.environ, "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0"}
results = {}
for name, (path, old, new) in MUTANTS.items():
    source = Path(path).read_text()
    assert source.count(old) == 1, (name, source.count(old))
    Path(path).write_text(source.replace(old, new))
    try:
        proc = subprocess.run(["taskset", "-c", "8,9", "nice", "-n", "19", sys.executable, "-m", "pytest", TEST, "-q",
                               "-p", "no:cacheprovider", "--basetemp", f"<USER_HOME>/wrf_gpu2_lanes/o1-restart/pytest/mut_{name}"],
                              capture_output=True, text=True, env=env, timeout=1200)
    finally:
        subprocess.run(["git", "checkout", "--", path], check=True)
    out = proc.stdout
    failed = re.findall(r"^FAILED (\S+)", out, re.M)
    errors = re.findall(r"^ERROR (\S+)", out, re.M)
    genuine = ("AssertionError" in out or "assert " in out or "DID NOT RAISE" in out or "Regex pattern did not match" in out
               or "ValueError" in out or "FileExistsError" in out)
    results[name] = {"rc": proc.returncode, "failed": failed, "setup_errors": errors,
                     "killed": bool(failed) and not errors and genuine, "summary": out.strip().splitlines()[-1]}
    print(name, json.dumps(results[name]), flush=True)
assert subprocess.run(["git", "diff", "--quiet", "--", "src"]).returncode == 0, "source not restored"
Path("<USER_HOME>/wrf_gpu2_lanes/o1-restart/mutants.json").write_text(json.dumps(results, indent=1) + "\n")
print("ALL_KILLED", all(r["killed"] for r in results.values()))
