"""Read-only inventory and inclusive CPU timing for the WN3 release cases."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from netCDF4 import Dataset, chartostring
from gpuwrf.io.gen2_accessor import parse_namelist

CASES = ("20260227_18z_a1", "20260502_18z_a1", "20260614_18z_a1")
ROOT = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen")
PROD = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")
MAIN = re.compile(r"Timing for main: time (\S+) on domain\s+(\d+):\s+([\d.]+) elapsed seconds")
WRITE = re.compile(r"Timing for Writing (wrfout_d\d+_(\S+)) for domain\s+(\d+):\s+([\d.]+) elapsed seconds")


def stamp(value):
    return datetime.strptime(value, "%Y-%m-%d_%H:%M:%S")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path):
    result = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if path.name.startswith(("wrfinput", "wrflowinp")):
        with Dataset(path) as ds:
            result["dimensions"] = {k: len(v) for k, v in ds.dimensions.items()}
            result["variables"] = list(ds.variables)
            result["times"] = chartostring(ds["Times"][:]).tolist()
    return result


def inventory(case, output):
    source = ROOT / f"wg_{case}" / "run/run"
    links = output.parent.parent / "cases" / case
    links.mkdir(parents=True, exist_ok=True)
    inputs = [source / "namelist.input", source / "wrfbdy_d01"]
    inputs += [source / f"{prefix}_d{d:02d}" for prefix in ("wrfinput", "wrflowinp") for d in range(1, 4)]
    files = {}
    for path in inputs:
        target = links / path.name
        if target.is_symlink():
            assert target.resolve() == path.resolve(), target
        elif target.exists():
            raise FileExistsError(target)
        else:
            target.symlink_to(path)
        files[path.name] = describe(path)
    nml = parse_namelist(source / "namelist.input")
    prod = parse_namelist(PROD / "namelist.input")
    diff = {f"{group}.{key}": {"PROD": prod.get(group, {}).get(key), "WN3": value}
            for group, values in nml.items() for key, value in values.items()
            if prod.get(group, {}).get(key) != value}
    history = {f"d{d:02d}": sorted(source.glob(f"wrfout_d{d:02d}_*")) for d in range(1, 4)}
    assert all(len(paths) == 73 for paths in history.values())
    first = stamp(history["d01"][0].name[11:])
    final = stamp(history["d01"][24].name[11:])
    assert (final - first).total_seconds() == 86400
    steps = {d: [] for d in range(1, 4)}
    writes = {d: [] for d in range(1, 4)}
    log = source / "rsl.out.0000"
    for line in log.read_text().splitlines():
        m = MAIN.search(line)
        if m:
            t, d, seconds = m.groups()
            steps[int(d)].append((stamp(t), float(seconds)))
        m = WRITE.search(line)
        if m:
            _, t, d, seconds = m.groups()
            writes[int(d)].append((stamp(t), float(seconds)))
    frames = []
    previous = first
    for frame in history["d01"][1:25]:
        end = stamp(frame.name[11:])
        seconds = sum(s for t, s in steps[1] if previous < t <= end)
        frames.append({"time": end.isoformat(), "inclusive_root_seconds": seconds,
                       "root_steps": sum(previous < t <= end for t, _ in steps[1])})
        previous = end
    root = sum(f["inclusive_root_seconds"] for f in frames)
    initial_io = sum(s for rows in writes.values() for t, s in rows if t == first)
    return {"case": case, "source": str(source), "input_dir": str(links), "inputs": files,
            "namelist_diff_vs_PROD": diff, "rsl_sha256": sha256(log),
            "history": {d: [{"name": p.name, "bytes": p.stat().st_size} for p in paths[:25]]
                        for d, paths in history.items()},
            "all_history_bytes": sum(p.stat().st_size for paths in history.values() for p in paths),
            "cpu_24h": {"frames": frames, "inclusive_root_seconds": root,
                        "initial_history_write_seconds": initial_io,
                        "seconds_per_case_sim_hour": root / 24,
                        "seconds_per_domain_frame": root / 72,
                        "initial_io_inclusive_seconds_per_case_sim_hour": (root + initial_io) / 24,
                        "own_domain_inclusive_seconds": {str(d): sum(s for t, s in rows if first < t <= final)
                                                         for d, rows in steps.items()},
                        "writing_seconds_already_in_root": {str(d): sum(s for t, s in rows if first < t <= final)
                                                            for d, rows in writes.items()},
                        "method": "d01 main includes children and in-step writes (module_integrate.F); excludes startup before first root timer"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = [inventory(case, args.output) for case in CASES]
    totals = [case["cpu_24h"]["inclusive_root_seconds"] for case in cases]
    result = {"cases": cases, "cpu_3x4_reference": {
        "status": "inferred replay of measured per-case 4-rank rates; cases were not simultaneous",
        "seconds_per_case_sim_hour": max(totals) / 72,
        "equal_work_makespan_seconds": max(totals), "completed_case_hours": 72,
        "startup_included": False}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in cases:
        print(case["case"], case["cpu_24h"]["seconds_per_case_sim_hour"], "s/case-sim-hour")
    print("3x4 replay [I]", result["cpu_3x4_reference"]["seconds_per_case_sim_hour"], "s/case-sim-hour")


if __name__ == "__main__":
    main()
