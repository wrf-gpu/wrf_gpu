"""Attach validated same-work/counter manifests alongside bench.json (CPU only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def summarize(payload: dict) -> dict:
    """Per-domain F2 counts for the release receipt.

    Under strict guards (the default) the guard rows count would-repair candidates
    (nothing is applied, E87); with GPUWRF_STRICT_GUARDS=0 they are applied repairs and
    guard_water reports the water they created or removed.
    """
    summary = {}
    for domain, record in payload["domains"].items():
        strict = bool(record["resolved_namelist"].get("disable_guards", False))
        guards = record["guards"].values()
        repaired = sum(row["repaired"] for row in guards)
        f2 = record.get("f2") or {}
        water = [row for stage in (f2.get("guard_water") or {}).values() for row in stage.values()]
        summary[domain] = {
            "strict_guards": strict,
            "nonfinite": sum(row["nonfinite"] for row in guards),
            "out_of_range": sum(row["out_of_range"] for row in guards),
            "applied_repairs": 0 if strict else repaired,
            "would_repair_candidates": repaired if strict else 0,
            "guard_water_added_kg": sum(row["added_kg"] for row in water),
            "guard_water_removed_kg": sum(row["removed_kg"] for row in water),
            "events": [{key: event[key] for key in ("phase", "field", "event", "count", "first_step", "first_value")}
                       for event in f2.get("events", [])],
        }
    return summary


def attach_bench(bench_path: Path) -> dict:
    import jsonschema

    bench = json.loads(bench_path.read_text())
    schema = json.loads(Path(__file__).with_name("census_schema.json").read_text())
    manifests, summaries = {}, {}
    for name, arm in bench["arms"].items():
        source = Path(arm["receipt"]).parent / "wrfout" / "census.json"
        if not source.exists():
            if str((arm.get("env") or {}).get("GPUWRF_CENSUS", "0")).lower() in {"1", "true", "yes", "on"}:
                raise ValueError(f"{name}: census-enabled arm has no manifest at {source}")
            continue
        payload = json.loads(source.read_text())
        jsonschema.validate(payload, schema)
        for domain, record in payload["domains"].items():
            if record["actual_work"]["steps"] != record["own_steps"]:
                raise ValueError(f"{name}/{domain}: device step count differs from scheduler own_steps")
        destination = bench_path.parent / f"{name}.census.json"
        shutil.copyfile(source, destination)
        manifests[name] = str(destination)
        arm["census_manifest"] = str(destination)
        summaries[name] = arm["census_summary"] = summarize(payload)
    bench["census_manifests"] = manifests
    bench["census_summary"] = summaries
    pending = bench_path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(bench, indent=1) + "\n")
    pending.replace(bench_path)
    return manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach-bench", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(attach_bench(args.attach_bench)))
    except Exception:
        # A failed manifest must invalidate the benchmark JSON as well as its exit code.
        bench = json.loads(args.attach_bench.read_text())
        bench["ok"] = False
        bench["census_validation"] = "FAIL"
        args.attach_bench.write_text(json.dumps(bench, indent=1) + "\n")
        raise


if __name__ == "__main__":
    main()
