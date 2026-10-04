"""D5/D6 CPU-WRF field gate with complete hours, finite fields and frozen bounds."""
import argparse
from datetime import datetime, timedelta
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
CPU = Path("<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/"
           "alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case")
MANIFEST = ROOT/"proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json"


def gate_failures(report, manifest, hours):
    failures = []
    pairing = report["pairing"]
    expected = set(range(hours+1))
    if set(pairing["common_leads_h"]) != expected or pairing["paired_file_count"] != hours+1:
        failures.append("hour coverage")
    initial = datetime.fromisoformat(pairing["init_time_utc"])
    for pair in pairing["pairs"]:
        if datetime.fromisoformat(pair["valid_time_utc"]) != initial+timedelta(hours=pair["lead_h"]):
            failures.append(f"non-hourly timestamp: {pair['valid_time_utc']}")
    fields = report["field_summaries"]
    required = manifest["score_policy"]["mandatory_presence_fields_for_final_runs"]
    for name in required:
        data = fields.get(name)
        if data is None or data.get("compared_lead_count") != hours+1:
            failures.append(f"missing/incompatible field or hour: {name}")
        elif data.get("missing_leads") or data.get("incompatible_leads"):
            failures.append(f"field coverage: {name}")
    for name, data in fields.items():
        if "overall" not in data:
            continue
        for stats in [data["overall"], *data["by_lead"]]:
            if stats.get("finite_pair_fraction") != 1.0:
                failures.append(f"nonfinite field: {name}")
        limits = manifest["fields"].get(name, {})
        if limits.get("gate") in {"hard_release_gate", "static_exactness"}:
            if data["tolerance_result"].get("pass") is not True:
                failures.append(f"pooled bound: {name}")
            for stats in data["by_lead"]:
                if stats["tolerance_result"].get("pass") is not True:
                    failures.append(f"hour {stats['lead_h']} bound: {name}")
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-dir", type=Path, required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--hours", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("bcore_grid_compare", ROOT/"scripts/compare_wrfout_grid.py")
    compare = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = compare
    spec.loader.exec_module(compare)
    manifest = json.loads(MANIFEST.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for domain in ("d01", "d02"):
        report = compare.build_report(argparse.Namespace(
            cpu_dir=CPU, gpu_dir=a.gpu_dir, domain=domain, init=a.init,
            min_lead=0, max_lead=a.hours, vars=None, tolerance_json=MANIFEST,
            no_spatial_splits=False, boundary_width=5, progress=25))
        failures = gate_failures(report, manifest, a.hours)
        report["b_core_gate"] = dict(passed=not failures, failures=failures,
            authority="D5 CPU-WRF output; D6 frozen manifest; every hour and required field")
        compare.write_json(a.out/(domain+".json"), report)
        compare.write_markdown(report, a.out/(domain+".md"))
        results[domain] = report["b_core_gate"]
    (a.out/"gate.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results),flush=True)
    return 0 if all(r["passed"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
