"""Sanitized nsys device-op/time census for per-call component captures."""
import argparse
import json
from pathlib import Path
import sqlite3


def census(path, calls):
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
    result = {"source": str(path), "calls": calls, "kinds": {}}
    for kind in ("KERNEL", "MEMCPY", "MEMSET"):
        table = f"CUPTI_ACTIVITY_KIND_{kind}"
        if table not in tables:
            result["kinds"][kind] = dict(count=0, device_ms=0.)
            continue
        count, ns = connection.execute(f"select count(*),sum(end-start) from {table}").fetchone()
        result["kinds"][kind] = dict(count=count, device_ms=(ns or 0)/1.e6,
                                     ops_per_call=count/calls, device_ms_per_call=(ns or 0)/1.e6/calls)
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        result["memcpy_copy_kinds"] = [dict(copy_kind=row[0], count=row[1], device_ms=row[2]/1.e6)
            for row in connection.execute("select copyKind,count(*),sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind")]
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        columns = {row[1] for row in connection.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        name_column = "demangledName" if "demangledName" in columns else "shortName"
        result["kernels"] = [dict(name=row[0], count=row[1], device_ms=row[2]/1.e6)
            for row in connection.execute(
                f"select s.value,count(*),sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k "
                f"join StringIds s on k.{name_column}=s.id group by s.value order by sum(k.end-k.start) desc")]
    result["device_ops_per_call"] = sum(row["count"] for row in result["kinds"].values())/calls
    result["device_ms_per_call"] = sum(row["device_ms"] for row in result["kinds"].values())/calls
    connection.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for db in sorted((args.directory / "raw").glob("*.sqlite")):
        bench = json.loads((args.directory / f"{db.stem}.json").read_text())
        reports[db.stem] = census(db, bench["calls"])
    (args.directory / "device_metrics.json").write_text(json.dumps(reports, indent=2)+"\n")
    for name, report in reports.items():
        print(name, report["device_ops_per_call"], report["device_ms_per_call"])


if __name__ == "__main__":
    main()
