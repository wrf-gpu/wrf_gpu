"""CPU-only parser for paired CENSUS NVTX ranges and node-traced GPU durations."""
import argparse
import json
import heapq
from pathlib import Path
import re
import sqlite3
import statistics


def parse_profile(database: Path):
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    labels = connection.execute("""
        SELECT n.start, n.end, COALESCE(n.text, s.value)
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
        WHERE COALESCE(n.text,s.value) LIKE '%CENSUS:tree:%' AND n.end>n.start
        ORDER BY n.start
    """).fetchall()
    marker_method = "explicit CENSUS NVTX ranges"
    if not labels:
        # Python TraceMe can be absent while the GPU plugin still records modules.
        # The frozen runner blocks all results before the next invocation and
        # captures exactly 30 pairs: off/on, on/off, repeated. Validate every ID.
        modules = connection.execute("""
            SELECT n.start,n.end,COALESCE(n.text,s.value)
            FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
            WHERE COALESCE(n.text,s.value) LIKE 'XlaModule:%hlo_module=jit_ordinary,%'
              AND n.end>n.start ORDER BY n.start
        """).fetchall()
        if len(modules) != 60:
            raise ValueError("missing CENSUS markers and not exactly 60 ordinary module calls")
        ids = [re.search(r"program_id=(\d+)", label).group(1) for _, _, label in modules]
        off_id, on_id = ids[:2]
        expected = [value for pair in range(30)
                    for value in ((off_id, on_id) if pair % 2 == 0 else (on_id, off_id))]
        if off_id == on_id or ids != expected:
            raise ValueError("ordinary module IDs do not match predeclared alternating pairs")
        names = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        last_end = max(connection.execute(f"SELECT MAX(end) FROM {table}").fetchone()[0] or 0
                       for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY",
                                     "CUPTI_ACTIVITY_KIND_MEMSET") if table in names)
        labels = []
        for index, (start, end, _) in enumerate(modules):
            stop = modules[index + 1][0] if index + 1 < len(modules) else max(end, last_end + 1)
            if end > stop:
                raise ValueError("overlapping ordinary module calls")
            mode = "off" if ids[index] == off_id else "on"
            labels.append((start, stop, f"CENSUS:tree:{mode}:{index // 2}"))
        marker_method = "XLA module starts; validated 30-pair executable-ID pattern and synchronous replay"
    rows = []
    for start, end, label in labels:
        match = re.search(r"CENSUS:tree:(off|on):(\d+)", label)
        if match:
            rows.append(dict(mode=match[1], iteration=int(match[2]), start=start, end=end,
                             kernels=0, kernel_sum_ns=0, device_union_ns=0, last_end=start,
                             copies_by_kind={}))
    if not rows:
        raise ValueError("no paired CENSUS ranges")
    index = 0
    for start, end in connection.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"):
        while index < len(rows) and start >= rows[index]["end"]:
            index += 1
        if index == len(rows):
            break
        row = rows[index]
        if start < row["start"] or end > row["end"]:
            raise ValueError("kernel crosses measured range boundary")
        row["kernels"] += 1
        row["kernel_sum_ns"] += end-start
        row["device_union_ns"] += max(0, end-max(start, row["last_end"]))
        row["last_end"] = max(row["last_end"], end)
    index = 0
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for start, end, kind in connection.execute(
                "SELECT start,end,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"):
            while index < len(rows) and start >= rows[index]["end"]:
                index += 1
            if index == len(rows):
                break
            row = rows[index]
            if row["start"] <= start and end <= row["end"]:
                key = str(kind)
                row["copies_by_kind"][key] = row["copies_by_kind"].get(key, 0)+1
    # Disclose full device service as well as the preregistered kernel metrics.
    # Memcpy/memset are real device work; retain the kernel gate separately.
    for row in rows:
        row.update(all_device_sum_ns=0, all_device_union_ns=0,
                   all_last_end=row["start"], device_ops_by_kind={})
    def operations(table, kind):
        for start, end in connection.execute(f"SELECT start,end FROM {table} ORDER BY start"):
            yield start, end, kind
    streams = [operations("CUPTI_ACTIVITY_KIND_KERNEL", "kernel")]
    streams += [operations(table, kind) for table, kind in (
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")) if table in tables]
    index = 0
    for start, end, kind in heapq.merge(*streams):
        while index < len(rows) and start >= rows[index]["end"]:
            index += 1
        if index == len(rows):
            break
        row = rows[index]
        if start < row["start"] or end > row["end"]:
            raise ValueError("device operation crosses measured range boundary")
        row["all_device_sum_ns"] += end-start
        row["all_device_union_ns"] += max(0, end-max(start, row["all_last_end"]))
        row["all_last_end"] = max(row["all_last_end"], end)
        row["device_ops_by_kind"][kind] = row["device_ops_by_kind"].get(kind, 0)+1
    connection.close()
    by_iteration = {}
    for row in rows:
        row.pop("last_end")
        row.pop("all_last_end")
        if row["kernels"] == 0:
            raise ValueError("range has no device kernels")
        pair = by_iteration.setdefault(row["iteration"], {})
        if row["mode"] in pair:
            raise ValueError("duplicate paired range")
        pair[row["mode"]] = row
    if any(set(pair) != {"on", "off"} for pair in by_iteration.values()):
        raise ValueError("unpaired measurement")
    metrics = {}
    for metric in ("kernel_sum_ns", "device_union_ns", "all_device_sum_ns", "all_device_union_ns"):
        off = [pair["off"][metric] for pair in by_iteration.values()]
        delta = [pair["on"][metric]-pair["off"][metric] for pair in by_iteration.values()]
        from scipy.stats import t
        stderr = statistics.stdev(delta) / len(delta)**0.5 if len(delta)>1 else float("inf")
        upper = statistics.mean(delta) + t.ppf(0.975, len(delta)-1)*stderr
        baseline = statistics.mean(off)
        metrics[metric] = dict(off_mean_ns=baseline, delta_mean_ns=statistics.mean(delta),
            paired_overhead_pct=100*statistics.mean(delta)/baseline,
            ci95_upper_pct=100*upper/baseline, pairs=len(delta))
    return {"scope": "paired real PROD d01 ordinary own-step device attribution; no wall speed claim",
            "marker_method": marker_method,
            "metrics": metrics, "ranges": rows,
            "gate_ordinary_lt_1pct": all(metrics[key]["ci95_upper_pct"]<1
                                        for key in ("kernel_sum_ns", "device_union_ns")),
            "gate_all_device_lt_1pct": all(metrics[key]["ci95_upper_pct"]<1
                                          for key in ("all_device_sum_ns", "all_device_union_ns"))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = parse_profile(args.database)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
