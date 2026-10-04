"""S0 post-session (CPU): graph-inclusive executed-op census from a node-level nsys sqlite export.
Bench-owned copy of the sprint tool; adds class_op_detail (per-class kernels/HtoD/DtoH/DtoD per step)
and device_span_s (span of the same device-op set as device_union_s, for a consistent idle denominator).

Input: `nsys export --type sqlite` of G1d (captured with --cuda-graph-trace=node, capture range =
root segment 2). Every executed kernel (stream launch OR graph node) and every memcpy/memset
row is one device op. Ops are attributed to the harness `S0:*` NVTX ranges (TraceAnnotation ->
NVTX "TSL:" ranges) through their launching runtime/driver API call: same thread and API start
inside the range; any-thread time containment is the fallback and is counted separately.
Radiation own-steps are those with step % cadence == 0 (d01 33, d02 100 on the frozen case).
Usage: python3 s0_nsys_census.py <export.sqlite> <out.json> [--cadence d01=33,d02=100]
"""
from __future__ import annotations

import bisect
import collections
import json
import re
import sqlite3
import statistics
import sys


def _q(c, sql, *a):
    try:
        return c.execute(sql, a).fetchall()
    except sqlite3.OperationalError:
        return []


def union_s(iv):
    iv = sorted(iv)
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None:
            cs, ce = s, e
        elif s > ce:
            tot += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    if cs is not None:
        tot += ce - cs
    return tot / 1e9


def main() -> int:
    db, out = sys.argv[1], sys.argv[2]
    cad = {"d01": 33, "d02": 100}
    if "--cadence" in sys.argv:
        for kv in sys.argv[sys.argv.index("--cadence") + 1].split(","):
            k, v = kv.split("=")
            cad[k] = int(v)
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    strings = dict(_q(c, "select id, value from StringIds"))
    # NVTX ranges (text may be inline or via textId)
    ranges = []
    for start, end, text, tid_text, gtid in _q(
        c, "select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null"
    ):
        t = text if text is not None else strings.get(tid_text, "")
        if t and "S0:" in t:
            ranges.append((int(start), int(end), t[t.index("S0:"):], gtid))
    ranges.sort()
    # API calls: correlationId -> (start, globalTid)
    api = {}
    for start, corr, gtid in _q(c, "select start, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
        api[corr] = (int(start), gtid)
    by_tid = collections.defaultdict(list)
    for r in ranges:
        by_tid[r[3]].append(r)
    starts_all = [r[0] for r in ranges]

    def owner(corr):
        a = api.get(corr)
        if a is None:
            return None, "no_api"
        t0, gtid = a
        best = None
        for r in by_tid.get(gtid, ()):
            if r[0] <= t0 <= r[1] and (best is None or r[0] >= best[0]):
                best = r
        if best is not None:
            return best, "thread"
        i = bisect.bisect_right(starts_all, t0) - 1
        while i >= 0:
            r = ranges[i]
            if r[0] <= t0 <= r[1] and not r[2].startswith("S0:segment"):
                return r, "time"
            i -= 1
        return None, "unattributed"

    ops = collections.defaultdict(lambda: {"kernels": 0, "graph_node_kernels": 0, "memcpy": {}, "memset": 0,
                                           "iv": []})
    how = collections.Counter()
    for start, end, corr, gnode in _q(c, "select start, end, correlationId, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL"):
        r, h = owner(corr)
        how[h] += 1
        key = r[2] if r else "UNATTRIBUTED"
        d = ops[key]
        d["kernels"] += 1
        d["graph_node_kernels"] += 1 if gnode else 0
        d["iv"].append((int(start), int(end)))
    for start, end, corr, kind in _q(c, "select start, end, correlationId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
        r, h = owner(corr)
        how[h] += 1
        d = ops[r[2] if r else "UNATTRIBUTED"]
        d["memcpy"][str(kind)] = d["memcpy"].get(str(kind), 0) + 1
        d["iv"].append((int(start), int(end)))
    for start, end, corr in _q(c, "select start, end, correlationId from CUPTI_ACTIVITY_KIND_MEMSET"):
        r, h = owner(corr)
        d = ops[r[2] if r else "UNATTRIBUTED"]
        d["memset"] += 1
        d["iv"].append((int(start), int(end)))
    # per-range op totals, then classes; also split kernels vs copy kinds per class (bench addition)
    per_class = collections.defaultdict(list)
    detail = collections.defaultdict(lambda: collections.defaultdict(list))

    def _norm(cls, d, norm):
        per_class[cls].append((d["kernels"] + sum(d["memcpy"].values()) + d["memset"]) / norm)
        e = detail[cls]
        e["ops"].append((d["kernels"] + sum(d["memcpy"].values()) + d["memset"]) / norm)
        e["kernels"].append(d["kernels"] / norm)
        e["memset"].append(d["memset"] / norm)
        for k, v in d["memcpy"].items():
            e[f"memcpy_{k}"].append(v / norm)

    for key, d in ops.items():
        m = re.match(r"S0:adv:(d\d+):(\d+):(\d+)", key)
        if m:
            dom, s0, n = m.group(1), int(m.group(2)), int(m.group(3))
            rad = any((s % cad.get(dom, 10**9)) == 0 for s in range(s0, s0 + n))
            _norm(f"{dom}:{'radiation' if rad else 'ordinary'}_per_step", d, max(n, 1))
        elif key.startswith("S0:output"):
            _norm("output_boundary", d, 1)
        elif key.startswith("S0:force"):
            _norm("force", d, 1)
        else:
            _norm(key.split(":")[0] + ":" + (key.split(":")[1] if ":" in key else ""), d, 1)
    all_iv = [iv for d in ops.values() for iv in d["iv"]]
    seg = [r for r in ranges if r[2].startswith("S0:segment")]
    # whole-report totals (one report per capture window in G1n: no NVTX attribution needed)
    kinds = collections.Counter(str(k) for (k,) in _q(c, "select copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"))
    graph_launch = 0
    for tab in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        graph_launch += sum(n for (n,) in _q(
            c, f"select count(*) from {tab} r join StringIds s on r.nameId = s.id "
               "where s.value like '%GraphLaunch%'"))
    span = _q(c, "select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL")
    totals = {
        "kernels": sum(n for (n,) in _q(c, "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL")),
        "graph_node_kernels": sum(n for (n,) in _q(
            c, "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is not null")),
        "inverted_kernel_rows": sum(n for (n,) in _q(
            c, "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where end < start")),
        "memcpy_by_kind": dict(kinds),
        "memset": sum(n for (n,) in _q(c, "select count(*) from CUPTI_ACTIVITY_KIND_MEMSET")),
        "graph_launch_api": graph_launch,
        "kernel_span_s": round((span[0][1] - span[0][0]) / 1e9, 4) if span and span[0][0] is not None else None,
    }
    totals["device_ops"] = totals["kernels"] + sum(kinds.values()) + totals["memset"]
    res = {
        "schema": "wrf_gpu2.v025.s0.nsys_node_census.v2", "db": db, "cadence": cad, "totals": totals,
        "n_s0_ranges": len(ranges), "attribution_method_counts": dict(how),
        "class_medians": {k: round(statistics.median(v), 1) for k, v in per_class.items() if v},
        "class_counts": {k: len(v) for k, v in per_class.items()},
        "total_ops": sum(d["kernels"] + sum(d["memcpy"].values()) + d["memset"] for d in ops.values()),
        "device_union_s": round(union_s(all_iv), 4) if all_iv else 0.0,
        "device_span_s": round((max(e for _, e in all_iv) - min(s for s, _ in all_iv)) / 1e9, 4) if all_iv else 0.0,
        "class_op_detail": {k: {kk: round(statistics.median(vv), 2) for kk, vv in v.items() if vv}
                            for k, v in detail.items() if v},
        "captured_segment_span_s": [round((r[1] - r[0]) / 1e9, 4) for r in seg],
        "unattributed_ops": ops["UNATTRIBUTED"]["kernels"] + sum(ops["UNATTRIBUTED"]["memcpy"].values())
        if "UNATTRIBUTED" in ops else 0,
    }
    open(out, "w").write(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("totals", "n_s0_ranges", "total_ops", "device_union_s", "class_medians",
                                          "unattributed_ops", "attribution_method_counts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
