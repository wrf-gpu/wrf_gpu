"""Read release timing receipts without importing JAX or relying on file mtimes.

W6 receipts have benchmark output timers; the public parallel launcher records
filename and first-publication time instead. Its one-second poll is disclosed
when used for timing. Compression may change file mtimes after either receipt.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import wrfout_pairs


def load(path):
    return json.loads(Path(path).read_text())


def unix(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def case_outputs(case: Path, expected_domains=None) -> dict[str, dict[int, float]]:
    """Return domain -> hourly lead -> publication/completion Unix time."""
    r = load(case / "receipt.json")
    if r.get("rc", 0) != 0 or r.get("errors"):
        raise ValueError(f"{case}: failed run cannot supply release timings")
    outputs = r.get("outputs", [])
    if not outputs:
        raise ValueError(f"{case}: no recorded outputs")
    modern = "t_published_s" in outputs[0]
    by_dom = {}
    if modern:
        valid = [wrfout_pairs._valid(o["file"])[1] for o in outputs]
        if any(t is None for t in valid):
            raise ValueError(f"{case}: invalid wrfout filename")
        init = min(valid)
        start = unix(r["start_utc"])
        for o, t in zip(outputs, valid):
            lead = (t - init).total_seconds() / 3600
            if lead != int(lead):
                raise ValueError(f"{case}: nonhourly output")
            dom = o["d"]
            if wrfout_pairs._valid(o["file"])[0] != dom:
                raise ValueError(f"{case}: domain does not match output filename")
            leads = by_dom.setdefault(dom, {})
            if int(lead) in leads:
                raise ValueError(f"{case}: duplicate {dom} lead {lead}")
            leads[int(lead)] = start + float(o["t_published_s"])
    else:
        # W6 has hourly history, including own_step=0. Refuse an ambiguous
        # receipt rather than shifting every lead by one hour.
        grouped = {}
        for o in outputs:
            grouped.setdefault(o["d"], []).append(o)
        for dom, rows in grouped.items():
            rows.sort(key=lambda o: o["own_step"])
            if rows[0]["own_step"] != 0:
                raise ValueError(f"{case}: legacy receipt lacks {dom} init frame")
            if len({o["own_step"] for o in rows}) != len(rows):
                raise ValueError(f"{case}: duplicate {dom} output")
            by_dom[dom] = {lead: float(r["t0_wall_utc"]) + o["t_start"] + o["s"]
                           for lead, o in enumerate(rows)}
    max_dom = (r.get("cli_summary") or {}).get("effective_max_dom")
    expected = expected_domains or ([f"d{i:02}" for i in range(1, int(max_dom) + 1)] if max_dom else None)
    if expected is not None and set(by_dom) != set(expected):
        raise ValueError(f"{case}: missing/unexpected domains: {sorted(by_dom)}, expected {expected}")
    return by_dom


def production_batch_summary(arm: Path) -> dict:
    """Whole six-case, 72 h FIFO production rate; separate from the N sweep."""
    run = load(arm / "parallel_run.json")
    args = run["cli_args"]
    if (float(args[args.index("--hours") + 1]) != 72 or len(run["cases"]) != 6
            or run.get("stop_reason") or run.get("compress_rc") not in (None, 0)
            or any(row.get("rc") != 0 for row in run["cases"].values())):
        raise ValueError("production row needs six successful, complete 72 h cases")
    expected = set(range(73))
    for name in run["cases"]:
        outputs = case_outputs(arm / name, expected_domains=("d01", "d02", "d03"))
        if any(set(leads) != expected for leads in outputs.values()):
            raise ValueError("production row needs every frame of every case")
    wall = float(run["wall_s"])
    if wall <= 0 or run.get("case_hours", 432) != 432:
        raise ValueError("invalid production wall or case-hour denominator")
    return {"wall_s": wall, "case_hours": 432, "cases": 6, "hours": 72,
            "peak_concurrency": run["peak_concurrency"], "s_per_case_h": wall / 432,
            "ref": str(arm / "parallel_run.json"),
            "method": "whole launcher, six cases including FIFO waves, compression and cleanup"}


def arm_info(arm: Path) -> dict:
    """Use the launcher run manifest when present; retain W6 analysis support."""
    if (arm / "parallel_run.json").exists():
        run = load(arm / "parallel_run.json")
        if run.get("stop_reason") or any(c.get("rc") != 0 for c in run["cases"].values()):
            raise ValueError(f"{arm}: unsuccessful parallel arm")
        if run.get("compress_rc") not in (None, 0):
            raise ValueError(f"{arm}: unsuccessful output compression")
        args = run["cli_args"]
        hours = float(args[args.index("--hours") + 1])
        n = len(run["cases"])
        if run.get("peak_concurrency", n) != n:
            raise ValueError(f"{arm}: {n} cases were queued, not all run concurrently")
        return {"start_unix": unix(run["start_utc"]), "hours": hours, "n_cases": n,
                "arm_wall_s": float(run["wall_s"]), "per_case": run["cases"],
                "ref": str(arm / "parallel_run.json"),
                "timing_method": "first-publication polling (1 s); includes output; excludes first forecast hour"}
    an = load(arm / "analysis.json")
    if any(c.get("rc", 0) != 0 for c in an["per_case"].values()):
        raise ValueError(f"{arm}: unsuccessful W6 arm")
    return {**an, "start_unix": load(arm / "start.json")["start_unix"],
            "ref": str(arm / "analysis.json"), "timing_method": "W6 benchmark segment timers"}


def t_to_lead(arm: Path, lead_h: int) -> tuple[float, int, list[Path]]:
    """Arm start to the last case's output at this lead, across all 3 domains."""
    info = arm_info(arm)
    cases = [arm / name for name in info["per_case"]]
    done = []
    for case in cases:
        domains = case_outputs(case, ["d01", "d02", "d03"])
        try:
            done.extend(leads[lead_h] for leads in domains.values())
        except KeyError as exc:
            raise ValueError(f"{case}: missing domain output at +{lead_h} h") from exc
    wall = max(done) - info["start_unix"]
    if wall <= 0:
        raise ValueError(f"{arm}: nonpositive output wall time")
    return wall, len(cases), cases


def benchmark_endpoint(arm: Path) -> dict:
    """Time an admitted N-sweep at its real horizon; queued FIFO arms are refused."""
    info = arm_info(arm)
    hours = float(info["hours"])
    if hours < 4 or hours != int(hours):
        raise ValueError("release benchmark needs at least four complete forecast hours")
    lead = int(hours)
    if (arm / "parallel_run.json").exists():
        publication_wall, n, cases = t_to_lead(arm, lead)
        wall = float(info["arm_wall_s"])
        if wall + 0.1 < publication_wall:
            raise ValueError(f"{arm}: launcher ended before its last recorded output")
        method = "whole launcher completion, including output compression and cleanup"
    else:
        if hours != 24:
            raise ValueError("legacy arm needs a 24 h process receipt; callbacks are not publication")
        wall, n = info["arm_wall_s"], info["n_cases"]
        cases = [arm / name for name in info["per_case"]]
        method = "24 h process completion (W6, includes output joins)"
    return {"wall_s": wall, "hours": lead, "n_cases": n, "case_dirs": cases,
            "last_publication_wall_s": publication_wall if (arm / "parallel_run.json").exists() else None,
            "s_per_case_h": wall / (n * lead), "timing_method": method}


def arm_rates(arm: Path) -> dict:
    info = arm_info(arm)
    n, hours = info["n_cases"], info["hours"]
    if "throughput_stepping_s_per_case_h" in info:
        return info
    if hours <= 1 or hours != int(hours):
        raise ValueError(f"{arm}: need at least 2 hourly forecast frames")
    last, _, _ = t_to_lead(arm, int(hours))
    first, _, _ = t_to_lead(arm, 1)
    # First-hour and last-hour barriers are each the last case's last domain.
    # This includes output and is deliberately called steady output throughput,
    # not kernel/stepping time. No unmeasured stepping-only value is invented.
    return {**info, "throughput_whole_run_s_per_case_h": info["arm_wall_s"] / (n * hours),
            "throughput_stepping_s_per_case_h": (last - first) / (n * (hours - 1)),
            "vram_sum_of_pid_peaks_mib": sum(v["vram_peak_mib"] for v in info["per_case"].values()),
            "timing_method": info["timing_method"]}


def prod_summary(cfg: dict) -> dict:
    """S1 ends at the last written file; callback return may precede its write.

Capture original file mtimes only inside the recorded process window. A saved
endpoint receipt lets plots be reproduced after rolling compression or cleanup.
"""
    path = Path(cfg["bench_json"])
    bench = load(path)
    arm = bench["arms"][cfg["whole_run_arm"]]
    if arm.get("rc") != 0 or arm.get("errors"):
        raise ValueError(f"{path}: unsuccessful PROD arm")
    r = load(arm["receipt"])
    args = r["cli_argv"]
    if "--hours" not in args or float(args[args.index("--hours") + 1]) != float(cfg["whole_run_hours"]):
        raise ValueError("configured forecast hours do not match the benchmark run")
    endpoint = cfg.get("endpoint_receipt")
    if isinstance(endpoint, (str, Path)):
        endpoint = load(endpoint)
    if endpoint is None:
        out = Path(args[args.index("--output-dir") + 1])
        files = sorted(p for p in out.glob("wrfout_d*") if wrfout_pairs._valid(p.name)[0])
        if len(files) != len(r["outputs"]):
            raise ValueError(f"{out}: need all original output files or a saved publication receipt")
        writes = {p.name: p.stat().st_mtime for p in files}
        t0 = float(r["t0_wall_utc"])
        if not writes or any(t < t0 or t > t0 + arm["t_end_s"] + 0.1 for t in writes.values()):
            raise ValueError(f"{out}: file mtimes changed; use the pre-compression publication receipt")
        endpoint = {"benchmark_receipt": arm["receipt"], "src_tree": arm.get("src_tree"),
                    "arm": cfg["whole_run_arm"], "hours": cfg["whole_run_hours"], "t0_wall_utc": t0,
                    "file_write_unix": writes, "wall_s": max(writes.values()) - t0,
                    "method": "original output file mtimes, captured inside the completed run window"}
    if (endpoint["benchmark_receipt"] != arm["receipt"] or endpoint["src_tree"] != arm.get("src_tree")
            or endpoint["arm"] != cfg["whole_run_arm"] or endpoint["hours"] != cfg["whole_run_hours"]):
        raise ValueError("publication receipt does not identify this benchmark arm")
    wall = float(endpoint["wall_s"])
    if (endpoint["t0_wall_utc"] != r["t0_wall_utc"] or len(endpoint["file_write_unix"]) != len(r["outputs"])
            or abs(max(endpoint["file_write_unix"].values()) - endpoint["t0_wall_utc"] - wall) > 1e-6
            or not 0 < wall <= arm["t_end_s"] + 0.1):
        raise ValueError("inconsistent publication clocks/count in endpoint receipt")
    return {"bench": bench, "receipt": r, "wall_s": wall,
            "s_per_fch": wall / float(cfg["whole_run_hours"]), "ref": arm["receipt"], "endpoint": endpoint}
