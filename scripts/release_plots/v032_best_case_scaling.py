"""CPU-only best-case scenarios; measured anchors must come from frozen VAL32.

collect: attest 24-hour N=4 sweep, geometry and PROD receipts into one input.
render: make ensemble/giant-domain scenarios, retaining assumptions and sources.
draft: layout only, without invented benchmark or extrapolation numbers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import textwrap

SCHEMA = "wrf_gpu.v032.best_case_anchors/1"
CAPTION = ("B200 measured on earlier versions; H100/B200/B300 extrapolated to "
           "v0.3.2, best case, not measured.")
HARDWARE = {
    "RTX 5090": {"physical_gb": 32, "bandwidth_tb_s": 1.792},
    "H100 SXM": {"physical_gb": 80, "bandwidth_tb_s": 3.35},
    "B200": {"physical_gb": 192, "bandwidth_tb_s": 8},
    "B300": {"physical_gb": 288, "bandwidth_tb_s": 8},
}
EFFICIENCY = {"low": 2.69 / (8 / 1.792), "central": 3.53 / (8 / 1.792), "high": 1.0}


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def load(path):
    return json.loads(Path(path).read_text())


def source(path):
    path = Path(path)
    return {"artifact": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def geometry(proof):
    rows = []
    for name, data in proof["metadata"]["domains"].items():
        shape = data["grid"]["mass_shape"]
        if len(shape) != 3 or any(int(n) <= 0 for n in shape):
            raise ValueError(f"invalid geometry for {name}")
        dt = positive(data["namelist"]["dt_s"], f"{name} dt")
        rows.append({"domain": name, "mass_shape": shape, "cells_3d": math.prod(shape),
                     "dt_s": dt, "cell_updates_per_forecast_hour": math.prod(shape) * 3600 / dt})
    if not rows:
        raise ValueError("missing measured geometry")
    return {"domains": rows, "cells_3d": sum(r["cells_3d"] for r in rows),
            "cell_updates_per_forecast_hour": sum(r["cell_updates_per_forecast_hour"] for r in rows)}


def collect(args):
    revision = Path(args.freeze_rev).read_text().strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("FREEZE_REV must be a full lowercase commit SHA")
    sweep, prod = load(args.sweep_summary), load(args.prod_receipt)
    if sweep["hours"] != 24:
        raise ValueError("final sweep anchor must be 24 h; six-hour figures are historical context")
    phase = sweep["phases"]["n4"]
    if phase["aggregate"]["n_cases"] != 4 or phase["peak_concurrency"] != 4 or phase.get("stop_reason"):
        raise ValueError("requires an admitted, completed N=4 phase")
    cases = list(phase["cases"].values())
    if len(cases) != 4 or any(c["rc"] != 0 or not c["aot_cache_warm"] for c in cases):
        raise ValueError("all four cases must exit successfully with warm executables")
    if any(c["source_head"] != revision for c in cases) or prod["git_head"] != revision:
        raise ValueError("measured sources do not match FREEZE_REV")
    if prod["rc"] != 0 or prod["end_reason"] != "completed" or prod["device"]["platform"] != "gpu":
        raise ValueError("PROD must be a successful measured GPU run")
    sweep_proof, prod_proof = load(args.sweep_proof), load(args.prod_proof)
    for proof in (sweep_proof, prod_proof):
        device = proof['device']
        gpu_device = device.get('platform') == 'gpu' if isinstance(device,dict) else str(device).startswith(('cuda:','gpu:'))
        if not gpu_device or proof["hours"] != 24 or not proof['all_outputs_present'] or not proof['all_domains_finite']:
            raise ValueError("both geometry proofs must attest GPU/24 h")
    sweep_geometry, prod_geometry = geometry(sweep_proof), geometry(prod_proof)
    for case in cases:
        segments = case["d01_hour_segments_s"]
        if len(segments) != 24:
            raise ValueError("24 per-hour segments are required, including the first")
        for n in segments:
            positive(n, "hour segment")
    aggregate = phase['aggregate']
    if 'steady_s_per_case_fc_h' in aggregate:
        steady = positive(aggregate['steady_s_per_case_fc_h'], 'all-running steady rate')
        steady_method = 'producer all-four-running window / measured case-hours in that window'
    else:
        steady = statistics.mean(statistics.mean(c["d01_hour_segments_s"][1:]) for c in cases) / 4
        steady_method = 'mean per-case h2-h24 means / 4'
    whole = positive(aggregate.get('whole_run_s_per_case_fc_h', aggregate.get('aggregate_s_per_case_fc_h')), "whole aggregate rate")
    if whole < steady:
        raise ValueError("whole clock is below the chosen steady estimator; review the clocks")
    prod_rates = prod["derived"]["segment_rate_s_per_fc_h"]
    if len(prod_rates) != 24:
        raise ValueError("PROD must contain 24 normalized segment rates")
    prod_steady = statistics.median(positive(x, "PROD segment") for x in prod_rates[1:])
    anchor = {"schema": SCHEMA, "status": "measured_anchors", "target_version": "0.3.2",
              "freeze_revision": revision, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "sources": [source(getattr(args, key)) for key in
                          ("freeze_rev", "sweep_summary", "sweep_proof", "prod_receipt", "prod_proof")],
              "sweep": {"hours": 24, "n": 4, "whole_s_per_case_fc_h": whole,
                        "steady_s_per_case_fc_h": steady,
                        "steady_method": steady_method,
                        "fixed_s_per_case": (whole - steady) * 24,
                        "warm_vram_mib_per_case": max(c["vram_peak_mib"] for c in cases),
                        "warm_host_high_water_kib_per_case": max(c["vmhwm_kb"] for c in cases),
                        "geometry": sweep_geometry, "clock_label": "M"},
              "prod": {"hours": 24, "whole_s_per_fc_h": positive(prod["t_end"], "PROD wall") / 24,
                       "steady_s_per_fc_h": prod_steady, "geometry": prod_geometry, "clock_label": "M"},
              "cpu": {"ensemble_12core_s_per_case_fc_h": 123.4,
                      "prod_12core_s_per_fc_h": 92.3, "clock_label": "M",
                      "source": "frozen original CPU-WRF WN3 3x4-core / PROD 12-core references"}}
    validate(anchor)
    Path(args.out).write_text(json.dumps(anchor, indent=2) + "\n")


def validate(anchor):
    if anchor.get("schema") != SCHEMA or anchor.get("status") != "measured_anchors":
        raise ValueError("validated VAL32 anchors are required; template is not measurement")
    if anchor.get("target_version") != "0.3.2" or anchor["sweep"]["hours"] != 24:
        raise ValueError("wrong release/window")
    if not anchor.get("sources") or any(len(s.get("sha256", "")) != 64 for s in anchor["sources"]):
        raise ValueError("missing source hashes")
    for name in ("sweep", "prod"):
        row = anchor[name]
        positive(row["steady_s_per_case_fc_h" if name == "sweep" else "steady_s_per_fc_h"], name + " steady")
        positive(row["geometry"]["cells_3d"], name + " cells")
        positive(row["geometry"]["cell_updates_per_forecast_hour"], name + " work")
    positive(anchor["sweep"]["warm_vram_mib_per_case"], "per-case VRAM")
    positive(anchor["sweep"]["warm_host_high_water_kib_per_case"], "per-case host memory")
    if anchor["sweep"]["fixed_s_per_case"] < 0:
        raise ValueError("negative fixed-cost proxy")


def scenarios(anchor, members=32, hours=72, levels=44, giant_dt=6, usable_fraction=.9, growth=1.1):
    validate(anchor)
    if min(members, hours, levels) <= 0 or not 0 < usable_fraction <= 1 or growth < 1:
        raise ValueError("invalid scenario controls")
    positive(giant_dt, "giant dt")
    sweep, prod = anchor["sweep"], anchor["prod"]
    case_bytes = sweep["warm_vram_mib_per_case"] * 2**20
    bytes_per_cell = case_bytes / sweep["geometry"]["cells_3d"]
    prod_updates_s = prod["geometry"]["cell_updates_per_forecast_hour"] / prod["steady_s_per_fc_h"]
    cpu_updates_s = prod["geometry"]["cell_updates_per_forecast_hour"] / anchor["cpu"]["prod_12core_s_per_fc_h"]
    out = {"caption": CAPTION, "target_version": "0.3.2", "freeze_revision": anchor["freeze_revision"],
           "sources": anchor["sources"], "controls": {"members": members, "hours": hours, "levels": levels,
           "giant_dt_s": giant_dt, "giant_dx_m": 1000, "usable_fraction": usable_fraction,
           "memory_growth_factor": growth, "global_reserve_bytes": 2**30},
           "memory": {"gross_bytes_per_cell": bytes_per_cell, "label": "I",
                      "note": "allocated per-PID peak / sum of measured mass-grid cells; includes pools/context/scratch"},
           "cpu": {"steady_throughput_case_fc_h_per_wall_h": 3600 / anchor["cpu"]["ensemble_12core_s_per_case_fc_h"],
                   "ensemble_minutes": members * hours * anchor["cpu"]["ensemble_12core_s_per_case_fc_h"] / 60,
                   "rate_label": "M", "duration_label": "I"}, "devices": {}}
    for name, hardware in HARDWARE.items():
        rho = hardware["bandwidth_tb_s"] / HARDWARE["RTX 5090"]["bandwidth_tb_s"]
        factors = {"central": 1.0, "low": 1.0, "high": 1.0} if name == "RTX 5090" else {
            key: rho * value for key, value in EFFICIENCY.items()}
        rate = sweep["steady_s_per_case_fc_h"] / factors["central"]
        fixed_rate = sweep["fixed_s_per_case"] / hours
        budget = hardware["physical_gb"] * 1e9 * usable_fraction
        cases_fit = int(budget // (case_bytes + 2**30))
        if cases_fit < 1:
            raise ValueError("no case fits the selected budget")
        giant_cells_limit = (budget - 2**30) / (bytes_per_cell * growth)
        side = math.isqrt(int(giant_cells_limit / levels))
        cells = side * side * levels
        work = cells * 3600 / giant_dt
        throughput_band = [3600 / (sweep["steady_s_per_case_fc_h"] / factors[k]) for k in ("low", "high")]
        out["devices"][name] = {"rate_label": "M-anchored" if name == "RTX 5090" else "I",
            "scenario_label": "I", "steady_throughput_case_fc_h_per_wall_h": 3600 / rate,
            "steady_throughput_band": throughput_band,
            "amortized_s_per_case_fc_h": rate + fixed_rate,
            "ensemble_minutes": members * hours * (rate + fixed_rate) / 60,
            "resident_case_capacity": cases_fit,
            "minimum_waves": math.ceil(members / cases_fit),
            "host_ram_gb_at_capacity": cases_fit * sweep["warm_host_high_water_kib_per_case"] * 1024 / 1e9,
            "host_reserve_excluded": True, "packing_budget_gb": budget / 1e9,
            "giant": {"side": side, "horizontal_cells": side * side, "cells_3d": cells,
                "levels": levels, "gpu_s_per_fc_h": work / (prod_updates_s * factors["central"]),
                "cpu_12core_s_per_fc_h": work / cpu_updates_s, "label": "I"}}
    out["assumptions"] = [
        "72-hour duration is inferred even for CPU/RTX measured-rate anchors; no 72-hour VAL32 timing claim.",
        "Best case: warm executables, steady saturated throughput, adequate host cores/RAM and output bandwidth.",
        "Fixed-cost proxy is unscaled per case and amortized; cold compilation excluded; FIFO tail under-utilization unmodelled.",
        "VRAM capacity is an allocated-peak linear scenario, not measured giant-domain feasibility or R1 scaling proof.",
        "Giant 1-km timestep is explicit; linear cell-update scaling ignores geometry/cadence/compiler-layout changes.",
        "Old B200 hardware efficiency transfers to new full-physics code by assumption; bands are not confidence intervals."]
    return out


def draw(data, out, dark=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#111827", "#eef3fa", "#b8c4d7") if dark else ("#fff", "#17243b", "#51627a")
    bg, fg, muted = colors
    suffix = "_dark" if dark else ""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 15, "text.color": fg,
                         "axes.labelcolor": fg, "xtick.color": fg, "ytick.color": fg}):
        for kind in ("ensemble", "giant_domain"):
            fig, ax = plt.subplots(figsize=(12, 7.4), dpi=160)
            fig.set_facecolor(bg); ax.set_facecolor(bg)
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.set_axisbelow(True); ax.grid(axis="y", alpha=.18, color=muted)
            if data is None:
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(.5, .5, "WAITING FOR FROZEN VAL32 RECEIPTS\nCPU script/layout preview; no projected values", ha="center", transform=ax.transAxes)
                ax.set_title("v0.3.2 best-case ensembles" if kind == "ensemble" else "v0.3.2 giant-domain scenario", fontsize=20)
                foot = CAPTION + "\nFinal numbers require the 24 h N=4 sweep, PROD S1 and matching geometry proofs."
            else:
                names = list(data["devices"])
                if kind == "ensemble":
                    values = [data["cpu"]["steady_throughput_case_fc_h_per_wall_h"]] + [data["devices"][n]["steady_throughput_case_fc_h_per_wall_h"] for n in names]
                    labels = ["CPU-WRF\n12 cores [M]"] + [n + "\n" + ("[I]" if data["devices"][n]["rate_label"] == "I" else data["devices"][n]["rate_label"]) for n in names]
                    ax.bar(range(5), values, color=["#94a3b8"] + ["#18a58b"] * 4)
                    for i, name in enumerate(names, start=1):
                        low, high = data['devices'][name]['steady_throughput_band']
                        ax.errorbar(i, values[i], yerr=[[values[i]-low],[high-values[i]]],color=fg,capsize=5)
                    for i, value in enumerate(values):
                        high = value if i == 0 else data['devices'][names[i-1]]['steady_throughput_band'][1]
                        ax.text(i, high * (1.8 if i == 0 else 1.06), f"{value:,.0f}\n×{value/values[0]:.0f} [I]", ha="center", weight="bold")
                    durations = [data["cpu"]["ensemble_minutes"]] + [data["devices"][n]["ensemble_minutes"] for n in names]
                    for i, minutes in enumerate(durations): ax.text(i, .06, f"{minutes:,.1f} min [I]", transform=ax.get_xaxis_transform(), ha="center", color=fg,
                                                                bbox={"facecolor": bg, "edgecolor": "none", "alpha": .95})
                    ax.set_yscale("log"); ax.set_ylim(min(values) * .55, max(values) * 3.4)
                    ax.set_xticks(range(5), labels); ax.set_ylabel("Steady case-forecast-hours per wall-hour")
                    ax.set_title(f"Best-case {data['controls']['members']}-member × {data['controls']['hours']} h ensemble", fontsize=20, pad=25)
                    capacities = "; ".join(f"{n}: {data['devices'][n]['resident_case_capacity']} cases / {data['devices'][n]['host_ram_gb_at_capacity']:.0f} GB host" for n in names)
                    foot = CAPTION + f"\n{data['controls']['hours']} h duration [I]: fixed start-up per case amortizes over longer runs; compilation excluded.\n" + capacities + "\nAdequate host cores/RAM assumed; batch-tail under-utilization unmodelled."
                else:
                    x = list(range(4)); gpu = [data["devices"][n]["giant"]["gpu_s_per_fc_h"] for n in names]
                    cpu = [data["devices"][n]["giant"]["cpu_12core_s_per_fc_h"] for n in names]
                    ax.bar([i-.18 for i in x], cpu, width=.35, color="#94a3b8", label="CPU 12-core at the same grid [I]")
                    ax.bar([i+.18 for i in x], gpu, width=.35, color="#18a58b", label="GPU at estimated capacity [I]")
                    labels = []
                    for i, name in enumerate(names):
                        g = data["devices"][name]["giant"]
                        labels.append(f"{name}\n{g['side']}×{g['side']} at 1 km\n{g['cells_3d']/1e6:.1f} M cells, {g['levels']} levels")
                        for position, value, speedup in ((i-.18,cpu[i],1),(i+.18,gpu[i],cpu[i]/gpu[i])):
                            ax.text(position,value*1.07,f"{value:,.1f}\n×{speedup:.0f} [I]",ha="center",fontsize=14)
                    ax.set_xticks(x, labels); ax.set_yscale("log"); ax.set_ylim(min(gpu)*.5,max(cpu)*3)
                    ax.set_ylabel("Seconds per forecast hour · matched grid [I]", fontsize=16)
                    ax.set_title("Largest estimated single domain per GPU · best-case potential [I]",fontsize=20,pad=25)
                    ax.legend(fontsize=14, frameon=False)
                    foot = CAPTION + f"\nGross {data['memory']['gross_bytes_per_cell']:.0f} bytes/cell from allocated VRAM; 90% capacity budget + 1 GiB reserve.\n" + f"Explicit 1 km / dt={data['controls']['giant_dt_s']:g} s / {data['controls']['levels']} levels; steady linear cell-work scaling.\nNot a measured giant-domain run, memory-scaling gate or fidelity claim."
            fig.subplots_adjust(left=.12,right=.98,top=.86,bottom=.33)
            foot = '\n'.join(textwrap.fill(line,width=100) for line in foot.splitlines())
            fig.text(.02,.025,foot,fontsize=14,color=muted,linespacing=1.3)
            fig.savefig(out/f"v032_best_case_{kind}{suffix}.png",facecolor=bg);plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    for key in ("freeze-rev", "sweep-summary", "sweep-proof", "prod-receipt", "prod-proof", "out"):
        c.add_argument("--"+key, type=Path, required=True)
    r = sub.add_parser("render"); r.add_argument("--anchors",type=Path,required=True);r.add_argument("--out-dir",type=Path,required=True)
    r.add_argument("--members",type=int,default=32);r.add_argument("--hours",type=int,default=72)
    r.add_argument("--levels",type=int,default=44);r.add_argument("--giant-dt",type=float,default=6)
    d = sub.add_parser("draft");d.add_argument("--out-dir",type=Path,required=True)
    args = ap.parse_args()
    if args.command == "collect":collect(args);return
    data = scenarios(load(args.anchors),args.members,args.hours,args.levels,args.giant_dt) if args.command == "render" else None
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if data is not None:(args.out_dir/"v032_best_case_scaling.json").write_text(json.dumps(data,indent=2)+'\n')
    draw(data,args.out_dir);draw(data,args.out_dir,dark=True)


if __name__ == "__main__":main()
