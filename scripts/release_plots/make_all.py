#!/usr/bin/env python3
"""Regenerate every release figure (light + dark) from docs/release/showcase_inputs.json.

  JAX_PLATFORMS=cpu python scripts/release_plots/make_all.py --inputs docs/release/showcase_inputs.json --out-dir docs/release/img
CPU only (numpy/netCDF4/matplotlib); never imports JAX.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

from run_data import arm_rates

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable


def run(args):
    print("+", " ".join(str(a) for a in args[:3]), "...", flush=True)
    subprocess.run([PY, *map(str, args)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--themes", nargs="+", default=["light", "dark"])
    ap.add_argument("--only", nargs="*", default=None, help="subset: bench identity maps points b200")
    a = ap.parse_args()
    cfg = json.loads(pathlib.Path(a.inputs).read_text())
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    want = set(a.only or ["bench", "identity", "maps", "points", "b200"])
    for th in a.themes:
        sfx = "" if th == "light" else "_dark"
        if "bench" in want:
            run([HERE / "bench_plots.py", "--inputs", a.inputs, "--out-dir", out, "--theme", th])
        if "identity" in want:
            for it in cfg.get("identity", []):
                args = [HERE / "identity_curves.py", "--out", out / f"{it['name']}{sfx}.png", "--theme", th,
                        "--title", it["title"], "--subtitle", it.get("subtitle", ""), "--footer", it.get("footer", ""),
                        "--summary-json", out / f"{it['name']}.json", "--domains", *it["domains"]]
                cargs = []
                for k, v in it["cases"].items():
                    cargs += ["--case", f"{k}={v}"]
                run(args + cargs)
                bias_args = [*args]
                bias_args[bias_args.index("--out") + 1] = out / f"{it['name']}_bias{sfx}.png"
                bias_args[bias_args.index("--summary-json") + 1] = out / f"{it['name']}_bias.json"
                bias_args[bias_args.index("--title") + 1] = it["title"].replace("RMSE", "bias") + " (GPU − CPU-WRF)"
                bias_args[bias_args.index("--footer") + 1] = it.get("footer", "").replace(
                    "dashed = release-gate RMSE limit", "dashed = zero bias") + "; lines = case median, shading = case range"
                run(bias_args + ["--metric", "bias"] + cargs)
                run([HERE / "identity_heatmap.py", "--out", out / f"{it['name']}_heatmap{sfx}.png", "--theme", th,
                     "--footer", it.get("footer", ""), "--domains", *it["domains"], *cargs])
        if "maps" in want:
            for it in cfg.get("maps", []):
                args = [HERE / "field_maps.py", "--cpu-dir", it["cpu_dir"], "--gpu-dir", it["gpu_dir"], "--domain",
                        it["domain"], "--footer", it.get("footer", ""), "--theme", th, "--out", out / f"{it['name']}{sfx}.png"]
                if it.get("lead") is not None:
                    args += ["--lead", it["lead"]]
                if it.get("init") is not None:
                    args += ["--init", it["init"]]
                run(args)
        if "points" in want:
            for it in cfg.get("points", []):
                run([HERE / "point_series.py", "--cpu-dir", it["cpu_dir"], "--gpu-dir", it["gpu_dir"], "--domain",
                     it["domain"], "--footer", it.get("footer", ""), "--theme", th, "--out", out / f"{it['name']}{sfx}.png"])
        if "b200" in want and "b200" in cfg:
            b = cfg["b200"]
            an = arm_rates(pathlib.Path(b["arm_dir"]))
            run([HERE / "b200_extrapolation.py", "--sat-stepping-s-per-case-h", an["throughput_stepping_s_per_case_h"],
                 "--sat-whole-s-per-case-h", an["throughput_whole_run_s_per_case_h"], "--sat-n", b["sat_n"],
                 "--rtx-board-w", b["rtx_board_w"], "--rtx-board-w-label", b["rtx_board_w_label"],
                 "--case-vram-gib", b["case_vram_gib"], "--tree", b.get("tree", cfg["tree_label"]), "--theme", th,
                 "--cpu-s-per-case-h", cfg["cpu"]["wn3_3x4_s_per_case_h"]["value"],
                 "--cpu-w-lo", cfg["cpu"]["package_w_12core"]["lo"], "--cpu-w-hi", cfg["cpu"]["package_w_12core"]["hi"],
                 "--cpu-power-label", cfg["cpu"]["package_w_12core"]["label"], "--timing-method", an["timing_method"],
                 "--out", out / f"b200{sfx}.png", "--json", out / "b200.json"])
    print("done ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
