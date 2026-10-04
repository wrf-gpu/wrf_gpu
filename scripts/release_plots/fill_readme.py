#!/usr/bin/env python3
"""Fill the {{KEY}} placeholders of docs/release/README_v0.3_draft.md from receipts, and bundle the evidence.

Reads docs/release/showcase_inputs.json (paths + manual values) and the make_all.py outputs in --img-dir
(numbers.json, b200.json, identity_curves_*.json). Every number is computed from a receipt. Each cited receipt is
copied (small) or slimmed (D6 scorer JSONs, run receipts, dmon logs) into --evidence-dir, and the provenance table
links those repo-relative copies; each copy keeps its original absolute path in "source_path" for internal audit.
Unfilled keys are reported and left visible.

  python scripts/release_plots/fill_readme.py --inputs docs/release/showcase_inputs.json --img-dir docs/release/img \
      --evidence-dir docs/release/evidence --template docs/release/README_v0.3_draft.md --out README.md \
      --numbers-out docs/release/numbers_final.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_plots import read_dmon  # noqa: E402
from run_data import arm_info, benchmark_endpoint, prod_summary, production_batch_summary  # noqa: E402
from fidelity_data import gate_summary, integrity_summary, approved_integrity_summary  # noqa: E402

GATE_FIELDS = ["T2", "U10", "V10", "PSFC", "RAINNC", "T", "U", "V", "W", "QVAPOR"]


def load(p):
    return json.loads(pathlib.Path(p).read_text())


class Evidence:
    """Small, publishable copies of every cited receipt (repo-relative links in the README)."""

    def __init__(self, root: pathlib.Path, rel_prefix: str):
        self.root, self.rel = root, rel_prefix.rstrip("/")
        root.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, str] = {}

    def _write(self, name: str, obj: dict, source: str) -> str:
        obj = {"source_path": source, **obj}
        # publishable: home-relative paths only
        (self.root / name).write_text(json.dumps(obj, indent=1).replace(str(pathlib.Path.home()) + "/", "~/"))
        self.index[name] = source
        return f"[{name}]({self.rel}/{name})"

    def copy(self, path, name: str) -> str:
        return self._write(name, {"content": load(path)}, str(path))

    def slim_d6(self, d6_dir, name: str) -> str:
        doms = {}
        for p in sorted(pathlib.Path(d6_dir).glob("d0?.json")):
            js = load(p)
            doms[p.stem] = {
                "schema": js.get("schema"), "inputs": js.get("inputs"),
                "summaries": {k: v for k, v in js["summaries"].items() if k != "top_drift_signals"},
                "tolerances": dict(js["tolerances"]),
                "coverage": gate_summary(js),
                "all_fields": {f: {k: fs.get(k) for k in ("classification", "compared_lead_count", "missing_leads",
                                                        "incompatible_leads", "overall", "tolerance_result", "drift")}
                               for f, fs in js["field_summaries"].items()},
                "gate_fields_by_lead": {f: [{k: r.get(k) for k in ("lead_h", "rmse", "bias", "max_abs", "n")}
                                            | {"pass": (r.get("tolerance_result") or {}).get("pass")}
                                            for r in js["field_summaries"][f]["by_lead"]]
                                        for f in GATE_FIELDS if f in js["field_summaries"]},
            }
        return self._write(name, {"domains": doms}, str(d6_dir))

    def receipt(self, case_dir, name: str) -> str:
        r = load(pathlib.Path(case_dir) / "receipt.json")
        recorded_env = r.get("env") or {}
        if isinstance(recorded_env, list):
            recorded_env = dict(item.split("=", 1) for item in recorded_env if "=" in item)
        env = {k: v for k, v in recorded_env.items()
               if k.startswith(("GPUWRF_", "XLA_", "JAX_", "OMP_")) and not k.endswith("_DIR")
               and not any(part in k for part in ("LOCK", "TOKEN", "LEASE", "APPROVAL"))}
        keep = {k: r.get(k) for k in ("schema", "tag", "name", "rc", "start_utc", "end_utc", "wall_s", "cli_summary",
                                     "t0_wall_utc", "segments", "outputs", "errors", "omp_num_threads",
                                     "git_head", "src_tree", "device", "end_reason", "host_vmhwm_kb",
                                     "vmhwm_kb", "vram_peak_mib")}
        return self._write(name, keep | {"env": env}, str(case_dir))

    def dmon(self, path, name: str) -> str:
        info = read_dmon(pathlib.Path(path))
        return self._write(name, {"summary": info, "note": "nvidia-smi board power; integration method and sample span recorded above"}, str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--img-rel", default="docs/release/img", help="image link prefix; use plots for a local preview")
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--evidence-rel", default="docs/release/evidence", help="link prefix used in the README")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--numbers-out", required=True)
    ap.add_argument("--require-final", action="store_true", help="reject prototypes, incomplete data and missing keys")
    a = ap.parse_args()
    cfg = load(a.inputs)
    img = pathlib.Path(a.img_dir)
    nums = load(img / "numbers.json")
    b200 = load(img / "b200.json") if (img / "b200.json").exists() else None
    man = cfg.get("manual", {})
    ev = Evidence(pathlib.Path(a.evidence_dir), a.evidence_rel)
    K: dict[str, str] = {}
    prov: list[tuple[str, str, str, str]] = []

    def put(key, value, fmtspec, label, ref, show):
        K[key] = format(value, fmtspec) if not isinstance(value, str) else value
        prov.append((show, K[key], label, ref))

    cpu = cfg["cpu"]
    # PROD whole-run S1 straight from bench.json (arm wall from process start to last wrfout / hours)
    bj = load(cfg["prod"]["bench_json"])
    e_bench = ev.copy(cfg["prod"]["bench_json"], "prod_bench.json")
    hrs = cfg["prod"]["whole_run_hours"]
    ps = prod_summary({**cfg["prod"], "endpoint_receipt": nums.get("prod_endpoint")})
    e_pr = ev.receipt(pathlib.Path(ps["ref"]).parent, "prod_whole_run_receipt.json")
    e_endpoint = ev._write("prod_publication_times.json", ps["endpoint"], ps["ref"])
    s1 = ps["s_per_fch"]
    put("PROD_S1_WHOLE", s1, ".2f", "M", f"{e_endpoint} last file written − process start, ÷ {hrs}",
        f"Canary (PROD) GPU s per forecast hour, {hrs:g} h whole run")
    K["PROD_HOURS"] = f"{hrs:g}"
    put("PROD_SPEEDUP", cpu["prod_12rank_s_per_fch"]["value"] / s1, ".1f", "M",
        f"CPU-WRF 12 ranks {cpu['prod_12rank_s_per_fch']['value']} s/fc-h ÷ GPU", "Canary speedup vs CPU-WRF, 12 cores")
    proxy = float(bj["s1_warm_s_per_fc_h"])
    base = cfg["history_baseline"]
    put("SPEEDUP_VS_V023", base["value"] / proxy, ".0f", "M",
        f"{base['value']} s/fc-h on the starting tree ÷ S1 proxy {proxy} ({e_bench} s1_warm_s_per_fc_h)",
        "speedup vs the 2026-10-01 starting tree (3 h warm proxy)")
    solo = cfg["wn3_solo"]["warm_s_per_simh"]
    one = [x for x in nums["wn3_arms"] if x["n"] == 1]
    if one:  # final data: R4 N=1 arm (steady stepping rate of a single case)
        e1 = ev.copy(one[0]["ref"], "wn3_solo_analysis.json")
        solo = {"value": one[0]["stepping_s_per_case_h"], "label": "M", "ref": f"{e1} throughput_stepping_s_per_case_h"}
    put("WN3_SOLO", solo["value"], ".1f", solo["label"], solo["ref"], "Tenerife one case, GPU s per simulated hour (steady)")
    put("WN3_SOLO_SPEEDUP", cpu["wn3_4rank_s_per_simh"]["value"] / solo["value"], ".1f", "M",
        f"CPU-WRF 4 ranks {cpu['wn3_4rank_s_per_simh']['value']} s ÷ GPU", "Tenerife one-case speedup vs CPU-WRF, 4 cores")

    arms = nums["wn3_arms"]
    eligible = [x for x in arms if x["hours"] >= 4]
    if not eligible:
        raise ValueError("need a >=4 h fully admitted WN3 benchmark arm")
    top = max(eligible, key=lambda x: (x["n"], x["hours"]))
    arm_dir = pathlib.Path(top["dir"])
    an = arm_info(arm_dir)
    e_an = ev.copy(an["ref"], "wn3_parallel_analysis.json")
    e_st = ev._write("wn3_parallel_start.json", {"start_unix": an["start_unix"], "timing_method": an["timing_method"]}, an["ref"])
    endpoint = benchmark_endpoint(arm_dir)
    bench_hours, cases = endpoint["hours"], endpoint["case_dirs"]
    e_rec = [ev.receipt(c, f"wn3_receipt_{c.name}.json") for c in cases]
    s_bench = endpoint["s_per_case_h"]
    K["WN3_BENCH_HOURS"] = str(bench_hours)
    put("WN3_NMAX", str(top["n"]), "", "M", e_an, "Tenerife cases run in parallel (max admitted)")
    put("WN3_NMAX_WHOLE", s_bench, ".2f", "M",
        f"arm start ({e_st}) → {endpoint['timing_method']} ({', '.join(e_rec)}) ÷ (N × {bench_hours})",
        f"Tenerife GPU s per case-hour at N_max, {bench_hours} h benchmark")
    put("WN3_NMAX_SPEEDUP", cpu["wn3_3x4_s_per_case_h"]["value"] / s_bench, ".2f", "M",
        f"CPU-WRF 3 cases × 4 cores {cpu['wn3_3x4_s_per_case_h']['value']} s per case-hour ÷ GPU",
        "Tenerife throughput speedup vs CPU-WRF, 12 cores")
    if cfg.get("production_batch"):
        batch = production_batch_summary(pathlib.Path(cfg["production_batch"]["dir"]))
        batch_ref = ev.copy(batch["ref"], "wn3_production_batch.json")
        put("WN3_PRODUCTION_WHOLE", batch["s_per_case_h"], ".2f", "M", batch_ref,
            "Tenerife 72 h production batch, six cases in FIFO waves, s per case-hour")
        put("WN3_PRODUCTION_SPEEDUP", cpu["wn3_3x4_s_per_case_h"]["value"] / batch["s_per_case_h"],
            ".1f", "M", batch_ref, "72 h production throughput vs CPU-WRF, 12 cores")
    vr = max(v["vram_peak_mib"] for v in an["per_case"].values()) / 1024
    put("WN3_VRAM_GIB", vr, ".1f", "M", f"{e_an} cases/per_case.vram_peak_mib (max)", "GPU memory per Tenerife case (process peak, GiB)")
    put("WN3_FIT", str(int(man.get("wn3_fit_32gb", top["n"]))), "", "M", "admission limit of the N sweep",
        "Tenerife cases that fit one 32 GB card")
    cpu_kj = (cpu["package_w_12core"]["lo"] + cpu["package_w_12core"]["hi"]) / 2 * cpu["wn3_3x4_s_per_case_h"]["value"] / 1000
    put("CPU_KJ", cpu_kj, ".0f", cpu["package_w_12core"]["label"], cpu["package_w_12core"]["ref"],
        "CPU-WRF energy per case-hour (kJ)")
    K["CPU_ENERGY_LABEL"] = "measured package" if cpu["package_w_12core"]["label"] == "M" else "package estimate"
    if "gpu_kj_per_case_h" in top:
        e_dm = ev.dmon(arm_dir / "dmon.log", "wn3_parallel_dmon.json")
        put("GPU_KJ", top["gpu_kj_per_case_h"], ".1f", "M", f"{e_dm} energy_j ÷ (N × hours)",
            "GPU board energy per case-hour at N_max (kJ)")
        power_label = cpu["package_w_12core"]["label"]
        put("ENERGY_RATIO", cpu_kj / top["gpu_kj_per_case_h"], ".1f", power_label,
            f"CPU [{power_label}] ÷ GPU [M]", "component energy ratio CPU package/GPU board")

    # release gate: D6 summaries of the gate cases (slimmed copies)
    fails, fields, bounded, refs, all_finite = 0, set(), set(), [], True
    for c in cfg.get("gate_d6_dirs", []):
        name = "gate_d6_" + "_".join(pathlib.Path(c).parts[-2:]) + ".json"
        refs.append(ev.slim_d6(c, name))
        for d in ("d01", "d02", "d03"):
            p = pathlib.Path(c) / f"{d}.json"
            if not p.exists():
                raise ValueError(f"missing release-gate domain: {p}")
            s = gate_summary(load(p))
            fails += s["failures"]
            fields.add(s["fields"])
            bounded.add(s["bounded_fields"])
            all_finite &= s["finite_gpu"]
    if refs:
        put("GATE_N_CASES", str(len(refs)), "", "M", ", ".join(refs), "release-gate cases (24 h, all variables)")
        put("GATE_N_FIELDS", str(min(fields)), "", "M", "inventory.common_variable_count (minimum over domains)",
            "variables compared per domain")
        put("GATE_N_BOUNDED", str(min(bounded)), "", "M", "fields with supplied frozen D6 bounds (minimum over domains)",
            "variables with numerical tolerances per domain")
        put("GATE_FAILURES", str(fails), "", "M", "sum of summaries.tolerance_failure_count", "release-gate tolerance failures")
        K["GATE_FINITE"] = "all compared numeric values finite" if all_finite else "nonfinite values found"
    integrity = []
    integrity_dirs = set()
    approvals = {str(pathlib.Path(row["report"]).resolve()): row
                 for row in cfg.get("output_integrity_approvals", [])}
    if len(approvals) != len(cfg.get("output_integrity_approvals", [])):
        raise ValueError("duplicate output-integrity approval records")
    for i, path in enumerate(cfg.get("output_integrity_jsons", [])):
        report = load(path)
        result = integrity_summary(report, require_full_header=a.require_final)
        ref = ev.copy(path, f"output_integrity_{i + 1}.json")
        approval = approvals.get(str(pathlib.Path(path).resolve()))
        if approval and not result["pass"]:
            result = approved_integrity_summary(report, load(approval["exception"]), approval["approval_reference"],
                                               pathlib.Path(approval["delivery_note"]).read_text())
            exception_ref = ev.copy(approval["exception"], f"output_integrity_exception_{i + 1}.json")
            note_ref = ev._write(f"output_integrity_approval_{i + 1}.json",
                                {"approval_reference": approval["approval_reference"],
                                 "delivery_note": pathlib.Path(approval["delivery_note"]).read_text()},
                                str(approval["delivery_note"]))
            ref += f"; disclosed exception {exception_ref}; approval {note_ref}"
        integrity.append(result)
        integrity_dirs.add(str(pathlib.Path(report["gpu_dir"]).resolve()))
        status = "PASS" if result["pass"] else ("raw FAIL; approved exception" if result.get("accepted_with_exception") else "FAIL")
        prov.append(("WRF output integrity", status, "M", ref))
    accepted_integrity = integrity and all(r["pass"] or r.get("accepted_with_exception") for r in integrity)
    K["OUTPUT_INTEGRITY_STATUS"] = ("PASS on all supplied strict reports" if integrity and all(r["pass"] for r in integrity)
                                    else "raw strict FAIL retained; accepted with disclosed, manager-approved exceptions" if accepted_integrity
                                    else "FAIL; release acceptance pending" if integrity
                                    else "not yet assessed; final writer validation pending")
    idw = img / "identity_curves_wn3.json"
    if idw.exists():
        e_id = ev.copy(idw, "identity_curves_wn3.json")
        s = load(idw)["fields"]
        worst = max(((f, d, v["worst_rmse"] / v["limit"], v["last_lead_h"]) for f, dd in s.items() for d, v in dd.items()
                     if v.get("limit")), key=lambda x: x[2])
        put("WN3_72H_CASES", str(max(v["cases"] for dd in s.values() for v in dd.values())), "", "M", e_id,
            "cases in the long-lead identity curves")
        put("WN3_72H_WORST_PCT", 100 * worst[2], ".0f", "M", e_id, f"worst gate variable, % of its D6 limit, 0–{worst[3]:.0f} h")
        K["WN3_72H_WORST_FIELD"] = f"{worst[0]} {worst[1]}"
        common_lead = min(v["last_lead_h"] for dd in s.values() for v in dd.values())
        K["WN3_LAST_LEAD"] = f"{common_lead:.0f}"
        K["FIDELITY_HEADLINE"] = (f"core fields stay within frozen RMSE limits through {common_lead:.0f} hours"
                                  if worst[2] <= 1 else f"core-field differences from CPU-WRF are shown through {common_lead:.0f} hours")
    idp = img / "identity_curves_prod162.json"
    if idp.exists():
        e_idp = ev.copy(idp, "identity_curves_prod162.json")
        fields = load(idp)["fields"]
        K["PROD_LAST_LEAD"] = f"{min(v['last_lead_h'] for dd in fields.values() for v in dd.values()):.0f}"
        worst = max(((f, d, v["worst_rmse"] / v["limit"]) for f, dd in fields.items() for d, v in dd.items()
                     if v.get("limit")), key=lambda x: x[2])
        put("PROD_WORST_PCT", 100 * worst[2], ".0f", "M", e_idp, "PROD worst core-field RMSE, % of reference limit")
        K["PROD_WORST_FIELD"] = f"{worst[0]} {worst[1]}"
    if b200:
        e_b = ev.copy(img / "b200.json", "b200_extrapolation.json")
        e = b200["b200"]
        lab = f"extrapolated from B200 runs on old version ({e_b}; method docs/release/B200_EXTRAPOLATION.md)"
        put("B200_CASE_H", e["mid"]["case_h_per_h_stepping"], ",.0f", "I", lab, "B200 Tenerife case-hours per hour")
        K["B200_LO"] = format(e["lo"]["case_h_per_h_stepping"], ",.0f")
        K["B200_HI"] = format(e["cap_bandwidth"]["case_h_per_h_stepping"], ",.0f")
        put("B200_KJ", e["mid"]["kj_per_case_h"], ".1f", "I", lab, "B200 energy per case-hour (kJ)")
        put("B200_FIT", str(b200["b200_cases_fit_vram"]), "", "I", "180 GB ÷ (case peak + 1 GiB)", "Tenerife cases fitting a B200")
    for k, v in man.items():
        if k.isupper():
            K[k] = str(v)
    K.setdefault("DATE", dt.date.today().isoformat())
    K["DRAFT_NOTICE"] = "" if a.require_final else "> **Preview using prototype data. Final-tree figures and release acceptance are pending.**\n"
    K["PROVENANCE_ROWS"] = "\n".join(f"| {n} | {v} | [{l}] {r} |" for n, v, l, r in prov)

    text = pathlib.Path(a.template).read_text()
    text = re.sub(r"<!--\s*FILL KEYS.*?-->\s*", "", text, flags=re.S)
    missing = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text)) - set(K))
    if a.require_final:
        if hrs != 24:
            raise ValueError("final S1 figure needs a measured 24 h whole run")
        if missing or "prototype" in cfg["tree_label"].lower() or "prototype" in man.get("TREE", "").lower():
            raise ValueError(f"final README needs final inputs and every key; missing {missing}")
        if len(refs) != 6 or fails or not all_finite:
            raise ValueError("final README needs all 6 complete, finite, passing WN3 gate reports")
        if len(integrity) != 6 or not accepted_integrity:
            raise ValueError("final README needs strict PASS or exact, explicitly approved disclosures for all 6 WN3 cases")
        expected_dirs = {str(pathlib.Path(load(pathlib.Path(p) / "d01.json")["inputs"]["gpu_dir"]).resolve())
                         for p in cfg["gate_d6_dirs"]}
        if len(expected_dirs) != 6 or len(integrity_dirs) != 6 or integrity_dirs != expected_dirs:
            raise ValueError("output-integrity reports must describe the same WN3 outputs as D6")
        wn3_lead = cfg.get("publication_wn3_lead_h", 72)
        if wn3_lead not in (24, 72):
            raise ValueError("WN3 publication window must be the 24 h gate or complete 72 h reports")
        for name, lead, domains in (("identity_curves_wn3", wn3_lead, {"d01", "d02", "d03"}),
                                    ("identity_curves_prod162", 162, {"d01", "d02"})):
            identity_report = load(img / f"{name}.json")
            if name == "identity_curves_wn3" and (len(identity_report["cases"]) != 6
                    or len(set(identity_report["cases"].values())) != 6):
                raise ValueError("final WN3 identity needs all six distinct release-window cases")
            fields_by_dom = identity_report["fields"]
            if set(fields_by_dom) != set(GATE_FIELDS) or any(set(dd) != domains for dd in fields_by_dom.values()):
                raise ValueError(f"final README needs every gate field and domain in {name}")
            if any(v["last_lead_h"] < lead for dd in fields_by_dom.values() for v in dd.values()):
                raise ValueError(f"final README needs {name} through {lead} h")
        for filename in re.findall(r'<img src="docs/release/img/([^\"]+\.png)"', text):
            if not (img / filename).exists() or not (img / (filename[:-4] + "_dark.png")).exists():
                raise ValueError(f"final README needs both themes of {filename}")
    text = re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda m: K.get(m.group(1), m.group(0)), text)

    # GitHub dark mode: every figure has an *_dark.png twin (make_all.py) -> <picture> with a dark <source>
    def picture(m):
        src = a.img_rel.rstrip("/") + "/" + m.group(2).rsplit("/", 1)[-1]
        return (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{src[:-4]}_dark.png">'
                f'{m.group(1)}{src}{m.group(3)}</picture>')
    text = re.sub(r'(<img src=")(docs/release/img/[^"]+\.png)("[^>]*>)', picture, text)
    pathlib.Path(a.out).write_text(text)
    (pathlib.Path(a.evidence_dir) / "INDEX.json").write_text(
        json.dumps(ev.index, indent=1).replace(str(pathlib.Path.home()) + "/", "~/"))
    pathlib.Path(a.numbers_out).write_text(json.dumps({"keys": K, "provenance": prov, "missing": missing}, indent=1))
    print("filled", len(K), "keys; evidence files", len(ev.index), "; MISSING:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
