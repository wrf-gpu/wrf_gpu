"""Deliver W8 twin manifests to alisios cross ONLY if all three gates pass per case (CPU). Usage: deliver.py ARM_DIR TAG
Gates: d6_24h/summary.json pass (frozen D6) AND output_integrity_pass AND frozen R32-01 formal pre-run RUN_COMPLETED.
Refuses to overwrite; prints a per-case verdict line; never modifies frames (they were deflated before the manifest)."""
import hashlib, json, shutil, statistics, sys
from pathlib import Path
arm, tag = Path(sys.argv[1]), sys.argv[2]
X = Path("<DATA_ROOT>/alisios/state/manager/cross")
ROLE = {"20260227_18z": "binding (clean)", "20260502_18z": "binding (clean)", "20260220_18z": "binding",
        "20260608_18z": "binding", "20260120_18z": "binding", "20260614_18z": "validation only"}
pin = json.loads((arm.parent / "source_pin.json").read_text())
for d in sorted(arm.glob("2026*_a1")):
    issue = d.name[:-3]
    s = json.loads((d / "d6_24h" / "summary.json").read_text()) if (d / "d6_24h" / "summary.json").exists() else {}
    pre = (d / "r32_prerun_formal.json").read_text().strip().splitlines() if (d / "r32_prerun_formal.json").exists() else []
    pre_ok = bool(pre) and json.loads(pre[-1]).get("status") == "RUN_COMPLETED"
    gates = {"d6": s.get("pass") is True, "output_integrity": s.get("output_integrity_pass") is True, "r32_formal": pre_ok}
    if not all(gates.values()):
        print(issue, "NOT DELIVERED", gates)
        continue
    m = json.loads((d / f"twin_manifest_{issue}_24h.json").read_text())
    for f in m["frames"]:  # frames must be exactly what the manifest says (deflated before manifest)
        assert hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest() == f["sha256"], f["path"]
    dst, md = X / f"FROM_WRF_GPU2_WN3_TWIN_{issue}_{tag}_24h.json", X / f"FROM_WRF_GPU2_WN3_TWIN_{issue}_{tag}_24h.md"
    if dst.exists() or md.exists():
        sys.exit(f"refuse overwrite {dst}")
    shutil.copyfile(d / f"twin_manifest_{issue}_24h.json", dst)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    rates = m["wallclock"]["segment_rate_s_per_sim_h"][1:]
    md.write_text(f"""# wrf_gpu2 WN3 twin {issue} — {tag}, first 24 h (PROVISIONAL, tau<=24) — RE-DELIVERY after the LW9 HELD verdict

- Role: {ROLE.get(issue, '?')}. Manifest `{dst.name}` (sha256 {sha[:16]}), R32-01 v2 schema. Your wrapper reads `..._LW9_24h.json` by
  name: point it at this file.
- Tree: wrf_gpu2 {tag} = {pin['base']} (snapshot {pin['commit'][:12]}, src tree {pin['src_tree'][:12]}), release defaults (no flag file).
  Full WRF history (release default); writer with WRF globals and populated land/accumulated fields; fid-q2 moist hdiff + fid-cloud
  MYNN cloud fields. Exact env and argv: namelist_diff_ref. topo_shading/slope_rad active on d01-d03 (harness receipt).
- Inputs: the 7 server files read in place (SHAs in manifest). Frames: 25 per domain, deflated losslessly BEFORE this manifest
  (compress_receipt.jsonl next to wrfout/; manifest SHAs = files on disk).
- Our gates (all PASS): D6 frozen manifest (375 fields x 25 frames x 3 domains); output integrity (no CPU variable missing, no constant
  field where CPU varies, required globals present); your pinned R32-01 v2 + freeze v1 run unchanged (formal mode, lane copy) completes.
- Wallclock: 1x RTX 5090, 3 cases concurrent: median {statistics.median(rates) if rates else float('nan'):.1f} s per sim-hour per case.
- Verdict: yours, with the same freeze. 72 h (f006-f078) follows in the release showcase window.
""")
    print(issue, "DELIVERED", sha[:16], gates)
