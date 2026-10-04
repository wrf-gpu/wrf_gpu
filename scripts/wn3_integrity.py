"""Output-integrity re-score of existing GPU history files against CPU-WRF (CPU only; see wn3_score.output_integrity).

Usage: wn3_integrity.py --cpu-dir CPU --gpu-dir GPU --out JSON [--domains d01,d02,d03]
GPU and CPU files are paired by identical filename (frames present in both); exit 1 if any domain fails.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("wn3_score", Path(__file__).with_name("wn3_score.py"))
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpu-dir", type=Path, required=True)
    ap.add_argument("--gpu-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--domains", default="d01,d02,d03")
    a = ap.parse_args()
    result = {"cpu_dir": str(a.cpu_dir), "gpu_dir": str(a.gpu_dir), "rule": "missing CPU variables + degenerate fields + WRF globals (full CPU header, every paired frame)",
              "domains": {}}
    for d in a.domains.split(","):
        names = sorted({p.name for p in a.gpu_dir.glob(f"wrfout_{d}_*") if not p.name.endswith(".tmp")}
                       & {p.name for p in a.cpu_dir.glob(f"wrfout_{d}_*")})
        if not names:
            result["domains"][d] = {"pass": False, "error": "no paired frames"}
            continue
        r = score.output_integrity([a.cpu_dir / n for n in names], [a.gpu_dir / n for n in names])
        result["domains"][d] = r
    result["pass"] = all(v["pass"] for v in result["domains"].values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"pass": result["pass"], "domains": {d: {"pass": v["pass"], "frames": v.get("frames_paired"),
          "missing_vars": len(v.get("missing_variables_vs_cpu", {})), "degenerate": len(v.get("degenerate_fields", {})),
          "missing_required_globals": len(v.get("global_attrs", {}).get("missing_required", [])),
          "missing_cpu_globals": len(v.get("global_attrs", {}).get("missing_vs_cpu", [])),
          "global_frames_failing": len(v.get("global_attrs", {}).get("by_frame", {}))} for d, v in result["domains"].items()}}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
