"""Compare idealized close-gate snapshot stats between two proof JSON dirs.

The fp64-core round-off vet: a Wave-A change is round-off-neutral iff the
idealized warm-bubble + density-current (Straka) snapshot SCALAR stats match the
baseline within a tight relative tolerance.

Compares the verdict JSONs (e.g. ``warm_bubble_verdict.json`` /
``density_current_verdict.json``) snapshot scalar stats:
  theta_prime_min_k/max_k, max_abs_w_m_s, max_abs_u_m_s, mass_total_pa,
  theta_symmetry_linf_k, w_symmetry_linf_m_s, *_center_*_m, front_position_m.

Usage:
  python proofs/v0100/compare_idealized_snapshots.py BASE_DIR NEW_DIR [--rtol 1e-12]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCALAR_KEYS = (
    "theta_prime_min_k", "theta_prime_max_k", "max_abs_w_m_s", "max_abs_u_m_s",
    "positive_theta_center_x_m", "positive_theta_center_z_m",
    "cold_theta_center_x_m", "cold_theta_center_z_m", "front_position_m",
    "mass_total_pa", "theta_symmetry_linf_k", "w_symmetry_linf_m_s",
)


def _reldiff(a, b):
    if a is None or b is None:
        return 0.0 if a == b else math.inf
    a = float(a); b = float(b)
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def _compare_case(base_json: Path, new_json: Path, rtol: float):
    base = json.loads(base_json.read_text())
    new = json.loads(new_json.read_text())
    rows = []
    worst = 0.0
    nb, nn = base.get("snapshots", []), new.get("snapshots", [])
    for i, (sb, sn) in enumerate(zip(nb, nn)):
        for k in SCALAR_KEYS:
            rd = _reldiff(sb.get(k), sn.get(k))
            if rd > rtol:
                rows.append({"snapshot": i, "second": sb.get("second"), "key": k,
                             "base": sb.get(k), "new": sn.get(k), "reldiff": rd})
            worst = max(worst, rd if math.isfinite(rd) else 1e9)
    return {"base_verdict": base.get("verdict"), "new_verdict": new.get("verdict"),
            "worst_reldiff": worst, "exceedances": rows,
            "n_snapshots": min(len(nb), len(nn))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir")
    ap.add_argument("new_dir")
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    base_dir, new_dir = Path(args.base_dir), Path(args.new_dir)
    result = {"rtol": args.rtol, "cases": {}}
    overall_ok = True
    for case in ("warm_bubble", "density_current"):
        bj = base_dir / f"{case}_verdict.json"
        nj = new_dir / f"{case}_verdict.json"
        if not bj.exists() or not nj.exists():
            result["cases"][case] = {"error": f"missing {bj if not bj.exists() else nj}"}
            overall_ok = False
            continue
        cmp = _compare_case(bj, nj, args.rtol)
        result["cases"][case] = cmp
        case_ok = (cmp["new_verdict"] == "PASS" and not cmp["exceedances"])
        result["cases"][case]["round_off_neutral_within_rtol"] = case_ok
        overall_ok = overall_ok and case_ok
    result["overall_round_off_neutral"] = overall_ok
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
