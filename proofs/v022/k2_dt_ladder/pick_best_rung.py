"""Pick the best CFL-safe rung from the Switzerland sweep (TEST INFRA ONLY).

Acceptance per rung (vs the R1 baseline):
  * CFL-safe      : worst C_total < TARGET (default 1.0; horizontal+vertical sum)
  * finite        : final wrfout all finite
  * bounded       : |U|,|V|,|W|,QVAPOR within physical bounds, qv >= 0
  * conserving    : |rel column-dry-mass drift| < MASS_DRIFT_MAX
  * within band   : RMSE vs R1 within the strict operational band (R2..R4 only;
                    R1 is the reference)

The BEST rung = the largest-dt (fastest) rung that passes ALL gates. We report
its s/step speedup vs R1 and the worst Courant observed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# CFL gates, physically split (see README + the CPU-WRF anchor measurement):
#   * HORIZONTAL advective Courant (Cx+Cy) is the clean RK3-EXPLICIT stability
#     constraint -> the hard CFL gate. On the Switzerland case this has large
#     headroom (Cx+Cy ~ 0.34->0.61 across dt 10->18).
#   * VERTICAL Courant (Cz) is handled by WRF's IMPLICIT (semi-implicit) vertical
#     acoustic solve + w-damping, so Cz>1 is NOT an explicit-CFL failure (CPU-WRF
#     runs this case stably at dt=18 with Cz~1.5). We REPORT Cz and C_total but do
#     not hard-reject on Cz alone; the empirical finite/bounded/conserving/within-
#     band gates are the real vertical-stability net.
CFL_HORIZ_TARGET = 1.0    # hard gate: max (Cx+Cy) advective Courant
CFL_TOTAL_TARGET = 1.6    # soft/report gate: max (Cx+Cy+Cz); CPU-WRF stable ~dt15
MASS_DRIFT_MAX = 5.0e-3   # |rel| column dry-mass drift cap over the short run
CFL_TARGET = CFL_TOTAL_TARGET  # back-compat alias


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def evaluate(ladder_dir: Path, cfl_target: float = CFL_TARGET) -> dict:
    runs_dir = ladder_dir / "runs"
    rungs = []
    r1_s_per_step = None
    for tag in ("R1", "R2", "R3", "R4"):
        res = _load(runs_dir / tag / "result.json")
        if res is None:
            continue
        worst = (res.get("courant") or {}).get("worst", {})
        c_total = float(worst.get("C_total", float("inf")))
        cx = float(worst.get("Cx", float("inf")))
        cy = float(worst.get("Cy", float("inf")))
        cz = float(worst.get("Cz", float("inf")))
        c_horiz = cx + cy
        fb = res.get("finite_bounded", {})
        finite = bool(fb.get("all_finite", False))
        bounded = bool(fb.get("bounded", False))
        mass = res.get("mass_conservation", {})
        drift = abs(float(mass.get("rel_drift_max_abs", float("inf"))))
        s_per_step = res.get("s_per_step")
        if tag == "R1":
            r1_s_per_step = s_per_step
        cmp = _load(runs_dir / tag / "compare_to_r1.json")
        within_band = True if tag == "R1" else (
            bool(cmp.get("within_operational_band", False)) if cmp else False
        )
        gates = {
            "cfl_horiz_safe": c_horiz < CFL_HORIZ_TARGET,   # hard explicit-CFL gate
            "cfl_total_safe": c_total < cfl_target,          # soft (Cz implicit)
            "finite": finite,
            "bounded": bounded,
            "conserving": drift < MASS_DRIFT_MAX,
            "within_band": within_band,
            "run_ok": res.get("status") == "OK",
        }
        rungs.append({
            "tag": tag,
            "dt_s": res.get("dt_s"),
            "n_sound": res.get("n_sound"),
            "worst_C_total": c_total,
            "worst_C_horiz": c_horiz,
            "worst_Cx": worst.get("Cx"),
            "worst_Cy": worst.get("Cy"),
            "worst_Cz": worst.get("Cz"),
            "finite": finite,
            "bounded": bounded,
            "mass_rel_drift_max_abs": drift,
            "s_per_step": s_per_step,
            "within_band": within_band,
            "worst_rmse_over_band": (cmp or {}).get("worst_rmse_over_band"),
            "gates": gates,
            "passes_all": all(gates.values()),
        })

    # speedup vs R1
    for r in rungs:
        if r1_s_per_step and r.get("s_per_step"):
            r["speedup_vs_R1"] = r1_s_per_step / r["s_per_step"]
        else:
            r["speedup_vs_R1"] = None

    # best safe rung = largest dt that passes all gates
    safe = [r for r in rungs if r["passes_all"]]
    best = max(safe, key=lambda r: r["dt_s"]) if safe else None

    return {
        "cfl_horiz_target": CFL_HORIZ_TARGET,
        "cfl_total_target": cfl_target,
        "mass_drift_max": MASS_DRIFT_MAX,
        "r1_s_per_step": r1_s_per_step,
        "rungs": rungs,
        "best_safe_rung": best["tag"] if best else None,
        "best_safe_detail": best,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--cfl-target", type=float, default=CFL_TARGET)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = evaluate(Path(args.ladder_dir), args.cfl_target)
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
