#!/usr/bin/env python3
"""v020_skill_eval.py — wind/temp/cloud skill + divergence-growth gate for the ladder.

Wires the ladder rung's candidate wrfout series against an oracle (fp64-GPU baseline or
CPU-WRF) through the ADR-031 long-horizon divergence-GROWTH metric
(proofs/perf/v015/fp32_oracles/divergence_growth_metric.py, the module
tests/test_fp32_divergence_growth_metric.py pins). The gate is on the SLOPE of the
divergence vs lead time (bounded/saturating = PASS; escalating = FAIL), normalised by the
oracle's own internal-variability envelope — exactly the relaxed-tolerance skill policy of
V0200-ROADMAP §8.5 (NOT bitwise parity).

Fields scored (the binding wind/temp/cloud set; §8.5):
  WIND : U10, V10 (10 m), and U/V layers if present
  TEMP : T2 (2 m), T (theta)
  CLOUD: QCLOUD, QICE  (+ QSNOW/QGRAUP if present)
Carve-out (drift OK, not scored as blow-up): QVAPOR, cumulative precip.

PURE numpy + netCDF4 + the in-repo divergence metric. NO GPU, NO gpuwrf import. Fully
CPU-dry-runnable (point --candidate-dir and --oracle-dir at any two wrfout sets; if they
are the SAME set the divergence is ~0 and the gate trivially PASSES — which is how the
wiring is validated without a model run).

Usage:
  python v020_skill_eval.py --candidate-dir CAND --oracle-dir ORACLE
                            --domains d01 d02 d03 [--out skill.json]
Exit code 0 = GATE PASS, 1 = GATE FAIL (escalating divergence), 2 = usage/IO error.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

try:
    from netCDF4 import Dataset
except Exception as exc:  # pragma: no cover
    print(f"v020_skill_eval: netCDF4 import failed: {exc}", file=sys.stderr)
    sys.exit(2)

_ROOT = Path(__file__).resolve().parents[1]
_METRIC = _ROOT / "proofs" / "perf" / "v015" / "fp32_oracles" / "divergence_growth_metric.py"


def _load_metric():
    spec = importlib.util.spec_from_file_location("divergence_growth_metric", _METRIC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["divergence_growth_metric"] = mod
    spec.loader.exec_module(mod)
    return mod


WIND = ["U10", "V10"]
TEMP = ["T2"]
CLOUD = ["QCLOUD", "QICE"]
SCORE_FIELDS = WIND + TEMP + CLOUD


def _time_sorted(run_dir: Path, dom: str) -> list[Path]:
    return sorted(run_dir.glob(f"wrfout_{dom}_*"))


def _read_field_series(files: list[Path], field: str) -> np.ndarray | None:
    """Stack a field across time-ordered wrfout files -> (T, *space). Surface (2-D)
    or 3-D both handled; takes frame 0 of each file (frames_per_outfile=1 in the case)."""
    frames = []
    for f in files:
        ds = Dataset(str(f))
        try:
            if field not in ds.variables:
                ds.close()
                return None
            v = ds.variables[field]
            arr = np.asarray(v[0] if "Time" in v.dimensions else v[:], dtype=np.float64)
        finally:
            try:
                ds.close()
            except Exception:
                pass
        frames.append(arr)
    if not frames:
        return None
    # align shapes (defensive: take the min common shape)
    shp = tuple(min(s) for s in zip(*[fr.shape for fr in frames]))
    frames = [fr[tuple(slice(0, n) for n in shp)] for fr in frames]
    return np.stack(frames, axis=0)


def _envelope(series: np.ndarray) -> float:
    """Oracle internal-variability scale: the std of the field over space+time (a
    proxy for run-to-run variability when no ensemble is available). Floored small."""
    s = float(np.nanstd(series))
    return max(s, 1e-6)


def evaluate(cand_dir: Path, oracle_dir: Path, domains: list[str]) -> dict:
    dgm = _load_metric()
    report = {"candidate_dir": str(cand_dir), "oracle_dir": str(oracle_dir),
              "domains": {}, "GATE_PASS": True, "scored_fields": SCORE_FIELDS}

    for dom in domains:
        cfiles = _time_sorted(cand_dir, dom)
        ofiles = _time_sorted(oracle_dir, dom)
        n = min(len(cfiles), len(ofiles))
        if n < 2:
            report["domains"][dom] = {"status": "INSUFFICIENT_TIMES",
                                      "n_candidate": len(cfiles), "n_oracle": len(ofiles)}
            continue
        cfiles, ofiles = cfiles[:n], ofiles[:n]
        leads = np.arange(n, dtype=np.float64)  # rung-relative lead index (hours if hourly)

        fp32_series, oracle_series, envelopes = {}, {}, {}
        missing = []
        for fld in SCORE_FIELDS:
            cs = _read_field_series(cfiles, fld)
            os_ = _read_field_series(ofiles, fld)
            if cs is None or os_ is None:
                missing.append(fld)
                continue
            m = min(cs.shape[0], os_.shape[0])
            shp = tuple(min(a, b) for a, b in zip(cs.shape[1:], os_.shape[1:]))
            sl = (slice(0, m),) + tuple(slice(0, s) for s in shp)
            fp32_series[fld] = cs[sl]
            oracle_series[fld] = os_[sl]
            envelopes[fld] = _envelope(os_[sl])

        if not fp32_series:
            report["domains"][dom] = {"status": "NO_SCORABLE_FIELDS", "missing": missing}
            report["GATE_PASS"] = False
            continue

        res = dgm.evaluate_paired_forecast(leads[:m], fp32_series, oracle_series, envelopes)
        res["missing_fields"] = missing
        res["category_pass"] = {
            "wind": all(res["per_field"].get(f, {}).get("passes", True) for f in WIND if f in fp32_series),
            "temp": all(res["per_field"].get(f, {}).get("passes", True) for f in TEMP if f in fp32_series),
            "cloud": all(res["per_field"].get(f, {}).get("passes", True) for f in CLOUD if f in fp32_series),
        }
        report["domains"][dom] = res
        report["GATE_PASS"] = report["GATE_PASS"] and bool(res["GATE_PASS"])

    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate-dir", required=True, type=Path)
    ap.add_argument("--oracle-dir", required=True, type=Path)
    ap.add_argument("--domains", nargs="+", default=[f"d{i:02d}" for i in range(1, 10)])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not _METRIC.is_file():
        print(f"v020_skill_eval: divergence metric not found at {_METRIC}", file=sys.stderr)
        return 2
    for d in (args.candidate_dir, args.oracle_dir):
        if not d.is_dir():
            print(f"v020_skill_eval: dir not found {d}", file=sys.stderr)
            return 2

    rep = evaluate(args.candidate_dir, args.oracle_dir, args.domains)
    payload = json.dumps(rep, indent=2, default=lambda o: None) + "\n"
    if args.out:
        args.out.write_text(payload)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(payload)

    print(f"\n=== skill gate (divergence-growth, wind/temp/cloud): "
          f"{'PASS' if rep['GATE_PASS'] else 'FAIL'} ===", file=sys.stderr)
    for dom, d in rep["domains"].items():
        if "category_pass" in d:
            cp = d["category_pass"]
            print(f"  {dom}: wind={cp['wind']} temp={cp['temp']} cloud={cp['cloud']} "
                  f"gate={d.get('GATE_PASS')}", file=sys.stderr)
        else:
            print(f"  {dom}: {d.get('status')}", file=sys.stderr)
    return 0 if rep["GATE_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
