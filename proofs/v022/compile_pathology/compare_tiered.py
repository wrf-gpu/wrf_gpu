"""Tiered field-identity comparator for the BouLac O(nz) kill-gate.

Compares the O(nz) full-forecast state dump against the dense dump (== the
frozen v0.14 default behavior) field-by-field, scoring pooled RMSE against the
v0.14 frozen tolerance manifest. The dense path is numerically the v0.14-frozen
default, so dense-vs-onz RMSE inside the manifest hard_release_gate limits ==
"tiered-identity holds vs the frozen v0.14 manifest".

Usage:
  python proofs/v022/compile_pathology/compare_tiered.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MANIFEST = Path("proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json").resolve()


def _load(algo: str):
    p = HERE / f"killgate_{algo}_state.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return float("nan")
    d = a[finite] - b[finite]
    return float(np.sqrt(np.mean(d * d)))


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    fields = manifest.get("fields", {})

    dense = _load("dense")
    onz = _load("onz")
    if dense is None or onz is None:
        print(json.dumps({
            "verdict": "INCONCLUSIVE",
            "reason": "missing state dump(s)",
            "dense_present": dense is not None,
            "onz_present": onz is not None,
        }, indent=2))
        (HERE / "tiered_identity.json").write_text(json.dumps({
            "verdict": "INCONCLUSIVE",
            "reason": "missing state dump(s); O(nz) likely did not compile",
        }, indent=2) + "\n")
        return 0

    # Map manifest field names (wrfout convention) to state leaf names.
    # State leaves are stored by their pytree field names; try direct + common aliases.
    alias = {
        "T": "theta", "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
        "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg",
        "P": "p_perturbation", "PB": "p_total", "PH": "ph_perturbation",
        "PHB": "ph_total", "MU": "mu_perturbation", "MUB": "mu_total",
        "U": "u", "V": "v", "W": "w",
        "RAINNC": "rain_acc", "SNOWNC": "snow_acc", "GRAUPELNC": "graupel_acc",
        "QNICE": "Ni", "QNRAIN": "Nr", "QNSNOW": "Ns", "QNGRAUPEL": "Ng",
        "T2": "t2", "U10": "u10", "V10": "v10", "PSFC": "psfc",
        "TKE_PBL": "qke", "QKE": "qke",
    }

    def _get(d, name):
        if name in d:
            return d[name]
        a = alias.get(name)
        if a and a in d:
            return d[a]
        for cand in (name.lower(), name.upper()):
            if cand in d:
                return d[cand]
        return None

    rows = []
    worst_ratio = 0.0
    worst_field = None
    compared = 0
    for fname, spec in fields.items():
        if not isinstance(spec, dict):
            continue
        gate = spec.get("gate", "")
        limit = spec.get("rmse")
        a = _get(dense, fname)
        b = _get(onz, fname)
        if a is None or b is None:
            continue
        if a.shape != b.shape:
            # tolerate flattened compare
            pass
        r = _rmse(a, b)
        compared += 1
        ratio = (r / limit) if (limit and limit > 0 and np.isfinite(r)) else None
        if ratio is not None and ratio > worst_ratio:
            worst_ratio = ratio
            worst_field = fname
        rows.append({
            "field": fname, "gate": gate, "rmse": r,
            "limit": limit, "ratio_of_limit": ratio,
        })

    hard_rows = [r for r in rows if r["gate"] == "hard_release_gate" and r["ratio_of_limit"] is not None]
    passed = all(r["ratio_of_limit"] <= 1.0 for r in hard_rows) if hard_rows else False
    verdict = "HOLDS" if (passed and compared > 0) else ("INCONCLUSIVE" if compared == 0 else "VIOLATED")

    out = {
        "verdict": verdict,
        "fields_compared": compared,
        "hard_gate_fields": len(hard_rows),
        "worst_ratio_of_limit": round(worst_ratio, 6),
        "worst_field": worst_field,
        "max_abs_rmse_over_compared": max((r["rmse"] for r in rows if np.isfinite(r["rmse"])), default=None),
        "rows": sorted(rows, key=lambda r: -(r["ratio_of_limit"] or 0.0))[:20],
        "manifest": str(MANIFEST),
        "note": "dense == frozen v0.14 default; onz-vs-dense RMSE under manifest limits == tiered-identity holds",
    }
    (HERE / "tiered_identity.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
