"""Compare two captured canary states leaf-by-leaf."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_meta(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _finite_stats(diff: np.ndarray) -> tuple[float, float, float]:
    if diff.size == 0:
        return 0.0, 0.0, 0.0
    abs_diff = np.abs(diff.astype(np.float64, copy=False))
    return float(np.max(abs_diff)), float(np.mean(abs_diff)), float(np.sqrt(np.mean(abs_diff * abs_diff)))


def compare(ref_npz: Path, ref_meta: Path, cand_npz: Path, cand_meta: Path) -> dict[str, object]:
    ref_info = _load_meta(ref_meta)
    cand_info = _load_meta(cand_meta)
    ref_leaves = ref_info["leaves"]
    cand_leaves = cand_info["leaves"]
    if not isinstance(ref_leaves, list) or not isinstance(cand_leaves, list):
        raise TypeError("bad metadata: leaves must be lists")
    if len(ref_leaves) != len(cand_leaves):
        raise ValueError(f"leaf count mismatch: ref={len(ref_leaves)} cand={len(cand_leaves)}")

    rows: list[dict[str, object]] = []
    exact = True
    max_abs_global = 0.0
    max_field = None
    mismatched = 0
    with np.load(ref_npz) as ref_data, np.load(cand_npz) as cand_data:
        for idx, (rmeta, cmeta) in enumerate(zip(ref_leaves, cand_leaves, strict=True)):
            rkey = rmeta["key"]
            ckey = cmeta["key"]
            if rkey != ckey or rmeta["shape"] != cmeta["shape"] or rmeta["dtype"] != cmeta["dtype"]:
                exact = False
                mismatched += 1
                row = {
                    "idx": idx,
                    "key": f"{rkey} != {ckey}",
                    "structural_mismatch": True,
                    "max_abs": float("inf"),
                    "mean_abs": float("inf"),
                    "rmse": float("inf"),
                    "byte_identical": False,
                }
                rows.append(row)
                max_abs_global = float("inf")
                max_field = row["key"]
                continue
            ref = ref_data[str(rmeta["array"])]
            cand = cand_data[str(cmeta["array"])]
            byte_identical = bool(np.array_equal(ref, cand))
            if byte_identical:
                max_abs = mean_abs = rmse = 0.0
            elif np.issubdtype(ref.dtype, np.number) and np.issubdtype(cand.dtype, np.number):
                max_abs, mean_abs, rmse = _finite_stats(cand.astype(np.float64) - ref.astype(np.float64))
            else:
                max_abs = mean_abs = rmse = float("inf")
            if not byte_identical:
                exact = False
                mismatched += 1
            if max_abs > max_abs_global:
                max_abs_global = max_abs
                max_field = rkey
            rows.append(
                {
                    "idx": idx,
                    "key": rkey,
                    "dtype": str(ref.dtype),
                    "shape": [int(v) for v in ref.shape],
                    "byte_identical": byte_identical,
                    "max_abs": max_abs,
                    "mean_abs": mean_abs,
                    "rmse": rmse,
                }
            )

    differing = [row for row in rows if not row["byte_identical"]]
    differing_sorted = sorted(differing, key=lambda row: float(row["max_abs"]), reverse=True)
    return {
        "ref_npz": str(ref_npz),
        "cand_npz": str(cand_npz),
        "ref_tag": ref_info.get("tag"),
        "cand_tag": cand_info.get("tag"),
        "exact": exact,
        "leaf_count": len(rows),
        "mismatched_leaf_count": mismatched,
        "max_abs_global": max_abs_global,
        "max_abs_field": max_field,
        "top_diffs": differing_sorted[:40],
        "all_diffs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-npz", required=True, type=Path)
    parser.add_argument("--ref-meta", required=True, type=Path)
    parser.add_argument("--cand-npz", required=True, type=Path)
    parser.add_argument("--cand-meta", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = compare(args.ref_npz, args.ref_meta, args.cand_npz, args.cand_meta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "COMPARE exact={exact} mismatched={mismatched_leaf_count}/{leaf_count} "
        "max_abs={max_abs_global:.12g} field={max_abs_field} output={output}".format(
            **result, output=args.output
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
