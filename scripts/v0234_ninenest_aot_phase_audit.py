#!/usr/bin/env python3
"""CPU-only audit of the two fused AOT phases from the v0234 nine-nest arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax

from gpuwrf.contracts.state import State


BOUNDARY_FIELDS = (
    "u_bdy",
    "v_bdy",
    "theta_bdy",
    "qv_bdy",
    "ph_bdy",
    "mu_bdy",
    "w_bdy",
    "p_bdy",
    "pb_bdy",
    "phb_bdy",
    "mub_bdy",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "canonical_payload_sha256"}
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def aval_tuple(record: dict[str, Any]) -> tuple[tuple[int, ...], str, bool]:
    return tuple(record["shape"]), str(record["dtype"]), bool(record["weak_type"])


def path_indices(path: tuple[Any, ...]) -> list[int]:
    result = []
    for key in path:
        value = getattr(key, "idx", getattr(key, "key", None))
        if isinstance(value, int):
            result.append(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fused_dir = args.cache_root / "aot/0.23.3-jax0.10.0-jaxlib0.10.0-cuda_sm120/fused_d02"
    meta_paths = sorted(fused_dir.glob("k_*.meta"), key=lambda path: path.stat().st_mtime_ns)
    metas = []
    for path in meta_paths:
        with path.open("rb") as handle:
            metas.append(pickle.load(handle))  # noqa: S301 - trusted, locally produced AOT metadata

    errors: list[str] = []
    if len(metas) != 2:
        errors.append(f"expected exactly two fused metadata files, found {len(metas)}")

    phases = []
    differences = []
    if len(metas) == 2:
        left, right = metas
        placeholder_tree = jax.tree_util.tree_unflatten(
            left.in_tree, list(range(len(left.in_avals)))
        )
        indexed_paths, _ = jax.tree_util.tree_flatten_with_path(placeholder_tree)
        path_by_leaf = {int(leaf): path for path, leaf in indexed_paths}
        kept = set(int(index) for index in left.kept_var_idx or ())
        for index, (before, after) in enumerate(zip(left.in_avals, right.in_avals, strict=True)):
            before_tuple = aval_tuple(before)
            after_tuple = aval_tuple(after)
            if before_tuple == after_tuple:
                continue
            path = path_by_leaf[index]
            indices = path_indices(path)
            child_index = indices[2] if len(indices) >= 5 else None
            state_index = indices[-1] if indices else None
            field = (
                State.__slots__[state_index]
                if isinstance(state_index, int) and 0 <= state_index < len(State.__slots__)
                else None
            )
            differences.append(
                {
                    "leaf_index": index,
                    "jax_path": jax.tree_util.keystr(path),
                    "child_index_zero_based": child_index,
                    "state_flat_index": state_index,
                    "state_field": field,
                    "phase_a": {
                        "shape": list(before_tuple[0]),
                        "dtype": before_tuple[1],
                        "weak_type": before_tuple[2],
                    },
                    "phase_b": {
                        "shape": list(after_tuple[0]),
                        "dtype": after_tuple[1],
                        "weak_type": after_tuple[2],
                    },
                    "in_kept_var_idx": index in kept,
                }
            )

        for path, meta in zip(meta_paths, metas, strict=True):
            blob = path.with_suffix(".xlaexec")
            phases.append(
                {
                    "meta_path": str(path),
                    "meta_bytes": path.stat().st_size,
                    "meta_sha256": sha256(path),
                    "blob_path": str(blob),
                    "blob_bytes": blob.stat().st_size if blob.is_file() else None,
                    "blob_sha256_from_authenticated_meta": meta.blob_sha256,
                    "cheap_key": meta.cheap_key,
                    "hlo_sha256": meta.hlo_sha256,
                    "input_leaf_count": len(meta.in_avals),
                    "kept_var_count": len(meta.kept_var_idx or ()),
                    "mtime_epoch_ns": path.stat().st_mtime_ns,
                }
            )

        fields = sorted({row["state_field"] for row in differences if row["state_field"]})
        children = sorted(
            {row["child_index_zero_based"] for row in differences if row["child_index_zero_based"] is not None}
        )
        first_dim_only = all(
            row["phase_a"]["shape"][0] == 1
            and row["phase_b"]["shape"][0] == 2
            and row["phase_a"]["shape"][1:] == row["phase_b"]["shape"][1:]
            and row["phase_a"]["dtype"] == row["phase_b"]["dtype"]
            and row["phase_a"]["weak_type"] == row["phase_b"]["weak_type"]
            for row in differences
        )
        checks = {
            "two_fused_phases": len(metas) == 2,
            "same_input_tree": left.in_tree == right.in_tree,
            "same_output_tree": left.out_tree == right.out_tree,
            "same_stablehlo_sha256": left.hlo_sha256 == right.hlo_sha256,
            "same_kept_var_idx": left.kept_var_idx == right.kept_var_idx,
            "input_leaf_count_888": len(left.in_avals) == len(right.in_avals) == 888,
            "kept_var_count_811": len(left.kept_var_idx or ()) == 811,
            "differing_aval_count_77": len(differences) == 77,
            "zero_differing_avals_consumed": not any(row["in_kept_var_idx"] for row in differences),
            "differences_are_all_11_boundary_fields": fields == sorted(BOUNDARY_FIELDS),
            "differences_cover_all_seven_children": children == list(range(7)),
            "differences_only_time_axis_1_to_2": first_dim_only,
        }
    else:
        checks = {"two_fused_phases": False}
        fields = []
        children = []

    verdict = "AOT_PHASE_AUDIT_PASS" if not errors and all(checks.values()) else "AOT_PHASE_AUDIT_FAIL"
    payload = {
        "schema": "wrfgpu2.v0234.ninenest-aot-phase-audit.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "cache_root": str(args.cache_root),
        "checks": checks,
        "errors": errors,
        "phase_count": len(phases),
        "phases": phases,
        "differing_aval_count": len(differences),
        "differing_kept_aval_count": sum(row["in_kept_var_idx"] for row in differences),
        "differing_state_fields": fields,
        "differing_child_indices_zero_based": children,
        "differences": differences,
        "interpretation": (
            "The persistent cheap key and in-process aval signature distinguish the two naive "
            "input shapes, but XLA consumes neither version of the 77 changed leaves. The "
            "identical StableHLO digest proves that these boundary-time-axis changes do not "
            "change the fused program. This is a bounded redundant cold-capture cost, not a "
            "numerical mismatch."
        ),
        "cpu_only": True,
        "gpu_queries": 0,
    }
    payload["canonical_payload_sha256"] = canonical(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "out": str(args.out),
                "canonical_payload_sha256": payload["canonical_payload_sha256"],
            },
            indent=2,
        )
    )
    return 0 if verdict == "AOT_PHASE_AUDIT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
