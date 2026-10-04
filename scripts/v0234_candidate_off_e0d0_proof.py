#!/usr/bin/env python3
"""CPU-only e0d0 candidate-off and candidate-on interface/IR proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v0234_nested_frozen_wrf_boundary_bundle_proof import run_probe, subprocess_probe


BASE_COMMIT = "e0d0b05a20b83502bbfe41e7808e9387168d7976"
PARENT_CANDIDATE = "5dbde92fbd5d9d6c44b52729581ead50d4bd542b"
FAILURE_CARRY = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/nested_frozen_bundle_e0d0b05a/"
    "failure/last-healthy-d03-step-8800.pkl"
)
FAILURE_CARRY_SHA256 = (
    "ccaf8a462506755c3d557f09c304a77e7a2722170c22fa466349cff643bca9aa"
)
FORBIDDEN_SOURCE = (
    "jax.device_get",
    "jax.pure_callback",
    "io_callback",
    "host_callback",
    "debug.callback",
)
OWNED_MODEL_FILES = (
    "src/gpuwrf/nesting/interp.py",
    "src/gpuwrf/nesting/boundary_construction.py",
    "src/gpuwrf/nesting/moving_driver.py",
    "src/gpuwrf/nesting/__init__.py",
    "src/gpuwrf/coupling/boundary_apply.py",
    "src/gpuwrf/runtime/domain_tree.py",
    "src/gpuwrf/runtime/operational_mode.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _normalize_probe(value: dict[str, object], label: str) -> dict[str, object]:
    result = dict(value)
    result["probe_root"] = label
    return result


def _source_transfer_audit() -> dict[str, object]:
    added = subprocess.check_output(
        (
            "git",
            "-C",
            str(REPO),
            "diff",
            "--unified=0",
            BASE_COMMIT,
            "--",
            *OWNED_MODEL_FILES,
        ),
        text=True,
    )
    added_lines = tuple(
        line[1:]
        for line in added.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    hits = {
        token: [line for line in added_lines if token in line]
        for token in FORBIDDEN_SOURCE
    }
    if any(hits.values()):
        raise RuntimeError(f"candidate adds callback/transfer source: {hits}")
    return {
        "baseline": BASE_COMMIT,
        "files": list(OWNED_MODEL_FILES),
        "added_lines_scanned": len(added_lines),
        "forbidden_token_hits": hits,
    }


def _production_interface() -> dict[str, object]:
    if _sha256(FAILURE_CARRY) != FAILURE_CARRY_SHA256:
        raise RuntimeError("retained production carry authority changed")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(REPO / "src"))
    import jax

    with FAILURE_CARRY.open("rb") as handle:
        carry = pickle.load(handle)
    leaves = jax.tree_util.tree_leaves(carry)
    signature = [
        {"shape": list(value.shape), "dtype": str(value.dtype)} for value in leaves
    ]
    if len(leaves) != 106:
        raise RuntimeError(f"production carry has {len(leaves)} leaves, expected 106")
    return {
        "authority": str(FAILURE_CARRY),
        "sha256": FAILURE_CARRY_SHA256,
        "leaf_count": len(leaves),
        "signature_sha256": _canonical(signature),
        "new_candidate_leaf": False,
    }


def build_proof() -> dict[str, object]:
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="v0234-e0d0-proof-") as raw_tmp:
        baseline = Path(raw_tmp) / "baseline"
        subprocess.run(
            (
                "git",
                "-C",
                str(REPO),
                "worktree",
                "add",
                "--detach",
                str(baseline),
                BASE_COMMIT,
            ),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            base_off = subprocess_probe(script, baseline, candidate_on=False)
            current_off = subprocess_probe(script, REPO, candidate_on=False)
            current_on = subprocess_probe(script, REPO, candidate_on=True)
        finally:
            subprocess.run(
                (
                    "git",
                    "-C",
                    str(REPO),
                    "worktree",
                    "remove",
                    "--force",
                    str(baseline),
                ),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

    base_off = _normalize_probe(base_off, "immutable-e0d0")
    current_off = _normalize_probe(current_off, "final-p1-repair-candidate-off")
    current_on = _normalize_probe(current_on, "final-p1-repair-candidate-on")
    compared = (
        "leaf_count",
        "input_value_sha256",
        "output_value_sha256",
        "signature_sha256",
        "treedef_sha256",
        "jaxpr_sha256",
        "jaxpr_bytes",
        "stablehlo_sha256",
        "stablehlo_bytes",
        "forbidden_hlo_tokens",
        "output_finite",
        "carry_signature_unchanged",
    )
    mismatches = {
        key: {"e0d0": base_off[key], "candidate_off": current_off[key]}
        for key in compared
        if base_off[key] != current_off[key]
    }
    if mismatches:
        raise RuntimeError(f"candidate-off differs from e0d0: {mismatches}")
    if current_off["feature_gate_active"] or not current_on["feature_gate_active"]:
        raise RuntimeError("feature gate activation audit failed")
    if any(current_on["forbidden_hlo_tokens"].values()):
        raise RuntimeError("candidate-on lowering contains a forbidden token")

    proof: dict[str, object] = {
        "schema": "gpuwrf.v0234.nested-boundary-final-p1-repair.interface-proof.v2",
        "verdict": "CANDIDATE_OFF_E0D0_EXACT_CANDIDATE_ON_INTERFACE_CLEAN",
        "authority": {
            "baseline_commit": BASE_COMMIT,
            "repair_parent": PARENT_CANDIDATE,
        },
        "candidate_off": {
            "e0d0": base_off,
            "current": current_off,
            "compared_fields": list(compared),
            "mismatches": mismatches,
            "exact": not mismatches,
        },
        "candidate_on": current_on,
        "production_interface": _production_interface(),
        "source_transfer_audit": _source_transfer_audit(),
        "gpu_commands_run": 0,
        "gpu_queries_run": 0,
    }
    proof["proof_sha256"] = _canonical(proof)
    return proof


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=("off", "on"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.probe is not None:
        root = Path(os.environ["GPUWRF_PROBE_ROOT"]).resolve()
        print(
            json.dumps(
                run_probe(root, candidate_on=args.probe == "on"),
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0

    proof = build_proof()
    rendered = json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "proof_sha256": proof["proof_sha256"],
                    "verdict": proof["verdict"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
