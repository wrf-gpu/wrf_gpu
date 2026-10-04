#!/usr/bin/env python3
"""Bounded GPU confirmation of the v0234 late-Ni fix (manager-authorized arm).

Runs ONE ordinary production advance of the parent domain d01 from the retained
last-green step-1147 carry -- the exact carry whose next ordinary dispatch
produced the first persisted nonfinite of the accepted trajectory
(``carry.t_2ave[0,39,40]``, plus ww / mudf / muave / muts) -- and health-checks
the step-1148 output for finiteness.

Pre-fix, this single dispatch produced 7,964 nonfinite ``t_2ave`` cells. With
``_balance_ice_number`` applied, the output must be fully finite.

Scope guardrails (matches the manager authorization exactly):
  * single ordinary production dispatch, ``advance("d01", carry, 1147, 1)``;
  * output materialization OFF (no writer, no I/O);
  * no recorder / phase tap / GPU query; health check on the returned carry only;
  * requires the fresh nonce and the CUDA backend.

Invoked by scripts/v0234-opus-late-ni-confirm.sh under the shared GPU lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-opus-late-ni-fix"
CASE_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
INPUT_DIR = CASE_ROOT / "run/wrf"
LINEAGE = CASE_ROOT / "corrected_ni_rca_max_22c2bd7a"
EXACT_DIR = LINEAGE / "v0234_gpt_late_ni_df242a850d6ebab1_exact"
INPUT_CARRY = EXACT_DIR / "last-green-advance-input-d01-step-1147.pkl"
EXPECTED_INPUT_SHA256 = "0bfc31891eb1da86326a2f5965cfdb8eb981a131d8c7c6a13ad69d2b3acb8414"

# The manager-authorized fresh nonce for THIS arm.
AUTHORIZED_NONCE = "ed8f781ae62e33cef6bd7618599e8c84dd907633ba1c39a3176b9a61d6b0c3f5"
NONCE_PATTERN = re.compile(r"^[0-9a-f]{32,128}$")

# d01 native step whose ordinary next dispatch first persisted a nonfinite.
INPUT_NATIVE_STEP = 1147
TARGET_DOMAIN = "d01"
# The predecessor's first ordered bad persisted output cell.
FIRST_RED_FIELD = "t_2ave"
FIRST_RED_INDEX = (0, 39, 40)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if not NONCE_PATTERN.fullmatch(args.nonce):
        raise SystemExit("nonce must be 32-128 lowercase hex")
    if args.nonce != AUTHORIZED_NONCE:
        raise SystemExit("nonce does not match the manager-authorized value")
    if os.environ.get("GPUWRF_OPUS_LATE_NI_NONCE") != args.nonce:
        raise SystemExit("locked-environment nonce missing/mismatched")
    if args.run_dir.exists():
        raise SystemExit(f"run dir must not pre-exist: {args.run_dir}")

    input_sha = sha256_file(INPUT_CARRY)
    if input_sha != EXPECTED_INPUT_SHA256:
        raise SystemExit(f"retained step-1147 carry SHA mismatch: {input_sha}")

    import jax
    import jax.numpy as jnp
    import numpy as np

    if jax.default_backend() != "gpu" or any(d.platform != "gpu" for d in jax.devices()):
        raise SystemExit(f"CUDA backend unavailable: {jax.devices()!r}")

    import gpuwrf
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains, domain_names_for
    from gpuwrf.runtime.domain_tree import DomainTree, _prepare_operational_domain_tree_runtime
    from gpuwrf.runtime.operational_mode import _rca_array_health

    gpuwrf_path = Path(gpuwrf.__file__).resolve()
    if ROOT not in gpuwrf_path.parents:
        raise SystemExit(f"gpuwrf imported outside launch worktree: {gpuwrf_path}")
    fix_present = hasattr(
        __import__("gpuwrf.physics.thompson_column", fromlist=["_balance_ice_number"]),
        "_balance_ice_number",
    )

    args.run_dir.mkdir(parents=True, exist_ok=False)

    config = NestedPipelineConfig(
        input_dir=INPUT_DIR,
        output_dir=args.run_dir / "unused-output",
        proof_dir=args.run_dir / "unused-pipeline-proof",
        hours=18,
        max_dom=3,
        feedback=False,
        emit_initial_history=True,
    )
    names = domain_names_for(3)
    hierarchy, bundles, metadata, run_start, dt_by_domain, carries = _load_domains(config, names)
    if names != ("d01", "d02", "d03") or dt_by_domain != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
        raise SystemExit(f"domain hierarchy changed: {names}/{dt_by_domain}")
    # Accepted 1/1 moist/scalar advection options (the V10 acceptance config).
    options = {
        name: [int(bundles[name].namelist.moist_adv_opt), int(bundles[name].namelist.scalar_adv_opt)]
        for name in names
    }
    if options != {name: [1, 1] for name in names}:
        raise SystemExit(f"accepted 1/1 scalar options changed: {options}")

    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    runtime = _prepare_operational_domain_tree_runtime(tree, feedback_enabled=False)

    with INPUT_CARRY.open("rb") as stream:
        input_carry = pickle.load(stream)

    # Health of every finite-checkable leaf, before and after the single dispatch.
    scratch_fields = (
        "t_2ave", "ww", "mudf", "muave", "muts", "ph_tend",
        "u_save", "v_save", "w_save", "t_save", "ph_save", "mu_save", "ww_save", "rthraten",
    )
    state_fields = tuple(
        f for f in ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "u", "v", "w", "ph", "mu", "T", "p")
        if getattr(input_carry.state, f, None) is not None
    )

    def leaf_health(carry) -> dict[str, Any]:
        out: dict[str, Any] = {}
        first_bad_field = None
        for src, fields in (("state", state_fields), ("carry", scratch_fields)):
            container = carry.state if src == "state" else carry
            for f in fields:
                v = getattr(container, f, None)
                if v is None:
                    continue
                h = np.asarray(jax.device_get(_rca_array_health(jnp.asarray(v))))
                bad_count = int(h[0])
                out[f"{src}.{f}"] = {
                    "nonfinite_count": bad_count,
                    "first_bad_flat": int(h[1]),
                    "max_abs": float(h[4]),
                }
                if bad_count > 0 and first_bad_field is None:
                    first_bad_field = f"{src}.{f}"
        out["_first_bad_field"] = first_bad_field
        return out

    input_health = leaf_health(input_carry)
    if input_health["_first_bad_field"] is not None:
        raise SystemExit(f"retained input carry is not all-finite: {input_health['_first_bad_field']}")

    # THE single ordinary production dispatch: one d01 native step from 1147.
    output = runtime.advance(TARGET_DOMAIN, input_carry, INPUT_NATIVE_STEP, 1)
    jax.block_until_ready(output.state.theta)
    output = jax.device_get(output)

    output_health = leaf_health(output)
    target_val = float(np.asarray(getattr(output, FIRST_RED_FIELD))[FIRST_RED_INDEX])
    ni = np.asarray(output.state.Ni)
    qi = np.asarray(output.state.qi)
    theta = np.asarray(output.state.theta)

    all_finite = output_health["_first_bad_field"] is None
    passed = bool(all_finite and np.isfinite(target_val))

    proof = {
        "schema": "gpuwrf.v0234.opus-late-ni-gpu-confirm.v1",
        "arm": "opus-late-ni-gpu-confirm",
        "authorized_nonce": args.nonce,
        "gpu_actions": 1,
        "dispatch": {
            "domain": TARGET_DOMAIN,
            "input_native_step": INPUT_NATIVE_STEP,
            "n_steps": 1,
            "output_materialization": False,
        },
        "fix_present_in_tree": fix_present,
        "model_commit": os.environ.get("GPUWRF_OPUS_MODEL_COMMIT"),
        "inputs": {"last_green_input": {"path": str(INPUT_CARRY), "sha256": input_sha}},
        "first_red_reference": {
            "field": f"carry.{FIRST_RED_FIELD}",
            "index": list(FIRST_RED_INDEX),
            "pre_fix_nonfinite_count_this_field": 7964,
        },
        "output_first_red_cell_value": target_val,
        "output_all_leaves_finite": all_finite,
        "output_first_bad_field": output_health["_first_bad_field"],
        "output_ranges": {
            "Ni_max": float(np.nanmax(ni)), "Ni_finite": bool(np.isfinite(ni).all()),
            "qi_max": float(np.nanmax(qi)), "qi_finite": bool(np.isfinite(qi).all()),
            "theta_min": float(np.nanmin(theta)), "theta_max": float(np.nanmax(theta)),
            "theta_finite": bool(np.isfinite(theta).all()),
        },
        "per_leaf_health": output_health,
        "verdict": (
            "OPUS_LATE_NI_GPU_CONFIRM_GREEN" if passed else "OPUS_LATE_NI_GPU_CONFIRM_RED"
        ),
    }
    proof["canonical_sha256"] = canonical_sha256(proof)

    out_path = SPRINT / "OPUS_LATE_NI_GPU_CONFIRM_RESULT.json"
    out_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    (args.run_dir / "result.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(json.dumps(proof, indent=2, sort_keys=True))
    print(f"\nverdict: {proof['verdict']}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
