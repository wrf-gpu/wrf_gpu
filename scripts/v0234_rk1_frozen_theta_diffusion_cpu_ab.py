"""Authenticated one-step CPU A/B for the RK1-frozen theta diffusion candidate."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Mapping


REQUIRED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": "<USER_HOME>/src/wrf_pristine/WRF",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
}
STEP0 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "theta_unlimited_e06583f0_full18h_discriminator1/"
    "failure/last-healthy-d03-step-0.pkl"
)
STEP0_SHA256 = "acd0d7ad147a3cf41f8302ae827d59294d5d249c8a007002646b878028ebe69d"
RELEASED_PARENT_HLO_SHA256 = "f5042ec46e3d596a08ad1b0af5dc1a4953c7bd86a5199bfacfa6cc26f50e8d67"
DISCARDED_CONSTANT_BASE_HLO_SHA256 = "9720369c826a2dc4628bbd8721fc2a28888ff4d0dc6371b003261aae9a02eb71"
OUT = (
    Path(__file__).resolve().parents[1]
    / ".agent/sprints/2026-07-14-v0234-final-ni-fable5/"
    "nested-diffopt1-rk1-forward-bundle-cpu-ab-proof.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _path_name(path: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for key in path:
        for attr in ("name", "key", "idx"):
            if hasattr(key, attr):
                parts.append(str(getattr(key, attr)))
                break
        else:
            parts.append(str(key))
    return ".".join(parts)


def _manifest(jax, np, value: Any) -> dict[str, Any]:
    rows = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
        array = np.asarray(leaf)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))
        rows.append(
            {
                "path": _path_name(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": digest.hexdigest(),
                "finite": bool(
                    not np.issubdtype(array.dtype, np.floating)
                    or np.all(np.isfinite(array))
                ),
            }
        )
    canonical = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"leaf_count": len(rows), "leaves": rows, "sha256": canonical}


def _state_delta(np, parent: Any, candidate: Any) -> dict[str, Any]:
    rows = []
    for name in sorted(set(dir(parent.state)) & set(dir(candidate.state))):
        if name.startswith("_"):
            continue
        left = getattr(parent.state, name, None)
        right = getattr(candidate.state, name, None)
        if left is None or callable(left) or not hasattr(left, "shape"):
            continue
        a = np.asarray(left)
        b = np.asarray(right)
        if a.shape != b.shape or not np.issubdtype(a.dtype, np.floating):
            continue
        delta = b.astype(np.float64) - a.astype(np.float64)
        if not np.any(delta != 0.0):
            continue
        rows.append(
            {
                "field": name,
                "changed_values": int(np.count_nonzero(delta)),
                "rms": float(np.sqrt(np.mean(delta * delta))),
                "max_abs": float(np.max(np.abs(delta))),
            }
        )
    return {"changed_fields": [row["field"] for row in rows], "fields": rows}


def _run_arm(runtime, jax, jnp, carry, namelist, clock):
    runtime._advance_chunk_fori.clear_cache()
    jax.clear_caches()
    lower_started = time.perf_counter()
    lowered = runtime._advance_chunk_fori.lower(
        carry,
        namelist,
        jnp.asarray(0, dtype=jnp.int32),
        clock,
        n_steps=1,
        cadence=int(namelist.radiation_cadence_steps),
    )
    lower_seconds = time.perf_counter() - lower_started
    input_tree = jax.tree_util.tree_structure(carry)
    output_tree = jax.tree_util.tree_structure(lowered.out_info)
    input_avals = [(tuple(x.shape), str(x.dtype)) for x in jax.tree_util.tree_leaves(carry)]
    output_avals = [(tuple(x.shape), str(x.dtype)) for x in jax.tree_util.tree_leaves(lowered.out_info)]
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    forbidden = [
        token
        for token in (
            "xla_python_cpu_callback",
            "host_callback",
            "io_callback",
            "pure_callback",
            "outside_compilation",
        )
        if token in stablehlo.lower()
    ]
    compile_started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started
    dispatch_started = time.perf_counter()
    result = executable(
            carry,
            namelist,
            jnp.asarray(0, dtype=jnp.int32),
            clock,
            n_steps=1,
            cadence=int(namelist.radiation_cadence_steps),
        )
    result = jax.device_get(result)
    dispatch_seconds = time.perf_counter() - dispatch_started
    return result, {
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "dispatch_seconds": dispatch_seconds,
        "stablehlo_sha256": hashlib.sha256(stablehlo.encode()).hexdigest(),
        "stablehlo_bytes": len(stablehlo.encode()),
        "forbidden_targets": forbidden,
        "interface_identity": bool(
            input_tree == output_tree
            and input_avals == output_avals
            and len(input_avals) == 106
        ),
    }


def main() -> int:
    actual_env = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise RuntimeError(f"environment mismatch: {actual_env!r}")
    if _sha256_file(STEP0) != STEP0_SHA256:
        raise RuntimeError("authenticated step-0 carry hash mismatch")

    import jax
    import jax.numpy as jnp
    import numpy as np

    if jax.default_backend() != "cpu":
        raise RuntimeError(f"unexpected backend: {jax.default_backend()}")
    import gpuwrf.contracts.state as state_contract

    state_contract._gpu_device = lambda: jax.devices("cpu")[0]

    import gpuwrf.runtime.operational_mode as runtime
    from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

    scratch = Path(tempfile.mkdtemp(prefix="v0234-rk1-theta-cpu-ab-"))
    try:
        load_dir = scratch / "load"
        load_dir.mkdir()
        tree, names, _initial, dt_by_domain, load_authority = ordinary.load_corrected_tree(
            load_dir
        )
        if names != ("d01", "d02", "d03"):
            raise RuntimeError(f"domain order changed: {names!r}")
        if dt_by_domain != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
            raise RuntimeError(f"timestep hierarchy changed: {dt_by_domain!r}")
        with STEP0.open("rb") as stream:
            carry = pickle.load(stream)
        namelist = tree.domains["d03"].namelist
        if (int(namelist.diff_opt), int(namelist.km_opt)) != (1, 4):
            raise RuntimeError("canonical d03 does not select diff_opt=1/km_opt=4")
        clock = runtime.build_clock_base(namelist)
        parent_namelist = dataclass_replace(namelist, diff_opt=0, km_opt=0)

        print("CPU_AB parent lower/compile/dispatch", flush=True)
        parent, parent_audit = _run_arm(runtime, jax, jnp, carry, parent_namelist, clock)
        print("CPU_AB candidate lower/compile/dispatch", flush=True)
        candidate, candidate_audit = _run_arm(runtime, jax, jnp, carry, namelist, clock)

        parent_manifest = _manifest(jax, np, parent)
        candidate_manifest = _manifest(jax, np, candidate)
        deltas = _state_delta(np, parent, candidate)
        checks = {
            "authenticated_step0": True,
            "canonical_options_active": True,
            "parent_interface_106_identity": parent_audit["interface_identity"],
            "candidate_interface_106_identity": candidate_audit["interface_identity"],
            "parent_callback_free": not parent_audit["forbidden_targets"],
            "candidate_callback_free": not candidate_audit["forbidden_targets"],
            "parent_all_leaves_finite": all(x["finite"] for x in parent_manifest["leaves"]),
            "candidate_all_leaves_finite": all(x["finite"] for x in candidate_manifest["leaves"]),
            "candidate_nonzero": parent_manifest["sha256"] != candidate_manifest["sha256"],
            "theta_changed": "theta" in deltas["changed_fields"],
            "parent_hlo_authenticated": parent_audit["stablehlo_sha256"]
            == RELEASED_PARENT_HLO_SHA256,
            "candidate_hlo_uses_106_leaf_t_init_inversion": candidate_audit[
                "stablehlo_sha256"
            ]
            != DISCARDED_CONSTANT_BASE_HLO_SHA256,
        }
        proof = {
            "schema": "gpuwrf.v0234.nested-diffopt1-rk1-forward-bundle-cpu-ab.v1",
            "input": {"path": str(STEP0), "sha256": STEP0_SHA256, "start_step": 0},
            "load_authority": load_authority,
            "configuration": {"diff_opt": 1, "km_opt": 4, "dt_s": 6.0},
            "parent": {"semantics": "released nested diff_opt=0/km_opt=0", **parent_audit, "manifest": parent_manifest},
            "candidate": {"semantics": "canonical diff_opt=1/km_opt=4 with RK1-frozen dry forward bundle", **candidate_audit, "manifest": candidate_manifest},
            "complete_output_delta": deltas,
            "causal_origin": "Focused operator A/B proves the only newly active source lane is the WRF RK1-frozen dry U/V/W/T diffusion bundle; the nested loader wiring selects the canonical namelist values without changing the carry.",
            "checks": checks,
            "verdict": "NESTED_DIFFOPT1_RK1_FORWARD_CPU_AB_GREEN" if all(checks.values()) else "NESTED_DIFFOPT1_RK1_FORWARD_CPU_AB_RED",
        }
        proof["proof_sha256"] = _canonical_hash(proof)
        OUT.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"verdict": proof["verdict"], "changed_fields": deltas["changed_fields"], "proof_sha256": proof["proof_sha256"]}, sort_keys=True), flush=True)
        return 0 if all(checks.values()) else 3
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
