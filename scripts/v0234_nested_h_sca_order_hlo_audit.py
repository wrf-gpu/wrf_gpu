"""Lower-only, target-aware CPU HLO audit for the h_sca order candidate."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "nested-h-sca-order-lowered-hlo-audit.json"
CANDIDATE = "395fb800df0bb619462db8e312f734ca0c538161"
CANDIDATE_TREE = "23b6cc942d4b0b0b32cf86bf88ebdec1c693fd4e"
STEP0 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_scalar_sixth_order_18d97595_full18h_discriminator1/"
    "failure/last-healthy-d03-step-0.pkl"
)
STEP0_SHA256 = "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
NAMELIST_CACHE = Path("/tmp/v0234-scalar-controls-d03-namelist-dc3fecdd.pkl")
NAMELIST_CACHE_SHA256 = "87cd017eb7a8e1194e5d1702b979f004e7a2533d549106e2d44bfb5508365e69"
ALLOWED_CPU_DEVICE_LIBRARY_TARGETS = frozenset({"lapack_dgtsv_ffi"})
FORBIDDEN_TEXT = (
    "xla_python_cpu_callback",
    "host_callback",
    "io_callback",
    "pure_callback",
    "debug_callback",
    "outside_compilation",
    "host_transfer",
    "device_to_host",
    "host_send",
    "host_recv",
    "stablehlo.send",
    "stablehlo.recv",
    "mhlo.send",
    "mhlo.recv",
    "infeed",
    "outfeed",
)
FORBIDDEN_TARGET_FRAGMENTS = (
    "python",
    "callback",
    "host",
    "outside",
    "send",
    "recv",
    "infeed",
    "outfeed",
    "debug",
    "io_",
    "pure_",
)
CUSTOM_CALL = re.compile(
    r'\bstablehlo\.custom_call\s+@(?:"([^"\n]+)"|([A-Za-z_.$][A-Za-z0-9_.$-]*))'
)
REQUIRED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": "<USER_HOME>/src/wrf_pristine/WRF",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return _text_sha(json.dumps(clean, sort_keys=True, separators=(",", ":")))


def _git(*args: str) -> str:
    return subprocess.check_output(("git", "-C", str(ROOT), *args), text=True).strip()


def _extract(stablehlo: str) -> dict[str, Any]:
    syntax_offsets = [
        match.start() for match in re.finditer(r"\bstablehlo\.custom_call\b", stablehlo)
    ]
    occurrences = []
    for index, match in enumerate(CUSTOM_CALL.finditer(stablehlo)):
        target = match.group(1) or match.group(2)
        start = max(0, match.start() - 256)
        end = min(len(stablehlo), match.end() + 1024)
        snippet = stablehlo[start:end]
        occurrences.append(
            {
                "index": index,
                "target": target,
                "byte_offset": match.start(),
                "line": stablehlo.count("\n", 0, match.start()) + 1,
                "snippet": snippet,
                "snippet_sha256": _text_sha(snippet),
            }
        )
    return {
        "stablehlo_sha256": _text_sha(stablehlo),
        "stablehlo_bytes": len(stablehlo.encode()),
        "custom_call_syntax_count": len(syntax_offsets),
        "parsed_custom_call_count": len(occurrences),
        "extraction_complete": len(syntax_offsets) == len(occurrences),
        "targets": sorted({row["target"] for row in occurrences}),
        "occurrences": occurrences,
    }


def evaluate_policy(stablehlo: str) -> dict[str, Any]:
    """Fail closed on callbacks, transfers, malformed, or unknown targets."""

    extraction = _extract(stablehlo)
    lowered_text = stablehlo.lower()
    forbidden_text = sorted({token for token in FORBIDDEN_TEXT if token in lowered_text})
    forbidden_targets = sorted(
        target
        for target in extraction["targets"]
        if any(fragment in target.lower() for fragment in FORBIDDEN_TARGET_FRAGMENTS)
    )
    unknown_targets = sorted(
        set(extraction["targets"]) - set(ALLOWED_CPU_DEVICE_LIBRARY_TARGETS)
    )
    return {
        "passed": extraction["extraction_complete"]
        and not forbidden_text
        and not forbidden_targets
        and not unknown_targets,
        "extraction": extraction,
        "forbidden_text_tokens": forbidden_text,
        "forbidden_custom_targets": forbidden_targets,
        "unknown_custom_targets": unknown_targets,
    }


def main() -> int:
    actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise RuntimeError(f"environment mismatch: {actual_env!r}")
    if _sha256(STEP0) != STEP0_SHA256:
        raise RuntimeError("authenticated Step0 mismatch")
    if _sha256(NAMELIST_CACHE) != NAMELIST_CACHE_SHA256:
        raise RuntimeError("authenticated d03 namelist cache mismatch")
    if _git("rev-parse", CANDIDATE) != CANDIDATE:
        raise RuntimeError("candidate commit mismatch")
    if _git("rev-parse", f"{CANDIDATE}^{{tree}}") != CANDIDATE_TREE:
        raise RuntimeError("candidate tree mismatch")
    if _git("diff", "--name-only", CANDIDATE, "--", "src/gpuwrf"):
        raise RuntimeError("model source changed after candidate commit")

    import jax
    import jax.numpy as jnp

    if jax.default_backend() != "cpu":
        raise RuntimeError(f"unexpected backend: {jax.default_backend()}")
    import gpuwrf.contracts.state as state_contract

    state_contract._gpu_device = lambda: jax.devices("cpu")[0]
    import gpuwrf.runtime.operational_mode as runtime

    with STEP0.open("rb") as stream:
        carry = pickle.load(stream)
    with NAMELIST_CACHE.open("rb") as stream:
        namelist, load_authority = pickle.load(stream)
    if (
        int(namelist.h_sca_adv_order),
        int(namelist.moist_adv_opt),
        int(namelist.scalar_adv_opt),
    ) != (5, 0, 0):
        raise RuntimeError("candidate d03 scalar controls changed")
    clock = runtime.build_clock_base(namelist)
    runtime._advance_chunk_fori.clear_cache()
    jax.clear_caches()
    print("H_SCA_HLO_AUDIT lower-only", flush=True)
    lowered = runtime._advance_chunk_fori.lower(
        carry,
        namelist,
        jnp.asarray(0, dtype=jnp.int32),
        clock,
        n_steps=1,
        cadence=int(namelist.radiation_cadence_steps),
    )
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    policy = evaluate_policy(stablehlo)
    extraction = policy["extraction"]
    forbidden_text = policy["forbidden_text_tokens"]
    forbidden_targets = policy["forbidden_custom_targets"]
    unknown_targets = policy["unknown_custom_targets"]
    input_leaves = jax.tree_util.tree_leaves(carry)
    output_leaves = jax.tree_util.tree_leaves(lowered.out_info)
    input_avals = [(tuple(x.shape), str(x.dtype)) for x in input_leaves]
    output_avals = [(tuple(x.shape), str(x.dtype)) for x in output_leaves]

    import jax._src.lax.linalg as lax_linalg
    import inspect

    linalg_source = Path(inspect.getsourcefile(lax_linalg) or "")
    export_fixture = linalg_source.parents[1] / (
        "internal_test_util/export_back_compat_test_data/"
        "cpu_tridiagonal_solve_lapack_gtsv.py"
    )
    source_text = linalg_source.read_text()
    fixture_text = export_fixture.read_text()
    checks = {
        "candidate_source_immutable": True,
        "authenticated_step0_and_namelist": True,
        "lower_only_no_compile_or_dispatch": True,
        "ordinary_interface_106_identity": len(input_avals) == 106
        and input_avals == output_avals
        and jax.tree_util.tree_structure(carry)
        == jax.tree_util.tree_structure(lowered.out_info),
        "custom_call_extraction_complete": extraction["extraction_complete"],
        "only_exact_cpu_device_library_target": extraction["targets"]
        == ["lapack_dgtsv_ffi"]
        and not unknown_targets,
        "no_callback_or_transfer_text": not forbidden_text,
        "no_forbidden_custom_target": not forbidden_targets,
        "jax_cpu_target_source_authenticated": (
            'prepare_lapack_call("gtsv_ffi", b_aval.dtype)' in source_text
            and "custom_call_targets=['lapack_dgtsv_ffi']" in fixture_text
        ),
        "all_retained_snippets_hashed": all(
            _text_sha(row["snippet"]) == row["snippet_sha256"]
            for row in extraction["occurrences"]
        ),
    }
    proof = {
        "schema": "gpuwrf.v0234.nested-h-sca-order-lowered-hlo-audit.v1",
        "candidate_commit": CANDIDATE,
        "candidate_tree": CANDIDATE_TREE,
        "environment": actual_env,
        "inputs": {
            "step0": {"path": str(STEP0), "sha256": STEP0_SHA256},
            "d03_namelist_cache": {
                "path": str(NAMELIST_CACHE),
                "sha256": NAMELIST_CACHE_SHA256,
                "load_authority": load_authority,
            },
        },
        "configuration": {
            "h_sca_adv_order": 5,
            "moist_adv_opt": 0,
            "scalar_adv_opt": 0,
            "leaf_count": 106,
        },
        "lowering": {
            "compile_calls": 0,
            "dispatch_calls": 0,
            "extraction": extraction,
            "allowed_cpu_device_library_targets": sorted(
                ALLOWED_CPU_DEVICE_LIBRARY_TARGETS
            ),
            "forbidden_text_tokens": forbidden_text,
            "forbidden_custom_targets": forbidden_targets,
            "unknown_custom_targets": unknown_targets,
        },
        "jax_target_authority": {
            "linalg_source": {"path": str(linalg_source), "sha256": _sha256(linalg_source)},
            "export_fixture": {"path": str(export_fixture), "sha256": _sha256(export_fixture)},
            "classification": "lapack_dgtsv_ffi is JAX's CPU fp64 LAPACK tridiagonal-solve device-library target, not a Python/host callback",
        },
        "checks": checks,
        "verdict": (
            "NESTED_H_SCA_ORDER_HLO_AUDIT_GREEN"
            if all(checks.values())
            else "NESTED_H_SCA_ORDER_HLO_AUDIT_RED"
        ),
    }
    proof["proof_sha256"] = _canonical(proof)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUT)
    print(
        json.dumps(
            {
                "verdict": proof["verdict"],
                "proof_sha256": proof["proof_sha256"],
                "stablehlo_sha256": extraction["stablehlo_sha256"],
                "custom_call_targets": extraction["targets"],
                "forbidden_text": forbidden_text,
                "unknown_targets": unknown_targets,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if all(checks.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
