#!/usr/bin/env python3
"""Rebind the terminal CPU proof to JAX-free orchestration-only repairs.

The real one-hour FAST observation is intentionally expensive.  This refresh is
valid only while the exact forecast child that produced that observation is
byte-identical.  It re-runs source/plan/mutation contracts in a fresh process,
updates only those static sections, and records the prior proof digest.  Any
change to the exact child, embedded six-way observation, production tree, or
CPU/device status refuses and requires a new full oracle run instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_m0_exact_boundary_evidence as builder  # noqa: E402
import m0_exact_boundary_contract as contract  # noqa: E402
import m0_three_window_executor as executor  # noqa: E402


DEFAULT_PROOF = REPO / "proofs/v025/m0/m0_exact_boundary_cpu_evidence.json"
REFRESH_SCHEMA = "wrf_gpu2.v025.m0.exact_boundary_static_refresh.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _accelerator_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if name == "jax"
        or name.startswith("jax.")
        or name == "jaxlib"
        or name.startswith("jaxlib.")
        or name == "gpuwrf"
        or name.startswith("gpuwrf.")
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _verify_expensive_observation(payload: dict[str, Any]) -> None:
    clock_accounting = payload.get("clock_accounting") or {}
    cache = payload.get("cpu_compile_cache") or {}
    _require(
        clock_accounting.get("status") == "PASS"
        and clock_accounting.get("pre_repair_proof_sha256")
        == builder.PRE_REPAIR_PROOF_SHA256
        and clock_accounting.get("proof_work_outside_readiness_clock") is True
        and clock_accounting.get("proof_work_outside_integration_clock") is True
        and (
            float(clock_accounting.get("readiness_reduction_seconds", -1.0))
            >= float(clock_accounting.get("required_reduction_seconds", 0.0))
            > 0.0
        ),
        "the pure-readiness clock accounting is absent or blocked",
    )
    _require(
        cache.get("status") == "PASS"
        and cache.get("enabled") is True
        and cache.get("locked") is True
        and cache.get("directory")
        == str(builder.REFERENCE_CPU_CACHE_DIR.resolve())
        and cache.get("top_level_entry_bytes")
        == builder.REFERENCE_CPU_CACHE_ENTRY_BYTES
        and cache.get("top_level_entry_sha256")
        == builder.REFERENCE_CPU_CACHE_ENTRY_SHA256
        and cache.get("unchanged_during_oracle") is True
        and cache.get("device_touched") is False,
        "the matched historical CPU compile-cache proof is absent",
    )
    identity = payload.get("real_fast_cpu_identity") or {}
    checks = identity.get("checks") or {}
    _require(
        set(checks)
        == {
            "arguments_identical",
            "lowered_program_identical",
            "called_function_identical",
            "output_digest_identical",
            "returned_state_semantics_identical",
            "hours_identical",
        }
        and all(value is True for value in checks.values()),
        "the embedded six-way real FAST CPU observation is not wholly green",
    )
    local = identity.get("local_exact_compiled") or {}
    public = identity.get("public_wrapper") or {}
    local_call = local.get("call") or {}
    public_call = public.get("captured_call") or {}
    local_result = (local.get("result") or {})
    public_result = (public.get("result") or {})
    _require(
        local_call.get("argument_identity")
        == public_call.get("argument_identity"),
        "embedded exact arguments no longer agree",
    )
    _require(
        local_call.get("lowered_program_sha256")
        == public_call.get("lowered_program_sha256"),
        "embedded lowered programs no longer agree",
    )
    _require(
        local_result.get("exact_value_sha256")
        == public_result.get("exact_value_sha256"),
        "embedded exact outputs no longer agree",
    )
    _require(
        local_result.get("semantics") == public_result.get("semantics"),
        "embedded returned-state semantics no longer agree",
    )


def refresh(*, source: Path, output: Path) -> dict[str, Any]:
    _require(not _accelerator_modules(), "accelerator module imported before refresh")
    source = Path(source)
    raw_sha256 = _sha256_file(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "terminal proof is not a JSON object")
    _require(
        payload.get("schema") == builder.SCHEMA
        and payload.get("status") == builder.STATUS,
        "terminal proof schema/status changed",
    )
    _require(
        payload.get("device_policy") == "closed"
        and payload.get("device_touched") is False
        and payload.get("device_queries") == []
        and payload.get("receipts_read") == []
        and payload.get("receipts_consumed") == [],
        "terminal proof does not describe the frozen CPU-only policy",
    )
    _require(
        payload.get("gpu_windows")
        == {"W1": "MISSING", "W2": "MISSING", "W3": "MISSING"}
        and payload.get("native_pallas_verdict") == "MISSING",
        "terminal proof overclaims a device or native-Pallas verdict",
    )
    _verify_expensive_observation(payload)

    prior_sources = (payload.get("source_contract") or {}).get(
        "source_sha256"
    ) or {}
    child_name = str(contract.CHILD.relative_to(REPO))
    observed_child_sha256 = prior_sources.get(child_name)
    current_child_sha256 = _sha256_file(contract.CHILD)
    _require(
        observed_child_sha256 == current_child_sha256,
        "exact forecast child changed; a full real FAST CPU oracle rerun is required",
    )

    production_identity = builder._production_identity()
    source_contract = contract.validate_sources()
    plan = executor.build_plan()
    plan_contract = contract.validate_plan(plan)
    mutation_matrix = builder._source_attack_matrix()
    _require(
        mutation_matrix.get("status") == "PASS"
        and mutation_matrix.get("rejected") == mutation_matrix.get("total"),
        "one or more refreshed source attacks were accepted",
    )
    current_sources = source_contract["source_sha256"]
    changed_static_sources = sorted(
        name
        for name in set(prior_sources) | set(current_sources)
        if name != child_name
        and prior_sources.get(name) != current_sources.get(name)
    )

    payload["production_identity"] = production_identity
    payload["source_contract"] = source_contract
    payload["plan_contract"] = plan_contract
    payload["plan_sha256"] = hashlib.sha256(
        json.dumps(
            plan,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    payload["mutation_matrix"] = mutation_matrix
    payload["static_refresh"] = {
        "schema": REFRESH_SCHEMA,
        "refreshed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "JAX-free parent/executor/sampler/contract closure repairs after "
            "the expensive exact-child observation"
        ),
        "input_proof_path": str(source),
        "input_proof_sha256": raw_sha256,
        "expensive_observation_child_path": child_name,
        "expensive_observation_child_sha256": current_child_sha256,
        "exact_child_unchanged": True,
        "six_real_fast_checks_preserved": True,
        "changed_static_sources": changed_static_sources,
        "accelerator_modules_imported": [],
        "device_touched": False,
        "gpu_windows": {
            "W1": "MISSING",
            "W2": "MISSING",
            "W3": "MISSING",
        },
        "native_pallas_verdict": "MISSING",
    }
    _require(
        not _accelerator_modules(),
        "static refresh imported an accelerator/model module",
    )
    builder._atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_PROOF)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROOF)
    args = parser.parse_args()
    payload = refresh(source=args.input, output=args.output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output),
                "source_contract": payload["source_contract"]["status"],
                "plan_contract": payload["plan_contract"]["status"],
                "mutations_rejected": payload["mutation_matrix"]["rejected"],
                "mutations_total": payload["mutation_matrix"]["total"],
                "exact_child_unchanged": payload["static_refresh"][
                    "exact_child_unchanged"
                ],
                "accelerator_modules_imported": [],
                "device_touched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
