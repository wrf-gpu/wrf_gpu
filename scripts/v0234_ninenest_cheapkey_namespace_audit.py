#!/usr/bin/env python3
"""Seal the CPU-only RCA for fresh-namespace AOT cheap-key fragmentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

from gpuwrf.io.gen2_accessor import Gen2Run  # noqa: E402


GLOBAL_COMPONENTS = (
    "program_config_hash",
    "fn_identity_hash",
    "source_fingerprint_hash",
    "global_trace_env_hash",
    "module_const_env_hash",
    "exec_env_hash",
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


def stage_content(manifest: dict[str, Any]) -> dict[str, tuple[int, str, str]]:
    return {
        row["name"]: (int(row["bytes"]), str(row["sha256"]), str(row["resolved"]))
        for row in manifest["files"]
    }


def gpu_key(log_text: str, domain: str) -> str | None:
    pattern = rf"domain={re.escape(domain)} loaded=false source=fallback:missing .*?/k_([0-9a-f]{{64}})\.xlaexec"
    match = re.search(pattern, log_text)
    return match.group(1) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-components", type=Path, required=True)
    parser.add_argument("--warm-components", type=Path, required=True)
    parser.add_argument("--cold-stage", type=Path, required=True)
    parser.add_argument("--warm-stage", type=Path, required=True)
    parser.add_argument("--cold-stage-manifest", type=Path, required=True)
    parser.add_argument("--warm-stage-manifest", type=Path, required=True)
    parser.add_argument("--cold-log", type=Path, required=True)
    parser.add_argument("--warm-log", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cold = json.loads(args.cold_components.read_text())
    warm = json.loads(args.warm_components.read_text())
    cold_manifest = json.loads(args.cold_stage_manifest.read_text())
    warm_manifest = json.loads(args.warm_stage_manifest.read_text())
    names = [f"d{index:02d}" for index in range(1, 10)]

    cold_run = Gen2Run(args.cold_stage)
    warm_run = Gen2Run(args.warm_stage)
    provenance_rows = []
    for name in names:
        cold_source = cold_run.grid(name).source_wrfout
        warm_source = warm_run.grid(name).source_wrfout
        provenance_rows.append(
            {
                "domain": name,
                "cold_source_wrfout": cold_source,
                "warm_source_wrfout": warm_source,
                "paths_differ": cold_source != warm_source,
                "same_basename": Path(cold_source).name == Path(warm_source).name,
                "same_resolved_file": Path(cold_source).resolve() == Path(warm_source).resolve(),
            }
        )

    global_rows = {
        component: {
            "cold": cold[component],
            "warm": warm[component],
            "equal": cold[component] == warm[component],
        }
        for component in GLOBAL_COMPONENTS
    }
    domain_rows = {}
    for name in names:
        left = cold["domains"][name]
        right = warm["domains"][name]
        domain_rows[name] = {
            "cold_static_config_hash": left["static_config_hash"],
            "warm_static_config_hash": right["static_config_hash"],
            "static_config_hash_differs": left["static_config_hash"] != right["static_config_hash"],
            "carry_aval_hash_equal": left["carry_aval_hash"] == right["carry_aval_hash"],
            "cadence_equal": left["cadence"] == right["cadence"],
            "cold_cheap_key": left["cheap_key"],
            "warm_cheap_key": right["cheap_key"],
            "cheap_key_differs": left["cheap_key"] != right["cheap_key"],
        }

    aot_source = (args.repo / "src/gpuwrf/runtime/aot_cheap_key.py").read_text()
    mode_source = (args.repo / "src/gpuwrf/runtime/operational_mode.py").read_text()
    grid_source = (args.repo / "src/gpuwrf/contracts/grid.py").read_text()
    accessor_source = (args.repo / "src/gpuwrf/io/gen2_accessor.py").read_text()
    source_checks = {
        "static_hash_uses_namelist_static_aux": "_children, aux = namelist.tree_flatten()\n    return canonical_digest(aux)" in aot_source,
        "namelist_static_aux_starts_with_grid": "aux = (\n            self.grid," in mode_source,
        "grid_static_aux_contains_terrain": "self.projection,\n            self.terrain," in grid_source,
        "terrain_provenance_contains_source_path": "class TerrainProvenance:" in grid_source and "source_path: str" in grid_source,
        "grid_provenance_takes_source_wrfout": "source_path=self.source_wrfout" in accessor_source,
        "source_wrfout_takes_staged_history_path": "source_wrfout=str(first)" in accessor_source,
    }

    cold_log = args.cold_log.read_text(errors="replace")
    warm_log = args.warm_log.read_text(errors="replace")
    cold_gpu_key = gpu_key(cold_log, "d01")
    warm_gpu_key = gpu_key(warm_log, "d01")
    checks = {
        "all_six_global_components_equal": all(row["equal"] for row in global_rows.values()),
        "all_nine_carry_aval_hashes_equal": all(row["carry_aval_hash_equal"] for row in domain_rows.values()),
        "all_nine_cadences_equal": all(row["cadence_equal"] for row in domain_rows.values()),
        "all_nine_static_config_hashes_differ": all(row["static_config_hash_differs"] for row in domain_rows.values()),
        "all_nine_cpu_cheap_keys_differ": all(row["cheap_key_differs"] for row in domain_rows.values()),
        "all_stage_file_bytes_hashes_and_targets_equal": stage_content(cold_manifest) == stage_content(warm_manifest),
        "all_nine_source_paths_differ": all(row["paths_differ"] for row in provenance_rows),
        "all_nine_source_paths_have_same_basename": all(row["same_basename"] for row in provenance_rows),
        "all_nine_source_paths_resolve_to_same_file": all(row["same_resolved_file"] for row in provenance_rows),
        "source_data_flow_confirmed": all(source_checks.values()),
        "gpu_d01_keys_parsed_and_differ": bool(cold_gpu_key and warm_gpu_key and cold_gpu_key != warm_gpu_key),
        "different_python_hash_seeds_exercised": cold["pythonhashseed"] != warm["pythonhashseed"],
        "jax_cpu_only": "CpuDevice" in cold["devices"] and "CpuDevice" in warm["devices"],
    }
    verdict = (
        "NINE_NEST_CHEAPKEY_NAMESPACE_PATH_RCA_PASS"
        if all(checks.values())
        else "NINE_NEST_CHEAPKEY_NAMESPACE_PATH_RCA_FAIL"
    )
    inputs = (
        args.cold_components,
        args.warm_components,
        args.cold_stage_manifest,
        args.warm_stage_manifest,
        args.cold_log,
        args.warm_log,
    )
    payload = {
        "schema": "wrfgpu2.v0234.ninenest-cheapkey-namespace-audit.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "checks": checks,
        "cold_pythonhashseed": cold["pythonhashseed"],
        "warm_pythonhashseed": warm["pythonhashseed"],
        "global_components": global_rows,
        "domains": domain_rows,
        "provenance": provenance_rows,
        "source_checks": source_checks,
        "gpu_d01": {"cold_cheap_key": cold_gpu_key, "warm_cheap_key": warm_gpu_key},
        "input_artifacts": {str(path): sha256(path) for path in inputs},
        "interpretation": (
            "The cold and warm stages resolve to byte-identical fixture files, and process/cache/target/carry "
            "components are stable. Only static_config_hash changes. The traced data flow shows that the fresh "
            "stage pathname enters OperationalNamelist static aux through GridSpec TerrainProvenance.source_path. "
            "That provenance string is operationally inert but fragments the AOT address across fresh namespaces."
        ),
        "cpu_only": True,
        "gpu_queries": 0,
    }
    payload["canonical_payload_sha256"] = canonical(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "canonical_payload_sha256": payload["canonical_payload_sha256"]}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
