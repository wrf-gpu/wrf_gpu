#!/usr/bin/env python3
"""Seal the ratified v0234 radiation re-admission proof chain.

This is a backend-dark audit of hash-pinned real-WRF A/B proofs, the fresh
paired d03 TSK proof, current production wiring, and the new channel gate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parent.parent
OLD = REPO / ".agent/sprints/2026-07-20-v0234-gpt-land-tsk-provenance"
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-mf-seam-closure"
FILES = {
    "lw_interface_real_wrf": OLD / "rrtmg-interface-real-wrf-proof.json",
    "sw_interface_real_wrf": OLD / "rrtmg-sw-interface-real-wrf-proof.json",
    "interface_tsk": OLD / "rrtmg-composed-tsk-proof.json",
    "clwrf_lw_real_wrf": OLD / "rrtmg-clwrf-production-real-wrf-proof.json",
    "clwrf_sw_real_wrf": OLD / "rrtmg-clwrf-production-sw-real-wrf-proof.json",
    "clwrf_tsk": OLD / "rrtmg-clwrf-production-tsk-proof.json",
    "o3_real_wrf": OLD / "rrtmg-o3-real-wrf-proof.json",
    "o3_tsk": OLD / "rrtmg-o3-tsk-proof.json",
    "combined_tsk": SPRINT / "combined-radiation-tsk-proof.json",
    "channel_gate": SPRINT / "channel-decomposed-sp2-gate.json",
    "coupler": REPO / "src/gpuwrf/coupling/physics_couplers.py",
    "ozone": REPO / "src/gpuwrf/physics/wrf_cam_ozone.py",
    "clwrf": REPO / "src/gpuwrf/physics/wrf_clwrf_ghg.py",
    "lw": REPO / "src/gpuwrf/physics/rrtmg_lw.py",
    "sw": REPO / "src/gpuwrf/physics/rrtmg_sw.py",
}
EXPECTED = {
    "lw_interface_real_wrf": "e2a1733ff7a63e4063017ae1a8d5e704bb90e2a7ed1f6446432868c5cf09bc0c",
    "sw_interface_real_wrf": "cbbde365b3c429c4decae7ce4fa7c8df710b23bbcf426146c5c46d2288156ecb",
    "interface_tsk": "728d8274d5d0aad3a8ff8e2cee98263e71f49c57d08f4c09de9e64337b474c71",
    "clwrf_lw_real_wrf": "f9e984f4b09a8bb3a226b167c59e269389953f70fe7e5627af2abfdee32628bc",
    "clwrf_sw_real_wrf": "d6ec82fda4bf75dfef95ccfaa9dfb007c93accf71791971e11c2ed95417417cd",
    "clwrf_tsk": "bff763c4fea6cbe5b32fe7ca0d3054a55692badb560b77ae4096aee7f875fa28",
    "o3_real_wrf": "333d716259c1d012b7db6d7092717611d3c708113ddf247683161872d753e4d9",
    "o3_tsk": "3505800030964c8706b631d9ed009a1194696854f4471cebe3798a1986eef6ac",
    "combined_tsk": "6cbc55d2c0f9018f90110abb963e3fe75e7d307b80f873cba967ddc12d5236a2",
    "channel_gate": "e73bbcf104575676ea3482838577e87a1914954249919fbe167e642a10168c25",
    "coupler": "84440c116f2f8309fa76390fdeea15ad55c58bb05c8bf783af79895411c537a2",
    "ozone": "ceb27bec0bc51822b7c15528dccb269491037b510553835aa9c5d7e7e75a68a5",
    "clwrf": "7bd666a5b55d43cbf79a6c339fd373b4138f269a0ce37edc65b165c3c13a4a12",
    "lw": "090ea05af5e5ffe1e2f3dba0c9c8b1680a5253044e4f4fd4440a41442fd4b1a9",
    "sw": "8381820a672f6c0d415973cbbc5e458588d19c93c7ad39cc899d0e4e663276f0",
}


class ProofFailure(RuntimeError):
    """Fail-closed source or proof authority failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Mapping[str, Any]) -> str:
    body = {
        key: item
        for key, item in value.items()
        if key != "canonical_payload_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def load(name: str) -> dict[str, Any]:
    path = FILES[name]
    actual = sha256_file(path)
    if actual != EXPECTED[name]:
        raise ProofFailure(f"HASH:{name}:{actual}")
    value = json.loads(path.read_text(encoding="utf-8"))
    declared = value.get("canonical_payload_sha256")
    if declared is not None and canonical(value) != declared:
        raise ProofFailure(f"CANONICAL:{name}")
    return value


def git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=False,
        text=not binary,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        error = result.stderr.decode() if binary else result.stderr
        raise ProofFailure(f"GIT:{' '.join(args)}:{error.strip()}")
    return result.stdout if binary else result.stdout.strip()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ProofFailure(f"OUTPUT_NOT_FRESH:{path}")
    payload = dict(value)
    payload["canonical_payload_sha256"] = canonical(payload)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ProofFailure(f"TEMP_NOT_FRESH:{temporary}")
    with temporary.open("xb") as stream:
        stream.write(
            (
                json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
                + "\n"
            ).encode()
        )
        stream.flush()
    os.link(temporary, path)
    temporary.unlink()


def source_gate() -> dict[str, Any]:
    for name in ("coupler", "ozone", "clwrf", "lw", "sw"):
        actual = sha256_file(FILES[name])
        if actual != EXPECTED[name]:
            raise ProofFailure(f"SOURCE_HASH:{name}:{actual}")
    coupler = FILES["coupler"].read_text(encoding="utf-8")
    tokens = (
        "_wrf_hydrostatic_pressure_profiles_from_state(",
        "_wrf_phy_prep_temperature_interfaces(",
        "sw_pressure_interfaces = radiation_pressure_interfaces",
        "lw_pressure_interfaces = radiation_pressure_interfaces",
        "wrf_cam_ozone_profile(",
        "ghg = clwrf_ssp245_gases_for_time(ghg_time)",
        "ozone_vmr=ozone_vmr",
        "cfc11_vmr=None if ghg is None else ghg.cfc11_vmr",
        "cfc12_vmr=None if ghg is None else ghg.cfc12_vmr",
    )
    if not all(token in coupler for token in tokens):
        raise ProofFailure("PRODUCTION_WIRING_TOKENS")
    return {
        "passed": True,
        "production_population": [
            "hydrostatic P3D/P8W and phy_prep T8W on both LW and SW",
            "exact o3input=2 CAM O3RAD on both LW and SW",
            "run-date CLWRF SSP245 CO2/N2O/CH4/CFC11/CFC12 metadata",
        ],
        "coupler_tokens": list(tokens),
        "source_sha256": {
            str(FILES[name].relative_to(REPO)): EXPECTED[name]
            for name in ("coupler", "ozone", "clwrf", "lw", "sw")
        },
    }


def validate_chain() -> dict[str, dict[str, Any]]:
    proofs = {
        name: load(name)
        for name in (
            "lw_interface_real_wrf",
            "sw_interface_real_wrf",
            "interface_tsk",
            "clwrf_lw_real_wrf",
            "clwrf_sw_real_wrf",
            "clwrf_tsk",
            "o3_real_wrf",
            "o3_tsk",
            "combined_tsk",
            "channel_gate",
        )
    }
    lw = proofs["lw_interface_real_wrf"]
    sw = proofs["sw_interface_real_wrf"]
    interface_tsk = proofs["interface_tsk"]
    clw = proofs["clwrf_lw_real_wrf"]
    csw = proofs["clwrf_sw_real_wrf"]
    o3 = proofs["o3_real_wrf"]
    o3_tsk = proofs["o3_tsk"]
    combined = proofs["combined_tsk"]
    channel = proofs["channel_gate"]
    if not (
        lw.get("status") == "PASS"
        and lw.get("is_self_compare") is False
        and lw["gates"]["real_wrf_glw_strictly_improves"] is True
        and sw.get("status") == "PASS"
        and sw.get("is_self_compare") is False
        and sw["gates"]["real_wrf_swdnb_rms_and_max_strictly_improve"]
        is True
        and interface_tsk.get("passed") is True
        and interface_tsk["gates"]["land_tsk_rms_and_max_strictly_improve"]
        is True
        and interface_tsk["gates"]["water_bitwise_invariant"] is True
        and clw.get("status") == "PASS"
        and clw.get("is_self_compare") is False
        and clw["gates"]["full_composition_real_wrf_glw_strictly_improves"]
        is True
        and csw.get("status") == "PASS"
        and csw.get("is_self_compare") is False
        and csw["gates"][
            "real_wrf_swdnb_production_composition_strictly_improves"
        ]
        is True
        and o3.get("status") == "PASS"
        and o3.get("is_self_compare") is False
        and o3["gates"]["real_wrf_glw_strictly_improves"] is True
        and o3["gates"]["real_wrf_swdnb_strictly_improves"] is True
        and o3_tsk.get("passed") is True
        and o3_tsk["gates"]["land_tsk_rms_and_max_strictly_improve"] is True
        and o3_tsk["gates"]["water_bitwise_invariant"] is True
        and combined.get("passed") is True
        and combined.get("verdict")
        == "WRF_PRODUCTION_STRICTLY_IMPROVES_D03_TSK"
        and combined["tsk_parity"]["land_rms_and_max_strictly_improve"]
        is True
        and combined["tsk_parity"]["water_bitwise_invariant"] is True
        and channel.get("passed") is True
        and channel.get("verdict")
        == "MASS_FLUX_SEAM_SOURCE_LOCALIZED_FIX_PROVEN"
    ):
        raise ProofFailure("UPSTREAM_GATE")
    return proofs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if git("rev-parse", "HEAD") != args.approved_head:
            raise ProofFailure("APPROVED_HEAD_DRIFT")
        if git("status", "--porcelain", "--untracked-files=no"):
            raise ProofFailure("TRACKED_WORKTREE_NOT_CLEAN")
        relative = Path(__file__).resolve().relative_to(REPO).as_posix()
        script_bytes = Path(__file__).resolve().read_bytes()
        if script_bytes != git("show", f"HEAD:{relative}", binary=True):
            raise ProofFailure("SCRIPT_NOT_HEAD")
        proofs = validate_chain()
        source = source_gate()

        lw = proofs["lw_interface_real_wrf"]["real_wrf_glw"]
        sw = proofs["sw_interface_real_wrf"]["real_wrf_swdnb"]
        clw = proofs["clwrf_lw_real_wrf"]["real_wrf_glw"]
        csw = proofs["clwrf_sw_real_wrf"]["real_wrf_swdnb"]
        o3 = proofs["o3_real_wrf"]
        combined = proofs["combined_tsk"]["tsk_parity"]
        channel = proofs["channel_gate"]["channel_decomposed_gate"]
        proof: dict[str, Any] = {
            "schema": "wrfgpu2-v0234-radiation-readmission-proof-v1",
            "generated_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "passed": True,
            "verdict": "RADIATION_READMISSION_COMPONENT_AND_CHANNEL_GATE_PROVEN",
            "authority": {
                "approved_head": args.approved_head,
                "script": {
                    "path": relative,
                    "sha256": hashlib.sha256(script_bytes).hexdigest(),
                    "git_blob": git("rev-parse", f"HEAD:{relative}"),
                },
                "gpu_actions": 0,
                "wrf_or_mpi_executions": 0,
                "synthetic_references": 0,
                "jax_vs_jax_truth_claims": 0,
            },
            "production_source_gate": source,
            "real_wrf_component_ab": {
                "dynamic_lw_glw_rms": {
                    "before": lw[
                        "accepted_top_buffer_midpoint_interfaces"
                    ]["rms"],
                    "after": lw["wrf_p8w_t8w_interfaces"]["rms"],
                    "strictly_improves": lw["strictly_improves"],
                },
                "dynamic_sw_swdnb_rms": {
                    "before": sw["hydrostatic_mass_midpoint"]["rms"],
                    "after": sw["wrf_p8w_t8w"]["rms"],
                    "strictly_improves": sw["strictly_improves"],
                },
                "clwrf_lw_full_composition_glw_rms": {
                    "before": clw[
                        "accepted_precomposition_static_interfaces_constant_gases"
                    ]["rms"],
                    "after": clw[
                        "candidate_dynamic_interfaces_cam_gases"
                    ]["rms"],
                    "strictly_improves": clw[
                        "full_composition_strictly_improves"
                    ],
                    "gas_increment_alone_strictly_improves": clw[
                        "gas_increment_strictly_improves"
                    ],
                },
                "clwrf_sw_full_composition_swdnb_rms": {
                    "before": csw["hydrostatic_mass_midpoint"]["rms"],
                    "after": csw["wrf_p8w_t8w_clwrf"]["rms"],
                    "strictly_improves": csw[
                        "production_composition_strictly_improves"
                    ],
                },
                "o3rad_glw_rms": {
                    "before": o3["real_wrf_glw"][
                        "accepted_static_top_annual_ozone"
                    ]["rms"],
                    "after": o3["real_wrf_glw"][
                        "candidate_static_top_o3rad"
                    ]["rms"],
                    "strictly_improves": True,
                },
                "o3rad_swdnb_rms": {
                    "before": o3["real_wrf_swdnb"][
                        "accepted_midpoint_annual_ozone"
                    ]["rms"],
                    "after": o3["real_wrf_swdnb"][
                        "candidate_midpoint_o3rad"
                    ]["rms"],
                    "strictly_improves": True,
                },
            },
            "fresh_combined_tsk_parity": {
                "baseline_land_rms_k": combined[
                    "accepted_qml_plus_top_buffer"
                ]["land"]["rms"],
                "candidate_land_rms_k": combined[
                    "candidate_plus_wrf_cam_ghg"
                ]["land"]["rms"],
                "candidate_land_max_abs_k": combined[
                    "candidate_plus_wrf_cam_ghg"
                ]["land"]["max_abs"],
                "land_rms_ratio": combined["land_rms_ratio"],
                "land_rms_and_max_strictly_improve": True,
                "water_bitwise_invariant": True,
            },
            "fresh_channel_gate": {
                "passed": True,
                "activation_port_only_columns": channel[
                    "activation_mask_primary"
                ]["s_aw"]["final_port_only_columns"],
                "u_rms_ratio": channel["target_adapter_error"]["u"][
                    "rms_ratio"
                ],
                "v_rms_ratio": channel["target_adapter_error"]["v"][
                    "rms_ratio"
                ],
                "protected_channel_rms_ratios": {
                    name: record["rms_ratio"]
                    for name, record in channel[
                        "protected_channel_shapley_rms"
                    ].items()
                },
            },
            "proof_roots": {
                name: {
                    "path": str(FILES[name]),
                    "sha256": EXPECTED[name],
                    **(
                        {
                            "canonical_payload_sha256": value[
                                "canonical_payload_sha256"
                            ]
                        }
                        if "canonical_payload_sha256" in value
                        else {}
                    ),
                }
                for name, value in proofs.items()
            },
            "scope_note": (
                "CLWRF gas increment alone does not improve LW GLW over the "
                "already-dynamic interface arm; the manager-ratified admission "
                "unit is the source-faithful full production composition, "
                "which improves both the precomposition GLW and SWDNB baselines."
            ),
            "drift_gate": {
                "included_here": False,
                "reason": (
                    "24/72 h forecast drift requires a separate authorized "
                    "trajectory comparison; this proof seals source, component, "
                    "TSK, capture, and channel clauses only"
                ),
            },
        }
        output = args.output.resolve()
        atomic_json(output, proof)
        print(
            json.dumps(
                {
                    "passed": True,
                    "output": str(output),
                    "land_tsk_rms_k": proof["fresh_combined_tsk_parity"][
                        "candidate_land_rms_k"
                    ],
                    "canonical_payload_sha256": canonical(proof),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - proof audit refuses closed
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
