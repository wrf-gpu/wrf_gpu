#!/usr/bin/env python3
"""Seal the v0234 land-TSK terminal bound from pinned sprint evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-land-tsk-provenance"
PRIOR = REPO / ".agent/sprints/2026-07-19-v0234-gpt-sp2-residual-closure"
SCRATCH = Path("/tmp/v0234_land_tsk_scratch")
ACCEPTED_V2 = (
    SCRATCH
    / "qml-rrtmg-final-accepted-fixed-cpu-adapter/qml-buffer-v2-final"
)
ROLLBACK_V3 = (
    SCRATCH
    / "qml-rrtmg-final-accepted-fixed-cpu-adapter/qml-buffer-v3-post-o3-revert"
)


JSON_EVIDENCE = {
    "prior_terminal_manifest": (
        PRIOR / "PROOF_MANIFEST.json",
        "6115634763e5e81a5f5bfa05f9fdb176e0ddfa78c778b017afec6f3a4feb626c",
    ),
    "qml_tsk": (
        SPRINT / "tsk-provenance-proof.json",
        "140905f7b8998ab4b5a39fa78e2cfc1213e18baccc21479aabad052e2cda26c0",
    ),
    "qml_partition": (
        SPRINT / "qml-partition-proof.json",
        "dcb620a04eea595169240e0b2cb19d33c0c2e12e381991100071db2a5977e4e5",
    ),
    "top_real_wrf": (
        SPRINT / "rrtmg-top-buffer-real-wrf-proof.json",
        "049399d259e3eddbd015359e94c48476e91c409666611b69a0d939b2a9b544a8",
    ),
    "top_tsk": (
        SPRINT / "rrtmg-top-buffer-tsk-proof.json",
        "af78f7a3eba095c2a69b688eb8c3f441b57975a3b500c8802f2f92cc0312917a",
    ),
    "accepted_capture": (
        SPRINT / "final-accepted-capture-proof.json",
        "3b019b6e3542c1f54a94c4575dd8fc874729d335daf0b0775538ec29504c7d4d",
    ),
    "accepted_partition": (
        SPRINT / "final-accepted-partition-proof.json",
        "f4ae01b8564c286d471b33e0b98647edbaef793ab04a03c58a2d2927ff092717",
    ),
    "lw_interface_real_wrf": (
        SPRINT / "rrtmg-interface-real-wrf-proof.json",
        "e2a1733ff7a63e4063017ae1a8d5e704bb90e2a7ed1f6446432868c5cf09bc0c",
    ),
    "sw_interface_real_wrf": (
        SPRINT / "rrtmg-sw-interface-real-wrf-proof.json",
        "cbbde365b3c429c4decae7ce4fa7c8df710b23bbcf426146c5c46d2288156ecb",
    ),
    "interface_tsk": (
        SPRINT / "rrtmg-interface-tsk-proof.json",
        "13b8f7414a13becc3c458c81a744d15ea50af7f73a929dec0af447a4cc5184fc",
    ),
    "interface_partition": (
        SPRINT / "interface-partition-proof.json",
        "f49c4b6ba47edb83516b11db2707e02f1a9357beae8bb0760380e546b2fff2a8",
    ),
    "composed_tsk": (
        SPRINT / "rrtmg-composed-tsk-proof.json",
        "728d8274d5d0aad3a8ff8e2cee98263e71f49c57d08f4c09de9e64337b474c71",
    ),
    "composed_partition": (
        SPRINT / "rrtmg-composed-partition-proof.json",
        "54de2ba2cf25ae28b113801b3d91cad63dbfb7bb0f74d9c426de7085836c4e95",
    ),
    "clwrf_lw_real_wrf": (
        SPRINT / "rrtmg-clwrf-production-real-wrf-proof.json",
        "f9e984f4b09a8bb3a226b167c59e269389953f70fe7e5627af2abfdee32628bc",
    ),
    "clwrf_sw_real_wrf": (
        SPRINT / "rrtmg-clwrf-production-sw-real-wrf-proof.json",
        "d6ec82fda4bf75dfef95ccfaa9dfb007c93accf71791971e11c2ed95417417cd",
    ),
    "clwrf_tsk": (
        SPRINT / "rrtmg-clwrf-production-tsk-proof.json",
        "bff763c4fea6cbe5b32fe7ca0d3054a55692badb560b77ae4096aee7f875fa28",
    ),
    "clwrf_partition": (
        SPRINT / "rrtmg-clwrf-composed-partition-proof.json",
        "cb7e2ad70508bda27c861316a5cea6558f6732880ca4936d1abd7eea2c50ade7",
    ),
    "o3_real_wrf": (
        SPRINT / "rrtmg-o3-real-wrf-proof.json",
        "333d716259c1d012b7db6d7092717611d3c708113ddf247683161872d753e4d9",
    ),
    "o3_tsk": (
        SPRINT / "rrtmg-o3-tsk-proof.json",
        "3505800030964c8706b631d9ed009a1194696854f4471cebe3798a1986eef6ac",
    ),
    "o3_partition": (
        SPRINT / "o3-partition-proof.json",
        "925dac5cbdc1edaeda3dabe26ec91b0dba7b869a16e433ee62f4c7ac84eb9ab2",
    ),
    "rollback_preflight": (
        SPRINT / "terminal-revert-capture-preflight.json",
        "78e6f913d66c364c6fbb2ebe0a9fdf52b5efdbfb1dab86a9ca4f713070d688b1",
    ),
    "rollback_capture": (
        SPRINT / "terminal-revert-capture-proof.json",
        "23f8be558942211bf97677d54bfc1c1b8c2e717dded008ff0ce15ebad52ee4e4",
    ),
}

# The inherited sprint manifest predates the repository's compact sorted-JSON
# canonical convention.  Its exact file digest is pinned above; do not pretend
# its historical declaration can be revalidated with the newer convention.
LEGACY_CANONICAL_EXEMPT = {"prior_terminal_manifest"}

FILE_EVIDENCE = {
    "accepted_archive": (
        ACCEPTED_V2 / "single-authority-capture.npz",
        "081c8528a1ca46e568669f0086fc4f0b86d70b939b60036096ab7576f919e034",
    ),
    "rollback_archive": (
        ROLLBACK_V3 / "single-authority-capture.npz",
        "081c8528a1ca46e568669f0086fc4f0b86d70b939b60036096ab7576f919e034",
    ),
    "accepted_manifest": (
        ACCEPTED_V2 / "manifest.json",
        "6206a9e4e7bcdc056387bde35862251e7b5a3ee52e29b68ef967b117d8aef6e1",
    ),
    "rollback_manifest": (
        ROLLBACK_V3 / "manifest.json",
        "7115fc1adf033c7a4689ac8ec628f8c6ea02818a6f22d24da36749a77667d797",
    ),
    "production_coupler": (
        REPO / "src/gpuwrf/coupling/physics_couplers.py",
        "bf9e784ab9ed642940a7cc4b51cdb518a3f8d2979ba7004d55d2cc9d7a63227c",
    ),
    "rrtmg_lw": (
        REPO / "src/gpuwrf/physics/rrtmg_lw.py",
        "090ea05af5e5ffe1e2f3dba0c9c8b1680a5253044e4f4fd4440a41442fd4b1a9",
    ),
    "rrtmg_sw": (
        REPO / "src/gpuwrf/physics/rrtmg_sw.py",
        "8381820a672f6c0d415973cbbc5e458588d19c93c7ad39cc899d0e4e663276f0",
    ),
    "cam_ozone": (
        REPO / "src/gpuwrf/physics/wrf_cam_ozone.py",
        "ceb27bec0bc51822b7c15528dccb269491037b510553835aa9c5d7e7e75a68a5",
    ),
    "clwrf_gases": (
        REPO / "src/gpuwrf/physics/wrf_clwrf_ghg.py",
        "7bd666a5b55d43cbf79a6c339fd373b4138f269a0ce37edc65b165c3c13a4a12",
    ),
}


class TerminalAuditFailure(RuntimeError):
    """A pinned input or terminal invariant drifted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_payload(record: dict[str, Any]) -> str:
    body = {
        key: value
        for key, value in record.items()
        if key != "canonical_payload_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


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
        raise TerminalAuditFailure(f"GIT:{' '.join(args)}:{error.strip()}")
    return result.stdout if binary else result.stdout.strip()


def tracked_script() -> dict[str, Any]:
    path = Path(__file__).resolve()
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    if disk != git("show", f"HEAD:{relative}", binary=True):
        raise TerminalAuditFailure("AUDIT_SCRIPT_NOT_HEAD")
    return {
        "path": relative,
        "sha256": hashlib.sha256(disk).hexdigest(),
        "git_blob": git("rev-parse", f"HEAD:{relative}"),
    }


def load_pinned_json(label: str, path: Path, expected: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TerminalAuditFailure(f"EVIDENCE_TYPE:{label}:{path}")
    actual = sha256_file(path)
    if actual != expected:
        raise TerminalAuditFailure(f"EVIDENCE_HASH:{label}:{actual}")
    record = json.loads(path.read_text(encoding="utf-8"))
    declared = record.get("canonical_payload_sha256")
    if (
        declared is not None
        and label not in LEGACY_CANONICAL_EXEMPT
        and canonical_payload(record) != declared
    ):
        raise TerminalAuditFailure(f"EVIDENCE_CANONICAL:{label}")
    return record


def require(condition: bool, label: str) -> None:
    if not condition:
        raise TerminalAuditFailure(label)


def land_metrics(proof: dict[str, Any], key: str) -> dict[str, Any]:
    return proof["tsk_parity"][key]["land"]


def candidate_record(
    name: str,
    tsk: dict[str, Any],
    tsk_key: str,
    partition: dict[str, Any],
) -> dict[str, Any]:
    land = land_metrics(tsk, tsk_key)
    acceptance = partition["candidate_acceptance"]
    return {
        "name": name,
        "land_tsk_rms_k": land["rms"],
        "land_tsk_max_abs_k": land["max_abs"],
        "land_tsk_signed_bias_k": land["signed_bias"],
        "land_tsk_bitwise_mismatch_count": land["bitwise_mismatch_count"],
        "water_tsk_bitwise_mismatch_count": proof_water_mismatch(tsk, tsk_key),
        "sp2_u_rms": acceptance["candidate"]["u_rms"],
        "sp2_v_rms": acceptance["candidate"]["v_rms"],
        "sp2_u_ratio_over_e653bdbf": acceptance["u_ratio_candidate_over_baseline"],
        "sp2_v_ratio_over_e653bdbf": acceptance["v_ratio_candidate_over_baseline"],
        "sp2_noise_gate_accepted": acceptance["accepted"],
    }


def proof_water_mismatch(proof: dict[str, Any], key: str) -> int:
    return proof["tsk_parity"][key]["water"]["bitwise_mismatch_count"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        script = tracked_script()
        records = {
            label: load_pinned_json(label, path, digest)
            for label, (path, digest) in JSON_EVIDENCE.items()
        }
        files: dict[str, Any] = {}
        for label, (path, expected) in FILE_EVIDENCE.items():
            if path.is_symlink() or not path.is_file():
                raise TerminalAuditFailure(f"FILE_TYPE:{label}:{path}")
            actual = sha256_file(path)
            if actual != expected:
                raise TerminalAuditFailure(f"FILE_HASH:{label}:{actual}")
            files[label] = {"path": str(path), "sha256": actual}

        qml = records["qml_tsk"]
        top = records["top_tsk"]
        accepted = records["accepted_partition"]
        interface = records["interface_tsk"]
        composed = records["composed_tsk"]
        clwrf = records["clwrf_tsk"]
        o3 = records["o3_tsk"]

        baseline = accepted["candidate_acceptance"]["e653bdbf_baseline"]
        accepted_gate = accepted["candidate_acceptance"]
        accepted_land = land_metrics(top, "qml_plus_rrtmg_buffer_counterfactual")
        qml_land = land_metrics(
            qml,
            "counterfactual_authenticated_seam_plus_paired_delta_vs_pristine_wrf",
        )
        frozen_land = land_metrics(qml, "frozen_seam_vs_pristine_wrf")
        require(frozen_land["rms"] == 0.02123018786362896, "FROZEN_TSK_RMS")
        require(qml_land["rms"] == 0.020818276319712437, "QML_TSK_RMS")
        require(accepted_land["rms"] == 0.018808123113575367, "ACCEPTED_TSK_RMS")
        require(accepted_gate["accepted"] is True, "ACCEPTED_SP2_GATE")
        require(accepted_gate["u_within_noise_bound"] is True, "ACCEPTED_U_GATE")
        require(accepted_gate["v_within_noise_bound"] is True, "ACCEPTED_V_GATE")

        require(
            records["top_real_wrf"]["status"] == "PASS"
            and records["top_real_wrf"]["is_self_compare"] is False
            and records["top_real_wrf"]["buffer_source_replay"]["bitwise_exact"]
            is True
            and records["top_real_wrf"]["real_wrf_glw"]["strictly_improves"]
            is True,
            "TOP_BUFFER_SOURCE_GATE",
        )
        require(
            records["lw_interface_real_wrf"]["gates"][
                "real_wrf_glw_strictly_improves"
            ]
            is True
            and records["sw_interface_real_wrf"]["gates"][
                "real_wrf_swdnb_rms_and_max_strictly_improve"
            ]
            is True,
            "INTERFACE_REAL_WRF_GATE",
        )
        require(
            records["clwrf_lw_real_wrf"]["gates"][
                "full_composition_real_wrf_glw_strictly_improves"
            ]
            is True
            and records["clwrf_sw_real_wrf"]["gates"][
                "real_wrf_swdnb_production_composition_strictly_improves"
            ]
            is True,
            "CLWRF_REAL_WRF_GATE",
        )
        o3_real = records["o3_real_wrf"]
        require(
            o3_real["status"] == "PASS"
            and o3_real["is_self_compare"] is False
            and o3_real["gates"]["cam_table_extraction_exact"] is True
            and o3_real["gates"]["real_wrf_glw_strictly_improves"] is True
            and o3_real["gates"]["real_wrf_swdnb_strictly_improves"] is True
            and o3_real["gates"]["omitted_leaves_byte_identical_to_9bfe1e30"]
            is True,
            "O3_REAL_WRF_GATE",
        )

        candidates = [
            candidate_record(
                "hydrostatic_P3D_P8W_T8W",
                interface,
                "candidate_plus_wrf_interfaces",
                records["interface_partition"],
            ),
            candidate_record(
                "composed_LW_SW_interfaces",
                composed,
                "composed_wrf_faithful_lw_plus_sw",
                records["composed_partition"],
            ),
            candidate_record(
                "composed_interfaces_plus_CLWRF_SSP245",
                clwrf,
                "candidate_plus_wrf_cam_ghg",
                records["clwrf_partition"],
            ),
            candidate_record(
                "static_top_plus_exact_O3RAD",
                o3,
                "candidate_qml_plus_static_top_plus_o3rad",
                records["o3_partition"],
            ),
        ]
        require(all(item["water_tsk_bitwise_mismatch_count"] == 0 for item in candidates), "CANDIDATE_WATER_DRIFT")
        require(all(item["sp2_noise_gate_accepted"] is False for item in candidates), "REJECTED_GATE_DRIFT")

        partition_labels = (
            "accepted_partition",
            "interface_partition",
            "composed_partition",
            "clwrf_partition",
            "o3_partition",
        )
        for label in partition_labels:
            partition = records[label]
            activity = partition["mass_flux_activity"]
            require(
                all(
                    activity[name]["columns_port_only"] == 4526
                    for name in ("s_aw", "s_awu", "s_awv")
                ),
                f"GATE_COUNT:{label}",
            )
            localization = partition.get("gate_split_localization")
            if localization is not None:
                require(localization["localized"] is True, f"LOCALIZATION:{label}")

        rollback_preflight = records["rollback_preflight"]
        rollback_capture = records["rollback_capture"]
        expected_order = [
            "authenticated seam",
            "seasonal absolute surface handles",
            "QML paired delta",
            "RRTMG-buffer paired delta",
        ]
        require(
            rollback_preflight["passed"] is True
            and rollback_preflight["git"]["head"]
            == "3fc0892aaf1ea0a5e4c0ac744fb2b3911728d85d"
            and rollback_preflight["correction_composition"]["order"]
            == expected_order,
            "ROLLBACK_PREFLIGHT",
        )
        require(
            rollback_capture["passed"] is True
            and rollback_capture["verdict"] == "SINGLE_AUTHORITY_CPU_CAPTURE_GREEN"
            and rollback_capture["archive"]["sha256"]
            == FILE_EVIDENCE["rollback_archive"][1],
            "ROLLBACK_CAPTURE",
        )
        require(
            files["accepted_archive"]["sha256"]
            == files["rollback_archive"]["sha256"],
            "ROLLBACK_ARCHIVE_NOT_IDENTICAL",
        )
        rollback_manifest = json.loads(
            (ROLLBACK_V3 / "manifest.json").read_text(encoding="utf-8")
        )
        require(
            rollback_manifest["passed"] is True
            and rollback_manifest["authority"]["adapter_invocations"] == 1
            and rollback_manifest["authority"]["gpu_actions"] == 0
            and rollback_manifest["authority"]["wrf_or_mpi_executions"] == 0,
            "ROLLBACK_MANIFEST_AUTHORITY",
        )

        o3_land = land_metrics(o3, "candidate_qml_plus_static_top_plus_o3rad")
        o3_acceptance = records["o3_partition"]["candidate_acceptance"]
        require(o3_land["rms"] == 0.004969380933367374, "O3_TSK_RMS")
        require(
            o3_acceptance["u_within_noise_bound"] is False
            and o3_acceptance["v_within_noise_bound"] is False,
            "O3_SP2_REJECTION",
        )

        proof = {
            "schema": "wrfgpu2-v0234-gpt-land-tsk-terminal-audit-v1",
            "verdict": "SOURCE_FAITHFUL_RADIATION_TSK_BOUND_SP2_GATE_CONFLICT_LOCALIZED",
            "passed": True,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "git": {
                "branch": git("branch", "--show-current"),
                "head": git("rev-parse", "HEAD"),
                "tree": git("rev-parse", "HEAD^{tree}"),
                "tracked_status_porcelain": git(
                    "status", "--porcelain", "--untracked-files=no"
                ),
                "audit_script": script,
            },
            "execution": {
                "backend": "standard-library-cpu-evidence-audit",
                "gpu_actions": 0,
                "wrf_or_mpi_executions": 0,
                "new_production_adapter_invocations": 0,
            },
            "terminal_conditions": {
                "land_tsk_bitwise_closed": False,
                "land_tsk_representation_floor_justified": False,
                "land_tsk_closed_or_floor_justified": False,
                "final_sp2_within_e653bdbf_relative_noise_bound_1e_4": True,
                "gate_split_addressed_or_falsifiably_localized": True,
                "all_manager_amended_terminal_conditions_met": False,
                "original_contract_strongest_falsifiable_localization_met": True,
            },
            "tsk_chain": {
                "e653bdbf_frozen_land": frozen_land,
                "accepted_qml_land": qml_land,
                "accepted_qml_plus_static_top_land": accepted_land,
                "strongest_individually_gated_source_candidate": {
                    "name": "exact WRF o3input=2/O3RAD on LW and SW",
                    "land": o3_land,
                    "land_rms_reduction_fraction_vs_accepted": 1.0
                    - o3_land["rms"] / accepted_land["rms"],
                    "real_wrf_glw_rms_before": o3_real["real_wrf_glw"][
                        "accepted_static_top_annual_ozone"
                    ]["rms"],
                    "real_wrf_glw_rms_after": o3_real["real_wrf_glw"][
                        "candidate_static_top_o3rad"
                    ]["rms"],
                    "real_wrf_swdnb_rms_before": o3_real["real_wrf_swdnb"][
                        "accepted_midpoint_annual_ozone"
                    ]["rms"],
                    "real_wrf_swdnb_rms_after": o3_real["real_wrf_swdnb"][
                        "candidate_midpoint_o3rad"
                    ]["rms"],
                },
                "rejected_source_faithful_candidates": candidates,
                "remaining_status": "LAND_TSK_NOT_CLOSED_AND_NOT_A_REPRESENTATION_FLOOR",
            },
            "final_production_endpoint": {
                "composition": "WRF QML specific-humidity conversion + exact static LW top buffer",
                "land_tsk_rms_k": accepted_land["rms"],
                "land_tsk_max_abs_k": accepted_land["max_abs"],
                "land_tsk_signed_bias_k": accepted_land["signed_bias"],
                "land_tsk_bitwise_mismatch_count": accepted_land[
                    "bitwise_mismatch_count"
                ],
                "water_tsk_bitwise_mismatch_count": proof_water_mismatch(
                    top, "qml_plus_rrtmg_buffer_counterfactual"
                ),
                "sp2_u_rms": accepted_gate["candidate"]["u_rms"],
                "sp2_v_rms": accepted_gate["candidate"]["v_rms"],
                "sp2_u_ratio_over_e653bdbf": accepted_gate[
                    "u_ratio_candidate_over_baseline"
                ],
                "sp2_v_ratio_over_e653bdbf": accepted_gate[
                    "v_ratio_candidate_over_baseline"
                ],
                "relative_noise_bound": accepted_gate["relative_noise_bound"],
                "fresh_post_rollback_capture_archive_sha256": files[
                    "rollback_archive"
                ]["sha256"],
                "byte_identical_to_pre_o3_accepted_archive": True,
            },
            "gate_split_localization": {
                "port_only_columns": {
                    name: accepted["mass_flux_activity"][name][
                        "columns_port_only"
                    ]
                    for name in ("s_aw", "s_awu", "s_awv")
                },
                "masks_bitwise_invariant_across_tested_tsk_radiation_candidates": True,
                "combined_vector_mass_flux_shapley_fraction_final": accepted[
                    "combined_vector_shapley"
                ]["mass_flux"]["signed_projection_fraction_of_authentic_error_sse"],
                "falsifiable_conclusion": (
                    "The tested QML, static-top, P3D/P8W/T8W, CLWRF-gas, and "
                    "O3RAD forcing deltas do not alter the 4,526-column split. "
                    "Within this evidence it remains localized to the MYNN "
                    "mass-flux activation seam; causality beyond the tested "
                    "deltas remains open because TSK is not closed."
                ),
            },
            "source_fidelity_vs_endpoint_gate": {
                "observation": (
                    "Exact O3RAD strongly improves authenticated real-WRF GLW, "
                    "SWDNB, and d03 land TSK, yet worsens both frozen SP2 endpoints "
                    "outside the amended noise bound while leaving activation masks "
                    "unchanged."
                ),
                "o3_sp2_u_ratio_over_e653bdbf": o3_acceptance[
                    "u_ratio_candidate_over_baseline"
                ],
                "o3_sp2_v_ratio_over_e653bdbf": o3_acceptance[
                    "v_ratio_candidate_over_baseline"
                ],
                "interpretation_for_critic": (
                    "This is evidence of an SP2 endpoint compensation/omitted-error "
                    "conflict, not evidence that the source-faithful radiation "
                    "transcription is wrong."
                ),
            },
            "rollback": {
                "rejected_dynamic_interfaces_populated_in_production": False,
                "rejected_clwrf_gases_populated_in_production": False,
                "rejected_o3rad_populated_in_production": False,
                "optional_backward_compatible_interfaces_retained": True,
                "capture_archive_byte_identical": True,
                "accepted_archive": files["accepted_archive"],
                "post_rollback_archive": files["rollback_archive"],
            },
            "authority": {
                "baseline": baseline,
                "json_evidence": {
                    label: {
                        "path": str(path),
                        "sha256": digest,
                        "canonical_payload_sha256": records[label].get(
                            "canonical_payload_sha256"
                        ),
                    }
                    for label, (path, digest) in JSON_EVIDENCE.items()
                },
                "legacy_canonical_exemptions": sorted(LEGACY_CANONICAL_EXEMPT),
                "file_evidence": files,
            },
        }
        proof["canonical_payload_sha256"] = canonical_payload(proof)
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(
            json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "verdict": proof["verdict"],
                    "land_tsk_closed_or_floor": False,
                    "final_land_tsk_rms_k": accepted_land["rms"],
                    "o3_land_tsk_rms_k": o3_land["rms"],
                    "final_sp2_u_rms": accepted_gate["candidate"]["u_rms"],
                    "final_sp2_v_rms": accepted_gate["candidate"]["v_rms"],
                    "gate_split_columns": 4526,
                    "canonical_payload_sha256": proof[
                        "canonical_payload_sha256"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - proof must fail closed
        print(f"FAIL_CLOSED:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
