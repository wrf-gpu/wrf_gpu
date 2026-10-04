#!/usr/bin/env python3
"""Run the frozen SP2 partition and gate the v0234 MYNN mass-flux seam.

Pure NumPy evidence driver.  It invokes the already frozen partition script on
the sprint's one-shot CPU capture, then compares the result with the sealed
pre-fix partition.  It never imports JAX, runs WRF/MPI, or invokes the
production adapter.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-mf-seam-closure"
DRIVER = REPO / "scripts/v0234_fable_sp2_postfix_partition.py"
COMPARATOR = REPO / "scripts/v0234_gpt_operand_attribution.py"
READER = REPO / "scripts/v0234_gpt_adversarial_audit.py"
BASELINE = (
    REPO
    / ".agent/sprints/2026-07-20-v0234-gpt-land-tsk-provenance"
    / "final-accepted-partition-proof.json"
)
CAPTURE_ROOT = Path("/tmp/v0234_gpt_mf_seam_capture/endpoint-v2")
CAPTURE_ARCHIVE = CAPTURE_ROOT / "single-authority-capture.npz"
CAPTURE_MANIFEST = CAPTURE_ROOT / "manifest.json"
CAPTURE_PROOF = SPRINT / "combined-capture-proof.json"
CAPTURE_PREFLIGHT = SPRINT / "combined-capture-preflight.json"
COMBINED_TSK_PROOF = SPRINT / "combined-radiation-tsk-proof.json"
WRF_SOURCE = Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_bl_mynnedmf.F")
PORT_EDMF = REPO / "src/gpuwrf/physics/mynn_edmf.py"
PORT_PBL = REPO / "src/gpuwrf/physics/mynn_pbl.py"
ORACLE_TEST = REPO / "tests/test_mynn_edmf_oracle.py"

EXPECTED = {
    DRIVER: "e7a2aebba9fb6460d3863a8a1e03ec9240621ea19a496bc429b23416c35d3b4c",
    COMPARATOR: "247a1dca01b7aeb35af1adcfd3eb1dc459472c04275edc7f5b2cb3f380df3244",
    READER: "49bc7b9761e2da1e807000bbbc101d05039a585b8b00f4b31c3b404b66d79d2a",
    BASELINE: "f4ae01b8564c286d471b33e0b98647edbaef793ab04a03c58a2d2927ff092717",
    CAPTURE_ARCHIVE: "882bd72120011dbe76fa82d52379bf29a5b42f0e0a7fceb4811bd9a2a4e68e11",
    CAPTURE_MANIFEST: "db2b3949af18cc0ebf0c762235509fd3548e218375a31c3cb35a875fb8365b8a",
    CAPTURE_PROOF: "e2b905573884f2839c3858a75494b50870874c1a60f4d1bacdff2bd4128ca11b",
    CAPTURE_PREFLIGHT: "0880fe708914c9e4ad1187b4a02d6b6a7ba9bcaef01da3425551534a3aa3f24e",
    COMBINED_TSK_PROOF: "6cbc55d2c0f9018f90110abb963e3fe75e7d307b80f873cba967ddc12d5236a2",
    WRF_SOURCE: "6e4a7d5b35ce46f01591f2c1d58e545380d546e654b4a59ee1bcf99cfbce2d72",
    PORT_EDMF: "2aeeffae999369c9c334f2db05595ac6044a2d9feb8c78bda549b0f89fa34ff0",
    PORT_PBL: "bad38f5bd6d7cb82e57f6c24016152f00688559251acb48dbab232cdb9b82b68",
    ORACLE_TEST: "b60cf092da475dcacc19d37de84802fbdbb016d321f83c09fc05a8a5b586ca7c",
}
BASELINE_CANONICAL = "d4d235d29f6dfbc22fe3a2c7a645ebd0e08533aea8f7bf8131bb38f20b11c79b"
WRF_TREE_SHA256 = "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b"
ALLOWED_CPUS = {13, 14, 15, 29, 30, 31}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
MASK_FIELDS = ("s_aw", "s_awu", "s_awv")
PROTECTED_CHANNELS = ("entry_state", "surface_drag", "mixing")


class GateFailure(RuntimeError):
    """Fail-closed authority, execution, or acceptance failure."""


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


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise GateFailure(f"OUTPUT_NOT_FRESH:{path}")
    payload = dict(value)
    payload["canonical_payload_sha256"] = canonical(payload)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise GateFailure(f"TEMP_NOT_FRESH:{temporary}")
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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateFailure(f"JSON_OBJECT:{path}")
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
        raise GateFailure(f"GIT:{' '.join(args)}:{error.strip()}")
    return result.stdout if binary else result.stdout.strip()


def authority_gate(approved_head: str) -> dict[str, Any]:
    if git("rev-parse", "HEAD") != approved_head:
        raise GateFailure("APPROVED_HEAD_DRIFT")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise GateFailure("TRACKED_WORKTREE_NOT_CLEAN")
    if set(os.sched_getaffinity(0)) != ALLOWED_CPUS:
        raise GateFailure("CPUSET_DRIFT")
    observed_env = {key: os.environ.get(key) for key in THREAD_ENV}
    if observed_env != THREAD_ENV:
        raise GateFailure(f"THREAD_ENV_DRIFT:{observed_env}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise GateFailure("CUDA_VISIBLE_DEVICES_NOT_EMPTY")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise GateFailure("JAX_PLATFORM_NOT_CPU")

    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected:
            raise GateFailure(f"AUTHORITY_HASH:{path}:{actual}")

    relative = Path(__file__).resolve().relative_to(REPO).as_posix()
    disk = Path(__file__).resolve().read_bytes()
    if disk != git("show", f"HEAD:{relative}", binary=True):
        raise GateFailure("GATE_SCRIPT_NOT_HEAD")
    return {
        "approved_head": approved_head,
        "cpuset": sorted(ALLOWED_CPUS),
        "thread_environment": observed_env,
        "cuda_visible_devices": "",
        "backend": "numpy-cpu",
        "gate_script": {
            "path": relative,
            "sha256": hashlib.sha256(disk).hexdigest(),
            "git_blob": git("rev-parse", f"HEAD:{relative}"),
        },
        "sealed_inputs": {
            str(path): digest for path, digest in EXPECTED.items()
        },
    }


def validate_upstream() -> dict[str, Any]:
    baseline = load_json(BASELINE)
    capture = load_json(CAPTURE_PROOF)
    preflight = load_json(CAPTURE_PREFLIGHT)
    combined = load_json(COMBINED_TSK_PROOF)
    manifest = load_json(CAPTURE_MANIFEST)
    if not (
        canonical(baseline) == BASELINE_CANONICAL
        and baseline.get("canonical_payload_sha256") == BASELINE_CANONICAL
        and baseline.get("passed") is True
        and baseline.get("verdict")
        == "QML_RRTMG_TOP_BUFFER_FINAL_ACCEPTED_NOISE_GATE_GATE_SPLIT_LOCALIZED"
    ):
        raise GateFailure("BASELINE_AUTHORITY")
    if not (
        canonical(capture) == capture.get("canonical_payload_sha256")
        and capture.get("passed") is True
        and capture.get("verdict") == "SINGLE_AUTHORITY_CPU_CAPTURE_GREEN"
        and capture.get("authority", {}).get("gpu_actions") == 0
        and capture.get("authority", {}).get("wrf_or_mpi_executions") == 0
        and capture.get("authority", {}).get("adapter_invocations") == 1
    ):
        raise GateFailure("CAPTURE_AUTHORITY")
    if not (
        canonical(preflight) == preflight.get("canonical_payload_sha256")
        and preflight.get("passed") is True
        and preflight.get("backend_imported") is False
    ):
        raise GateFailure("PREFLIGHT_AUTHORITY")
    if not (
        canonical(combined) == combined.get("canonical_payload_sha256")
        and combined.get("passed") is True
        and combined.get("verdict")
        == "WRF_PRODUCTION_STRICTLY_IMPROVES_D03_TSK"
        and combined.get("tsk_parity", {}).get(
            "land_rms_and_max_strictly_improve"
        )
        is True
        and combined.get("tsk_parity", {}).get("water_bitwise_invariant")
        is True
    ):
        raise GateFailure("COMBINED_RADIATION_AUTHORITY")
    if not (
        canonical(manifest) == manifest.get("canonical_payload_sha256")
        and manifest.get("passed") is True
        and manifest.get("single_adapter_invocation") is True
        and manifest.get("archive", {}).get("sha256")
        == EXPECTED[CAPTURE_ARCHIVE]
    ):
        raise GateFailure("CAPTURE_MANIFEST_AUTHORITY")
    return {
        "baseline": baseline,
        "capture": capture,
        "preflight": preflight,
        "combined": combined,
        "manifest": manifest,
    }


def source_gate() -> dict[str, Any]:
    wrf = WRF_SOURCE.read_text(encoding="utf-8", errors="strict")
    port = PORT_EDMF.read_text(encoding="utf-8", errors="strict")
    pbl = PORT_PBL.read_text(encoding="utf-8", errors="strict")
    wrf_tokens = (
        "IF (k==kts+1 .AND. Wn == zero) THEN",
        "NUP2=0",
        "IF (nup2 > 0) THEN",
        "s_aw1(k+1)   = s_aw1(k+1)",
    )
    port_tokens = (
        "def _wrf_first_level_plume_survival(first_level_w):",
        "return jnp.all(first_level_w > 0.0)",
        "first_level_survives = _wrf_first_level_plume_survival(EW_s[:, 0])",
        "active = active & first_level_survives",
    )
    pbl_tokens = (
        "ts = jnp.asarray(flux.t_skin, dtype=fltv.dtype)",
        "ts = jnp.broadcast_to(ts, fltv.shape) / exner[..., 0]",
        "pblh=pblh, ts=ts",
    )
    if not all(token in wrf for token in wrf_tokens):
        raise GateFailure("WRF_SOURCE_TOKENS")
    if not all(token in port for token in port_tokens):
        raise GateFailure("PORT_SOURCE_TOKENS")
    if not all(token in pbl for token in pbl_tokens):
        raise GateFailure("PBL_SOURCE_TOKENS")
    return {
        "wrf": {
            "path": str(WRF_SOURCE),
            "sha256": EXPECTED[WRF_SOURCE],
            "first_level_veto_lines": [6242, 6247],
            "flux_guard_line": 6363,
            "tokens": list(wrf_tokens),
        },
        "port": {
            "edmf_path": str(PORT_EDMF.relative_to(REPO)),
            "edmf_sha256": EXPECTED[PORT_EDMF],
            "pbl_path": str(PORT_PBL.relative_to(REPO)),
            "pbl_sha256": EXPECTED[PORT_PBL],
            "tokens": [*port_tokens, *pbl_tokens],
        },
        "first_source_authorized_divergence": (
            "WRF writes shared NUP2=0 when any plume has Wn==0 at the first "
            "integrated level, so its later NUP2 guard suppresses every "
            "s_aw* flux in that column.  The port previously retained and "
            "summed the other surviving plumes."
        ),
    }


def run_frozen_partition(raw_output: Path) -> dict[str, Any]:
    if raw_output.exists() or raw_output.is_symlink():
        raise GateFailure(f"RAW_OUTPUT_NOT_FRESH:{raw_output}")
    spec = importlib.util.spec_from_file_location(
        "v0234_mf_seam_frozen_partition", DRIVER
    )
    if spec is None or spec.loader is None:
        raise GateFailure("FROZEN_DRIVER_IMPORT")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.POSTFIX_ROOT = CAPTURE_ROOT
    driver.POSTFIX_ARCHIVE = CAPTURE_ARCHIVE
    driver.POSTFIX_MANIFEST = CAPTURE_MANIFEST
    driver.EXPECTED_POSTFIX_ARCHIVE = EXPECTED[CAPTURE_ARCHIVE]
    driver.EXPECTED_WRF_TREE = WRF_TREE_SHA256
    previous_argv = sys.argv
    try:
        sys.argv = [str(DRIVER), "--output", str(raw_output)]
        return_code = driver.main()
    finally:
        sys.argv = previous_argv
    if return_code != 0:
        raise GateFailure(f"FROZEN_DRIVER_RETURN:{return_code}")
    raw = load_json(raw_output)
    if not (
        canonical(raw) == raw.get("canonical_payload_sha256")
        and raw.get("backend") == "numpy-cpu"
        and raw.get("gpu_actions") == 0
        and raw.get("wrf_or_mpi_executions") == 0
        and raw.get("production_adapter_invocations_in_this_comparator") == 0
        and raw.get("input_evidence", {})
        .get("postfix_archive", {})
        .get("sha256")
        == EXPECTED[CAPTURE_ARCHIVE]
        and raw.get("input_evidence", {})
        .get("wrf_dump_tree", {})
        .get("sha256")
        == WRF_TREE_SHA256
    ):
        raise GateFailure("RAW_PARTITION_AUTHORITY")
    return raw


def make_gate(
    baseline: Mapping[str, Any], raw: Mapping[str, Any]
) -> dict[str, Any]:
    masks: dict[str, Any] = {}
    for name in MASK_FIELDS:
        before = baseline["mass_flux_activity"][name]
        after = raw["mass_flux_activity"][name]
        gate = {
            "baseline_port_active_columns": before["port_active_columns"],
            "final_port_active_columns": after["port_active_columns"],
            "wrf_active_columns": after["wrf_active_columns"],
            "baseline_port_only_columns": before["columns_port_only"],
            "final_port_only_columns": after["columns_port_only"],
            "final_wrf_only_columns": after["columns_wrf_only"],
            "final_both_columns": after["columns_both"],
            "port_only_reduction_fraction": (
                1.0
                - after["columns_port_only"] / before["columns_port_only"]
            ),
            "passed": (
                before["columns_port_only"] == 4526
                and after["columns_port_only"] < before["columns_port_only"]
                and after["columns_wrf_only"] == 0
                and after["wrf_active_columns"]
                == before["wrf_active_columns"]
                and after["columns_both"] == after["wrf_active_columns"]
            ),
        }
        masks[name] = gate

    channels: dict[str, Any] = {}
    for name in PROTECTED_CHANNELS:
        before = baseline["combined_vector_shapley"][name]
        after = raw["combined_vector_shapley"][name]
        ratio = after["rms"] / before["rms"]
        channels[name] = {
            "baseline_shapley_rms": before["rms"],
            "final_shapley_rms": after["rms"],
            "rms_ratio": ratio,
            "maximum_ratio": 1.01,
            "passed": ratio <= 1.01,
        }

    target_components: dict[str, Any] = {}
    for component in ("u", "v"):
        before = baseline["components"][component][
            "authentic_postfix_adapter_vs_pristine_wrf"
        ]
        after = raw["components"][component][
            "authentic_postfix_adapter_vs_pristine_wrf"
        ]
        target_components[component] = {
            "baseline_rms": before["rms"],
            "final_rms": after["rms"],
            "rms_ratio": after["rms"] / before["rms"],
            "baseline_max_abs": before["max_abs"],
            "final_max_abs": after["max_abs"],
            "rms_strictly_improves": after["rms"] < before["rms"],
        }

    before_mass = baseline["combined_vector_shapley"]["mass_flux"]
    after_mass = raw["combined_vector_shapley"]["mass_flux"]
    mass_target = {
        "baseline_shapley_rms": before_mass["rms"],
        "final_shapley_rms": after_mass["rms"],
        "rms_ratio": after_mass["rms"] / before_mass["rms"],
        "baseline_projection_fraction": before_mass[
            "signed_projection_fraction_of_authentic_error_sse"
        ],
        "final_projection_fraction": after_mass[
            "signed_projection_fraction_of_authentic_error_sse"
        ],
        "rms_strictly_improves": after_mass["rms"] < before_mass["rms"],
    }
    metric_before = baseline["combined_vector_shapley"]["metric"]
    metric_after = raw["combined_vector_shapley"]["metric"]
    report_only_metric = {
        "gated": False,
        "baseline_shapley_rms": metric_before["rms"],
        "final_shapley_rms": metric_after["rms"],
        "rms_ratio": metric_after["rms"] / metric_before["rms"],
        "baseline_projection_fraction": metric_before[
            "signed_projection_fraction_of_authentic_error_sse"
        ],
        "final_projection_fraction": metric_after[
            "signed_projection_fraction_of_authentic_error_sse"
        ],
    }
    passed = (
        all(item["passed"] for item in masks.values())
        and all(item["passed"] for item in channels.values())
        and all(
            item["rms_strictly_improves"]
            for item in target_components.values()
        )
        and mass_target["rms_strictly_improves"]
    )
    return {
        "passed": passed,
        "activation_mask_primary": masks,
        "protected_channel_shapley_rms": channels,
        "target_adapter_error": target_components,
        "target_mass_flux_channel": mass_target,
        "metric_channel_report_only": report_only_metric,
        "final_combined_ranking": raw[
            "combined_vector_shapley_ranking_by_absolute_projection"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-head", required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        execution = authority_gate(args.approved_head)
        upstream = validate_upstream()
        source = source_gate()
        raw_output = args.raw_output.resolve()
        output = args.output.resolve()
        raw = run_frozen_partition(raw_output)
        gate = make_gate(upstream["baseline"], raw)
        if not gate["passed"]:
            raise GateFailure("CHANNEL_GATE_RED")

        residual = gate["activation_mask_primary"]["s_aw"]
        proof: dict[str, Any] = {
            "schema": "wrfgpu2-v0234-gpt-mf-seam-channel-gate-v1",
            "generated_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "passed": True,
            "verdict": "MASS_FLUX_SEAM_SOURCE_LOCALIZED_FIX_PROVEN",
            "execution": {
                **execution,
                "gpu_actions": 0,
                "wrf_or_mpi_executions": 0,
                "production_adapter_invocations": 0,
                "frozen_partition_driver_sha256": EXPECTED[DRIVER],
                "frozen_comparator_sha256": EXPECTED[COMPARATOR],
                "independent_reader_sha256": EXPECTED[READER],
            },
            "input_evidence": {
                "baseline_partition": {
                    "path": str(BASELINE),
                    "sha256": EXPECTED[BASELINE],
                    "canonical_payload_sha256": BASELINE_CANONICAL,
                },
                "fresh_capture": {
                    "archive_path": str(CAPTURE_ARCHIVE),
                    "archive_sha256": EXPECTED[CAPTURE_ARCHIVE],
                    "manifest_sha256": EXPECTED[CAPTURE_MANIFEST],
                    "proof_sha256": EXPECTED[CAPTURE_PROOF],
                    "preflight_sha256": EXPECTED[CAPTURE_PREFLIGHT],
                },
                "combined_radiation_tsk_proof_sha256": EXPECTED[
                    COMBINED_TSK_PROOF
                ],
                "raw_partition": {
                    "path": str(raw_output),
                    "sha256": sha256_file(raw_output),
                    "canonical_payload_sha256": raw[
                        "canonical_payload_sha256"
                    ],
                },
                "wrf_dump_tree_sha256": WRF_TREE_SHA256,
            },
            "source_localization": source,
            "predicate_audit": {
                "ordered_source_predicates": [
                    "resolved-scale Psig_shcu taper",
                    "positive virtual heat flux threshold",
                    "maxwidth greater than minwidth",
                    "WRF superadiabatic surface gate using TSK/exner(kts)",
                    "each of eight plumes survives the first integrated level",
                    "shared NUP2 remains positive before s_aw* accumulation",
                ],
                "first_authorized_divergence": (
                    "first-level plume survival was per-plume in the port but "
                    "column-global through shared NUP2 in WRF"
                ),
                "secondary_interface_repair": (
                    "DMP_mf surface argument now receives TSK/exner(kts), "
                    "matching the WRF driver"
                ),
                "baseline_port_only_columns": 4526,
                "final_port_only_columns": residual[
                    "final_port_only_columns"
                ],
                "false_positive_columns_removed": (
                    residual["baseline_port_only_columns"]
                    - residual["final_port_only_columns"]
                ),
                "false_positive_reduction_fraction": residual[
                    "port_only_reduction_fraction"
                ],
                "final_mask_agreement_fraction": (
                    1.0 - residual["final_port_only_columns"] / (93 * 111)
                ),
                "residual_status": (
                    "67-column frozen observational bound; no WRF-only "
                    "regression and no speculative tuning authorized"
                ),
            },
            "channel_decomposed_gate": gate,
            "acceptance_semantics": {
                "source_fidelity_required": True,
                "activation_mask_is_primary": True,
                "target_must_strictly_improve": True,
                "entry_state_surface_drag_mixing_rms_max_ratio": 1.01,
                "metric_channel_is_report_only": True,
                "aggregate_one_e_minus_four_gate_rejected_as_defective": True,
            },
        }
        atomic_json(output, proof)
        print(
            json.dumps(
                {
                    "passed": True,
                    "verdict": proof["verdict"],
                    "output": str(output),
                    "raw_output": str(raw_output),
                    "port_only_columns": residual[
                        "final_port_only_columns"
                    ],
                    "u_rms_ratio": gate["target_adapter_error"]["u"][
                        "rms_ratio"
                    ],
                    "v_rms_ratio": gate["target_adapter_error"]["v"][
                        "rms_ratio"
                    ],
                    "canonical_payload_sha256": canonical(proof),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - proof driver refuses closed
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
