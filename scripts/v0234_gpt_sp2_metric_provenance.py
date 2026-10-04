#!/usr/bin/env python3
"""Prove the first v0234 SP2 divergence is CPU-capture metric authority.

Pure NumPy/netCDF analysis over the authenticated post-QKE seam, the sealed
single-adapter CPU archive, and the pristine WRF dump.  It reproduces both the
incorrect analytic-flat rho that entered the old capture and the corrected
WRF-hybrid rho; no adapter, GPU, WRF, or MPI execution occurs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from netCDF4 import Dataset
import numpy as np


REPO = Path(__file__).resolve().parent.parent
STAGED = Path("/tmp/v0234_gpt_sp2_residual_evidence")
SEAM = STAGED / "seam-capture"
POSTFIX = STAGED / "postfix"
WRF_ROOT = STAGED / "wrf-dumps"
WRFINPUT = STAGED / "inputs/wrfinput_d03"
AUDIT_READER = REPO / "scripts/v0234_gpt_adversarial_audit.py"
COUPLER = REPO / "src/gpuwrf/coupling/physics_couplers.py"
GEN2 = REPO / "src/gpuwrf/io/gen2_accessor.py"
METRICS = REPO / "src/gpuwrf/dynamics/metrics.py"
CAPTURE_HARNESS = REPO / "scripts/v0234_gpt_single_authority_capture.py"
PROVENANCE_SCRIPT = Path(__file__).resolve()

EXPECTED = {
    "seam_manifest": "ae584d25d0e8fc56afedc4a1745f9109d52fecf76635007324180f40bb709873",
    "seam_canonical": "f6d14eee33d1f43364238c9a42849cb0993aaa71fa9b660c397d0c3ee7096d6a",
    "postfix_archive": "0f778d51658148fa2d482493fd8fae8a15c5862c2052e6d5942e839e16a850e0",
    "postfix_manifest": "4722c6faf5df0bf09675ff3d9d51d61f6d81b2793c7c4ee9ea496f474fe9afe9",
    "postfix_manifest_canonical": "7b168f4b1aa372e4232ca00df8f77a7f194f4f94d8366237e28808bc0ba1a388",
    "wrf_tree": "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b",
    "wrfinput": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
}


class ProvenanceFailure(RuntimeError):
    """Fail-closed evidence or arithmetic error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_without_self(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items()
               if key != "canonical_payload_sha256"}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ProvenanceFailure(f"OUTPUT_NOT_FRESH:{path}")
    payload = dict(value)
    payload["canonical_payload_sha256"] = canonical_without_self(payload)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(
            payload, sort_keys=True, indent=2, allow_nan=False,
        ) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    if lhs.shape != rhs.shape or not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise ProvenanceFailure(f"METRIC_INPUT:{lhs.shape}:{rhs.shape}")
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    rms = float(np.sqrt(np.mean(delta * delta)))
    reference_rms = float(np.sqrt(np.mean(rhs.astype(np.float64) ** 2)))
    return {
        "rms": rms,
        "max_abs": float(np.max(np.abs(delta))),
        "exact_fraction": float(np.mean(lhs == rhs)),
        "reference_rms": reference_rms,
        "relative_rms": rms / reference_rms if reference_rms else None,
    }


def load_manifest(path: Path, file_sha: str, canonical_sha: str) -> dict[str, Any]:
    if path.is_symlink() or sha256_file(path) != file_sha:
        raise ProvenanceFailure(f"MANIFEST_FILE:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not (
        canonical_without_self(value) == canonical_sha
        and value.get("canonical_payload_sha256") == canonical_sha
    ):
        raise ProvenanceFailure(f"MANIFEST_CANONICAL:{path}")
    return value


def seam_leaf(manifest: Mapping[str, Any], name: str) -> np.ndarray:
    matches = [item for item in manifest["state_leaves"] if item["slot_name"] == name]
    if len(matches) != 1 or matches[0]["kind"] != "array":
        raise ProvenanceFailure(f"SEAM_LEAF:{name}")
    item = matches[0]
    path = SEAM / item["file"]
    if sha256_file(path) != item["file_sha256"]:
        raise ProvenanceFailure(f"SEAM_FILE_HASH:{name}")
    value = np.load(path, allow_pickle=False)
    if not (
        list(value.shape) == item["shape"]
        and value.dtype.str == item["dtype"]
        and array_sha(value) == item["logical_c_bitpayload_sha256"]
    ):
        raise ProvenanceFailure(f"SEAM_ARRAY_RECORD:{name}")
    return np.asarray(value)


def rho_from_hybrid(
    ph_total: np.ndarray,
    mu_total: np.ndarray,
    qv: np.ndarray,
    c3h: np.ndarray,
    c4h: np.ndarray,
    c3f: np.ndarray,
    c4f: np.ndarray,
    p_top: np.ndarray,
) -> np.ndarray:
    ph = np.asarray(ph_total, dtype=np.float32)
    mut = np.asarray(mu_total, dtype=np.float32)
    vapor = np.asarray(qv, dtype=np.float32)
    top = np.asarray(p_top, dtype=np.float32).reshape(())
    p_up = (c3f[1:, None, None] * mut + c4f[1:, None, None] + top).astype(np.float32)
    p_down = (c3f[:-1, None, None] * mut + c4f[:-1, None, None] + top).astype(np.float32)
    p_mid = (c3h[:, None, None] * mut + c4h[:, None, None] + top).astype(np.float32)
    dph = (ph[1:] - ph[:-1]).astype(np.float32)
    alt = (dph / p_mid / np.log(p_down / p_up).astype(np.float32)).astype(np.float32)
    return ((np.float32(1.0) + vapor) / alt).astype(np.float32)


def pressure_from_hybrid(
    mu_total: np.ndarray,
    qtot: np.ndarray,
    c1h: np.ndarray,
    c2h: np.ndarray,
    dnw: np.ndarray,
    p_top: np.ndarray,
) -> np.ndarray:
    mut = np.asarray(mu_total, dtype=np.float32)
    moist = np.asarray(qtot, dtype=np.float32)
    next_face = np.broadcast_to(np.asarray(p_top, dtype=np.float32).reshape(()), mut.shape).copy()
    top_to_bottom = [next_face]
    for k in range(moist.shape[0] - 1, -1, -1):
        mass = (c1h[k] * mut + c2h[k]).astype(np.float32)
        next_face = (next_face - (np.float32(1.0) + moist[k]) * mass * dnw[k]).astype(np.float32)
        top_to_bottom.append(next_face)
    faces = np.stack(tuple(reversed(top_to_bottom)), axis=0)
    return (np.float32(0.5) * (faces[:-1] + faces[1:])).astype(np.float32)


def import_reader() -> Any:
    spec = importlib.util.spec_from_file_location("v0234_metric_provenance_reader", AUDIT_READER)
    if spec is None or spec.loader is None:
        raise ProvenanceFailure("READER_IMPORT")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IndependentWrfDump(WRF_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        seam_manifest = load_manifest(
            SEAM / "manifest.json", EXPECTED["seam_manifest"], EXPECTED["seam_canonical"]
        )
        if not (
            seam_manifest.get("status") == "CAPTURE_COMPLETE_HOLD_5_OF_28"
            and seam_manifest.get("seam", {}).get("trace_invocation_count") == 1
            and seam_manifest.get("seam", {}).get("pbl_guard_invocation_count") == 0
            and seam_manifest.get("provenance", {}).get("synthetic_reconstruction") is False
        ):
            raise ProvenanceFailure("SEAM_AUTHORITY")

        postfix_manifest_path = POSTFIX / "manifest.json"
        postfix_manifest_file_sha = EXPECTED["postfix_manifest"]
        postfix_manifest = load_manifest(
            postfix_manifest_path,
            postfix_manifest_file_sha,
            EXPECTED["postfix_manifest_canonical"],
        )
        archive_path = POSTFIX / "single-authority-capture.npz"
        if sha256_file(archive_path) != EXPECTED["postfix_archive"]:
            raise ProvenanceFailure("POSTFIX_ARCHIVE_HASH")
        if not (
            postfix_manifest.get("passed") is True
            and postfix_manifest.get("single_adapter_invocation") is True
            and postfix_manifest.get("authority", {}).get("adapter_invocations") == 1
            and postfix_manifest.get("authority", {}).get("gpu_actions") == 0
        ):
            raise ProvenanceFailure("POSTFIX_AUTHORITY")

        if sha256_file(WRFINPUT) != EXPECTED["wrfinput"]:
            raise ProvenanceFailure("WRFINPUT_HASH")
        tree = {
            path.relative_to(WRF_ROOT).as_posix(): sha256_file(path)
            for path in sorted(WRF_ROOT.rglob("*")) if path.is_file()
        }
        tree_sha = hashlib.sha256(json.dumps(
            tree, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        if len(tree) != 462 or tree_sha != EXPECTED["wrf_tree"]:
            raise ProvenanceFailure(f"WRF_TREE:{len(tree)}:{tree_sha}")

        ph_total = seam_leaf(seam_manifest, "ph_total")
        mu_total = seam_leaf(seam_manifest, "mu_total")
        qv = seam_leaf(seam_manifest, "qv")
        theta_m = seam_leaf(seam_manifest, "theta")
        hydrometeors = sum(
            seam_leaf(seam_manifest, name)
            for name in ("qv", "qc", "qr", "qi", "qs", "qg")
        )

        with Dataset(WRFINPUT) as dataset:
            wrf_metric = {
                name.lower(): np.asarray(dataset[name][0], dtype=np.float32)
                for name in (
                    "C1H", "C2H", "C3H", "C4H", "C1F", "C2F", "C3F", "C4F",
                    "DNW", "ZNW",
                )
            }
            p_top = np.asarray(dataset["P_TOP"][0], dtype=np.float32)

        znw = wrf_metric["znw"]
        flat_metric = {
            "c1h": (np.float32(0.5) * (znw[:-1] + znw[1:])).astype(np.float32),
            "c2h": np.zeros_like(wrf_metric["c2h"]),
            "c3h": (np.float32(0.5) * (znw[:-1] + znw[1:])).astype(np.float32),
            "c4h": np.zeros_like(wrf_metric["c4h"]),
            "c1f": znw,
            "c2f": np.zeros_like(wrf_metric["c2f"]),
            "c3f": znw,
            "c4f": np.zeros_like(wrf_metric["c4f"]),
        }

        reader = import_reader()
        wrf_rho = reader.columns("driver_rho")
        wrf_dz = reader.columns("driver_dz")
        with np.load(archive_path, allow_pickle=False) as archive:
            captured_rho = np.asarray(archive["surface_terms_state_rho"])
            captured_p = np.asarray(archive["surface_terms_state_p"])
            captured_theta = np.asarray(archive["surface_terms_state_theta"])
            captured_dz = np.asarray(archive["surface_terms_state_dz"])
            for name, value in (
                ("surface_terms_state_rho", captured_rho),
                ("surface_terms_state_p", captured_p),
                ("surface_terms_state_theta", captured_theta),
                ("surface_terms_state_dz", captured_dz),
            ):
                declared = postfix_manifest["archive"]["arrays"][name]
                if not (
                    list(value.shape) == declared["shape"]
                    and value.dtype.str == declared["dtype"]
                    and array_sha(value) == declared["logical_c_bitpayload_sha256"]
                ):
                    raise ProvenanceFailure(f"POSTFIX_ARRAY:{name}")

        rho_flat = rho_from_hybrid(
            ph_total, mu_total, qv,
            flat_metric["c3h"], flat_metric["c4h"],
            flat_metric["c3f"], flat_metric["c4f"], p_top,
        )
        rho_wrf_metric = rho_from_hybrid(
            ph_total, mu_total, qv,
            wrf_metric["c3h"], wrf_metric["c4h"],
            wrf_metric["c3f"], wrf_metric["c4f"], p_top,
        )
        p_flat = pressure_from_hybrid(
            mu_total, hydrometeors,
            flat_metric["c1h"], flat_metric["c2h"], wrf_metric["dnw"], p_top,
        )
        p_wrf_metric = pressure_from_hybrid(
            mu_total, hydrometeors,
            wrf_metric["c1h"], wrf_metric["c2h"], wrf_metric["dnw"], p_top,
        )
        theta_dry = theta_m / (1.0 + (461.6 / 287.0) * qv)
        dz_from_seam = (ph_total[1:] - ph_total[:-1]) / 9.81

        rho_capture_vs_wrf = metrics(captured_rho, wrf_rho)
        rho_corrected_vs_wrf = metrics(rho_wrf_metric, wrf_rho)
        rho_flat_closure = metrics(rho_flat, captured_rho)
        p_flat_closure = metrics(p_flat, captured_p)
        p_metric_delta = metrics(p_wrf_metric, captured_p)
        theta_closure = metrics(theta_dry, captured_theta)
        dz_closure = metrics(dz_from_seam, captured_dz)
        dz_vs_wrf = metrics(captured_dz, wrf_dz)

        improvement = rho_capture_vs_wrf["rms"] / rho_corrected_vs_wrf["rms"]
        if not (
            rho_flat_closure["rms"] <= 1.0e-7
            and p_flat_closure["rms"] <= 5.0e-3
            and rho_corrected_vs_wrf["relative_rms"] <= 1.0e-5
            and improvement >= 1000.0
            and theta_closure["rms"] <= 1.0e-12
            and dz_closure["rms"] <= 1.0e-10
        ):
            raise ProvenanceFailure(
                f"SOURCE_GATE:rho_flat={rho_flat_closure['rms']}:"
                f"p_flat={p_flat_closure['rms']}:rho_fix={rho_corrected_vs_wrf['relative_rms']}:"
                f"improvement={improvement}:theta={theta_closure['rms']}:dz={dz_closure['rms']}"
            )

        proof = {
            "schema": "wrfgpu2-v0234-gpt-sp2-metric-provenance-v1",
            "verdict": "CPU_CAPTURE_ANALYTIC_FLAT_METRIC_DIVERGENCE_PROVEN",
            "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "backend": "numpy-cpu",
            "gpu_actions": 0,
            "wrf_or_mpi_executions": 0,
            "production_adapter_invocations": 0,
            "authority": {
                "seam_manifest_sha256": EXPECTED["seam_manifest"],
                "seam_manifest_canonical_sha256": EXPECTED["seam_canonical"],
                "postfix_archive_sha256": EXPECTED["postfix_archive"],
                "postfix_manifest_sha256": postfix_manifest_file_sha,
                "postfix_manifest_canonical_sha256": EXPECTED["postfix_manifest_canonical"],
                "wrf_dump_tree": {"file_count": len(tree), "sha256": tree_sha},
                "wrfinput_d03_sha256": EXPECTED["wrfinput"],
            },
            "first_source_authorized_divergence": {
                "stage": "CPU materializer GridSpec construction before MYNN adapter entry",
                "old_path": (
                    "Gen2Run(INPUT_DIR).grid('d03').as_grid_spec() leaves "
                    "GridSpec.__post_init__ analytic DycoreMetrics.flat in place"
                ),
                "operational_path": (
                    "integration/d02_replay.py loads load_wrfinput_metrics(wrfinput_d03) "
                    "and replaces GridSpec.metrics"
                ),
                "effect": (
                    "MYNN phy_prep pressure/rho used c3~=eta,c4=0 and c1~=eta,c2=0 "
                    "instead of WRF C1H..C4F"
                ),
                "not_upstream_state": (
                    "The authenticated seam PH/MU/QV fed the WRF hybrid formula matches "
                    "pristine driver_rho; no PH/MU/QV correction is authorized."
                ),
            },
            "metric_payload_differences": {
                name: metrics(flat_metric[name], wrf_metric[name])
                for name in ("c1h", "c2h", "c3h", "c4h", "c1f", "c2f", "c3f", "c4f")
            },
            "rho_chain": {
                "old_capture_vs_pristine_wrf": rho_capture_vs_wrf,
                "analytic_flat_formula_vs_old_capture": rho_flat_closure,
                "wrf_metric_formula_vs_pristine_wrf": rho_corrected_vs_wrf,
                "rms_improvement_factor": improvement,
            },
            "other_entry_operands": {
                "analytic_flat_pressure_formula_vs_old_capture": p_flat_closure,
                "wrf_metric_pressure_vs_old_capture": p_metric_delta,
                "authenticated_state_moist_to_dry_theta_vs_old_capture": theta_closure,
                "authenticated_state_geopotential_to_dz_vs_old_capture": dz_closure,
                "old_capture_dz_vs_pristine_wrf": dz_vs_wrf,
            },
            "source_files": {
                path.relative_to(REPO).as_posix(): sha256_file(path)
                for path in (
                    AUDIT_READER, COUPLER, GEN2, METRICS, CAPTURE_HARNESS,
                    PROVENANCE_SCRIPT,
                )
            },
            "gates": {
                "old_capture_reproduced_by_flat_rho_rms_lte_1e-7": True,
                "old_capture_reproduced_by_flat_pressure_rms_lte_5e-3_pa": True,
                "corrected_rho_relative_rms_lte_1e-5": True,
                "rho_rms_improvement_gte_1000x": True,
                "theta_and_dz_non_metric_paths_close": True,
            },
            "passed": True,
        }
        atomic_json(output, proof)
        print(json.dumps({
            "passed": True,
            "verdict": proof["verdict"],
            "old_rho_relative_rms": rho_capture_vs_wrf["relative_rms"],
            "corrected_rho_relative_rms": rho_corrected_vs_wrf["relative_rms"],
            "rho_rms_improvement_factor": improvement,
            "output": str(output),
            "canonical_payload_sha256": canonical_without_self(proof),
        }, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
