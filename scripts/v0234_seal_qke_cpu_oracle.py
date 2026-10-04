"""Seal the authenticated current-tree JAX-CPU cold QKE array.

The source NPY was emitted by the CPU-only cold-load diagnosis and is accepted
only if its dtype/shape/content hash matches both independent cold-load proofs.
This script performs no JAX/gpuwrf import, model compile/step, GPU action, or
WRF/MPI execution.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import traceback


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-rootcause"
OUT = SPRINT / "QKE_CPU_ORACLE.json"
BLOCKER = SPRINT / "QKE_CPU_ORACLE_BLOCKER.json"
SOURCE = Path("/tmp/v0234-current-cpu-qke.npy")
LINEAGE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a"
)
ORACLE_DIR = LINEAGE / "v0234_gpt_v10_qke_cpu_oracle_680a4cab02f7198b"
ORACLE = ORACLE_DIR / "qke-current-cpu.npy"
COLD_PROOF = SPRINT / "COLD_STEP0_ANCHOR_REPAIR.json"
COLD_PROOF_FILE_SHA256 = (
    "e98c689ad9aab02a59e4eb4896e92bb4b34a7cd85371d766c332e0df6a8beeef"
)
COLD_PROOF_SHA256 = (
    "eada51b566dfe9497261226fd6e29b43278a8a9bf0f4e4d77e46325ca2d3aaad"
)
QKE_PROOF = SPRINT / "NOCTURNAL_STEP1_QKE_BACKEND.json"
QKE_PROOF_FILE_SHA256 = (
    "cb596f27c327d55b7f55b88c2982739d424d5bd7a39e72d98f288250760ff0c3"
)
QKE_PROOF_SHA256 = (
    "680a4cab02f7198b9d04c9864420a5a24bb2997a04ebb2ab86ae09f8272e7a0e"
)
ARRAY_SHA256 = (
    "07ea2a840d9a6793eb2822640eb75dd5c7d7adacf7bf11ef96020ab7f0f5ac95"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: dict) -> str:
    value = dict(payload)
    value.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _array_sha256(np, value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    stage = "startup"
    temporary: Path | None = None
    try:
        import numpy as np

        if _sha256(COLD_PROOF) != COLD_PROOF_FILE_SHA256:
            raise RuntimeError("cold-load proof file changed")
        cold = json.loads(COLD_PROOF.read_text())
        if cold.get("proof_sha256") != COLD_PROOF_SHA256:
            raise RuntimeError("cold-load proof semantics changed")
        if _sha256(QKE_PROOF) != QKE_PROOF_FILE_SHA256:
            raise RuntimeError("QKE Step1 proof file changed")
        qke_proof = json.loads(QKE_PROOF.read_text())
        if (
            qke_proof.get("proof_sha256") != QKE_PROOF_SHA256
            or qke_proof.get("verdict")
            != "QKE_BACKEND_SEED_STEP1_UV_NEGLIGIBLE__HARNESS_GATE_REPAIRABLE"
        ):
            raise RuntimeError("QKE Step1 proof semantics changed")
        if not SOURCE.is_file() or SOURCE.is_symlink():
            raise RuntimeError("authenticated temporary QKE NPY is absent or symlinked")

        stage = "source_validation"
        array = np.load(SOURCE, allow_pickle=False)
        source_array_sha = _array_sha256(np, array)
        expected_row = cold["repaired_candidate"]["cold_state"]["arrays"]["qke"]
        if (
            tuple(array.shape) != (44, 93, 111)
            or str(array.dtype) != "float64"
            or not bool(np.all(np.isfinite(array)))
            or source_array_sha != ARRAY_SHA256
            or expected_row["sha256"] != ARRAY_SHA256
            or qke_proof["authority"]["current_CPU_qke_sha256"] != ARRAY_SHA256
        ):
            raise RuntimeError("temporary QKE NPY differs from authenticated cold field")

        stage = "seal"
        if ORACLE_DIR.exists() or ORACLE_DIR.is_symlink():
            raise RuntimeError(f"oracle namespace already exists: {ORACLE_DIR}")
        ORACLE_DIR.mkdir()
        fd, name = tempfile.mkstemp(prefix=".qke-current-cpu.", dir=ORACLE_DIR)
        os.close(fd)
        temporary = Path(name)
        shutil.copyfile(SOURCE, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, ORACLE)
        temporary = None
        sealed = np.load(ORACLE, allow_pickle=False)
        if _array_sha256(np, sealed) != ARRAY_SHA256 or not np.array_equal(
            array, sealed
        ):
            raise RuntimeError("sealed QKE oracle failed byte validation")

        proof = {
            "schema": "gpuwrf.v0234.qke-cpu-cold-oracle.v1",
            "verdict": "CURRENT_TREE_JAX_CPU_QKE_ORACLE_SEALED",
            "scope": (
                "serialization of an already-authenticated CPU cold-load QKE "
                "array; no JAX/gpuwrf import, compile, step, WRF/MPI, GPU command, "
                "or GPU query"
            ),
            "oracle": {
                "path": str(ORACLE),
                "bytes": ORACLE.stat().st_size,
                "file_sha256": _sha256(ORACLE),
                "array_sha256": ARRAY_SHA256,
                "shape": list(sealed.shape),
                "dtype": str(sealed.dtype),
                "finite": bool(np.all(np.isfinite(sealed))),
                "min": float(np.min(sealed)),
                "max": float(np.max(sealed)),
            },
            "authority": {
                "cold_step0_proof": {
                    "path": str(COLD_PROOF),
                    "file_sha256": COLD_PROOF_FILE_SHA256,
                    "proof_sha256": COLD_PROOF_SHA256,
                },
                "qke_step1_discriminator": {
                    "path": str(QKE_PROOF),
                    "file_sha256": QKE_PROOF_FILE_SHA256,
                    "proof_sha256": QKE_PROOF_SHA256,
                },
                "source_npy_path": str(SOURCE),
                "source_npy_file_sha256": _sha256(SOURCE),
                "source_array_sha256": source_array_sha,
            },
            "gpu_commands": 0,
            "gpu_queries": 0,
            "jax_or_gpuwrf_imported": False,
            "model_compiles": 0,
            "model_steps": 0,
            "wrf_or_mpi_executions": 0,
        }
        proof["proof_sha256"] = _canonical(proof)
        _atomic_json(OUT, proof)
        print(json.dumps(proof, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        blocker = {
            "schema": "gpuwrf.v0234.qke-cpu-cold-oracle-blocker.v1",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "gpu_commands": 0,
            "gpu_queries": 0,
            "jax_or_gpuwrf_imported": False,
            "model_compiles": 0,
            "model_steps": 0,
            "wrf_or_mpi_executions": 0,
        }
        blocker["proof_sha256"] = _canonical(blocker)
        _atomic_json(BLOCKER, blocker)
        print(json.dumps(blocker, sort_keys=True), flush=True)
        return 4
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
