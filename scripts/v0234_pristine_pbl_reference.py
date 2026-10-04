#!/usr/bin/env python3
"""Fail-closed V0234 PBL-entry 23-array materializer and 28-array validator.

Import is backend-dark.  Static G2 admission proves the exact inventories and
authorities without importing JAX/gpuwrf.  The materialization mode is CPU-only
and may run only after the authentic one-shot capture has sealed successfully.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-pristine-pbl-entry-closure-gpt"
CONTRACT = SPRINT / "CONTRACT.md"
CONTRACT_COMMIT = "5bbfe0416979eb41dd72bf80a8539191740f8eea"
CONTRACT_SHA256 = "db445ed5ca18250c932651c4bf506799778895baf9390d692f65c9dfa21b6f35"
COORDINATION_COMMIT = "8231225573683dc56828caba2be3326508bdfd73"
AUTHORITY_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_0b18530a1dc9cac2")
OUTPUT_AUTHORITY_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084")
CAPTURE_ROOT = OUTPUT_AUTHORITY_ROOT / "capture/authentic-ac6712170cbe5084"
REFERENCE_ROOT = OUTPUT_AUTHORITY_ROOT / "cpu-reference"
REFERENCE_ARCHIVE = REFERENCE_ROOT / "pbl-sp2-reference-28.npz"
REFERENCE_MANIFEST = REFERENCE_ROOT / "pbl-sp2-reference-manifest.json"
REFERENCE_VALIDATION = REFERENCE_ROOT / "pbl-sp2-reference-validation.json"

NONCE = "ac6712170cbe508475ea8572464c7742a6906d559b53ed9930292d6822229ed8"
CARRY = AUTHORITY_ROOT / "inputs/last-healthy-d03-step-0.pkl"
CARRY_SHA256 = "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
INPUT_DIR = AUTHORITY_ROOT / "capture-inputs"
DUMP_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_mynn_sp2_d03_horizon55_gpt_fresh01/evidence-dumps-fresh-d03-runtime")
DUMP_TREE_SHA256 = "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b"
DUMP_ADMISSION = Path(
    "<USER_HOME>/src/wrf_gpu2_wt/v0234-mynn-sp2-input-provenance/"
    ".agent/sprints/2026-07-19-v0234-offline-comparator-preparation-gpt/"
    "dump-identity-admission.json"
)
DUMP_ADMISSION_SHA256 = "e326a8881710dfec38bacd40c1af3ef55cbeec5679940358e8eb03c08f093e78"
PRIOR_REFERENCE = DUMP_ADMISSION.with_name("retained-carry-reference-bundle.json")
PRIOR_REFERENCE_SHA256 = "2fb52f7cd44c09fbe925293c80a5ca9fca27dda4632bf610ef0347050dc707b2"

SOURCE_SHA256 = {
    "src/gpuwrf/contracts/state.py": "f959da39d8957ef72f165c2e4fad98a03e8235a58379e5a2743f9a6da93af9f8",
    "src/gpuwrf/coupling/physics_couplers.py": "cf569c4780cecca17e144c63386e3c456a78db8db48253aa75d0720441cc24bd",
    "src/gpuwrf/io/gen2_accessor.py": "3f552e3b3552ee16de6770170881d52330e91a3b9a3960fb174c2e9dc568c898",
    "src/gpuwrf/physics/mynn_constants.py": "0172e9424a126e2bde760066be7ba3f483240b53889b834c1f6441a7ff058aee",
    "src/gpuwrf/physics/mynn_pbl.py": "e4778816aeeba28acabdee49043eeac2693ab54b218b6c221b684c89281a1434",
    "src/gpuwrf/physics/noahmp_coupler.py": "cd857efa18731d796fd4a60047eb833fae94588e3150d2973cbd8100e34ced76",
    "src/gpuwrf/physics/tridiagonal_solver.py": "1f141e6581c41daf3a720ecbedfbe557b72d4a0c84bf61a7d3b3a7d449da193f",
    "src/gpuwrf/runtime/operational_mode.py": "68efe76b9d1f91860e9a49a6e6c9ab9e573986dd11a47677fa161badb3a79282",
}

AVAILABLE = ("surface_ust", "surface_tsk", "sp2_rublten", "sp2_rvblten", "xland")
MISSING = (
    "surface_hfx", "surface_qfx", "surface_ch",
    "mix_qke_initialized", "mix_el", "mix_sm", "mix_dfm",
    "solve_u_a", "solve_u_b", "solve_u_c", "solve_u_d", "solve_u_x",
    "solve_v_a", "solve_v_b", "solve_v_c", "solve_v_d", "solve_v_x",
    "residual_after_surface_rublten", "residual_after_surface_rvblten",
    "residual_after_mixing_rublten", "residual_after_mixing_rvblten",
    "residual_after_lower_bc_rublten", "residual_after_lower_bc_rvblten",
)
REQUIRED = (
    "surface_ust", "surface_hfx", "surface_qfx", "surface_tsk", "surface_ch",
    "mix_qke_initialized", "mix_el", "mix_sm", "mix_dfm",
    "solve_u_a", "solve_u_b", "solve_u_c", "solve_u_d", "solve_u_x",
    "solve_v_a", "solve_v_b", "solve_v_c", "solve_v_d", "solve_v_x",
    "sp2_rublten", "sp2_rvblten", "xland",
    "residual_after_surface_rublten", "residual_after_surface_rvblten",
    "residual_after_mixing_rublten", "residual_after_mixing_rvblten",
    "residual_after_lower_bc_rublten", "residual_after_lower_bc_rvblten",
)
RESIDUALS = MISSING[-6:]
RESIDUAL_STAGE = {
    **{name: "surface" for name in RESIDUALS[:2]},
    **{name: "mixing" for name in RESIDUALS[2:4]},
    **{name: "lower_bc" for name in RESIDUALS[4:]},
}
EXPECTED_SHAPES = {
    name: ((93, 111) if name.startswith("surface_") or name == "xland" else (44, 93, 111))
    for name in REQUIRED
}
EXPECTED_DTYPE = "<f8"
ALLOWED_CPUS = {13, 14, 15, 29, 30, 31}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OMP_THREAD_LIMIT": "1", "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


class ReferenceError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def canonical_without_self(value: Mapping[str, Any]) -> str:
    return canonical({key: item for key, item in value.items() if key != "canonical_payload_sha256"})


def require_file(path: Path, expected: str | None = None) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ReferenceError(f"AUTHORITY_NOT_REGULAR:{path}")
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ReferenceError(f"AUTHORITY_DRIFT:{path}:{actual}")
    info = path.stat()
    return {"path": str(path), "sha256": actual, "size": info.st_size, "mode": info.st_mode & 0o777}


def load_canonical(path: Path, expected: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    record = require_file(path, expected)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or canonical_without_self(value) != value.get("canonical_payload_sha256"):
        raise ReferenceError(f"CANONICAL_DRIFT:{path}")
    return value, record


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ReferenceError(f"OUTPUT_NOT_FRESH:{path}")
    payload = dict(value)
    payload["canonical_payload_sha256"] = canonical_without_self(payload)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def git_text(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(REPO), *args], check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ReferenceError(f"GIT_FAILED:{' '.join(args)}:{result.stderr}")
    return result.stdout.strip()


def resource_gate(*, cpu_backend: bool) -> dict[str, Any]:
    affinity = set(os.sched_getaffinity(0))
    environment = {key: os.environ.get(key) for key in THREAD_ENV}
    if affinity != ALLOWED_CPUS:
        raise ReferenceError(f"CPUSET_ESCAPE:{sorted(affinity)}")
    if environment != THREAD_ENV:
        raise ReferenceError(f"THREAD_ENV_DRIFT:{environment}")
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    ioprio = ctypes.CDLL(None, use_errno=True).syscall(252, 1, 0)
    if nice < 15 or ioprio < 0 or int(ioprio) >> 13 != 3:
        raise ReferenceError(f"PRIORITY_DRIFT:{nice}:{ioprio}")
    if Path("/tmp/PREEMPT_CPU").exists() or Path("/tmp/PREEMPT_CPU").is_symlink():
        raise ReferenceError("PREEMPT_CPU")
    meminfo = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0]) * 1024
    storage = os.statvfs(AUTHORITY_ROOT)
    storage_available = storage.f_bavail * storage.f_frsize
    if meminfo.get("MemAvailable", 0) < 16 * 1024**3:
        raise ReferenceError("MEMORY_PRESSURE")
    if storage_available < 120 * 1024**3:
        raise ReferenceError("STORAGE_GUARD")
    if cpu_backend and (os.environ.get("JAX_PLATFORMS") != "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES", "x") != ""):
        raise ReferenceError("CPU_BACKEND_ENV")
    return {
        "logical_cpu_ids": sorted(affinity), "thread_environment": environment,
        "nice": nice, "ionice_class": int(ioprio) >> 13,
        "memory_available_bytes": meminfo["MemAvailable"],
        "storage_available_bytes": storage_available, "preempt_cpu_absent": True,
        "cpu_backend_required": cpu_backend,
    }


def assert_backend_dark() -> None:
    forbidden = sorted(name for name in sys.modules if name == "jax" or name.startswith(("jax.", "gpuwrf.")))
    if forbidden:
        raise ReferenceError(f"BACKEND_IMPORTED_EARLY:{forbidden}")


def authority_graph() -> dict[str, Any]:
    graph, record = load_canonical(AUTHORITY_ROOT / "control/receipt-graph.json")
    if graph.get("complete") is not True or "capture-input-manifest.json" not in graph.get("manifests", {}):
        raise ReferenceError("RECEIPT_GRAPH_INCOMPLETE")
    for name, digest in graph["manifests"].items():
        require_file(AUTHORITY_ROOT / "control" / name, digest)
    return {"record": record, "canonical_payload_sha256": graph["canonical_payload_sha256"]}


def source_authority() -> dict[str, Any]:
    records = {name: require_file(REPO / name, digest) for name, digest in SOURCE_SHA256.items()}
    source = (REPO / "src/gpuwrf/physics/mynn_pbl.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    required_symbols = (
        "_mym_turbulence", "_solve_tridiagonal", "_step_mynn_pbl_impl_with_pblh",
        "_apply_mean_tendencies", "_diffusion_solve_with_mf",
    )
    symbols = {}
    for name in required_symbols:
        matches = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(matches) != 1:
            raise ReferenceError(f"SOURCE_SYMBOL_DRIFT:{name}")
        symbols[name] = {"line": matches[0].lineno, "ast_sha256": sha256_bytes(ast.dump(matches[0]).encode())}
    return {"files": records, "symbols": symbols}


def prior_five_authority() -> dict[str, Any]:
    value, record = load_canonical(PRIOR_REFERENCE, PRIOR_REFERENCE_SHA256)
    if value.get("available_arrays") != list(AVAILABLE) or value.get("available_array_count") != 5:
        raise ReferenceError("PRIOR_5_OF_28_DRIFT")
    matrix = {item["name"]: item for item in value.get("availability_matrix", []) if item.get("available")}
    if tuple(matrix) != AVAILABLE:
        raise ReferenceError("PRIOR_AVAILABLE_ORDER_DRIFT")
    return {"record": record, "arrays": matrix}


def dump_authority(*, deep: bool) -> dict[str, Any]:
    require_file(DUMP_ADMISSION, DUMP_ADMISSION_SHA256)
    admission = json.loads(DUMP_ADMISSION.read_text(encoding="utf-8"))
    leaves = admission.get("leaves", [])
    if admission.get("dump_tree_sha256") != DUMP_TREE_SHA256 or len(leaves) != 462:
        raise ReferenceError("DUMP_ADMISSION_DRIFT")
    expected = {item["path"]: item["sha256"] for item in leaves}
    if len(expected) != 462:
        raise ReferenceError("DUMP_DUPLICATE_PATH")
    if deep:
        observed = {}
        for path in sorted(DUMP_ROOT.rglob("*")):
            if path.is_symlink():
                raise ReferenceError(f"DUMP_SYMLINK:{path}")
            if path.is_file():
                observed[path.relative_to(DUMP_ROOT).as_posix()] = sha256_file(path)
        if observed != expected or canonical(observed) != DUMP_TREE_SHA256:
            raise ReferenceError("DUMP_TREE_DRIFT")
    return {"root": str(DUMP_ROOT), "tree_sha256": DUMP_TREE_SHA256, "leaf_count": 462, "deep": deep}


def g2_preflight(approved_head: str) -> dict[str, Any]:
    assert_backend_dark()
    if git_text("rev-parse", "HEAD") != approved_head or git_text("status", "--porcelain"):
        raise ReferenceError("WORKTREE_OR_HEAD_DRIFT")
    for commit in (CONTRACT_COMMIT, COORDINATION_COMMIT):
        if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, approved_head]).returncode:
            raise ReferenceError(f"COMMIT_NOT_ANCESTOR:{commit}")
    return {
        "schema": "wrfgpu2-v0234-pristine-pbl-g2-reference-preflight-v1",
        "approved_head": approved_head,
        "contract": require_file(CONTRACT, CONTRACT_SHA256),
        "coordination_commit": COORDINATION_COMMIT,
        "authority_graph": authority_graph(),
        "sources": source_authority(),
        "prior_reference": prior_five_authority(),
        "dump": dump_authority(deep=True),
        "resource_gate": resource_gate(cpu_backend=False),
        "reference_inventory": {
            "required": list(REQUIRED), "available": list(AVAILABLE), "missing": list(MISSING),
            "required_count": 28, "available_count": 5, "missing_count": 23,
            "status": "HOLD_5_OF_28", "comparator_dispatch_ready": False,
        },
        "capture_present": CAPTURE_ROOT.exists(),
        "backend_imported": False,
        "gpu_action_or_query": False,
        "passed": True,
        "checked_utc": now(),
    }


def array_record(array: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(array)
    if value.dtype.hasobject or not np.isfinite(value).all():
        raise ReferenceError("ARRAY_NONFINITE_OR_OBJECT")
    return {
        "shape": list(value.shape), "dtype": value.dtype.str,
        "byte_order": value.dtype.byteorder, "nbytes": int(value.nbytes),
        "c_contiguous": bool(value.flags.c_contiguous),
        "f_contiguous": bool(value.flags.f_contiguous),
        "logical_c_bitpayload_sha256": sha256_bytes(value.tobytes(order="C")),
    }


def validate_reference(
    manifest_path: Path,
    archive_path: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    manifest, manifest_record = load_canonical(manifest_path)
    if manifest.get("schema") != "wrfgpu2-v0234-pristine-pbl-cpu-reference-v1":
        raise ReferenceError("REFERENCE_SCHEMA")
    require_file(archive_path, manifest.get("archive_sha256"))
    if Path(manifest.get("archive_path", "")) != archive_path:
        raise ReferenceError("REFERENCE_ARCHIVE_PATH")
    with np.load(archive_path, allow_pickle=False) as archive:
        names = tuple(archive.files)
        expected = AVAILABLE if allow_partial else REQUIRED
        if names != expected:
            raise ReferenceError(f"REFERENCE_INVENTORY:{names}")
        declared = manifest.get("arrays")
        if (
            not isinstance(declared, dict)
            or set(declared) != set(names)
            or manifest.get("array_order") != list(names)
        ):
            raise ReferenceError("REFERENCE_DECLARED_INVENTORY")
        for name in names:
            observed = array_record(archive[name])
            if observed != declared[name]:
                raise ReferenceError(f"REFERENCE_ARRAY_DRIFT:{name}")
            if tuple(observed["shape"]) != EXPECTED_SHAPES[name] or observed["dtype"] != EXPECTED_DTYPE:
                raise ReferenceError(f"REFERENCE_SHAPE_OR_DTYPE:{name}")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != set(names):
        raise ReferenceError("REFERENCE_PROVENANCE_INVENTORY")
    for name in names:
        item = provenance[name]
        if not isinstance(item, dict) or item.get("synthetic") is not False:
            raise ReferenceError(f"REFERENCE_PROVENANCE:{name}")
        wrf_edges = item.get("wrf_edges", [])
        if name not in RESIDUALS and wrf_edges:
            raise ReferenceError(f"WRF_EDGE_TO_NONRESIDUAL:{name}")
        if name in RESIDUALS:
            if item.get("residual_stage") != RESIDUAL_STAGE[name] or not wrf_edges:
                raise ReferenceError(f"RESIDUAL_STAGE_DRIFT:{name}")
    if allow_partial:
        if manifest.get("reference_status") != "HOLD_5_OF_28" or manifest.get("missing_arrays") != list(MISSING):
            raise ReferenceError("PARTIAL_STATUS_DRIFT")
    else:
        if (
            manifest.get("reference_status") != "AUTHENTIC_28_OF_28"
            or manifest.get("newly_materialized_arrays") != list(MISSING)
            or manifest.get("comparator_dispatch_ready") is not True
        ):
            raise ReferenceError("COMPLETE_STATUS_DRIFT")
    return {
        "schema": "wrfgpu2-v0234-pristine-pbl-reference-validation-v1",
        "manifest": manifest_record, "archive_sha256": sha256_file(archive_path),
        "array_count": len(names), "array_order": list(names),
        "reference_status": manifest["reference_status"],
        "independent_validation": True,
        "valid": True,
    }


def parse_meta(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] in {"rank", "real_bytes", "ids_ide_jds_jde_kds_kde", "ims_ime_jms_jme_kms_kme", "ips_ipe_jps_jpe_kps_kpe"}:
            parsed[parts[0]] = [int(item) for item in parts[1:]]
        elif parts:
            parsed[parts[0]] = " ".join(parts[1:])
    if parsed.get("schema") != "wrfgpu2-mynn-sp2-v1" or parsed.get("real_bytes") != [4]:
        raise ReferenceError(f"DUMP_META_DRIFT:{path}")
    return parsed


def _bounds(meta: dict[str, Any]) -> tuple[range, range, range, range, range, range]:
    ids, ide, jds, jde, kds, kde = meta["ids_ide_jds_jde_kds_kde"]
    ims, ime, jms, jme, kms, kme = meta["ims_ime_jms_jme_kms_kme"]
    return range(ids, ide), range(jds, jde), range(kds, kde), range(ims, ime + 1), range(jms, jme + 1), range(kms, kme + 1)


def load_outer(metas: list[dict[str, Any]], tag: str, dimensions: int) -> np.ndarray:
    gi, gj, gk, _mi, _mj, _mk = _bounds(metas[0])
    shape = (len(gj), len(gi)) if dimensions == 2 else (len(gk), len(gj), len(gi))
    result = np.empty(shape, dtype=np.float32)
    occupied = np.zeros(shape, dtype=bool)
    for rank, meta in enumerate(metas):
        _gi, _gj, _gk, mi, mj, mk = _bounds(meta)
        ips, ipe, jps, jpe, kps, kpe = meta["ips_ipe_jps_jpe_kps_kpe"]
        raw = np.fromfile(DUMP_ROOT / "mynnsp2" / f"rank{rank:04d}" / f"step000001_outer__{tag}.bin", dtype=">f4")
        memory = raw.reshape((len(mi), len(mj)), order="F") if dimensions == 2 else raw.reshape((len(mi), len(mk), len(mj)), order="F")
        for j in range(max(jps, gj.start), min(jpe, gj.stop - 1) + 1):
            for i in range(max(ips, gi.start), min(ipe, gi.stop - 1) + 1):
                if dimensions == 2:
                    target = (j - gj.start, i - gi.start)
                    result[target] = memory[i - mi.start, j - mj.start]
                    occupied[target] = True
                else:
                    for k in range(max(kps, gk.start), min(kpe, gk.stop - 1) + 1):
                        target = (k - gk.start, j - gj.start, i - gi.start)
                        result[target] = memory[i - mi.start, k - mk.start, j - mj.start]
                        occupied[target] = True
    if not occupied.all():
        raise ReferenceError(f"DUMP_COVERAGE:{tag}")
    return result


def load_columns(metas: list[dict[str, Any]], tag: str) -> np.ndarray:
    gi, gj, _gk, _mi, _mj, _mk = _bounds(metas[0])
    columns: dict[tuple[int, int], np.ndarray] = {}
    bounds = None
    for rank in range(6):
        path = DUMP_ROOT / "mynnsp2" / f"rank{rank:04d}" / f"step000001_columns__{tag}.bin"
        with path.open("rb") as stream:
            while True:
                header = np.fromfile(stream, dtype=">i4", count=5)
                if not header.size:
                    break
                if header.size != 5:
                    raise ReferenceError(f"DUMP_COLUMN_HEADER:{tag}")
                i, j, lower, upper, count = (int(value) for value in header)
                values = np.fromfile(stream, dtype=">f4", count=count).astype(np.float32)
                if values.size != count or count != upper - lower + 1 or (i, j) in columns:
                    raise ReferenceError(f"DUMP_COLUMN_RECORD:{tag}:{i}:{j}")
                columns[(i, j)] = values
                bounds = (lower, upper) if bounds is None else bounds
                if bounds != (lower, upper):
                    raise ReferenceError(f"DUMP_COLUMN_BOUNDS:{tag}")
    if len(columns) != len(gi) * len(gj) or bounds is None:
        raise ReferenceError(f"DUMP_COLUMN_COVERAGE:{tag}")
    result = np.empty((bounds[1] - bounds[0] + 1, len(gj), len(gi)), dtype=np.float32)
    for (i, j), values in columns.items():
        result[:, j - gj.start, i - gi.start] = values
    return result


def load_capture_state(jax: Any, State: Any) -> tuple[Any, dict[str, Any]]:
    manifest_path = CAPTURE_ROOT / "manifest.json"
    manifest, record = load_canonical(manifest_path)
    status, _ = load_canonical(CAPTURE_ROOT / "run-status.json")
    if (
        manifest.get("nonce") != NONCE
        or manifest.get("status") != "CAPTURE_COMPLETE_HOLD_5_OF_28"
        or status.get("result", {}).get("status") != "CAPTURE_COMPLETE_HOLD_5_OF_28"
        or manifest.get("seam", {}).get("pbl_guard_invocation_count") != 0
        or manifest.get("seam", {}).get("trace_invocation_count") != 1
    ):
        raise ReferenceError("CAPTURE_TERMINAL_DRIFT")
    values = []
    for item in manifest.get("state_leaves", []):
        values.append(None if item["kind"] == "none" else jax.device_put(np.load(CAPTURE_ROOT / item["file"], allow_pickle=False)))
    if len(values) != len(State.__slots__):
        raise ReferenceError("CAPTURE_STATE_SCHEMA")
    state = State.tree_unflatten(None, tuple(values))
    return state, {"manifest": record, "value": manifest}


def _to_mass(couplers: Any, value: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(couplers._from_columns(value)))


def _run_pbl(
    state: Any,
    grid: Any,
    jax: Any,
    couplers: Any,
    mynn: Any,
    *,
    initialized_state: Any | None = None,
    dfm_substitution: np.ndarray | None = None,
    el_substitution: np.ndarray | None = None,
    record: bool = False,
) -> tuple[dict[str, np.ndarray], Any]:
    initialized = initialized_state or couplers._mynn_state_with_first_call_qke(state, grid, True)
    column = couplers._mynn_column_from_state(initialized, grid)
    surface = couplers._surface_fluxes_from_state(initialized)
    ny, nx = column.theta.shape[:2]
    column_b = couplers._flatten_columns_to_batch(column, ny, nx)
    surface_b = couplers._flatten_columns_to_batch(surface, ny, nx)
    solves: list[tuple[Any, Any, Any, Any, Any]] = []
    turbulence: dict[str, Any] = {}
    original_solve = mynn._solve_tridiagonal
    original_turbulence = mynn._mym_turbulence
    exact_sm: dict[str, Any] = {}

    def turbulence_trace(frame: Any, event: str, _arg: Any) -> Any:
        if event == "call":
            return turbulence_trace if frame.f_code is original_turbulence.__code__ else None
        if frame.f_code is original_turbulence.__code__ and event == "return":
            if "value" in exact_sm:
                raise ReferenceError("TURBULENCE_TRACE_REPEAT")
            if "sm" not in frame.f_locals:
                raise ReferenceError("TURBULENCE_TRACE_LOCAL")
            exact_sm["value"] = frame.f_locals["sm"]
        return turbulence_trace

    def solve_hook(a, b, c, d):
        x = original_solve(a, b, c, d)
        solves.append((a, b, c, d, x))
        return x

    def turbulence_hook(*args, **kwargs):
        result = dict(original_turbulence(*args, **kwargs))
        if el_substitution is not None:
            result["el"] = jax.numpy.asarray(el_substitution).transpose(1, 2, 0).reshape(ny * nx, -1)
        if dfm_substitution is not None:
            result["dfm"] = jax.numpy.asarray(dfm_substitution).transpose(1, 2, 0).reshape(ny * nx, -1)
        turbulence.update(result)
        return result

    mynn._solve_tridiagonal = solve_hook
    mynn._mym_turbulence = turbulence_hook
    sys.settrace(turbulence_trace)
    try:
        output, _pblh = mynn._step_mynn_pbl_impl_with_pblh(
            column_b, 6.0, False, surface_b, True, 1000.0
        )
    finally:
        sys.settrace(None)
        mynn._solve_tridiagonal = original_solve
        mynn._mym_turbulence = original_turbulence
    if len(solves) != 6:
        raise ReferenceError(f"TRIDIAGONAL_CALL_ORDER:{len(solves)}")
    result = {
        "rublten": (_to_mass(couplers, output.u.reshape((ny, nx, -1))) - np.asarray(couplers._u_mass(initialized))) / 6.0,
        "rvblten": (_to_mass(couplers, output.v.reshape((ny, nx, -1))) - np.asarray(couplers._v_mass(initialized))) / 6.0,
    }
    if record:
        result["mix_qke_initialized"] = np.asarray(initialized.qke)
        result["mix_el"] = _to_mass(couplers, turbulence["el"].reshape((ny, nx, -1)))
        if "value" not in exact_sm:
            raise ReferenceError("TURBULENCE_TRACE_MISSING")
        result["mix_sm"] = _to_mass(couplers, exact_sm["value"].reshape((ny, nx, -1)))
        result["mix_dfm"] = _to_mass(couplers, turbulence["dfm"].reshape((ny, nx, -1)))
        for component, index in (("u", 2), ("v", 3)):
            for name, value in zip(("a", "b", "c", "d", "x"), solves[index]):
                result[f"solve_{component}_{name}"] = _to_mass(couplers, value.reshape((ny, nx, -1)))
    return {name: np.ascontiguousarray(np.asarray(value)) for name, value in result.items()}, initialized


def ordered_surface_mixing_residuals(
    surface_run: Mapping[str, Any],
    mixing_run: Mapping[str, Any],
    wrf_target: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Emit the preregistered residual stages without changing their algebra."""

    result: dict[str, np.ndarray] = {}
    for stage, run in (("surface", surface_run), ("mixing", mixing_run)):
        for field in ("rublten", "rvblten"):
            result[f"residual_after_{stage}_{field}"] = np.asarray(run[field]) - np.asarray(
                wrf_target[field]
            )
    return result


def materialize(approved_head: str) -> dict[str, Any]:
    assert_backend_dark()
    if git_text("rev-parse", "HEAD") != approved_head or git_text("status", "--porcelain"):
        raise ReferenceError("WORKTREE_OR_HEAD_DRIFT")
    resource = resource_gate(cpu_backend=True)
    graph = authority_graph()
    sources = source_authority()
    prior = prior_five_authority()
    dump = dump_authority(deep=True)
    if any(REFERENCE_ROOT.iterdir()):
        raise ReferenceError("REFERENCE_OUTPUT_NOT_FRESH")
    sys.path.insert(0, str(REPO / "src"))
    import jax
    import jax.numpy as jnp
    from gpuwrf.contracts.state import State
    from gpuwrf.coupling import physics_couplers as couplers
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.physics import mynn_pbl as mynn

    if jax.default_backend() != "cpu" or any(device.platform != "cpu" for device in jax.devices()):
        raise ReferenceError("NON_CPU_BACKEND")
    state, capture = load_capture_state(jax, State)
    # The materializer needs only the authenticated d03 grid contract.  Loading
    # full operational domain bundles calls ``State.zeros()``, whose production
    # constructor correctly requires a GPU.  Build the same grid directly from
    # the sealed namelist/wrfinput bytes so this oracle remains CPU-only.
    grid = Gen2Run(INPUT_DIR).grid("d03").as_grid_spec()
    baseline, initialized = _run_pbl(state, grid, jax, couplers, mynn, record=True)

    metas = [parse_meta(DUMP_ROOT / "mynnsp2" / f"rank{rank:04d}" / "meta.txt") for rank in range(6)]
    wrf_surface = {name: load_outer(metas, name, 2) for name in ("ust", "hfx", "qfx", "tsk", "ch")}
    wrf_mix = {
        "qke": load_columns(metas, "mix_qke_initialized"),
        "el": load_columns(metas, "mix_el_for_dfm"),
        "sm": load_columns(metas, "mix_sm_for_dfm"),
        "dfm": load_columns(metas, "mix_dfm"),
    }
    wrf_system = {f"solve_{uv}_{name}": load_columns(metas, f"solve_{uv}_{name}")
                  for uv in ("u", "v") for name in ("a", "b", "c", "d", "x")}
    wrf_target = {name: load_outer(metas, f"{name}_exit", 3) for name in ("rublten", "rvblten")}

    qv0 = np.asarray(state.qv)[0]
    cpm = 1004.5 * (1.0 + 0.84 * np.maximum(qv0, 0.0))
    missing: dict[str, np.ndarray] = {
        "surface_hfx": np.asarray(state.theta_flux) * np.asarray(state.rhosfc) * cpm,
        "surface_qfx": np.asarray(state.qv_flux) * np.asarray(state.rhosfc),
        "surface_ch": np.load(CAPTURE_ROOT / "auxiliary/surface_ch.npy", allow_pickle=False),
        **{name: baseline[name] for name in MISSING[3:17]},
    }
    # Use the source-local sm value reconstructed at its exact dfm assignment;
    # reject if it does not re-emit dfm to floating representation.
    if not np.all(np.isfinite(missing["mix_sm"])):
        raise ReferenceError("MIX_SM_RECONSTRUCTION")

    surface_state = state.replace(
        ustar=jnp.asarray(wrf_surface["ust"], dtype=state.ustar.dtype),
        theta_flux=jnp.asarray(wrf_surface["hfx"] / (np.asarray(state.rhosfc) * cpm), dtype=state.theta_flux.dtype),
        qv_flux=jnp.asarray(wrf_surface["qfx"] / np.asarray(state.rhosfc), dtype=state.qv_flux.dtype),
        t_skin=jnp.asarray(wrf_surface["tsk"], dtype=state.t_skin.dtype),
    )
    fltv = (1.0 + 0.61 * np.maximum(qv0, 0.0)) * np.asarray(surface_state.theta_flux) \
        + 0.61 * np.asarray(state.theta)[0] * np.asarray(surface_state.qv_flux)
    surface_state = surface_state.replace(fltv=jnp.asarray(fltv, dtype=state.fltv.dtype))
    surface_run, _ = _run_pbl(surface_state, grid, jax, couplers, mynn)

    mix_initialized = state.replace(qke=jnp.asarray(wrf_mix["qke"], dtype=state.qke.dtype))
    mixing_run, _ = _run_pbl(
        state, grid, jax, couplers, mynn, initialized_state=mix_initialized,
        dfm_substitution=wrf_mix["dfm"], el_substitution=wrf_mix["el"],
    )
    missing.update(ordered_surface_mixing_residuals(surface_run, mixing_run, wrf_target))
    u_before = np.asarray(couplers._u_mass(initialized))
    v_before = np.asarray(couplers._v_mass(initialized))
    missing["residual_after_lower_bc_rublten"] = (wrf_system["solve_u_x"] - u_before) / 6.0 - wrf_target["rublten"]
    missing["residual_after_lower_bc_rvblten"] = (wrf_system["solve_v_x"] - v_before) / 6.0 - wrf_target["rvblten"]
    if tuple(missing) != MISSING:
        raise ReferenceError(f"MATERIALIZED_ORDER:{tuple(missing)}")

    with CARRY.open("rb") as stream:
        carry = pickle.load(stream)
    available = {
        "surface_ust": np.asarray(carry.state.ustar),
        "surface_tsk": np.asarray(carry.state.t_skin),
        "sp2_rublten": np.load(Path(prior["arrays"]["sp2_rublten"]["source_path"]), allow_pickle=False),
        "sp2_rvblten": np.load(Path(prior["arrays"]["sp2_rvblten"]["source_path"]), allow_pickle=False),
        "xland": np.asarray(carry.state.xland),
    }
    for name, array in available.items():
        if array_record(array)["logical_c_bitpayload_sha256"] != prior["arrays"][name]["logical_c_bitpayload_sha256"]:
            raise ReferenceError(f"PRIOR_ARRAY_DRIFT:{name}")
    arrays = {name: np.ascontiguousarray(available[name] if name in available else missing[name]) for name in REQUIRED}
    temporary = REFERENCE_ROOT / f".reference.tmp-{os.getpid()}.npz"
    np.savez(temporary, **arrays)
    os.link(temporary, REFERENCE_ARCHIVE)
    temporary.unlink()
    provenance = {}
    for name in REQUIRED:
        if name in AVAILABLE:
            provenance[name] = {"synthetic": False, "source": "prior_authenticated_5_of_28", "wrf_edges": []}
        elif name in RESIDUALS:
            provenance[name] = {
                "synthetic": False, "source": "source_ordered_authentic_wrf_substitution",
                "residual_stage": RESIDUAL_STAGE[name],
                "wrf_edges": ["sealed_substitution_operand", "sealed_rublten_or_rvblten_target"],
            }
        else:
            provenance[name] = {
                "synthetic": False, "source": "authenticated_gpu_capture_and_frozen_formula",
                "wrf_edges": [],
            }
    manifest = {
        "schema": "wrfgpu2-v0234-pristine-pbl-cpu-reference-v1",
        "contract_commit": CONTRACT_COMMIT, "approved_head": approved_head,
        "capture_nonce": NONCE, "capture_manifest_sha256": capture["manifest"]["sha256"],
        "authority_receipt_graph_sha256": graph["record"]["sha256"],
        "prior_reference_sha256": PRIOR_REFERENCE_SHA256,
        "wrf_dump_tree_sha256": dump["tree_sha256"],
        "source_sha256": SOURCE_SHA256, "source_symbols": sources["symbols"],
        "backend": "cpu", "gpu_actions": 0, "synthetic_values": False,
        "archive_path": str(REFERENCE_ARCHIVE), "archive_sha256": sha256_file(REFERENCE_ARCHIVE),
        "array_order": list(REQUIRED), "arrays": {name: array_record(arrays[name]) for name in REQUIRED},
        "available_arrays": list(AVAILABLE), "newly_materialized_arrays": list(MISSING),
        "missing_arrays": [], "reference_status": "AUTHENTIC_28_OF_28",
        "comparator_dispatch_ready": True, "provenance": provenance,
        "resource_gate": resource, "materialized_utc": now(),
    }
    atomic_json(REFERENCE_MANIFEST, manifest)
    validation = validate_reference(REFERENCE_MANIFEST, REFERENCE_ARCHIVE)
    atomic_json(REFERENCE_VALIDATION, validation)
    return {
        "verdict": "AUTHENTIC_28_OF_28", "archive_sha256": sha256_file(REFERENCE_ARCHIVE),
        "manifest_sha256": sha256_file(REFERENCE_MANIFEST),
        "validation_sha256": sha256_file(REFERENCE_VALIDATION),
        "gpu_actions": 0,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--g2-preflight", action="store_true")
    modes.add_argument("--materialize", action="store_true")
    modes.add_argument("--validate", action="store_true")
    value.add_argument("--approved-head")
    value.add_argument("--output", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--archive", type=Path)
    value.add_argument("--allow-partial", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.g2_preflight:
            if not args.approved_head or args.output is None:
                raise ReferenceError("G2_ARGS")
            result = g2_preflight(args.approved_head)
            atomic_json(args.output.resolve(), result)
        elif args.materialize:
            if not args.approved_head:
                raise ReferenceError("MATERIALIZE_ARGS")
            result = materialize(args.approved_head)
        else:
            result = validate_reference(
                (args.manifest or REFERENCE_MANIFEST).resolve(),
                (args.archive or REFERENCE_ARCHIVE).resolve(),
                allow_partial=args.allow_partial,
            )
            if args.output is not None:
                atomic_json(args.output.resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ReferenceError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
