#!/usr/bin/env python3
"""One-shot, fail-closed V0234 post-surface/pre-PBL State capture.

The module is CPU/dark at import time: it imports no JAX, gpuwrf, CUDA, WRF,
MPI, or comparator package.  The ``--execute`` path dynamically imports the
frozen model only after it has authenticated a live canonical lock-v2 lease.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import re
import signal
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-pristine-pbl-entry-closure-gpt"
CONTRACT = SPRINT / "CONTRACT.md"
CONTRACT_SHA256 = "db445ed5ca18250c932651c4bf506799778895baf9390d692f65c9dfa21b6f35"
CONTRACT_COMMIT = "5bbfe0416979eb41dd72bf80a8539191740f8eea"
CONTRACT_PATCH = REPO / ".agent/patches/2026-07-19-v0234-corrected-g3-fresh-arm.md"
CONTRACT_PATCH_SHA256 = "979ccda45f85aca4315fd06647b5f9e50edef4b9070b593a20c767ab4a560dbb"
CONTRACT_PATCH_COMMIT = "aa16ef9024ad3fcbfc1d37c65e09d3540ffa390b"
COORDINATION_COMMIT = "8231225573683dc56828caba2be3326508bdfd73"
AUTHORITY_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_0b18530a1dc9cac2")
OUTPUT_AUTHORITY_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084")

NONCE = "ac6712170cbe508475ea8572464c7742a6906d559b53ed9930292d6822229ed8"
OUTPUT_DIR = OUTPUT_AUTHORITY_ROOT / "capture/authentic-ac6712170cbe5084"
EXECUTION_RECEIPT = SPRINT / "execution-receipt.json"
INPUT_DIR = AUTHORITY_ROOT / "capture-inputs"
CARRY = AUTHORITY_ROOT / "inputs/last-healthy-d03-step-0.pkl"
CARRY_SHA256 = "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
NAMELIST = INPUT_DIR / "namelist.input"
NAMELIST_SHA256 = "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838"
WRFINPUT = INPUT_DIR / "wrfinput_d03"
WRFINPUT_SHA256 = "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a"
CAPTURE_INPUT_SHA256 = {
    "namelist.input": NAMELIST_SHA256,
    "wrfinput_d01": "afd069d201c14f0001308696f3b5b2f30f4e1b77f368d15a0d9d7c5b77dc0756",
    "wrfinput_d02": "ec281f234940d7353b9fdf544833db635b4388e83bab4a7a71e025f23f36b964",
    "wrfinput_d03": WRFINPUT_SHA256,
    "wrfbdy_d01": "1b5b20408b3384b2e81dc09f3ec029c3110117745e623dc0b7669f0976121fec",
}
REQUIRED_AUTHORITY_MANIFESTS = (
    "authority-manifest.json",
    "build-manifest.json",
    "capture-input-manifest.json",
    "environment-manifest.json",
    "external-source-manifest.json",
    "input-manifest.json",
    "runtime-manifest.json",
    "source-manifest.json",
    "toolchain-manifest.json",
)

MODEL_ROOT = REPO
MODEL_SOURCE = MODEL_ROOT / "src/gpuwrf/runtime/operational_mode.py"
MODEL_SOURCE_SHA256 = "68efe76b9d1f91860e9a49a6e6c9ab9e573986dd11a47677fa161badb3a79282"
MODEL_STATE_SOURCE = MODEL_ROOT / "src/gpuwrf/contracts/state.py"
MODEL_STATE_SOURCE_SHA256 = "f959da39d8957ef72f165c2e4fad98a03e8235a58379e5a2743f9a6da93af9f8"
NOAHMP_COUPLER_SOURCE = MODEL_ROOT / "src/gpuwrf/physics/noahmp_coupler.py"
NOAHMP_COUPLER_SOURCE_SHA256 = "cd857efa18731d796fd4a60047eb833fae94588e3150d2973cbd8100e34ced76"

PYTHON_EXECUTABLE = Path("<USER_HOME>/miniconda3/bin/python3.13")
PYTHON_EXECUTABLE_SHA256 = "3851f1998dcd3a08bd2554a3a0f1f439c9227e80afff2791d4df6da1b3456492"
NUMPY_INIT = Path("<USER_HOME>/.local/lib/python3.13/site-packages/numpy/__init__.py")
NUMPY_INIT_SHA256 = "2e8da3e4385e79c4885b3f7324a8b957e6f01732b239e99e266a12c62a008b8d"
NUMPY_VERSION = "2.4.4"

LOCK_ROOT = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-gpu-lock-v2")
LOCK_COMMIT = "8152309aff1e85e1052d44d549a5a5409e710bdd"
LOCK_TREE = "309e65f998de7c9d65daf8c106be40a7415d0b8a"
LOCK_WRAPPER = LOCK_ROOT / "scripts/with_gpu_lock.sh"
LOCK_WRAPPER_SHA256 = "c75b3a4eda17e94df921986e1c15fcc71e182e01d51b6fe877e4c52533077e1a"
LOCK_VERIFIER = LOCK_ROOT / "scripts/gpu_lock_v2.py"
LOCK_VERIFIER_SHA256 = "ec911f565ee9d2c58dee81d8d1e17411ed677be500a57e73ded5f793e2c4187d"
LOCK_LABEL = "v0234-pristine-pbl-capture-ac6712170cbe5084"
LOCK_FILE = Path("/tmp/wrf_gpu2_gpu.lock")
LOCK_HOLDER = Path("/tmp/wrf_gpu2_gpu.lock.holder")

PREEMPT_PATHS = (
    Path("/tmp/PREEMPT_GPU"),
    Path("/tmp/PREEMPT_PRODUCTION"),
    Path("<DATA_ROOT>/alisios/state/PREEMPT_PRODUCTION"),
    Path("<DATA_ROOT>/alisios/state/PREEMPT_GPU"),
)
ALLOWED_CPUS = {13, 14, 15, 29, 30, 31}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
REQUIRED_RUNTIME_ENV = {
    "JAX_PLATFORMS": "cuda",
    "JAX_ENABLE_X64": "true",
    "CUDA_VISIBLE_DEVICES": "0",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "GPUWRF_ALLOCATOR": "cuda_async",
    "GPUWRF_FINITE_CHECK": "1",
    "GPUWRF_NESTED_FUSE": "0",
    "GPUWRF_NESTED_DEFUSE_COMPILE": "0",
    "GPUWRF_NESTED_PARALLEL_COMPILE": "0",
    "GPUWRF_NESTED_AOT": "0",
    "GPUWRF_AOT_VERIFY": "0",
    "GPUWRF_NESTED_ASYNC_OUTPUT": "0",
    "GPUWRF_NEST_OUTPUT_PIPELINE": "0",
    "GPUWRF_BATCH_ENSEMBLE": "1",
    "GPUWRF_NESTED_SYNC_MODE": "root",
    "GPUWRF_BITWISE": "1",
    "GPUWRF_ADVANCE_CHUNK_LOOP": "fori",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": str(AUTHORITY_ROOT / "runtime/wrf"),
}


class GateError(RuntimeError):
    """Fail-closed admission or evidence error."""


class CaptureStop(BaseException):
    """Private source-seam stop before PBL."""


class CapturePreempted(BaseException):
    """Production preemption signal or sentinel."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def canonical_without_self(value: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: item for key, item in value.items() if key != "canonical_payload_sha256"
    })


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: Mapping[str, Any], *, replace: bool = True) -> None:
    if path.is_symlink() or (not replace and path.exists()):
        raise GateError(f"refuse JSON destination: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}")
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        os.link(temporary, path)
        temporary.unlink()
    _fsync_directory(path.parent)


def require_file(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise GateError(f"invalid authority file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise GateError(f"AUTHORITY_DRIFT:{path}:{actual}")
    info = path.stat()
    return {
        "path": str(path),
        "sha256": actual,
        "size": info.st_size,
        "mode": info.st_mode & 0o777,
    }


def git_text(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise GateError(f"git {' '.join(arguments)} failed at {root}: {result.stderr}")
    return result.stdout.strip()


def resource_attestation() -> dict[str, Any]:
    affinity = set(os.sched_getaffinity(0))
    env = {name: os.environ.get(name) for name in THREAD_ENV}
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    if affinity != ALLOWED_CPUS:
        raise GateError(f"CPU_AFFINITY:{sorted(affinity)}")
    if env != THREAD_ENV:
        raise GateError(f"THREAD_ENV:{env}")
    if nice < 15:
        raise GateError(f"NICENESS:{nice}")
    if os.uname().machine != "x86_64":
        raise GateError("IOPRIO_ARCH")
    ioprio = ctypes.CDLL(None, use_errno=True).syscall(252, 1, 0)
    if ioprio < 0 or int(ioprio) >> 13 != 3:
        raise GateError(f"IOPRIO:{ioprio}")
    cpu_preempt = Path("/tmp/PREEMPT_CPU")
    if cpu_preempt.exists() or cpu_preempt.is_symlink():
        raise CapturePreempted("resource-attestation:/tmp/PREEMPT_CPU")
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0]) * 1024
    available_memory = meminfo.get("MemAvailable", 0)
    if available_memory < 16 * 1024**3:
        raise GateError(f"MEMORY_PRESSURE:{available_memory}")
    storage = os.statvfs(AUTHORITY_ROOT)
    available_storage = storage.f_bavail * storage.f_frsize
    if available_storage < 120 * 1024**3:
        raise GateError(f"STORAGE_GUARD:{available_storage}")
    return {
        "logical_cpu_ids": sorted(affinity),
        "thread_environment": env,
        "niceness": nice,
        "ionice_class": int(ioprio) >> 13,
        "memory_available_bytes": available_memory,
        "storage_available_bytes": available_storage,
        "preempt_cpu_absent": True,
    }


def assert_no_backend_imported() -> None:
    forbidden = sorted(
        name for name in sys.modules
        if name == "jax" or name.startswith(("jax.", "gpuwrf.", "mpi4py", "netCDF4"))
    )
    if forbidden:
        raise GateError(f"BACKEND_IMPORTED_BEFORE_LEASE:{forbidden}")


def interpreter_authority() -> dict[str, Any]:
    observed = Path(sys.executable).resolve()
    if observed != PYTHON_EXECUTABLE:
        raise GateError(f"PYTHON_EXECUTABLE_DRIFT:{observed}")
    numpy_source = Path(np.__file__).resolve()
    if numpy_source != NUMPY_INIT or np.__version__ != NUMPY_VERSION:
        raise GateError(f"NUMPY_AUTHORITY_DRIFT:{numpy_source}:{np.__version__}")
    return {
        "python": require_file(PYTHON_EXECUTABLE, PYTHON_EXECUTABLE_SHA256),
        "numpy": require_file(NUMPY_INIT, NUMPY_INIT_SHA256),
        "numpy_version": NUMPY_VERSION,
    }


def assert_preemption_clear(label: str) -> dict[str, Any]:
    present = [str(path) for path in PREEMPT_PATHS if path.exists() or path.is_symlink()]
    if present:
        raise CapturePreempted(f"{label}:{present}")
    return {"label": label, "checked_utc": utc_now(), "sentinels_absent": [str(p) for p in PREEMPT_PATHS]}


def load_canonical_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise GateError(f"CANONICAL_JSON_NOT_REGULAR:{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateError(f"CANONICAL_JSON_INVALID:{path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"CANONICAL_JSON_NOT_OBJECT:{path}")
    claimed = value.get("canonical_payload_sha256")
    if not isinstance(claimed, str) or canonical_without_self(value) != claimed:
        raise GateError(f"CANONICAL_JSON_DRIFT:{path}")
    return value, require_file(path, sha256_file(path))


def authority_graph() -> dict[str, Any]:
    control = AUTHORITY_ROOT / "control"
    graph_path = control / "receipt-graph.json"
    graph, record = load_canonical_json(graph_path)
    if graph.get("schema") != "wrfgpu2-v0234-receipt-graph-v1" or graph.get("complete") is not True:
        raise GateError("AUTHORITY_GRAPH_INCOMPLETE")
    manifests = graph.get("manifests")
    if not isinstance(manifests, dict):
        raise GateError("AUTHORITY_GRAPH_MANIFESTS")
    authenticated: dict[str, Any] = {}
    for name in REQUIRED_AUTHORITY_MANIFESTS:
        expected = manifests.get(name)
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise GateError(f"AUTHORITY_MANIFEST_NOT_BOUND:{name}")
        path = control / name
        manifest, manifest_record = load_canonical_json(path)
        if manifest_record["sha256"] != expected:
            raise GateError(f"AUTHORITY_MANIFEST_DRIFT:{name}")
        authenticated[name] = {
            "record": manifest_record,
            "schema": manifest.get("schema"),
            "canonical_payload_sha256": manifest["canonical_payload_sha256"],
        }
    build, _ = load_canonical_json(control / "build-manifest.json")
    if (
        build.get("verdict") != "FRESH_PINNED_WRF_BUILD_COMPLETE"
        or build.get("wrf_commit") != "f52c197ed39d12e087d02c50f412d90d418f6186"
        or build.get("wrf_tree") != "6b658fbc98077fe0648cba724921679126464181"
        or build.get("wrf_or_mpi_executions") != 0
        or build.get("gpu_actions") != 0
    ):
        raise GateError("BUILD_AUTHORITY_TERMINAL_DRIFT")
    capture_inputs, _ = load_canonical_json(control / "capture-input-manifest.json")
    if capture_inputs.get("complete") is not True or capture_inputs.get("required_files") != list(CAPTURE_INPUT_SHA256):
        raise GateError("CAPTURE_INPUT_MANIFEST_INCOMPLETE")
    return {
        "root": str(AUTHORITY_ROOT),
        "receipt_graph": record,
        "receipt_graph_canonical_sha256": graph["canonical_payload_sha256"],
        "manifests": authenticated,
        "complete": True,
    }


def _git_authority(root: Path, commit: str, tree: str, *, tracked_clean: bool) -> dict[str, Any]:
    observed_commit = git_text(root, "rev-parse", "HEAD")
    observed_tree = git_text(root, "rev-parse", "HEAD^{tree}")
    if (observed_commit, observed_tree) != (commit, tree):
        raise GateError(f"GIT_AUTHORITY_DRIFT:{root}:{observed_commit}:{observed_tree}")
    dirty = git_text(root, "status", "--porcelain", "--untracked-files=no")
    if tracked_clean and dirty:
        raise GateError(f"TRACKED_WORKTREE_DIRTY:{root}:{dirty}")
    return {"root": str(root), "commit": commit, "tree": tree, "tracked_clean": not bool(dirty)}


def seam_authority(source_path: Path = MODEL_SOURCE, expected_sha256: str = MODEL_SOURCE_SHA256) -> dict[str, Any]:
    record = require_file(source_path, expected_sha256)
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(source_path))
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_physics_step_forcing"]
    if len(functions) != 1:
        raise GateError("PHYSICS_FUNCTION_NOT_UNIQUE")
    function = functions[0]
    matches: list[tuple[int, ast.Assign]] = []
    for index, statement in enumerate(function.body):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id == "pbl_entry_state":
            if not isinstance(statement.value, ast.Name) or statement.value.id != "next_state":
                raise GateError("PBL_ENTRY_ASSIGNMENT_DRIFT")
            matches.append((index, statement))
    if len(matches) != 1:
        raise GateError("PBL_ENTRY_ASSIGNMENT_NOT_UNIQUE")
    index, assignment = matches[0]
    if index + 1 >= len(function.body) or not isinstance(function.body[index + 1], ast.If):
        raise GateError("PBL_BRANCH_NOT_IMMEDIATELY_AFTER_ENTRY_ASSIGNMENT")
    branch = function.body[index + 1]
    return {
        "source": record,
        "function_first_line": function.lineno,
        "assignment_line": assignment.lineno,
        "stop_line": branch.lineno,
        "function_ast_sha256": sha256_bytes(ast.dump(function, include_attributes=True).encode()),
        "assignment": "pbl_entry_state = next_state",
        "next_statement_type": "If",
    }


def state_schema_authority(
    source_path: Path = MODEL_STATE_SOURCE,
    expected_sha256: str = MODEL_STATE_SOURCE_SHA256,
) -> dict[str, Any]:
    """Read the frozen State slot schema without importing JAX or the model."""

    record = require_file(source_path, expected_sha256)
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    classes = [node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "State"]
    if len(classes) != 1:
        raise GateError("STATE_CLASS_NOT_UNIQUE")
    assignments: list[ast.Assign] = []
    for statement in classes[0].body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id == "__slots__":
            assignments.append(statement)
    if len(assignments) != 1:
        raise GateError("STATE_SLOTS_NOT_UNIQUE")
    literal = assignments[0].value
    if not isinstance(literal, ast.Tuple):
        raise GateError("STATE_SLOTS_NOT_LITERAL_TUPLE")
    names: list[str] = []
    for item in literal.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str) or not item.value:
            raise GateError("STATE_SLOT_NOT_NONEMPTY_LITERAL")
        names.append(item.value)
    if not names or len(names) != len(set(names)):
        raise GateError("STATE_SLOT_DUPLICATE_OR_EMPTY")
    return {
        "source": record,
        "class_line": classes[0].lineno,
        "slots_line": assignments[0].lineno,
        "state_slot_count": len(names),
        "slot_names": names,
        "slot_names_sha256": canonical_sha256(names),
    }


def assert_fresh_output_and_nonce(
    *,
    output: Path,
    receipt: Path,
) -> None:
    parent = output.parent
    if (
        parent.is_symlink()
        or parent.resolve() != parent
        or not parent.is_dir()
        or not os.access(parent, os.W_OK | os.X_OK)
    ):
        raise GateError(f"OUTPUT_PARENT_NOT_ADMITTED:{parent}")
    if output.exists() or output.is_symlink():
        raise GateError(f"OUTPUT_NOT_FRESH:{output}")
    if receipt.exists() or receipt.is_symlink():
        raise GateError("NONCE_ALREADY_CONSUMED_BY_LAUNCH_RECEIPT")


def static_preflight(approved_head: str, *, require_output_absent: bool = True) -> dict[str, Any]:
    assert_no_backend_imported()
    if git_text(REPO, "rev-parse", "HEAD") != approved_head:
        raise GateError("APPROVED_HEAD_MISMATCH")
    if git_text(REPO, "status", "--porcelain"):
        raise GateError("CAPTURE_WORKTREE_NOT_CLEAN")
    for commit, label in (
        (CONTRACT_COMMIT, "CONTRACT"),
        (CONTRACT_PATCH_COMMIT, "CONTRACT_PATCH"),
        (COORDINATION_COMMIT, "COORDINATION"),
    ):
        ancestor = subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, approved_head],
            check=False,
        )
        if ancestor.returncode != 0:
            raise GateError(f"{label}_NOT_ANCESTOR")
    if require_output_absent:
        assert_fresh_output_and_nonce(
            output=OUTPUT_DIR,
            receipt=EXECUTION_RECEIPT,
        )
    preemption = assert_preemption_clear("static-preflight")
    graph = authority_graph()
    inputs = {name: require_file(INPUT_DIR / name, digest) for name, digest in CAPTURE_INPUT_SHA256.items()}
    return {
        "schema": "wrfgpu2-v0234-pbl-entry-capture-preflight-v1",
        "approved_head": approved_head,
        "contract": {
            "commit": CONTRACT_COMMIT,
            "record": require_file(CONTRACT, CONTRACT_SHA256),
            "patch_commit": CONTRACT_PATCH_COMMIT,
            "patch_record": require_file(CONTRACT_PATCH, CONTRACT_PATCH_SHA256),
            "coordination_commit": COORDINATION_COMMIT,
        },
        "authority_graph": graph,
        "nonce": NONCE,
        "output": {"path": str(OUTPUT_DIR), "absent": True},
        "model": {
            "git": _git_authority(
                MODEL_ROOT,
                approved_head,
                git_text(REPO, "rev-parse", f"{approved_head}^{{tree}}"),
                tracked_clean=True,
            ),
            "seam": seam_authority(),
            "state_schema": state_schema_authority(),
            "noahmp_coupler_source": require_file(
                NOAHMP_COUPLER_SOURCE, NOAHMP_COUPLER_SOURCE_SHA256
            ),
        },
        "inputs": {
            "carry": require_file(CARRY, CARRY_SHA256),
            "capture_input_closure": inputs,
        },
        "lock_v2": {
            "git": _git_authority(LOCK_ROOT, LOCK_COMMIT, LOCK_TREE, tracked_clean=False),
            "wrapper": require_file(LOCK_WRAPPER, LOCK_WRAPPER_SHA256),
            "verifier": require_file(LOCK_VERIFIER, LOCK_VERIFIER_SHA256),
            "label": LOCK_LABEL,
            "intent": "production-preemptible",
            "timeout_seconds": 0,
        },
        "preemption": preemption,
        "resources": resource_attestation(),
        "interpreter": interpreter_authority(),
        "backend_imported": False,
        "gpu_query_performed": False,
        "passed": True,
        "checked_utc": utc_now(),
    }


def load_lock_verifier() -> ModuleType:
    require_file(LOCK_VERIFIER, LOCK_VERIFIER_SHA256)
    spec = importlib.util.spec_from_file_location("v0234_capture_lock_v2", LOCK_VERIFIER)
    if spec is None or spec.loader is None:
        raise GateError("LOCK_VERIFIER_IMPORT_REFUSED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_live_lock() -> dict[str, Any]:
    required = {
        "GPUWRF_GPU_LOCK_HELD": "1",
        "GPUWRF_GPU_LOCK_FD": "9",
        "GPUWRF_GPU_LOCK_FILE": str(LOCK_FILE),
        "GPUWRF_GPU_LOCK_HOLDER_FILE": str(LOCK_HOLDER),
        "GPUWRF_GPU_LOCK_LABEL": LOCK_LABEL,
    }
    for name, expected in required.items():
        if os.environ.get(name) != expected:
            raise GateError(f"LIVE_LOCK_ENV:{name}:{os.environ.get(name)!r}")
    token = os.environ.get("GPUWRF_GPU_LOCK_TOKEN", "")
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise GateError("LIVE_LOCK_TOKEN")
    if git_text(LOCK_ROOT, "rev-parse", "HEAD") != LOCK_COMMIT:
        raise GateError("LIVE_LOCK_COMMIT")
    module = load_lock_verifier()
    lease = module._read_sidecar(LOCK_FILE)
    module._validate_schema(dict(lease))
    if lease.get("intent") != "production-preemptible":
        raise GateError("LIVE_LOCK_NOT_PREEMPTIBLE")
    if lease.get("label") != LOCK_LABEL or lease.get("lease_id") != token:
        raise GateError("LIVE_LOCK_LEASE_MISMATCH")
    module._validate_lock_lease_fd(LOCK_FILE, 9, lease)
    probe = module._open_gpu_lock(LOCK_FILE)
    try:
        module._validate_live_lease(LOCK_FILE, probe, expected=lease)
    finally:
        os.close(probe)
    return {
        "schema": lease["schema"],
        "version": lease["version"],
        "intent": lease["intent"],
        "label": lease["label"],
        "lease_id_sha256": sha256_bytes(token.encode()),
        "fd": 9,
        "live_verified": True,
    }


def validate_runtime_environment() -> dict[str, Any]:
    assert_no_backend_imported()
    observed = {key: os.environ.get(key) for key in REQUIRED_RUNTIME_ENV}
    if observed != REQUIRED_RUNTIME_ENV:
        raise GateError(f"RUNTIME_ENV_DRIFT:{observed}")
    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath.split(os.pathsep)[0] != str(MODEL_ROOT / "src"):
        raise GateError(f"PYTHONPATH_MODEL_ROOT:{pythonpath}")
    return {"required": observed, "pythonpath": pythonpath, "checked_before_jax_import": True}


class RuntimeJournal:
    def __init__(self, output: Path):
        self.output = output
        self.events: list[dict[str, Any]] = []

    def add(self, event: str, **details: Any) -> None:
        self.events.append({"index": len(self.events), "utc": utc_now(), "event": event, **details})
        atomic_json(self.output / "events.json", {"events": self.events})


class SentinelWatcher:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.observed: list[str] = []
        self.thread = threading.Thread(target=self._run, name="v0234-preempt-watch", daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(0.05):
            present = [str(path) for path in PREEMPT_PATHS if path.exists() or path.is_symlink()]
            if present:
                self.observed = present
                os.kill(os.getpid(), signal.SIGINT)
                return

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)


class SeamTracer:
    def __init__(self, code: Any, stop_line: int, noahmp_code: Any | None = None):
        self.code = code
        self.noahmp_code = noahmp_code
        self.stop_line = int(stop_line)
        self.count = 0
        self.state: Any = None
        self.frame_evidence: dict[str, Any] = {}
        self.noahmp_count = 0
        self.surface_sfclay_ch: Any = None
        self.surface_noahmp_land_ch: Any = None
        self.surface_is_land: Any = None

    def __call__(self, frame: Any, event: str, _arg: Any) -> Any:
        if event == "call":
            return self if frame.f_code in (self.code, self.noahmp_code) else None
        if frame.f_code is self.noahmp_code and event == "return":
            if self.noahmp_count != 0:
                raise GateError("NOAHMP_SURFACE_TRACE_REPEAT")
            local = frame.f_locals
            diag = local.get("diag")
            land_state_out = local.get("land_state_out")
            if (
                diag is None
                or land_state_out is None
                or "is_land" not in local
                or "ch_seed" not in local
            ):
                raise GateError("NOAHMP_SURFACE_TRACE_LOCALS")
            # ``SurfaceLayerDiagnostics`` does not expose CH.  The exact value
            # consumed by Noah-MP is the adapter's already-materialized
            # ``ch_seed`` local (which may intentionally fall back to the land
            # carry), so retain that live operand rather than re-deriving it.
            self.surface_sfclay_ch = local["ch_seed"]
            self.surface_noahmp_land_ch = getattr(land_state_out, "ch")
            self.surface_is_land = local["is_land"]
            self.noahmp_count = 1
            return self
        if frame.f_code is self.code and event == "line" and frame.f_lineno == self.stop_line:
            if self.count != 0:
                raise GateError("SEAM_TRACE_REPEAT")
            local = frame.f_locals
            if "pbl_entry_state" not in local or local.get("pbl_entry_state") is not local.get("next_state"):
                raise GateError("SEAM_LOCAL_IDENTITY")
            self.count = 1
            self.state = local["pbl_entry_state"]
            self.frame_evidence = {
                "function": frame.f_code.co_name,
                "filename": str(Path(frame.f_code.co_filename).resolve()),
                "first_line": frame.f_code.co_firstlineno,
                "stop_line": frame.f_lineno,
                "bl_opt": int(local.get("bl_opt")),
                "source_leaf_mode": bool(local.get("source_leaf_mode")),
            }
            raise CaptureStop
        return self


class PBLGuard:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        self.count += 1
        raise GateError("PBL_EXECUTION_FORBIDDEN")


def _safe_slot_filename(index: int, name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return f"{index:03d}-{safe}.npy"


def _write_npy_exclusive(path: Path, array: np.ndarray) -> None:
    if array.dtype.hasobject:
        raise GateError(f"OBJECT_DTYPE:{path.name}")
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _runtime_slot_values(state: Any, expected_names: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    state_type = type(state)
    raw_slots = state_type.__dict__.get("__slots__")
    if not isinstance(raw_slots, tuple):
        raise GateError("RUNTIME_STATE_SLOTS_NOT_TUPLE")
    names = tuple(raw_slots)
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise GateError("RUNTIME_STATE_SLOT_DUPLICATE_OR_EMPTY")
    if hasattr(state, "__dict__") or "__dict__" in names or "__weakref__" in names:
        raise GateError("RUNTIME_STATE_DYNAMIC_OR_INHERITED")
    for base in state_type.__mro__[1:-1]:
        inherited = base.__dict__.get("__slots__")
        if inherited not in (None, ()):
            raise GateError("RUNTIME_STATE_DYNAMIC_OR_INHERITED")
    if names != expected_names:
        raise GateError("RUNTIME_STATE_SLOT_SCHEMA_DRIFT")
    values: list[Any] = []
    for name in names:
        try:
            values.append(getattr(state, name))
        except AttributeError as exc:
            raise GateError(f"RUNTIME_STATE_SLOT_UNSET:{name}") from exc
    flatten = getattr(state, "tree_flatten", None)
    if not callable(flatten):
        raise GateError("RUNTIME_STATE_TREE_FLATTEN_MISSING")
    direct_children, aux = flatten()
    if aux is not None or not isinstance(direct_children, tuple) or len(direct_children) != len(values):
        raise GateError("RUNTIME_STATE_DIRECT_TREE_SCHEMA")
    if any(child is not value for child, value in zip(direct_children, values)):
        raise GateError("RUNTIME_STATE_DIRECT_TREE_ORDER")
    return names, tuple(values)


def materialize_state(
    state: Any,
    output: Path,
    jax: ModuleType,
    expected_slot_names: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schema = state_schema_authority()
    expected = tuple(schema["slot_names"]) if expected_slot_names is None else expected_slot_names
    names, values = _runtime_slot_values(state, expected)
    for name, value in zip(names, values):
        if value is not None and (not hasattr(value, "shape") or not hasattr(value, "dtype")):
            raise GateError(f"UNSUPPORTED_STATE_SLOT:{name}:{type(value)!r}")
    path_leaves, _treedef = jax.tree_util.tree_flatten_with_path(state)
    array_slots = [(index, name, value) for index, (name, value) in enumerate(zip(names, values)) if value is not None]
    if len(path_leaves) != len(array_slots):
        raise GateError(f"PYTREE_ARRAY_COUNT:{len(path_leaves)}:{len(array_slots)}")
    for (path, leaf), (slot_index, slot_name, slot_value) in zip(path_leaves, array_slots):
        if leaf is not slot_value:
            raise GateError(f"PYTREE_SLOT_ORDER:{slot_index}:{slot_name}:{jax.tree_util.keystr(path)}")

    arrays_dir = output / "arrays"
    arrays_dir.mkdir(mode=0o700, exist_ok=False)
    leaves: list[dict[str, Any]] = []
    array_count = 0
    none_count = 0
    host_arrays: dict[int, np.ndarray] = {}
    for index, (name, value) in enumerate(zip(names, values)):
        if value is None:
            none_count += 1
            leaves.append({"slot_name": name, "kind": "none", "slot_index": index})
            continue
        array = np.asarray(jax.device_get(value))
        if array.dtype.hasobject:
            raise GateError(f"OBJECT_STATE_SLOT:{name}")
        contiguous = np.ascontiguousarray(array)
        host_arrays[index] = contiguous
        payload = contiguous.tobytes(order="C")
        relative = Path("arrays") / _safe_slot_filename(index, name)
        path = output / relative
        _write_npy_exclusive(path, contiguous)
        array_count += 1
        leaves.append({
            "slot_name": name,
            "kind": "array",
            "slot_index": index,
            "shape": list(contiguous.shape),
            "dtype": contiguous.dtype.str,
            "byte_order": contiguous.dtype.byteorder,
            "nbytes": int(contiguous.nbytes),
            "logical_c_bitpayload_sha256": sha256_bytes(payload),
            "file": relative.as_posix(),
            "file_sha256": sha256_file(path),
            "file_bytes": path.stat().st_size,
        })
    _fsync_directory(arrays_dir)
    pytree_records: list[dict[str, Any]] = []
    for (path, value), (slot_index, slot_name, _slot_value) in zip(path_leaves, array_slots):
        independent = np.ascontiguousarray(np.asarray(jax.device_get(value)))
        primary = host_arrays[slot_index]
        if (
            independent.shape != primary.shape
            or independent.dtype != primary.dtype
            or independent.nbytes != primary.nbytes
            or independent.tobytes(order="C") != primary.tobytes(order="C")
        ):
            raise GateError(f"PYTREE_SLOT_PAYLOAD:{slot_index}:{slot_name}")
        pytree_records.append({
            "slot_index": slot_index,
            "slot_name": slot_name,
            "path": jax.tree_util.keystr(path),
            "shape": list(independent.shape),
            "dtype": independent.dtype.str,
            "nbytes": int(independent.nbytes),
            "logical_c_bitpayload_sha256": sha256_bytes(independent.tobytes(order="C")),
        })
    summary = {
        "state_slot_count": len(names),
        "array_slot_count": array_count,
        "none_slot_count": none_count,
        "slot_names": list(names),
        "slot_names_sha256": canonical_sha256(list(names)),
        "direct_tree_child_count": len(values),
        "direct_tree_aux_is_none": True,
        "pytree_leaf_count": len(pytree_records),
        "pytree_inventory_sha256": canonical_sha256(pytree_records),
        "pytree_leaves": pytree_records,
        "leaf_inventory_sha256": canonical_sha256(leaves),
    }
    return leaves, summary


def materialize_surface_ch_auxiliary(tracer: SeamTracer, output: Path, jax: ModuleType) -> list[dict[str, Any]]:
    if tracer.noahmp_count != 1:
        raise GateError(f"NOAHMP_SURFACE_TRACE_COUNT:{tracer.noahmp_count}")
    auxiliary_dir = output / "auxiliary"
    auxiliary_dir.mkdir(mode=0o700, exist_ok=False)
    host = {
        "sfclay_ch": np.ascontiguousarray(np.asarray(jax.device_get(tracer.surface_sfclay_ch))),
        "noahmp_land_ch": np.ascontiguousarray(np.asarray(jax.device_get(tracer.surface_noahmp_land_ch))),
        "is_land": np.ascontiguousarray(np.asarray(jax.device_get(tracer.surface_is_land))),
    }
    if len({array.shape for array in host.values()}) != 1:
        raise GateError(f"SURFACE_CH_AUX_SHAPES:{[array.shape for array in host.values()]}")
    host["surface_ch"] = np.ascontiguousarray(
        np.where(host["is_land"], host["noahmp_land_ch"], host["sfclay_ch"])
    )
    records: list[dict[str, Any]] = []
    for name in ("sfclay_ch", "noahmp_land_ch", "is_land", "surface_ch"):
        array = host[name]
        if array.dtype.hasobject or (name != "is_land" and not np.isfinite(array).all()):
            raise GateError(f"SURFACE_CH_AUX_PAYLOAD:{name}")
        relative = Path("auxiliary") / f"{name}.npy"
        path = output / relative
        _write_npy_exclusive(path, array)
        records.append({
            "name": name,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "byte_order": array.dtype.byteorder,
            "nbytes": int(array.nbytes),
            "logical_c_bitpayload_sha256": sha256_bytes(array.tobytes(order="C")),
            "file": relative.as_posix(),
            "file_sha256": sha256_file(path),
            "file_bytes": path.stat().st_size,
        })
    _fsync_directory(auxiliary_dir)
    return records


def validate_snapshot_manifest(
    manifest_path: Path,
    *,
    expected_nonce: str,
    expected_slot_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not manifest_path.is_absolute() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise GateError(f"MANIFEST_PATH:{manifest_path}")
    root = manifest_path.parent.resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema") != "wrfgpu2-v0234-authentic-pbl-entry-snapshot-v1":
        raise GateError("MANIFEST_SCHEMA")
    if value.get("nonce") != expected_nonce:
        raise GateError("MANIFEST_NONCE")
    if canonical_without_self(value) != value.get("canonical_payload_sha256"):
        raise GateError("MANIFEST_CANONICAL")
    leaves = value.get("state_leaves")
    if not isinstance(leaves, list) or not leaves:
        raise GateError("MANIFEST_LEAVES")
    slots = [item.get("slot_name") for item in leaves if isinstance(item, dict)]
    if len(slots) != len(leaves) or len(set(slots)) != len(slots):
        raise GateError("MANIFEST_DUPLICATE_SLOT")
    authoritative = (
        tuple(state_schema_authority()["slot_names"])
        if expected_slot_names is None
        else expected_slot_names
    )
    if tuple(slots) != authoritative:
        raise GateError("MANIFEST_SLOT_SCHEMA_DRIFT")
    arrays = 0
    none = 0
    file_records: list[dict[str, Any]] = []
    for expected_index, item in enumerate(leaves):
        if item.get("slot_index") != expected_index:
            raise GateError("MANIFEST_SLOT_ORDER")
        if item.get("kind") == "none":
            none += 1
            if set(item) != {"slot_name", "kind", "slot_index"}:
                raise GateError("MANIFEST_NONE_SCHEMA")
            continue
        if item.get("kind") != "array":
            raise GateError("MANIFEST_FIELD_KIND")
        required = {
            "slot_name", "kind", "slot_index", "shape", "dtype", "byte_order", "nbytes",
            "logical_c_bitpayload_sha256", "file", "file_sha256", "file_bytes",
        }
        if set(item) != required:
            raise GateError(f"MANIFEST_ARRAY_SCHEMA:{item.get('slot_name')}")
        relative = Path(item["file"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("arrays",):
            raise GateError("MANIFEST_FILE_ESCAPE")
        path = root / relative
        if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
            raise GateError("MANIFEST_ARRAY_PATH")
        if sha256_file(path) != item["file_sha256"] or path.stat().st_size != item["file_bytes"]:
            raise GateError(f"MANIFEST_FILE_HASH:{item['slot_name']}")
        array = np.load(path, allow_pickle=False)
        if array.dtype.hasobject:
            raise GateError("MANIFEST_OBJECT_DTYPE")
        contiguous = np.ascontiguousarray(array)
        if (
            list(contiguous.shape) != item["shape"]
            or contiguous.dtype.str != item["dtype"]
            or contiguous.dtype.byteorder != item["byte_order"]
            or contiguous.nbytes != item["nbytes"]
            or sha256_bytes(contiguous.tobytes(order="C")) != item["logical_c_bitpayload_sha256"]
        ):
            raise GateError(f"MANIFEST_ARRAY_DRIFT:{item['slot_name']}")
        arrays += 1
        file_records.append({"file": item["file"], "sha256": item["file_sha256"]})
    auxiliary = value.get("capture_auxiliary")
    if not isinstance(auxiliary, list) or [item.get("name") for item in auxiliary] != [
        "sfclay_ch", "noahmp_land_ch", "is_land", "surface_ch"
    ]:
        raise GateError("MANIFEST_AUXILIARY_INVENTORY")
    required_aux = {
        "name", "shape", "dtype", "byte_order", "nbytes",
        "logical_c_bitpayload_sha256", "file", "file_sha256", "file_bytes",
    }
    aux_arrays: dict[str, np.ndarray] = {}
    for item in auxiliary:
        if set(item) != required_aux:
            raise GateError(f"MANIFEST_AUXILIARY_SCHEMA:{item.get('name')}")
        relative = Path(item["file"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("auxiliary",):
            raise GateError("MANIFEST_AUXILIARY_ESCAPE")
        path = root / relative
        if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
            raise GateError("MANIFEST_AUXILIARY_PATH")
        array = np.ascontiguousarray(np.load(path, allow_pickle=False))
        if (
            sha256_file(path) != item["file_sha256"]
            or path.stat().st_size != item["file_bytes"]
            or list(array.shape) != item["shape"]
            or array.dtype.str != item["dtype"]
            or array.dtype.byteorder != item["byte_order"]
            or array.nbytes != item["nbytes"]
            or sha256_bytes(array.tobytes(order="C")) != item["logical_c_bitpayload_sha256"]
        ):
            raise GateError(f"MANIFEST_AUXILIARY_DRIFT:{item['name']}")
        aux_arrays[item["name"]] = array
        file_records.append({"file": item["file"], "sha256": item["file_sha256"]})
    rebuilt_ch = np.where(
        aux_arrays["is_land"], aux_arrays["noahmp_land_ch"], aux_arrays["sfclay_ch"]
    )
    if not np.array_equal(rebuilt_ch, aux_arrays["surface_ch"]):
        raise GateError("MANIFEST_AUXILIARY_CH_RECONSTRUCTION")
    inventory = value.get("state_inventory", {})
    pytree = inventory.get("pytree_leaves")
    array_leaves = [item for item in leaves if item["kind"] == "array"]
    if not isinstance(pytree, list) or len(pytree) != len(array_leaves):
        raise GateError("MANIFEST_PYTREE_COUNT")
    for record, leaf in zip(pytree, array_leaves):
        required_pytree = {
            "slot_index", "slot_name", "path", "shape", "dtype", "nbytes",
            "logical_c_bitpayload_sha256",
        }
        if (
            not isinstance(record, dict)
            or set(record) != required_pytree
            or record["slot_index"] != leaf["slot_index"]
            or record["slot_name"] != leaf["slot_name"]
            or record["shape"] != leaf["shape"]
            or record["dtype"] != leaf["dtype"]
            or record["nbytes"] != leaf["nbytes"]
            or record["logical_c_bitpayload_sha256"] != leaf["logical_c_bitpayload_sha256"]
            or not isinstance(record["path"], str)
            or not record["path"]
        ):
            raise GateError("MANIFEST_PYTREE_DRIFT")
    if (
        inventory.get("state_slot_count") != len(leaves)
        or inventory.get("array_slot_count") != arrays
        or inventory.get("none_slot_count") != none
        or inventory.get("slot_names") != slots
        or inventory.get("slot_names_sha256") != canonical_sha256(slots)
        or inventory.get("direct_tree_child_count") != len(leaves)
        or inventory.get("direct_tree_aux_is_none") is not True
        or inventory.get("pytree_leaf_count") != arrays
        or inventory.get("pytree_inventory_sha256") != canonical_sha256(pytree)
        or inventory.get("leaf_inventory_sha256") != canonical_sha256(leaves)
    ):
        raise GateError("MANIFEST_INVENTORY")
    return {
        "manifest": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "canonical_payload_sha256": value["canonical_payload_sha256"],
        "slot_count": len(leaves),
        "array_count": arrays,
        "none_count": none,
        "payload_file_inventory_sha256": canonical_sha256(file_records),
        "auxiliary_count": len(auxiliary),
        "valid": True,
    }


def _signal_handlers() -> tuple[dict[int, Any], dict[str, Any]]:
    state: dict[str, Any] = {"signal": None, "received_utc": None}

    def handle(signum: int, _frame: Any) -> None:
        state["signal"] = int(signum)
        state["received_utc"] = utc_now()
        raise CapturePreempted(f"signal:{signum}")

    old = {sig: signal.signal(sig, handle) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    return old, state


def _restore_handlers(old: Mapping[int, Any]) -> None:
    for sig, handler in old.items():
        signal.signal(sig, handler)


def _exclusive_output_dir() -> None:
    if OUTPUT_DIR.exists() or OUTPUT_DIR.is_symlink():
        raise GateError(f"OUTPUT_NOT_FRESH:{OUTPUT_DIR}")
    os.mkdir(OUTPUT_DIR, mode=0o700)
    _fsync_directory(OUTPUT_DIR.parent)


def _normal_load_and_capture(
    journal: RuntimeJournal,
    lock: dict[str, Any],
    approved_head: str,
) -> dict[str, Any]:
    runtime_env = validate_runtime_environment()
    assert_preemption_clear("before-model-import")
    sys.path.insert(0, str(MODEL_ROOT / "src"))
    import jax  # type: ignore[import-not-found]
    import jax.numpy as jnp  # type: ignore[import-not-found]
    import gpuwrf  # type: ignore[import-not-found]
    from gpuwrf.integration import nested_pipeline  # type: ignore[import-not-found]
    from gpuwrf.physics import noahmp_coupler  # type: ignore[import-not-found]
    from gpuwrf.runtime import operational_mode  # type: ignore[import-not-found]

    if MODEL_ROOT not in Path(gpuwrf.__file__).resolve().parents:
        raise GateError(f"MODEL_IMPORT_ROOT:{gpuwrf.__file__}")
    journal.add("model-imported", model_file=str(Path(gpuwrf.__file__).resolve()))
    assert_preemption_clear("before-config-load")

    config = nested_pipeline.NestedPipelineConfig(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR / "unused-output",
        proof_dir=OUTPUT_DIR / "unused-proof",
        hours=0,
        max_dom=3,
        feedback=False,
    )
    original_commit = nested_pipeline._commit_to_operational_device
    nested_pipeline._commit_to_operational_device = lambda value: value
    try:
        hierarchy, bundles, load_meta, run_start, dt_by_domain, _cold = nested_pipeline._load_domains(
            config, nested_pipeline.domain_names_for(3)
        )
    finally:
        nested_pipeline._commit_to_operational_device = original_commit
    del hierarchy, _cold
    namelist = bundles["d03"].namelist
    route = {
        "mp_physics": int(namelist.mp_physics),
        "sf_sfclay_physics": int(namelist.sf_sfclay_physics),
        "sf_surface_physics": int(namelist.sf_surface_physics),
        "bl_pbl_physics": int(namelist.bl_pbl_physics),
        "rad_rk_tendf": int(namelist.rad_rk_tendf),
        "dt_s": float(namelist.dt_s),
        "use_noahmp": bool(namelist.use_noahmp),
        "nx": int(namelist.grid.nx),
        "ny": int(namelist.grid.ny),
        "nz": int(namelist.grid.nz),
    }
    expected_route = {
        "mp_physics": 8,
        "sf_sfclay_physics": 5,
        "sf_surface_physics": 4,
        "bl_pbl_physics": 5,
        "rad_rk_tendf": 1,
        "dt_s": 6.0,
        "use_noahmp": True,
        "nx": 111,
        "ny": 93,
        "nz": 44,
    }
    if route != expected_route:
        raise GateError(f"RUNTIME_ROUTE:{route}")
    if run_start.isoformat() != "2025-03-01T00:00:00+00:00":
        raise GateError(f"RUN_START:{run_start.isoformat()}")
    if dt_by_domain != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
        raise GateError(f"DT_HIERARCHY:{dt_by_domain}")
    journal.add("d03-config-authenticated", route=route)

    require_file(CARRY, CARRY_SHA256)
    with CARRY.open("rb") as stream:
        host_carry = pickle.load(stream)
        if stream.read(1):
            raise GateError("CARRY_TRAILING_BYTES")
    host_leaves = jax.tree_util.tree_leaves(host_carry)
    if len(host_leaves) != 106:
        raise GateError(f"CARRY_LEAF_COUNT:{len(host_leaves)}")
    carry = jax.device_put(host_carry)
    journal.add("retained-carry-device-placed", leaf_count=len(host_leaves))

    seam = seam_authority()
    target = operational_mode._physics_step_forcing
    if Path(inspect.getsourcefile(target) or "").resolve() != MODEL_SOURCE:
        raise GateError("LIVE_FUNCTION_SOURCE")
    if target.__code__.co_firstlineno != seam["function_first_line"]:
        raise GateError("LIVE_FUNCTION_FIRST_LINE")
    if Path(inspect.getsourcefile(noahmp_coupler.noahmp_surface_adapter) or "").resolve() != NOAHMP_COUPLER_SOURCE:
        raise GateError("LIVE_NOAHMP_COUPLER_SOURCE")
    require_file(NOAHMP_COUPLER_SOURCE, NOAHMP_COUPLER_SOURCE_SHA256)
    tracer = SeamTracer(
        target.__code__,
        int(seam["stop_line"]),
        noahmp_code=noahmp_coupler.noahmp_surface_adapter.__code__,
    )
    guard = PBLGuard()
    originals = {
        "mynn_adapter_with_source_leaves": operational_mode.mynn_adapter_with_source_leaves,
        "mynn_adapter": operational_mode.mynn_adapter,
        "myj_pbl_adapter": operational_mode.myj_pbl_adapter,
        "PBL_SCAN_ADAPTERS": operational_mode.PBL_SCAN_ADAPTERS,
    }
    operational_mode.mynn_adapter_with_source_leaves = guard
    operational_mode.mynn_adapter = guard
    operational_mode.myj_pbl_adapter = guard
    operational_mode.PBL_SCAN_ADAPTERS = {key: guard for key in originals["PBL_SCAN_ADAPTERS"]}
    step_index = jnp.asarray(1, dtype=jnp.int32)
    lead_seconds = step_index.astype(jnp.float64) * float(namelist.dt_s)
    run_radiation = jnp.equal(jnp.mod(step_index, int(namelist.radiation_cadence_steps)), 0)
    first_timestep = jnp.equal(step_index, 1)
    clock_base = operational_mode.build_clock_base(namelist)
    assert_preemption_clear("before-prefix")
    journal.add("prefix-started", domain="d03", step=1)
    sys.settrace(tracer)
    stopped = False
    try:
        target(
            carry,
            namelist,
            lead_seconds,
            run_radiation=run_radiation,
            first_timestep=first_timestep,
            clock_base=clock_base,
        )
    except CaptureStop:
        stopped = True
    finally:
        sys.settrace(None)
        operational_mode.mynn_adapter_with_source_leaves = originals["mynn_adapter_with_source_leaves"]
        operational_mode.mynn_adapter = originals["mynn_adapter"]
        operational_mode.myj_pbl_adapter = originals["myj_pbl_adapter"]
        operational_mode.PBL_SCAN_ADAPTERS = originals["PBL_SCAN_ADAPTERS"]
    if not stopped or tracer.count != 1 or tracer.state is None:
        raise GateError(f"SEAM_CAPTURE_COUNT:{stopped}:{tracer.count}")
    if guard.count != 0:
        raise GateError(f"PBL_GUARD_INVOKED:{guard.count}")
    if tracer.noahmp_count != 1:
        raise GateError(f"NOAHMP_SURFACE_TRACE_COUNT:{tracer.noahmp_count}")
    if tracer.frame_evidence.get("bl_opt") != 5 or tracer.frame_evidence.get("source_leaf_mode") is not True:
        raise GateError(f"SEAM_FRAME_ROUTE:{tracer.frame_evidence}")
    assert_preemption_clear("at-capture-seam")
    journal.add("seam-captured", frame=tracer.frame_evidence, pbl_guard_calls=guard.count)

    leaves, inventory = materialize_state(tracer.state, OUTPUT_DIR, jax)
    auxiliary = materialize_surface_ch_auxiliary(tracer, OUTPUT_DIR, jax)
    assert_preemption_clear("after-state-materialization")
    journal.add("state-materialized", inventory=inventory)
    graph = authority_graph()
    manifest: dict[str, Any] = {
        "schema": "wrfgpu2-v0234-authentic-pbl-entry-snapshot-v1",
        "nonce": NONCE,
        "status": "CAPTURE_COMPLETE_HOLD_5_OF_28",
        "tuple": {
            "retained_carry_sha256": CARRY_SHA256,
            "producer_commit": approved_head,
            "producer_tree": git_text(REPO, "rev-parse", f"{approved_head}^{{tree}}"),
            "producer_source_sha256": MODEL_SOURCE_SHA256,
            "producer_state_source_sha256": MODEL_STATE_SOURCE_SHA256,
            "producer_noahmp_coupler_source_sha256": NOAHMP_COUPLER_SOURCE_SHA256,
            "namelist_sha256": NAMELIST_SHA256,
            "wrfinput_d03_sha256": WRFINPUT_SHA256,
            "cycle": "20250228_18z",
            "domain": "d03",
            "outer_step": 1,
            "dt_s": 6.0,
            "route": route,
        },
        "seam": {
            "after": "microphysics_and_surface_land",
            "assignment": "pbl_entry_state = next_state",
            "before": "mynn_adapter_with_source_leaves",
            "source_line": seam["assignment_line"],
            "stop_line": seam["stop_line"],
            "function_ast_sha256": seam["function_ast_sha256"],
            "trace_invocation_count": tracer.count,
            "pbl_guard_invocation_count": guard.count,
            "noahmp_surface_trace_invocation_count": tracer.noahmp_count,
            "frame": tracer.frame_evidence,
        },
        "provenance": {
            "contract_commit": CONTRACT_COMMIT,
            "coordination_commit": COORDINATION_COMMIT,
            "authority_receipt_graph_sha256": graph["receipt_graph"]["sha256"],
            "authority_receipt_graph_canonical_sha256": graph["receipt_graph_canonical_sha256"],
            "source_instrumented_seam": True,
            "wrf_dump_edge": False,
            "comparator_edge": False,
            "synthetic_reconstruction": False,
            "one_authorized_arm": True,
        },
        "lock_v2": lock,
        "runtime_environment": runtime_env,
        "load_metadata_sha256": canonical_sha256(load_meta),
        "state_inventory": inventory,
        "state_leaves": leaves,
        "capture_auxiliary": auxiliary,
        "reference_status": "HOLD_5_OF_28",
        "comparator_dispatch_ready": False,
        "captured_utc": utc_now(),
    }
    manifest["canonical_payload_sha256"] = canonical_without_self(manifest)
    candidate = OUTPUT_DIR / ".candidate-manifest.json"
    atomic_json(candidate, manifest, replace=False)
    validation = validate_snapshot_manifest(candidate, expected_nonce=NONCE)
    assert_preemption_clear("after-inert-reread")
    final = OUTPUT_DIR / "manifest.json"
    os.link(candidate, final)
    candidate.unlink()
    _fsync_directory(OUTPUT_DIR)
    validation["manifest"] = str(final)
    validation["manifest_file_sha256"] = sha256_file(final)
    atomic_json(OUTPUT_DIR / "validation.json", validation, replace=False)
    journal.add("manifest-sealed", validation=validation)
    return {
        "status": "CAPTURE_COMPLETE_HOLD_5_OF_28",
        "manifest": str(final),
        "manifest_file_sha256": sha256_file(final),
        "manifest_canonical_sha256": manifest["canonical_payload_sha256"],
        "state_inventory": inventory,
        "pbl_guard_invocation_count": guard.count,
        "seam_trace_invocation_count": tracer.count,
        "gpu_query_performed": False,
        "wrf_mpi_comparator_executed": False,
    }


def execute_payload(approved_head: str) -> int:
    output_created = False
    journal: RuntimeJournal | None = None
    watcher: SentinelWatcher | None = None
    old_handlers, signal_state = _signal_handlers()
    try:
        assert_no_backend_imported()
        pre = static_preflight(approved_head, require_output_absent=True)
        lock = assert_live_lock()
        assert_preemption_clear("inside-lease-before-output")
        _exclusive_output_dir()
        output_created = True
        consumption = {
            "schema": "wrfgpu2-v0234-capture-nonce-consumption-v1",
            "nonce": NONCE,
            "contract_commit": CONTRACT_COMMIT,
            "approved_head": approved_head,
            "consumed": True,
            "consumed_utc": utc_now(),
            "reason": "authorized-payload-started",
        }
        consumption["canonical_payload_sha256"] = canonical_without_self(consumption)
        atomic_json(OUTPUT_DIR / "nonce-consumed.json", consumption, replace=False)
        journal = RuntimeJournal(OUTPUT_DIR)
        journal.add("payload-admitted", lock=lock, preflight_sha256=canonical_sha256(pre))
        watcher = SentinelWatcher()
        watcher.start()
        result = _normal_load_and_capture(journal, lock, approved_head)
        status = {
            "schema": "wrfgpu2-v0234-pbl-entry-capture-status-v1",
            "nonce": NONCE,
            "terminal": True,
            "result": result,
            "preemption_signal": signal_state,
            "watcher_observed": watcher.observed,
            "finished_utc": utc_now(),
        }
        status["canonical_payload_sha256"] = canonical_without_self(status)
        atomic_json(OUTPUT_DIR / "run-status.json", status, replace=False)
        return 0
    except CapturePreempted as exc:
        if output_created:
            failure = {
                "schema": "wrfgpu2-v0234-pbl-entry-capture-status-v1",
                "nonce": NONCE,
                "terminal": True,
                "status": "CAPTURE_PREEMPTED_NONCE_CONSUMED",
                "error": str(exc),
                "preemption_signal": signal_state,
                "watcher_observed": [] if watcher is None else watcher.observed,
                "finished_utc": utc_now(),
            }
            failure["canonical_payload_sha256"] = canonical_without_self(failure)
            atomic_json(OUTPUT_DIR / "run-status.json", failure)
        return 130
    except Exception as exc:
        if output_created:
            failure = {
                "schema": "wrfgpu2-v0234-pbl-entry-capture-status-v1",
                "nonce": NONCE,
                "terminal": True,
                "status": "CAPTURE_FAILED_NONCE_CONSUMED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "preemption_signal": signal_state,
                "watcher_observed": [] if watcher is None else watcher.observed,
                "finished_utc": utc_now(),
            }
            failure["canonical_payload_sha256"] = canonical_without_self(failure)
            try:
                atomic_json(OUTPUT_DIR / "run-status.json", failure)
            except Exception:
                pass
        print(f"CAPTURE_REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 75
    finally:
        if watcher is not None:
            watcher.stop()
        _restore_handlers(old_handlers)


def build_lock_command(approved_head: str) -> list[str]:
    return [
        str(LOCK_WRAPPER),
        "--timeout", "0",
        "--label", LOCK_LABEL,
        "--intent", "production-preemptible",
        "--",
        "/usr/bin/taskset", "-c", "13,14,15,29,30,31",
        "/usr/bin/nice", "-n", "15",
        "/usr/bin/ionice", "-c", "3",
        str(PYTHON_EXECUTABLE), str(Path(__file__).resolve()),
        "--execute", "--approved-head", approved_head,
    ]


def minimal_launch_environment() -> dict[str, str]:
    environment = {
        "HOME": os.environ.get("HOME", "<USER_HOME>"),
        "USER": os.environ.get("USER", "user"),
        "LOGNAME": os.environ.get("LOGNAME", os.environ.get("USER", "user")),
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "PYTHONPATH": os.pathsep.join((str(MODEL_ROOT / "src"), str(REPO))),
        **THREAD_ENV,
        **REQUIRED_RUNTIME_ENV,
    }
    for optional in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
        if os.environ.get(optional):
            environment[optional] = os.environ[optional]
    return environment


def _release_evidence(stderr: str, returncode: int) -> dict[str, Any]:
    expected = f"[with_gpu_lock] {LOCK_LABEL} released GPU lock (rc={returncode})"
    lines = stderr.splitlines()
    matches = [line for line in lines if line == expected]
    acquired = f"[with_gpu_lock] {LOCK_LABEL} ACQUIRED GPU lock" in lines
    if len(matches) != 1:
        raise GateError(f"TERMINAL_LOCK_RELEASE_LINE:{matches}:{expected}")
    holder_snapshot: dict[str, Any] | None = None
    if LOCK_HOLDER.exists() and not LOCK_HOLDER.is_symlink():
        try:
            holder_snapshot = json.loads(LOCK_HOLDER.read_text(encoding="utf-8"))
        except Exception as exc:
            holder_snapshot = {"unreadable": str(exc)}
    if isinstance(holder_snapshot, dict) and holder_snapshot.get("label") == LOCK_LABEL:
        raise GateError("OWN_LOCK_HOLDER_REMAINS")
    return {
        "acquired": acquired,
        "release_line": expected,
        "release_line_count": len(matches),
        "own_holder_absent": not (
            isinstance(holder_snapshot, dict) and holder_snapshot.get("label") == LOCK_LABEL
        ),
        "post_exit_holder": holder_snapshot,
    }


def launch_once(approved_head: str, preflight_path: Path, preflight_sha256: str, receipt: Path) -> int:
    if receipt.exists() or receipt.is_symlink():
        raise GateError("LAUNCH_RECEIPT_ALREADY_EXISTS_NONCE_CONSUMED")
    started = utc_now()
    command = build_lock_command(approved_head)
    try:
        pre = static_preflight(approved_head, require_output_absent=True)
        if require_file(preflight_path, preflight_sha256)["sha256"] != preflight_sha256:
            raise GateError("PREFLIGHT_FILE")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        preflight_head = str(preflight.get("approved_head", ""))
        ancestor = subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor", preflight_head, approved_head],
            check=False,
        )
        if preflight.get("passed") is not True or ancestor.returncode != 0:
            raise GateError("PREFLIGHT_AUTHORITY")
        assert_preemption_clear("launcher-final")
    except (CapturePreempted, GateError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        status = (
            "CAPTURE_PREEMPTED_NONCE_CONSUMED"
            if isinstance(exc, CapturePreempted)
            else "CAPTURE_REFUSED_NONCE_CONSUMED"
        )
        refused: dict[str, Any] = {
            "schema": "wrfgpu2-v0234-single-capture-execution-receipt-v1",
            "nonce": NONCE,
            "nonce_consumed": True,
            "approved_head": approved_head,
            "preflight_file_sha256": preflight_sha256,
            "lock_command": command,
            "lock_invocation_count": 0,
            "started_utc": started,
            "finished_utc": utc_now(),
            "returncode": 130 if isinstance(exc, CapturePreempted) else 75,
            "terminal_status": status,
            "prelaunch_error_type": type(exc).__name__,
            "prelaunch_error": str(exc),
            "lock_release": {"not_acquired": True, "terminal": True},
            "retry_performed": False,
            "alternate_arm_performed": False,
            "gpu_query_performed": False,
        }
        refused["canonical_payload_sha256"] = canonical_without_self(refused)
        atomic_json(receipt, refused, replace=False)
        return int(refused["returncode"])

    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=minimal_launch_environment(),
        )
    except OSError as exc:
        failure: dict[str, Any] = {
            "schema": "wrfgpu2-v0234-single-capture-execution-receipt-v1",
            "nonce": NONCE,
            "nonce_consumed": True,
            "approved_head": approved_head,
            "preflight_file_sha256": preflight_sha256,
            "lock_command": command,
            "lock_invocation_count": 1,
            "started_utc": started,
            "finished_utc": utc_now(),
            "returncode": 75,
            "terminal_status": "CAPTURE_REFUSED_NONCE_CONSUMED",
            "launch_error_type": type(exc).__name__,
            "launch_error": str(exc),
            "lock_release": {"verified": False, "wrapper_did_not_start": True},
            "retry_performed": False,
            "alternate_arm_performed": False,
            "gpu_query_performed": False,
        }
        failure["canonical_payload_sha256"] = canonical_without_self(failure)
        atomic_json(receipt, failure, replace=False)
        return 75

    release_verified = True
    try:
        release = _release_evidence(result.stderr, result.returncode)
    except GateError as exc:
        release_verified = False
        release = {"verified": False, "error": str(exc)}
    output_status = None
    status_path = OUTPUT_DIR / "run-status.json"
    if status_path.is_file() and not status_path.is_symlink():
        output_status = {
            "path": str(status_path),
            "sha256": sha256_file(status_path),
            "value": json.loads(status_path.read_text(encoding="utf-8")),
        }
    receipt_value: dict[str, Any] = {
        "schema": "wrfgpu2-v0234-single-capture-execution-receipt-v1",
        "nonce": NONCE,
        "nonce_consumed": True,
        "approved_head": approved_head,
        "preflight_file_sha256": preflight_sha256,
        "static_preflight_revalidated_sha256": canonical_sha256(pre),
        "lock_command": command,
        "lock_invocation_count": 1,
        "started_utc": started,
        "finished_utc": utc_now(),
        "returncode": result.returncode if release_verified else 75,
        "payload_or_wrapper_returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "lock_release": release,
        "output_status": output_status,
        "retry_performed": False,
        "alternate_arm_performed": False,
        "gpu_query_performed": False,
    }
    receipt_value["canonical_payload_sha256"] = canonical_without_self(receipt_value)
    atomic_json(receipt, receipt_value, replace=False)
    return int(receipt_value["returncode"])


def tree_inventory(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise GateError(f"OUTPUT_SYMLINK:{path}")
        if path.is_file():
            records.append({
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return {"file_count": len(records), "files": records, "tree_sha256": canonical_sha256(records)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--validate-output", action="store_true")
    value.add_argument("--approved-head")
    value.add_argument("--proof", type=Path)
    value.add_argument("--preflight-path", type=Path)
    value.add_argument("--preflight-sha256")
    value.add_argument("--receipt", type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.preflight:
            if not args.approved_head or args.proof is None:
                raise GateError("PREFLIGHT_ARGS")
            result = static_preflight(args.approved_head)
            result["canonical_payload_sha256"] = canonical_without_self(result)
            atomic_json(args.proof.resolve(), result, replace=False)
            print(json.dumps({"verdict": "PREFLIGHT_PASS", "canonical_payload_sha256": result["canonical_payload_sha256"]}, sort_keys=True))
            return 0
        if args.execute:
            if not args.approved_head:
                raise GateError("EXECUTE_ARGS")
            return execute_payload(args.approved_head)
        if args.launch:
            if not all((args.approved_head, args.preflight_path, args.preflight_sha256, args.receipt)):
                raise GateError("LAUNCH_ARGS")
            return launch_once(
                args.approved_head,
                args.preflight_path.resolve(),
                args.preflight_sha256,
                args.receipt.resolve(),
            )
        if args.proof is None:
            raise GateError("VALIDATE_ARGS")
        validation = validate_snapshot_manifest(
            OUTPUT_DIR / "manifest.json", expected_nonce=NONCE
        )
        validation["output_tree"] = tree_inventory(OUTPUT_DIR)
        validation["validated_utc"] = utc_now()
        validation["canonical_payload_sha256"] = canonical_without_self(validation)
        atomic_json(args.proof.resolve(), validation, replace=False)
        print(json.dumps({"verdict": "OUTPUT_VALID", "canonical_payload_sha256": validation["canonical_payload_sha256"]}, sort_keys=True))
        return 0
    except CapturePreempted as exc:
        print(f"PREEMPT:{exc}", file=sys.stderr)
        return 130
    except (GateError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
