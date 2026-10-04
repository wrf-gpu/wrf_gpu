#!/usr/bin/env python3
"""Prove the source-authorized v0234 fresh-start MYNN QKE lifecycle fix.

The proof reuses the sealed pre-PBL state and invokes only the lifecycle/QKE
initializer.  It does not invoke a production adapter, a MYNN timestep, GPU,
WRF, or MPI.  The pre-fix side is the one-shot capture made at the parent
commit; the post-fix side is the current committed lifecycle function on the
identical state.  Pristine WRF initialized QKE is the target.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-single-authority-attribution"
CAPTURE_INPUT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084/"
    "capture/authentic-ac6712170cbe5084"
)
CAPTURE_INPUT_MANIFEST = CAPTURE_INPUT / "manifest.json"
CAPTURE_ARCHIVE = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_gpt_single_authority_attribution/"
    "cpu-production-adapter-authority-v1/single-authority-capture.npz"
)
CAPTURE_RUN_MANIFEST = CAPTURE_ARCHIVE.with_name("manifest.json")
GRID_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_0b18530a1dc9cac2/"
    "capture-inputs"
)
NAMELIST = GRID_ROOT / "namelist.input"
RECEIPT = GRID_ROOT.parent / "control/capture-input-manifest.json"
WRF_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_mynn_sp2_d03_horizon55_gpt_fresh01/"
    "evidence-dumps-fresh-d03-runtime"
)
WRF_SOURCE = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_0b18530a1dc9cac2/"
    "source/instrumented/phys/MYNN-EDMF/module_bl_mynnedmf.F90"
)
AUDIT_SCRIPT = REPO / "scripts/v0234_gpt_adversarial_audit.py"
COUPLER_SOURCE = REPO / "src/gpuwrf/coupling/physics_couplers.py"
FOCUSED_TEST = REPO / "tests/test_v0234_mynn_qke_lifecycle.py"
LEGACY_TEST = REPO / "tests/test_v014_mynn_coldstart_init.py"
JUNIT = SPRINT / "qke-lifecycle-pytest.xml"

EXPECTED_INPUT_MANIFEST = (
    "295760b17cba2fde8e931188caab76fbf0564b7e9229784e8f739270f75309e6"
)
EXPECTED_CAPTURE_ARCHIVE = (
    "bc576333ba54d26db7272c8692b327b56f477e6183a6c02d596bf0e6685cd2fe"
)
EXPECTED_CAPTURE_RUN_MANIFEST = (
    "7835e8fecce89fb84d1dafaf316cb8ea0f448acb241bae789095886406b23e7a"
)
EXPECTED_NAMELIST = (
    "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838"
)
EXPECTED_RECEIPT = (
    "be3ee7df849fed756a9b6c0432b8c7ace7ccf0c411fb8630f4252bcecf7c2f40"
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


class FixProofFailure(RuntimeError):
    """Fail-closed fix proof failure."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        raise FixProofFailure(f"OUTPUT_NOT_FRESH:{path}")
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


def finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject or not np.isfinite(array).all():
        raise FixProofFailure(f"NONFINITE_OR_OBJECT:{name}")
    return array


def rms(value: np.ndarray) -> float:
    array = finite("rms", np.asarray(value, dtype=np.float64))
    return float(np.sqrt(np.mean(array * array)))


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    lhs = finite("left", np.asarray(left))
    rhs = finite("right", np.asarray(right))
    if lhs.shape != rhs.shape:
        raise FixProofFailure(f"SHAPE:{lhs.shape}:{rhs.shape}")
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    reference_rms = rms(rhs)
    return {
        "rms": rms(delta),
        "max_abs": float(np.max(np.abs(delta))),
        "sse": float(np.sum(delta * delta)),
        "exact_fraction": float(np.mean(lhs == rhs)),
        "reference_rms": reference_rms,
        "relative_rms": rms(delta) / reference_rms if reference_rms else None,
    }


def resource_gate() -> dict[str, Any]:
    affinity = set(os.sched_getaffinity(0))
    if affinity != ALLOWED_CPUS:
        raise FixProofFailure(f"CPUSET:{sorted(affinity)}")
    observed_threads = {key: os.environ.get(key) for key in THREAD_ENV}
    if observed_threads != THREAD_ENV:
        raise FixProofFailure(f"THREAD_ENV:{observed_threads}")
    backend_env = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
        "JAX_ENABLE_COMPILATION_CACHE": os.environ.get(
            "JAX_ENABLE_COMPILATION_CACHE"
        ),
    }
    if backend_env != {
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
    }:
        raise FixProofFailure(f"BACKEND_ENV:{backend_env}")
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    ioprio = ctypes.CDLL(None, use_errno=True).syscall(252, 1, 0)
    if nice < 15 or ioprio < 0 or int(ioprio) >> 13 != 3:
        raise FixProofFailure(f"PRIORITY:{nice}:{ioprio}")
    if Path("/tmp/PREEMPT_CPU").exists() or Path("/tmp/PREEMPT_CPU").is_symlink():
        raise FixProofFailure("PREEMPT_CPU")
    return {
        "logical_cpu_ids": sorted(affinity),
        "thread_environment": observed_threads,
        "backend_environment": backend_env,
        "nice": nice,
        "ionice_class": int(ioprio) >> 13,
        "preempt_cpu_absent": True,
    }


def read_canonical(path: Path, expected_sha: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FixProofFailure(f"NOT_REGULAR:{path}")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise FixProofFailure(f"FILE_HASH:{path}:{actual}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if canonical_without_self(value) != value.get("canonical_payload_sha256"):
        raise FixProofFailure(f"CANONICAL:{path}")
    return value


def validate_junit() -> dict[str, Any]:
    if JUNIT.is_symlink() or not JUNIT.is_file():
        raise FixProofFailure("JUNIT_MISSING")
    root = ET.parse(JUNIT).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        key: sum(int(float(suite.attrib.get(key, "0"))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if totals["tests"] < 9 or totals["failures"] or totals["errors"]:
        raise FixProofFailure(f"JUNIT_RED:{totals}")
    return {"path": str(JUNIT), "sha256": sha256_file(JUNIT), **totals}


def validate_static() -> tuple[dict[str, Any], list[np.ndarray | None], dict[str, Any]]:
    if any(name == "jax" or name.startswith(("jax.", "gpuwrf.")) for name in sys.modules):
        raise FixProofFailure("BACKEND_IMPORTED_BEFORE_PREFLIGHT")
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"], check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    if status:
        raise FixProofFailure(f"WORKTREE:{status}")

    input_manifest = read_canonical(CAPTURE_INPUT_MANIFEST, EXPECTED_INPUT_MANIFEST)
    capture_manifest = read_canonical(
        CAPTURE_RUN_MANIFEST, EXPECTED_CAPTURE_RUN_MANIFEST
    )
    receipt = read_canonical(RECEIPT, EXPECTED_RECEIPT)
    if not receipt.get("complete") or receipt.get("schema") != (
        "wrfgpu2-v0234-capture-input-closure-v1"
    ):
        raise FixProofFailure("RECEIPT_STATUS")
    receipt_namelist = receipt.get("inputs", {}).get("namelist.input", {})
    copy = receipt_namelist.get("copy", {})
    origin = receipt_namelist.get("origin", {})
    if not (
        copy.get("path") == str(NAMELIST)
        and copy.get("sha256") == EXPECTED_NAMELIST
        and origin.get("sha256") == EXPECTED_NAMELIST
        and sha256_file(NAMELIST) == EXPECTED_NAMELIST
    ):
        raise FixProofFailure("NAMELIST_RECEIPT")
    restart_lines = re.findall(
        r"(?im)^\s*restart\s*=\s*([^,\n]+)",
        NAMELIST.read_text(encoding="utf-8"),
    )
    if [value.strip().lower() for value in restart_lines] != [".false."]:
        raise FixProofFailure(f"RESTART_AUTHORITY:{restart_lines}")
    if not (
        input_manifest.get("seam", {}).get("trace_invocation_count") == 1
        and input_manifest.get("seam", {}).get("pbl_guard_invocation_count") == 0
        and input_manifest.get("provenance", {}).get("synthetic_reconstruction") is False
        and capture_manifest.get("single_adapter_invocation") is True
        and capture_manifest.get("passed") is True
    ):
        raise FixProofFailure("CAPTURE_AUTHORITY")
    if sha256_file(CAPTURE_ARCHIVE) != EXPECTED_CAPTURE_ARCHIVE:
        raise FixProofFailure("CAPTURE_ARCHIVE_HASH")

    state_values: list[np.ndarray | None] = []
    for item in input_manifest.get("state_leaves", []):
        if item.get("slot_index") != len(state_values):
            raise FixProofFailure("STATE_SLOT_ORDER")
        if item.get("kind") == "none":
            state_values.append(None)
            continue
        path = CAPTURE_INPUT / item["file"]
        if path.is_symlink() or sha256_file(path) != item["file_sha256"]:
            raise FixProofFailure(f"STATE_FILE:{path}")
        value = finite(item["slot_name"], np.load(path, allow_pickle=False))
        observed = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "nbytes": value.nbytes,
            "logical_c_bitpayload_sha256": array_sha(value),
        }
        if observed != {key: item[key] for key in observed}:
            raise FixProofFailure(f"STATE_ARRAY:{item['slot_name']}")
        state_values.append(value)
    if len(state_values) != 71:
        raise FixProofFailure(f"STATE_COUNT:{len(state_values)}")

    sources = {}
    for name, path in {
        "proof_script": Path(__file__).resolve(),
        "wrf_mynn": WRF_SOURCE,
        "port_coupler": COUPLER_SOURCE,
        "focused_test": FOCUSED_TEST,
        "legacy_coldstart_test": LEGACY_TEST,
        "wrf_reader": AUDIT_SCRIPT,
    }.items():
        if path.is_symlink() or not path.is_file():
            raise FixProofFailure(f"SOURCE:{path}")
        sources[name] = {"path": str(path), "sha256": sha256_file(path)}
    wrf_text = WRF_SOURCE.read_text(encoding="utf-8")
    port_text = COUPLER_SOURCE.read_text(encoding="utf-8")
    for needle in (
        "IF (initflag > 0 .and. .not.restart) THEN",
        "ELSE ! not cycling or restarting:",
        "INITIALIZE_QKE = .TRUE.",
        "qke1        =zero",
        "CALL mym_initialize",
    ):
        if needle not in wrf_text:
            raise FixProofFailure(f"WRF_SOURCE_DRIFT:{needle}")
    for needle in (
        "restart: bool = False",
        "if restart:",
        "qke_seed = seed(None)",
        "mynn_coldstart_qke_from_state(state, grid)",
    ):
        if needle not in port_text:
            raise FixProofFailure(f"PORT_SOURCE_DRIFT:{needle}")
    old_coupler = capture_manifest.get("imported_gpuwrf_sources", {}).get(
        "gpuwrf.coupling.physics_couplers", {}
    )
    if old_coupler.get("sha256") == sources["port_coupler"]["sha256"]:
        raise FixProofFailure("BEFORE_AFTER_SOURCE_NOT_DISTINCT")

    return input_manifest, state_values, {
        "git_head": head,
        "resource_gate": resource_gate(),
        "namelist_receipt": {
            "path": str(RECEIPT),
            "sha256": EXPECTED_RECEIPT,
            "canonical_payload_sha256": receipt["canonical_payload_sha256"],
            "namelist_path": str(NAMELIST),
            "namelist_sha256": EXPECTED_NAMELIST,
            "origin_namelist_sha256": origin["sha256"],
            "restart_values": [".false."],
            "complete": True,
        },
        "capture": {
            "input_manifest_sha256": EXPECTED_INPUT_MANIFEST,
            "run_manifest_sha256": EXPECTED_CAPTURE_RUN_MANIFEST,
            "archive_sha256": EXPECTED_CAPTURE_ARCHIVE,
            "before_coupler_sha256": old_coupler.get("sha256"),
        },
        "sources": sources,
        "tests": validate_junit(),
    }


def load_wrf_reader() -> Any:
    spec = importlib.util.spec_from_file_location("v0234_fix_wrf_reader", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise FixProofFailure("READER_SPEC")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.WRF_DUMP_ROOT != WRF_ROOT:
        raise FixProofFailure("READER_ROOT")
    return module.IndependentWrfDump(WRF_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        print(f"REFUSE:OUTPUT_NOT_FRESH:{output}", file=sys.stderr)
        return 74
    try:
        input_manifest, host_values, evidence = validate_static()
        reader = load_wrf_reader()
        wrf_entry = reader.columns("driver_qke_entry")
        wrf_initialized = reader.columns("mix_qke_initialized")
        with np.load(CAPTURE_ARCHIVE, allow_pickle=False) as archive:
            before_entry = finite("before_entry", archive["entry_state_qke"])
            before_initialized = finite(
                "before_initialized", archive["initialized_state_qke"]
            )

        sys.path.insert(0, str(REPO / "src"))
        import jax
        from gpuwrf.contracts.state import State
        from gpuwrf.coupling import physics_couplers as couplers
        from gpuwrf.io.gen2_accessor import Gen2Run

        if jax.default_backend() != "cpu" or any(
            device.platform != "cpu" for device in jax.devices()
        ):
            raise FixProofFailure("NON_CPU_BACKEND")
        if not bool(jax.config.jax_enable_x64):
            raise FixProofFailure("JAX_X64")
        state = State.tree_unflatten(
            None,
            tuple(None if value is None else jax.device_put(value)
                  for value in host_values),
        )
        grid = Gen2Run(GRID_ROOT).grid("d03").as_grid_spec()
        original = couplers.mynn_coldstart_qke_from_state
        calls = {"value": 0}

        def counted_seed(*seed_args, **seed_kwargs):
            calls["value"] += 1
            return original(*seed_args, **seed_kwargs)

        couplers.mynn_coldstart_qke_from_state = counted_seed
        try:
            fixed_state = couplers._mynn_state_with_first_call_qke(
                state, grid, True, restart=False
            )
            jax.block_until_ready(fixed_state.qke)
            calls_after_fresh = calls["value"]
            restart_state = couplers._mynn_state_with_first_call_qke(
                state, grid, True, restart=True
            )
            jax.block_until_ready(restart_state.qke)
        finally:
            couplers.mynn_coldstart_qke_from_state = original
        if calls_after_fresh != 1 or calls["value"] != 1:
            raise FixProofFailure(f"INITIALIZER_COUNT:{calls}")
        after = finite("after", np.asarray(jax.device_get(fixed_state.qke)))
        restart_qke = finite(
            "restart_qke", np.asarray(jax.device_get(restart_state.qke))
        )

        before = metrics(before_initialized, wrf_initialized)
        after_target = metrics(after, wrf_initialized)
        improvement = before["rms"] / after_target["rms"]
        thresholds = {
            "pre_fix_entry_vs_initialized_max_abs_max": 0.0,
            "wrf_entry_vs_initialized_rms_min": 0.1,
            "pre_fix_initialized_vs_wrf_rms_min": 0.02,
            "post_fix_initialized_vs_wrf_rms_max": 0.005,
            "post_fix_improvement_factor_min": 5.0,
            "restart_preservation_max_abs_max": 0.0,
        }
        checks = {
            "pre_fix_skipped": metrics(before_entry, before_initialized)["max_abs"]
            <= thresholds["pre_fix_entry_vs_initialized_max_abs_max"],
            "wrf_initialized": metrics(wrf_entry, wrf_initialized)["rms"]
            >= thresholds["wrf_entry_vs_initialized_rms_min"],
            "pre_fix_material": before["rms"]
            >= thresholds["pre_fix_initialized_vs_wrf_rms_min"],
            "post_fix_close": after_target["rms"]
            <= thresholds["post_fix_initialized_vs_wrf_rms_max"],
            "post_fix_improves": improvement
            >= thresholds["post_fix_improvement_factor_min"],
            "restart_preserved": metrics(restart_qke, np.asarray(state.qke))[
                "max_abs"
            ] <= thresholds["restart_preservation_max_abs_max"],
        }
        if not all(checks.values()):
            raise FixProofFailure(f"FIX_GATE:{checks}")

        proof = {
            "schema": "wrfgpu2-v0234-gpt-qke-lifecycle-fix-proof-v1",
            "generated_utc": now(),
            "status": "SOURCE_AUTHORIZED_FRESH_QKE_INIT_FIX_PROVEN",
            "terminal_eligible": True,
            "backend": "jax-cpu",
            "gpu_actions": 0,
            "wrf_or_mpi_executions": 0,
            "production_adapter_invocations": 0,
            "mynn_timestep_invocations": 0,
            "fresh_initializer_invocations": calls_after_fresh,
            "restart_initializer_invocations": calls["value"] - calls_after_fresh,
            "input_evidence": evidence,
            "capture_authority": {
                "nonce": input_manifest["nonce"],
                "seam": input_manifest["seam"],
                "synthetic_reconstruction": input_manifest["provenance"][
                    "synthetic_reconstruction"
                ],
            },
            "runtime": {
                "jax_version": jax.__version__,
                "jaxlib_version": __import__("jaxlib").__version__,
                "numpy_version": np.__version__,
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
            },
            "thresholds": thresholds,
            "checks": checks,
            "before_after": {
                "pre_fix_entry_vs_pre_fix_initialized": metrics(
                    before_entry, before_initialized
                ),
                "wrf_entry_vs_wrf_initialized": metrics(
                    wrf_entry, wrf_initialized
                ),
                "pre_fix_initialized_vs_wrf_initialized": before,
                "post_fix_initialized_vs_wrf_initialized": after_target,
                "post_fix_vs_pre_fix_initialized": metrics(
                    after, before_initialized
                ),
                "post_fix_rms_improvement_factor": improvement,
                "post_fix_bitpayload_sha256": array_sha(after),
                "restart_output_vs_input": metrics(
                    restart_qke, np.asarray(state.qke)
                ),
            },
            "source_decision": {
                "sealed_fixture": (
                    "Receipt-bound namelist has restart=.false.; step 1 therefore "
                    "takes WRF's fresh non-restart branch."
                ),
                "faithful_fix": (
                    "Fresh first call unconditionally rebuilds QKE after surface "
                    "fluxes, independent of incoming QKE magnitude."
                ),
                "restart_behavior": (
                    "Explicit restart=True skips the initialization, matching "
                    "WRF's outer .not.restart guard."
                ),
                "full_sp2_closure_claimed": False,
                "reason": (
                    "The one-run rule forbids a second production adapter capture; "
                    "this proof establishes the source-localized lifecycle fix, "
                    "not closure of downstream mass-flux/mixing SP2 error."
                ),
            },
            "passed": True,
        }
        atomic_json(output, proof)
        print(json.dumps({
            "passed": True,
            "output": str(output),
            "pre_fix_rms": before["rms"],
            "post_fix_rms": after_target["rms"],
            "improvement_factor": improvement,
            "status": proof["status"],
            "canonical_payload_sha256": canonical_without_self(proof),
        }, sort_keys=True))
        return 0
    except (FixProofFailure, OSError, ValueError, TypeError, KeyError,
            json.JSONDecodeError, subprocess.CalledProcessError,
            ET.ParseError) as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
