#!/usr/bin/env python3
"""One-shot CPU production-adapter capture for v0234 MYNN SP2 attribution.

The capture mode calls ``mynn_adapter_with_source_leaves`` exactly once.  JAX
debug callbacks observe the operands inside that same compiled CPU execution;
they do not recompute the PBL step.  The namespace is consumed before backend
import and is never reused after success or failure.

Preflight mode is backend-dark and does not create the external namespace.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-single-authority-attribution"
CAPTURE_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084/"
    "capture/authentic-ac6712170cbe5084"
)
CAPTURE_MANIFEST = CAPTURE_ROOT / "manifest.json"
CAPTURE_MANIFEST_SHA256 = (
    "295760b17cba2fde8e931188caab76fbf0564b7e9229784e8f739270f75309e6"
)
INPUT_DIR = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_0b18530a1dc9cac2/"
    "capture-inputs"
)
NAMELIST_SHA256 = (
    "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838"
)
WRFINPUT_D03_SHA256 = (
    "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a"
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
STATIC_SOURCE_FILES = (
    "src/gpuwrf/_x64_config.py",
    "src/gpuwrf/contracts/state.py",
    "src/gpuwrf/coupling/physics_couplers.py",
    "src/gpuwrf/dynamics/metrics.py",
    "src/gpuwrf/io/gen2_accessor.py",
    "src/gpuwrf/physics/mynn_constants.py",
    "src/gpuwrf/physics/mynn_edmf.py",
    "src/gpuwrf/physics/mynn_pbl.py",
    "src/gpuwrf/physics/mynn_sgs_cloud.py",
    "src/gpuwrf/physics/mynn_surface_stub.py",
    "src/gpuwrf/physics/tridiagonal_solver.py",
)
NAMESPACE_PARENT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_gpt_single_authority_attribution"
)
DT = 6.0
NY = 93
NX = 111
NCOL = NY * NX


class CaptureFailure(RuntimeError):
    """Fail-closed capture error."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_without_self(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items()
               if key != "canonical_payload_sha256"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return sha256_bytes(encoded)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CaptureFailure(f"OUTPUT_NOT_FRESH:{path}")
    payload = dict(value)
    payload["canonical_payload_sha256"] = canonical_without_self(payload)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(
            (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
            .encode()
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    if result.returncode:
        stderr = result.stderr.decode() if binary else result.stderr
        raise CaptureFailure(f"GIT_FAILED:{' '.join(args)}:{stderr}")
    return result.stdout if binary else result.stdout.strip()


def tracked_source_record(relative: str) -> dict[str, Any]:
    path = REPO / relative
    if not path.is_file() or path.is_symlink():
        raise CaptureFailure(f"SOURCE_NOT_REGULAR:{relative}")
    head_bytes = git("show", f"HEAD:{relative}", binary=True)
    disk_bytes = path.read_bytes()
    if disk_bytes != head_bytes:
        raise CaptureFailure(f"SOURCE_NOT_HEAD:{relative}")
    return {
        "path": str(path),
        "relative": relative,
        "sha256": sha256_bytes(disk_bytes),
        "git_blob": str(git("rev-parse", f"HEAD:{relative}")),
        "size": len(disk_bytes),
    }


def resource_gate() -> dict[str, Any]:
    affinity = set(os.sched_getaffinity(0))
    if affinity != ALLOWED_CPUS:
        raise CaptureFailure(f"CPUSET_DRIFT:{sorted(affinity)}")
    environment = {key: os.environ.get(key) for key in THREAD_ENV}
    if environment != THREAD_ENV:
        raise CaptureFailure(f"THREAD_ENV_DRIFT:{environment}")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise CaptureFailure("JAX_PLATFORM_NOT_CPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise CaptureFailure("CUDA_VISIBLE_DEVICES_NOT_EMPTY")
    if os.environ.get("JAX_ENABLE_X64", "").lower() not in {"1", "true"}:
        raise CaptureFailure("JAX_X64_NOT_EXPLICIT")
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    ioprio = ctypes.CDLL(None, use_errno=True).syscall(252, 1, 0)
    if nice < 15 or ioprio < 0 or int(ioprio) >> 13 != 3:
        raise CaptureFailure(f"PRIORITY_DRIFT:{nice}:{ioprio}")
    for sentinel in (Path("/tmp/PREEMPT_CPU"),):
        if sentinel.exists() or sentinel.is_symlink():
            raise CaptureFailure(f"PREEMPT:{sentinel}")
    for name in os.environ:
        if name.startswith("GPUWRF_MYNN_"):
            raise CaptureFailure(f"MYNN_ENV_OVERRIDE:{name}")
    mem_available = 0
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            mem_available = int(line.split()[1]) * 1024
            break
    if mem_available < 8 * 1024**3:
        raise CaptureFailure(f"MEMORY_PRESSURE:{mem_available}")
    return {
        "logical_cpu_ids": sorted(affinity),
        "thread_environment": environment,
        "jax_platforms": os.environ["JAX_PLATFORMS"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "jax_enable_x64": os.environ["JAX_ENABLE_X64"],
        "nice": nice,
        "ionice_class": int(ioprio) >> 13,
        "memory_available_bytes": mem_available,
        "preempt_cpu_absent": True,
    }


def static_preflight(namespace: Path, proof_output: Path) -> dict[str, Any]:
    if "jax" in sys.modules or any(
        name.startswith("jax.") or name.startswith("gpuwrf.")
        for name in sys.modules
    ):
        raise CaptureFailure("BACKEND_IMPORTED_BEFORE_PREFLIGHT")
    if not namespace.is_absolute() or namespace.parent != NAMESPACE_PARENT:
        raise CaptureFailure(f"NAMESPACE_SCOPE:{namespace}")
    if namespace.exists() or namespace.is_symlink():
        raise CaptureFailure(f"NAMESPACE_CONSUMED:{namespace}")
    if proof_output.exists() or proof_output.is_symlink():
        raise CaptureFailure(f"PROOF_OUTPUT_NOT_FRESH:{proof_output}")
    if proof_output.parent.resolve() != SPRINT.resolve():
        raise CaptureFailure(f"PROOF_SCOPE:{proof_output}")
    # Proof/source identity is enforced byte-for-byte by
    # ``tracked_source_record``.  Permit unrelated untracked evidence objects in
    # a shared worktree while still refusing every tracked modification.
    if git("status", "--porcelain", "--untracked-files=no"):
        raise CaptureFailure("TRACKED_WORKTREE_NOT_CLEAN")
    if sha256_file(CAPTURE_MANIFEST) != CAPTURE_MANIFEST_SHA256:
        raise CaptureFailure("INPUT_CAPTURE_MANIFEST_DRIFT")
    if sha256_file(INPUT_DIR / "namelist.input") != NAMELIST_SHA256:
        raise CaptureFailure("NAMELIST_DRIFT")
    if sha256_file(INPUT_DIR / "wrfinput_d03") != WRFINPUT_D03_SHA256:
        raise CaptureFailure("WRFINPUT_D03_DRIFT")
    capture_manifest = json.loads(CAPTURE_MANIFEST.read_text(encoding="utf-8"))
    if (
        capture_manifest.get("status") != "CAPTURE_COMPLETE_HOLD_5_OF_28"
        or capture_manifest.get("seam", {}).get("pbl_guard_invocation_count") != 0
        or capture_manifest.get("seam", {}).get("trace_invocation_count") != 1
    ):
        raise CaptureFailure("INPUT_CAPTURE_STATUS_DRIFT")
    sources = {
        relative: tracked_source_record(relative)
        for relative in STATIC_SOURCE_FILES
    }
    return {
        "schema": "wrfgpu2-v0234-gpt-single-authority-preflight-v1",
        "passed": True,
        "backend_imported": False,
        "gpu_actions": 0,
        "wrf_or_mpi_executions": 0,
        "namespace_fresh": True,
        "namespace": str(namespace),
        "proof_output": str(proof_output),
        "git": {
            "head": git("rev-parse", "HEAD"),
            "tree": git("rev-parse", "HEAD^{tree}"),
            "branch": git("branch", "--show-current"),
            "worktree_clean": True,
        },
        "resource_gate": resource_gate(),
        "input_capture": {
            "root": str(CAPTURE_ROOT),
            "manifest_sha256": CAPTURE_MANIFEST_SHA256,
            "nonce": capture_manifest.get("nonce"),
            "state_leaf_count": len(capture_manifest.get("state_leaves", [])),
            "seam": capture_manifest.get("seam"),
        },
        "input_grid": {
            "root": str(INPUT_DIR),
            "namelist_sha256": NAMELIST_SHA256,
            "wrfinput_d03_sha256": WRFINPUT_D03_SHA256,
        },
        "static_sources": sources,
        "checked_utc": now(),
    }


def array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject or not np.isfinite(array).all():
        raise CaptureFailure("ARRAY_NONFINITE_OR_OBJECT")
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "nbytes": int(array.nbytes),
        "logical_c_bitpayload_sha256": sha256_bytes(array.tobytes()),
    }


def delta_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise CaptureFailure(f"METRIC_SHAPE:{left_array.shape}:{right_array.shape}")
    delta = left_array.astype(np.float64) - right_array.astype(np.float64)
    if not np.isfinite(delta).all():
        raise CaptureFailure("METRIC_NONFINITE")
    return {
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "sse": float(np.sum(delta * delta)),
        "exact_fraction": float(np.mean(left_array == right_array)),
    }


def mass_order(value: np.ndarray) -> np.ndarray:
    """Convert flattened production columns to (z,y,x) or (y,x)."""

    array = np.asarray(value)
    if array.ndim == 1 and array.shape[0] == NCOL:
        return np.ascontiguousarray(array.reshape(NY, NX))
    if array.ndim == 2 and array.shape[0] == NCOL:
        return np.ascontiguousarray(
            array.reshape(NY, NX, array.shape[1]).transpose(2, 0, 1)
        )
    return np.ascontiguousarray(array)


class RuntimeRecorder:
    def __init__(self, jax_module: Any):
        self.jax = jax_module
        self.scheduled: list[str] = []
        self.values: dict[str, dict[str, np.ndarray]] = {}

    def schedule(self, name: str, values: Mapping[str, Any]) -> None:
        if name in self.scheduled:
            raise CaptureFailure(f"CALLBACK_SCHEDULE_DUPLICATE:{name}")
        keys = tuple(values)
        leaves = tuple(values[key] for key in keys)
        self.scheduled.append(name)

        def callback(*runtime_values: Any) -> None:
            if name in self.values:
                raise CaptureFailure(f"CALLBACK_RUNTIME_DUPLICATE:{name}")
            captured = {
                key: np.array(value, copy=True)
                for key, value in zip(keys, runtime_values, strict=True)
            }
            for key, value in captured.items():
                if value.dtype.hasobject or not np.isfinite(value).all():
                    raise CaptureFailure(f"CALLBACK_NONFINITE:{name}:{key}")
            self.values[name] = captured

        self.jax.debug.callback(callback, *leaves, ordered=True)


def imported_gpuwrf_sources() -> dict[str, Any]:
    records: dict[str, Any] = {}
    repo_resolved = REPO.resolve()
    for module_name, module in sorted(sys.modules.items()):
        if not (module_name == "gpuwrf" or module_name.startswith("gpuwrf.")):
            continue
        source = getattr(module, "__file__", None)
        if not source:
            continue
        path = Path(source).resolve()
        if path.suffix == ".pyc":
            path = Path(str(path)[:-1])
        try:
            relative = path.relative_to(repo_resolved).as_posix()
        except ValueError:
            raise CaptureFailure(f"GPUWRF_MODULE_OUTSIDE_REPO:{module_name}:{path}")
        if path.suffix != ".py":
            continue
        records[module_name] = tracked_source_record(relative)
    if not records:
        raise CaptureFailure("NO_IMPORTED_GPUWRF_SOURCES")
    return records


def grid_with_wrfinput_metrics(run: Any, input_dir: Path, metrics_loader: Any) -> Any:
    """Replace ``as_grid_spec`` fallback metrics with the real-case payload."""

    grid = run.grid("d03").as_grid_spec()
    return dataclass_replace(
        grid,
        metrics=metrics_loader(input_dir / "wrfinput_d03"),
    )


def execute_capture(namespace: Path, proof_output: Path) -> dict[str, Any]:
    preflight = static_preflight(namespace, proof_output)
    namespace.parent.mkdir(parents=True, exist_ok=True)
    namespace.mkdir(mode=0o755)
    start = {
        "schema": "wrfgpu2-v0234-gpt-single-authority-run-start-v1",
        "namespace_consumed": True,
        "started_utc": now(),
        "pid": os.getpid(),
        "command": [sys.executable, *sys.argv],
        "preflight_canonical_sha256": canonical_without_self(preflight),
        "git_head": preflight["git"]["head"],
    }
    atomic_json(namespace / "run-start.json", start)

    try:
        sys.path.insert(0, str(REPO / "src"))
        import jax
        import jax.numpy as jnp
        from gpuwrf.contracts.state import State
        from gpuwrf.coupling import physics_couplers as couplers
        from gpuwrf.dynamics.metrics import load_wrfinput_metrics
        from gpuwrf.io.gen2_accessor import Gen2Run
        from gpuwrf.physics import mynn_pbl as mynn

        if jax.default_backend() != "cpu" or any(
            device.platform != "cpu" for device in jax.devices()
        ):
            raise CaptureFailure("NON_CPU_BACKEND")
        if not bool(jax.config.jax_enable_x64):
            raise CaptureFailure("JAX_X64_DISABLED")
        if not (
            bool(mynn._MYNN_SGS_CLOUD)
            and bool(mynn._MYNN_COLUMN_TILING)
            and int(mynn._MYNN_COLUMN_TILE_COLS) == 16384
            and bool(couplers._MYNN_EDMF)
        ):
            raise CaptureFailure(
                "OPERATIONAL_SWITCH_DRIFT:"
                f"sgs={mynn._MYNN_SGS_CLOUD}:tiling={mynn._MYNN_COLUMN_TILING}:"
                f"tile_cols={mynn._MYNN_COLUMN_TILE_COLS}:edmf={couplers._MYNN_EDMF}"
            )

        capture_manifest = json.loads(CAPTURE_MANIFEST.read_text(encoding="utf-8"))
        state_values = []
        for item in capture_manifest.get("state_leaves", []):
            if item["kind"] == "none":
                state_values.append(None)
            else:
                state_values.append(
                    jax.device_put(
                        np.load(CAPTURE_ROOT / item["file"], allow_pickle=False)
                    )
                )
        if len(state_values) != len(State.__slots__):
            raise CaptureFailure("STATE_SCHEMA_DRIFT")
        state = State.tree_unflatten(None, tuple(state_values))
        # ``Gen2GridSpec.as_grid_spec`` deliberately supplies analytic-flat
        # fallback metrics; the operational real-case loader immediately
        # replaces them with the WRF payload (d02_replay.py:2109-2114).  The
        # CPU authority must do the same.  Leaving the fallback here silently
        # fed MYNN eta-coordinate placeholders (c3~=eta, c4=0), corrupting the
        # phy_prep pressure/density reconstructed from the otherwise authentic
        # pre-PBL State.
        grid = grid_with_wrfinput_metrics(
            Gen2Run(INPUT_DIR), INPUT_DIR, load_wrfinput_metrics,
        )
        if (
            tuple(np.asarray(state.theta).shape) != (44, NY, NX)
            or float(couplers._mynn_dx(grid)) != 1000.0
        ):
            raise CaptureFailure("FROZEN_GRID_DRIFT")

        recorder = RuntimeRecorder(jax)
        solve_trace_count = {"value": 0}
        initialization_count = {"value": 0}
        initialized_host: dict[str, np.ndarray] = {}
        entry_host: dict[str, np.ndarray] = {}

        original_initialize = couplers._mynn_state_with_first_call_qke
        original_surface_terms = mynn._surface_terms
        original_edmf = mynn._edmf_arrays_from_state
        original_turbulence = mynn._mym_turbulence
        original_predict_qke = mynn._mym_predict_qke
        original_predict_qsq = mynn._mym_predict_qsq
        original_mean = mynn._apply_mean_tendencies
        original_solve = mynn._solve_tridiagonal

        selected_state_fields = (
            "u", "v", "theta", "qv", "qke", "p", "rho", "dz", "ustar",
            "theta_flux", "qv_flux", "rhosfc", "fltv", "xland", "qsq",
        )

        def initialize_hook(
            input_state: Any,
            input_grid: Any,
            first_timestep: Any,
            *,
            restart: bool = False,
        ):
            initialization_count["value"] += 1
            if initialization_count["value"] != 1:
                raise CaptureFailure("INITIALIZATION_REPEAT")
            if restart is not False:
                raise CaptureFailure("POSTFIX_CAPTURE_REQUIRES_FRESH_START")
            for field in selected_state_fields:
                value = getattr(input_state, field, None)
                if value is not None:
                    entry_host[field] = np.ascontiguousarray(np.asarray(value))
            initialized = original_initialize(
                input_state, input_grid, first_timestep, restart=restart
            )
            for field in selected_state_fields:
                value = getattr(initialized, field, None)
                if value is not None:
                    initialized_host[field] = np.ascontiguousarray(np.asarray(value))
            return initialized

        def surface_terms_hook(column_state: Any, surface: Any = None):
            flux, wind, fltv, rhosfc = original_surface_terms(column_state, surface)
            values: dict[str, Any] = {
                **{
                    f"state_{field}": getattr(column_state, field)
                    for field in (
                        "u", "v", "w", "theta", "qv", "tke", "p", "rho",
                        "dz", "qc", "qi", "qs", "qsq",
                    )
                },
                **{
                    f"flux_{field}": getattr(flux, field)
                    for field in flux._fields
                },
                "wind": wind,
                "fltv": fltv,
                "rhosfc": rhosfc,
            }
            recorder.schedule("surface_terms", values)
            return flux, wind, fltv, rhosfc

        def edmf_hook(column_state: Any, flux: Any, fltv: Any, pblh: Any,
                      dt: Any, dx: Any):
            result = original_edmf(column_state, flux, fltv, pblh, dt, dx)
            required = (
                "s_aw", "s_awu", "s_awv", "s_awthl", "s_awqv", "edmf_a",
                "edmf_qc", "edmf_qt",
            )
            missing = [name for name in required if name not in result]
            if missing:
                raise CaptureFailure(f"EDMF_SCHEMA:{missing}")
            recorder.schedule(
                "mass_flux",
                {name: result[name] for name in required},
            )
            return result

        def turbulence_hook(column_state: Any, qke: Any, fltv: Any,
                            ustar: Any, dx: Any, xland: Any = 1.0):
            result = original_turbulence(
                column_state, qke, fltv, ustar, dx, xland
            )
            recorder.schedule(
                "turbulence",
                {
                    "qke_input": qke,
                    "state_qc_bl": column_state.qc_bl,
                    "state_qi_bl": column_state.qi_bl,
                    "state_cldfra_bl": column_state.cldfra_bl,
                    "fltv": fltv,
                    "ustar": ustar,
                    "xland": xland,
                    "qkw": result["qkw"],
                    "el": result["el"],
                    "pblh": result["pblh"],
                    "dfm": result["dfm"],
                    "dfh": result["dfh"],
                    "pdk": result["pdk"],
                },
            )
            return result

        def predict_qke_hook(column_state: Any, qke: Any, turb: Any,
                             dt: Any, ustar: Any, flux: Any):
            result = original_predict_qke(
                column_state, qke, turb, dt, ustar, flux
            )
            recorder.schedule(
                "qke_predict",
                {
                    "qke_before": qke,
                    "qke_after": result[0],
                    "qwt": result[1],
                    "qdiss": result[2],
                    "pdk": result[3],
                },
            )
            return result

        def predict_qsq_hook(column_state: Any, qsq: Any, turb: Any,
                             dt: Any, s_aw: Any = None):
            result = original_predict_qsq(
                column_state, qsq, turb, dt, s_aw=s_aw
            )
            recorder.schedule(
                "qsq_predict", {"qsq_before": qsq, "qsq_after": result}
            )
            return result

        def solve_hook(a: Any, b: Any, c: Any, d: Any):
            index = solve_trace_count["value"]
            solve_trace_count["value"] += 1
            result = original_solve(a, b, c, d)
            recorder.schedule(
                f"solve_{index}", {"a": a, "b": b, "c": c, "d": d, "x": result}
            )
            return result

        def mean_hook(column_state: Any, turb: Any, dt: Any, flux: Any,
                      wind: Any, rhosfc: Any, mf: Any = None):
            if mf is None:
                raise CaptureFailure("MASS_FLUX_INACTIVE")
            raw_kmdz = mynn._rho_interfaces(column_state, turb["dfm"])
            floored_kmdz = mynn._apply_s_aw_stability_floor(
                raw_kmdz, mf["s_aw"]
            )
            dz = column_state.dz
            rho = column_state.rho
            rhoz_i = (
                rho[..., 1:] * dz[..., :-1] + rho[..., :-1] * dz[..., 1:]
            ) / (dz[..., :-1] + dz[..., 1:])
            rhoz_i = jnp.maximum(rhoz_i, 1.0e-4)
            rhoz = jnp.concatenate(
                (rho[..., :1], rhoz_i, rhoz_i[..., -1:]), axis=-1
            )
            result = original_mean(
                column_state, turb, dt, flux, wind, rhosfc, mf=mf
            )
            recorder.schedule(
                "mean_tendencies",
                {
                    "state_u": column_state.u,
                    "state_v": column_state.v,
                    "rho": rho,
                    "dz": dz,
                    "dfm": turb["dfm"],
                    "kmdz_raw": raw_kmdz,
                    "kmdz_floored": floored_kmdz,
                    "rhoz": rhoz,
                    "dtz": dt / dz,
                    "rhoinv": 1.0 / jnp.maximum(rho, 1.0e-4),
                    "ustar": flux.ustar,
                    "wind": wind,
                    "rhosfc": rhosfc,
                    "drag": rhosfc * flux.ustar * flux.ustar / wind,
                    "s_aw": mf["s_aw"],
                    "s_awu": mf["s_awu"],
                    "s_awv": mf["s_awv"],
                    "output_u": result[0],
                    "output_v": result[1],
                },
            )
            return result

        couplers._mynn_state_with_first_call_qke = initialize_hook
        mynn._surface_terms = surface_terms_hook
        mynn._edmf_arrays_from_state = edmf_hook
        mynn._mym_turbulence = turbulence_hook
        mynn._mym_predict_qke = predict_qke_hook
        mynn._mym_predict_qsq = predict_qsq_hook
        mynn._solve_tridiagonal = solve_hook
        mynn._apply_mean_tendencies = mean_hook

        adapter_invocations = 0
        try:
            adapter_invocations += 1
            leaves = couplers.mynn_adapter_with_source_leaves(
                state, DT, grid, first_timestep=True, restart=False
            )
            jax.tree_util.tree_map(
                lambda value: value.block_until_ready()
                if hasattr(value, "block_until_ready") else value,
                leaves,
            )
        finally:
            couplers._mynn_state_with_first_call_qke = original_initialize
            mynn._surface_terms = original_surface_terms
            mynn._edmf_arrays_from_state = original_edmf
            mynn._mym_turbulence = original_turbulence
            mynn._mym_predict_qke = original_predict_qke
            mynn._mym_predict_qsq = original_predict_qsq
            mynn._solve_tridiagonal = original_solve
            mynn._apply_mean_tendencies = original_mean

        expected_callbacks = {
            "surface_terms", "mass_flux", "turbulence", "qke_predict",
            "qsq_predict", "mean_tendencies",
            *(f"solve_{index}" for index in range(6)),
        }
        if adapter_invocations != 1:
            raise CaptureFailure(f"ADAPTER_INVOCATIONS:{adapter_invocations}")
        if initialization_count["value"] != 1:
            raise CaptureFailure(
                f"INITIALIZATION_INVOCATIONS:{initialization_count['value']}"
            )
        if solve_trace_count["value"] != 6:
            raise CaptureFailure(f"SOLVE_TRACE_COUNT:{solve_trace_count['value']}")
        if set(recorder.values) != expected_callbacks:
            raise CaptureFailure(
                f"CALLBACK_SET:{sorted(recorder.values)}:"
                f"EXPECTED:{sorted(expected_callbacks)}"
            )

        arrays: dict[str, np.ndarray] = {}
        for prefix, source_values in (
            ("entry_state", entry_host),
            ("initialized_state", initialized_host),
        ):
            for name, value in source_values.items():
                arrays[f"{prefix}_{name}"] = np.ascontiguousarray(value)
        for stage in (
            "surface_terms", "mass_flux", "turbulence", "qke_predict",
            "qsq_predict", "mean_tendencies",
        ):
            for name, value in recorder.values[stage].items():
                arrays[f"{stage}_{name}"] = mass_order(value)
        for component, index in (("u", 2), ("v", 3)):
            for name, value in recorder.values[f"solve_{index}"].items():
                arrays[f"solve_{component}_{name}"] = mass_order(value)
        arrays["adapter_rublten"] = np.ascontiguousarray(
            np.asarray(leaves.rublten)
        )
        arrays["adapter_rvblten"] = np.ascontiguousarray(
            np.asarray(leaves.rvblten)
        )
        arrays["adapter_rthblten"] = np.ascontiguousarray(
            np.asarray(leaves.rthblten)
        )
        arrays["adapter_rqvblten"] = np.ascontiguousarray(
            np.asarray(leaves.rqvblten)
        )
        for name, value in arrays.items():
            if value.dtype.hasobject or not np.isfinite(value).all():
                raise CaptureFailure(f"ARCHIVE_ARRAY_NONFINITE:{name}")

        self_consistency: dict[str, Any] = {}
        for component, field in (("u", "rublten"), ("v", "rvblten")):
            implied = (
                arrays[f"solve_{component}_x"]
                - arrays[f"mean_tendencies_state_{component}"]
            ) / DT
            self_consistency[component] = {
                "solve_x_vs_mean_output": delta_metrics(
                    arrays[f"solve_{component}_x"],
                    arrays[f"mean_tendencies_output_{component}"],
                ),
                "solve_implied_vs_adapter_tendency": delta_metrics(
                    implied, arrays[f"adapter_{field}"]
                ),
                "implied_bitpayload_sha256": sha256_bytes(
                    np.ascontiguousarray(implied).tobytes()
                ),
            }
            if (
                self_consistency[component]["solve_x_vs_mean_output"]["max_abs"]
                != 0.0
                or self_consistency[component][
                    "solve_implied_vs_adapter_tendency"
                ]["max_abs"] > 1.0e-14
            ):
                raise CaptureFailure(f"SINGLE_AUTHORITY_IDENTITY:{component}")

        archive_path = namespace / "single-authority-capture.npz"
        temporary_archive = namespace / f".capture.tmp-{os.getpid()}.npz"
        np.savez(temporary_archive, **arrays)
        with temporary_archive.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary_archive, archive_path)
        temporary_archive.unlink()

        sources = imported_gpuwrf_sources()
        if git("status", "--porcelain", "--untracked-files=no"):
            raise CaptureFailure("TRACKED_WORKTREE_CHANGED_DURING_CAPTURE")
        manifest = {
            "schema": "wrfgpu2-v0234-gpt-single-authority-capture-v1",
            "status": "SINGLE_AUTHORITY_CAPTURE_COMPLETE",
            "namespace": str(namespace),
            "started_utc": start["started_utc"],
            "finished_utc": now(),
            "authority": {
                "adapter_entry": (
                    "mynn_adapter_with_source_leaves(state,6.0,grid,"
                    "first_timestep=True,restart=False)"
                ),
                "adapter_invocations": adapter_invocations,
                "initialization_invocations": initialization_count["value"],
                "tridiagonal_invocations": solve_trace_count["value"],
                "callback_schedule": recorder.scheduled,
                "callback_runtime": list(recorder.values),
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "jax_version": jax.__version__,
                "jaxlib_version": __import__("jaxlib").__version__,
                "numpy_version": np.__version__,
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "edmf": bool(couplers._MYNN_EDMF),
                "sgs_cloud": bool(mynn._MYNN_SGS_CLOUD),
                "column_tiling": bool(mynn._MYNN_COLUMN_TILING),
                "column_tile_cols": int(mynn._MYNN_COLUMN_TILE_COLS),
                "dx": float(couplers._mynn_dx(grid)),
                "dt": DT,
                "gpu_actions": 0,
                "wrf_or_mpi_executions": 0,
            },
            "git": preflight["git"],
            "resource_gate": preflight["resource_gate"],
            "input_capture": preflight["input_capture"],
            "input_grid": preflight["input_grid"],
            "imported_gpuwrf_sources": sources,
            "archive": {
                "path": str(archive_path),
                "sha256": sha256_file(archive_path),
                "size": archive_path.stat().st_size,
                "array_order": list(arrays),
                "arrays": {name: array_record(value)
                           for name, value in arrays.items()},
            },
            "self_consistency": self_consistency,
            "single_namespace": True,
            "single_adapter_invocation": True,
            "production_operands_and_tendencies_same_invocation": True,
            "passed": True,
        }
        manifest_path = namespace / "manifest.json"
        atomic_json(manifest_path, manifest)
        manifest_file_sha = sha256_file(manifest_path)
        result = {
            "schema": "wrfgpu2-v0234-gpt-single-authority-capture-proof-v1",
            "verdict": "SINGLE_AUTHORITY_CPU_CAPTURE_GREEN",
            "namespace": str(namespace),
            "namespace_consumed_once": True,
            "manifest": {
                "path": str(manifest_path),
                "sha256": manifest_file_sha,
                "canonical_payload_sha256": canonical_without_self(manifest),
            },
            "archive": manifest["archive"],
            "authority": manifest["authority"],
            "git": manifest["git"],
            "resource_gate": manifest["resource_gate"],
            "input_capture": manifest["input_capture"],
            "self_consistency": self_consistency,
            "passed": True,
        }
        atomic_json(proof_output, result)
        atomic_json(
            namespace / "run-complete.json",
            {
                "schema": "wrfgpu2-v0234-gpt-single-authority-run-complete-v1",
                "status": "SINGLE_AUTHORITY_CAPTURE_COMPLETE",
                "manifest_sha256": manifest_file_sha,
                "archive_sha256": manifest["archive"]["sha256"],
                "proof_path": str(proof_output),
                "proof_sha256": sha256_file(proof_output),
                "finished_utc": now(),
            },
        )
        return result
    except BaseException as exc:
        failure_path = namespace / "run-failure.json"
        if not failure_path.exists():
            try:
                atomic_json(
                    failure_path,
                    {
                        "schema": "wrfgpu2-v0234-gpt-single-authority-run-failure-v1",
                        "status": "NAMESPACE_CONSUMED_CAPTURE_FAILED",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "failed_utc": now(),
                        "retry_same_namespace_permitted": False,
                    },
                )
            except BaseException:
                pass
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--capture", action="store_true")
    value.add_argument("--namespace-root", type=Path, required=True)
    value.add_argument("--proof-output", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    namespace = args.namespace_root.resolve()
    proof_output = args.proof_output.resolve()
    try:
        if args.preflight:
            result = static_preflight(namespace, proof_output)
            atomic_json(proof_output, result)
        else:
            result = execute_capture(namespace, proof_output)
        print(json.dumps({
            "verdict": result.get("verdict", "PREFLIGHT_GREEN"),
            "passed": result["passed"],
            "namespace": str(namespace),
            "proof": str(proof_output),
        }, sort_keys=True))
        return 0
    except (
        CaptureFailure, OSError, ValueError, TypeError, KeyError,
        json.JSONDecodeError, subprocess.SubprocessError,
    ) as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
