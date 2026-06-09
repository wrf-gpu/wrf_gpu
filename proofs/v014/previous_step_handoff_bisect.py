#!/usr/bin/env python3
"""V0.14 previous-step handoff bisection for the h10 d02 bad carry.

Evidence-only proof.  The live replay path is CPU-preflighted first, but the
current native-domain loader is GPU-only because ``State.zeros`` requires a JAX
GPU.  When a GPU replay is explicitly allowed, this script captures compact d02
patch statistics around the final parent/child partial subcycle and caches the
compact replay artifact outside git.  Later CPU validation runs reuse that
artifact and rewrite the repo proof files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


os.environ.setdefault("JAX_ENABLE_X64", "true")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proofs.v014 import prestep_carry_source_trace as trace  # noqa: E402


OUT_JSON = ROOT / "proofs/v014/previous_step_handoff_bisect.json"
OUT_MD = ROOT / "proofs/v014/previous_step_handoff_bisect.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-previous-step-handoff-bisect.md"

SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-09-v014-previous-step-handoff-bisect/sprint-contract.md"
)
PROJECT_CONSTITUTION = ROOT / "PROJECT_CONSTITUTION.md"
AGENTS = ROOT / "AGENTS.md"
VALIDATING_PHYSICS_SKILL = ROOT / ".agent/skills/validating-physics/SKILL.md"
WRF_ORACLE_SKILL = ROOT / ".agent/skills/building-wrf-oracles/SKILL.md"
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"

PRE_RK_JSON = ROOT / "proofs/v014/pre_rk_input_boundary.json"
SOURCE_TRACE_JSON = ROOT / "proofs/v014/prestep_carry_source_trace.json"
SOURCE_TRACE_MD = ROOT / "proofs/v014/prestep_carry_source_trace.md"
PRODUCER_SCRIPT = ROOT / "proofs/v014/jax_h10_prestep_carry_producer.py"
PRODUCER_JSON = ROOT / "proofs/v014/jax_h10_prestep_carry_producer.json"
DOMAIN_TREE_MODULE = ROOT / "src/gpuwrf/runtime/domain_tree.py"
OPERATIONAL_MODE_MODULE = ROOT / "src/gpuwrf/runtime/operational_mode.py"
STATE_MODULE = ROOT / "src/gpuwrf/contracts/state.py"

CHECKPOINT = Path(
    os.environ.get(
        "WRFGPU2_H10_PRESTEP_CARRY",
        "/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl",
    )
)
CHECKPOINT_PROVENANCE = CHECKPOINT.with_suffix(".provenance.json")

RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
DEFAULT_INPUT_ROOTS = (
    Path("/tmp/v0120_merged_run_root"),
    Path("/mnt/data/canairy_meteo/runs/wrf_l2"),
)
TARGET_DOMAIN = "d02"
TARGET_FIELDS = ("T", "P", "PB", "MU", "MUB")
STATIC_BASE_FIELDS = ("PB", "MUB")
TARGET_STEP = 6000
PRESTEP_COMPLETED_STEPS = TARGET_STEP - 1
TARGET_DT_S = 6.0
THETA_OFFSET_K = 300.0
TOLERANCE_MAX_ABS = 2.0e-6
MAX_DOM = 2

ARTIFACT_ROOT_CANDIDATES = (
    Path(os.environ["WRFGPU2_PREVIOUS_STEP_HANDOFF_BISECT_ROOT"])
    if os.environ.get("WRFGPU2_PREVIOUS_STEP_HANDOFF_BISECT_ROOT")
    else None,
    Path("/mnt/data/wrf_gpu2/v014_previous_step_handoff_bisect"),
    Path("/tmp/wrf_gpu2_v014_previous_step_handoff_bisect"),
)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def artifact_root() -> Path:
    for candidate in ARTIFACT_ROOT_CANDIDATES:
        if candidate is None:
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    raise OSError("could not create v014_previous_step_handoff_bisect artifact root")


ARTIFACT_ROOT = artifact_root()
REPLAY_ARTIFACT_JSON = ARTIFACT_ROOT / "previous_step_handoff_bisect.live_replay_compact.json"


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path),
    }


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def run_command(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "returncode": int(proc.returncode),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "wall_s": float(time.perf_counter() - start),
            "error": repr(exc),
        }


def environment_snapshot() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "XLA_PYTHON_CLIENT_ALLOCATOR": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    try:
        import jax  # noqa: PLC0415

        devices = [str(device) for device in jax.devices()]
        env.update(
            {
                "jax_import_error": None,
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": devices,
                "gpu_device_count": len([device for device in devices if "cuda" in device.lower() or "gpu" in device.lower()]),
            }
        )
    except Exception as exc:
        env["jax_import_error"] = repr(exc)
        env["gpu_device_count"] = None
    return env


CPU_PREFLIGHT_CODE = r"""
import os
import time
from pathlib import Path
os.environ.setdefault("JAX_ENABLE_X64", "true")
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains, domain_names_for
RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
roots = [Path("/tmp/v0120_merged_run_root"), Path("/mnt/data/canairy_meteo/runs/wrf_l2")]
run_dir = next((root / RUN_ID for root in roots if (root / RUN_ID / "wrfinput_d01").exists()), None)
if run_dir is None:
    raise FileNotFoundError("missing native L2 run directory")
config = NestedPipelineConfig(
    input_dir=run_dir,
    output_dir=Path("/tmp/wrf_gpu2_v014_previous_step_cpu_preflight_unused"),
    proof_dir=Path("/tmp/wrf_gpu2_v014_previous_step_cpu_preflight_setup"),
    hours=10,
    max_dom=2,
    feedback=False,
)
t0 = time.perf_counter()
_load_domains(config, domain_names_for(2))
print({"status": "CPU_LOAD_OK", "wall_s": time.perf_counter() - t0, "run_dir": str(run_dir)})
"""


def cpu_replay_preflight() -> dict[str, Any]:
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = "src"
    result = run_command([sys.executable, "-c", CPU_PREFLIGHT_CODE], cwd=ROOT, env=env)
    status = "CPU_LOAD_OK" if result.get("returncode") == 0 else "CPU_LOAD_REQUIRES_GPU"
    detail = None
    stderr = str(result.get("stderr_tail") or "")
    if "State.zeros requires a GPU device" in stderr:
        detail = "src/gpuwrf/contracts/state.py::State.zeros requires a visible JAX GPU during _load_domains"
    return {
        "status": status,
        "practical_for_full_replay": status == "CPU_LOAD_OK",
        "detail": detail,
        "command_result": result,
        "exact_source_file_if_cpu_replay_required": "src/gpuwrf/contracts/state.py",
        "exact_hook_or_change_if_cpu_replay_required": (
            "Make State.zeros/_load_domains CPU-capable for native L2 domain initialization, "
            "or provide a CPU-loadable d01+d02 carry checkpoint at d01=1999/d02=5997."
        ),
    }


def nvidia_smi_query(query: str) -> dict[str, Any]:
    command = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    return run_command(command)


class VramSampler:
    def __init__(self, interval_s: float = 15.0) -> None:
        self.interval_s = float(interval_s)
        self.samples_mib: list[int] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "VramSampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            result = nvidia_smi_query("memory.used")
            if result.get("returncode") == 0:
                for raw in str(result.get("stdout_tail") or "").splitlines():
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        self.samples_mib.append(int(float(raw.split(",")[0].strip())))
                    except ValueError:
                        self.errors.append(raw)
            elif result.get("error"):
                self.errors.append(str(result["error"]))
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, Any]:
        return {
            "samples_mib": self.samples_mib[-40:],
            "max_periodic_sample_mib": max(self.samples_mib) if self.samples_mib else None,
            "sample_count": len(self.samples_mib),
            "errors_tail": self.errors[-5:],
        }


def input_run_dir() -> Path:
    for root in DEFAULT_INPUT_ROOTS:
        path = root / RUN_ID
        if path.is_dir() and (path / "wrfinput_d01").exists() and (path / "wrfinput_d02").exists():
            return path
    searched = ", ".join(str(root / RUN_ID) for root in DEFAULT_INPUT_ROOTS)
    raise FileNotFoundError(f"missing L2 native-init run directory; searched {searched}")


def patch_bounds_from_truth(wrf_truth: Mapping[str, Any]) -> dict[str, int]:
    records = wrf_truth["records"]["MASS_K1"]
    ys = [int(key[0]) for key in records]
    xs = [int(key[1]) for key in records]
    return {
        "y0": min(ys),
        "y1": max(ys) + 1,
        "x0": min(xs),
        "x1": max(xs) + 1,
        "count": len(records),
    }


def stats(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite_count": 0, "max_abs": None, "rmse": None}
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "max_abs": float(np.max(np.abs(finite))),
        "rmse": float(math.sqrt(float(np.mean(finite * finite)))),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def array_summary(array: Any) -> dict[str, Any]:
    arr = np.asarray(array)
    finite = np.asarray(arr, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    out: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "count": int(arr.size),
        "finite_count": int(finite.size),
    }
    if finite.size:
        out.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "max_abs": float(np.max(np.abs(finite))),
            }
        )
    return out


def patch_arrays_from_carry(carry: Any, bounds: Mapping[str, int]) -> dict[str, np.ndarray]:
    import jax  # noqa: PLC0415

    y0, y1, x0, x1 = (int(bounds[name]) for name in ("y0", "y1", "x0", "x1"))
    state = carry.state

    def host(value: Any) -> np.ndarray:
        return np.asarray(jax.device_get(value), dtype=np.float64)

    p_pert = state.p_perturbation[0, y0:y1, x0:x1]
    mu_pert = state.mu_perturbation[y0:y1, x0:x1]
    target = {
        "T": host(state.theta[0, y0:y1, x0:x1] - THETA_OFFSET_K),
        "P": host(p_pert),
        "PB": host(state.p_total[0, y0:y1, x0:x1] - p_pert),
        "MU": host(mu_pert),
        "MUB": host(state.mu_total[y0:y1, x0:x1] - mu_pert),
    }
    state_mub = state.mu_total[y0:y1, x0:x1] - mu_pert
    target.update(
        {
            "scratch.T_from_t_2ave_minus_300": host(carry.t_2ave[0, y0:y1, x0:x1] - THETA_OFFSET_K),
            "scratch.T_from_t_save_minus_300": host(carry.t_save[0, y0:y1, x0:x1] - THETA_OFFSET_K),
            "scratch.MU_from_mu_save": host(carry.mu_save[y0:y1, x0:x1]),
            "scratch.MU_from_muts_minus_state_MUB": host(carry.muts[y0:y1, x0:x1] - state_mub),
            "scratch.MUB_from_muts_minus_mu_save": host(carry.muts[y0:y1, x0:x1] - carry.mu_save[y0:y1, x0:x1]),
            "scratch.MUTS_total": host(carry.muts[y0:y1, x0:x1]),
        }
    )
    return target


def patch_arrays_from_jit_tuple(values: Any) -> dict[str, np.ndarray]:
    import jax  # noqa: PLC0415

    names = (
        "T",
        "P",
        "PB",
        "MU",
        "MUB",
        "scratch.T_from_t_2ave_minus_300",
        "scratch.T_from_t_save_minus_300",
        "scratch.MU_from_mu_save",
        "scratch.MU_from_muts_minus_state_MUB",
        "scratch.MUB_from_muts_minus_mu_save",
        "scratch.MUTS_total",
    )
    return {name: np.asarray(jax.device_get(value), dtype=np.float64) for name, value in zip(names, values)}


def compare_patch_to_truth(
    *,
    name: str,
    source_expression: str,
    field: str,
    patch: Any,
    wrf_truth: Mapping[str, Any],
    bounds: Mapping[str, int],
    target_leaf_eligible: bool,
) -> dict[str, Any]:
    tag, wrf_field, wrf_convention = trace.WRF_FIELD_SOURCE[field]
    arr = np.asarray(patch, dtype=np.float64)
    y0, x0 = int(bounds["y0"]), int(bounds["x0"])
    diffs: list[float] = []
    worst: dict[str, Any] | None = None
    skipped: list[dict[str, Any]] = []
    for key, record in sorted(wrf_truth["records"].get(tag, {}).items()):
        if wrf_field not in record:
            skipped.append({"native_key": list(key), "reason": f"missing {wrf_field}"})
            continue
        rel_y, rel_x = int(key[0]) - y0, int(key[1]) - x0
        try:
            candidate = float(arr[rel_y, rel_x])
        except Exception as exc:
            skipped.append({"native_key": list(key), "reason": repr(exc)})
            continue
        truth_value = float(record[wrf_field])
        diff = candidate - truth_value
        diffs.append(diff)
        if worst is None or abs(diff) > worst["abs_diff"]:
            worst = {
                "native_key": list(key),
                "patch_index": [rel_y, rel_x],
                "jax_candidate": candidate,
                "wrf_truth": truth_value,
                "diff_jax_minus_wrf": diff,
                "abs_diff": abs(diff),
            }
    entry = {
        "name": name,
        "field": field,
        "source_expression": source_expression,
        "target_leaf_eligible": bool(target_leaf_eligible),
        "wrf_source_field": wrf_field,
        "wrf_source_convention": wrf_convention,
        "array_summary": array_summary(arr),
        **stats(diffs),
        "worst": worst,
        "skipped_record_count": len(skipped),
        "skipped_records": skipped,
        "tolerance_max_abs": TOLERANCE_MAX_ABS,
    }
    entry["status"] = (
        "MATCH"
        if entry.get("max_abs") is not None and float(entry["max_abs"]) <= TOLERANCE_MAX_ABS
        else "DIFF"
    )
    return entry


TARGET_SOURCE = {
    "T": "carry.state.theta - 300 K, k=1",
    "P": "carry.state.p_perturbation, k=1",
    "PB": "carry.state.p_total - carry.state.p_perturbation, k=1",
    "MU": "carry.state.mu_perturbation",
    "MUB": "carry.state.mu_total - carry.state.mu_perturbation",
}

SCRATCH_SOURCE = {
    "scratch.T_from_t_2ave_minus_300": ("T", "carry.t_2ave - 300 K, k=1"),
    "scratch.T_from_t_save_minus_300": ("T", "carry.t_save - 300 K, k=1"),
    "scratch.MU_from_mu_save": ("MU", "carry.mu_save"),
    "scratch.MU_from_muts_minus_state_MUB": (
        "MU",
        "carry.muts - (carry.state.mu_total - carry.state.mu_perturbation)",
    ),
    "scratch.MUB_from_muts_minus_mu_save": ("MUB", "carry.muts - carry.mu_save"),
}

CHRONOLOGICAL_SNAPSHOT_ORDER = (
    "after_segment_replay_d02_step5997_before_final_partial_parent",
    "before_parent_d01_step2000_child_d02_step5997",
    "after_parent_d01_step2000_before_child_force",
    "before_operational_force_d02_step5997",
    "after_operational_force_before_child_step5998",
    "after_child_advance_step5998_midscan_capture",
    "after_child_advance_step5999_midscan_capture",
    "after_child_advance_step5999_before_checkpoint_write",
)


def compare_arrays(name: str, left: Any, right: Any) -> dict[str, Any]:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape != right_arr.shape:
        return {
            "name": name,
            "status": "SHAPE_MISMATCH",
            "left_shape": list(left_arr.shape),
            "right_shape": list(right_arr.shape),
        }
    diff = np.asarray(left_arr, dtype=np.float64) - np.asarray(right_arr, dtype=np.float64)
    finite = diff[np.isfinite(diff)]
    exact = bool(np.array_equal(left_arr, right_arr))
    return {
        "name": name,
        "status": "EXACT" if exact else "DIFF",
        "array_equal": exact,
        "shape": list(left_arr.shape),
        "left_dtype": str(left_arr.dtype),
        "right_dtype": str(right_arr.dtype),
        "max_abs": float(np.max(np.abs(finite))) if finite.size else None,
        "rmse": float(math.sqrt(float(np.mean(finite * finite)))) if finite.size else None,
    }


def make_snapshot(
    *,
    label: str,
    kind: str,
    own_steps: Mapping[str, int],
    patch_arrays: Mapping[str, np.ndarray],
    wrf_truth: Mapping[str, Any],
    bounds: Mapping[str, int],
    previous_patch_arrays: Mapping[str, np.ndarray] | None,
    checkpoint_patch_arrays: Mapping[str, np.ndarray] | None,
    notes: str,
) -> dict[str, Any]:
    target: dict[str, Any] = {}
    scratch: dict[str, Any] = {}
    for field in TARGET_FIELDS:
        target[field] = compare_patch_to_truth(
            name=f"{label}.{field}",
            source_expression=TARGET_SOURCE[field],
            field=field,
            patch=patch_arrays[field],
            wrf_truth=wrf_truth,
            bounds=bounds,
            target_leaf_eligible=True,
        )
    for candidate, (field, source) in SCRATCH_SOURCE.items():
        scratch[candidate] = compare_patch_to_truth(
            name=f"{label}.{candidate}",
            source_expression=source,
            field=field,
            patch=patch_arrays[candidate],
            wrf_truth=wrf_truth,
            bounds=bounds,
            target_leaf_eligible=False,
        )
    scratch["scratch.MUTS_total"] = {
        "source_expression": "carry.muts",
        "array_summary": array_summary(patch_arrays["scratch.MUTS_total"]),
    }

    delta_previous = None
    if previous_patch_arrays is not None:
        delta_previous = {
            field: compare_arrays(
                f"{label}.{field}.delta_vs_previous_snapshot",
                patch_arrays[field],
                previous_patch_arrays[field],
            )
            for field in TARGET_FIELDS
        }
    delta_checkpoint = None
    if checkpoint_patch_arrays is not None:
        delta_checkpoint = {
            field: compare_arrays(
                f"{label}.{field}.delta_vs_existing_checkpoint",
                patch_arrays[field],
                checkpoint_patch_arrays[field],
            )
            for field in TARGET_FIELDS
        }

    worst_field = max(
        (field for field in TARGET_FIELDS if target[field].get("max_abs") is not None),
        key=lambda field: float(target[field]["max_abs"]),
    )
    return {
        "label": label,
        "kind": kind,
        "own_steps": {key: int(value) for key, value in own_steps.items()},
        "notes": notes,
        "target_leaf_comparisons": target,
        "scratch_comparisons": scratch,
        "all_target_fields_match_wrf_truth": all(target[field]["status"] == "MATCH" for field in TARGET_FIELDS),
        "static_base_fields_match_wrf_truth": all(target[field]["status"] == "MATCH" for field in STATIC_BASE_FIELDS),
        "worst_target_field": {
            "field": worst_field,
            "max_abs": target[worst_field].get("max_abs"),
            "rmse": target[worst_field].get("rmse"),
        },
        "delta_vs_previous_snapshot": delta_previous,
        "delta_vs_existing_checkpoint": delta_checkpoint,
    }


def instrumented_child_advance_available() -> bool:
    return True


def build_instrumented_child_advance():
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from functools import partial  # noqa: PLC0415
    from gpuwrf.runtime.operational_mode import _physics_boundary_step  # noqa: PLC0415

    @partial(jax.jit, static_argnames=("n_steps", "cadence", "y0", "y1", "x0", "x1"))
    def instrumented_advance(carry, namelist, start_step, *, n_steps: int, cadence: int, y0: int, y1: int, x0: int, x1: int):
        start_step = jnp.asarray(start_step, dtype=jnp.int32)
        indices = start_step + jnp.arange(int(n_steps), dtype=jnp.int32)
        run_physics = bool(namelist.run_physics)

        def compact(next_carry):
            state = next_carry.state
            p_pert = state.p_perturbation[0, y0:y1, x0:x1]
            mu_pert = state.mu_perturbation[y0:y1, x0:x1]
            state_mub = state.mu_total[y0:y1, x0:x1] - mu_pert
            return (
                state.theta[0, y0:y1, x0:x1] - THETA_OFFSET_K,
                p_pert,
                state.p_total[0, y0:y1, x0:x1] - p_pert,
                mu_pert,
                state.mu_total[y0:y1, x0:x1] - mu_pert,
                next_carry.t_2ave[0, y0:y1, x0:x1] - THETA_OFFSET_K,
                next_carry.t_save[0, y0:y1, x0:x1] - THETA_OFFSET_K,
                next_carry.mu_save[y0:y1, x0:x1],
                next_carry.muts[y0:y1, x0:x1] - state_mub,
                next_carry.muts[y0:y1, x0:x1] - next_carry.mu_save[y0:y1, x0:x1],
                next_carry.muts[y0:y1, x0:x1],
            )

        def body(scan_carry, step_index):
            if run_physics:
                run_radiation = jnp.equal(jnp.mod(step_index, int(cadence)), 0)
            else:
                run_radiation = False
            next_carry = _physics_boundary_step(
                scan_carry,
                namelist,
                step_index,
                run_radiation=run_radiation,
                debug=False,
            )
            return next_carry, compact(next_carry)

        final_carry, compact_by_step = jax.lax.scan(body, carry, indices)
        return final_carry, compact_by_step

    return instrumented_advance


def source_nodes() -> dict[str, Any]:
    return {
        "producer_produce_checkpoint": trace.extract_ast_node(PRODUCER_SCRIPT, "produce_checkpoint"),
        "domain_tree_operational_force": trace.extract_ast_node(DOMAIN_TREE_MODULE, "_operational_force"),
        "domain_tree_operational_advance_factory": trace.extract_ast_node(DOMAIN_TREE_MODULE, "_operational_advance_factory"),
        "domain_tree_run_operational_domain_tree": trace.extract_ast_node(DOMAIN_TREE_MODULE, "run_operational_domain_tree"),
        "operational_mode_advance_chunk": trace.extract_ast_node(OPERATIONAL_MODE_MODULE, "_advance_chunk"),
        "operational_mode_physics_boundary_step": trace.extract_ast_node(OPERATIONAL_MODE_MODULE, "_physics_boundary_step"),
        "state_zeros": trace.extract_ast_node(STATE_MODULE, "zeros"),
    }


def input_records() -> dict[str, Any]:
    return {
        "project_constitution": path_info(PROJECT_CONSTITUTION),
        "agents": path_info(AGENTS),
        "sprint_contract": path_info(SPRINT_CONTRACT),
        "validating_physics_skill": path_info(VALIDATING_PHYSICS_SKILL),
        "building_wrf_oracles_skill": path_info(WRF_ORACLE_SKILL),
        "handoff": path_info(HANDOFF),
        "pre_rk_input_boundary_json": path_info(PRE_RK_JSON),
        "prestep_carry_source_trace_json": path_info(SOURCE_TRACE_JSON),
        "prestep_carry_source_trace_md": path_info(SOURCE_TRACE_MD),
        "producer_script": path_info(PRODUCER_SCRIPT),
        "producer_json": path_info(PRODUCER_JSON),
        "checkpoint": path_info(CHECKPOINT),
        "checkpoint_provenance": path_info(CHECKPOINT_PROVENANCE),
        "domain_tree_module": path_info(DOMAIN_TREE_MODULE),
        "operational_mode_module": path_info(OPERATIONAL_MODE_MODULE),
        "state_module": path_info(STATE_MODULE),
    }


def required_commands() -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "required_validation": [
            "python -m py_compile proofs/v014/previous_step_handoff_bisect.py",
            "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/previous_step_handoff_bisect.py",
            "python -m json.tool proofs/v014/previous_step_handoff_bisect.json >/tmp/previous_step_handoff_bisect.validated.json",
        ],
        "gpu_replay_command_used": os.environ.get("WRFGPU2_PREVIOUS_STEP_HANDOFF_BISECT_COMMAND_DISPLAY"),
    }


def proof_objects() -> dict[str, str]:
    return {
        "json": str(OUT_JSON),
        "markdown": str(OUT_MD),
        "review": str(OUT_REVIEW),
        "compact_replay_artifact": str(REPLAY_ARTIFACT_JSON),
    }


def classify(
    snapshots: Mapping[str, Any],
    final_identity: Mapping[str, Any],
    capture_identity: Mapping[str, Any],
) -> tuple[str, list[str], str]:
    if not final_identity.get("all_target_fields_exact"):
        return (
            "REPRODUCER_MISMATCH_FINAL_CHECKPOINT_TARGET_LEAVES",
            [
                "The producer-shaped final d02 step-5999 replay did not exactly match the existing checkpoint target leaves.",
                "This is a replay/provenance mismatch, not a valid producer-path bisection result.",
            ],
            "Re-run a producer-shaped replay from the exact original provenance or add d01/d02 step-5997 carry savepoints before bisection.",
        )

    first = snapshots["after_segment_replay_d02_step5997_before_final_partial_parent"]
    if not first["static_base_fields_match_wrf_truth"]:
        bad_fields = [
            field
            for field in STATIC_BASE_FIELDS
            if first["target_leaf_comparisons"][field]["status"] == "DIFF"
        ]
        return (
            "BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE",
            [
                "The final producer-shaped replay exactly matches the existing bad checkpoint target leaves.",
                f"Static/base target leaves {bad_fields} already differ from CPU-WRF h10 pre-RK truth at d02 completed step 5997, before parent step 2000, _operational_force, or child step 5998/5999.",
                "Parent advance and _operational_force do not change the inspected child target leaves in this replay; their deltas are recorded as exact-zero where applicable.",
            ],
            "Open a narrower earlier-handoff/source sprint before d02 step 5997; do not target _operational_force or final child _advance_chunk first.",
        )

    after_parent = snapshots["after_parent_d01_step2000_before_child_force"]
    if not after_parent["static_base_fields_match_wrf_truth"]:
        return (
            "BAD_AFTER_PARENT_ADVANCE",
            [
                "The child static/base target leaves were clean at d02 step 5997 but bad after parent d01 step 2000.",
                "This would implicate final parent advancement or parent-derived state used for child forcing.",
            ],
            "Open a source-changing fix sprint around parent d01 step 2000 state production.",
        )

    after_force = snapshots["after_operational_force_before_child_step5998"]
    if not after_force["static_base_fields_match_wrf_truth"]:
        return (
            "BAD_AFTER_OPERATIONAL_FORCE",
            [
                "The child target leaves first became bad immediately after _operational_force.",
            ],
            "Open a source-changing fix sprint around _operational_force/build_child_boundary_package.",
        )

    step5998 = snapshots.get("after_child_advance_step5998_midscan_capture")
    if step5998 and not step5998["all_target_fields_match_wrf_truth"]:
        if not capture_identity.get("instrumented_final_matches_producer_shape"):
            return (
                "BISECTION_BLOCKED_MIDSCAN_CAPTURE_PERTURBS_FINAL",
                [
                    "The proof-only midscan capture did not reproduce the producer-shaped n_steps=2 final carry exactly.",
                    "An exact internal step-5998 savepoint requires a source hook inside _advance_chunk.",
                ],
                "Add a proof-only compact savepoint hook in src/gpuwrf/runtime/operational_mode.py::_advance_chunk and rerun this bisection.",
            )
        return (
            "BAD_AFTER_CHILD_ADVANCE_STEP_5998",
            [
                "The first target-field mismatch appears after child step 5998 in the proof-only midscan capture.",
                "The instrumented final capture matches the producer-shaped final carry, so the midscan surface is accepted.",
            ],
            "Open a source-changing fix sprint around child _advance_chunk step 5998.",
        )

    final = snapshots["after_child_advance_step5999_before_checkpoint_write"]
    if not final["all_target_fields_match_wrf_truth"]:
        return (
            "BAD_AFTER_CHILD_ADVANCE_STEP_5999",
            [
                "The final producer-shaped step-5999 carry matches the checkpoint and first differs from WRF truth at step 5999.",
            ],
            "Open a source-changing fix sprint around child _advance_chunk step 5999.",
        )

    return (
        "REPRODUCER_MISMATCH_NO_FINAL_WRF_MISMATCH",
        [
            "The final replay matched the checkpoint but did not reproduce the known WRF mismatch.",
        ],
        "Re-run proofs/v014/prestep_carry_source_trace.py and verify the checkpoint/truth identity.",
    )


def build_payload_from_live_replay(cpu_preflight: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from gpuwrf.integration.nested_pipeline import (  # noqa: PLC0415
        NestedPipelineConfig,
        _load_domains,
        domain_names_for,
    )
    from gpuwrf.runtime.checkpoint import read_checkpoint_with_runtime_state  # noqa: PLC0415
    from gpuwrf.runtime.domain_tree import (  # noqa: PLC0415
        DomainTree,
        _operational_advance_factory,
        _operational_force,
        run_operational_domain_tree,
    )

    started = time.perf_counter()
    if not PRE_RK_JSON.exists():
        raise FileNotFoundError(PRE_RK_JSON)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    pre_rk = load_json(PRE_RK_JSON)
    source_trace = load_json(SOURCE_TRACE_JSON)
    producer = load_json(PRODUCER_JSON)
    checkpoint_provenance = load_optional_json(CHECKPOINT_PROVENANCE)
    wrf_truth = trace.parse_pre_rk_truth(pre_rk)
    bounds = patch_bounds_from_truth(wrf_truth)

    run_dir = input_run_dir()
    config = NestedPipelineConfig(
        input_dir=run_dir,
        output_dir=ARTIFACT_ROOT / "unused_wrfouts",
        proof_dir=ARTIFACT_ROOT / "nested_setup",
        hours=10,
        max_dom=MAX_DOM,
        feedback=False,
    )
    names = domain_names_for(MAX_DOM)

    nvidia_before = nvidia_smi_query("timestamp,index,name,memory.used,memory.total,utilization.gpu")
    segment_records: list[dict[str, Any]] = []
    snapshots: dict[str, Any] = {}
    patch_cache: dict[str, dict[str, np.ndarray]] = {}

    with VramSampler() as sampler:
        load_start = time.perf_counter()
        hierarchy, bundles, meta, run_start, dt_by_domain = _load_domains(config, names)
        tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
        load_record = {
            "wall_s": float(time.perf_counter() - load_start),
            "run_start_utc": run_start.isoformat(),
            "dt_by_domain": {name: float(value) for name, value in dt_by_domain.items()},
        }
        parent = hierarchy.parent(TARGET_DOMAIN)
        if parent is None:
            raise ValueError(f"{TARGET_DOMAIN} has no live-nest parent")
        edge = next(edge for edge in tree.children(parent) if edge.child == TARGET_DOMAIN)
        ratio = int(edge.parent_grid_ratio)
        target_dt = float(dt_by_domain[TARGET_DOMAIN])
        if abs(target_dt - TARGET_DT_S) > 1.0e-9:
            raise ValueError(f"{TARGET_DOMAIN} dt_s={target_dt:g}, expected {TARGET_DT_S:g}")
        full_parent_steps = PRESTEP_COMPLETED_STEPS // ratio
        child_remainder = PRESTEP_COMPLETED_STEPS - full_parent_steps * ratio
        if child_remainder != 2:
            raise ValueError(f"expected child_remainder=2, got {child_remainder}")

        root_segment_steps = max(1, int(os.environ.get("WRFGPU2_H10_ROOT_SEGMENT_STEPS", "200")))
        carries: dict[str, Any] | None = None
        own_steps: dict[str, int] = {name: 0 for name in names}
        completed_parent = 0
        while completed_parent < full_parent_steps:
            seg = min(root_segment_steps, full_parent_steps - completed_parent)
            seg_start = time.perf_counter()
            result = run_operational_domain_tree(
                tree,
                root_steps=int(seg),
                feedback_enabled=False,
                output=None,
                output_cadence_steps=None,
                block_between=True,
                carries=carries,
                initial_own_steps=own_steps,
            )
            jax.block_until_ready(tuple(carry.state.theta for carry in result.carries.values()))
            carries = dict(result.carries)
            own_steps = dict(result.own_steps)
            completed_parent = int(own_steps[parent])
            record = {
                "segment": len(segment_records) + 1,
                "root_steps": int(seg),
                "wall_s": float(time.perf_counter() - seg_start),
                "own_steps": dict(own_steps),
            }
            segment_records.append(record)
            print(
                f"bisect_progress segment={record['segment']} d01={own_steps.get('d01')} d02={own_steps.get('d02')} wall_s={record['wall_s']:.1f}",
                flush=True,
            )

        if carries is None:
            raise RuntimeError("no carries produced by segmented replay")
        if int(own_steps[TARGET_DOMAIN]) != 5997:
            raise RuntimeError(f"expected d02 step 5997 before partial subcycle, got {own_steps[TARGET_DOMAIN]}")

        checkpoint_state, checkpoint_namelist, checkpoint_grid, checkpoint_step, checkpoint_carry = read_checkpoint_with_runtime_state(CHECKPOINT)
        del checkpoint_state, checkpoint_namelist, checkpoint_grid
        if checkpoint_carry is None:
            raise ValueError(f"{CHECKPOINT} has no runtime_state")
        if int(checkpoint_step) != PRESTEP_COMPLETED_STEPS:
            raise ValueError(f"checkpoint step {checkpoint_step}, expected {PRESTEP_COMPLETED_STEPS}")
        checkpoint_patch = patch_arrays_from_carry(checkpoint_carry, bounds)

        def add_snapshot(label: str, kind: str, note: str) -> None:
            previous_arrays = None
            if snapshots:
                previous_label = next(reversed(snapshots))
                previous_arrays = patch_cache[previous_label]
            arrays = patch_arrays_from_carry(carries[TARGET_DOMAIN], bounds)
            patch_cache[label] = arrays
            snapshots[label] = make_snapshot(
                label=label,
                kind=kind,
                own_steps=own_steps,
                patch_arrays=arrays,
                wrf_truth=wrf_truth,
                bounds=bounds,
                previous_patch_arrays=previous_arrays,
                checkpoint_patch_arrays=checkpoint_patch,
                notes=note,
            )

        add_snapshot(
            "after_segment_replay_d02_step5997_before_final_partial_parent",
            "producer_replay_surface",
            "After segmented native L2 live replay to d01=1999/d02=5997, before parent step 2000.",
        )
        add_snapshot(
            "before_parent_d01_step2000_child_d02_step5997",
            "producer_replay_surface",
            "Same child carry immediately before parent d01 step 2000.",
        )

        parent_before_patch = patch_arrays_from_carry(carries[parent], bounds)
        advance = _operational_advance_factory(tree)
        parent_start_step = int(own_steps[parent]) + 1
        parent_advance_start = time.perf_counter()
        carries[parent] = advance(parent, carries[parent], parent_start_step, 1)
        jax.block_until_ready(carries[parent].state.theta)
        own_steps[parent] += 1
        parent_advance_record = {
            "parent_start_step": int(parent_start_step),
            "wall_s": float(time.perf_counter() - parent_advance_start),
            "own_steps_after": dict(own_steps),
            "parent_source_patch_delta_before_after": {
                field: compare_arrays(
                    f"parent_d01_step2000.{field}.delta",
                    parent_before_patch[field],
                    patch_arrays_from_carry(carries[parent], bounds)[field],
                )
                for field in TARGET_FIELDS
                if parent_before_patch[field].shape == patch_arrays_from_carry(carries[parent], bounds)[field].shape
            },
        }

        add_snapshot(
            "after_parent_d01_step2000_before_child_force",
            "producer_replay_surface",
            "Child carry after parent d01 step 2000; child target leaves are not directly advanced here.",
        )
        add_snapshot(
            "before_operational_force_d02_step5997",
            "producer_replay_surface",
            "Child carry immediately before _operational_force.",
        )

        force_start = time.perf_counter()
        carries[TARGET_DOMAIN] = _operational_force(edge, carries[parent], carries[TARGET_DOMAIN])
        jax.block_until_ready(carries[TARGET_DOMAIN].state.theta)
        force_record = {
            "wall_s": float(time.perf_counter() - force_start),
            "edge": {"parent": edge.parent, "child": edge.child, "parent_grid_ratio": int(edge.parent_grid_ratio)},
        }
        add_snapshot(
            "after_operational_force_before_child_step5998",
            "producer_replay_surface",
            "Child carry immediately after _operational_force and before the final child advance.",
        )

        after_force_carry = carries[TARGET_DOMAIN]
        child_start_step = int(own_steps[TARGET_DOMAIN]) + 1
        child_namelist = tree.domains[TARGET_DOMAIN].namelist
        child_cadence = int(child_namelist.radiation_cadence_steps)

        instrumented = build_instrumented_child_advance()
        capture_start = time.perf_counter()
        capture_final_carry, compact_by_step = instrumented(
            after_force_carry,
            child_namelist,
            jnp.asarray(child_start_step, dtype=jnp.int32),
            n_steps=int(child_remainder),
            cadence=child_cadence,
            y0=int(bounds["y0"]),
            y1=int(bounds["y1"]),
            x0=int(bounds["x0"]),
            x1=int(bounds["x1"]),
        )
        jax.block_until_ready(capture_final_carry.state.theta)
        capture_wall_s = float(time.perf_counter() - capture_start)
        step_patch_tuples = []
        for idx in range(int(child_remainder)):
            step_patch_tuples.append(tuple(value[idx] for value in compact_by_step))

        step5998_arrays = patch_arrays_from_jit_tuple(step_patch_tuples[0])
        own_steps_5998 = dict(own_steps)
        own_steps_5998[TARGET_DOMAIN] = int(child_start_step)
        previous_label = next(reversed(snapshots))
        snapshots["after_child_advance_step5998_midscan_capture"] = make_snapshot(
            label="after_child_advance_step5998_midscan_capture",
            kind="proof_only_midscan_capture",
            own_steps=own_steps_5998,
            patch_arrays=step5998_arrays,
            wrf_truth=wrf_truth,
            bounds=bounds,
            previous_patch_arrays=patch_cache[previous_label],
            checkpoint_patch_arrays=checkpoint_patch,
            notes=(
                "Proof-only compact midscan capture of child step 5998 using the same _physics_boundary_step body as _advance_chunk; "
                "accepted only if the captured final carry matches producer-shaped n_steps=2."
            ),
        )
        patch_cache["after_child_advance_step5998_midscan_capture"] = step5998_arrays

        capture_final_arrays = patch_arrays_from_carry(capture_final_carry, bounds)
        snapshots["after_child_advance_step5999_midscan_capture"] = make_snapshot(
            label="after_child_advance_step5999_midscan_capture",
            kind="proof_only_midscan_capture",
            own_steps={**own_steps, TARGET_DOMAIN: PRESTEP_COMPLETED_STEPS},
            patch_arrays=capture_final_arrays,
            wrf_truth=wrf_truth,
            bounds=bounds,
            previous_patch_arrays=step5998_arrays,
            checkpoint_patch_arrays=checkpoint_patch,
            notes="Proof-only compact midscan final state from the instrumented two-step scan.",
        )
        patch_cache["after_child_advance_step5999_midscan_capture"] = capture_final_arrays

        producer_child_start = time.perf_counter()
        producer_shape_final = advance(TARGET_DOMAIN, after_force_carry, child_start_step, int(child_remainder))
        jax.block_until_ready(producer_shape_final.state.theta)
        producer_child_record = {
            "child_start_step": int(child_start_step),
            "child_steps": int(child_remainder),
            "wall_s": float(time.perf_counter() - producer_child_start),
        }
        carries[TARGET_DOMAIN] = producer_shape_final
        own_steps[TARGET_DOMAIN] += int(child_remainder)

        producer_shape_arrays = patch_arrays_from_carry(producer_shape_final, bounds)
        snapshots["after_child_advance_step5999_before_checkpoint_write"] = make_snapshot(
            label="after_child_advance_step5999_before_checkpoint_write",
            kind="producer_shaped_surface",
            own_steps=own_steps,
            patch_arrays=producer_shape_arrays,
            wrf_truth=wrf_truth,
            bounds=bounds,
            previous_patch_arrays=patch_cache["after_operational_force_before_child_step5998"],
            checkpoint_patch_arrays=checkpoint_patch,
            notes="Producer-shaped final child _advance_chunk call with n_steps=2, immediately before checkpoint write.",
        )
        patch_cache["after_child_advance_step5999_before_checkpoint_write"] = producer_shape_arrays

        final_identity = {
            "target_fields": {
                field: compare_arrays(
                    f"final_producer_shape_vs_existing_checkpoint.{field}",
                    producer_shape_arrays[field],
                    checkpoint_patch[field],
                )
                for field in TARGET_FIELDS
            }
        }
        final_identity["all_target_fields_exact"] = all(
            item.get("status") == "EXACT" and item.get("max_abs") in {0.0, None}
            for item in final_identity["target_fields"].values()
        )
        capture_identity = {
            "target_fields": {
                field: compare_arrays(
                    f"instrumented_final_vs_producer_shape.{field}",
                    capture_final_arrays[field],
                    producer_shape_arrays[field],
                )
                for field in TARGET_FIELDS
            },
            "capture_wall_s": capture_wall_s,
            "instrumented_child_advance_available_without_src_edits": instrumented_child_advance_available(),
        }
        capture_identity["instrumented_final_matches_producer_shape"] = all(
            item.get("status") == "EXACT" and item.get("max_abs") in {0.0, None}
            for item in capture_identity["target_fields"].values()
        )

        vram_summary = sampler.summary()

    nvidia_after = nvidia_smi_query("timestamp,index,name,memory.used,memory.total,utilization.gpu")
    classification, rationale, next_decision = classify(snapshots, final_identity, capture_identity)

    if classification.startswith("REPRODUCER_MISMATCH"):
        verdict = classification
    elif classification.startswith("BISECTION_BLOCKED"):
        verdict = classification
    else:
        verdict = classification

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.previous_step_handoff_bisect.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "classification": classification,
        "classification_rationale": rationale,
        "blocked": None
        if not classification.startswith("BISECTION_BLOCKED")
        else {
            "reason": classification.removeprefix("BISECTION_BLOCKED_"),
            "exact_hook_or_source_file_needed": "src/gpuwrf/runtime/operational_mode.py::_advance_chunk",
            "next_command_needed": "Add a compact proof-only midscan savepoint hook, then rerun proofs/v014/previous_step_handoff_bisect.py.",
        },
        "target": {
            "domain": TARGET_DOMAIN,
            "wrf_step": TARGET_STEP,
            "prestep_completed_steps": PRESTEP_COMPLETED_STEPS,
            "valid_time_utc": "2026-05-02T04:00:00Z",
            "wrf_surface": "dyn_em/solve_em.F after grid%itimestep increment before current-step physics/RK",
            "target_fields": list(TARGET_FIELDS),
            "static_base_fields_used_for_early_classification": list(STATIC_BASE_FIELDS),
            "tolerance_max_abs": TOLERANCE_MAX_ABS,
            "patch_bounds": dict(bounds),
        },
        "cpu_preflight": cpu_preflight,
        "gpu_used": True,
        "gpu_replay": {
            "why_cpu_replay_not_practical": cpu_preflight.get("detail")
            or "CPU live replay preflight failed; see cpu_preflight.command_result.",
            "command_display": os.environ.get("WRFGPU2_PREVIOUS_STEP_HANDOFF_BISECT_COMMAND_DISPLAY"),
            "environment": environment_snapshot(),
            "backend": jax.default_backend(),
            "allocator": {
                "XLA_PYTHON_CLIENT_ALLOCATOR": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
                "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            },
            "nvidia_smi_before": nvidia_before,
            "nvidia_smi_after": nvidia_after,
            "observed_vram_mib": vram_summary,
        },
        "cpu_validation_reuses_compact_artifact": True,
        "production_src_edits": False,
        "wrf_source_edits": False,
        "tost_run": False,
        "switzerland_validation_run": False,
        "fp32_source_landing": False,
        "current_step_rk_acoustic_debug_run": False,
        "inputs_read": input_records(),
        "source_nodes": source_nodes(),
        "wrf_pre_rk_truth": {
            "source_json": str(PRE_RK_JSON),
            "source_verdict": pre_rk.get("verdict"),
            "files": wrf_truth["files"],
            "file_info": wrf_truth["file_info"],
            "metadata": wrf_truth["metadata"],
            "unique_counts": wrf_truth["unique_counts"],
            "duplicate_count": wrf_truth["duplicate_count"],
            "duplicate_max_delta": wrf_truth["duplicate_max_delta"],
            "field_source": trace.WRF_FIELD_SOURCE,
        },
        "starting_fact": {
            "source_trace_verdict": source_trace.get("verdict"),
            "source_trace_classification": source_trace.get("classification"),
            "checkpoint_identity_from_source_trace": source_trace.get("checkpoint_identity"),
        },
        "producer_provenance": {
            "producer_json_verdict": producer.get("verdict"),
            "checkpoint_provenance": checkpoint_provenance,
            "run_dir": str(run_dir),
            "load": load_record,
            "nesting": {
                "parent": parent,
                "parent_grid_ratio": ratio,
                "full_parent_steps": full_parent_steps,
                "child_remainder_steps": child_remainder,
                "root_segment_steps": root_segment_steps,
                "metadata": meta,
            },
            "segments": segment_records,
            "partial_subcycle": {
                "parent_advance": parent_advance_record,
                "operational_force": force_record,
                "child_advance_producer_shape": producer_child_record,
                "own_steps_after": dict(own_steps),
            },
        },
        "snapshots": snapshots,
        "final_reproducer_identity": final_identity,
        "midscan_capture_identity": capture_identity,
        "unreachable_without_source_hook": [
            {
                "surface": "final_child_step_rk3_pre_halo_state",
                "status": "UNREACHABLE_WITHOUT_SOURCE_HOOK",
                "exact_hook_or_source_file_needed": "src/gpuwrf/runtime/operational_mode.py::_physics_boundary_step/_advance_chunk",
                "reason": "The public DomainTree advance adapter returns only the post-step OperationalCarry; _rk_scan_step_with_pre_halo_capture is private and not threaded through _physics_boundary_step or _advance_chunk.",
            }
        ],
        "acceptance_notes": {
            "uses_cpu_wrf_pre_rk_truth_from_pre_rk_input_boundary": True,
            "distinguishes_reproducer_mismatch_from_bisection": True,
            "final_reproduced_d02_step5999_matches_existing_checkpoint": final_identity["all_target_fields_exact"],
            "production_source_edited": False,
            "wrf_source_edited": False,
            "no_tost": True,
            "no_switzerland_validation": True,
            "no_fp32": True,
            "no_hermes_or_telegram": True,
            "top_level_output_compact": True,
        },
        "commands": required_commands(),
        "proof_objects": proof_objects(),
        "unresolved_risks": [
            "The WRF oracle is the existing final h10 pre-RK patch only; no WRF step-5997 or step-5998 oracle was generated in this evidence sprint.",
            "The final RK3 pre-halo internal state remains behind a missing _advance_chunk/_physics_boundary_step hook.",
            "The live replay required GPU because the current native-domain loader is not CPU-capable.",
        ],
        "next_decision": next_decision,
        "wall_s": float(time.perf_counter() - started),
    }
    return payload


def blocked_payload(reason: str, detail: str, next_action: str, cpu_preflight: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": "wrfgpu2.v014.previous_step_handoff_bisect.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": f"BISECTION_BLOCKED_{reason}",
        "classification": f"BISECTION_BLOCKED_{reason}",
        "classification_rationale": [detail],
        "blocked": {
            "reason": reason,
            "detail": detail,
            "exact_hook_or_source_file_needed": next_action,
            "next_command_needed": next_action,
        },
        "cpu_preflight": cpu_preflight,
        "gpu_used": False,
        "production_src_edits": False,
        "wrf_source_edits": False,
        "tost_run": False,
        "switzerland_validation_run": False,
        "fp32_source_landing": False,
        "current_step_rk_acoustic_debug_run": False,
        "inputs_read": input_records(),
        "source_nodes": source_nodes(),
        "commands": required_commands(),
        "proof_objects": proof_objects(),
        "unresolved_risks": [detail],
        "next_decision": next_action,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# V0.14 Previous-Step Handoff Bisection",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Summary",
        "",
        f"- Classification: `{payload['classification']}`.",
        f"- Final replay matches checkpoint: `{payload.get('final_reproducer_identity', {}).get('all_target_fields_exact')}`.",
        f"- GPU used: `{payload.get('gpu_used')}`.",
        f"- CPU preflight: `{(payload.get('cpu_preflight') or {}).get('status')}`.",
        "",
    ]
    if payload.get("blocked"):
        blocked = payload["blocked"]
        lines.extend(
            [
                "## Blocker",
                "",
                f"- Reason: `{blocked['reason']}`.",
                f"- Detail: {blocked.get('detail')}",
                f"- Next: `{blocked.get('next_command_needed')}`.",
                "",
            ]
        )
    if payload.get("snapshots"):
        lines.extend(
            [
                "## Snapshot Results",
                "",
                "| surface | d01 | d02 | all target match WRF | static PB/MUB match | worst field | max abs |",
                "| --- | ---: | ---: | --- | --- | --- | ---: |",
            ]
        )
        snapshot_items = []
        snapshots = payload["snapshots"]
        for label in CHRONOLOGICAL_SNAPSHOT_ORDER:
            if label in snapshots:
                snapshot_items.append((label, snapshots[label]))
        for label, snap in snapshots.items():
            if label not in CHRONOLOGICAL_SNAPSHOT_ORDER:
                snapshot_items.append((label, snap))
        for label, snap in snapshot_items:
            own = snap.get("own_steps", {})
            worst = snap.get("worst_target_field", {})
            lines.append(
                f"| `{label}` | {own.get('d01')} | {own.get('d02')} | "
                f"`{snap.get('all_target_fields_match_wrf_truth')}` | "
                f"`{snap.get('static_base_fields_match_wrf_truth')}` | "
                f"`{worst.get('field')}` | {worst.get('max_abs')} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            str(payload.get("next_decision")),
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Previous-Step Handoff Bisection",
        "",
        f"verdict: `{payload['verdict']}`",
        "",
        "objective: bisect the live nested replay producer path that writes the bad h10 d02 step-5999 OperationalCarry.",
        "",
        "files changed:",
        "- `proofs/v014/previous_step_handoff_bisect.py`",
        "- `proofs/v014/previous_step_handoff_bisect.json`",
        "- `proofs/v014/previous_step_handoff_bisect.md`",
        "- `.agent/reviews/2026-06-09-v014-previous-step-handoff-bisect.md`",
        "",
        "commands run:",
    ]
    for command in payload.get("commands", {}).get("required_validation", []):
        lines.append(f"- `{command}`")
    gpu_command = payload.get("gpu_replay", {}).get("command_display")
    if gpu_command:
        lines.append(f"- `{gpu_command}`")
    lines.extend(
        [
            "",
            "proof objects produced:",
            "- `proofs/v014/previous_step_handoff_bisect.json`",
            "- `proofs/v014/previous_step_handoff_bisect.md`",
            "- `.agent/reviews/2026-06-09-v014-previous-step-handoff-bisect.md`",
            f"- `{REPLAY_ARTIFACT_JSON}`",
            "",
            "unresolved risks:",
        ]
    )
    for risk in payload.get("unresolved_risks", []):
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload.get('next_decision')}", ""])
    return "\n".join(lines)


def write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")


def main() -> int:
    force_replay = truthy(os.environ.get("WRFGPU2_PREVIOUS_STEP_HANDOFF_BISECT_FORCE_REPLAY"))
    allow_gpu = truthy(os.environ.get("WRFGPU2_PREVIOUS_STEP_HANDOFF_BISECT_ALLOW_GPU"))

    if REPLAY_ARTIFACT_JSON.exists() and not force_replay:
        payload = load_json(REPLAY_ARTIFACT_JSON)
        write_outputs(payload)
        print(payload["verdict"])
        print(f"json={OUT_JSON}")
        print(f"markdown={OUT_MD}")
        print(f"review={OUT_REVIEW}")
        return 0

    cpu_preflight = cpu_replay_preflight()
    if not allow_gpu:
        detail = (
            "CPU live replay is not practical in the current code path: "
            f"{cpu_preflight.get('detail') or 'CPU preflight failed'}."
        )
        payload = blocked_payload(
            "CPU_LOAD_REQUIRES_GPU",
            detail,
            (
                "Run a targeted GPU replay with WRFGPU2_PREVIOUS_STEP_HANDOFF_BISECT_ALLOW_GPU=1, "
                "or add a CPU-capable initialization hook in src/gpuwrf/contracts/state.py."
            ),
            cpu_preflight,
        )
        write_outputs(payload)
        print(payload["verdict"])
        print(f"json={OUT_JSON}")
        print(f"markdown={OUT_MD}")
        print(f"review={OUT_REVIEW}")
        return 0

    try:
        payload = build_payload_from_live_replay(cpu_preflight)
    except Exception as exc:
        payload = blocked_payload(
            "REPLAY_EXCEPTION",
            repr(exc),
            "Inspect the exception and rerun proofs/v014/previous_step_handoff_bisect.py with the same replay command.",
            cpu_preflight,
        )
    write_json(REPLAY_ARTIFACT_JSON, payload)
    write_outputs(payload)
    print(payload["verdict"])
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"review={OUT_REVIEW}")
    print(f"artifact={REPLAY_ARTIFACT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
