"""CPU-side GPU run preflight guards.

The nested forecast path is a long, expensive JAX/XLA job. These checks run
before any forecast compile/device work so an unsafe local GPU environment fails
closed with an actionable message instead of OOMing mid-integration.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Mapping, Sequence, TextIO

LOCK_FILE = Path("/tmp/wrf_gpu2_gpu.lock")
HOLDER_FILE = Path(f"{LOCK_FILE}.holder")
DEFAULT_MIN_FREE_VRAM_GIB = 24.0
DEFAULT_MIN_FREE_VRAM_FRACTION = 0.50

_FALSEY = {"0", "false", "no", "off", ""}
_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class VramSnapshot:
    """One ``nvidia-smi`` memory row for the selected GPU."""

    index: str
    name: str
    free_mib: int
    total_mib: int
    used_mib: int
    uuid: str = ""

    @property
    def free_gib(self) -> float:
        return self.free_mib / 1024.0

    @property
    def total_gib(self) -> float:
        return self.total_mib / 1024.0

    @property
    def used_gib(self) -> float:
        return self.used_mib / 1024.0


@dataclass(frozen=True)
class VramThreshold:
    """Resolved free-VRAM gate for one selected GPU."""

    min_free_gib: float
    absolute_floor_gib: float
    fraction: float
    fractional_gib: float | None
    source: str


class GpuPreflightError(RuntimeError):
    """Raised when a nested GPU run should not start."""

    def __init__(self, message: str, payload: Mapping[str, object]) -> None:
        super().__init__(message)
        self.payload = dict(payload)


def env_bool(name: str, default: bool, environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _FALSEY:
        return False
    if value in _TRUTHY:
        return True
    return default


def _explicit_min_free_vram_gib(environ: Mapping[str, str] | None = None) -> float | None:
    env = os.environ if environ is None else environ
    raw = env.get("GPUWRF_MIN_FREE_VRAM_GIB")
    if raw is None or not raw.strip():
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def min_free_vram_fraction(environ: Mapping[str, str] | None = None) -> float:
    """Return the fractional card-memory floor for nested-run preflight."""

    env = os.environ if environ is None else environ
    raw = env.get("GPUWRF_MIN_FREE_VRAM_FRACTION")
    if raw is None or not raw.strip():
        return DEFAULT_MIN_FREE_VRAM_FRACTION
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return DEFAULT_MIN_FREE_VRAM_FRACTION


def resolve_min_free_vram_threshold(
    environ: Mapping[str, str] | None = None,
    *,
    total_gib: float | None = None,
) -> VramThreshold:
    """Resolve the fail-closed free-VRAM threshold for the selected GPU."""

    env = os.environ if environ is None else environ
    explicit = _explicit_min_free_vram_gib(env)
    if explicit is not None:
        return VramThreshold(
            min_free_gib=explicit,
            absolute_floor_gib=explicit,
            fraction=0.0,
            fractional_gib=None,
            source="GPUWRF_MIN_FREE_VRAM_GIB explicit override",
        )
    fraction = min_free_vram_fraction(env)
    fractional_gib = None if total_gib is None else max(0.0, fraction * max(0.0, total_gib))
    min_free = DEFAULT_MIN_FREE_VRAM_GIB
    if fractional_gib is not None:
        min_free = max(DEFAULT_MIN_FREE_VRAM_GIB, fractional_gib)
    return VramThreshold(
        min_free_gib=min_free,
        absolute_floor_gib=DEFAULT_MIN_FREE_VRAM_GIB,
        fraction=fraction,
        fractional_gib=fractional_gib,
        source="max(default absolute floor, GPUWRF_MIN_FREE_VRAM_FRACTION * total VRAM)",
    )


def min_free_vram_gib(
    environ: Mapping[str, str] | None = None,
    *,
    total_gib: float | None = None,
) -> float:
    """Return the resolved fail-closed nested-run free-VRAM threshold."""

    return resolve_min_free_vram_threshold(environ, total_gib=total_gib).min_free_gib


def force_gpu_run_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the operator explicitly bypassed lock/headroom failures."""

    return env_bool("GPUWRF_FORCE_GPU_RUN", False, environ)


def read_gpu_lock_holder(holder_file: Path = HOLDER_FILE) -> str:
    try:
        text = holder_file.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return text


def _fd_points_to_lock(fd_text: str, lock_file_text: str) -> bool:
    try:
        fd = int(fd_text)
    except (TypeError, ValueError):
        return False
    if fd < 0:
        return False
    proc_fd = Path("/proc/self/fd") / str(fd)
    try:
        target = proc_fd.resolve(strict=True)
        lock_file = Path(lock_file_text).resolve(strict=False)
    except OSError:
        return False
    return target == lock_file


def gpu_lock_status(
    environ: Mapping[str, str] | None = None,
    *,
    holder_file: Path = HOLDER_FILE,
    lock_file: Path = LOCK_FILE,
) -> dict[str, object]:
    """Best-effort advisory check for a ``scripts/with_gpu_lock.sh`` holder.

    This is a cooperative same-box environment/holder-token check, not a hard
    proof that this process owns the kernel ``flock`` mutex.
    """

    env = os.environ if environ is None else environ
    holder = read_gpu_lock_holder(holder_file)
    fd_text = env.get("GPUWRF_GPU_LOCK_FD", "")
    lock_file_text = env.get("GPUWRF_GPU_LOCK_FILE", str(lock_file))
    token = env.get("GPUWRF_GPU_LOCK_TOKEN", "")
    held_flag = env.get("GPUWRF_GPU_LOCK_HELD", "") == "1"
    fd_ok = _fd_points_to_lock(fd_text, lock_file_text)
    token_ok = bool(token and token in holder)
    ok = bool(held_flag and fd_ok and token_ok)
    reason = ""
    if not held_flag:
        reason = "GPUWRF_GPU_LOCK_HELD is not set by scripts/with_gpu_lock.sh"
    elif not fd_ok:
        reason = f"GPU lock fd {fd_text!r} does not point to {lock_file_text}"
    elif not token_ok:
        reason = "GPU lock token is absent from the holder file"
    return {
        "ok": ok,
        "reason": reason,
        "holder": holder,
        "lock_file": str(lock_file),
        "holder_file": str(holder_file),
        "fd": fd_text,
        "token_present": bool(token),
    }


def _visible_device_tokens(environ: Mapping[str, str]) -> list[str] | None:
    raw = environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    return [token.strip() for token in raw.split(",") if token.strip()]


def _first_target_device_token(environ: Mapping[str, str]) -> str:
    tokens = _visible_device_tokens(environ)
    if tokens is None:
        return "0"
    if not tokens or tokens[0].lower() in {"-1", "none", "void", "nodevfiles"}:
        raise RuntimeError("CUDA_VISIBLE_DEVICES hides all CUDA GPUs")
    return tokens[0]


def _parse_nvidia_smi_memory_row(row: str) -> VramSnapshot:
    parts = [part.strip() for part in row.split(",")]
    if len(parts) < 6:
        raise RuntimeError(f"unexpected nvidia-smi memory row: {row!r}")
    try:
        free_mib = int(parts[2])
        total_mib = int(parts[3])
        used_mib = int(parts[4])
    except ValueError as exc:
        raise RuntimeError(f"could not parse nvidia-smi memory row: {row!r}") from exc
    return VramSnapshot(
        index=parts[0],
        name=", ".join(part for part in parts[5:] if part),
        free_mib=free_mib,
        total_mib=total_mib,
        used_mib=used_mib,
        uuid=parts[1],
    )


def _device_token_matches(snapshot: VramSnapshot, token: str) -> bool:
    if snapshot.index == token:
        return True
    if snapshot.uuid == token:
        return True
    return bool(snapshot.uuid and snapshot.uuid.startswith(token))


def select_nvidia_smi_snapshot(
    rows: Sequence[str],
    environ: Mapping[str, str] | None = None,
) -> VramSnapshot:
    """Select the ``nvidia-smi`` row for the GPU JAX will see as device 0."""

    env = os.environ if environ is None else environ
    snapshots = [_parse_nvidia_smi_memory_row(row) for row in rows if row.strip()]
    if not snapshots:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    target = _first_target_device_token(env)
    for snapshot in snapshots:
        if _device_token_matches(snapshot, str(target)):
            return snapshot
    choices = ", ".join(
        f"index={snapshot.index} uuid={snapshot.uuid or '<unknown>'}" for snapshot in snapshots
    )
    raise RuntimeError(
        f"CUDA_VISIBLE_DEVICES selects GPU {target!r}, but nvidia-smi returned only: {choices}"
    )


def query_nvidia_smi_memory(environ: Mapping[str, str] | None = None) -> VramSnapshot:
    """Query free memory on the CUDA-visible GPU JAX will use first."""

    env = os.environ if environ is None else environ
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise RuntimeError("nvidia-smi not found on PATH")
    proc = subprocess.run(
        [
            exe,
            "--query-gpu=index,uuid,memory.free,memory.total,memory.used,name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"nvidia-smi failed with rc={proc.returncode}: {err}")
    rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return select_nvidia_smi_snapshot(rows, env)


def _snapshot_payload(snapshot: VramSnapshot | None, error: str | None) -> dict[str, object]:
    if snapshot is None:
        return {"ok": False, "error": error or "unknown"}
    return {
        "ok": True,
        "index": snapshot.index,
        "name": snapshot.name,
        "uuid": snapshot.uuid,
        "free_mib": snapshot.free_mib,
        "used_mib": snapshot.used_mib,
        "total_mib": snapshot.total_mib,
        "free_gib": round(snapshot.free_gib, 3),
        "used_gib": round(snapshot.used_gib, 3),
        "total_gib": round(snapshot.total_gib, 3),
    }


def _format_status(payload: Mapping[str, object]) -> str:
    lock = payload.get("lock") if isinstance(payload.get("lock"), Mapping) else {}
    vram = payload.get("vram") if isinstance(payload.get("vram"), Mapping) else {}
    threshold = payload.get("vram_threshold") if isinstance(payload.get("vram_threshold"), Mapping) else {}
    holder = lock.get("holder") if isinstance(lock, Mapping) else ""
    if isinstance(vram, Mapping) and vram.get("ok"):
        free = vram.get("free_gib")
        total = vram.get("total_gib")
        device = vram.get("name")
        index = vram.get("index")
        vram_text = f"free_vram={free} GiB / {total} GiB on {device} (index={index})"
    else:
        reason = vram.get("error") if isinstance(vram, Mapping) else "unknown"
        vram_text = f"free_vram=unavailable ({reason})"
    if isinstance(threshold, Mapping):
        fractional = threshold.get("fractional_gib", "n/a")
        threshold_text = (
            f"threshold={threshold.get('min_free_gib')} GiB "
            f"(source={threshold.get('source')}; floor={threshold.get('absolute_floor_gib')} GiB; "
            f"fraction={threshold.get('fraction')}; fractional={fractional} GiB)"
        )
    else:
        threshold_text = "threshold=<unknown>"
    return f"holder={holder or '<empty>'}; {vram_text}; {threshold_text}"


def _threshold_payload(threshold: VramThreshold) -> dict[str, object]:
    payload: dict[str, object] = {
        "min_free_gib": round(threshold.min_free_gib, 3),
        "absolute_floor_gib": round(threshold.absolute_floor_gib, 3),
        "fraction": round(threshold.fraction, 6),
        "source": threshold.source,
    }
    if threshold.fractional_gib is not None:
        payload["fractional_gib"] = round(threshold.fractional_gib, 3)
    return payload


def run_nested_gpu_preflight(
    *,
    force: bool = False,
    environ: Mapping[str, str] | None = None,
    query_memory: Callable[[], VramSnapshot] = query_nvidia_smi_memory,
    holder_file: Path = HOLDER_FILE,
    lock_file: Path = LOCK_FILE,
    stderr: TextIO | None = None,
) -> dict[str, object]:
    """Fail closed unless the nested forecast owns the GPU lock and has headroom."""

    env = os.environ if environ is None else environ
    err = sys.stderr if stderr is None else stderr
    force = bool(force or force_gpu_run_enabled(env))
    lock = gpu_lock_status(env, holder_file=holder_file, lock_file=lock_file)

    snapshot: VramSnapshot | None = None
    memory_error: str | None = None
    try:
        snapshot = query_memory()
    except Exception as exc:  # noqa: BLE001 - surfaced in fail-closed payload
        memory_error = f"{type(exc).__name__}: {exc}"
    threshold = resolve_min_free_vram_threshold(
        env,
        total_gib=snapshot.total_gib if snapshot is not None else None,
    )

    payload: dict[str, object] = {
        "schema": "GpuwrfNestedGpuPreflight",
        "schema_version": 1,
        "status": "PASS",
        "forced": bool(force),
        "min_free_vram_gib": float(threshold.min_free_gib),
        "min_free_vram_fraction": float(threshold.fraction),
        "vram_threshold": _threshold_payload(threshold),
        "lock": lock,
        "vram": _snapshot_payload(snapshot, memory_error),
        "action": (
            "Run nested GPU forecasts through "
            "`scripts/with_gpu_lock.sh --label <name> -- <cmd>`; set "
            "GPUWRF_MIN_FREE_VRAM_GIB for an explicit GiB threshold override or "
            "GPUWRF_MIN_FREE_VRAM_FRACTION to tune the card-relative threshold; set "
            "GPUWRF_FORCE_GPU_RUN=1 or pass --force-gpu-run only for a deliberate override."
        ),
    }

    failures: list[str] = []
    if not bool(lock.get("ok")):
        failures.append(str(lock.get("reason") or "GPU lock is not held"))
    if snapshot is None:
        failures.append(f"could not query free VRAM before the run ({memory_error})")
    elif snapshot.free_gib < threshold.min_free_gib:
        failures.append(
            f"free VRAM {snapshot.free_gib:.2f} GiB is below "
            f"resolved threshold {threshold.min_free_gib:.2f} GiB ({threshold.source})"
        )

    if failures and not force:
        payload["status"] = "FAIL"
        payload["failures"] = failures
        message = "nested GPU preflight failed: " + "; ".join(failures)
        print(f"gpuwrf: {message} ({_format_status(payload)})", file=err)
        raise GpuPreflightError(message, payload)

    if failures and force:
        payload["status"] = "FORCED"
        payload["failures"] = failures
        print(f"gpuwrf: WARNING: forcing nested GPU run despite preflight failures: {'; '.join(failures)}", file=err)
    else:
        print(f"gpuwrf: nested GPU preflight PASS: {_format_status(payload)}", file=err)
    return payload


__all__ = [
    "DEFAULT_MIN_FREE_VRAM_GIB",
    "DEFAULT_MIN_FREE_VRAM_FRACTION",
    "GpuPreflightError",
    "VramSnapshot",
    "VramThreshold",
    "env_bool",
    "force_gpu_run_enabled",
    "gpu_lock_status",
    "min_free_vram_fraction",
    "min_free_vram_gib",
    "query_nvidia_smi_memory",
    "read_gpu_lock_holder",
    "resolve_min_free_vram_threshold",
    "run_nested_gpu_preflight",
    "select_nvidia_smi_snapshot",
]
