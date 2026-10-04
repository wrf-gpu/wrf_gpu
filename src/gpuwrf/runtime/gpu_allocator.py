"""Case-sized CLI CUDA pools; importing this module never initializes JAX."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

GIB = 1024**3
SCHEMA = "gpuwrf-cli-memory-plan-v1"
SAFETY_FACTOR = 1.20
HEADROOM_BYTES = GIB // 2
_SESSION: dict[str, Any] | None = None


def case_key(source: str, flags: dict, geometry: dict) -> str:
    payload = json.dumps([SCHEMA, source, flags, geometry], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def memory_record(executable: Any) -> dict[str, int]:
    """Read compiled metadata only; no device execution or synchronization."""
    loaded = getattr(executable, "loaded_executable", None)
    stats = (loaded.get_compiled_memory_stats() if loaded is not None
             else executable.memory_analysis())
    fields = ("argument_size_in_bytes", "output_size_in_bytes",
              "temp_size_in_bytes", "alias_size_in_bytes", "generated_code_size_in_bytes")
    result = {name: int(getattr(stats, name)) for name in fields}
    if any(value < 0 for value in result.values()):
        raise ValueError("negative compiled memory statistic")
    return result


def speculative_allowance(programs: dict[str, list[dict]]) -> int:
    """Unused AOT prefetch residency: at most one speculative variant per domain.

    A mismatched prefetch is another variant of the same domain program, so its
    generated code is bounded by that domain's largest record. GPU statistics
    carry no constant sizes (about 0.1 GB per domain); the margin covers them.
    """
    return sum(max(int(record["generated_code_size_in_bytes"]) for record in variants)
               for variants in programs.values())


def pool_budget(programs: dict[str, list[dict]], live_peak_bytes: int) -> int:
    """Conservative sum of domain footprints, with measured live peak as a floor.

    Variants of one domain are alternatives; separate domains retain their
    carries/executables. This bound deliberately avoids adding all variants.
    """
    if not programs or any(not variants for variants in programs.values()):
        raise ValueError("incomplete domain memory records")
    total = 0
    for variants in programs.values():
        footprints = []
        for record in variants:
            footprint = sum(int(record[key]) for key in
                            ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes"))
            footprint -= int(record["alias_size_in_bytes"])
            if footprint <= 0:
                raise ValueError("invalid executable memory footprint")
            footprints.append(footprint)
        total += max(footprints)
    peak = max(total, int(live_peak_bytes)) + speculative_allowance(programs)
    return math.ceil(peak * SAFETY_FACTOR) + HEADROOM_BYTES


def record_executable(domain: str, executable: Any, identity: str) -> None:
    """One telemetry hook, active only inside a product CLI memory session."""
    if _SESSION is None:
        return
    try:
        record = memory_record(executable)
        _SESSION["programs"].setdefault(str(domain), {})[str(identity)] = record
    except Exception:  # optional telemetry must never change execution behavior
        _SESSION["incomplete"] = True


def write_plan(path: Path, key: str, domains: int, programs: dict, live_peak_bytes: int) -> dict:
    if len(programs) != domains:
        raise ValueError("not all case domains supplied compiled memory statistics")
    variants = {domain: list(records.values()) for domain, records in programs.items()}
    plan = {"schema": SCHEMA, "case_key": key, "domains": domains,
            "programs": variants, "live_peak_bytes": int(live_peak_bytes),
            "speculative_bytes": speculative_allowance(variants),
            "budget_bytes": pool_budget(variants, live_peak_bytes)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.{os.getpid()}.new")
    try:
        temporary.write_text(json.dumps(plan, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return plan


def read_plan(path: Path, key: str, domains: int) -> dict:
    plan = json.loads(path.read_text())
    if plan.get("schema") != SCHEMA or plan.get("case_key") != key or plan.get("domains") != domains:
        raise ValueError("stale case memory plan")
    budget = pool_budget(plan["programs"], plan["live_peak_bytes"])
    if plan["budget_bytes"] != budget or len(plan["programs"]) != domains:
        raise ValueError("incomplete or inconsistent case memory plan")
    return plan


def outside_headroom(plan: dict) -> int:
    """CUDA context, modules and resident code outside the pool (VR19: 824 MiB)."""
    code = sum(int(r["generated_code_size_in_bytes"]) for v in plan["programs"].values() for r in v)
    return max(GIB, HEADROOM_BYTES + code)


def planned_fraction(plan: dict, total_bytes: int, free_bytes: int) -> str:
    budget = int(plan["budget_bytes"])
    if total_bytes <= 0 or budget + outside_headroom(plan) > free_bytes:
        raise ValueError("case pool plus outside-pool headroom exceeds free GPU memory")
    fraction = math.ceil(budget / total_bytes * 1000000) / 1000000
    if not 0 < fraction < 1:
        raise ValueError("case pool exceeds GPU capacity")
    return f"{fraction:.6f}"


def _case_description(args: Any) -> tuple[dict, int]:
    # Header/namelist reads only: no arrays are materialized on a device.
    from gpuwrf.io.netcdf_lock import Dataset
    directory = Path(args.input_dir)
    namelist_path = Path(getattr(args, "namelist", None) or directory / "namelist.input")
    # Importing the full input accessor initializes JAX through its contracts.
    # Storage planning needs only max_dom and a conservative case-setting digest.
    text = namelist_path.read_text()
    match = re.search(r"(?mi)^\s*max_dom\s*=\s*(\d+)", text)
    domains = int(getattr(args, "max_dom", None) or (match.group(1) if match else 1))
    geometry = {}
    for index in range(1, domains + 1):
        name = f"d{index:02d}"
        with Dataset(directory / f"wrfinput_{name}") as dataset:
            geometry[name] = {"dimensions": {key: len(dim) for key, dim in dataset.dimensions.items()},
                              "dx": float(dataset.DX), "dy": float(dataset.DY)}
    # Dates change data, not storage. Other case settings remain load-bearing.
    settings = "\n".join(line for line in text.splitlines() if not re.match(
        r"\s*(?:start_|end_|run_)[A-Za-z_]+\s*=", line, re.IGNORECASE))
    return {"domains": geometry, "namelist_settings_sha256": hashlib.sha256(settings.encode()).hexdigest(),
            "feedback": bool(getattr(args, "feedback", False)),
            "initial_history": bool(getattr(args, "emit_initial_history", False))}, domains


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _pool_device_memory() -> tuple[int, int]:
    """(CUDA-visible total, free) bytes of the first visible GPU via NVML only.

    No CUDA context is created. CUDA's total equals NVML total minus the driver
    reservation, which is the denominator JAX applies the fraction to.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    result = subprocess.run(["nvidia-smi", "-i", visible,
                             "--query-gpu=memory.total,memory.reserved,memory.free",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("ambiguous GPU memory query")
    total, reserved, free = (int(value.strip()) * 1024**2 for value in lines[0].split(","))
    return total - reserved, free


def configure_cli_pool(args: Any, env: dict[str, str]) -> bool:
    """Set CLI defaults before re-exec; explicit operator settings win.

    Return whether allocator settings changed. A fresh/incompatible case uses
    demand allocation until its successful run provides a verified plan.
    """
    global _SESSION
    before = dict(env)
    requested = env.get("GPUWRF_ALLOCATOR", "cuda_async").strip().lower() or "cuda_async"
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "default" if requested == "bfc" else requested)
    try:
        from gpuwrf.runtime.aot_cheap_key import is_process_infra_env  # shared with the AOT cheap key
        geometry, domains = _case_description(args)
        flags = {key: value for key, value in env.items() if key.startswith("GPUWRF_")
                 and not is_process_infra_env(key)
                 and not key.startswith(("GPUWRF_GPU_LOCK", "GPUWRF_MIN_FREE", "GPUWRF_KEEP", "GPUWRF_TMP", "GPUWRF_SCRATCH"))
                 and not key.endswith("_DIR") and key != "GPUWRF_ALLOCATOR"}
        flags["XLA_FLAGS"] = env.get("XLA_FLAGS", "")
        flags["JAX_ENABLE_X64"] = env.get("JAX_ENABLE_X64", "")
        flags["jax_version"] = importlib.metadata.version("jax")
        flags["jaxlib_version"] = importlib.metadata.version("jaxlib")
        flags["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
        key = case_key(_source_fingerprint(), flags, geometry)
        cache = Path(env.get("GPUWRF_JAX_CACHE_DIR") or env.get("JAX_COMPILATION_CACHE_DIR")
                     or Path.home() / ".cache/gpuwrf/jit")
        path = cache / "cli_memory_plans" / (key + ".json")
        _SESSION = {"key": key, "path": path, "domains": domains, "programs": {}, "incomplete": False}
        if env.get("_GPUWRF_C_AUTO_BUDGET"):
            _SESSION["applied_budget"] = int(env["_GPUWRF_C_AUTO_BUDGET"])
    except (OSError, ValueError, KeyError, AttributeError, TypeError, ImportError) as exc:
        _SESSION = None
        path = None
        reason = f"case memory metadata unavailable ({type(exc).__name__})"
    owned = env.get("_GPUWRF_C_AUTO") == "1"
    explicit = not owned and any(env.get(key) for key in
        ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION"))
    if env["XLA_PYTHON_CLIENT_ALLOCATOR"] != "cuda_async" or explicit:
        return env != before
    # The one-shot re-exec already carries the chosen defaults. Keep them and
    # recreate only the recording session in the fresh process.
    if owned:
        return env != before
    try:
        if path is None:
            raise ValueError(reason)
        plan = read_plan(path, key, domains)
        total, free = _pool_device_memory()
        fraction = planned_fraction(plan, total, free)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        env["XLA_CLIENT_MEM_FRACTION"] = fraction
        env["_GPUWRF_C_AUTO_BUDGET"] = str(plan["budget_bytes"])
        env["_GPUWRF_C_AUTO_HEADROOM"] = str(outside_headroom(plan))  # nested GPU preflight
        _SESSION["applied_budget"] = plan["budget_bytes"]
        message = f"C-auto case pool {plan['budget_bytes'] / GIB:.2f} GiB, fraction={fraction}"
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env.pop("_GPUWRF_C_AUTO_BUDGET", None)
        env.pop("_GPUWRF_C_AUTO_HEADROOM", None)
        message = f"C-auto unavailable ({exc}); using demand allocation and recording this case for later runs"
    env["_GPUWRF_C_AUTO"] = "1"
    print("gpuwrf: " + message, file=sys.stderr)
    return env != before


def finish_cli_pool() -> None:
    """Save only complete, successful real-forecast statistics."""
    if _SESSION is None:
        return
    try:
        import jax
        device = jax.devices()[0]
        if device.platform != "gpu":
            return
        live = int(device.memory_stats()["peak_bytes_in_use"])
        applied = _SESSION.get("applied_budget")
        if applied and live > applied:  # soft threshold: the excess was re-mapped every sync
            print(f"gpuwrf: warning: live peak {live / GIB:.2f} GiB exceeded the C-auto pool "
                  f"{applied / GIB:.2f} GiB; refreshing this case plan", file=sys.stderr)
            _SESSION["path"].unlink(missing_ok=True)
        if not _SESSION["incomplete"]:
            write_plan(_SESSION["path"], _SESSION["key"], _SESSION["domains"], _SESSION["programs"], live)
    except Exception:  # optional plan persistence must never fail a forecast
        # An unavailable plan means a future demand fallback, never a guessed cap.
        pass


def failed_cli_pool(error: Exception) -> None:
    """Discard an insufficient estimate; never restart over existing output."""
    if _SESSION is None or not _SESSION.get("applied_budget"):
        return
    text = str(error).lower()
    if not any(marker in text for marker in ("out of memory", "out_of_memory", "resource_exhausted")):
        return
    try:
        _SESSION["path"].unlink(missing_ok=True)
    except OSError:
        pass
    print("gpuwrf: C-auto case estimate was exceeded; saved plan invalidated. "
          "The next CLI launch uses demand allocation to record a fresh plan. "
          "Check other GPU users and available memory before restarting.", file=sys.stderr)
