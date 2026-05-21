"""Gen2 d02 WRF history inventory helpers for M6.5-D1."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
from netCDF4 import Dataset

from gpuwrf.io.gen2_accessor import GEN2_READ_ONLY_ROOT


DEFAULT_GEN2_WRF_L3_ROOT = GEN2_READ_ONLY_ROOT / "runs" / "wrf_l3"
DEFAULT_DOMAIN = "d02"
DEFAULT_COMPLETE_MIN_HOURS = 24
WRFOUT_TIME_RE = re.compile(
    r"^wrfout_(?P<domain>d\d{2})_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$"
)
RUN_ID_RE = re.compile(
    r"^(?P<ymd>\d{8})_(?P<hour>\d{2})z_l(?P<level>\d+)_(?P<hours>\d+)h_(?P<created>\d{8}T\d{6}Z)$"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def reject_gen2_write_target(path: str | Path) -> None:
    """Reject writes into the read-only Gen2 data domain."""

    target = Path(path).expanduser()
    if _is_under(target, GEN2_READ_ONLY_ROOT):
        raise PermissionError(f"refusing to write inside read-only Gen2 data domain: {target}")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    reject_gen2_write_target(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_run_id(run_id: str) -> dict[str, Any]:
    match = RUN_ID_RE.match(run_id)
    if match is None:
        return {
            "start_date": None,
            "cycle_hour_utc": None,
            "forecast_hours_advertised": None,
            "created_utc": None,
        }
    ymd = match.group("ymd")
    hour = int(match.group("hour"))
    start = datetime.strptime(f"{ymd}{hour:02d}", "%Y%m%d%H").replace(tzinfo=timezone.utc)
    return {
        "start_date": start.date().isoformat(),
        "cycle_hour_utc": hour,
        "forecast_hours_advertised": int(match.group("hours")),
        "created_utc": datetime.strptime(match.group("created"), "%Y%m%dT%H%M%SZ")
        .replace(tzinfo=timezone.utc)
        .isoformat(),
    }


def parse_wrfout_valid_time(path: str | Path) -> datetime:
    match = WRFOUT_TIME_RE.match(Path(path).name)
    if match is None:
        raise ValueError(f"not a WRF history filename with valid time: {path}")
    return datetime.strptime(match.group("stamp"), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)


def _decode_times_variable(dataset: Dataset) -> list[str]:
    if "Times" not in dataset.variables:
        return []
    raw = np.asarray(dataset.variables["Times"][:])
    decoded: list[str] = []
    for row in raw:
        if row.dtype.kind == "S":
            decoded.append(b"".join(row.tolist()).decode("ascii", errors="replace").strip())
        else:
            decoded.append("".join(row.astype(str).tolist()).strip())
    return decoded


def _dataset_metadata(path: Path) -> dict[str, Any]:
    with Dataset(path, "r") as dataset:
        dims = {name: int(len(dim)) for name, dim in dataset.dimensions.items()}
        variables = sorted(dataset.variables.keys())
        attrs = {}
        for name in ("DX", "DY", "MAP_PROJ", "CEN_LAT", "CEN_LON", "TRUELAT1", "TRUELAT2", "STAND_LON"):
            if hasattr(dataset, name):
                value = getattr(dataset, name)
                attrs[name] = value.item() if hasattr(value, "item") else value
        return {
            "dimensions": dims,
            "variables_present": variables,
            "times_variable": _decode_times_variable(dataset),
            "global_attributes": attrs,
        }


def discover_gen2_run_dirs(root: str | Path = DEFAULT_GEN2_WRF_L3_ROOT) -> list[Path]:
    """Return run directories under the Gen2 wrf_l3 root.

    The primary inventory count follows the existing Gen2 run marker
    `wrfbdy_d01`, so runs with zero retained d02 history are still represented.
    Direct child directories without that marker are included as a fallback.
    """

    base = Path(root)
    marker_dirs = {path.parent for path in base.glob("*/wrfbdy_d01") if path.is_file()}
    direct_dirs = {path for path in base.iterdir() if path.is_dir()} if base.exists() else set()
    return sorted(marker_dirs | direct_dirs, key=lambda path: path.name)


def expected_valid_times(init_time: datetime | None, hours: int | None) -> list[datetime]:
    if init_time is None or hours is None:
        return []
    return [init_time + timedelta(hours=hour) for hour in range(int(hours) + 1)]


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def inventory_run(
    run_dir: str | Path,
    *,
    domain: str = DEFAULT_DOMAIN,
    complete_min_hours: int = DEFAULT_COMPLETE_MIN_HOURS,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    parsed = parse_run_id(run_path.name)
    files = sorted(run_path.glob(f"wrfout_{domain}_*"), key=lambda path: path.name)
    total_bytes = sum(path.stat().st_size for path in files if path.is_file())
    valid_times: list[datetime] = []
    parse_errors: list[str] = []
    for path in files:
        try:
            valid_times.append(parse_wrfout_valid_time(path))
        except ValueError as exc:
            parse_errors.append(str(exc))
    valid_times = sorted(valid_times)

    init_time: datetime | None
    if valid_times:
        init_time = valid_times[0]
    elif parsed["start_date"] is not None and parsed["cycle_hour_utc"] is not None:
        init_time = datetime.strptime(
            f"{parsed['start_date']}T{parsed['cycle_hour_utc']:02d}:00:00+0000",
            "%Y-%m-%dT%H:%M:%S%z",
        )
    else:
        init_time = None

    advertised_hours = parsed["forecast_hours_advertised"]
    if advertised_hours is None and valid_times:
        advertised_hours = int((valid_times[-1] - valid_times[0]).total_seconds() // 3600)
    expected_times = expected_valid_times(init_time, advertised_hours)
    observed = set(valid_times)
    missing_times = [time for time in expected_times if time not in observed]
    observed_hours = 0
    if valid_times:
        observed_hours = int((valid_times[-1] - valid_times[0]).total_seconds() // 3600)

    metadata: dict[str, Any] = {}
    metadata_error: str | None = None
    if files:
        try:
            metadata = _dataset_metadata(files[0])
        except Exception as exc:  # pragma: no cover - depends on corrupt external files.
            metadata_error = f"{type(exc).__name__}: {exc}"

    complete_by_hours = observed_hours >= int(complete_min_hours)
    complete_by_count = advertised_hours is None or len(missing_times) == 0
    complete = bool(files and complete_by_hours and complete_by_count and not parse_errors and metadata_error is None)
    return {
        "run_id": run_path.name,
        "run_path": str(run_path),
        "domain": domain,
        "start_date": parsed["start_date"],
        "cycle_hour_utc": parsed["cycle_hour_utc"],
        "hours": int(advertised_hours) if advertised_hours is not None else int(observed_hours),
        "observed_hours": int(observed_hours),
        "d02_wrfout_file_count": int(len(files)),
        "expected_d02_wrfout_file_count": int(advertised_hours + 1) if advertised_hours is not None else None,
        "total_bytes": int(total_bytes),
        "complete": complete,
        "complete_or_partial": "complete" if complete else "partial",
        "init_time_utc": _iso_or_none(init_time),
        "valid_time_range": {
            "start": _iso_or_none(valid_times[0] if valid_times else None),
            "end": _iso_or_none(valid_times[-1] if valid_times else None),
        },
        "valid_times_utc": [time.isoformat() for time in valid_times],
        "missing_time_step_count": int(len(missing_times)),
        "missing_valid_times_utc": [time.isoformat() for time in missing_times],
        "first_file": files[0].name if files else None,
        "last_file": files[-1].name if files else None,
        "files": [
            {
                "name": path.name,
                "path": str(path),
                "valid_time_utc": parse_wrfout_valid_time(path).isoformat(),
                "size_bytes": int(path.stat().st_size),
            }
            for path in files
            if WRFOUT_TIME_RE.match(path.name)
        ],
        "metadata": metadata,
        "parse_errors": parse_errors,
        "metadata_error": metadata_error,
    }


def build_gen2_d02_inventory(
    root: str | Path = DEFAULT_GEN2_WRF_L3_ROOT,
    *,
    domain: str = DEFAULT_DOMAIN,
    complete_min_hours: int = DEFAULT_COMPLETE_MIN_HOURS,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    run_dirs = discover_gen2_run_dirs(root)
    runs = [
        inventory_run(run_dir, domain=domain, complete_min_hours=complete_min_hours)
        for run_dir in run_dirs
    ]
    inventory = {
        "schema": "Gen2D02Inventory",
        "schema_version": 1,
        "generated_utc": generated_utc or utc_now_iso(),
        "root": str(Path(root)),
        "domain": domain,
        "complete_min_hours": int(complete_min_hours),
        "run_count": int(len(runs)),
        "complete_run_count": int(sum(1 for run in runs if run["complete"])),
        "partial_run_count": int(sum(1 for run in runs if not run["complete"])),
        "wrfbdy_d01_run_marker_count": int(sum(1 for run_dir in run_dirs if (run_dir / "wrfbdy_d01").is_file())),
        "wrfout_d02_file_count": int(sum(run["d02_wrfout_file_count"] for run in runs)),
        "total_bytes": int(sum(run["total_bytes"] for run in runs)),
        "runs": runs,
    }
    validate_gen2_d02_inventory(inventory)
    return inventory


def _assert_type(schema: str, field: str, value: Any, expected: type | tuple[type, ...]) -> None:
    if not isinstance(value, expected):
        names = expected if isinstance(expected, tuple) else (expected,)
        raise TypeError(f"{schema}.{field} expected {', '.join(item.__name__ for item in names)}, got {type(value).__name__}")


def validate_gen2_d02_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    schema = "Gen2D02Inventory"
    required = {
        "schema": str,
        "schema_version": int,
        "generated_utc": str,
        "root": str,
        "domain": str,
        "complete_min_hours": int,
        "run_count": int,
        "complete_run_count": int,
        "partial_run_count": int,
        "wrfout_d02_file_count": int,
        "total_bytes": int,
        "runs": list,
    }
    for field, expected in required.items():
        if field not in inventory:
            raise ValueError(f"{schema} missing required field {field!r}")
        _assert_type(schema, field, inventory[field], expected)
    if inventory["schema"] != schema:
        raise ValueError(f"inventory schema must be {schema!r}")
    if inventory["run_count"] != len(inventory["runs"]):
        raise ValueError("Gen2D02Inventory.run_count does not match runs length")
    if inventory["wrfout_d02_file_count"] != sum(run["d02_wrfout_file_count"] for run in inventory["runs"]):
        raise ValueError("Gen2D02Inventory.wrfout_d02_file_count does not match run records")
    if inventory["complete_run_count"] != sum(1 for run in inventory["runs"] if run["complete"]):
        raise ValueError("Gen2D02Inventory.complete_run_count does not match run records")
    for run in inventory["runs"]:
        for field in ("run_id", "run_path", "start_date", "hours", "d02_wrfout_file_count", "total_bytes", "complete_or_partial", "init_time_utc", "valid_time_range"):
            if field not in run:
                raise ValueError(f"Gen2D02Inventory run {run.get('run_id', '<unknown>')} missing {field!r}")
    return inventory


def load_inventory(path: str | Path) -> dict[str, Any]:
    return validate_gen2_d02_inventory(json.loads(Path(path).read_text(encoding="utf-8")))


def _parse_yyyymmdd(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)


def _run_start_datetime(run: dict[str, Any]) -> datetime | None:
    if run.get("init_time_utc"):
        return datetime.fromisoformat(run["init_time_utc"])
    if run.get("start_date") and run.get("cycle_hour_utc") is not None:
        return datetime.strptime(f"{run['start_date']} {int(run['cycle_hour_utc']):02d}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc)
    return None


def filter_inventory_runs(
    inventory: dict[str, Any],
    *,
    start: str | None = None,
    end: str | None = None,
    min_hours: int = DEFAULT_COMPLETE_MIN_HOURS,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    validate_gen2_d02_inventory(inventory)
    start_dt = _parse_yyyymmdd(start)
    end_dt = _parse_yyyymmdd(end)
    if end_dt is not None:
        end_dt = end_dt + timedelta(days=1)
    selected: list[dict[str, Any]] = []
    for run in inventory["runs"]:
        run_start = _run_start_datetime(run)
        if run_start is None:
            continue
        if start_dt is not None and run_start < start_dt:
            continue
        if end_dt is not None and run_start >= end_dt:
            continue
        if int(run.get("observed_hours", 0)) < int(min_hours):
            continue
        if require_complete and not bool(run["complete"]):
            continue
        selected.append(run)
    return selected


def build_subset_manifest(
    inventory: dict[str, Any],
    *,
    start: str | None = None,
    end: str | None = None,
    min_hours: int = DEFAULT_COMPLETE_MIN_HOURS,
    tag: str = "selected",
    require_complete: bool = True,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    runs = filter_inventory_runs(
        inventory,
        start=start,
        end=end,
        min_hours=min_hours,
        require_complete=require_complete,
    )
    manifest = {
        "schema": "Gen2D02SubsetManifest",
        "schema_version": 1,
        "generated_utc": generated_utc or utc_now_iso(),
        "source_inventory_schema": inventory["schema"],
        "source_inventory_root": inventory["root"],
        "tag": tag,
        "filters": {
            "start": start,
            "end": end,
            "min_hours": int(min_hours),
            "require_complete": bool(require_complete),
        },
        "run_count": int(len(runs)),
        "wrfout_d02_file_count": int(sum(run["d02_wrfout_file_count"] for run in runs)),
        "total_bytes": int(sum(run["total_bytes"] for run in runs)),
        "runs": runs,
    }
    validate_subset_manifest(manifest)
    return manifest


def validate_subset_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    for field in ("schema", "schema_version", "generated_utc", "tag", "filters", "run_count", "wrfout_d02_file_count", "total_bytes", "runs"):
        if field not in manifest:
            raise ValueError(f"Gen2D02SubsetManifest missing required field {field!r}")
    if manifest["schema"] != "Gen2D02SubsetManifest":
        raise ValueError("subset manifest schema must be 'Gen2D02SubsetManifest'")
    if manifest["run_count"] != len(manifest["runs"]):
        raise ValueError("subset run_count does not match runs length")
    return manifest


def iter_complete_runs(inventory: dict[str, Any], *, min_hours: int = DEFAULT_COMPLETE_MIN_HOURS) -> Iterable[dict[str, Any]]:
    for run in inventory["runs"]:
        if run["complete"] and int(run.get("observed_hours", 0)) >= int(min_hours):
            yield run


__all__ = [
    "DEFAULT_COMPLETE_MIN_HOURS",
    "DEFAULT_DOMAIN",
    "DEFAULT_GEN2_WRF_L3_ROOT",
    "build_gen2_d02_inventory",
    "build_subset_manifest",
    "discover_gen2_run_dirs",
    "filter_inventory_runs",
    "inventory_run",
    "iter_complete_runs",
    "load_inventory",
    "parse_run_id",
    "parse_wrfout_valid_time",
    "reject_gen2_write_target",
    "utc_now_iso",
    "validate_gen2_d02_inventory",
    "validate_subset_manifest",
    "write_json",
]
