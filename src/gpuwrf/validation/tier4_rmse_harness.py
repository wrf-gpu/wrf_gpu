"""M7 Tier-4 RMSE corpus bridge harness.

This module is intentionally smaller than the M6 probtest tolerance freeze. It
audits the available Gen2 corpus and emits a bounded probationary status when
the caller explicitly opts into the non-operational N=5 bridge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np
from netCDF4 import Dataset

from gpuwrf.validation.tier4_probtest import DEFAULT_ENDING_CYCLE, DEFAULT_HELDOUT_CYCLE


DEFAULT_GEN2_ROOTS = (
    Path("/mnt/data/canairy_meteo/runs/wrf_l3"),
    Path("/mnt/data/canairy_meteo/runs/wrf_l2"),
)
DEFAULT_DOMAIN = "d02"
DEFAULT_PINNED_GRID_YX = (66, 159)
DEFAULT_VARIABLES = ("U10", "V10", "T2")
DEFAULT_LEADS_H = (1,)
OPERATIONAL_MEMBER_FLOOR = 10
NON_OPERATIONAL_MEMBER_FLOOR = 5

RUN_DIR_RE = re.compile(
    r"^(?P<cycle>\d{8}_\d{2}z)_(?P<level>l[23])_(?P<hours>\d+)h_(?P<created>\d{8}T\d{6}Z)$"
)
WRFOUT_RE = re.compile(
    r"^wrfout_(?P<domain>d\d{2})_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$"
)


@dataclass(frozen=True)
class CorpusMember:
    """One complete pinned-grid Gen2 member visible to the bridge harness."""

    run_id: str
    cycle: str
    created: str
    level: str
    advertised_hours: int
    path: Path
    domain: str
    history_count: int
    grid_shape_yx: tuple[int, int]

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["path"] = str(self.path)
        record["grid_shape_yx"] = list(self.grid_shape_yx)
        return record


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def _wrfout_files(path: Path, domain: str) -> list[Path]:
    files = sorted(path.glob(f"wrfout_{domain}_*"))
    return [item for item in files if WRFOUT_RE.match(item.name)]


def _grid_shape_yx(path: Path) -> tuple[int, int]:
    with Dataset(path, "r") as dataset:
        dims = dataset.dimensions
        return int(len(dims["south_north"])), int(len(dims["west_east"]))


def _member_from_dir(
    path: Path,
    *,
    domain: str,
    ending_cycle: str,
    heldout_cycle: str | None,
    required_leads_h: Sequence[int],
    pinned_grid_yx: tuple[int, int],
) -> CorpusMember | None:
    match = RUN_DIR_RE.match(path.name)
    if match is None or not path.is_dir():
        return None
    cycle = match.group("cycle")
    if cycle > ending_cycle:
        return None
    if heldout_cycle is not None and cycle == heldout_cycle:
        return None
    files = _wrfout_files(path, domain)
    required_history_count = max([0, *[int(item) for item in required_leads_h]]) + 1
    if len(files) < required_history_count or len(files) < 25:
        return None
    try:
        shape = _grid_shape_yx(files[0])
    except Exception:
        return None
    if shape != pinned_grid_yx:
        return None
    return CorpusMember(
        run_id=path.name,
        cycle=cycle,
        created=match.group("created"),
        level=match.group("level"),
        advertised_hours=int(match.group("hours")),
        path=path,
        domain=domain,
        history_count=len(files),
        grid_shape_yx=shape,
    )


def discover_corpus_members(
    roots: Iterable[str | Path] = DEFAULT_GEN2_ROOTS,
    *,
    domain: str = DEFAULT_DOMAIN,
    ending_cycle: str = DEFAULT_ENDING_CYCLE,
    heldout_cycle: str | None = DEFAULT_HELDOUT_CYCLE,
    required_leads_h: Sequence[int] = DEFAULT_LEADS_H,
    pinned_grid_yx: tuple[int, int] = DEFAULT_PINNED_GRID_YX,
) -> list[CorpusMember]:
    """Return complete pinned-grid members across the requested Gen2 roots."""

    members: list[CorpusMember] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for child in sorted(root_path.iterdir()):
            member = _member_from_dir(
                child,
                domain=domain,
                ending_cycle=ending_cycle,
                heldout_cycle=heldout_cycle,
                required_leads_h=required_leads_h,
                pinned_grid_yx=pinned_grid_yx,
            )
            if member is not None:
                members.append(member)
    return sorted(members, key=lambda item: (item.cycle, item.level, item.created, item.run_id))


def _load_surface(path: Path, variable: str) -> np.ndarray:
    with Dataset(path, "r") as dataset:
        netcdf_var = dataset.variables[variable]
        data = netcdf_var[0] if netcdf_var.dimensions and netcdf_var.dimensions[0] == "Time" else netcdf_var[:]
        return np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)


def build_rmse_records(
    members: Sequence[CorpusMember],
    *,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    leads_h: Sequence[int] = DEFAULT_LEADS_H,
    domain: str = DEFAULT_DOMAIN,
) -> list[dict[str, Any]]:
    """Build finite pairwise RMSE records against the first visible member.

    The bridge uses these records as a smoke skeleton only; they are not a
    production tolerance freeze and are marked as such in the final artifact.
    """

    if len(members) < 2:
        return []
    reference = members[0]
    reference_files = _wrfout_files(reference.path, domain)
    records: list[dict[str, Any]] = []
    for member in members[1:]:
        member_files = _wrfout_files(member.path, domain)
        for lead_h in leads_h:
            lead_index = int(lead_h)
            if lead_index >= len(reference_files) or lead_index >= len(member_files):
                continue
            for variable in variables:
                try:
                    ref = _load_surface(reference_files[lead_index], variable)
                    actual = _load_surface(member_files[lead_index], variable)
                except Exception:
                    continue
                if ref.shape != actual.shape:
                    continue
                valid = np.isfinite(ref) & np.isfinite(actual)
                if not bool(np.any(valid)):
                    continue
                diff = actual[valid] - ref[valid]
                records.append(
                    {
                        "record_kind": "pairwise_corpus_smoke",
                        "variable": variable,
                        "lead_h": lead_index,
                        "reference_run_id": reference.run_id,
                        "member_run_id": member.run_id,
                        "rmse": float(np.sqrt(np.mean(diff * diff))),
                        "bias": float(np.mean(diff)),
                        "finite": True,
                        "grid_cell_count": int(diff.size),
                    }
                )
    return records


def run_tier4_rmse_harness(
    *,
    roots: Iterable[str | Path] = DEFAULT_GEN2_ROOTS,
    output_path: str | Path | None = None,
    non_operational: bool = False,
    domain: str = DEFAULT_DOMAIN,
    ending_cycle: str = DEFAULT_ENDING_CYCLE,
    heldout_cycle: str | None = DEFAULT_HELDOUT_CYCLE,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    leads_h: Sequence[int] = DEFAULT_LEADS_H,
    pinned_grid_yx: tuple[int, int] = DEFAULT_PINNED_GRID_YX,
) -> dict[str, Any]:
    """Run the M7 corpus bridge audit and optionally write its JSON artifact."""

    required_count = NON_OPERATIONAL_MEMBER_FLOOR if non_operational else OPERATIONAL_MEMBER_FLOOR
    members = discover_corpus_members(
        roots,
        domain=domain,
        ending_cycle=ending_cycle,
        heldout_cycle=heldout_cycle,
        required_leads_h=leads_h,
        pinned_grid_yx=pinned_grid_yx,
    )
    member_count = len(members)
    needed = max(0, required_count - member_count)
    if member_count >= required_count:
        status = "PASS_PROBATIONARY" if non_operational else "PASS"
        corpus_gate = status
        message = f"corpus floor met with {member_count}/{required_count} members"
    elif non_operational:
        status = "PASS_PROBATIONARY_PENDING"
        corpus_gate = "BLOCKED_CORPUS"
        message = f"non-operational bridge needs +{needed} member(s) to reach N={required_count}"
    else:
        status = "BLOCKED_CORPUS"
        corpus_gate = "BLOCKED_CORPUS"
        message = f"operational Tier-4 floor needs +{needed} member(s) to reach N={required_count}"

    selected = members[-min(member_count, required_count) :] if members else []
    rmse_records = build_rmse_records(selected, variables=variables, leads_h=leads_h, domain=domain)
    payload: dict[str, Any] = {
        "artifact_type": "m7_tier4_rmse_corpus_bridge",
        "created_utc": utc_now_iso(),
        "status": status,
        "corpus_gate": corpus_gate,
        "message": message,
        "mode": "non_operational" if non_operational else "operational",
        "non_operational": bool(non_operational),
        "corpus_size_class": "bounded" if non_operational else "standard",
        "M7_close_class": "probationary" if non_operational else "operational",
        "required_member_count": int(required_count),
        "member_count": int(member_count),
        "needed_members": int(needed),
        "domain": domain,
        "ending_cycle": ending_cycle,
        "heldout_cycle_excluded": heldout_cycle,
        "pinned_grid_yx": list(pinned_grid_yx),
        "variables": list(variables),
        "leads_h": [int(item) for item in leads_h],
        "member_split": {
            "available_count": int(member_count),
            "selected_count": int(len(selected)),
            "available": [member.to_record() for member in members],
            "selected": [member.to_record() for member in selected],
        },
        "rmse_records": rmse_records,
        "finite_rmse_record_count": int(sum(1 for record in rmse_records if record.get("finite"))),
        "rmse_record_policy": (
            "pairwise corpus smoke against the first selected member; not a production M7 tolerance freeze"
        ),
    }
    if output_path is not None:
        write_json(output_path, payload)
    return payload


__all__ = [
    "DEFAULT_DOMAIN",
    "DEFAULT_GEN2_ROOTS",
    "DEFAULT_LEADS_H",
    "DEFAULT_PINNED_GRID_YX",
    "DEFAULT_VARIABLES",
    "NON_OPERATIONAL_MEMBER_FLOOR",
    "OPERATIONAL_MEMBER_FLOOR",
    "CorpusMember",
    "build_rmse_records",
    "discover_corpus_members",
    "run_tier4_rmse_harness",
    "write_json",
]
