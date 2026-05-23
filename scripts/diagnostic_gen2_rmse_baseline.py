#!/usr/bin/env python
"""Characterize Gen2 forecast-to-forecast RMSE noise floor.

This is an analysis-only sprint tool. It reads existing Gen2 WRF history files
from the read-only archive and writes a small CSV summary for the M6 Tier-4
threshold discussion.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
from netCDF4 import Dataset


DEFAULT_FIELDS = ("T2", "U10", "V10")
DEFAULT_LEADS = (24, 72)
RUN_RE = re.compile(
    r"^(?P<ymd>\d{8})_(?P<hour>\d{2})z_(?P<label>.+)_(?P<hours>\d+)h_(?P<created>\d{8}T\d{6}Z)$"
)
WRFOUT_RE = re.compile(
    r"^wrfout_(?P<domain>d\d{2})_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$"
)
THIN_RE = re.compile(r"^thin_gridded_(?P<domain>d\d{2})(?P<suffix>.*?)_v\d+\.nc$")


@dataclass(frozen=True)
class TimeRef:
    path: Path
    time_index: int | None = None
    source_kind: str = "wrfout"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    path: Path
    root: Path
    label: str | None
    advertised_hours: int | None
    created_utc: datetime | None
    init_time_utc: datetime | None
    valid_files: dict[datetime, TimeRef]

    @property
    def file_count(self) -> int:
        return len(self.valid_files)

    @property
    def observed_hours(self) -> int:
        if not self.valid_files:
            return 0
        times = sorted(self.valid_files)
        return int((times[-1] - times[0]).total_seconds() // 3600)

    @property
    def is_complete(self) -> bool:
        if self.advertised_hours is None:
            return False
        return self.file_count >= self.advertised_hours + 1 and self.observed_hours >= self.advertised_hours


@dataclass(frozen=True)
class Pair:
    lead_hours: int
    valid_time_utc: datetime
    lhs: RunRecord
    rhs: RunRecord
    lhs_path: TimeRef
    rhs_path: TimeRef
    rhs_lead_hours: int


@dataclass
class FieldResult:
    field: str
    lead_hours: int
    spatial_mean_rmse: float
    p95_rmse: float
    sample_pairs: int
    units: str
    notes: str
    heatmap: str | None = None
    spatial_summary: str | None = None


def parse_wrfout_time(path: Path, domain: str) -> datetime | None:
    match = WRFOUT_RE.match(path.name)
    if match is None or match.group("domain") != domain:
        return None
    return datetime.strptime(match.group("stamp"), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)


def decode_wrf_times(raw: np.ndarray) -> list[datetime]:
    times: list[datetime] = []
    for row in raw:
        chars: list[str] = []
        for item in row:
            value = item.item() if hasattr(item, "item") else item
            chars.append(value.decode("ascii") if isinstance(value, bytes) else str(value))
        stamp = "".join(chars).strip()
        times.append(datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc))
    return times


def parse_run_name(path: Path) -> tuple[str | None, int | None, datetime | None, datetime | None]:
    match = RUN_RE.match(path.name)
    if match is None:
        return None, None, None, None
    init = datetime.strptime(match.group("ymd") + match.group("hour"), "%Y%m%d%H").replace(tzinfo=timezone.utc)
    created = datetime.strptime(match.group("created"), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return match.group("label"), int(match.group("hours")), created, init


def discover_runs(root: Path, *, domain: str) -> list[RunRecord]:
    records: list[RunRecord] = []
    if not root.exists():
        return records
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label, advertised_hours, created_utc, parsed_init = parse_run_name(run_dir)
        valid_files: dict[datetime, TimeRef] = {}
        thin_candidates = sorted(
            path
            for path in run_dir.glob(f"thin_gridded_{domain}*.nc")
            if THIN_RE.match(path.name) and THIN_RE.match(path.name).group("domain") == domain
        )
        if thin_candidates:
            thin = thin_candidates[0]
            try:
                with Dataset(thin, "r") as dataset:
                    if "Times" in dataset.variables:
                        for index, valid_time in enumerate(decode_wrf_times(dataset.variables["Times"][:])):
                            valid_files[valid_time] = TimeRef(thin, index, "thin")
            except OSError:
                valid_files = {}
        if not valid_files:
            for wrfout in run_dir.glob(f"wrfout_{domain}_*"):
                valid_time = parse_wrfout_time(wrfout, domain)
                if valid_time is not None:
                    valid_files[valid_time] = TimeRef(wrfout, None, "wrfout")
        init_time = min(valid_files) if valid_files else parsed_init
        records.append(
            RunRecord(
                run_id=run_dir.name,
                path=run_dir,
                root=root,
                label=label,
                advertised_hours=advertised_hours,
                created_utc=created_utc,
                init_time_utc=init_time,
                valid_files=valid_files,
            )
        )
    return records


def canonical_by_init(records: Iterable[RunRecord], *, preferred_label: str | None) -> list[RunRecord]:
    grouped: dict[datetime, list[RunRecord]] = {}
    for record in records:
        if record.init_time_utc is None or not record.valid_files:
            continue
        grouped.setdefault(record.init_time_utc, []).append(record)

    selected: list[RunRecord] = []
    for init_time, candidates in grouped.items():
        def score(record: RunRecord) -> tuple[int, int, int, datetime]:
            exact_label = int(preferred_label is not None and record.label == preferred_label)
            complete = int(record.is_complete)
            created = record.created_utc or datetime.min.replace(tzinfo=timezone.utc)
            return complete, record.file_count, exact_label, created

        _ = init_time
        selected.append(max(candidates, key=score))
    return sorted(selected, key=lambda record: record.init_time_utc or datetime.max.replace(tzinfo=timezone.utc))


def preferred_label_for_root(root: Path) -> str | None:
    name = root.name
    if name.startswith("wrf_"):
        return name.removeprefix("wrf_")
    return None


def candidate_roots(gen2_root: Path) -> list[Path]:
    roots = [gen2_root]
    if gen2_root.name == "wrf_l3":
        sibling = gen2_root.parent / "wrf_l2"
        if sibling.exists():
            roots.append(sibling)
    return roots


def build_method_a_pairs(records: list[RunRecord], *, lead_hours: int) -> list[Pair]:
    by_init = {record.init_time_utc: record for record in records if record.init_time_utc is not None}
    pairs: list[Pair] = []
    for lhs in records:
        if lhs.init_time_utc is None:
            continue
        rhs = by_init.get(lhs.init_time_utc + timedelta(hours=24))
        if rhs is None or rhs.init_time_utc is None:
            continue
        valid_time = lhs.init_time_utc + timedelta(hours=lead_hours)
        lhs_path = lhs.valid_files.get(valid_time)
        rhs_path = rhs.valid_files.get(valid_time)
        if lhs_path is None or rhs_path is None:
            continue
        rhs_lead = int((valid_time - rhs.init_time_utc).total_seconds() // 3600)
        pairs.append(Pair(lead_hours, valid_time, lhs, rhs, lhs_path, rhs_path, rhs_lead))
    return pairs


def build_method_c_pairs(records: list[RunRecord], *, lead_hours: int) -> list[Pair]:
    grouped: dict[datetime, list[RunRecord]] = {}
    for record in records:
        if record.init_time_utc is not None and record.valid_files:
            grouped.setdefault(record.init_time_utc, []).append(record)
    pairs: list[Pair] = []
    for init_time, candidates in sorted(grouped.items()):
        if len(candidates) < 2:
            continue
        valid_time = init_time + timedelta(hours=lead_hours)
        available = [record for record in candidates if valid_time in record.valid_files]
        if len(available) < 2:
            continue
        lhs, rhs = sorted(available, key=lambda record: record.run_id)[:2]
        pairs.append(
            Pair(
                lead_hours,
                valid_time,
                lhs,
                rhs,
                lhs.valid_files[valid_time],
                rhs.valid_files[valid_time],
                lead_hours,
            )
        )
    return pairs


def read_field(ref: TimeRef, field: str) -> tuple[np.ndarray, str]:
    with Dataset(ref.path, "r") as dataset:
        if field not in dataset.variables:
            raise KeyError(f"{field} not present in {ref.path}")
        variable = dataset.variables[field]
        data = variable[:]
        if variable.dimensions and variable.dimensions[0] == "Time":
            data = data[ref.time_index or 0]
        units = str(getattr(variable, "units", "")).strip()
    array = np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)
    return array, normalize_units(units, field)


def read_static(ref: TimeRef) -> dict[str, np.ndarray]:
    static: dict[str, np.ndarray] = {}
    with Dataset(ref.path, "r") as dataset:
        for name in ("XLAT", "XLONG", "HGT", "LANDMASK"):
            if name not in dataset.variables:
                continue
            variable = dataset.variables[name]
            data = variable[:]
            if variable.dimensions and variable.dimensions[0] == "Time":
                data = data[ref.time_index or 0]
            static[name] = np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)
    return static


def read_wrfout_metadata(ref: TimeRef, fields: Iterable[str]) -> dict[str, object]:
    with Dataset(ref.path, "r") as dataset:
        variables = sorted(dataset.variables.keys())
        target_attrs = {}
        for field in fields:
            if field in dataset.variables:
                variable = dataset.variables[field]
                target_attrs[field] = {
                    "dimensions": list(variable.dimensions),
                    "shape": list(variable.shape),
                    "units": normalize_units(str(getattr(variable, "units", "")).strip(), field),
                    "description": str(getattr(variable, "description", "")).strip(),
                }
        dims = {name: len(dim) for name, dim in dataset.dimensions.items()}
    return {
        "dimension_summary": dims,
        "variable_count": len(variables),
        "first_variables": variables[:80],
        "target_attrs": target_attrs,
    }


def normalize_units(units: str, field: str) -> str:
    if field == "T2":
        return "K"
    if field in {"U10", "V10"}:
        return "m/s"
    return units


def block_reduce_mean(array: np.ndarray, *, rows: int, cols: int) -> np.ndarray:
    y_edges = np.linspace(0, array.shape[0], rows + 1, dtype=int)
    x_edges = np.linspace(0, array.shape[1], cols + 1, dtype=int)
    reduced = np.full((rows, cols), np.nan, dtype=np.float64)
    for y in range(rows):
        for x in range(cols):
            block = array[y_edges[y] : y_edges[y + 1], x_edges[x] : x_edges[x + 1]]
            if np.isfinite(block).any():
                reduced[y, x] = float(np.nanmean(block))
    return reduced


def ascii_heatmap(array: np.ndarray, *, rows: int = 12, cols: int = 24) -> str:
    reduced = block_reduce_mean(array, rows=rows, cols=cols)
    finite = reduced[np.isfinite(reduced)]
    if finite.size == 0:
        return "(no finite cells)"
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    chars = ".:-=+*#%@"
    lines: list[str] = []
    for row in reduced:
        line = []
        for value in row:
            if not np.isfinite(value):
                line.append("?")
            elif high <= low:
                line.append(chars[-1])
            else:
                idx = int(round((len(chars) - 1) * (float(value) - low) / (high - low)))
                line.append(chars[max(0, min(len(chars) - 1, idx))])
        lines.append("".join(line).rstrip())
    return "\n".join(lines)


def coastline_mask(landmask: np.ndarray) -> np.ndarray:
    land = np.asarray(landmask) >= 0.5
    coast = np.zeros(land.shape, dtype=bool)
    coast[1:, :] |= land[1:, :] != land[:-1, :]
    coast[:-1, :] |= land[:-1, :] != land[1:, :]
    coast[:, 1:] |= land[:, 1:] != land[:, :-1]
    coast[:, :-1] |= land[:, :-1] != land[:, 1:]
    return coast


def spatial_summary(cell_rmse: np.ndarray, static: dict[str, np.ndarray]) -> str:
    finite = np.isfinite(cell_rmse)
    if not finite.any():
        return "no finite RMSE cells"
    threshold = float(np.nanpercentile(cell_rmse, 95.0))
    top = finite & (cell_rmse >= threshold)
    ny, nx = cell_rmse.shape
    yy, xx = np.indices(cell_rmse.shape)
    boundary_width = max(3, min(ny, nx) // 10)
    boundary = (yy < boundary_width) | (yy >= ny - boundary_width) | (xx < boundary_width) | (xx >= nx - boundary_width)
    parts = [
        f"top5_threshold={threshold:.6g}",
        f"top5_boundary_fraction={np.mean(boundary[top]):.3f}",
    ]
    if "XLAT" in static and "XLONG" in static:
        parts.append(f"top5_mean_lat={np.nanmean(static['XLAT'][top]):.4f}")
        parts.append(f"top5_mean_lon={np.nanmean(static['XLONG'][top]):.4f}")
    if "LANDMASK" in static:
        landmask = static["LANDMASK"]
        parts.append(f"top5_land_fraction={np.nanmean((landmask[top] >= 0.5).astype(float)):.3f}")
        coast = coastline_mask(landmask)
        parts.append(f"top5_coastline_fraction={np.mean(coast[top]):.3f}")
    if "HGT" in static:
        hgt = static["HGT"]
        terrain_cut = float(np.nanpercentile(hgt[finite], 75.0))
        parts.append(f"top5_mean_hgt_m={np.nanmean(hgt[top]):.1f}")
        parts.append(f"top5_high_terrain_fraction={np.mean(hgt[top] >= terrain_cut):.3f}")
    return "; ".join(parts)


def compute_field_result(
    *,
    field: str,
    lead_hours: int,
    pairs: list[Pair],
    method: str,
    root: Path,
    domain: str,
) -> FieldResult:
    if not pairs:
        return FieldResult(
            field=field,
            lead_hours=lead_hours,
            spatial_mean_rmse=math.nan,
            p95_rmse=math.nan,
            sample_pairs=0,
            units=normalize_units("", field),
            notes=f"No {method} pairs available for {lead_hours}h lead; root={root}; domain={domain}",
        )

    cell_sumsq: np.ndarray | None = None
    cell_count: np.ndarray | None = None
    total_sumsq = 0.0
    total_count = 0
    units = normalize_units("", field)
    used_pairs = 0
    skipped_shape = 0
    expected_shape: tuple[int, ...] | None = None
    static_ref: TimeRef | None = None
    pair_rmse: list[float] = []

    for pair in pairs:
        lhs, lhs_units = read_field(pair.lhs_path, field)
        rhs, rhs_units = read_field(pair.rhs_path, field)
        units = lhs_units or rhs_units or units
        if lhs.shape != rhs.shape:
            skipped_shape += 1
            continue
        if expected_shape is None:
            expected_shape = lhs.shape
            static_ref = pair.lhs_path
        elif lhs.shape != expected_shape:
            skipped_shape += 1
            continue
        diff = lhs - rhs
        finite = np.isfinite(diff)
        sq = np.where(finite, diff * diff, 0.0)
        if cell_sumsq is None:
            cell_sumsq = np.zeros(lhs.shape, dtype=np.float64)
            cell_count = np.zeros(lhs.shape, dtype=np.int64)
        cell_sumsq += sq
        cell_count += finite.astype(np.int64)
        total_sumsq += float(np.sum(sq))
        total_count += int(np.sum(finite))
        pair_count = int(np.sum(finite))
        if pair_count:
            pair_rmse.append(math.sqrt(float(np.sum(sq)) / pair_count))
        used_pairs += 1

    if cell_sumsq is None or cell_count is None or static_ref is None:
        return FieldResult(
            field=field,
            lead_hours=lead_hours,
            spatial_mean_rmse=math.nan,
            p95_rmse=math.nan,
            sample_pairs=0,
            units=units,
            notes=(
                f"No same-grid {method} pairs available for {lead_hours}h lead; "
                f"root={root}; domain={domain}; skipped_shape={skipped_shape}"
            ),
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        cell_rmse = np.sqrt(cell_sumsq / cell_count)
    cell_rmse[cell_count == 0] = np.nan

    static = read_static(static_ref)
    time_summary = (
        f"pair_rmse_mean={np.mean(pair_rmse):.6g}; pair_rmse_std={np.std(pair_rmse):.6g}; "
        f"pair_rmse_min={np.min(pair_rmse):.6g}; pair_rmse_max={np.max(pair_rmse):.6g}"
        if pair_rmse
        else "pair_rmse_unavailable"
    )
    rhs_leads = sorted({pair.rhs_lead_hours for pair in pairs})
    notes = (
        f"{method} consecutive-day overlap; root={root.name}; domain={domain}; "
        f"lhs_lead={lead_hours}h; rhs_leads={rhs_leads}; "
        f"valid_times={pairs[0].valid_time_utc.isoformat()}..{pairs[-1].valid_time_utc.isoformat()}; "
        f"source_kind={static_ref.source_kind}; skipped_shape_pairs={skipped_shape}"
    )
    return FieldResult(
        field=field,
        lead_hours=lead_hours,
        spatial_mean_rmse=float(np.nanmean(cell_rmse)),
        p95_rmse=float(np.nanpercentile(cell_rmse, 95.0)),
        sample_pairs=used_pairs,
        units=units,
        notes=notes,
        heatmap=ascii_heatmap(cell_rmse),
        spatial_summary=spatial_summary(cell_rmse, static) + "; " + time_summary,
    )


def longest_consecutive_dates(records: Iterable[RunRecord]) -> list[str]:
    dates = sorted({record.init_time_utc.date() for record in records if record.init_time_utc is not None and record.valid_files})
    if not dates:
        return []
    best: list[object] = []
    current = [dates[0]]
    for date in dates[1:]:
        if date == current[-1] + timedelta(days=1):
            current.append(date)
        else:
            if len(current) > len(best):
                best = current
            current = [date]
    if len(current) > len(best):
        best = current
    return [date.isoformat() for date in best]


def inventory_lines(root: Path, records: list[RunRecord], *, domain: str) -> list[str]:
    usable = [record for record in records if record.valid_files]
    complete = [record for record in usable if record.is_complete]
    longest = longest_consecutive_dates(usable)
    lines = [
        f"Inventory root: {root}",
        f"Domain: {domain}",
        f"Run directories: {len(records)}",
        f"Runs with {domain} retained product: {len(usable)}",
        f"Complete advertised runs with {domain} retained product: {len(complete)}",
        "Longest consecutive usable init-date span: "
        + (f"{longest[0]}..{longest[-1]} ({len(longest)} days)" if longest else "none"),
        "Runs with retained products:",
    ]
    for record in usable:
        start = min(record.valid_files).isoformat()
        end = max(record.valid_files).isoformat()
        status = "complete" if record.is_complete else "partial"
        first_ref = next(iter(record.valid_files.values()))
        lines.append(
            f"  - {record.run_id}: files={record.file_count}, observed_hours={record.observed_hours}, "
            f"advertised_hours={record.advertised_hours}, {status}, source={first_ref.source_kind}, "
            f"file={first_ref.path.name}, valid={start}..{end}"
        )
    return lines


def select_root_and_pairs(
    roots_to_records: dict[Path, list[RunRecord]],
    *,
    lead_hours: int,
    method: str,
) -> tuple[Path, list[RunRecord], list[Pair]]:
    best: tuple[Path, list[RunRecord], list[Pair]] | None = None
    best_score: tuple[int, int, int] | None = None
    for root, all_records in roots_to_records.items():
        selected = canonical_by_init(all_records, preferred_label=preferred_label_for_root(root))
        pairs = build_method_c_pairs(selected, lead_hours=lead_hours) if method == "C" else build_method_a_pairs(selected, lead_hours=lead_hours)
        max_hours = max((record.advertised_hours or record.observed_hours for record in all_records), default=0)
        capable = int(max_hours >= lead_hours)
        complete_count = sum(1 for record in all_records if record.is_complete)
        score = (len(pairs), capable, complete_count)
        if best_score is None or score > best_score:
            best = (root, selected, pairs)
            best_score = score
    assert best is not None
    return best


def write_csv(path: Path, results: list[FieldResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["field", "lead_hours", "spatial_mean_rmse", "p95_rmse", "sample_pairs", "units", "notes"],
            lineterminator="\n",
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "field": result.field,
                    "lead_hours": result.lead_hours,
                    "spatial_mean_rmse": format_float(result.spatial_mean_rmse),
                    "p95_rmse": format_float(result.p95_rmse),
                    "sample_pairs": result.sample_pairs,
                    "units": result.units,
                    "notes": result.notes,
                }
            )


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.10g}"


def print_report(
    *,
    roots_to_records: dict[Path, list[RunRecord]],
    metadata: dict[str, object] | None,
    method: str,
    output: Path,
    results: list[FieldResult],
    pair_counts: dict[int, tuple[Path, list[Pair]]],
) -> None:
    print("Gen2 RMSE baseline characterization")
    print("Method requested:", method)
    print("Method used:", "A consecutive-day overlap" if method == "A" else "C same-valid rerun comparison")
    print()
    for root, records in roots_to_records.items():
        for line in inventory_lines(root, records, domain=args_domain()):
            print(line)
        print()

    if metadata is not None:
        print("WRF output metadata sample")
        print("Dimensions:", metadata["dimension_summary"])
        print("Variable count:", metadata["variable_count"])
        print("First variables:", ", ".join(metadata["first_variables"]))
        print("Target variable metadata:", metadata["target_attrs"])
        print("Output frequency: hourly for complete retained runs (valid-time deltas are 1h).")
        print(
            "File naming convention: <run-id>/wrfout_<domain>_YYYY-MM-DD_HH:MM:SS for raw retained "
            "history, or <run-id>/thin_gridded_<domain>_v1.nc for compact hourly retained products"
        )
        print("Run-id convention: YYYYMMDD_18z_l{2|3}_{24|72}h_YYYYMMDDTHHMMSSZ, with observed rerun variants.")
        print()

    print("Comparison pairs")
    for lead, (root, pairs) in pair_counts.items():
        print(f"Lead {lead}h source root: {root}")
        if not pairs:
            print("  no valid pairs")
        for pair in pairs:
            print(
                f"  {pair.valid_time_utc.isoformat()}: {pair.lhs.run_id} lead {lead}h "
                f"vs {pair.rhs.run_id} lead {pair.rhs_lead_hours}h"
            )
    print()

    print("RMSE summary")
    print("field,lead_hours,spatial_mean_rmse,p95_rmse,sample_pairs,units")
    for result in results:
        print(
            ",".join(
                [
                    result.field,
                    str(result.lead_hours),
                    format_float(result.spatial_mean_rmse),
                    format_float(result.p95_rmse),
                    str(result.sample_pairs),
                    result.units,
                ]
            )
        )
    print()

    for result in results:
        print(f"Spatial pattern: {result.field} {result.lead_hours}h")
        print(result.spatial_summary or "no spatial summary")
        print(result.heatmap or "(no heatmap)")
        print()

    print("Threshold recommendation")
    print(
        "Recommend RMSE < 2.0x Gen2 noise floor as the default M6 Tier-4 rejection threshold, "
        "because ADR-007 makes U10/V10/T2 operational RMSE binding and a 2x multiplier allows "
        "minor implementation differences while still rejecting large forecast-regime errors."
    )
    print(
        "Caveat: rows with sample_pairs=0 are not threshold anchors; rerun this diagnostic after "
        "at least 7 consecutive retained daily cycles and 72h overlaps are available."
    )
    print(f"CSV written: {output}")


_REPORT_DOMAIN = "d02"


def args_domain() -> str:
    return _REPORT_DOMAIN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen2-root", required=True, help="Gen2 wrf_l3 root; wrf_l2 sibling is scanned for 72h if present")
    parser.add_argument("--output", required=True, help="Small CSV output path")
    parser.add_argument("--method", choices=("A", "C"), default="A", help="A=consecutive-day overlap, C=same-valid rerun fallback")
    parser.add_argument("--domain", default="d02", help="WRF domain to analyze")
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS), help="Comma-separated WRF fields")
    parser.add_argument("--leads", default=",".join(str(lead) for lead in DEFAULT_LEADS), help="Comma-separated lead hours")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    global _REPORT_DOMAIN
    _REPORT_DOMAIN = args.domain

    root = Path(args.gen2_root).expanduser().resolve()
    output = Path(args.output)
    fields = tuple(field.strip() for field in args.fields.split(",") if field.strip())
    leads = tuple(int(item.strip()) for item in args.leads.split(",") if item.strip())
    roots = candidate_roots(root)
    roots_to_records = {candidate: discover_runs(candidate, domain=args.domain) for candidate in roots}

    results: list[FieldResult] = []
    pair_counts: dict[int, tuple[Path, list[Pair]]] = {}
    selected_metadata_ref: TimeRef | None = None
    for lead in leads:
        selected_root, _selected_records, pairs = select_root_and_pairs(
            roots_to_records,
            lead_hours=lead,
            method=args.method,
        )
        pair_counts[lead] = (selected_root, pairs)
        if selected_metadata_ref is None and pairs:
            selected_metadata_ref = pairs[0].lhs_path
        for field in fields:
            results.append(
                compute_field_result(
                    field=field,
                    lead_hours=lead,
                    pairs=pairs,
                    method=f"Method {args.method}",
                    root=selected_root,
                    domain=args.domain,
                )
            )

    metadata = read_wrfout_metadata(selected_metadata_ref, fields) if selected_metadata_ref is not None else None

    write_csv(output, results)
    print_report(
        roots_to_records=roots_to_records,
        metadata=metadata,
        method=args.method,
        output=output,
        results=results,
        pair_counts=pair_counts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
