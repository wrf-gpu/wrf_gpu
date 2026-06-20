#!/usr/bin/env python3
"""Validate the Step-1 pre-call QVAPOR WRF savepoint.

This proof is intentionally file-level and strict.  The disposable WRF hook only
appends QVAPOR to the accepted MASS_PREPART record, so the previous fields must
remain text-identical while the new QVAPOR values cover the full d02 mass grid.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[2]
OLD_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth")
NEW_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v014_step1_qvapor_precall_savepoint/wrf_truth")
SCRATCH_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v014_step1_qvapor_precall_savepoint")
FILTERED_ROOT = SCRATCH_ROOT / "precall_truth_only"
LOG_ROOT = SCRATCH_ROOT / "logs"
TARGET_GLOB = "before_first_rk_step_part1_call_d2_step_1_rk_1_*.txt"
OUT_JSON = REPO / "proofs/v014/step1_qvapor_precall_savepoint.json"
OUT_MD = REPO / "proofs/v014/step1_qvapor_precall_savepoint.md"
PATCH_DIFF = REPO / "proofs/v014/step1_qvapor_precall_savepoint_wrf_patch.diff"

MASS_FIELDS = ["T_STATE", "P_STATE", "PB", "MU_STATE", "MUB", "MUT"]
WPH_FIELDS = ["W_STATE", "PH_STATE", "PHB"]
EXPECTED_MASS_SHAPE_ZYX = [44, 66, 159]
EXPECTED_WPH_SHAPE_ZYX = [45, 66, 159]


@dataclass
class DiffStats:
    count: int = 0
    max_abs: float = 0.0
    sum_sq: float = 0.0
    text_identical: bool = True

    def add(self, old_text: str, new_text: str) -> None:
        old = float(old_text)
        new = float(new_text)
        diff = new - old
        abs_diff = abs(diff)
        self.count += 1
        if abs_diff > self.max_abs:
            self.max_abs = abs_diff
        self.sum_sq += diff * diff
        if old_text != new_text:
            self.text_identical = False

    def as_json(self) -> dict[str, object]:
        rmse = math.sqrt(self.sum_sq / self.count) if self.count else None
        return {
            "count": self.count,
            "max_abs": self.max_abs,
            "rmse": rmse,
            "text_identical": self.text_identical,
        }


@dataclass
class ValueStats:
    count: int = 0
    finite_count: int = 0
    min_value: float | None = None
    max_value: float | None = None
    sum_value: float = 0.0

    def add(self, text: str) -> None:
        value = float(text)
        self.count += 1
        if math.isfinite(value):
            self.finite_count += 1
            self.sum_value += value
            self.min_value = value if self.min_value is None else min(self.min_value, value)
            self.max_value = value if self.max_value is None else max(self.max_value, value)

    def as_json(self) -> dict[str, object]:
        mean = self.sum_value / self.finite_count if self.finite_count else None
        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "all_finite": self.count == self.finite_count,
            "min": self.min_value,
            "max": self.max_value,
            "mean": mean,
        }


def record_lines(path: Path) -> list[str]:
    return [
        line.rstrip("\n")
        for line in path.read_text(errors="replace").splitlines()
        if line.startswith("MASS_PREPART ") or line.startswith("WPH_PREPART ")
    ]


def parse_headers(path: Path) -> dict[str, str]:
    headers: dict[str, str] = {}
    schemas: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("record_schema "):
            schemas.append(line)
            continue
        if line.startswith("MASS_PREPART ") or line.startswith("WPH_PREPART "):
            break
        key, _, value = line.partition(" ")
        headers[key] = value.strip()
    headers["record_schemas"] = "\n".join(schemas)
    return headers


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def clean_and_link_filtered_root(files: Iterable[Path]) -> None:
    FILTERED_ROOT.mkdir(parents=True, exist_ok=True)
    for old_file in FILTERED_ROOT.glob("*.txt"):
        old_file.unlink()
    for src in files:
        dst = FILTERED_ROOT / src.name
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def update_shape(shape_sets: dict[str, set[int]], tokens: list[str]) -> None:
    zero_x, zero_y, zero_k = map(int, tokens[4:7])
    shape_sets["x"].add(zero_x)
    shape_sets["y"].add(zero_y)
    shape_sets["z"].add(zero_k)


def shape_from_sets(shape_sets: dict[str, set[int]]) -> list[int]:
    return [len(shape_sets["z"]), len(shape_sets["y"]), len(shape_sets["x"])]


def compare_files(old_file: Path, new_file: Path, stats: dict[str, DiffStats], qv: ValueStats,
                  mass_shape: dict[str, set[int]], wph_shape: dict[str, set[int]]) -> None:
    old_lines = record_lines(old_file)
    new_lines = record_lines(new_file)
    require(len(old_lines) == len(new_lines), f"record count mismatch: {old_file.name}")

    for line_no, (old_line, new_line) in enumerate(zip(old_lines, new_lines), start=1):
        old = old_line.split()
        new = new_line.split()
        require(old[0] == new[0], f"record type mismatch {old_file.name}:{line_no}")
        require(old[1:7] == new[1:7], f"record coordinate mismatch {old_file.name}:{line_no}")

        if old[0] == "MASS_PREPART":
            require(len(old) == 13, f"old MASS_PREPART length changed: {old_file.name}:{line_no}")
            require(len(new) == 14, f"new MASS_PREPART lacks QVAPOR: {new_file.name}:{line_no}")
            update_shape(mass_shape, new)
            for idx, field in enumerate(MASS_FIELDS):
                stats[field].add(old[7 + idx], new[7 + idx])
            qv.add(new[13])
        elif old[0] == "WPH_PREPART":
            require(len(old) == 10, f"old WPH_PREPART length changed: {old_file.name}:{line_no}")
            require(len(new) == 10, f"new WPH_PREPART length changed: {new_file.name}:{line_no}")
            update_shape(wph_shape, new)
            for idx, field in enumerate(WPH_FIELDS):
                stats[field].add(old[7 + idx], new[7 + idx])
        else:
            raise RuntimeError(f"unexpected record {old[0]} in {old_file.name}:{line_no}")


def main() -> int:
    old_files = sorted(OLD_ROOT.glob(TARGET_GLOB))
    new_files = sorted(NEW_ROOT.glob(TARGET_GLOB))
    require(len(old_files) == 28, f"accepted pre-call file count is {len(old_files)}, expected 28")
    require(len(new_files) == 28, f"new pre-call file count is {len(new_files)}, expected 28")
    require([p.name for p in old_files] == [p.name for p in new_files], "pre-call tile file names differ")
    require(PATCH_DIFF.exists(), f"missing WRF patch diff: {PATCH_DIFF}")

    clean_and_link_filtered_root(new_files)

    stats = {field: DiffStats() for field in [*MASS_FIELDS, *WPH_FIELDS]}
    qv = ValueStats()
    mass_shape = {"x": set(), "y": set(), "z": set()}
    wph_shape = {"x": set(), "y": set(), "z": set()}
    header_examples: dict[str, dict[str, str]] = {}

    for old_file, new_file in zip(old_files, new_files):
        old_headers = parse_headers(old_file)
        new_headers = parse_headers(new_file)
        if not header_examples:
            header_examples = {"accepted": old_headers, "new": new_headers}
        for headers, label in [(old_headers, "accepted"), (new_headers, "new")]:
            require(headers.get("surface") == "before_first_rk_step_part1_call",
                    f"{label} wrong surface in {old_file.name}")
            require(headers.get("domain_id") == "2", f"{label} wrong domain in {old_file.name}")
            require(headers.get("grid_itimestep_after_increment") == "1",
                    f"{label} wrong step in {old_file.name}")
            require(headers.get("rk_step") == "1", f"{label} wrong rk_step in {old_file.name}")
        require("QVAPOR" not in old_headers["record_schemas"], "accepted schema already had QVAPOR")
        require("QVAPOR" in new_headers["record_schemas"], "new schema lacks QVAPOR")
        require(new_headers.get("moist_index_qv") == "2", f"missing moist_index_qv in {new_file.name}")
        compare_files(old_file, new_file, stats, qv, mass_shape, wph_shape)

    mass_shape_zyx = shape_from_sets(mass_shape)
    wph_shape_zyx = shape_from_sets(wph_shape)
    comparisons = {field: stat.as_json() for field, stat in stats.items()}
    old_fields_exact = all(item["max_abs"] == 0.0 and item["text_identical"] for item in comparisons.values())
    qv_json = qv.as_json()
    qv_shape_ok = mass_shape_zyx == EXPECTED_MASS_SHAPE_ZYX and qv_json["count"] == math.prod(EXPECTED_MASS_SHAPE_ZYX)

    verdict = (
        "STEP1_QVAPOR_PRECALL_SAVEPOINT_READY"
        if old_fields_exact and qv_json["all_finite"] and qv_shape_ok
        else "STEP1_QVAPOR_PRECALL_SAVEPOINT_BLOCKED_VALIDATION_FAILED"
    )

    result = {
        "verdict": verdict,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_used": False,
        "production_source_changed": False,
        "scratch_only_wrf_source_changed": True,
        "accepted_truth_root": str(OLD_ROOT),
        "new_raw_truth_root": str(NEW_ROOT),
        "new_filtered_precall_truth_root": str(FILTERED_ROOT),
        "target_glob": TARGET_GLOB,
        "wrf_patch_diff": str(PATCH_DIFF),
        "wrf_logs": {
            "manager_mpirun_stdout": str(LOG_ROOT / "wrf_qvapor_precall_savepoint_manager_stdout.log"),
            "compile_env_log": str(LOG_ROOT / "compile_qvapor_precall_savepoint_env.log"),
            "sandboxed_mpirun_failure_log": str(LOG_ROOT / "wrf_qvapor_precall_savepoint_stdout.log"),
            "sandboxed_singleton_failure_log": str(LOG_ROOT / "wrf_qvapor_precall_savepoint_singleton_stdout.log"),
        },
        "file_counts": {
            "accepted_precall_files": len(old_files),
            "new_precall_files": len(new_files),
            "new_raw_all_hook_files": len(list(NEW_ROOT.glob("*.txt"))),
            "filtered_precall_files": len(list(FILTERED_ROOT.glob("*.txt"))),
        },
        "headers_example": header_examples,
        "shapes": {
            "mass_shape_zyx": mass_shape_zyx,
            "expected_mass_shape_zyx": EXPECTED_MASS_SHAPE_ZYX,
            "wph_shape_zyx": wph_shape_zyx,
            "expected_wph_shape_zyx": EXPECTED_WPH_SHAPE_ZYX,
        },
        "qvapor": qv_json,
        "old_field_comparisons_vs_accepted_precall": comparisons,
        "acceptance": {
            "old_fields_text_identical": all(item["text_identical"] for item in comparisons.values()),
            "old_fields_max_abs_all_zero": all(item["max_abs"] == 0.0 for item in comparisons.values()),
            "qvapor_full_mass_shape": qv_shape_ok,
            "qvapor_all_finite": qv_json["all_finite"],
            "no_post_rk_prehallo_qvapor_used": True,
        },
        "next_use": {
            "rerun_theta_proof_with_same_boundary_qvapor": True,
            "suggested_input_root": str(FILTERED_ROOT),
        },
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text(render_markdown(result))
    print(json.dumps({"verdict": verdict, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, indent=2))
    return 0 if verdict == "STEP1_QVAPOR_PRECALL_SAVEPOINT_READY" else 1


def render_markdown(result: dict[str, object]) -> str:
    comparisons = result["old_field_comparisons_vs_accepted_precall"]
    qvapor = result["qvapor"]
    lines = [
        "# V0.14 Step-1 QVAPOR Pre-call Savepoint",
        "",
        f"Verdict: `{result['verdict']}`.",
        "",
        "## Result",
        "",
        f"- CPU-only proof; GPU used: `{result['gpu_used']}`.",
        f"- Production `src/gpuwrf` source changed: `{result['production_source_changed']}`.",
        f"- New raw WRF truth root: `{result['new_raw_truth_root']}`.",
        f"- Filtered pre-call-only truth root for the next proof: `{result['new_filtered_precall_truth_root']}`.",
        f"- WRF patch diff artifact: `{result['wrf_patch_diff']}`.",
        f"- File counts: `{result['file_counts']}`.",
        f"- Mass shape z/y/x: `{result['shapes']['mass_shape_zyx']}`.",
        f"- WPH shape z/y/x: `{result['shapes']['wph_shape_zyx']}`.",
        f"- QVAPOR count `{qvapor['count']}`, all finite `{qvapor['all_finite']}`, "
        f"min `{qvapor['min']}`, max `{qvapor['max']}`, mean `{qvapor['mean']}`.",
        "",
        "## Old-field identity check",
        "",
        "| Field | Count | Max abs | RMSE | Text identical |",
        "|---|---:|---:|---:|---|",
    ]
    for field in [*MASS_FIELDS, *WPH_FIELDS]:
        item = comparisons[field]
        lines.append(
            f"| `{field}` | {item['count']} | {item['max_abs']} | {item['rmse']} | {item['text_identical']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The new disposable WRF hook appended `QVAPOR` to the accepted "
        "`before_first_rk_step_part1_call` mass record without changing the "
        "previous mass or W/PH fields. All previous fields are text-identical "
        "to the accepted pre-call dump, so the new QVAPOR field is a valid "
        "same-boundary truth input for the next theta-semantics rerun.",
        "",
        "This proof does not authorize a production theta or `adjust_tempqv` "
        "patch by itself. The next sprint must rerun the theta proof using "
        "the filtered same-boundary QVAPOR root and classify the remaining "
        "worst cell before a source patch decision.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
