#!/usr/bin/env python3
"""V0.14 earlier-source bisection for the h10 d02 OperationalCarry.

Evidence-only proof.  The live nested loader is not CPU-capable in this branch
because ``State.zeros`` requires a visible JAX GPU, so the required CPU command
validates and renders a compact replay artifact.  A targeted GPU run may create
that compact artifact when explicitly allowed.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
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
from proofs.v014 import previous_step_handoff_bisect as prev  # noqa: E402


OUT_JSON = ROOT / "proofs/v014/earlier_source_bisect.json"
OUT_MD = ROOT / "proofs/v014/earlier_source_bisect.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-earlier-source-bisect.md"

PROJECT_CONSTITUTION = ROOT / "PROJECT_CONSTITUTION.md"
AGENTS = ROOT / "AGENTS.md"
SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-09-v014-earlier-source-bisect/sprint-contract.md"
)
VALIDATING_PHYSICS_SKILL = ROOT / ".agent/skills/validating-physics/SKILL.md"
WRF_ORACLE_SKILL = ROOT / ".agent/skills/building-wrf-oracles/SKILL.md"
REPORTING_SKILL = ROOT / ".agent/skills/reporting-to-human/SKILL.md"
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"

PRE_RK_JSON = ROOT / "proofs/v014/pre_rk_input_boundary.json"
SOURCE_TRACE_JSON = ROOT / "proofs/v014/prestep_carry_source_trace.json"
PREVIOUS_STEP_JSON = ROOT / "proofs/v014/previous_step_handoff_bisect.json"
BASE_STATE_JSON = ROOT / "proofs/v014/base_state_writer_attribution.json"
STATIC_BASE_JSON = ROOT / "proofs/v014/static_metric_base_parity.json"
PRODUCER_SCRIPT = ROOT / "proofs/v014/jax_h10_prestep_carry_producer.py"
PRODUCER_JSON = ROOT / "proofs/v014/jax_h10_prestep_carry_producer.json"
DOMAIN_TREE_MODULE = ROOT / "src/gpuwrf/runtime/domain_tree.py"
OPERATIONAL_MODE_MODULE = ROOT / "src/gpuwrf/runtime/operational_mode.py"
STATE_MODULE = ROOT / "src/gpuwrf/contracts/state.py"
D02_REPLAY_MODULE = ROOT / "src/gpuwrf/integration/d02_replay.py"
NESTED_PIPELINE_MODULE = ROOT / "src/gpuwrf/integration/nested_pipeline.py"

RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
DEFAULT_INPUT_ROOTS = (
    Path("/tmp/v0120_merged_run_root"),
    Path("/mnt/data/canairy_meteo/runs/wrf_l2"),
)
CPU_WRFOUT_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output") / RUN_ID
TARGET_DOMAIN = "d02"
TARGET_FIELDS = ("T", "P", "PB", "MU", "MUB")
STATIC_BASE_FIELDS = ("PB", "MUB")
TARGET_STEP = 6000
PRESTEP_COMPLETED_STEPS = TARGET_STEP - 1
TARGET_DT_S = 6.0
THETA_OFFSET_K = 300.0
TOLERANCE_MAX_ABS = 2.0e-6
MAX_DOM = 2

START_TIME = datetime(2026, 5, 1, 18, 0, 0, tzinfo=timezone.utc)
SCHEDULE_D02_STEPS = (0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 5997)

ARTIFACT_ROOT_CANDIDATES = (
    Path(os.environ["WRFGPU2_EARLIER_SOURCE_BISECT_ROOT"])
    if os.environ.get("WRFGPU2_EARLIER_SOURCE_BISECT_ROOT")
    else None,
    Path("/mnt/data/wrf_gpu2/v014_earlier_source_bisect"),
    Path("/tmp/wrf_gpu2_v014_earlier_source_bisect"),
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
    raise OSError("could not create v014_earlier_source_bisect artifact root")


ARTIFACT_ROOT = artifact_root()
REPLAY_ARTIFACT_JSON = ARTIFACT_ROOT / "earlier_source_bisect.live_replay_compact.json"


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


def path_info(path: Path) -> dict[str, Any]:
    return prev.path_info(path)


def run_command(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    return prev.run_command(command, cwd=cwd, env=env)


def nvidia_smi_query(query: str) -> dict[str, Any]:
    return prev.nvidia_smi_query(query)


def input_run_dir() -> Path:
    for root in DEFAULT_INPUT_ROOTS:
        path = root / RUN_ID
        if path.is_dir() and (path / "wrfinput_d01").exists() and (path / "wrfinput_d02").exists():
            return path
    searched = ", ".join(str(root / RUN_ID) for root in DEFAULT_INPUT_ROOTS)
    raise FileNotFoundError(f"missing L2 native-init run directory; searched {searched}")


def valid_time_for_d02_step(step: int) -> datetime:
    return START_TIME + timedelta(seconds=float(step) * TARGET_DT_S)


def wrfout_for_d02_step(step: int) -> Path | None:
    if step % 600 != 0:
        return None
    label = valid_time_for_d02_step(step).strftime("%Y-%m-%d_%H:%M:%S")
    return CPU_WRFOUT_DIR / f"wrfout_d02_{label}"


def nc_array(dataset: Any, field: str) -> Any:
    var = dataset.variables[field]
    if var.dimensions and var.dimensions[0] == "Time":
        return var[0]
    return var[:]


def nc_index(field: str, key: tuple[int, ...]) -> tuple[int, ...]:
    if field in {"T", "P", "PB"}:
        return (0, key[0], key[1])
    if field in {"MU", "MUB"}:
        return (key[0], key[1])
    raise KeyError(field)


def compare_patch_to_netcdf(
    *,
    name: str,
    source_expression: str,
    field: str,
    patch: Any,
    netcdf_path: Path | None,
    bounds: Mapping[str, int],
    keys: Iterable[tuple[int, ...]],
    truth_surface: str,
    same_step_truth: bool,
    eligible_for_verdict: bool,
) -> dict[str, Any]:
    if netcdf_path is None:
        return {
            "name": name,
            "field": field,
            "status": "TRUTH_UNAVAILABLE",
            "reason": "no same-step hourly CPU-WRF wrfout for this d02 step",
            "truth_surface": truth_surface,
            "same_step_truth": bool(same_step_truth),
            "eligible_for_verdict": bool(eligible_for_verdict),
        }
    if not netcdf_path.exists():
        return {
            "name": name,
            "field": field,
            "status": "TRUTH_MISSING",
            "truth_file": str(netcdf_path),
            "truth_surface": truth_surface,
            "same_step_truth": bool(same_step_truth),
            "eligible_for_verdict": bool(eligible_for_verdict),
        }
    import netCDF4  # noqa: PLC0415

    arr = np.asarray(patch, dtype=np.float64)
    y0, x0 = int(bounds["y0"]), int(bounds["x0"])
    diffs: list[float] = []
    worst: dict[str, Any] | None = None
    skipped: list[dict[str, Any]] = []
    with netCDF4.Dataset(netcdf_path, "r") as ds:
        if field not in ds.variables:
            return {
                "name": name,
                "field": field,
                "status": "TRUTH_MISSING_FIELD",
                "truth_file": str(netcdf_path),
                "truth_surface": truth_surface,
                "same_step_truth": bool(same_step_truth),
                "eligible_for_verdict": bool(eligible_for_verdict),
            }
        truth_arr = nc_array(ds, field)
        for key in sorted(keys):
            rel_y, rel_x = int(key[0]) - y0, int(key[1]) - x0
            try:
                candidate = float(arr[rel_y, rel_x])
                truth = float(truth_arr[nc_index(field, key)])
            except Exception as exc:  # pragma: no cover - recorded in proof output
                skipped.append({"native_key": list(key), "reason": repr(exc)})
                continue
            diff = candidate - truth
            diffs.append(diff)
            if worst is None or abs(diff) > worst["abs_diff"]:
                worst = {
                    "native_key": list(key),
                    "patch_index": [rel_y, rel_x],
                    "jax_candidate": candidate,
                    "wrf_truth": truth,
                    "diff_jax_minus_wrf": diff,
                    "abs_diff": abs(diff),
                }
    out = {
        "name": name,
        "field": field,
        "source_expression": source_expression,
        "truth_file": str(netcdf_path),
        "truth_surface": truth_surface,
        "same_step_truth": bool(same_step_truth),
        "eligible_for_verdict": bool(eligible_for_verdict),
        "array_summary": prev.array_summary(arr),
        **stats(diffs),
        "worst": worst,
        "skipped_record_count": len(skipped),
        "skipped_records": skipped,
        "tolerance_max_abs": TOLERANCE_MAX_ABS,
    }
    out["status"] = (
        "MATCH"
        if out.get("max_abs") is not None and float(out["max_abs"]) <= TOLERANCE_MAX_ABS
        else "DIFF"
    )
    return out


def compare_patch_to_pre_rk_static(
    *,
    name: str,
    source_expression: str,
    field: str,
    patch: Any,
    wrf_truth: Mapping[str, Any],
    bounds: Mapping[str, int],
    eligible_for_verdict: bool,
) -> dict[str, Any]:
    entry = prev.compare_patch_to_truth(
        name=name,
        source_expression=source_expression,
        field=field,
        patch=patch,
        wrf_truth=wrf_truth,
        bounds=bounds,
        target_leaf_eligible=True,
    )
    entry["truth_surface"] = "CPU-WRF pre-RK hook at d02 step 6000; static PB/MUB only"
    entry["same_step_truth"] = False
    entry["static_base_truth"] = field in STATIC_BASE_FIELDS
    entry["eligible_for_verdict"] = bool(eligible_for_verdict and field in STATIC_BASE_FIELDS)
    return entry


def unavailable_dynamic_entry(label: str, field: str, step: int) -> dict[str, Any]:
    return {
        "name": f"{label}.{field}",
        "field": field,
        "status": "TRUTH_UNAVAILABLE",
        "truth_surface": "no same-step CPU-WRF internal/savepoint truth",
        "same_step_truth": False,
        "eligible_for_verdict": False,
        "reason": (
            f"d02 completed step {step} is not an hourly wrfout boundary, and this sprint did not "
            "add a WRF/JAX same-step savepoint hook."
        ),
    }


def target_source(field: str) -> str:
    return prev.TARGET_SOURCE[field]


def make_snapshot(
    *,
    label: str,
    step: int,
    own_steps: Mapping[str, int],
    patch_arrays: Mapping[str, np.ndarray],
    wrf_truth: Mapping[str, Any],
    bounds: Mapping[str, int],
    keys: Iterable[tuple[int, ...]],
    previous_patch: Mapping[str, np.ndarray] | None,
    initial_patch: Mapping[str, np.ndarray] | None,
    notes: str,
) -> dict[str, Any]:
    wrfinput = input_run_dir() / "wrfinput_d02"
    wrfout = wrfout_for_d02_step(step)
    has_same_step_wrfout = wrfout is not None and wrfout.exists()

    native_wrfinput = {
        field: compare_patch_to_netcdf(
            name=f"{label}.native_wrfinput.{field}",
            source_expression=target_source(field),
            field=field,
            patch=patch_arrays[field],
            netcdf_path=wrfinput,
            bounds=bounds,
            keys=keys,
            truth_surface="native wrfinput_d02 consumed by _load_domains/build_replay_case",
            same_step_truth=(step == 0),
            eligible_for_verdict=(step == 0),
        )
        for field in TARGET_FIELDS
    }
    cpu_wrfout = {
        field: compare_patch_to_netcdf(
            name=f"{label}.cpu_wrfout.{field}",
            source_expression=target_source(field),
            field=field,
            patch=patch_arrays[field],
            netcdf_path=wrfout,
            bounds=bounds,
            keys=keys,
            truth_surface=f"CPU-WRF hourly wrfout at d02 completed step {step}",
            same_step_truth=has_same_step_wrfout,
            eligible_for_verdict=has_same_step_wrfout,
        )
        for field in TARGET_FIELDS
    }
    pre_rk_static = {
        field: (
            compare_patch_to_pre_rk_static(
                name=f"{label}.pre_rk_static.{field}",
                source_expression=target_source(field),
                field=field,
                patch=patch_arrays[field],
                wrf_truth=wrf_truth,
                bounds=bounds,
                eligible_for_verdict=True,
            )
            if field in STATIC_BASE_FIELDS
            else unavailable_dynamic_entry(label, field, step)
        )
        for field in TARGET_FIELDS
    }

    if has_same_step_wrfout:
        primary = cpu_wrfout
        primary_surface = "same_step_cpu_wrf_hourly_wrfout"
    elif step == PRESTEP_COMPLETED_STEPS:
        primary = pre_rk_static
        primary_surface = "static_base_only_cpu_wrf_pre_rk_hook"
    else:
        primary = {
            field: (
                pre_rk_static[field]
                if field in STATIC_BASE_FIELDS
                else unavailable_dynamic_entry(label, field, step)
            )
            for field in TARGET_FIELDS
        }
        primary_surface = "static_base_only_cpu_wrf_pre_rk_hook"

    delta_previous = None
    if previous_patch is not None:
        delta_previous = {
            field: prev.compare_arrays(
                f"{label}.{field}.delta_vs_previous_snapshot",
                patch_arrays[field],
                previous_patch[field],
            )
            for field in TARGET_FIELDS
        }
    delta_initial = None
    if initial_patch is not None:
        delta_initial = {
            field: prev.compare_arrays(
                f"{label}.{field}.delta_vs_initial_native_load",
                patch_arrays[field],
                initial_patch[field],
            )
            for field in TARGET_FIELDS
        }

    eligible_primary = [
        field
        for field in TARGET_FIELDS
        if primary[field].get("eligible_for_verdict") and primary[field].get("max_abs") is not None
    ]
    worst_field = None
    if eligible_primary:
        worst_field = max(eligible_primary, key=lambda field: float(primary[field]["max_abs"]))

    return {
        "label": label,
        "d02_completed_step": int(step),
        "valid_time_utc": valid_time_for_d02_step(step).isoformat().replace("+00:00", "Z"),
        "own_steps": {key: int(value) for key, value in own_steps.items()},
        "notes": notes,
        "primary_truth_surface": primary_surface,
        "same_step_cpu_wrfout_available": bool(has_same_step_wrfout),
        "native_wrfinput_comparisons": native_wrfinput,
        "cpu_wrfout_comparisons": cpu_wrfout,
        "pre_rk_static_comparisons": pre_rk_static,
        "primary_truth_comparisons": primary,
        "native_wrfinput_static_base_match": all(
            native_wrfinput[field]["status"] == "MATCH" for field in STATIC_BASE_FIELDS
        ),
        "cpu_wrf_static_base_match": all(
            primary[field]["status"] == "MATCH" for field in STATIC_BASE_FIELDS
        ),
        "all_target_fields_match_same_step_cpu_wrf": (
            has_same_step_wrfout and all(cpu_wrfout[field]["status"] == "MATCH" for field in TARGET_FIELDS)
        ),
        "worst_primary_field": (
            {
                "field": worst_field,
                "max_abs": primary[worst_field].get("max_abs"),
                "rmse": primary[worst_field].get("rmse"),
            }
            if worst_field is not None
            else None
        ),
        "delta_vs_previous_snapshot": delta_previous,
        "delta_vs_initial_native_load": delta_initial,
    }


def compare_netcdf_to_netcdf(
    *,
    left: Path,
    right: Path,
    fields: Iterable[str],
    keys: Iterable[tuple[int, ...]],
    name: str,
) -> dict[str, Any]:
    import netCDF4  # noqa: PLC0415

    out: dict[str, Any] = {"name": name, "left": str(left), "right": str(right), "fields": {}}
    with netCDF4.Dataset(left, "r") as lds, netCDF4.Dataset(right, "r") as rds:
        for field in fields:
            diffs: list[float] = []
            worst: dict[str, Any] | None = None
            if field not in lds.variables or field not in rds.variables:
                out["fields"][field] = {"status": "MISSING"}
                continue
            la = nc_array(lds, field)
            ra = nc_array(rds, field)
            for key in sorted(keys):
                lv = float(la[nc_index(field, key)])
                rv = float(ra[nc_index(field, key)])
                diff = lv - rv
                diffs.append(diff)
                if worst is None or abs(diff) > worst["abs_diff"]:
                    worst = {
                        "native_key": list(key),
                        "left": lv,
                        "right": rv,
                        "diff": diff,
                        "abs_diff": abs(diff),
                    }
            item = {**stats(diffs), "worst": worst, "tolerance_max_abs": TOLERANCE_MAX_ABS}
            item["status"] = (
                "MATCH"
                if item.get("max_abs") is not None and float(item["max_abs"]) <= TOLERANCE_MAX_ABS
                else "DIFF"
            )
            out["fields"][field] = item
    out["all_static_base_match"] = all(
        out["fields"].get(field, {}).get("status") == "MATCH" for field in STATIC_BASE_FIELDS
    )
    return out


def compare_netcdf_to_pre_rk_static(
    *,
    netcdf_path: Path,
    wrf_truth: Mapping[str, Any],
    fields: Iterable[str],
    keys: Iterable[tuple[int, ...]],
    name: str,
) -> dict[str, Any]:
    import netCDF4  # noqa: PLC0415

    field_map = {"PB": "PB", "MUB": "MUB"}
    out: dict[str, Any] = {"name": name, "netcdf": str(netcdf_path), "pre_rk_files": wrf_truth["files"], "fields": {}}
    with netCDF4.Dataset(netcdf_path, "r") as ds:
        for field in fields:
            diffs: list[float] = []
            worst: dict[str, Any] | None = None
            arr = nc_array(ds, field)
            for key in sorted(keys):
                rec = wrf_truth["records"]["MASS_K1"][key]
                lv = float(arr[nc_index(field, key)])
                rv = float(rec[field_map[field]])
                diff = lv - rv
                diffs.append(diff)
                if worst is None or abs(diff) > worst["abs_diff"]:
                    worst = {
                        "native_key": list(key),
                        "netcdf": lv,
                        "pre_rk": rv,
                        "diff": diff,
                        "abs_diff": abs(diff),
                    }
            item = {**stats(diffs), "worst": worst, "tolerance_max_abs": TOLERANCE_MAX_ABS}
            item["status"] = (
                "MATCH"
                if item.get("max_abs") is not None and float(item["max_abs"]) <= TOLERANCE_MAX_ABS
                else "DIFF"
            )
            out["fields"][field] = item
    out["all_static_base_match"] = all(
        out["fields"].get(field, {}).get("status") == "MATCH" for field in STATIC_BASE_FIELDS
    )
    return out


def source_nodes() -> dict[str, Any]:
    return {
        "d02_replay_build_replay_case": trace.extract_ast_node(D02_REPLAY_MODULE, "build_replay_case"),
        "nested_pipeline_load_domains": trace.extract_ast_node(NESTED_PIPELINE_MODULE, "_load_domains"),
        "domain_tree_run_operational_domain_tree": trace.extract_ast_node(DOMAIN_TREE_MODULE, "run_operational_domain_tree"),
        "domain_tree_operational_advance_factory": trace.extract_ast_node(DOMAIN_TREE_MODULE, "_operational_advance_factory"),
        "operational_mode_initial_carry_for_run": trace.extract_ast_node(OPERATIONAL_MODE_MODULE, "_initial_carry_for_run"),
        "state_zeros": trace.extract_ast_node(STATE_MODULE, "zeros"),
        "producer_produce_checkpoint": trace.extract_ast_node(PRODUCER_SCRIPT, "produce_checkpoint"),
    }


def input_records(run_dir: Path | None = None) -> dict[str, Any]:
    native_wrfinput = (run_dir or (DEFAULT_INPUT_ROOTS[0] / RUN_ID)) / "wrfinput_d02"
    return {
        "project_constitution": path_info(PROJECT_CONSTITUTION),
        "agents": path_info(AGENTS),
        "sprint_contract": path_info(SPRINT_CONTRACT),
        "validating_physics_skill": path_info(VALIDATING_PHYSICS_SKILL),
        "building_wrf_oracles_skill": path_info(WRF_ORACLE_SKILL),
        "reporting_to_human_skill": path_info(REPORTING_SKILL),
        "handoff": path_info(HANDOFF),
        "pre_rk_input_boundary_json": path_info(PRE_RK_JSON),
        "prestep_carry_source_trace_json": path_info(SOURCE_TRACE_JSON),
        "previous_step_handoff_bisect_json": path_info(PREVIOUS_STEP_JSON),
        "base_state_writer_attribution_json": path_info(BASE_STATE_JSON),
        "static_metric_base_parity_json": path_info(STATIC_BASE_JSON),
        "producer_json": path_info(PRODUCER_JSON),
        "native_wrfinput_d02": path_info(native_wrfinput),
        "cpu_wrfout_dir": {"path": str(CPU_WRFOUT_DIR), "exists": CPU_WRFOUT_DIR.exists(), "is_dir": CPU_WRFOUT_DIR.is_dir()},
    }


def required_commands() -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "required_validation": [
            "python -m py_compile proofs/v014/earlier_source_bisect.py",
            "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/earlier_source_bisect.py",
            "python -m json.tool proofs/v014/earlier_source_bisect.json >/tmp/earlier_source_bisect.validated.json",
        ],
        "gpu_replay_command_used": os.environ.get("WRFGPU2_EARLIER_SOURCE_BISECT_COMMAND_DISPLAY"),
    }


def proof_objects() -> dict[str, str]:
    return {
        "json": str(OUT_JSON),
        "markdown": str(OUT_MD),
        "review": str(OUT_REVIEW),
        "compact_replay_artifact": str(REPLAY_ARTIFACT_JSON),
    }


def classify(
    *,
    snapshots: Mapping[str, Any],
    static_invariance: Mapping[str, Any],
    previous_step: Mapping[str, Any],
    source_trace: Mapping[str, Any],
) -> tuple[str, list[str], str, dict[str, Any] | None]:
    if source_trace.get("classification") != "PRODUCER_WRITES_BAD_FINAL_CARRY":
        return (
            "REPRODUCER_MISMATCH_STARTING_SOURCE_TRACE",
            ["Starting source trace no longer says PRODUCER_WRITES_BAD_FINAL_CARRY."],
            "Rerun proofs/v014/prestep_carry_source_trace.py, then repeat this sprint.",
            None,
        )
    if previous_step.get("classification") != "BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE":
        return (
            "REPRODUCER_MISMATCH_PREVIOUS_STEP_HANDOFF",
            ["Starting previous-step proof no longer says BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE."],
            "Rerun proofs/v014/previous_step_handoff_bisect.py, then repeat this sprint.",
            None,
        )

    initial = snapshots["initial_native_load_carry"]
    if initial["native_wrfinput_static_base_match"] and not initial["cpu_wrf_static_base_match"]:
        return (
            "BASE_STATE_SPLIT_DEFINITION_MISMATCH",
            [
                "Initial d02 OperationalCarry PB/MUB match the native wrfinput_d02 split on the target patch.",
                "The same initial carry PB/MUB already differ from CPU-WRF h0 history and from the CPU-WRF h10 pre-RK static base truth.",
                "CPU-WRF PB/MUB are invariant across h0, h1, h10 wrfout and the h10 pre-RK hook on this patch, so replay-time drift is not needed to explain the bad h10 base carry.",
            ],
            (
                "Open a source-changing fix sprint for src/gpuwrf/integration/d02_replay.py::build_replay_case "
                "native child base-state split construction; reproduce WRF's post-initialization PB/MUB split or load an accepted h0 base-state oracle before replay."
            ),
            None,
        )

    if not initial["cpu_wrf_static_base_match"]:
        return (
            "BAD_AT_NATIVE_LOAD_OR_INITIAL_CARRY",
            [
                "Initial d02 OperationalCarry PB/MUB already differ from CPU-WRF static base truth.",
                "The mismatch is present before any live nested replay segment.",
            ],
            "Open a source-changing fix sprint around native load / initial OperationalCarry construction.",
            None,
        )

    ordered = [snapshots[f"after_replay_segment_d02_step_{step}"] for step in SCHEDULE_D02_STEPS[1:]]
    for idx, snap in enumerate(ordered):
        if not snap["cpu_wrf_static_base_match"]:
            step = snap["d02_completed_step"]
            if idx == 0:
                return (
                    "BAD_AFTER_FIRST_REPLAY_SEGMENT",
                    [f"PB/MUB first differ from CPU-WRF static base truth after d02 step {step}."],
                    "Open a source-changing fix sprint around the first replay segment after native load.",
                    None,
                )
            prev_step = ordered[idx - 1]["d02_completed_step"]
            return (
                f"DRIFTS_BETWEEN_SEGMENTS_{prev_step}_{step}",
                [f"PB/MUB first differ from CPU-WRF static base truth between d02 steps {prev_step} and {step}."],
                "Open a narrower source/hook sprint inside that replay interval.",
                None,
            )

    if not static_invariance.get("h0_vs_h10_pre_rk_static", {}).get("all_static_base_match"):
        return (
            "EARLIER_SOURCE_BLOCKED_STATIC_BASE_ORACLE_INCONSISTENT",
            ["CPU-WRF h0 static base and h10 pre-RK static base were not invariant on the target patch."],
            (
                "Add a WRF savepoint hook for PB/MUB at d02 completed step 5997 before classifying replay drift."
            ),
            {
                "reason": "STATIC_BASE_ORACLE_INCONSISTENT",
                "exact_hook_or_source_file_needed": "WRF dyn_em/solve_em.F pre-RK input hook at d02 completed step 5997",
                "next_command_needed": "Run the existing pre_rk_input_boundary hook with START_STEP=5998 END_STEP=5998 for grid 2.",
            },
        )

    return (
        "EARLIER_SOURCE_BLOCKED_NO_BAD_TRANSITION_FOUND",
        ["No bad transition was found on the reachable schedule despite the starting proofs."],
        "Rerun the producer and previous-step proofs from the same branch, then repeat this sprint.",
        {
            "reason": "NO_BAD_TRANSITION_FOUND",
            "exact_hook_or_source_file_needed": "src/gpuwrf/runtime/domain_tree.py run_operational_domain_tree compact savepoint API",
            "next_command_needed": "Add a compact proof-only segment savepoint API and rerun.",
        },
    )


def build_payload_from_live_replay(cpu_preflight: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    from gpuwrf.integration.nested_pipeline import (  # noqa: PLC0415
        NestedPipelineConfig,
        _load_domains,
        domain_names_for,
    )
    from gpuwrf.runtime.domain_tree import DomainTree, run_operational_domain_tree  # noqa: PLC0415

    started = time.perf_counter()
    pre_rk = load_json(PRE_RK_JSON)
    source_trace = load_json(SOURCE_TRACE_JSON)
    previous_step = load_json(PREVIOUS_STEP_JSON)
    base_state = load_optional_json(BASE_STATE_JSON)
    static_base = load_optional_json(STATIC_BASE_JSON)
    producer = load_optional_json(PRODUCER_JSON)
    wrf_truth = trace.parse_pre_rk_truth(pre_rk)
    bounds = prev.patch_bounds_from_truth(wrf_truth)
    keys = tuple(sorted(wrf_truth["records"]["MASS_K1"]))

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
    snapshots: dict[str, Any] = {}
    patch_cache: dict[str, dict[str, np.ndarray]] = {}
    segment_records: list[dict[str, Any]] = []

    with prev.VramSampler() as sampler:
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

        initial = run_operational_domain_tree(
            tree,
            root_steps=0,
            feedback_enabled=False,
            output=None,
            output_cadence_steps=None,
            block_between=True,
            carries=None,
            initial_own_steps={name: 0 for name in names},
        )
        jax.block_until_ready(tuple(carry.state.theta for carry in initial.carries.values()))
        carries = dict(initial.carries)
        own_steps = dict(initial.own_steps)
        initial_patch = prev.patch_arrays_from_carry(carries[TARGET_DOMAIN], bounds)
        patch_cache["initial_native_load_carry"] = initial_patch
        snapshots["initial_native_load_carry"] = make_snapshot(
            label="initial_native_load_carry",
            step=0,
            own_steps=own_steps,
            patch_arrays=initial_patch,
            wrf_truth=wrf_truth,
            bounds=bounds,
            keys=keys,
            previous_patch=None,
            initial_patch=None,
            notes="Immediately after _load_domains and _initial_carry_for_run; before any parent or child replay step.",
        )

        parent_targets = [step // ratio for step in SCHEDULE_D02_STEPS[1:]]
        completed_parent = 0
        for step, parent_target in zip(SCHEDULE_D02_STEPS[1:], parent_targets):
            seg = int(parent_target) - int(completed_parent)
            if seg <= 0:
                raise ValueError(f"non-positive parent segment for d02 step {step}: {seg}")
            seg_start = time.perf_counter()
            result = run_operational_domain_tree(
                tree,
                root_steps=seg,
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
            if int(own_steps[TARGET_DOMAIN]) != int(step):
                raise RuntimeError(f"expected d02 step {step}, got {own_steps[TARGET_DOMAIN]}")
            record = {
                "segment": len(segment_records) + 1,
                "root_steps": int(seg),
                "wall_s": float(time.perf_counter() - seg_start),
                "own_steps": dict(own_steps),
            }
            segment_records.append(record)
            label = f"after_replay_segment_d02_step_{step}"
            previous_label = next(reversed(patch_cache))
            arrays = prev.patch_arrays_from_carry(carries[TARGET_DOMAIN], bounds)
            patch_cache[label] = arrays
            snapshots[label] = make_snapshot(
                label=label,
                step=step,
                own_steps=own_steps,
                patch_arrays=arrays,
                wrf_truth=wrf_truth,
                bounds=bounds,
                keys=keys,
                previous_patch=patch_cache[previous_label],
                initial_patch=initial_patch,
                notes=(
                    "Coarse producer-path replay boundary before d02 completed step 5997."
                    if step < PRESTEP_COMPLETED_STEPS
                    else "Final reachable boundary before parent step 2000, _operational_force, and d02 steps 5998-5999."
                ),
            )
            print(
                f"earlier_bisect_progress d01={own_steps.get('d01')} d02={own_steps.get('d02')} "
                f"wall_s={record['wall_s']:.1f}",
                flush=True,
            )

        vram_summary = sampler.summary()

    nvidia_after = nvidia_smi_query("timestamp,index,name,memory.used,memory.total,utilization.gpu")
    h0 = wrfout_for_d02_step(0)
    h1 = wrfout_for_d02_step(600)
    h10 = CPU_WRFOUT_DIR / "wrfout_d02_2026-05-02_04:00:00"
    static_invariance = {
        "h0_vs_h1_static": compare_netcdf_to_netcdf(
            left=h0,
            right=h1,
            fields=STATIC_BASE_FIELDS,
            keys=keys,
            name="cpu_wrf_h0_vs_h1_static_base",
        )
        if h0 and h1 and h0.exists() and h1.exists()
        else {"status": "MISSING"},
        "h0_vs_h10_wrfout_static": compare_netcdf_to_netcdf(
            left=h0,
            right=h10,
            fields=STATIC_BASE_FIELDS,
            keys=keys,
            name="cpu_wrf_h0_vs_h10_wrfout_static_base",
        )
        if h0 and h0.exists() and h10.exists()
        else {"status": "MISSING"},
        "h0_vs_h10_pre_rk_static": compare_netcdf_to_pre_rk_static(
            netcdf_path=h0,
            wrf_truth=wrf_truth,
            fields=STATIC_BASE_FIELDS,
            keys=keys,
            name="cpu_wrf_h0_wrfout_vs_h10_pre_rk_static_base",
        )
        if h0 and h0.exists()
        else {"status": "MISSING"},
        "from_static_metric_base_parity": {
            "cpu_wrfout_h0_vs_cpu_wrfout_h1": (
                static_base.get("comparisons", {}).get("cpu_wrfout_h0_vs_cpu_wrfout_h1", {})
                if isinstance(static_base, Mapping)
                else None
            )
        },
    }

    classification, rationale, next_decision, blocked = classify(
        snapshots=snapshots,
        static_invariance=static_invariance,
        previous_step=previous_step,
        source_trace=source_trace,
    )

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.earlier_source_bisect.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": classification,
        "classification": classification,
        "classification_rationale": rationale,
        "blocked": blocked,
        "target": {
            "domain": TARGET_DOMAIN,
            "wrf_step": TARGET_STEP,
            "prestep_completed_steps": PRESTEP_COMPLETED_STEPS,
            "target_fields": list(TARGET_FIELDS),
            "static_base_fields_used_for_initial_classification": list(STATIC_BASE_FIELDS),
            "tolerance_max_abs": TOLERANCE_MAX_ABS,
            "patch_bounds": dict(bounds),
            "schedule_d02_completed_steps": list(SCHEDULE_D02_STEPS),
        },
        "cpu_preflight": cpu_preflight,
        "gpu_used": True,
        "gpu_replay": {
            "why_cpu_replay_not_practical": cpu_preflight.get("detail")
            or "CPU live replay preflight failed; see cpu_preflight.command_result.",
            "command_display": os.environ.get("WRFGPU2_EARLIER_SOURCE_BISECT_COMMAND_DISPLAY"),
            "environment": prev.environment_snapshot(),
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
        "final_partial_subcycle_debug_run": False,
        "inputs_read": input_records(run_dir),
        "source_nodes": source_nodes(),
        "wrf_truth": {
            "pre_rk_input_boundary_json": str(PRE_RK_JSON),
            "pre_rk_verdict": pre_rk.get("verdict"),
            "pre_rk_files": wrf_truth["files"],
            "pre_rk_file_info": wrf_truth["file_info"],
            "pre_rk_unique_counts": wrf_truth["unique_counts"],
            "native_wrfinput": str(run_dir / "wrfinput_d02"),
            "cpu_wrfout_dir": str(CPU_WRFOUT_DIR),
            "hourly_wrfout_steps": {
                str(step): str(wrfout_for_d02_step(step)) for step in SCHEDULE_D02_STEPS if wrfout_for_d02_step(step)
            },
            "static_base_invariance": static_invariance,
        },
        "starting_facts": {
            "prestep_carry_source_trace_classification": source_trace.get("classification"),
            "previous_step_handoff_bisect_classification": previous_step.get("classification"),
            "producer_json_verdict": producer.get("verdict") if isinstance(producer, Mapping) else None,
            "base_state_summary": base_state.get("summary") if isinstance(base_state, Mapping) else None,
        },
        "producer_provenance": {
            "run_dir": str(run_dir),
            "load": load_record,
            "nesting": {
                "parent": parent,
                "parent_grid_ratio": ratio,
                "metadata": meta,
            },
            "segments": segment_records,
        },
        "snapshots": snapshots,
        "acceptance_notes": {
            "uses_cpu_wrf_pre_rk_truth_from_pre_rk_input_boundary": True,
            "uses_cpu_wrf_hourly_wrfouts_for_same_step_segment_truth": True,
            "uses_native_wrfinput_to_separate_loader_identity_from_wrf_base_convention": True,
            "jax_vs_jax_self_compare_used_as_truth": False,
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
            "Dynamic T/P/MU at d02 step 5997 still lack same-step CPU-WRF internal truth; only PB/MUB are classified there as static base fields.",
            "The replay required a targeted GPU run because State.zeros/_load_domains is not CPU-capable in this branch.",
            "The conclusion is patch-local to the existing h10 target patch, not a full-grid validation campaign.",
        ],
        "next_decision": next_decision,
        "wall_s": float(time.perf_counter() - started),
    }
    return payload


def blocked_payload(reason: str, detail: str, next_action: str, cpu_preflight: Mapping[str, Any] | None = None) -> dict[str, Any]:
    classification = f"EARLIER_SOURCE_BLOCKED_{reason}"
    return {
        "schema": "wrfgpu2.v014.earlier_source_bisect.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": classification,
        "classification": classification,
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
        "final_partial_subcycle_debug_run": False,
        "inputs_read": input_records(None),
        "source_nodes": source_nodes(),
        "commands": required_commands(),
        "proof_objects": proof_objects(),
        "unresolved_risks": [detail],
        "next_decision": next_action,
    }


def snapshot_order(payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    snapshots = payload.get("snapshots") or {}
    labels = ["initial_native_load_carry", *[f"after_replay_segment_d02_step_{step}" for step in SCHEDULE_D02_STEPS[1:]]]
    out: list[tuple[str, Mapping[str, Any]]] = []
    for label in labels:
        if label in snapshots:
            out.append((label, snapshots[label]))
    for label, snap in snapshots.items():
        if label not in labels:
            out.append((label, snap))
    return out


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# V0.14 Earlier-Source Bisection",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Summary",
        "",
        f"- Classification: `{payload['classification']}`.",
        f"- GPU used: `{payload.get('gpu_used')}`.",
        f"- CPU preflight: `{(payload.get('cpu_preflight') or {}).get('status')}`.",
    ]
    if payload.get("blocked"):
        blocked = payload["blocked"]
        lines.extend(
            [
                "",
                "## Blocker",
                "",
                f"- Reason: `{blocked.get('reason')}`.",
                f"- Detail: {blocked.get('detail')}",
                f"- Next: `{blocked.get('next_command_needed')}`.",
            ]
        )
    if payload.get("snapshots"):
        lines.extend(
            [
                "",
                "## Snapshot Results",
                "",
                "| surface | d01 | d02 | native PB/MUB match wrfinput | CPU-WRF PB/MUB match | worst primary field | max abs |",
                "| --- | ---: | ---: | --- | --- | --- | ---: |",
            ]
        )
        for label, snap in snapshot_order(payload):
            own = snap.get("own_steps", {})
            worst = snap.get("worst_primary_field") or {}
            lines.append(
                f"| `{label}` | {own.get('d01')} | {own.get('d02')} | "
                f"`{snap.get('native_wrfinput_static_base_match')}` | "
                f"`{snap.get('cpu_wrf_static_base_match')}` | "
                f"`{worst.get('field')}` | {worst.get('max_abs')} |"
            )
    lines.extend(["", "## Decision", "", str(payload.get("next_decision")), ""])
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Earlier-Source Bisection",
        "",
        f"verdict: `{payload['verdict']}`",
        "",
        "objective: bisect whether the bad h10 d02 OperationalCarry source is native load/initial carry or an earlier replay segment before completed step 5997.",
        "",
        "files changed:",
        "- `proofs/v014/earlier_source_bisect.py`",
        "- `proofs/v014/earlier_source_bisect.json`",
        "- `proofs/v014/earlier_source_bisect.md`",
        "- `.agent/reviews/2026-06-09-v014-earlier-source-bisect.md`",
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
            "- `proofs/v014/earlier_source_bisect.json`",
            "- `proofs/v014/earlier_source_bisect.md`",
            "- `.agent/reviews/2026-06-09-v014-earlier-source-bisect.md`",
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


def payload_for_cpu_validation_from_artifact() -> dict[str, Any]:
    payload = load_json(REPLAY_ARTIFACT_JSON)
    payload["cpu_validation_environment"] = prev.environment_snapshot()
    payload["cpu_validation_reused_compact_artifact_utc"] = datetime.now(timezone.utc).isoformat()
    payload["commands"] = required_commands()
    payload["proof_objects"] = proof_objects()
    return payload


def main() -> int:
    force_replay = truthy(os.environ.get("WRFGPU2_EARLIER_SOURCE_BISECT_FORCE_REPLAY"))
    allow_gpu = truthy(os.environ.get("WRFGPU2_EARLIER_SOURCE_BISECT_ALLOW_GPU"))

    if REPLAY_ARTIFACT_JSON.exists() and not force_replay:
        payload = payload_for_cpu_validation_from_artifact()
        write_outputs(payload)
        print(payload["verdict"])
        print(f"json={OUT_JSON}")
        print(f"markdown={OUT_MD}")
        print(f"review={OUT_REVIEW}")
        return 0

    cpu_preflight = prev.cpu_replay_preflight()
    if not allow_gpu:
        detail = (
            "CPU live replay is not practical in the current code path: "
            f"{cpu_preflight.get('detail') or 'CPU preflight failed'}."
        )
        payload = blocked_payload(
            "CPU_LOAD_REQUIRES_GPU",
            detail,
            (
                "Run a targeted GPU replay with WRFGPU2_EARLIER_SOURCE_BISECT_ALLOW_GPU=1, "
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
            "Inspect the exception and rerun proofs/v014/earlier_source_bisect.py with the same replay command.",
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
