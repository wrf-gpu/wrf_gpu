from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

from netCDF4 import Dataset
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v0234_corrected_fullbuffer_gate.py"
SPEC = importlib.util.spec_from_file_location("v0234_corrected_fullbuffer_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)
REAL_LOAD_FROZEN_GEOMETRY = gate.load_frozen_geometry
REAL_WRF_DEPENDENCY_MANIFEST = copy.deepcopy(gate.WRF_DEPENDENCY_MANIFEST)
REAL_WRF_LOADER_SOURCE_AUTHORITY = dict(gate.WRF_LOADER_SOURCE_AUTHORITY)


@pytest.fixture(autouse=True)
def synthetic_frozen_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    arrays = {
        "XLAT": np.full((93, 111), 28.2),
        "XLONG": np.full((93, 111), -16.5),
        "HGT": np.full((93, 111), 500.0),
        "LANDMASK": np.full((93, 111), 1.0),
    }
    monkeypatch.setattr(
        gate, "load_frozen_geometry",
        lambda: (arrays, {"path": "/synthetic/frozen-geo", "sha256": gate.GRID_SHA256}),
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_frame(
    path: Path,
    stamp: str,
    *,
    nonfinite: str | None = None,
    grid_id: int = 3,
    omit: set[str] | None = None,
) -> None:
    omit = omit or set()
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("bottom_top", 2)
        dataset.createDimension("south_north", 93)
        dataset.createDimension("west_east", 111)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = np.asarray(list(stamp), dtype="S1")
        dataset.GRID_ID = grid_id
        dataset.setncattr("WEST-EAST_GRID_DIMENSION", 112)
        dataset.setncattr("SOUTH-NORTH_GRID_DIMENSION", 94)
        values = {
            "T": 0.0, "U": 2.0, "V": -2.0, "W": 0.01,
            "T2": 290.0, "U10": 3.0, "V10": -1.0, "PSFC": 90000.0,
            "QVAPOR": 0.005, "RAINC": 0.0, "RAINNC": 0.0,
            "P": 5000.0, "PH": 100.0, "MU": 80000.0,
        }
        for name, value in values.items():
            if name in omit:
                continue
            variable = dataset.createVariable(
                name, "f8", ("Time", "bottom_top", "south_north", "west_east"),
            )
            variable[:] = value
            if nonfinite == name:
                variable[0, 0, 0, 0] = np.nan
        for name, value in {"XLAT": 28.2, "XLONG": -16.5, "HGT": 500.0, "LANDMASK": 1.0}.items():
            if name in omit:
                continue
            variable = dataset.createVariable(
                name, "f8", ("Time", "south_north", "west_east"),
            )
            variable[:] = value


def write_closed_first_pair(cpu_dir: Path, gpu_dir: Path) -> str:
    first = "wrfout_d03_2025-03-01_00:00:00"
    successor = "wrfout_d03_2025-03-01_00:20:00"
    write_frame(cpu_dir / first, "2025-03-01_00:00:00")
    shutil.copy2(cpu_dir / first, gpu_dir / first)
    write_frame(cpu_dir / successor, "2025-03-01_00:20:00")
    shutil.copy2(cpu_dir / successor, gpu_dir / successor)
    return first


def write_cpu_raw(path: Path, stamp: str, *, child: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    required = gate.CPU_CHILD_REQUIRED if child else gate.CPU_PARENT_REQUIRED
    values = {
        "T2": 290.0, "U10": 2.0, "V10": -1.0, "PSFC": 90000.0,
        "Q2": 0.005, "TSK": 291.0, "U": 1.0, "V": -1.0, "W": 0.01,
        "T": 0.0, "QVAPOR": 0.004, "PH": 100.0, "PHB": 1000.0,
        "HGT": 500.0, "XLAT": 28.2, "XLONG": -16.5,
        "LANDMASK": 1.0, "CLDFRA": 0.2,
    }
    with Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("bottom_top", 1)
        dataset.createDimension("south_north", 1)
        dataset.createDimension("west_east", 1)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = np.asarray(list(stamp), dtype="S1")
        for field in sorted(required - {"Times"}):
            dims = ("Time", "south_north", "west_east") if field in {"XLAT", "XLONG", "HGT", "LANDMASK"} else ("Time", "bottom_top", "south_north", "west_east")
            variable = dataset.createVariable(field, "f8", dims)
            variable[:] = values[field]


def write_cpu_thin(path: Path, stamp: str, raw_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "XLAT": 28.2, "XLONG": -16.5, "LANDMASK": 1.0, "HGT": 500.0,
        "U10": 2.0, "V10": -1.0, "T2": 290.0, "Q2": 0.005,
        "TSK": 291.0, "PSFC": 90000.0, "TCC": 0.2,
        "CLD_LOW": 0.1, "CLD_MID": 0.1, "CLD_HIGH": 0.0,
    }
    with Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("south_north", 1)
        dataset.createDimension("west_east", 1)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = np.asarray(list(stamp), dtype="S1")
        for field in sorted(gate.CPU_THIN_REQUIRED - {"Times"}):
            variable = dataset.createVariable(field, "f8", ("Time", "south_north", "west_east"))
            variable[:] = values[field]
        dataset.grid_id = gate.GRID_ID
        dataset.geo_em_sha256 = gate.GRID_SHA256
        dataset.source_raw = str(raw_path)


def namelist_text() -> str:
    return """&time_control
 run_days = 0,
 run_hours = 18,
 run_minutes = 0,
 run_seconds = 0,
/
&domains
 max_dom = 3,
 e_we = 100, 100, 112,
 e_sn = 100, 100, 94,
 i_parent_start = 1, 10, 92,
 j_parent_start = 1, 10, 36,
 parent_grid_ratio = 1, 3, 3,
/
&physics
 sf_surface_physics = 4, 4, 4,
 ra_lw_physics = 4, 4, 4,
 ra_sw_physics = 4, 4, 4,
/
"""


def cpu_identity() -> dict[str, object]:
    descendants = [
        {"pid": 4100 + index, "cpuset": "0-11", "command": "./wrf.exe"}
        for index in range(12)
    ]
    return {
        "pid": 4000,
        "start_ticks": 123456,
        "state": "S (sleeping)",
        "cpuset": "0-11",
        "command": " ".join(gate.PRTERUN_ARGV),
        "descendants": descendants,
        "observed_wrf_rank_count": 12,
    }


@pytest.fixture
def authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    fixture = tmp_path / "fixture"
    run_root = fixture / "run"
    wrf = run_root / "wrf"
    wrf.mkdir(parents=True)
    compatibility = fixture / "attempts/attempt_repaired_v2"
    compatibility.parent.mkdir()
    compatibility.symlink_to("../run")
    plan = json.loads(gate.LAUNCH_PLAN.read_text(encoding="utf-8"))
    plan["outputs"]["workspace"] = str(run_root)
    launch_path = fixture / "launch_plan.json"
    write_json(launch_path, plan)
    checksums = fixture / "provenance" / "checksums.sha256"
    launch_checksums = fixture / "provenance" / "checksums_launch_time.sha256"
    checksums.parent.mkdir(parents=True)
    checksums.write_text("current canonical provenance\n", encoding="utf-8")
    launch_checksums.write_text("historical launch provenance\n", encoding="utf-8")
    historical_plan = copy.deepcopy(plan)
    historical_plan["status"] = "repair_preflight_pass_awaiting_explicit_alisios_release"
    historical_plan["outputs"]["workspace"] = str(compatibility)
    historical_plan_sha = "2" * 64

    (wrf / "namelist.input").write_text(namelist_text(), encoding="utf-8")
    for name in gate.INPUT_NAMES[1:]:
        (wrf / name).write_bytes((name + "\n").encode())
    for path in (wrf / name for name in gate.INPUT_NAMES):
        path.chmod(0o444)

    qa_root = run_root / "early_gpu_ready_qa"
    groups = []
    for stamp in gate.EARLY_REQUIRED_STAMPS:
        raw = wrf / f"wrfout_d03_{stamp}"
        write_frame(raw, stamp)
        valid = datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
        independent = gate.qa_d03_frame(raw, valid)
        observed = raw.stat()
        qa = {
            "schema": gate.UPSTREAM_QA_SCHEMA,
            "checked_utc": "2026-07-12T00:01:00Z",
            "path": str(raw),
            "bytes": observed.st_size,
            "mtime_ns": observed.st_mtime_ns,
            "closed": True,
            "readable": True,
            "status": "pass",
            "valid_time": stamp,
            "variables": list(gate.STRICT_FIELDS),
            "ranges": {name: independent["ranges"][name] for name in gate.STRICT_FIELDS},
        }
        qa_path = qa_root / f"d03_{stamp.replace(':', '').replace('-', '').replace('_', 'T')}.json"
        write_json(qa_path, qa)
        groups.append({"valid_time": stamp, "raw_path": str(raw), "qa_evidence": str(qa_path), "bytes": observed.st_size})

    records = []
    for name in gate.INPUT_NAMES:
        path = wrf / name
        observed = path.stat()
        records.append({
            "name": name, "path": str(compatibility / "wrf" / name), "sha256": gate.sha256_file(path),
            "bytes": observed.st_size, "mtime_ns": observed.st_mtime_ns,
            "mode": stat.filemode(observed.st_mode), "open_by_pids": [],
        })
    provenance = {
        "checksums_path": str(checksums),
        "checksums_sha256": gate.sha256_file(launch_checksums),
        "launch_plan_path": str(launch_path),
        "launch_plan_sha256": historical_plan_sha,
        "launch_plan": historical_plan,
    }
    seal = {
        "schema": gate.UPSTREAM_SEAL_SCHEMA,
        "status": "sealed_immutable_monitored",
        "sealed_utc": "2026-07-12T00:00:00Z",
        "case_id": gate.CASE_ID,
        "grid_id": gate.GRID_ID,
        "grid_sha256": gate.GRID_SHA256,
        "immutability": "mode_0444_plus_continuous_sha256_monitor_until_cpu_exit",
        "files": records,
        "provenance": provenance,
    }
    seal_path = run_root / "input_seal.json"
    write_json(seal_path, seal)

    cpu = cpu_identity()
    resource = {
        "schema": gate.UPSTREAM_RESOURCE_SCHEMA,
        "status": "pass",
        "checked_utc": "2026-07-12T00:02:00Z",
        "cpu_oracle": cpu,
        "cpu_auxiliary": {
            "cpuset": "12-15", "free": True,
            "proof_method": "synthetic exact process-affinity inventory",
            "compute_blockers": [],
        },
        "gpu_lock": {
            "path": "/tmp/wrf_gpu2_gpu.lock", "free": True,
            "authority": "kernel_flock_exclusive_nonblocking_probe",
            "device": 42, "inode": 84, "holder_sidecar_at_probe": "",
        },
        "gpu_device": {
            "compute_query": [
                "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            "gpu_query": [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            "gpu": {
                "index": 0, "name": "NVIDIA GeForce RTX 5090",
                "memory_total_mib": 32607, "memory_used_mib": 2600,
                "memory_free_mib": 30000, "utilization_percent": 0,
            },
            "baseline_compute_processes": [], "unexpected_compute_processes": [],
            "minimum_free_mib": 24000, "maximum_utilization_percent": 20,
            "free": True,
        },
        "preemption": {
            "checked_utc": "2026-07-12T00:02:00Z",
            "sentinels_checked": [
                "/tmp/PREEMPT_PRODUCTION", "/tmp/PREEMPT_GPU",
                "<DATA_ROOT>/alisios/state/PREEMPT_PRODUCTION",
                "<DATA_ROOT>/alisios/state/PREEMPT_GPU",
            ],
            "sentinels_present": [],
            "nightly_active_path": "<DATA_ROOT>/alisios/state/nightly18z/active.json",
            "nightly_active": False,
        },
    }
    resource_path = run_root / "EARLY_GPU_READY_RESOURCE_PROOF.json"
    write_json(resource_path, resource)
    marker = {
        "schema": gate.UPSTREAM_MARKER_SCHEMA,
        "status": "EARLY_GPU_READY",
        "emitted_utc": "2026-07-12T00:03:00Z",
        "case_id": gate.CASE_ID,
        "grid_id": gate.GRID_ID,
        "grid_sha256": gate.GRID_SHA256,
        "input_seal_path": str(seal_path),
        "input_hashes": {record["name"]: record["sha256"] for record in records},
        "provenance": provenance,
        "cpu_oracle": cpu,
        "complete_regular_d03_groups": groups,
        "initialization_group_counted": False,
        "resource_proof_path": str(resource_path),
        "resource_proof": resource,
        "full_cpu_verdict": "pending_55_of_55",
        "gpu_contract": "consumer_must_acquire_authoritative_gpu_lock; final CPU QA and plot gates remain mandatory",
        "callback_targets": ["0:fable-mgr.0", "0:gpt-mgr.0"],
    }
    marker_path = run_root / "EARLY_GPU_READY.json"
    write_json(marker_path, marker)

    table_source = tmp_path / "wrf-source"
    table_source.mkdir()
    table_hashes = {}
    dependency_manifest = {}
    for relative, expected in gate.WRF_DEPENDENCY_MANIFEST.items():
        path = table_source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("frozen-" + relative).encode())
        dependency_manifest[relative] = {
            "sha256": gate.sha256_file(path), "size": path.stat().st_size,
            "resolved_relative_path": relative, "provenance": expected["provenance"],
        }
        if relative.startswith("run/"):
            table_hashes[Path(relative).name] = gate.sha256_file(path)

    monkeypatch.setattr(gate, "CPU_RUN_ROOT", run_root)
    monkeypatch.setattr(gate, "CPU_CASE_ROOT", fixture)
    monkeypatch.setattr(gate, "CPU_INPUT_DIR", wrf)
    monkeypatch.setattr(gate, "CPU_MANIFEST", run_root / "cpu_oracle_manifest.json")
    monkeypatch.setattr(gate, "EARLY_MARKER", marker_path)
    monkeypatch.setattr(gate, "EARLY_REVOKED", run_root / "EARLY_GPU_READY_REVOKED.json")
    monkeypatch.setattr(gate, "EARLY_INVALIDATED", run_root / "EARLY_GPU_READY.invalidated.json")
    monkeypatch.setattr(gate, "INPUT_SEAL", seal_path)
    monkeypatch.setattr(gate, "RESOURCE_PROOF", resource_path)
    monkeypatch.setattr(gate, "LAUNCH_PLAN", launch_path)
    monkeypatch.setattr(gate, "LAUNCH_PLAN_SHA256", gate.sha256_file(launch_path))
    monkeypatch.setattr(gate, "CURRENT_CHECKSUMS", checksums)
    monkeypatch.setattr(gate, "CURRENT_CHECKSUMS_SHA256", gate.sha256_file(checksums))
    monkeypatch.setattr(gate, "LAUNCH_TIME_CHECKSUMS", launch_checksums)
    monkeypatch.setattr(gate, "LAUNCH_TIME_CHECKSUMS_SHA256", gate.sha256_file(launch_checksums))
    monkeypatch.setattr(gate, "WRF_SOURCE_ROOT", table_source)
    monkeypatch.setattr(gate, "TABLE_HASHES", table_hashes)
    monkeypatch.setattr(gate, "WRF_DEPENDENCY_MANIFEST", dependency_manifest)
    monkeypatch.setattr(gate, "WRF_LOADER_SOURCE_AUTHORITY", {})
    monkeypatch.setattr(gate, "require_current_no_preemption", lambda: {
        "sentinels_present": [], "nightly_active": False,
        "nightly_active_path": "<DATA_ROOT>/alisios/state/nightly18z/active.json",
    })
    fake_canonical = {
        "contract": {"pre_wrf_seal": {"payload": seal}},
        "binding": {"input_seal_launch_plan_sha256": historical_plan_sha},
        "launch_plan": {"path": str(launch_path), "sha256": gate.sha256_file(launch_path), "payload": plan},
        "upstream": {"commit": gate.EARLY_AUTHORITY_COMMIT, "emitter_sha256": gate.EARLY_AUTHORITY_SOURCE_SHA256},
        "cpu_runtime": {"wrf_pid": cpu["pid"], "wrf_start_ticks": cpu["start_ticks"]},
        "contract_artifact": {"sha256": gate.CANONICAL_V2_CONTRACT_SHA256},
        "binding_artifact": {"sha256": gate.CANONICAL_BINDING_SHA256},
    }
    monkeypatch.setattr(gate, "validate_canonical_authority_v2", lambda: fake_canonical)
    return {
        "root": tmp_path, "run": run_root, "wrf": wrf, "marker": marker_path,
        "seal": seal_path, "resource": resource_path, "tables": table_source,
        "launch": launch_path, "plan": plan, "canonical": fake_canonical,
    }


@pytest.fixture
def terminal_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    case_root = tmp_path / "terminal-case"
    run_root = case_root / "run"
    wrf = run_root / "wrf"
    thin_root = run_root / "thin"
    wrf.mkdir(parents=True)
    thin_root.mkdir()
    monkeypatch.setattr(gate, "CPU_RUN_ROOT", run_root)
    monkeypatch.setattr(gate, "CPU_INPUT_DIR", wrf)
    monkeypatch.setattr(gate, "CPU_MANIFEST", run_root / "cpu_oracle_manifest.json")
    monkeypatch.setattr(gate, "CPU_PAIR_INDEX", run_root / "cpu_oracle_pair_index.jsonl")
    monkeypatch.setattr(gate, "CPU_THIN_DIR", thin_root)
    terminal_marker = run_root / "EARLY_GPU_READY.json"
    write_json(terminal_marker, {
        "schema": gate.UPSTREAM_MARKER_SCHEMA, "status": "EARLY_GPU_READY",
        "case_id": gate.CASE_ID, "grid_id": gate.GRID_ID, "grid_sha256": gate.GRID_SHA256,
    })
    monkeypatch.setattr(gate, "EARLY_MARKER", terminal_marker)
    monkeypatch.setattr(
        gate, "validate_input_seal",
        lambda *_args, **_kwargs: ({}, {}, {"synthetic": "already covered by seal adversaries"}),
    )

    static = case_root / "static/geo_em.d03.nc"
    static.parent.mkdir(parents=True)
    with Dataset(static, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("south_north", 1)
        dataset.createDimension("west_east", 1)
        for field, value in {"XLAT_M": 28.2, "XLONG_M": -16.5, "HGT_M": 500.0, "LANDMASK": 1.0}.items():
            variable = dataset.createVariable(field, "f8", ("Time", "south_north", "west_east"))
            variable[:] = value
    terminal_geo = {
        "XLAT": np.full((1, 1), 28.2), "XLONG": np.full((1, 1), -16.5),
        "HGT": np.full((1, 1), 500.0), "LANDMASK": np.full((1, 1), 1.0),
    }
    monkeypatch.setattr(
        gate, "load_frozen_geometry",
        lambda: (terminal_geo, {"path": str(static), "sha256": gate.GRID_SHA256}),
    )

    expected = {
        "d01": [value.strftime("%Y-%m-%d_%H:%M:%S") for value in gate.EXPECTED_PARENT_TIMES],
        "d02": [value.strftime("%Y-%m-%d_%H:%M:%S") for value in gate.EXPECTED_PARENT_TIMES],
        "d03": [value.strftime("%Y-%m-%d_%H:%M:%S") for value in gate.EXPECTED_TIMES],
    }
    raw: dict[str, dict[str, Path]] = {}
    for domain, stamps in expected.items():
        raw[domain] = {}
        for stamp in stamps:
            path = wrf / f"wrfout_{domain}_{stamp}"
            write_cpu_raw(path, stamp, child=domain == "d03")
            raw[domain][stamp] = path
    thin: dict[str, Path] = {}
    for stamp in expected["d03"]:
        path = thin_root / f"wrfout_d03_{stamp}.thin.nc"
        write_cpu_thin(path, stamp, raw["d03"][stamp])
        thin[stamp] = path
    pair_rows = gate._expected_cpu_pair_rows(raw, thin)
    gate.CPU_PAIR_INDEX.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pair_rows), encoding="utf-8",
    )
    surface_ranges = {
        "T2": [290.0, 290.0], "U10": [2.0, 2.0], "V10": [-1.0, -1.0],
        "PSFC": [90000.0, 90000.0], "Q2": [0.005, 0.005], "TSK": [291.0, 291.0],
    }
    manifest = {
        "schema": "tenerife_fullbuffer_cpu_oracle_v1", "status": "complete",
        "qa_status": "pass", "case_id": gate.CASE_ID, "grid_id": gate.GRID_ID,
        "grid_sha256": gate.GRID_SHA256, "engine": "cpu_wrf_max_dom3",
        "raw_frame_counts": {"d01": 19, "d02": 19, "d03": 55},
        "thin_frame_count": 55,
        "pair_counts": {"nine_to_three": 19, "aifs_to_1km_physical_target": 55, "three_to_one": 55},
        "pair_index": str(gate.CPU_PAIR_INDEX), "pair_index_rows": 129,
        "surface_ranges": surface_ranges, "all_required_fields_finite": True,
        "all_grid_fields_match_frozen_geo": True,
        "forcing_path": "<DATA_ROOT>/alisios/forcing/historical/aifs/20250228_18z/aifs_pure_wps_20250228_18z.grib2",
        "forcing_sha256": gate.AIFS_SHA256,
        "gpu_identity_contract": {
            "consumer": "wrf_gpu 0:1", "matched_raw_domain": "d03",
            "valid_times": expected["d03"], "exclude_fields": list(gate.REPORT_ONLY_FIELDS),
        },
        "eligibility": {
            "aifs_to_1km": "pending_se_cache_and_model_qa",
            "three_to_one": "pending_training_cache_and_cv",
            "nine_to_three": "unchanged_parent_pair_evidence",
        },
        "generated_utc": "2026-07-12T02:00:00Z",
    }
    write_json(gate.CPU_MANIFEST, manifest)
    return {"root": case_root, "run": run_root, "wrf": wrf, "thin": thin_root, "manifest": gate.CPU_MANIFEST, "pair_index": gate.CPU_PAIR_INDEX}


def build(authority: dict[str, object], suffix: str = "work") -> dict[str, object]:
    return gate.build_early_ready_proof(
        marker_path=authority["marker"],
        work_dir=authority["root"] / suffix,
        interval_seconds=0.0,
        sleep_fn=lambda _: None,
        live_identity_checker=lambda cpu: {
            "pid": cpu["pid"], "start_ticks": cpu["start_ticks"],
            "wrf_rank_pids": [row["pid"] for row in cpu["descendants"] if "wrf.exe" in row["command"]],
            "cpuset": "0-11",
        },
        gpu_baseline_checker=lambda _resource: [],
        now=datetime(2026, 7, 12, 0, 3, 30, tzinfo=timezone.utc),
    )


def mutate_json(path: Path, mutation) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    write_json(path, payload)


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "test-manager"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.invalid"], check=True)


def git_commit_all(path: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", message], check=True)
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def test_suite_is_pinned_to_cpu_12_15() -> None:
    assert set(os.sched_getaffinity(0)) == {12, 13, 14, 15}


@pytest.mark.parametrize("mode", ["audit", "command", "run"])
def test_actual_entry_rejects_superseded_early_live_modes(tmp_path: Path, mode: str) -> None:
    argv = [sys.executable, str(SCRIPT), mode, "--work-dir", str(tmp_path / "work")]
    if mode in {"command", "run"}:
        argv.extend(["--early-proof", str(tmp_path / "early.json")])
    if mode == "run":
        argv.extend(["--runtime-acceptance-path", "forbidden.json"])
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode == 75
    assert "EARLY_AUTHORITY_SUPERSEDED_BY_TERMINAL_CONTRACT" in result.stderr
    assert not (tmp_path / "work").exists()


def test_real_early_authority_commit_source_and_report_are_bound() -> None:
    result = gate.validate_upstream_early_authority()
    assert result["commit"] == gate.EARLY_AUTHORITY_COMMIT
    assert result["emitter_sha256"] == gate.EARLY_AUTHORITY_SOURCE_SHA256
    assert result["report_sha256"] == gate.EARLY_AUTHORITY_REPORT_SHA256


def test_real_terminal_authority_builds_no_live_no_marker_proof_and_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "load_frozen_geometry", REAL_LOAD_FROZEN_GEOMETRY)
    terminal = gate.validate_terminal_contract_authority(
        interval_seconds=0.0, sleep_fn=lambda _: None,
    )
    assert terminal["artifact"]["sha256"] == gate.TERMINAL_CONTRACT_SHA256
    assert terminal["terminal_cpu"]["recomputed"]["raw_counts"] == {"d01": 19, "d02": 19, "d03": 55}
    assert terminal["terminal_cpu"]["recomputed"]["thin_count"] == 55
    assert terminal["live_cpu_requirement"] is False
    assert terminal["early_marker_claim"] is False
    monkeypatch.setattr(gate, "validate_terminal_contract_authority", lambda *_args, **_kwargs: terminal)
    monkeypatch.setattr(gate, "require_current_no_preemption", lambda: {"nightly_active": False})
    work_dir = tmp_path / "future-run"
    proof = gate.build_terminal_cpu_proof(
        contract_path=gate.TERMINAL_CONTRACT, work_dir=work_dir,
        interval_seconds=0.0, sleep_fn=lambda _: None,
        now=datetime(2026, 7, 12, 7, 0, tzinfo=timezone.utc),
    )
    assert proof["status"] == "TERMINAL_CPU_AUTHORITY_READY"
    assert "cpu_live_identity" not in proof and "upstream_marker" not in proof
    run_repo = tmp_path / "candidate"
    for relative in gate.WRF_LOADER_SOURCE_AUTHORITY:
        target = run_repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gate.RUN_REPO / relative, target)
    monkeypatch.setattr(gate, "RUN_REPO", run_repo)
    command = gate.build_exact_command(
        early_proof=proof, work_dir=work_dir,
        candidate_authority={"path": str(run_repo), "head": gate.BASE_SHA, "clean": True, "detached": True},
        lock_authority={
            "root": str(gate.LOCK_ROOT), "commit": gate.LOCK_COMMIT,
            "wrapper": str(gate.LOCK_WRAPPER), "wrapper_sha256": gate.LOCK_WRAPPER_SHA256,
            "intent": gate.LOCK_INTENT,
        },
    )
    assert command["cpu_authority_mode"] == "terminal"
    assert command["cpu_authority_sha256"] == proof["canonical_authority_sha256"]
    assert command["argv"][:2] == ["/usr/bin/env", "-i"]
    assert command["argv"].index(str(gate.LOCK_WRAPPER)) > len(gate.COMMAND_ENV_KEYS)
    assert gate.validate_exact_command_argv(command) == command["argv_authority"]
    assert command["environment"]["JAX_PLATFORMS"] == "cuda"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.__setitem__("live_cpu_pid", 4038058),
        lambda p: p["terminal_cpu_process"].__setitem__("live_cpu_requirement", True),
        lambda p: p["terminal_cpu_process"].__setitem__("early_marker_claim", "EARLY_GPU_READY"),
        lambda p: p.__setitem__("gpu_authority", "AUTHORIZED"),
        lambda p: p["terminal_cpu_process"].__setitem__("wrf_exit_code", 1),
        lambda p: p["terminal_cpu_process"].__setitem__("wrapper_status", "success"),
        lambda p: p["corrected_cpu_qa"]["raw_frame_counts"].__setitem__("d01", 18),
        lambda p: p["parent_schedule_contract"]["d01"].__setitem__(1, "2025-03-01_01:00:00"),
        lambda p: p["parent_schedule_contract"]["d02"].__setitem__(1, "2025-03-01_01:00:18"),
        lambda p: p["parent_schedule_contract"]["d03"].pop(),
        lambda p: p["immutable_input_lineage"]["sealed_inputs"][4].__setitem__("sha256", "0" * 64),
        lambda p: p["source_authority"]["report"].__setitem__("sha256", "0" * 64),
        lambda p: p["immutable_input_lineage"]["canonical_binding"].__setitem__("sha256", "0" * 64),
        lambda p: p["early_marker_plan_supersession"].__setitem__("sha256", "0" * 64),
    ],
)
def test_terminal_contract_semantic_substitution_fails_after_rehash(
    tmp_path: Path, mutation,
) -> None:
    payload = json.loads(gate.TERMINAL_CONTRACT.read_text(encoding="utf-8"))
    mutation(payload)
    substitute = tmp_path / "terminal.json"
    write_json(substitute, payload)
    loaded, artifact = gate.stable_json_authority(
        substitute, interval_seconds=0.0, sleep_fn=lambda _: None,
    )
    assert artifact["sha256"] == gate.sha256_file(substitute)
    with pytest.raises(gate.GateError):
        gate.validate_terminal_contract_payload(loaded)


def test_terminal_runtime_acceptance_shape_has_no_marker_or_live_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime-authority"
    runtime_root.mkdir(mode=0o700)
    monkeypatch.setattr(gate, "RUNTIME_AUTHORITY_ROOT", runtime_root)
    input_authority = {
        name: {"sha256": str(index + 1) * 64}
        for index, name in enumerate(gate.INPUT_NAMES)
    }
    proof = {
        "status": "TERMINAL_CPU_AUTHORITY_READY",
        "canonical_authority_sha256": "a" * 64,
        "input_authority": input_authority,
        "terminal_authority": {
            "process": {"artifact": {"path": str(gate.TERMINAL_PROCESS_STATUS), "sha256": gate.TERMINAL_PROCESS_STATUS_SHA256}},
            "terminal_cpu": {"recomputed": {"manifest_artifact": {"path": str(gate.CPU_MANIFEST), "sha256": gate.TERMINAL_MANIFEST_SHA256}}},
        },
    }
    gpu_runtime = {
        "gpu": {"index": 0, "name": "NVIDIA GeForce RTX 5090", "memory_total_mib": 32607},
        "gpu_lock": {"path": "/tmp/wrf_gpu2_gpu.lock", "device": 42, "inode": 84},
        "baseline_compute_processes": [],
    }
    payload = gate._runtime_acceptance_expected(
        early=proof, command={"command_sha256": "b" * 64, "paths": {"work": "/future"}},
        tooling_commit="c" * 40, code_review_commit="d" * 40, nonce="e" * 64,
        gpu_runtime_authority=gpu_runtime,
    )
    assert payload["authority_mode"] == "terminal_cpu_no_live_pid_no_marker"
    assert "marker" not in payload and "cpu_live_identity" not in payload
    assert payload["terminal_contract"]["sha256"] == gate.TERMINAL_CONTRACT_SHA256


def test_terminal_runtime_acceptance_exact_manager_blob_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime-authority"
    runtime_root.mkdir(mode=0o700)
    monkeypatch.setattr(gate, "RUNTIME_AUTHORITY_ROOT", runtime_root)
    contract = json.loads(gate.TERMINAL_CONTRACT.read_text(encoding="utf-8"))
    proof = {
        "status": "TERMINAL_CPU_AUTHORITY_READY",
        "canonical_authority_sha256": "a" * 64,
        "input_authority": {name: {"sha256": str(index + 1) * 64} for index, name in enumerate(gate.INPUT_NAMES)},
        "terminal_authority": {
            "contract": contract,
            "process": {"artifact": {"path": str(gate.TERMINAL_PROCESS_STATUS), "sha256": gate.TERMINAL_PROCESS_STATUS_SHA256}},
            "terminal_cpu": {"recomputed": {"manifest_artifact": {"path": str(gate.CPU_MANIFEST), "sha256": gate.TERMINAL_MANIFEST_SHA256}}},
        },
    }
    command = {"command_sha256": "b" * 64, "paths": {"work": "/future"}}
    gpu_runtime = {
        "gpu": {"index": 0, "name": "NVIDIA GeForce RTX 5090", "memory_total_mib": 32607},
        "gpu_lock": {"path": "/tmp/wrf_gpu2_gpu.lock", "device": 42, "inode": 84},
        "baseline_compute_processes": [],
    }
    now = datetime.now(timezone.utc).replace(microsecond=0)
    payload = gate._runtime_acceptance_expected(
        early=proof, command=command, tooling_commit="c" * 40,
        code_review_commit="d" * 40, nonce="e" * 64,
        gpu_runtime_authority=gpu_runtime,
    )
    payload["issued_utc"] = (now - gate.timedelta(seconds=1)).isoformat()
    payload["expires_utc"] = (now + gate.timedelta(seconds=60)).isoformat()
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    monkeypatch.setattr(gate, "validate_code_review_acceptance", lambda: {
        "commit": "d" * 40, "payload": {"tooling_commit": "c" * 40},
    })
    monkeypatch.setattr(gate, "validate_executing_tooling", lambda commit: {"commit": commit})
    monkeypatch.setattr(gate, "_git_blob_at_ref", lambda *_args: ("f" * 40, raw))

    def checked_output(args, **kwargs):
        if "--format=%ct" in args:
            return str(int(now.timestamp())) + "\n"
        return gate.MANAGER_POLICY.read_bytes()

    monkeypatch.setattr(gate.subprocess, "check_output", checked_output)
    accepted = gate.validate_runtime_acceptance(
        relative_path=f"{gate.RUNTIME_ACCEPTANCE_PREFIX}{'e' * 64}.json",
        command=command, early=proof, now=now,
    )
    assert accepted["payload"]["authority_mode"] == "terminal_cpu_no_live_pid_no_marker"
    assert "marker" not in accepted["payload"] and "cpu_live_identity" not in accepted["payload"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("pid", True), ("start_ticks", 0), ("process_name", ""), ("exe_inode", -1)],
)
def test_terminal_manager_gpu_baseline_malformed_identity_fails(
    field: str, value: object,
) -> None:
    row: dict[str, object] = {
        "pid": 42, "process_name": "/opt/google/chrome/chrome --type=gpu-process",
        "start_ticks": 123, "argv0": "/opt/google/chrome/chrome",
        "exe_path": "/opt/google/chrome/chrome", "exe_device": 8, "exe_inode": 9,
    }
    row[field] = value
    with pytest.raises(gate.GateError, match="TERMINAL_GPU_RUNTIME_BASELINE"):
        gate.validate_terminal_gpu_runtime_authority({
            "gpu": {"index": 0, "name": "NVIDIA GeForce RTX 5090", "memory_total_mib": 32607},
            "gpu_lock": {"path": "/tmp/wrf_gpu2_gpu.lock", "device": 42, "inode": 84},
            "baseline_compute_processes": [row],
        })


def test_real_canonical_v2_authority_and_distinct_provenance_are_bound() -> None:
    assert gate.sha256_file(gate.CANONICAL_V2_CONTRACT) == gate.CANONICAL_V2_CONTRACT_SHA256
    assert gate.sha256_file(gate.CANONICAL_BINDING) == gate.CANONICAL_BINDING_SHA256
    assert gate.sha256_file(gate.LAUNCH_TIME_CHECKSUMS) == gate.LAUNCH_TIME_CHECKSUMS_SHA256
    assert gate.sha256_file(gate.CURRENT_CHECKSUMS) == gate.CURRENT_CHECKSUMS_SHA256
    with pytest.raises(gate.GateError, match="CANONICAL_LAUNCH_RUNTIME"):
        gate.validate_canonical_authority_v2()
    retired = json.loads((gate.CPU_RUN_ROOT / "launch_runtime.json").read_text(encoding="utf-8"))
    assert retired["status"] == "failed_final_qa"
    assert retired["authority"] == "revoked_until_early_gpu_ready_marker"
    assert not gate.EARLY_MARKER.exists()


def install_rehashed_binding_and_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
) -> tuple[dict[str, object], dict[str, object]]:
    binding = json.loads(gate.CANONICAL_BINDING.read_text(encoding="utf-8"))
    mutation(binding)
    binding_path = tmp_path / "canonical_run_binding.json"
    write_json(binding_path, binding)
    binding_hash = gate.sha256_file(binding_path)
    contract = json.loads(gate.CANONICAL_V2_CONTRACT.read_text(encoding="utf-8"))
    contract["canonical_promotion"] = {"path": str(binding_path), "sha256": binding_hash}
    contract_path = tmp_path / "canonical_authority_contract_v2.json"
    write_json(contract_path, contract)
    monkeypatch.setattr(gate, "CANONICAL_BINDING", binding_path)
    monkeypatch.setattr(gate, "CANONICAL_BINDING_SHA256", binding_hash)
    monkeypatch.setattr(gate, "CANONICAL_V2_CONTRACT", contract_path)
    monkeypatch.setattr(gate, "CANONICAL_V2_CONTRACT_SHA256", gate.sha256_file(contract_path))
    return binding, contract


def test_binding_launch_checksum_equation_rejects_current_role_after_full_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_rehashed_binding_and_contract(
        tmp_path, monkeypatch,
        lambda binding: binding.__setitem__(
            "input_seal_checksums_sha256", gate.CURRENT_CHECKSUMS_SHA256,
        ),
    )
    with pytest.raises(gate.GateError, match="CANONICAL_BINDING_CHECKSUM_ROLE"):
        gate.validate_canonical_authority_v2()


def test_binding_historical_plan_equation_rejects_current_role_after_full_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_rehashed_binding_and_contract(
        tmp_path, monkeypatch,
        lambda binding: binding.__setitem__("input_seal_launch_plan_sha256", gate.LAUNCH_PLAN_SHA256),
    )
    with pytest.raises(gate.GateError, match="CANONICAL_BINDING_LAUNCH_PLAN_ROLE"):
        gate.validate_canonical_authority_v2()


def test_binding_two_way_current_role_swap_fails_after_full_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def collapse(binding: dict[str, object]) -> None:
        binding["input_seal_checksums_sha256"] = gate.CURRENT_CHECKSUMS_SHA256
        binding["input_seal_launch_plan_sha256"] = gate.LAUNCH_PLAN_SHA256

    install_rehashed_binding_and_contract(tmp_path, monkeypatch, collapse)
    with pytest.raises(
        gate.GateError,
        match="CANONICAL_BINDING_CHECKSUM_ROLE|CANONICAL_BINDING_LAUNCH_PLAN_ROLE",
    ):
        gate.validate_canonical_authority_v2()


@pytest.mark.parametrize(
    ("substitute_pid", "fd_present", "error"),
    [
        (4000, False, "CANONICAL_LIVE_FD_RANK"),
        (4101, False, "CANONICAL_LIVE_FD_DESCRIPTOR"),
    ],
)
def test_contract_live_fd_pid_substitution_fails_after_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitute_pid: int,
    fd_present: bool,
    error: str,
) -> None:
    contract = json.loads(gate.CANONICAL_V2_CONTRACT.read_text(encoding="utf-8"))
    contract["live_open_fd"]["pid"] = substitute_pid
    substitute = tmp_path / "canonical_authority_contract_v2.json"
    write_json(substitute, contract)
    loaded, artifact = gate.stable_json_authority(
        substitute, interval_seconds=0.0, sleep_fn=lambda _: None,
    )
    assert artifact["sha256"] == gate.sha256_file(substitute)
    runtime = {"wrf_pid": 4000, "wrf_start_ticks": 123456}
    target = tmp_path / "wrfbdy_d01"
    target.write_bytes(b"sealed")
    sealed = {"wrfbdy_d01": gate.file_authority(target)}

    def live(pid: int) -> dict[str, object]:
        is_root = pid == runtime["wrf_pid"]
        return {
            "start_ticks": 123456 if is_root else 234567,
            "parent_pid": 1 if is_root else runtime["wrf_pid"],
            "status": {"Cpus_allowed_list": "0-11"},
            "command": " ".join(gate.PRTERUN_ARGV) if is_root else "./wrf.exe",
        }

    monkeypatch.setattr(gate, "_live_cpu_proc", live)
    monkeypatch.setattr(gate, "_is_descendant_of", lambda pid, root: (pid != root, [pid]))
    monkeypatch.setattr(
        gate, "_stable_relevant_fd_observation",
        lambda *_args: ({
            "pid": substitute_pid, "process_start_ticks": 234567, "fd": 37,
            "flags": os.O_RDONLY, "access": "read_only",
            "device": sealed["wrfbdy_d01"]["device"],
            "inode": sealed["wrfbdy_d01"]["inode"], "path_identity_match": True,
        } if fd_present else None),
    )
    with pytest.raises(gate.GateError, match=error):
        gate.validate_contract_live_fd_rank(loaded["live_open_fd"], runtime, sealed)


def test_superseded_d7_contract_hash_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gate, "CANONICAL_V2_CONTRACT_SHA256",
        "d7b266bf490e72b8c35977995defd4e6357bc98b5e85467fba4a26ebf30ad877",
    )
    with pytest.raises(gate.GateError, match="CANONICAL_V2_CONTRACT_HASH|CANONICAL_V2_SUPERSEDED"):
        gate.validate_canonical_authority_v2()


@pytest.mark.parametrize("role", ["status", "emitter", "report"])
def test_contract_semantic_drift_fails_after_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str,
) -> None:
    contract = json.loads(gate.CANONICAL_V2_CONTRACT.read_text(encoding="utf-8"))
    if role == "status":
        contract["status"] = "substituted"
    else:
        contract[role]["sha256"] = "0" * 64
    substitute = tmp_path / "canonical_authority_contract_v2.json"
    write_json(substitute, contract)
    monkeypatch.setattr(gate, "CANONICAL_V2_CONTRACT", substitute)
    monkeypatch.setattr(gate, "CANONICAL_V2_CONTRACT_SHA256", gate.sha256_file(substitute))
    with pytest.raises(gate.GateError, match="CANONICAL_V2_CONTRACT_FIELD|CANONICAL_SOURCE_BINDING"):
        gate.validate_canonical_authority_v2()


def test_binding_semantic_drift_fails_after_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = json.loads(gate.CANONICAL_BINDING.read_text(encoding="utf-8"))
    binding["status"] = "substituted"
    substitute = tmp_path / "canonical_run_binding.json"
    write_json(substitute, binding)
    monkeypatch.setattr(gate, "CANONICAL_BINDING", substitute)
    monkeypatch.setattr(gate, "CANONICAL_BINDING_SHA256", gate.sha256_file(substitute))
    with pytest.raises(gate.GateError, match="CANONICAL_BINDING_FIELD"):
        gate.validate_canonical_authority_v2()


def test_forcing_semantic_drift_fails_after_full_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    forcing = json.loads(gate.FORCING_PREFLIGHT.read_text(encoding="utf-8"))
    forcing["status"] = "substituted"
    forcing_path = tmp_path / "forcing.json"
    write_json(forcing_path, forcing)
    forcing_hash = gate.sha256_file(forcing_path)
    contract = json.loads(gate.CANONICAL_V2_CONTRACT.read_text(encoding="utf-8"))
    contract["forcing_preflight"] = {"path": str(forcing_path), "sha256": forcing_hash}
    contract_path = tmp_path / "contract.json"
    write_json(contract_path, contract)
    monkeypatch.setattr(gate, "FORCING_PREFLIGHT", forcing_path)
    monkeypatch.setattr(gate, "FORCING_PREFLIGHT_SHA256", forcing_hash)
    monkeypatch.setattr(gate, "CANONICAL_V2_CONTRACT", contract_path)
    monkeypatch.setattr(gate, "CANONICAL_V2_CONTRACT_SHA256", gate.sha256_file(contract_path))
    with pytest.raises(gate.GateError, match="CANONICAL_FORCING_FIELD"):
        gate.validate_canonical_authority_v2()


def test_old_launch_plan_substitution_fails_after_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = json.loads(gate.LAUNCH_PLAN.read_text(encoding="utf-8"))
    plan["status"] = "repair_preflight_pass_awaiting_explicit_alisios_release"
    substitute = tmp_path / "launch_plan.json"
    write_json(substitute, plan)
    monkeypatch.setattr(gate, "LAUNCH_PLAN", substitute)
    monkeypatch.setattr(gate, "LAUNCH_PLAN_SHA256", gate.sha256_file(substitute))
    with pytest.raises(gate.GateError, match="LAUNCH_PLAN_FIELD|LAUNCH_PLAN_SEMANTICS"):
        gate.validate_launch_plan(substitute)


def test_compatibility_symlink_cannot_be_replaced_by_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    compatibility = tmp_path / "attempts/attempt_repaired_v2"
    compatibility.parent.mkdir()
    compatibility.symlink_to("../run")
    monkeypatch.setattr(gate, "CPU_RUN_ROOT", run)
    binding = {"compatibility_symlink": {"path": str(compatibility), "target": "../run"}}
    gate._validate_compatibility_symlink(binding, compatibility)
    compatibility.unlink()
    compatibility.write_text("../run\n", encoding="utf-8")
    with pytest.raises(gate.GateError, match="CANONICAL_COMPATIBILITY_SYMLINK"):
        gate._validate_compatibility_symlink(binding, compatibility)


@pytest.mark.parametrize(
    ("access", "flags", "identity_match", "error"),
    [
        ("read_write", os.O_RDWR, True, "INPUT_FD_WRITABLE"),
        ("read_only", os.O_WRONLY, True, "INPUT_FD_WRITABLE"),
        ("read_only", os.O_RDONLY, False, "INPUT_FD_INODE_MISMATCH"),
    ],
)
def test_current_input_fd_substitution_fails_closed(
    access: str, flags: int, identity_match: bool, error: str,
) -> None:
    authority = {
        "wrfbdy_d01": {
            "path": "/canonical/wrfbdy_d01", "device": 42, "inode": 84,
            "mode": 0o444, "sha256": "a" * 64,
        },
    }
    row = {
        "pid": 123, "process_start_ticks": 456, "fd": 37,
        "flags": flags, "access": access,
        "device": 42, "inode": 84, "path_identity_match": identity_match,
    }
    with pytest.raises(gate.GateError, match=error):
        gate.validate_input_fd_observations(authority, [row])


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("path_identity_match", None, "INPUT_FD_TYPE_OR_RANGE"),
        ("path_identity_match", "false", "INPUT_FD_TYPE_OR_RANGE"),
        ("path_identity_match", False, "INPUT_FD_INODE_MISMATCH"),
        ("pid", True, "INPUT_FD_TYPE_OR_RANGE"),
        ("pid", 1, "INPUT_FD_TYPE_OR_RANGE"),
        ("pid", -2, "INPUT_FD_TYPE_OR_RANGE"),
        ("pid", "123", "INPUT_FD_TYPE_OR_RANGE"),
        ("process_start_ticks", True, "INPUT_FD_TYPE_OR_RANGE"),
        ("process_start_ticks", 0, "INPUT_FD_TYPE_OR_RANGE"),
        ("process_start_ticks", "456", "INPUT_FD_TYPE_OR_RANGE"),
        ("fd", True, "INPUT_FD_TYPE_OR_RANGE"),
        ("fd", -1, "INPUT_FD_TYPE_OR_RANGE"),
        ("fd", "37", "INPUT_FD_TYPE_OR_RANGE"),
        ("flags", True, "INPUT_FD_TYPE_OR_RANGE"),
        ("flags", -1, "INPUT_FD_TYPE_OR_RANGE"),
        ("flags", "0", "INPUT_FD_TYPE_OR_RANGE"),
        ("device", True, "INPUT_FD_TYPE_OR_RANGE"),
        ("device", 0, "INPUT_FD_TYPE_OR_RANGE"),
        ("device", "42", "INPUT_FD_TYPE_OR_RANGE"),
        ("inode", True, "INPUT_FD_TYPE_OR_RANGE"),
        ("inode", 0, "INPUT_FD_TYPE_OR_RANGE"),
        ("inode", "84", "INPUT_FD_TYPE_OR_RANGE"),
        ("access", None, "INPUT_FD_TYPE_OR_RANGE"),
        ("access", 0, "INPUT_FD_TYPE_OR_RANGE"),
        ("access", "read_write", "INPUT_FD_WRITABLE"),
    ],
)
def test_malformed_fd_evidence_types_and_ranges_fail_closed(
    field: str, value: object, error: str,
) -> None:
    authority = {
        "wrfbdy_d01": {
            "path": "/canonical/wrfbdy_d01", "device": 42, "inode": 84,
            "mode": 0o444, "sha256": "a" * 64,
        },
    }
    row: dict[str, object] = {
        "pid": 123, "process_start_ticks": 456, "fd": 37,
        "flags": os.O_RDONLY, "access": "read_only",
        "device": 42, "inode": 84, "path_identity_match": True,
    }
    row[field] = value
    with pytest.raises(gate.GateError, match=error):
        gate.validate_input_fd_observations(authority, [row])


@pytest.mark.parametrize("changed_field", ["process_start_ticks", "flags", "inode", "target"])
def test_relevant_fd_reuse_or_race_fails_double_snapshot(
    monkeypatch: pytest.MonkeyPatch, changed_field: str,
) -> None:
    anchor = {
        "pid": 123, "process_start_ticks": 456, "fd": 37,
        "device": 42, "inode": 84, "target": "/canonical/wrfbdy_d01",
    }
    first = {**anchor, "flags": os.O_RDONLY}
    second = dict(first)
    second[changed_field] = {
        "process_start_ticks": 457, "flags": os.O_RDWR,
        "inode": 85, "target": "/replacement/wrfbdy_d01",
    }[changed_field]
    snapshots = iter((first, second))
    monkeypatch.setattr(gate, "_fd_target_snapshot", lambda *_args: dict(anchor))
    monkeypatch.setattr(gate, "_fd_snapshot", lambda *_args: dict(next(snapshots)))
    with pytest.raises(gate.GateError, match="INPUT_FD_CHANGED_DURING_SCAN"):
        gate._stable_relevant_fd_observation(
            Path("/proc/123"), Path("/proc/123/fd/37"),
            {(42, 84): "wrfbdy_d01"}, {"/canonical/wrfbdy_d01": "wrfbdy_d01"},
        )


def test_relevant_fd_inaccessible_on_second_read_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = {
        "pid": 123, "process_start_ticks": 456, "fd": 37,
        "device": 42, "inode": 84, "target": "/canonical/wrfbdy_d01",
    }
    calls = 0

    def snapshot(*_args) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("descriptor closed")
        return {**anchor, "flags": os.O_RDONLY}

    monkeypatch.setattr(gate, "_fd_target_snapshot", lambda *_args: dict(anchor))
    monkeypatch.setattr(gate, "_fd_snapshot", snapshot)
    with pytest.raises(gate.GateError, match="INPUT_FD_CHANGED_DURING_SCAN"):
        gate._stable_relevant_fd_observation(
            Path("/proc/123"), Path("/proc/123/fd/37"),
            {(42, 84): "wrfbdy_d01"}, {"/canonical/wrfbdy_d01": "wrfbdy_d01"},
        )


def test_wrong_prterun_argv_binary_and_pid_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu = cpu_identity()
    runtime = {"wrf_pid": cpu["pid"], "wrf_start_ticks": cpu["start_ticks"]}
    cpu["command"] = "mpirun -np 12 ./wrf.exe"
    with pytest.raises(gate.GateError, match="CPU_IDENTITY"):
        gate.validate_cpu_identity_payload(cpu, runtime)
    cpu = cpu_identity()
    cpu["pid"] += 1
    with pytest.raises(gate.GateError, match="CPU_IDENTITY"):
        gate.validate_cpu_identity_payload(cpu, runtime)
    monkeypatch.setattr(gate, "PRTE_SHA256", "0" * 64)
    with pytest.raises(gate.GateError, match="PRTERUN_BINARY"):
        gate.validate_prterun_authority()


def test_two_regular_frames_pass_while_final_remains_pending(authority: dict[str, object]) -> None:
    proof = build(authority)
    assert [row["upstream"]["valid_time"] for row in proof["early_frame_qa"]] == list(gate.EARLY_REQUIRED_STAMPS)
    assert proof["upstream_marker"]["initialization_group_counted"] is False
    assert proof["final_verdict"] == "FINAL_PENDING_CPU_55_OF_55"
    pair_state = authority["run"] / "incremental-pairs.json"
    gpu_dir = authority["run"] / "gpu-output"
    assert gate.final_verdict(
        gate.CPU_MANIFEST, pair_state,
        authority["run"] / "identity-numbers-first.jpg", gpu_dir,
    )["verdict"] == "FINAL_PENDING"


def test_attempt_local_marker_alias_is_rejected(authority: dict[str, object]) -> None:
    attempt_marker = authority["root"] / "fixture/attempts/attempt_repaired_v2/EARLY_GPU_READY.json"
    assert attempt_marker.resolve() == authority["marker"].resolve()
    with pytest.raises(gate.GateError, match="EARLY_MARKER_CANONICAL_PATH"):
        gate.build_early_ready_proof(
            marker_path=attempt_marker,
            work_dir=authority["root"] / "attempt-local-work",
            interval_seconds=0.0,
            sleep_fn=lambda _: None,
        )


def test_historical_provenance_cannot_be_rewritten_as_current(
    authority: dict[str, object],
) -> None:
    marker = json.loads(authority["marker"].read_text(encoding="utf-8"))
    seal = json.loads(authority["seal"].read_text(encoding="utf-8"))
    current_sha = gate.LAUNCH_PLAN_SHA256
    current_plan = authority["plan"]
    for payload in (marker, seal):
        payload["provenance"]["launch_plan_sha256"] = current_sha
        payload["provenance"]["launch_plan"] = current_plan
    write_json(authority["marker"], marker)
    write_json(authority["seal"], seal)
    canonical = authority["canonical"]
    canonical["contract"]["pre_wrf_seal"]["payload"] = seal
    canonical["binding"]["input_seal_launch_plan_sha256"] = current_sha
    with pytest.raises(gate.GateError, match="PROVENANCE_FALSE_CURRENT_EQUIVALENCE"):
        build(authority)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "substitute"),
        ("status", "NOT_READY"),
        ("grid_sha256", "0" * 64),
        ("initialization_group_counted", True),
        ("full_cpu_verdict", "pass"),
        ("callback_targets", ["attacker"]),
    ],
)
def test_marker_semantic_substitutions_fail_closed(authority: dict[str, object], field: str, value: object) -> None:
    mutate_json(authority["marker"], lambda payload: payload.__setitem__(field, value))
    with pytest.raises(gate.GateError):
        build(authority)


def test_marker_extra_field_fails_exact_schema(authority: dict[str, object]) -> None:
    mutate_json(authority["marker"], lambda payload: payload.__setitem__("alternate_policy", True))
    with pytest.raises(gate.GateError, match="EARLY_MARKER_SCHEMA"):
        build(authority)


def test_atomic_marker_replacement_during_stability_fails(authority: dict[str, object]) -> None:
    called = False

    def replace(_: float) -> None:
        nonlocal called
        if not called:
            called = True
            payload = json.loads(authority["marker"].read_text())
            replacement = authority["marker"].with_suffix(".new")
            write_json(replacement, payload)
            os.replace(replacement, authority["marker"])

    with pytest.raises(gate.GateError, match="ATOMIC_AUTHORITY_CHANGED"):
        gate.stable_json_authority(authority["marker"], interval_seconds=0, sleep_fn=replace)


def test_stable_json_identity_includes_ctime_and_mode(tmp_path: Path) -> None:
    path = tmp_path / "authority.json"
    write_json(path, {"status": "stable"})
    original_mode = stat.S_IMODE(path.stat().st_mode)

    def change_ctime_without_changing_bytes_or_final_mode(_: float) -> None:
        path.chmod(0o600)
        path.chmod(original_mode)

    with pytest.raises(gate.GateError, match="ATOMIC_AUTHORITY_CHANGED"):
        gate.stable_json_authority(path, interval_seconds=0.0, sleep_fn=change_ctime_without_changing_bytes_or_final_mode)


def test_duplicate_key_marker_fails_safe_deserialization(authority: dict[str, object]) -> None:
    authority["marker"].write_text('{"schema":"first","schema":"second"}\n', encoding="utf-8")
    with pytest.raises(gate.GateError, match="INVALID_JSON"):
        build(authority)


def test_revocation_requires_stop_and_lock_release(authority: dict[str, object]) -> None:
    write_json(authority["run"] / "EARLY_GPU_READY_REVOKED.json", {
        "schema": gate.UPSTREAM_REVOKE_SCHEMA,
        "status": "revoked_gpu_stop_required",
        "gpu_action": "STOP_FAIL_CLOSED_AND_RELEASE_LOCK",
    })
    with pytest.raises(gate.GateError, match="EARLY_REVOKED_STOP_AND_RELEASE"):
        build(authority)


def test_malformed_revocation_still_fails_closed(authority: dict[str, object]) -> None:
    write_json(authority["run"] / "EARLY_GPU_READY_REVOKED.json", {"status": "ignore"})
    with pytest.raises(gate.GateError):
        build(authority)


def test_input_hash_substitution_fails_without_rehash_trust(authority: dict[str, object]) -> None:
    mutate_json(authority["marker"], lambda payload: payload["input_hashes"].__setitem__("wrfbdy_d01", "0" * 64))
    with pytest.raises(gate.GateError, match="INPUT_HASH_BINDING"):
        build(authority)


def test_sealed_input_content_mutation_fails(authority: dict[str, object]) -> None:
    target = authority["wrf"] / "wrfinput_d03"
    target.chmod(0o644)
    target.write_bytes(b"substituted")
    target.chmod(0o444)
    with pytest.raises(gate.GateError, match="INPUT_MUTATION"):
        build(authority)


def test_sealed_input_writable_mode_fails(authority: dict[str, object]) -> None:
    (authority["wrf"] / "wrfinput_d02").chmod(0o644)
    with pytest.raises(gate.GateError, match="INPUT_IMMUTABILITY"):
        build(authority)


def test_input_mutation_across_bounded_stability_interval_fails(authority: dict[str, object]) -> None:
    target = authority["wrf"] / "wrfbdy_d01"

    def mutate(_: float) -> None:
        target.chmod(0o644)
        target.write_bytes(b"changed during interval")
        target.chmod(0o444)

    with pytest.raises(gate.GateError, match="UNSTABLE_FILE"):
        gate.stable_file_authorities([target], interval_seconds=0, sleep_fn=mutate)


def test_provenance_launch_plan_substitution_fails(authority: dict[str, object]) -> None:
    mutate_json(authority["seal"], lambda payload: payload["provenance"].__setitem__("launch_plan_sha256", "0" * 64))
    marker = json.loads(authority["marker"].read_text())
    marker["provenance"] = json.loads(authority["seal"].read_text())["provenance"]
    write_json(authority["marker"], marker)
    with pytest.raises(gate.GateError, match="PROVENANCE_HISTORICAL_PAYLOAD"):
        build(authority)


@pytest.mark.parametrize(
    ("path", "field", "value"),
    [
        (("cpu_oracle",), "observed_wrf_rank_count", 11),
        (("cpu_oracle",), "cpuset", "0-10"),
        (("cpu_auxiliary",), "free", False),
        (("gpu_lock",), "free", False),
        (("gpu_device",), "free", False),
        (("preemption",), "nightly_active", True),
    ],
)
def test_resource_semantic_adversaries_fail(authority: dict[str, object], path: tuple[str, ...], field: str, value: object) -> None:
    resource = json.loads(authority["resource"].read_text())
    resource[path[0]][field] = value
    write_json(authority["resource"], resource)
    marker = json.loads(authority["marker"].read_text())
    marker["resource_proof"] = resource
    if path == ("cpu_oracle",):
        marker["cpu_oracle"] = resource["cpu_oracle"]
    write_json(authority["marker"], marker)
    with pytest.raises(gate.GateError):
        build(authority)


def test_embedded_resource_different_from_atomic_file_fails(authority: dict[str, object]) -> None:
    mutate_json(authority["marker"], lambda payload: payload["resource_proof"]["gpu_lock"].__setitem__("inode", 999))
    with pytest.raises(gate.GateError, match="RESOURCE_PROOF_BINDING"):
        build(authority)


def test_early_group_0000_substitution_fails(authority: dict[str, object]) -> None:
    mutate_json(authority["marker"], lambda payload: payload["complete_regular_d03_groups"][0].__setitem__("valid_time", "2025-03-01_00:00:00"))
    with pytest.raises(gate.GateError, match="EARLY_GROUP_TIMES"):
        build(authority)


def test_missing_early_frame_fails_closed(authority: dict[str, object]) -> None:
    marker = json.loads(authority["marker"].read_text())
    Path(marker["complete_regular_d03_groups"][1]["raw_path"]).unlink()
    with pytest.raises(gate.GateError, match="MISSING_FILE"):
        build(authority)


def test_wrong_netcdf_time_fails_independent_qa(authority: dict[str, object]) -> None:
    marker = json.loads(authority["marker"].read_text())
    group = marker["complete_regular_d03_groups"][0]
    path = Path(group["raw_path"])
    write_frame(path, "2025-03-01_00:00:00")
    evidence_path = Path(group["qa_evidence"])
    evidence = json.loads(evidence_path.read_text())
    observed = path.stat()
    evidence["bytes"] = observed.st_size
    evidence["mtime_ns"] = observed.st_mtime_ns
    group["bytes"] = observed.st_size
    write_json(evidence_path, evidence)
    write_json(authority["marker"], marker)
    with pytest.raises(gate.GateError, match="FRAME_TIME"):
        build(authority)


def test_nonfinite_early_frame_fails_independent_qa(authority: dict[str, object]) -> None:
    group = json.loads(authority["marker"].read_text())["complete_regular_d03_groups"][0]
    path = Path(group["raw_path"])
    write_frame(path, group["valid_time"], nonfinite="T")
    evidence_path = Path(group["qa_evidence"])
    evidence = json.loads(evidence_path.read_text())
    observed = path.stat()
    evidence["bytes"] = observed.st_size
    evidence["mtime_ns"] = observed.st_mtime_ns
    group["bytes"] = observed.st_size
    write_json(evidence_path, evidence)
    marker = json.loads(authority["marker"].read_text())
    marker["complete_regular_d03_groups"][0] = group
    write_json(authority["marker"], marker)
    with pytest.raises(gate.GateError, match="FRAME_NONFINITE"):
        build(authority)


def test_wrong_grid_early_frame_fails(authority: dict[str, object]) -> None:
    group = json.loads(authority["marker"].read_text())["complete_regular_d03_groups"][0]
    path = Path(group["raw_path"])
    write_frame(path, group["valid_time"], grid_id=2)
    evidence_path = Path(group["qa_evidence"])
    evidence = json.loads(evidence_path.read_text())
    observed = path.stat()
    evidence["bytes"] = observed.st_size
    evidence["mtime_ns"] = observed.st_mtime_ns
    group["bytes"] = observed.st_size
    write_json(evidence_path, evidence)
    marker = json.loads(authority["marker"].read_text())
    marker["complete_regular_d03_groups"][0] = group
    write_json(authority["marker"], marker)
    with pytest.raises(gate.GateError, match="FRAME_GRID"):
        build(authority)


def test_qa_range_substitution_fails_even_when_json_rehashed(authority: dict[str, object]) -> None:
    marker = json.loads(authority["marker"].read_text())
    qa_path = Path(marker["complete_regular_d03_groups"][0]["qa_evidence"])
    mutate_json(qa_path, lambda payload: payload["ranges"]["T"].__setitem__(1, 99.0))
    with pytest.raises(gate.GateError, match="EARLY_QA_RANGE_SUBSTITUTION"):
        build(authority)


def test_table_snapshot_exact_bytes_and_post_mutation_gate(authority: dict[str, object]) -> None:
    proof = build(authority)
    work_dir = authority["root"] / "work"
    verified = gate.verify_table_snapshot(
        proof["table_authority"], expected_work_dir=work_dir,
    )
    assert {name: row["sha256"] for name, row in verified.items()} == {
        relative: expected["sha256"] for relative, expected in gate.WRF_DEPENDENCY_MANIFEST.items()
    }
    target = Path(proof["table_authority"]["snapshot_root"]) / "run" / "MPTABLE.TBL"
    target.chmod(0o644)
    target.write_bytes(b"changed")
    with pytest.raises(gate.GateError, match="WRF_DEPENDENCY_MUTATION"):
        gate.verify_table_snapshot(proof["table_authority"], expected_work_dir=work_dir)


def test_wrong_source_table_hash_fails(authority: dict[str, object]) -> None:
    target = authority["tables"] / "run" / "GENPARM.TBL"
    target.write_bytes(b"wrong")
    with pytest.raises(gate.GateError, match="WRF_DEPENDENCY_HASH"):
        build(authority)


def test_real_maxdom3_initialization_dependency_inventory_and_cpu_loader_smoke(tmp_path: Path) -> None:
    work_dir = tmp_path / "private-smoke"
    authority = gate.materialize_table_snapshot(gate.WRF_SOURCE_ROOT, work_dir)
    assert set(authority["inventory"]) == set(REAL_WRF_DEPENDENCY_MANIFEST)
    assert {
        relative: (row["sha256"], row["size"], row["provenance"])
        for relative, row in authority["inventory"].items()
    } == {
        relative: (expected["sha256"], expected["size"], expected["provenance"])
        for relative, expected in REAL_WRF_DEPENDENCY_MANIFEST.items()
    }
    assert authority["loader_source_authority"] == REAL_WRF_LOADER_SOURCE_AUTHORITY
    observed = gate.cpu_initialization_dependency_smoke(
        authority, expected_work_dir=work_dir,
    )
    assert observed["real_jax_backend_imported"] is False
    assert observed["lw_source_sha256"] == "c7a5238612aa8a4213c8d3af6708ec6a5248e6701e19758a80e563905d306de3"
    assert observed["production_loader"] == "gpuwrf.physics.rrtmg_lw._native_lw_tables"
    assert set(authority["inventory"]) == {
        "run/MPTABLE.TBL", "run/SOILPARM.TBL", "run/GENPARM.TBL",
        "phys/module_ra_rrtmg_lw.F",
    }


def test_fully_rehashed_relocated_root_rejected_by_external_workdir(
    authority: dict[str, object],
) -> None:
    proof = build(authority)
    original = proof["table_authority"]
    intended_work_dir = authority["root"] / "work"
    outside_work_dir = authority["root"] / "outside-intended-workdir"
    outside_authority = outside_work_dir / "authority"
    outside_authority.mkdir(parents=True)
    outside_root = outside_authority / "wrf_root"
    shutil.copytree(Path(original["snapshot_root"]), outside_root, copy_function=shutil.copy2)
    relocated = copy.deepcopy(original)
    relocated["snapshot_root"] = str(outside_root)
    relocated["work_dir_authority"] = gate._directory_binding(
        outside_work_dir, "TEST_OUTSIDE_WORK",
    )
    relocated["authority_dir_authority"] = gate._directory_binding(
        outside_authority, "TEST_OUTSIDE_AUTHORITY",
    )
    for relative, detail in relocated["inventory"].items():
        target = outside_root / relative
        info = target.lstat()
        detail.update({
            "snapshot_path": str(target), "snapshot_device": info.st_dev,
            "snapshot_inode": info.st_ino, "snapshot_mtime_ns": info.st_mtime_ns,
            "snapshot_ctime_ns": info.st_ctime_ns,
        })
    unsigned = dict(relocated)
    unsigned.pop("authority_sha256")
    relocated["authority_sha256"] = gate.authority_digest(unsigned)
    with pytest.raises(gate.GateError, match="WRF_DEPENDENCY_AUTHORITY_BINDING"):
        gate.verify_table_snapshot(relocated, expected_work_dir=intended_work_dir)
    with pytest.raises(gate.GateError, match="WRF_DEPENDENCY_AUTHORITY_BINDING"):
        gate.build_exact_command(
            early_proof=proof, work_dir=outside_work_dir,
            candidate_authority={
                "path": str(gate.RUN_REPO), "head": gate.BASE_SHA,
                "clean": True, "detached": True,
            },
            lock_authority={
                "root": str(gate.LOCK_ROOT), "commit": gate.LOCK_COMMIT,
                "wrapper": str(gate.LOCK_WRAPPER),
                "wrapper_sha256": gate.LOCK_WRAPPER_SHA256, "intent": gate.LOCK_INTENT,
            },
        )
    for current, _dirnames, _filenames in os.walk(outside_work_dir, topdown=False):
        Path(current).chmod(0o700)


def test_atomic_publish_collision_never_replaces_racing_target(
    authority: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_dir = authority["root"] / "collision-work"
    real_rename = gate._rename_noreplace

    def collide(dir_fd: int, source_name: str, destination_name: str) -> None:
        destination = work_dir / "authority" / destination_name
        destination.mkdir()
        (destination / "sentinel").write_text("racing owner\n", encoding="utf-8")
        real_rename(dir_fd, source_name, destination_name)

    monkeypatch.setattr(gate, "_rename_noreplace", collide)
    with pytest.raises(gate.GateError, match="WRF_DEPENDENCY_PUBLISH_COLLISION"):
        gate.materialize_table_snapshot(gate.WRF_SOURCE_ROOT, work_dir)
    target = work_dir / "authority/wrf_root"
    assert (target / "sentinel").read_text(encoding="utf-8") == "racing owner\n"
    assert not any((work_dir / "authority").glob(".wrf_root.tmp-*"))


def test_materializer_fsyncs_after_chmod_and_dirs_before_noreplace_publish(
    authority: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int, int]] = []
    real_fsync = gate.os.fsync
    real_fchmod = gate.os.fchmod
    real_rename = gate._rename_noreplace

    def observed_fsync(fd: int) -> None:
        info = os.fstat(fd)
        events.append(("fsync", info.st_ino, stat.S_IMODE(info.st_mode)))
        real_fsync(fd)

    def observed_fchmod(fd: int, mode: int) -> None:
        real_fchmod(fd, mode)
        info = os.fstat(fd)
        events.append(("fchmod", info.st_ino, stat.S_IMODE(info.st_mode)))

    def observed_publish(dir_fd: int, source_name: str, destination_name: str) -> None:
        info = os.fstat(dir_fd)
        events.append(("publish", info.st_ino, stat.S_IMODE(info.st_mode)))
        real_rename(dir_fd, source_name, destination_name)

    monkeypatch.setattr(gate.os, "fsync", observed_fsync)
    monkeypatch.setattr(gate.os, "fchmod", observed_fchmod)
    monkeypatch.setattr(gate, "_rename_noreplace", observed_publish)
    work_dir = authority["root"] / "durability-work"
    snapshot = gate.materialize_table_snapshot(gate.WRF_SOURCE_ROOT, work_dir)
    for detail in snapshot["inventory"].values():
        inode = detail["snapshot_inode"]
        sequence = [(operation, mode) for operation, observed_inode, mode in events if observed_inode == inode]
        assert sequence.index(("fsync", 0o400)) < sequence.index(("fchmod", 0o444))
        assert sequence.index(("fchmod", 0o444)) < sequence.index(("fsync", 0o444))
    publish_index = next(index for index, event in enumerate(events) if event[0] == "publish")
    for directory in [Path(snapshot["snapshot_root"]), Path(snapshot["snapshot_root"]) / "run", Path(snapshot["snapshot_root"]) / "phys"]:
        inode = directory.lstat().st_ino
        assert any(
            index < publish_index and event == ("fsync", inode, 0o555)
            for index, event in enumerate(events)
        )
    authority_inode = (work_dir / "authority").lstat().st_ino
    assert any(
        index > publish_index and event[0] == "fsync" and event[1] == authority_inode
        for index, event in enumerate(events)
    )


def test_archived_pretimestep_failure_evidence_is_exact_and_non_scientific() -> None:
    evidence = gate.validate_pretimestep_failure_evidence()
    assert evidence == {
        "manager_commit": "50fafe7d1f0e3681f95bcf8516a057823809c3f0",
        "json_path": gate.PRETIMESTEP_FAILURE_JSON_RELATIVE,
        "json_sha256": "4a4b4a6ebe672c0b0d79eea1e8e5f052da5516e8ce08af9cc6b2b930dcf10201",
        "log_path": gate.PRETIMESTEP_FAILURE_LOG_RELATIVE,
        "log_sha256": "9e05b9b285a95cf4c906fe390a8f3c9185f1b80ad4405f394143e5905944cda7",
        "classification": "PRE_TIMESTEP_HARNESS_SOURCE_MATERIALIZATION_FAILURE",
        "observed_missing_relative": "phys/module_ra_rrtmg_lw.F",
        "pre_timestep": True,
        "gpu_output_frames": 0,
        "scientific_result": False,
        "prior_workdir_immutable": True,
        "runtime_ref_consumed_no_reuse": True,
    }


def test_observed_lw_missing_and_rehashed_source_path_substitution_fail(
    authority: dict[str, object],
) -> None:
    proof = build(authority)
    snapshot = proof["table_authority"]
    root = Path(snapshot["snapshot_root"])
    root.chmod(0o755)
    (root / "phys").chmod(0o755)
    (root / "phys/module_ra_rrtmg_lw.F").unlink()
    (root / "phys").chmod(0o555)
    root.chmod(0o555)
    with pytest.raises(gate.GateError, match="WRF_DEPENDENCY_EXTRA_OR_MISSING"):
        gate.verify_table_snapshot(snapshot, expected_work_dir=authority["root"] / "work")

    rebuilt = gate.materialize_table_snapshot(
        gate.WRF_SOURCE_ROOT, authority["root"] / "replacement-authority",
    )
    substituted = copy.deepcopy(rebuilt)
    substituted["inventory"]["phys/module_ra_rrtmg_lw.F"]["canonical_source_path"] = (
        "/substituted/phys/module_ra_rrtmg_lw.F"
    )
    unsigned = dict(substituted)
    unsigned.pop("authority_sha256")
    substituted["authority_sha256"] = gate.authority_digest(unsigned)
    with pytest.raises(gate.GateError, match="WRF_DEPENDENCY_MUTATION"):
        gate.verify_table_snapshot(
            substituted, expected_work_dir=authority["root"] / "replacement-authority",
        )


def test_dependency_snapshot_rejects_symlink_hardlink_extra_and_schema_contradiction(
    authority: dict[str, object], request: pytest.FixtureRequest,
) -> None:
    cleanup_roots: list[Path] = []

    def cleanup() -> None:
        for tree in cleanup_roots:
            if not tree.exists() and not tree.is_symlink():
                continue
            for current, dirnames, filenames in os.walk(tree, topdown=False, followlinks=False):
                current_path = Path(current)
                current_path.chmod(0o700)
                for filename in filenames:
                    path = current_path / filename
                    if path.is_symlink():
                        path.unlink(missing_ok=True)
                    else:
                        path.chmod(0o600)
                for dirname in dirnames:
                    path = current_path / dirname
                    if path.is_symlink():
                        path.unlink(missing_ok=True)
                    else:
                        path.chmod(0o700)

    request.addfinalizer(cleanup)
    proof = build(authority)
    original = proof["table_authority"]

    extra_root = Path(original["snapshot_root"])
    cleanup_roots.append(extra_root)
    extra_root.chmod(0o755)
    (extra_root / "unexpected").write_bytes(b"ambient fallback")
    extra_root.chmod(0o555)
    with pytest.raises(gate.GateError, match="WRF_DEPENDENCY_EXTRA_OR_MISSING"):
        gate.verify_table_snapshot(original, expected_work_dir=authority["root"] / "work")

    symlinked = gate.materialize_table_snapshot(
        gate.WRF_SOURCE_ROOT, authority["root"] / "symlink-authority",
    )
    symlink_root = Path(symlinked["snapshot_root"])
    cleanup_roots.append(symlink_root)
    symlink_root.chmod(0o755)
    (symlink_root / "phys").chmod(0o755)
    target = symlink_root / "phys/module_ra_rrtmg_lw.F"
    external = authority["root"] / "external-lw"
    external.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(external)
    (symlink_root / "phys").chmod(0o555)
    symlink_root.chmod(0o555)
    with pytest.raises(gate.GateError, match="WRF_DEPENDENCY_TARGET_SYMLINK"):
        gate.verify_table_snapshot(
            symlinked, expected_work_dir=authority["root"] / "symlink-authority",
        )

    hardlinked = gate.materialize_table_snapshot(
        gate.WRF_SOURCE_ROOT, authority["root"] / "hardlink-authority",
    )
    hardlink_root = Path(hardlinked["snapshot_root"])
    cleanup_roots.append(hardlink_root)
    hardlink_root.chmod(0o755)
    (hardlink_root / "phys").chmod(0o755)
    hardlink_target = hardlink_root / "phys/module_ra_rrtmg_lw.F"
    external_hardlink = authority["root"] / "external-hardlink-lw"
    external_hardlink.write_bytes(hardlink_target.read_bytes())
    hardlink_target.unlink()
    os.link(external_hardlink, hardlink_target)
    hardlink_target.chmod(0o444)
    (hardlink_root / "phys").chmod(0o555)
    hardlink_root.chmod(0o555)
    with pytest.raises(gate.GateError, match="WRF_DEPENDENCY_MUTATION"):
        gate.verify_table_snapshot(
            hardlinked, expected_work_dir=authority["root"] / "hardlink-authority",
        )

    semantic_substitution = copy.deepcopy(original)
    semantic_substitution["same_uid_chmod_residual"] = "owning UID cannot mutate files"
    unsigned = dict(semantic_substitution)
    unsigned.pop("authority_sha256")
    semantic_substitution["authority_sha256"] = gate.authority_digest(unsigned)
    with pytest.raises(gate.GateError, match="WRF_DEPENDENCY_AUTHORITY_BINDING"):
        gate.verify_table_snapshot(
            semantic_substitution, expected_work_dir=authority["root"] / "work",
        )

    contradictory = copy.deepcopy(hardlinked)
    contradictory["unexpected"] = True
    unsigned = dict(contradictory)
    unsigned.pop("authority_sha256")
    contradictory["authority_sha256"] = gate.authority_digest(unsigned)
    with pytest.raises(gate.GateError, match="WRF_DEPENDENCY_AUTHORITY_SCHEMA"):
        gate.verify_table_snapshot(
            contradictory, expected_work_dir=authority["root"] / "hardlink-authority",
        )


def test_dead_or_reused_cpu_pid_fails_live_identity_gate(authority: dict[str, object]) -> None:
    cpu = json.loads(authority["marker"].read_text())["cpu_oracle"]
    cpu["pid"] = 999_999_999
    with pytest.raises(gate.GateError, match="CPU_IDENTITY"):
        gate.verify_live_cpu_identity(cpu)


def test_each_rank_start_tick_and_ancestry_are_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu = cpu_identity()
    monkeypatch.setattr(gate, "validate_canonical_authority_v2", lambda: {
        "cpu_runtime": {"wrf_pid": cpu["pid"], "wrf_start_ticks": cpu["start_ticks"]},
    })
    starts = {cpu["pid"]: cpu["start_ticks"]}
    starts.update({row["pid"]: 9000 + index for index, row in enumerate(cpu["descendants"])})

    def live(pid: int) -> dict[str, object]:
        root = pid == cpu["pid"]
        return {
            "start_ticks": starts[pid], "parent_pid": 1 if root else cpu["pid"],
            "status": {"Cpus_allowed_list": "0-11"},
            "command": " ".join(gate.PRTERUN_ARGV) if root else "./wrf.exe",
        }

    monkeypatch.setattr(gate, "_live_cpu_proc", live)
    monkeypatch.setattr(gate, "_is_descendant_of", lambda pid, root: (True, [pid]))
    monkeypatch.setattr(gate, "validate_live_prterun_executable", lambda _pid: gate.PRTE_RESOLVED)
    bound = gate.verify_live_cpu_identity(cpu)
    assert len(bound["ranks"]) == 12
    starts[cpu["descendants"][0]["pid"]] += 1
    with pytest.raises(gate.GateError, match="CPU_LIVE_IDENTITY_CHANGED"):
        gate.verify_live_cpu_identity(cpu, bound)
    monkeypatch.setattr(gate, "_is_descendant_of", lambda pid, root: (False, [pid]))
    with pytest.raises(gate.GateError, match="CPU_RANK_IDENTITY_CHANGED"):
        gate.verify_live_cpu_identity(cpu)


def test_current_preempt_recheck_is_mandatory(authority: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked() -> dict[str, object]:
        raise gate.GateError("CURRENT_PREEMPT_STOP_AND_RELEASE", "Nightly active")

    monkeypatch.setattr(gate, "require_current_no_preemption", blocked)
    with pytest.raises(gate.GateError, match="CURRENT_PREEMPT_STOP_AND_RELEASE"):
        build(authority)


def test_marker_resource_qa_and_derived_freshness_fail_closed(authority: dict[str, object]) -> None:
    mutate_json(authority["marker"], lambda payload: payload.__setitem__("emitted_utc", "2026-07-11T23:00:00Z"))
    with pytest.raises(gate.GateError, match="STALE_AUTHORITY"):
        build(authority)


def test_resource_preemption_and_qa_time_order_fail_closed(authority: dict[str, object]) -> None:
    resource = json.loads(authority["resource"].read_text())
    resource["checked_utc"] = "2026-07-12T00:02:30Z"
    resource["preemption"]["checked_utc"] = "2026-07-12T00:02:15Z"
    write_json(authority["resource"], resource)
    marker = json.loads(authority["marker"].read_text())
    marker["resource_proof"] = resource
    write_json(authority["marker"], marker)
    with pytest.raises(gate.GateError, match="AUTHORITY_TIME_ORDER"):
        build(authority)


@pytest.mark.parametrize("target", ["resource", "preemption", "qa"])
def test_each_nested_authority_timestamp_has_maximum_age(authority: dict[str, object], target: str) -> None:
    if target in {"resource", "preemption"}:
        resource = json.loads(authority["resource"].read_text())
        if target == "resource":
            resource["checked_utc"] = "2026-07-11T23:59:00Z"
        else:
            resource["preemption"]["checked_utc"] = "2026-07-11T23:59:00Z"
        write_json(authority["resource"], resource)
        marker = json.loads(authority["marker"].read_text())
        marker["resource_proof"] = resource
        write_json(authority["marker"], marker)
    else:
        marker = json.loads(authority["marker"].read_text())
        qa_path = Path(marker["complete_regular_d03_groups"][0]["qa_evidence"])
        mutate_json(qa_path, lambda payload: payload.__setitem__("checked_utc", "2026-07-11T23:59:00Z"))
    with pytest.raises(gate.GateError, match="STALE_AUTHORITY"):
        build(authority)


def test_consumed_proof_is_reconstructed_not_self_authenticated(authority: dict[str, object]) -> None:
    proof = build(authority)
    proof_path = authority["root"] / "early-proof.json"
    proof["canonical_authority_sha256"] = "0" * 64
    unsigned = dict(proof)
    unsigned.pop("authority_sha256")
    proof["authority_sha256"] = gate.authority_digest(unsigned)
    write_json(proof_path, proof)
    with pytest.raises(gate.GateError, match="EARLY_PROOF_NOT_CANONICAL"):
        gate.verify_early_ready_proof(
            proof_path,
            expected_work_dir=authority["root"] / "work",
            now=datetime(2026, 7, 12, 0, 3, 40, tzinfo=timezone.utc),
            sleep_fn=lambda _: None,
            live_identity_checker=lambda cpu: {
                "pid": cpu["pid"], "start_ticks": cpu["start_ticks"],
                "wrf_rank_pids": [row["pid"] for row in cpu["descendants"] if "wrf.exe" in row["command"]],
                "cpuset": "0-11",
            },
            gpu_baseline_checker=lambda _resource: [],
        )


def test_derived_proof_staleness_fails_before_canonical_consume(authority: dict[str, object]) -> None:
    proof = build(authority)
    proof["created_utc"] = "2026-07-12T00:00:00+00:00"
    unsigned = dict(proof)
    unsigned.pop("authority_sha256")
    proof["authority_sha256"] = gate.authority_digest(unsigned)
    proof_path = authority["root"] / "stale-proof.json"
    write_json(proof_path, proof)
    with pytest.raises(gate.GateError, match="derived_proof"):
        gate.verify_early_ready_proof(
            proof_path, expected_work_dir=authority["root"] / "work",
            now=datetime(2026, 7, 12, 0, 3, 40, tzinfo=timezone.utc),
            sleep_fn=lambda _: None,
            live_identity_checker=lambda _cpu: {}, gpu_baseline_checker=lambda _resource: [],
        )


def test_immediate_pre_spawn_runs_every_live_gate(authority: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    proof = build(authority)
    calls: list[str] = []
    monkeypatch.setattr(gate, "verify_early_ready_proof", lambda *_args, **_kwargs: proof)
    monkeypatch.setattr(gate, "verify_live_cpu_identity", lambda *_args: calls.append("cpu") or proof["cpu_live_identity"])
    monkeypatch.setattr(gate, "validate_live_cpu_lane", lambda: calls.append("lane") or {"free": True})
    monkeypatch.setattr(gate, "validate_live_gpu_device", lambda *_args: calls.append("gpu") or {"free": True})
    monkeypatch.setattr(gate, "validate_live_lock_available", lambda: calls.append("lock") or {"free": True, "device": 42, "inode": 84})
    monkeypatch.setattr(gate, "require_current_no_preemption", lambda: calls.append("preempt") or {"nightly_active": False})
    rebuilt, result = gate.immediate_pre_spawn_checks(
        authority["root"] / "proof.json",
        expected_work_dir=authority["root"] / "work",
        now=datetime(2026, 7, 12, 0, 3, 40, tzinfo=timezone.utc), interval_seconds=0,
    )
    assert rebuilt is proof
    assert calls == ["cpu", "lock", "lane", "gpu", "preempt"]
    assert result["pre_spawn_sha256"] == gate.authority_digest({key: value for key, value in result.items() if key != "pre_spawn_sha256"})


def test_terminal_pre_spawn_preserves_live_resource_gates_without_cpu_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = {
        "input_authority": {"input": {"sha256": "a" * 64}},
        "table_authority": {"authority_sha256": "b" * 64},
        "canonical_authority_sha256": "c" * 64,
    }
    calls: list[str] = []
    monkeypatch.setattr(gate, "verify_terminal_cpu_proof", lambda *_args, **_kwargs: proof)
    monkeypatch.setattr(gate, "validate_live_lock_available", lambda: calls.append("lock") or {
        "free": True, "path": "/tmp/wrf_gpu2_gpu.lock", "device": 42, "inode": 84,
    })
    monkeypatch.setattr(gate, "validate_live_cpu_lane", lambda: calls.append("lane") or {"free": True})
    monkeypatch.setattr(gate, "validate_live_gpu_device", lambda *_args: calls.append("gpu") or {"free": True})
    monkeypatch.setattr(gate, "require_current_no_preemption", lambda: calls.append("preempt") or {"nightly_active": False})
    gpu_runtime = {
        "gpu": {"index": 0, "name": "NVIDIA GeForce RTX 5090", "memory_total_mib": 32607},
        "gpu_lock": {"path": "/tmp/wrf_gpu2_gpu.lock", "device": 42, "inode": 84},
        "baseline_compute_processes": [],
    }
    rebuilt, result = gate.immediate_terminal_pre_spawn_checks(
        Path("/terminal-proof"), gpu_runtime,
        expected_work_dir=Path("/terminal-work"),
        now=datetime(2026, 7, 12, 7, 0, tzinfo=timezone.utc), interval_seconds=0.0,
    )
    assert rebuilt is proof
    assert calls == ["lock", "lane", "gpu", "preempt"]
    assert result["cpu_authority_mode"] == "terminal_no_live_pid"
    assert "cpu_identity" not in result


def test_monitor_keeps_cpu_live_until_strict_terminal_and_always_checks_qa(
    authority: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = build(authority)
    calls: list[str] = []
    monkeypatch.setattr(gate, "verify_live_cpu_identity", lambda *_args: calls.append("live") or proof["cpu_live_identity"])
    work_dir = authority["root"] / "work"
    assert gate.monitor_once(
        early_proof=proof, expected_work_dir=work_dir,
        cpu_manifest_path=gate.CPU_MANIFEST,
    )["status"] == "CONTINUE_CPU_LIVE"
    assert calls == ["live"]
    write_json(gate.CPU_MANIFEST, {"terminal": True})
    monkeypatch.setattr(gate, "validate_cpu_terminal_manifest", lambda *_args, **_kwargs: {"authenticated": True})
    calls.clear()
    assert gate.monitor_once(
        early_proof=proof, expected_work_dir=work_dir,
        cpu_manifest_path=gate.CPU_MANIFEST,
    )["status"] == "CONTINUE_CPU_TERMINAL_AUTHENTICATED"
    assert calls == []
    qa_path = Path(proof["early_frame_qa"][0]["artifact"]["path"])
    qa_path.write_text("{}\n")
    with pytest.raises(gate.GateError, match="EARLY_QA_MUTATED"):
        gate.monitor_once(
            early_proof=proof, expected_work_dir=work_dir,
            cpu_manifest_path=gate.CPU_MANIFEST,
        )


def test_chrome_nvidia_name_is_bound_separately_from_proc_argv0(monkeypatch: pytest.MonkeyPatch) -> None:
    nvidia_name = "/opt/google/chrome/chrome --type=gpu-process --use-gl=angle"
    process_result = subprocess.CompletedProcess([], 0, f"4242, {nvidia_name}, 321\n", "")
    device_result = subprocess.CompletedProcess([], 0, "0, NVIDIA RTX, 32607, 2600, 30000, 2\n", "")
    results = iter((process_result, device_result))
    monkeypatch.setattr(gate.subprocess, "run", lambda *_args, **_kwargs: next(results))
    bound_process = {
        "start_ticks": 777, "argv0": "/opt/google/chrome/chrome",
        "exe_path": "/opt/google/chrome/chrome", "exe_device": 8, "exe_inode": 9,
    }
    monkeypatch.setattr(gate, "live_process_binding", lambda pid: dict(bound_process))
    expected = {
        "baseline_compute_processes": [{
            "pid": 4242, "process_name": nvidia_name, "used_memory_mib": 300,
            "baseline_allowed": True,
        }],
        "unexpected_compute_processes": [],
        "gpu": {
            "index": 0, "name": "NVIDIA RTX", "memory_total_mib": 32607,
            "memory_used_mib": 2500, "memory_free_mib": 30107, "utilization_percent": 0,
        },
    }
    result = gate.validate_live_gpu_device(
        expected, [{"pid": 4242, "process_name": nvidia_name, **bound_process}],
    )
    assert result["free"] is True
    assert result["baseline_compute_processes"][0]["process_name"] == nvidia_name


def test_gpu_baseline_addition_and_pid_reuse_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "/usr/bin/plasmashell"
    process_result = subprocess.CompletedProcess([], 0, f"42, {name}, 100\n43, /tmp/model, 1000\n", "")
    monkeypatch.setattr(gate.subprocess, "run", lambda *_args, **_kwargs: process_result)
    expected = {
        "baseline_compute_processes": [{"pid": 42, "process_name": name, "baseline_allowed": True}],
        "unexpected_compute_processes": [],
        "gpu": {
            "index": 0, "name": "NVIDIA RTX", "memory_total_mib": 32607,
            "memory_used_mib": 2500, "memory_free_mib": 30107, "utilization_percent": 0,
        },
    }
    bound = {"pid": 42, "process_name": name, "start_ticks": 1, "argv0": name, "exe_path": name, "exe_device": 1, "exe_inode": 2}
    with pytest.raises(gate.GateError, match="GPU_BASELINE_IDENTITY_CHANGED"):
        gate.validate_live_gpu_device(expected, [bound])
    process_only = subprocess.CompletedProcess([], 0, f"42, {name}, 100\n", "")
    device = subprocess.CompletedProcess([], 0, "0, NVIDIA RTX, 32607, 2600, 30000, 0\n", "")
    results = iter((process_only, device))
    monkeypatch.setattr(gate.subprocess, "run", lambda *_args, **_kwargs: next(results))
    monkeypatch.setattr(
        gate, "live_process_binding",
        lambda pid: {"start_ticks": 2, "argv0": name, "exe_path": name, "exe_device": 1, "exe_inode": 2},
    )
    with pytest.raises(gate.GateError, match="GPU_BASELINE_PID_REUSED"):
        gate.validate_live_gpu_device(expected, [bound])


def test_live_gpu_index_name_and_total_memory_must_match_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    process = subprocess.CompletedProcess([], 0, "", "")
    changed = subprocess.CompletedProcess([], 0, "1, Substitute GPU, 24576, 10, 24566, 0\n", "")
    results = iter((process, changed))
    monkeypatch.setattr(gate.subprocess, "run", lambda *_args, **_kwargs: next(results))
    expected = {
        "baseline_compute_processes": [], "unexpected_compute_processes": [],
        "gpu": {
            "index": 0, "name": "NVIDIA RTX", "memory_total_mib": 32607,
            "memory_used_mib": 1, "memory_free_mib": 32606, "utilization_percent": 0,
        },
    }
    with pytest.raises(gate.GateError, match="GPU_DEVICE_IDENTITY_CHANGED"):
        gate.validate_live_gpu_device(expected, [])


def test_resource_gpu_query_and_object_are_exact_schema(authority: dict[str, object]) -> None:
    resource = json.loads(authority["resource"].read_text())
    resource["gpu_device"]["gpu"]["alias"] = "forbidden"
    write_json(authority["resource"], resource)
    marker = json.loads(authority["marker"].read_text())
    marker["resource_proof"] = resource
    write_json(authority["marker"], marker)
    with pytest.raises(gate.GateError, match="GPU_OBJECT_SCHEMA"):
        build(authority)


def test_all_four_static_fields_are_mandatory_per_pair(tmp_path: Path) -> None:
    cpu = tmp_path / "wrfout_d03_2025-03-01_00:00:00"
    gpu = tmp_path / "gpu" / cpu.name
    gpu.parent.mkdir()
    write_frame(cpu, "2025-03-01_00:00:00")
    write_frame(gpu, "2025-03-01_00:00:00", omit={"HGT"})
    with pytest.raises(gate.GateError, match="FROZEN_GEOMETRY_FIELD"):
        gate.compare_pair(cpu, gpu, gate.EXPECTED_TIMES[0])


def test_frozen_geo_bytes_are_exactly_authenticated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    geo = tmp_path / "geo_em.d03.nc"
    with Dataset(geo, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("south_north", 1)
        dataset.createDimension("west_east", 1)
        for field, value in {"XLAT_M": 28.2, "XLONG_M": -16.5, "HGT_M": 500.0, "LANDMASK": 1.0}.items():
            variable = dataset.createVariable(field, "f8", ("Time", "south_north", "west_east"))
            variable[:] = value
    monkeypatch.setattr(gate, "FROZEN_GEO_PATH", geo)
    monkeypatch.setattr(gate, "GRID_SHA256", gate.sha256_file(geo))
    arrays, authority = REAL_LOAD_FROZEN_GEOMETRY()
    assert authority["sha256"] == gate.GRID_SHA256
    assert set(arrays) == set(gate.STATIC_GATE_FIELDS)
    with geo.open("ab") as handle:
        handle.write(b"one-bit-substitution")
    with pytest.raises(gate.GateError, match="FROZEN_GEO_HASH"):
        REAL_LOAD_FROZEN_GEOMETRY()


def test_pair_matching_wrong_geometry_fails_even_when_both_sides_match(tmp_path: Path) -> None:
    cpu = tmp_path / "wrfout_d03_2025-03-01_00:00:00"
    gpu = tmp_path / "gpu" / cpu.name
    gpu.parent.mkdir()
    write_frame(cpu, "2025-03-01_00:00:00")
    write_frame(gpu, "2025-03-01_00:00:00")
    for path in (cpu, gpu):
        with Dataset(path, "r+") as dataset:
            dataset.variables["LANDMASK"][0, 0, 0] = 0.0
    with pytest.raises(gate.GateError, match="FROZEN_GEOMETRY_EXACT"):
        gate.compare_pair(cpu, gpu, gate.EXPECTED_TIMES[0])


def test_authenticated_terminal_cpu_frame_pairs_with_boundary_hgt_and_static_stays_bit_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "load_frozen_geometry", REAL_LOAD_FROZEN_GEOMETRY)
    manifest = json.loads(gate.CPU_MANIFEST.read_text(encoding="utf-8"))
    declared = manifest["raw_artifacts"]["d03"][0]
    cpu = Path(declared["path"])
    assert cpu == gate.CPU_INPUT_DIR / "wrfout_d03_2025-03-01_00:00:00"
    assert gate.sha256_file(cpu) == declared["sha256"]
    gpu = tmp_path / "gpu" / cpu.name
    gpu.parent.mkdir()
    shutil.copy2(cpu, gpu)
    row = gate.compare_pair(
        cpu, gpu, gate.EXPECTED_TIMES[0],
        frozen_geometry_policy="terminal_nested_boundary_v1",
    )
    assert row["frozen_geometry_policy"] == "terminal_nested_boundary_v1"
    assert row["per_frame_static_pass"] is True
    assert all(row["metrics"][field]["max_abs"] == 0.0 for field in gate.STATIC_GATE_FIELDS)
    with Dataset(gpu, "r+") as dataset:
        dataset.variables["HGT"][0, 0, 0] += 1.0
    mismatch = gate.compare_pair(
        cpu, gpu, gate.EXPECTED_TIMES[0],
        frozen_geometry_policy="terminal_nested_boundary_v1",
    )
    assert mismatch["metrics"]["HGT"]["max_abs"] == 1.0
    assert mismatch["metrics"]["HGT"]["pass"] is False
    assert mismatch["per_frame_static_pass"] is False
    with Dataset(gpu, "r+") as dataset:
        dataset.variables["HGT"][0, 20, 20] += 1.0
    with pytest.raises(gate.GateError, match="TERMINAL_CPU_GEOMETRY.*HGT_INTERIOR"):
        gate.compare_pair(
            cpu, gpu, gate.EXPECTED_TIMES[0],
            frozen_geometry_policy="terminal_nested_boundary_v1",
        )


def test_missing_owner_report_field_is_explicit_not_silently_green(tmp_path: Path) -> None:
    cpu = tmp_path / "wrfout_d03_2025-03-01_00:00:00"
    gpu = tmp_path / "gpu" / cpu.name
    gpu.parent.mkdir()
    write_frame(cpu, "2025-03-01_00:00:00")
    write_frame(gpu, "2025-03-01_00:00:00", omit={"RAINC"})
    row = gate.compare_pair(cpu, gpu, gate.EXPECTED_TIMES[0])
    assert "RAINC" in row["inventory"]["cpu_only_fields"]
    pooled, inventory = gate.pooled_metrics([row])
    assert "RAINC" not in pooled
    assert inventory["fields"]["RAINC"]["policy"] == "OWNER_DIRECTIVE_REPORT_ONLY"
    assert inventory["fields"]["RAINC"]["missing_gpu_valid_times"] == [gate.EXPECTED_TIMES[0].isoformat()]


def test_exact_command_uses_lock_v2_cuda_and_no_shell(authority: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    proof = build(authority, "future-run")
    run_repo = authority["root"] / "candidate"
    monkeypatch.setattr(gate, "RUN_REPO", run_repo)
    candidate = {"path": str(run_repo), "head": gate.BASE_SHA, "clean": True, "detached": True}
    lock = {
        "root": str(gate.LOCK_ROOT), "commit": gate.LOCK_COMMIT,
        "wrapper": str(gate.LOCK_WRAPPER), "wrapper_sha256": gate.LOCK_WRAPPER_SHA256,
        "intent": gate.LOCK_INTENT,
    }
    command = gate.build_exact_command(
        early_proof=proof, work_dir=authority["root"] / "future-run",
        candidate_authority=candidate, lock_authority=lock,
    )
    assert command["shell"] is False
    assert command["argv"][:2] == ["/usr/bin/env", "-i"]
    lock_index = command["argv"].index(str(gate.LOCK_WRAPPER))
    separator = command["argv"].index("--")
    assert gate.SYSTEMD_USER_ENVIRONMENT == (
        ("XDG_RUNTIME_DIR", "/run/user/1000"),
        ("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus"),
    )
    assert set(gate.COMMAND_ENV_KEYS) - set(gate.PRE_SYSTEMD_COMMAND_ENV_KEYS) == {
        "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS",
    }
    assert gate.COMMAND_ENV_KEYS[2:4] == ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
    assert tuple(command["environment"]) == gate.COMMAND_ENV_KEYS
    assert command["argv"][2:lock_index] == [
        f"{key}={command['environment'][key]}" for key in gate.COMMAND_ENV_KEYS
    ]
    assert command["argv"][4:6] == [
        "XDG_RUNTIME_DIR=/run/user/1000",
        "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus",
    ]
    assert command["argv"][separator + 1:separator + 6] == [
        "/usr/bin/taskset", "-c", "12-15", str(gate.PYTHON_BIN), "-m",
    ]
    assert command["argv"][separator + 6:separator + 8] == ["gpuwrf", "run"]
    assert "/usr/bin/env" not in command["argv"][separator + 1:]
    assert "-i" not in command["argv"][separator + 1:]
    assert command["argv"][command["argv"].index("--intent") + 1] == "production-preemptible"
    assert command["argv"][command["argv"].index("-c") + 1] == "12-15"
    assert command["environment"]["JAX_PLATFORMS"] == "cuda"
    assert command["environment"]["GPUWRF_WRF_ROOT"] == proof["table_authority"]["snapshot_root"]
    assert "gpu" not in {command["environment"]["JAX_PLATFORMS"]}
    assert str(ROOT / "scripts/with_gpu_lock.sh") not in command["argv"]
    assert command["argv"][command["argv"].index("--timeout") + 1] == "0"
    assert command["identity_policy_sha256"] == proof["identity_policy"]["authority_sha256"]
    assert gate.validate_exact_command_argv(command) == command["argv_authority"]


def test_sanitized_private_lock_wrapper_injects_attested_environment_for_payload(tmp_path: Path) -> None:
    lock_path = tmp_path / "private.lock"
    holder_path = tmp_path / "private.lock.holder"
    wrapper = tmp_path / "private-lock-v2"
    wrapper.write_text(
        "#!/usr/bin/python3\n"
        "import fcntl, json, os, sys\n"
        "args = sys.argv[1:]\n"
        "separator = args.index('--')\n"
        "label = args[args.index('--label') + 1]\n"
        "required = {'XDG_RUNTIME_DIR': '/run/user/1000', "
        "'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/1000/bus'}\n"
        "if any(os.environ.get(key) != value for key, value in required.items()):\n"
        "    raise SystemExit(71)\n"
        f"lock_path = {str(lock_path)!r}\n"
        f"holder_path = {str(holder_path)!r}\n"
        "fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)\n"
        "fcntl.flock(fd, fcntl.LOCK_EX)\n"
        "os.set_inheritable(fd, True)\n"
        "token = 'private-attested-token'\n"
        "open(holder_path, 'w').write(json.dumps({'lease_id': token, 'pid': os.getpid()}))\n"
        "env = os.environ.copy()\n"
        "env.update({'GPUWRF_GPU_LOCK_HELD': '1', 'GPUWRF_GPU_LOCK_FD': str(fd), "
        "'GPUWRF_GPU_LOCK_FILE': lock_path, 'GPUWRF_GPU_LOCK_HOLDER_FILE': holder_path, "
        "'GPUWRF_GPU_LOCK_LABEL': label, 'GPUWRF_GPU_LOCK_TOKEN': token})\n"
        "payload = args[separator + 1:]\n"
        "os.execvpe(payload[0], payload, env)\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    checker = tmp_path / "check-lock-env.py"
    checker.write_text(
        "import json, os\n"
        "keys = sorted(k for k in os.environ if k.startswith('GPUWRF_GPU_LOCK_'))\n"
        "fd = int(os.environ['GPUWRF_GPU_LOCK_FD'])\n"
        "os.fstat(fd)\n"
        "print(json.dumps({'lock': {k: os.environ[k] for k in keys}, "
        "'systemd': {k: os.environ[k] for k in "
        "('XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS')}, "
        "'unauthorized_present': 'UNAUTHORIZED_AMBIENT' in os.environ}, sort_keys=True))\n",
        encoding="utf-8",
    )
    environment = gate._command_environment(tmp_path / "work", tmp_path / "tables")
    payload = ["/usr/bin/taskset", "-c", "12-15", sys.executable, str(checker)]
    argv = gate._lock_wrapped_argv(environment, payload, lock_wrapper=wrapper)
    inherited = os.environ.copy()
    inherited["GPUWRF_GPU_LOCK_HELD"] = "malicious-preexisting-value"
    inherited["XDG_RUNTIME_DIR"] = "/malicious/runtime"
    inherited["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=/malicious/bus"
    inherited["UNAUTHORIZED_AMBIENT"] = "must-not-pass-env-i"
    result = subprocess.run(
        argv, env=inherited, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["lock"] == {
        "GPUWRF_GPU_LOCK_FD": observed["lock"]["GPUWRF_GPU_LOCK_FD"],
        "GPUWRF_GPU_LOCK_FILE": str(lock_path),
        "GPUWRF_GPU_LOCK_HELD": "1",
        "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder_path),
        "GPUWRF_GPU_LOCK_LABEL": "v0234-corrected-fullbuffer-20250228",
        "GPUWRF_GPU_LOCK_TOKEN": "private-attested-token",
    }
    assert observed["lock"]["GPUWRF_GPU_LOCK_FD"].isdigit()
    assert observed["systemd"] == dict(gate.SYSTEMD_USER_ENVIRONMENT)
    assert observed["unauthorized_present"] is False


def test_exact_command_rejects_prelock_or_postlock_reordering(
    authority: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = build(authority, "future-run")
    run_repo = authority["root"] / "candidate"
    monkeypatch.setattr(gate, "RUN_REPO", run_repo)
    command = gate.build_exact_command(
        early_proof=proof, work_dir=authority["root"] / "future-run",
        candidate_authority={"path": str(run_repo), "head": gate.BASE_SHA, "clean": True, "detached": True},
        lock_authority={
            "root": str(gate.LOCK_ROOT), "commit": gate.LOCK_COMMIT,
            "wrapper": str(gate.LOCK_WRAPPER), "wrapper_sha256": gate.LOCK_WRAPPER_SHA256,
            "intent": gate.LOCK_INTENT,
        },
    )
    mutations: list[tuple[dict[str, object], str]] = []
    swapped_assignments = copy.deepcopy(command)
    swapped_assignments["argv"][2], swapped_assignments["argv"][3] = (
        swapped_assignments["argv"][3], swapped_assignments["argv"][2],
    )
    mutations.append((swapped_assignments, "COMMAND_ARGV_ORDER"))
    lock_before_sanitizer = copy.deepcopy(command)
    lock_index = lock_before_sanitizer["argv"].index(str(gate.LOCK_WRAPPER))
    lock_before_sanitizer["argv"].insert(0, lock_before_sanitizer["argv"].pop(lock_index))
    mutations.append((lock_before_sanitizer, "COMMAND_ARGV_ORDER"))
    post_lock_clear = copy.deepcopy(command)
    separator = post_lock_clear["argv"].index("--")
    post_lock_clear["argv"][separator + 1:separator + 1] = ["/usr/bin/env", "-i"]
    mutations.append((post_lock_clear, "COMMAND_ARGV_ORDER"))
    missing_xdg = copy.deepcopy(command)
    missing_xdg["argv"].remove("XDG_RUNTIME_DIR=/run/user/1000")
    mutations.append((missing_xdg, "COMMAND_ARGV_ORDER"))
    substituted_bus = copy.deepcopy(command)
    bus_index = substituted_bus["argv"].index(
        "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus",
    )
    substituted_bus["argv"][bus_index] = "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/wrong"
    mutations.append((substituted_bus, "COMMAND_ARGV_ORDER"))
    reordered_systemd = copy.deepcopy(command)
    xdg_index = reordered_systemd["argv"].index("XDG_RUNTIME_DIR=/run/user/1000")
    bus_index = reordered_systemd["argv"].index(
        "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus",
    )
    reordered_systemd["argv"][xdg_index], reordered_systemd["argv"][bus_index] = (
        reordered_systemd["argv"][bus_index], reordered_systemd["argv"][xdg_index],
    )
    mutations.append((reordered_systemd, "COMMAND_ARGV_ORDER"))
    substituted_environment = copy.deepcopy(command)
    substituted_environment["environment"]["XDG_RUNTIME_DIR"] = "/run/user/999"
    mutations.append((substituted_environment, "COMMAND_ENV_ALLOWLIST"))
    for malformed, error in mutations:
        with pytest.raises(gate.GateError, match=error):
            gate.validate_exact_command_argv(malformed)
    popen_called = False

    def forbidden_popen(*_args: object, **_kwargs: object) -> object:
        nonlocal popen_called
        popen_called = True
        raise AssertionError("malformed command reached Popen")

    for malformed, error in mutations:
        with pytest.raises(gate.GateError, match=error):
            gate.run_reviewed_command(malformed, popen_factory=forbidden_popen)
    assert popen_called is False


def test_code_review_acceptance_requires_fixed_independent_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "review"
    init_git_repo(repo)
    (repo / "base.txt").write_text("tooling\n")
    tooling = git_commit_all(repo, "tooling")
    acceptance_path = repo / gate.CODE_REVIEW_ACCEPTANCE_RELATIVE
    acceptance_path.parent.mkdir(parents=True)
    write_json(acceptance_path, {
        "schema": gate.CODE_REVIEW_SCHEMA, "verdict": "ACCEPT",
        "reviewer_role": "independent_gpt56_sol_xhigh_critic",
        "tooling_commit": tooling, "manager_policy_sha256": gate.MANAGER_POLICY_SHA256,
        "release_policy_sha256": gate.RELEASE_POLICY_SHA256,
        "runtime_marker_status": "MAY_BE_PENDING_CODE_REVIEW_ONLY",
        "reviewed_utc": "2026-07-12T00:00:00Z",
    })
    monkeypatch.setattr(gate, "REVIEW_REPO", repo)
    review_commit = git_commit_all(repo, "independent accept")
    with pytest.raises(gate.GateError, match="INDEPENDENT_AUTHORITY_REF"):
        gate.validate_code_review_acceptance()
    subprocess.run(
        ["git", "-C", str(repo), "update-ref", gate.CODE_REVIEW_ACCEPTANCE_REF, review_commit], check=True,
    )
    accepted = gate.validate_code_review_acceptance()
    assert accepted["commit"] == review_commit
    acceptance_path.write_text('{"schema":"self-minted-working-tree"}\n')
    assert gate.validate_code_review_acceptance()["payload"]["verdict"] == "ACCEPT"


def test_executing_tooling_is_file_rooted_clean_detached_and_blob_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "tooling"
    init_git_repo(repo)
    script = repo / "scripts/v0234_corrected_fullbuffer_gate.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(SCRIPT.read_bytes())
    commit = git_commit_all(repo, "frozen tooling")
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "--detach", commit], check=True)
    monkeypatch.setattr(gate, "__file__", str(script))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert gate.validate_executing_tooling(commit)["commit"] == commit
    script.write_text("# substituted\n", encoding="utf-8")
    with pytest.raises(gate.GateError, match="EXECUTING_TOOLING_STATE"):
        gate.validate_executing_tooling(commit)


def test_runtime_nonce_is_atomically_one_use_outside_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority_root = tmp_path / "outside-run"
    authority_root.mkdir(mode=0o700)
    monkeypatch.setattr(gate, "RUNTIME_AUTHORITY_ROOT", authority_root)
    nonce = "a" * 64
    acceptance = {
        "commit": "manager-commit", "blob_sha256": "b" * 64,
        "payload": {
            "nonce_authority": gate.runtime_nonce_boundary_authority(), "nonce": nonce,
            "command_sha256": "c" * 64, "canonical_authority_sha256": "d" * 64,
        },
    }
    pre_spawn = {"pre_spawn_sha256": "e" * 64}
    first = gate.consume_runtime_nonce(
        acceptance, pre_spawn, now=datetime(2026, 7, 12, 0, 4, tzinfo=timezone.utc),
    )
    assert first["entry"]["status"] == "BURNED_BEFORE_SPAWN_NO_REUSE"
    assert gate.verify_runtime_nonce_burn(first, acceptance)["inode"] == first["artifact"]["inode"]
    changed_acceptance = copy.deepcopy(acceptance)
    changed_acceptance["commit"] = "ref-swapped-after-burn"
    with pytest.raises(gate.GateError, match="NONCE_BURN_ACCEPTANCE_CHANGED"):
        gate.verify_runtime_nonce_burn(first, changed_acceptance)
    with pytest.raises(gate.GateError, match="RUNTIME_NONCE_REUSED"):
        gate.consume_runtime_nonce(
            acceptance, pre_spawn, now=datetime(2026, 7, 12, 0, 4, 1, tzinfo=timezone.utc),
        )


def test_nonce_burn_detects_delete_recreate_and_directory_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "authority"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(gate, "RUNTIME_AUTHORITY_ROOT", root)
    nonce = "7" * 64
    acceptance = {
        "commit": "manager", "blob_sha256": "8" * 64,
        "payload": {
            "nonce_authority": gate.runtime_nonce_boundary_authority(), "nonce": nonce,
            "command_sha256": "9" * 64, "canonical_authority_sha256": "a" * 64,
        },
    }
    burn = gate.consume_runtime_nonce(
        acceptance, {"pre_spawn_sha256": "b" * 64},
        now=datetime(2026, 7, 12, 0, 4, tzinfo=timezone.utc),
    )
    burned_path = Path(burn["artifact"]["path"])
    original = root / "original-burn"
    burned_path.rename(original)
    burned_path.write_bytes(original.read_bytes())
    burned_path.chmod(0o400)
    with pytest.raises(gate.GateError, match="NONCE_BURN_TAMPERED"):
        gate.verify_runtime_nonce_burn(burn, acceptance)
    burned_path.unlink()
    with pytest.raises(gate.GateError, match="MISSING_FILE"):
        gate.verify_runtime_nonce_burn(burn, acceptance)
    replacement_parent = tmp_path / "old-authority"
    root.rename(replacement_parent)
    root.mkdir(mode=0o700)
    with pytest.raises(gate.GateError, match="NONCE_AUTHORITY_REPLACED"):
        gate.verify_runtime_nonce_burn(burn, acceptance)


def test_nonce_burn_concurrent_replay_has_exactly_one_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "authority"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(gate, "RUNTIME_AUTHORITY_ROOT", root)
    acceptance = {
        "commit": "manager", "blob_sha256": "1" * 64,
        "payload": {
            "nonce_authority": gate.runtime_nonce_boundary_authority(), "nonce": "2" * 64,
            "command_sha256": "3" * 64, "canonical_authority_sha256": "4" * 64,
        },
    }

    def attempt() -> str:
        try:
            gate.consume_runtime_nonce(
                acceptance, {"pre_spawn_sha256": "5" * 64},
                now=datetime(2026, 7, 12, 0, 4, tzinfo=timezone.utc),
            )
            return "won"
        except gate.GateError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: attempt(), range(2)))
    assert sorted(outcomes) == ["RUNTIME_NONCE_REUSED", "won"]


def test_runtime_order_is_preflight_then_nonce_then_spawn() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    main_source = source[source.index("def main("):]
    preflight = main_source.rindex("immediate_pre_spawn_checks(")
    consume = main_source.rindex("consume_runtime_nonce(")
    verify = main_source.rindex("verify_runtime_nonce_burn(")
    acceptance = main_source.rindex("validate_runtime_acceptance(")
    spawn = main_source.rindex("run_reviewed_command(")
    assert consume < preflight < acceptance < verify < spawn


def test_runtime_acceptance_requires_exact_tracked_manager_blob(
    authority: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = build(authority, "authorized-run")
    run_repo = authority["root"] / "candidate"
    monkeypatch.setattr(gate, "RUN_REPO", run_repo)
    command = gate.build_exact_command(
        early_proof=proof, work_dir=authority["root"] / "authorized-run",
        candidate_authority={"path": str(run_repo), "head": gate.BASE_SHA, "clean": True, "detached": True},
        lock_authority={
            "root": str(gate.LOCK_ROOT), "commit": gate.LOCK_COMMIT,
            "wrapper": str(gate.LOCK_WRAPPER), "wrapper_sha256": gate.LOCK_WRAPPER_SHA256,
            "intent": gate.LOCK_INTENT,
        },
    )
    review_repo = authority["root"] / "review-repo"
    init_git_repo(review_repo)
    (review_repo / "tooling.txt").write_text("tooling\n")
    tooling = git_commit_all(review_repo, "tooling")
    review_path = review_repo / gate.CODE_REVIEW_ACCEPTANCE_RELATIVE
    review_path.parent.mkdir(parents=True)
    write_json(review_path, {
        "schema": gate.CODE_REVIEW_SCHEMA, "verdict": "ACCEPT",
        "reviewer_role": "independent_gpt56_sol_xhigh_critic",
        "tooling_commit": tooling, "manager_policy_sha256": gate.MANAGER_POLICY_SHA256,
        "release_policy_sha256": gate.RELEASE_POLICY_SHA256,
        "runtime_marker_status": "MAY_BE_PENDING_CODE_REVIEW_ONLY",
        "reviewed_utc": "2026-07-12T00:03:00Z",
    })
    review_commit = git_commit_all(review_repo, "review accept")
    subprocess.run(
        ["git", "-C", str(review_repo), "update-ref", gate.CODE_REVIEW_ACCEPTANCE_REF, review_commit], check=True,
    )
    monkeypatch.setattr(gate, "REVIEW_REPO", review_repo)

    manager_repo = authority["root"] / "manager-repo"
    init_git_repo(manager_repo)
    policy_path = manager_repo / gate.MANAGER_POLICY_RELATIVE
    policy_path.parent.mkdir(parents=True)
    policy_path.write_bytes(gate.MANAGER_POLICY.read_bytes())
    policy_commit = git_commit_all(manager_repo, "policy")
    monkeypatch.setattr(gate, "MANAGER_REPO", manager_repo)
    monkeypatch.setattr(gate, "MANAGER_POLICY", policy_path)
    monkeypatch.setattr(gate, "MANAGER_POLICY_COMMIT", policy_commit)
    runtime_root = authority["root"] / "runtime-authority"
    runtime_root.mkdir(mode=0o700)
    monkeypatch.setattr(gate, "RUNTIME_AUTHORITY_ROOT", runtime_root)
    monkeypatch.setattr(gate, "validate_executing_tooling", lambda commit: {"commit": commit})
    nonce = "9" * 64
    relative = f"{gate.RUNTIME_ACCEPTANCE_PREFIX}{nonce}.json"
    now = datetime.now(timezone.utc).replace(microsecond=0)
    payload = gate._runtime_acceptance_expected(
        early=proof, command=command, tooling_commit=tooling,
        code_review_commit=review_commit, nonce=nonce,
    )
    payload["issued_utc"] = (now - gate.timedelta(seconds=1)).isoformat()
    payload["expires_utc"] = (now + gate.timedelta(seconds=60)).isoformat()
    runtime_path = manager_repo / relative
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(runtime_path, payload)
    with pytest.raises(gate.GateError, match="INDEPENDENT_AUTHORITY_REF"):
        gate.validate_runtime_acceptance(
            relative_path=relative, command=command, early=proof, now=now,
        )
    manager_commit = git_commit_all(manager_repo, "one use runtime accept")
    subprocess.run(
        ["git", "-C", str(manager_repo), "update-ref", gate.MANAGER_RUNTIME_ACCEPTANCE_REF, manager_commit], check=True,
    )
    accepted = gate.validate_runtime_acceptance(
        relative_path=relative, command=command, early=proof, now=now,
    )
    assert accepted["commit"] == manager_commit
    assert accepted["payload"]["nonce"] == nonce
    burn = gate.consume_runtime_nonce(
        accepted, {"pre_spawn_sha256": "f" * 64}, now=now,
    )
    with pytest.raises(gate.GateError, match="RUNTIME_ACCEPTANCE_TIME_ORDER"):
        gate.validate_runtime_acceptance(
            relative_path=relative, command=command, early=proof,
            now=gate.parse_utc(payload["expires_utc"]),
        )
    assert Path(burn["artifact"]["path"]).exists()


def test_incremental_pair_only_compares_exact_same_valid_time(authority: dict[str, object]) -> None:
    cpu_dir = authority["root"] / "pair-cpu"
    gpu_dir = authority["root"] / "pair-gpu"
    cpu_dir.mkdir()
    gpu_dir.mkdir()
    write_frame(cpu_dir / "wrfout_d03_2025-03-01_00:00:00", "2025-03-01_00:00:00")
    write_frame(gpu_dir / "wrfout_d03_2025-03-01_00:20:00", "2025-03-01_00:20:00")
    state = gate.incremental_pair(cpu_dir, gpu_dir, authority["root"] / "pairs.json")
    assert state["matched_count"] == 0
    assert state["final_status"] == "FINAL_PENDING"
    shutil.copy2(
        cpu_dir / "wrfout_d03_2025-03-01_00:00:00",
        gpu_dir / "wrfout_d03_2025-03-01_00:00:00",
    )
    write_frame(cpu_dir / "wrfout_d03_2025-03-01_00:20:00", "2025-03-01_00:20:00")
    gate.incremental_pair(cpu_dir, gpu_dir, authority["root"] / "pairs.json")
    state = gate.incremental_pair(cpu_dir, gpu_dir, authority["root"] / "pairs.json")
    assert state["matched_count"] == 1
    assert state["pairs"][0]["valid_time"] == "2025-03-01T00:00:00+00:00"


def test_terminal_last_frame_requires_both_producers_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_dir = tmp_path / "cpu"
    gpu_dir = tmp_path / "gpu"
    cpu_dir.mkdir()
    gpu_dir.mkdir()
    stamp = gate.EXPECTED_TIMES[-1].strftime("%Y-%m-%d_%H:%M:%S")
    monkeypatch.setattr(gate, "EXPECTED_TIMES", (gate.EXPECTED_TIMES[-1],))
    name = f"wrfout_d03_{stamp}"
    write_frame(cpu_dir / name, stamp)
    shutil.copy2(cpu_dir / name, gpu_dir / name)
    manifest = tmp_path / "terminal.json"
    monkeypatch.setattr(gate, "CPU_MANIFEST", manifest)
    state_path = tmp_path / "pairs.json"
    assert gate.incremental_pair(cpu_dir, gpu_dir, state_path)["matched_count"] == 0
    write_json(manifest, {"terminal": True})
    calls: list[str] = []
    monkeypatch.setattr(gate, "validate_cpu_terminal_manifest", lambda *_args: calls.append("cpu") or {})
    monkeypatch.setattr(gate, "validate_gpu_completion_frames", lambda *_args, **_kwargs: calls.append("gpu") or {})
    monkeypatch.setattr(gate, "gpu_terminal_inventory_ready", lambda *_args, **_kwargs: True)
    assert gate.incremental_pair(cpu_dir, gpu_dir, state_path)["matched_count"] == 1
    assert calls == ["cpu", "gpu"]


def test_terminal_pair_source_hash_substitution_fails(authority: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    cpu_dir = authority["root"] / "source-cpu"
    gpu_dir = authority["root"] / "source-gpu"
    cpu_dir.mkdir()
    gpu_dir.mkdir()
    name = write_closed_first_pair(cpu_dir, gpu_dir)
    state_path = authority["root"] / "source-pairs.json"
    gate.incremental_pair(cpu_dir, gpu_dir, state_path)
    state = gate.incremental_pair(cpu_dir, gpu_dir, state_path)
    monkeypatch.setattr(gate, "CPU_INPUT_DIR", cpu_dir)
    stamp = "2025-03-01_00:00:00"
    cpu_authority = gate.file_authority(cpu_dir / name)
    gpu_authority = gate.file_authority(gpu_dir / name)
    cpu_terminal = {"recomputed": {"raw_qa": {"d03": {stamp: {"authority": cpu_authority}}}}}
    gpu_completion = {"frame_authority": {"d03": {stamp: dict(gpu_authority)}}}
    gpu_completion["frame_authority"]["d03"][stamp]["sha256"] = "0" * 64
    with pytest.raises(gate.GateError, match="PAIR_TERMINAL_SOURCE_BINDING"):
        gate.cross_bind_terminal_pair_sources(state, cpu_terminal, gpu_completion, gpu_dir)


def test_terminal_cpu_manifest_still_pending_until_55_pairs(authority: dict[str, object]) -> None:
    cpu_dir = authority["root"] / "pending-cpu"
    gpu_dir = authority["root"] / "gpu-output"
    cpu_dir.mkdir()
    gpu_dir.mkdir()
    name = "wrfout_d03_2025-03-01_00:00:00"
    write_frame(cpu_dir / name, "2025-03-01_00:00:00")
    shutil.copy2(cpu_dir / name, gpu_dir / name)
    state_path = authority["root"] / "incremental-pairs.json"
    gate.incremental_pair(cpu_dir, gpu_dir, state_path)
    manifest_path = authority["root"] / "terminal-manifest.json"
    write_json(manifest_path, {
        "schema": "tenerife_fullbuffer_cpu_oracle_v1",
        "status": "complete", "qa_status": "pass",
        "case_id": gate.CASE_ID, "grid_id": gate.GRID_ID,
        "grid_sha256": gate.GRID_SHA256,
        "raw_frame_counts": {"d01": 19, "d02": 19, "d03": 55},
        "thin_frame_count": 55,
        "pair_counts": {"nine_to_three": 19, "aifs_to_1km_physical_target": 55, "three_to_one": 55},
        "all_required_fields_finite": True,
        "all_grid_fields_match_frozen_geo": True,
        "forcing_sha256": gate.AIFS_SHA256,
        "gpu_identity_contract": {
            "matched_raw_domain": "d03",
            "exclude_fields": ["QVAPOR", "RAINNC"],
            "valid_times": [value.strftime("%Y-%m-%d_%H:%M:%S") for value in gate.EXPECTED_TIMES],
        },
    })
    gate.CPU_MANIFEST.write_bytes(manifest_path.read_bytes())
    result = gate.final_verdict(
        gate.CPU_MANIFEST, state_path, authority["root"] / "identity-numbers-first.jpg",
        gpu_dir,
    )
    assert result == {"verdict": "FINAL_PENDING", "reason": "matched_frames_not_55"}


def test_gpu_completion_requires_19_19_55_not_d03_only(tmp_path: Path) -> None:
    output = tmp_path / "gpu-output"
    output.mkdir()
    with pytest.raises(gate.GateError, match="d01: expected=19 observed=0"):
        gate.validate_gpu_completion_frames(output)


def test_terminal_gpu_completion_uses_exact_offset_d01_and_rejects_normalized_substitution(
    tmp_path: Path,
) -> None:
    assert gate.gpu_terminal_inventory_ready(gate.CPU_INPUT_DIR, terminal_mode=True) is True
    assert gate.gpu_terminal_inventory_ready(gate.CPU_INPUT_DIR, terminal_mode=False) is False
    completion = gate.validate_gpu_completion_frames(gate.CPU_INPUT_DIR, terminal_mode=True)
    assert completion["counts"] == {"d01": 19, "d02": 19, "d03": 55}
    normalized = tmp_path / "normalized-gpu"
    normalized.mkdir()
    schedules = {
        "d01": gate.EXPECTED_PARENT_TIMES,
        "d02": gate.EXPECTED_PARENT_TIMES,
        "d03": gate.EXPECTED_TIMES,
    }
    for domain, times in schedules.items():
        for valid in times:
            (normalized / f"wrfout_{domain}_{valid.strftime('%Y-%m-%d_%H:%M:%S')}").write_bytes(b"")
    exact_d01 = set(gate.TERMINAL_D01_TIMES)
    normalized_d01 = set(gate.EXPECTED_PARENT_TIMES)
    assert len(exact_d01 - normalized_d01) == 12
    assert len(normalized_d01 - exact_d01) == 12
    assert gate.gpu_terminal_inventory_ready(normalized, terminal_mode=True) is False
    with pytest.raises(gate.GateError, match="GPU_FRAME_SET.*d01"):
        gate.validate_gpu_completion_frames(normalized, terminal_mode=True)


def test_terminal_cpu_authority_recomputes_raw_thin_pair_and_policy(terminal_cpu: dict[str, Path]) -> None:
    result = gate.validate_cpu_terminal_manifest(terminal_cpu["manifest"], sleep_fn=lambda _: None)
    assert result["recomputed"]["raw_counts"] == {"d01": 19, "d02": 19, "d03": 55}
    assert result["recomputed"]["thin_count"] == 55
    assert result["recomputed"]["pair_index_rows"] == 129
    assert result["recomputed"]["effective_owner_report_only_fields"] == ["QVAPOR", "RAINC", "RAINNC"]


def test_terminal_cpu_manifest_arbitrary_path_rejected(terminal_cpu: dict[str, Path]) -> None:
    alternate = terminal_cpu["run"] / "self-minted.json"
    alternate.write_bytes(terminal_cpu["manifest"].read_bytes())
    with pytest.raises(gate.GateError, match="CPU_MANIFEST_CANONICAL_PATH"):
        gate.validate_cpu_terminal_manifest(alternate, sleep_fn=lambda _: None)


def test_terminal_cpu_pair_index_and_raw_claim_substitution_fail(terminal_cpu: dict[str, Path]) -> None:
    with terminal_cpu["pair_index"].open("a", encoding="utf-8") as handle:
        handle.write('{"self_minted":true}\n')
    with pytest.raises(gate.GateError, match="CPU_PAIR_INDEX_SUBSTITUTION"):
        gate.validate_cpu_terminal_manifest(terminal_cpu["manifest"], sleep_fn=lambda _: None)


def test_terminal_cpu_raw_and_thin_are_recomputed_not_manifest_assertions(terminal_cpu: dict[str, Path]) -> None:
    stamp = gate.EXPECTED_TIMES[0].strftime("%Y-%m-%d_%H:%M:%S")
    raw = terminal_cpu["wrf"] / f"wrfout_d03_{stamp}"
    with Dataset(raw, "r+") as dataset:
        dataset.variables["T2"][0, 0, 0] = np.nan
    with pytest.raises(gate.GateError, match="CPU_RAW_NONFINITE"):
        gate.validate_cpu_terminal_manifest(terminal_cpu["manifest"], sleep_fn=lambda _: None)


def test_terminal_cpu_raw_thin_pair_mismatch_fails(terminal_cpu: dict[str, Path]) -> None:
    stamp = gate.EXPECTED_TIMES[0].strftime("%Y-%m-%d_%H:%M:%S")
    thin = terminal_cpu["thin"] / f"wrfout_d03_{stamp}.thin.nc"
    with Dataset(thin, "r+") as dataset:
        dataset.variables["T2"][0, 0, 0] = 291.0
    with pytest.raises(gate.GateError, match="CPU_RAW_THIN_MISMATCH"):
        gate.validate_cpu_terminal_manifest(terminal_cpu["manifest"], sleep_fn=lambda _: None)


def test_terminal_raw_one_bit_geometry_substitution_fails(terminal_cpu: dict[str, Path]) -> None:
    stamp = gate.EXPECTED_TIMES[0].strftime("%Y-%m-%d_%H:%M:%S")
    raw = terminal_cpu["wrf"] / f"wrfout_d03_{stamp}"
    with Dataset(raw, "r+") as dataset:
        dataset.variables["HGT"][0, 0, 0] = np.nextafter(500.0, 501.0)
    with pytest.raises(gate.GateError, match="CPU_RAW_GEOMETRY"):
        gate.validate_cpu_terminal_manifest(terminal_cpu["manifest"], sleep_fn=lambda _: None)


def test_terminal_thin_one_bit_geometry_substitution_fails(terminal_cpu: dict[str, Path]) -> None:
    stamp = gate.EXPECTED_TIMES[0].strftime("%Y-%m-%d_%H:%M:%S")
    thin = terminal_cpu["thin"] / f"wrfout_d03_{stamp}.thin.nc"
    with Dataset(thin, "r+") as dataset:
        dataset.variables["XLAT"][0, 0, 0] = np.nextafter(28.2, 29.0)
    with pytest.raises(gate.GateError, match="FROZEN_GEOMETRY_EXACT"):
        gate.validate_cpu_terminal_manifest(terminal_cpu["manifest"], sleep_fn=lambda _: None)


def test_pair_source_mutation_after_snapshot_fails_closed(authority: dict[str, object]) -> None:
    cpu_dir = authority["root"] / "mut-cpu"
    gpu_dir = authority["root"] / "mut-gpu"
    cpu_dir.mkdir()
    gpu_dir.mkdir()
    name = write_closed_first_pair(cpu_dir, gpu_dir)
    state_path = authority["root"] / "mut-pairs.json"
    gate.incremental_pair(cpu_dir, gpu_dir, state_path)
    gate.incremental_pair(cpu_dir, gpu_dir, state_path)
    with Dataset(gpu_dir / name, "r+") as dataset:
        dataset.variables["T"][0, 0, 0, 0] = 9.0
    with pytest.raises(gate.GateError, match="PAIR_SOURCE_CHANGED_AFTER_SNAPSHOT"):
        gate.incremental_pair(cpu_dir, gpu_dir, state_path)


def test_pair_source_change_during_fd_copy_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.nc"
    source.write_bytes(b"original stable bytes")
    original_read = gate.os.read
    mutated = False

    def read_and_mutate(fd: int, size: int) -> bytes:
        nonlocal mutated
        chunk = original_read(fd, size)
        if chunk and not mutated:
            mutated = True
            source.write_bytes(b"changed while copied")
        return chunk

    monkeypatch.setattr(gate.os, "read", read_and_mutate)
    with pytest.raises(gate.GateError, match="PAIR_SOURCE_CHANGED_DURING_COPY"):
        gate.immutable_file_snapshot(source, tmp_path / "snapshots" / "copy.nc")


def test_pair_metric_substitution_fails_even_with_recomputed_digest(authority: dict[str, object]) -> None:
    cpu_dir = authority["root"] / "metric-cpu"
    gpu_dir = authority["root"] / "metric-gpu"
    cpu_dir.mkdir()
    gpu_dir.mkdir()
    write_closed_first_pair(cpu_dir, gpu_dir)
    state_path = authority["root"] / "metric-pairs.json"
    gate.incremental_pair(cpu_dir, gpu_dir, state_path)
    gate.incremental_pair(cpu_dir, gpu_dir, state_path)
    state = json.loads(state_path.read_text())
    state["pairs"][0]["metrics"]["T"]["rmse"] = 0.25
    unsigned = dict(state)
    unsigned.pop("authority_sha256")
    state["authority_sha256"] = gate.authority_digest(unsigned)
    write_json(state_path, state)
    with pytest.raises(gate.GateError, match="PAIR_METRIC_SUBSTITUTION"):
        gate.validate_pair_state_integrity(gate.load_exact_json(state_path), recompute=True)


def test_pair_metrics_use_n_pooled_policy_and_complete_inventory(authority: dict[str, object]) -> None:
    cpu_dir = authority["root"] / "schema-cpu"
    gpu_dir = authority["root"] / "schema-gpu"
    cpu_dir.mkdir()
    gpu_dir.mkdir()
    write_closed_first_pair(cpu_dir, gpu_dir)
    state_path = authority["root"] / "schema-pairs.json"
    gate.incremental_pair(cpu_dir, gpu_dir, state_path)
    state = gate.incremental_pair(cpu_dir, gpu_dir, state_path)
    metric = state["pairs"][0]["metrics"]["T"]
    assert metric["n"] > 0 and "count" not in metric
    assert metric["gate"] == "STRICT_POOLED_RMSE"
    assert state["pooled_metrics"]["T"]["frame_count"] == 1
    assert state["pooled_metrics"]["T"]["pass"] is False
    assert state["pairs"][0]["per_frame_static_pass"] is True
    assert state["identity_policy"]["owner_report_only_fields"] == ["QVAPOR", "RAINC", "RAINNC"]
    assert "fields" in state["complete_field_inventory"]


def test_steady_pair_poll_full_reads_are_frontier_bounded_independent_of_history(
    authority: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_dir = authority["root"] / "bounded-cpu"
    gpu_dir = authority["root"] / "bounded-gpu"
    cpu_dir.mkdir()
    gpu_dir.mkdir()
    first_name = write_closed_first_pair(cpu_dir, gpu_dir)
    state_path = authority["root"] / "bounded-pairs.json"
    gate.incremental_pair(cpu_dir, gpu_dir, state_path)
    base = gate.incremental_pair(cpu_dir, gpu_dir, state_path)
    base_row = base["pairs"][0]
    cpu_source = cpu_dir / first_name
    gpu_source = gpu_dir / first_name

    monkeypatch.setattr(gate, "verify_pinned_file_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(gate, "source_pinned_metadata", lambda *_args, **_kwargs: {})
    history_size = 1

    def discover(directory: Path) -> dict[datetime, Path]:
        source = cpu_source if directory == cpu_dir else gpu_source
        return {valid: source for valid in gate.EXPECTED_TIMES[:history_size + 2]}

    monkeypatch.setattr(gate, "discover_d03_frames", discover)
    original_sha = gate.sha256_file
    counts = {"sha": 0, "qa": 0}

    def counted_sha(path: Path) -> str:
        if Path(path).name.startswith("wrfout_d03_"):
            counts["sha"] += 1
        return original_sha(Path(path))

    def counted_qa(path: Path, _valid: datetime) -> dict[str, object]:
        counts["qa"] += 1
        return {"sha256": gate.sha256_file(path)}

    monkeypatch.setattr(gate, "sha256_file", counted_sha)
    monkeypatch.setattr(gate, "qa_d03_frame", counted_qa)

    def one_poll(size: int) -> tuple[int, int]:
        nonlocal history_size
        history_size = size
        state = copy.deepcopy(base)
        state["pairs"] = []
        for index in range(size):
            row = copy.deepcopy(base_row)
            row["valid_time"] = gate.EXPECTED_TIMES[index].isoformat()
            state["pairs"].append(row)
        state["matched_count"] = size
        state["closure_observations"] = {}
        state["pooled_metrics"], state["complete_field_inventory"] = gate.pooled_metrics(state["pairs"])
        unsigned = dict(state)
        unsigned.pop("authority_sha256", None)
        state["authority_sha256"] = gate.authority_digest(unsigned)
        write_json(state_path, state)
        counts.update(sha=0, qa=0)
        result = gate.incremental_pair(cpu_dir, gpu_dir, state_path)
        assert result["matched_count"] == size
        return counts["sha"], counts["qa"]

    one_pair = one_poll(1)
    forty_pairs = one_poll(40)
    fifty_four_pairs = one_poll(54)
    assert one_pair == forty_pairs
    assert forty_pairs[0] <= 8
    assert forty_pairs[1] <= 4
    assert fifty_four_pairs[0] <= 4
    assert fifty_four_pairs[1] <= 2


def test_revocation_preempts_delayed_pair_without_entering_pair_scan(
    authority: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = build(authority)
    write_json(gate.EARLY_REVOKED, {
        "schema": gate.UPSTREAM_REVOKE_SCHEMA,
        "status": "revoked_gpu_stop_required",
        "gpu_action": "STOP_FAIL_CLOSED_AND_RELEASE_LOCK",
    })
    pair_called = False

    def forbidden_pair(*_args, **_kwargs):
        nonlocal pair_called
        pair_called = True
        raise AssertionError("historical pair scan must not precede revocation")

    monkeypatch.setattr(gate, "incremental_pair", forbidden_pair)
    with pytest.raises(gate.GateError, match="EARLY_REVOKED_STOP_AND_RELEASE"):
        gate.active_monitor_poll(
            early_proof=proof, expected_work_dir=authority["root"] / "work",
            cpu_manifest_path=gate.CPU_MANIFEST,
            cpu_dir=gate.CPU_INPUT_DIR, gpu_dir=authority["root"] / "delayed-gpu",
            pair_state_path=authority["root"] / "delayed-pairs.json", terminal_cache={},
        )
    assert pair_called is False


def test_fd_holder_policy_failure_preempts_pairing_every_poll(
    authority: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = build(authority)
    pair_called = False

    def rejected_scan(*_args, **_kwargs):
        raise gate.GateError("INPUT_FD_WRITABLE", "deterministic current holder")

    def forbidden_pair(*_args, **_kwargs):
        nonlocal pair_called
        pair_called = True
        raise AssertionError("pairing must follow continuous FD safety")

    monkeypatch.setattr(gate, "scan_current_input_open_fds", rejected_scan)
    monkeypatch.setattr(gate, "incremental_pair", forbidden_pair)
    with pytest.raises(gate.GateError, match="INPUT_FD_WRITABLE"):
        gate.active_monitor_poll(
            early_proof=proof, expected_work_dir=authority["root"] / "work",
            cpu_manifest_path=gate.CPU_MANIFEST,
            cpu_dir=gate.CPU_INPUT_DIR, gpu_dir=authority["root"] / "delayed-gpu",
            pair_state_path=authority["root"] / "delayed-pairs.json", terminal_cache={},
        )
    assert pair_called is False


def test_monitor_repins_input_inode_mode_and_rescans_fds_each_time(
    authority: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = build(authority)
    calls = 0

    def scan(*_args, **_kwargs) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"current_policy": "read_only_exact_inode_holders_only", "targets": {}}

    monkeypatch.setattr(gate, "scan_current_input_open_fds", scan)
    monkeypatch.setattr(
        gate, "verify_live_cpu_identity", lambda *_args: proof["cpu_live_identity"],
    )
    gate.monitor_once(
        early_proof=proof, expected_work_dir=authority["root"] / "work",
        cpu_manifest_path=gate.CPU_MANIFEST,
    )
    gate.monitor_once(
        early_proof=proof, expected_work_dir=authority["root"] / "work",
        cpu_manifest_path=gate.CPU_MANIFEST,
    )
    assert calls == 2
    target = authority["wrf"] / "wrfinput_d03"
    target.chmod(0o644)
    with pytest.raises(gate.GateError, match="INPUT_AUTHORITY_CHANGED_DURING_RUN"):
        gate.monitor_once(
            early_proof=proof, expected_work_dir=authority["root"] / "work",
            cpu_manifest_path=gate.CPU_MANIFEST,
        )


def test_terminal_monitor_preserves_preemption_input_fd_and_cache_without_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "wrfinput_d03"
    input_path.write_bytes(b"sealed")
    input_path.chmod(0o444)
    authority = gate.file_authority(input_path)
    artifact_path = tmp_path / "authority.json"
    write_json(artifact_path, {"terminal": True})
    artifact = gate.file_authority(artifact_path)
    cpu_terminal = {"recomputed": {"manifest_artifact": artifact, "pinned": authority}}
    proof = {
        "table_authority": {"authority_sha256": "a" * 64},
        "input_authority": {"wrfinput_d03": authority},
        "terminal_cpu": cpu_terminal,
        "terminal_authority": {
            "artifact": artifact, "process": {"artifact": artifact},
            "terminal_cpu": {"recomputed": {"manifest_artifact": artifact, "pair_index": artifact}},
        },
    }
    for name in ("EARLY_MARKER", "EARLY_REVOKED", "EARLY_INVALIDATED"):
        monkeypatch.setattr(gate, name, tmp_path / name)
    calls: list[str] = []
    monkeypatch.setattr(gate, "require_current_no_preemption", lambda: calls.append("preempt") or {})
    monkeypatch.setattr(gate, "verify_table_snapshot", lambda *_args, **_kwargs: calls.append("table") or {})
    monkeypatch.setattr(gate, "scan_current_input_open_fds", lambda *_args: calls.append("fd") or {
        "targets": {"wrfinput_d03": {"holders": []}},
    })
    monkeypatch.setattr(gate, "verify_cached_terminal_metadata", lambda *_args: calls.append("cache") or {})
    monkeypatch.setattr(gate, "verify_cached_terminal_manifest_sha", lambda *_args: calls.append("manifest") or {})
    result = gate.terminal_monitor_once(
        terminal_proof=proof, expected_work_dir=tmp_path, terminal_cache={},
    )
    assert result["status"] == "CONTINUE_TERMINAL_CPU_AUTHENTICATED"
    assert calls == ["preempt", "table", "fd", "cache", "manifest"]
    (tmp_path / "EARLY_MARKER").write_text("forbidden", encoding="utf-8")
    with pytest.raises(gate.GateError, match="TERMINAL_REJECTS_EARLY_MARKER"):
        gate.terminal_monitor_once(
            terminal_proof=proof, expected_work_dir=tmp_path, terminal_cache={},
        )


def test_terminal_cpu_validation_is_cached_then_metadata_only(
    authority: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = build(authority)
    write_json(gate.CPU_MANIFEST, {"terminal": True})
    _payload, manifest_artifact = gate.stable_json_authority(
        gate.CPU_MANIFEST, interval_seconds=0.0, sleep_fn=lambda _: None,
    )
    calls = 0

    def validate(*_args, **_kwargs) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"recomputed": {"manifest_artifact": manifest_artifact}}

    monkeypatch.setattr(gate, "validate_cpu_terminal_manifest", validate)
    cache: dict[str, object] = {}
    gate.monitor_once(
        early_proof=proof, expected_work_dir=authority["root"] / "work",
        cpu_manifest_path=gate.CPU_MANIFEST, terminal_cache=cache,
    )
    gate.monitor_once(
        early_proof=proof, expected_work_dir=authority["root"] / "work",
        cpu_manifest_path=gate.CPU_MANIFEST, terminal_cache=cache,
    )
    assert calls == 1


def test_real_stable_manifest_cache_rejects_same_size_mutation_with_restored_mtime(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "cpu_oracle_manifest.json"
    manifest.write_bytes(b'{"status":"complete"}\n')
    _payload, artifact = gate.stable_json_authority(
        manifest, interval_seconds=0.0, sleep_fn=lambda _: None,
    )
    cache = {"recomputed": {"manifest_artifact": artifact}}
    incomplete = copy.deepcopy(cache)
    del incomplete["recomputed"]["manifest_artifact"]["ctime_ns"]
    with pytest.raises(gate.GateError, match="lack complete metadata"):
        gate.verify_cached_terminal_metadata(incomplete, "CPU_TERMINAL_CACHE_CHANGED")
    gate.verify_cached_terminal_metadata(cache, "CPU_TERMINAL_CACHE_CHANGED")
    before = manifest.stat()
    with manifest.open("r+b") as handle:
        handle.write(b'{"status":"rejected"}\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.utime(manifest, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = manifest.stat()
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
    )
    assert after.st_ctime_ns != before.st_ctime_ns
    with pytest.raises(gate.GateError, match="CPU_TERMINAL_CACHE_CHANGED"):
        gate.verify_cached_terminal_metadata(cache, "CPU_TERMINAL_CACHE_CHANGED")
    with pytest.raises(gate.GateError, match="CPU_TERMINAL_CACHE_MANIFEST_SHA"):
        gate.verify_cached_terminal_manifest_sha(cache, manifest)


def test_no_jax_gpuwrf_or_cuda_probe_in_tool_imports() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = __import__("ast").parse(source)
    imports = {
        alias.name
        for node in __import__("ast").walk(tree)
        if isinstance(node, __import__("ast").Import)
        for alias in node.names
    }
    assert "jax" not in imports
    assert "gpuwrf" not in imports
    assert "validate_live_gpu_device(" in source


def test_cli_missing_marker_fails_without_importing_jax(tmp_path: Path) -> None:
    sentinel = tmp_path / "jax.py"
    sentinel.write_text("raise RuntimeError('JAX IMPORTED')\n", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "audit", "--work-dir", str(tmp_path / "work"), "--marker", str(tmp_path / "absent.json"), "--stability-seconds", "0"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=False,
    )
    assert result.returncode == 75
    assert "JAX IMPORTED" not in result.stdout
