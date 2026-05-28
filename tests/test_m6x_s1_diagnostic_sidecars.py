from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "diagnostic",
    "input",
    "measurements",
    "units",
    "status",
    "source_citations",
}


def _module(script_name: str) -> ModuleType:
    path = SCRIPTS / script_name
    spec = importlib.util.spec_from_file_location(script_name.removesuffix(".py"), path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(tmp_path: Path, script_name: str, payload: dict | None = None, input_path: Path | None = None) -> dict:
    module = _module(script_name)
    source = input_path or _write_json(tmp_path / f"{script_name}.input.json", payload or {})
    output = tmp_path / f"{script_name}.output.json"
    assert module.main(["--input", str(source), "--output", str(output)]) == 0
    proof = json.loads(output.read_text(encoding="utf-8"))
    assert REQUIRED_TOP_LEVEL <= set(proof)
    assert proof["schema_version"] == "m6x-s1-diagnostic-sidecar-v1"
    assert proof["diagnostic"]["name"]
    return proof


def test_bound_violation_tracer_smoke(tmp_path: Path) -> None:
    proof = _run(
        tmp_path,
        "diagnostic_bound_violation_tracer.py",
        {
            "series": [
                {"step": 0, "time_s": 0.0, "theta_K": 300.0},
                {"step": 1, "time_s": 1.0, "theta_K": 420.0, "i": 2, "j": 1, "k": 0},
            ],
            "bounds": {"theta_K": {"min": 150.0, "max": 400.0, "units": "K"}},
        },
    )
    assert "first_violation" in proof["measurements"]


def test_sanitizer_audit_smoke(tmp_path: Path) -> None:
    proof = _run(
        tmp_path,
        "diagnostic_sanitizer_audit.py",
        {"sanitizer_steps": [{"step": 1, "candidate_nonfinite_count": 0}, {"step": 2, "candidate_clip_count": 3}]},
    )
    assert "first_bad_candidate_step" in proof["measurements"]


def test_limiter_activation_tracker_smoke(tmp_path: Path) -> None:
    proof = _run(
        tmp_path,
        "diagnostic_limiter_activation_tracker.py",
        {"limiter_steps": [{"step": 1, "raw_dmu": [[100.0, -5.0]], "bounded_dmu": [[10.0, -5.0]]}]},
    )
    assert "max_saturation_fraction" in proof["measurements"]


def test_field_rmse_timeline_smoke(tmp_path: Path) -> None:
    proof = _run(
        tmp_path,
        "diagnostic_field_rmse_timeline.py",
        {
            "leads": [
                {
                    "lead_time_s": 60.0,
                    "forecast": {"T2": [[300.0, 301.0]], "U10": [[1.0, 2.0]]},
                    "reference": {"T2": [[299.0, 301.5]], "U10": [[0.0, 2.5]]},
                }
            ]
        },
    )
    assert proof["measurements"]["timeline"]


def test_spatial_divergence_map_smoke(tmp_path: Path) -> None:
    proof = _run(
        tmp_path,
        "diagnostic_spatial_divergence_map.py",
        {
            "field": "T2",
            "forecast": {"T2": [[1.0, 2.0], [3.0, 4.0]]},
            "reference": {"T2": [[1.5, 1.5], [2.5, 5.0]]},
            "elevation_m": [[0.0, 10.0], [20.0, 30.0]],
            "landmask": [[0, 1], [1, 1]],
        },
    )
    assert Path(proof["artifacts"]["spatial_error_npz"]).exists()


def test_conservation_tracker_smoke(tmp_path: Path) -> None:
    proof = _run(
        tmp_path,
        "diagnostic_conservation_tracker.py",
        {
            "states": [
                {"step": 0, "mass": [[1.0, 1.0]], "qv": [[0.01, 0.02]], "u": [[1.0, 0.0]], "theta": [[300.0, 301.0]]},
                {"step": 1, "mass": [[1.1, 0.9]], "qv": [[0.01, 0.02]], "u": [[1.1, 0.0]], "theta": [[300.5, 301.0]]},
            ]
        },
    )
    assert "max_abs_relative_drift" in proof["measurements"]


def test_boundary_ring_error_profiler_smoke(tmp_path: Path) -> None:
    proof = _run(
        tmp_path,
        "diagnostic_boundary_ring_error_profiler.py",
        {
            "field": "T2",
            "forecast": {"T2": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]},
            "reference": {"T2": [[1.0, 1.0, 3.0], [1.0, 2.0, 4.0], [0.0, 2.0, 3.0]]},
        },
    )
    assert len(proof["measurements"]["ring_rmse"]) == 4


def test_vertical_column_phase_space_smoke(tmp_path: Path) -> None:
    proof = _run(
        tmp_path,
        "diagnostic_vertical_column_phase_space.py",
        {
            "columns": [
                {
                    "name": "center",
                    "i": 2,
                    "j": 3,
                    "profiles": {"w": [0.0, 0.1], "theta": [300.0, 301.0]},
                    "time_series": {"w": [0.0, 0.1], "theta": [300.0, 301.0], "p": [0.0, 2.0], "mu": [0.0, 1.0]},
                }
            ]
        },
    )
    assert proof["measurements"]["columns"]


def test_operator_term_budget_tracer_smoke(tmp_path: Path) -> None:
    proof = _run(
        tmp_path,
        "diagnostic_operator_term_budget_tracer.py",
        {"terms": {"buoyancy": [[0.1, 0.2]], "pressure_restoring": [[-1.0, 0.5]], "rayleigh": [[0.0, 0.0]]}},
    )
    assert "ranking" in proof["measurements"]


def test_transfer_launch_timeline_smoke(tmp_path: Path) -> None:
    proof = _run(
        tmp_path,
        "diagnostic_transfer_launch_timeline.py",
        {
            "transfer_audit": {
                "static": {"host_callback_free": True},
                "trace": {"host_to_device_bytes_post_init": 0, "device_to_host_bytes_post_init": 0},
            },
            "launch_count": 2,
            "peak_gpu_memory": {"peak_bytes_in_use": 1024},
        },
    )
    assert proof["measurements"]["post_init_total_transfer_bytes"] == 0


def test_timestep_convergence_dashboard_smoke(tmp_path: Path) -> None:
    proof = _run(
        tmp_path,
        "diagnostic_timestep_convergence_dashboard.py",
        {"dt_pairs": [{"lead_time_s": 10.0, "dt_coarse_s": 2.0, "dt_fine_s": 1.0, "coarse": {"w": [1.0]}, "fine": {"w": [0.8]}}]},
    )
    assert proof["measurements"]["convergence_verdict"] == "PLACEHOLDER_PENDING_S4"


def test_stabilizer_provenance_scanner_smoke(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "candidate.py").write_text(
        "\"\"\"WRF source anchor: module_small_step_em.F:562.\"\"\"\n"
        "def apply_smdiv_pressure(p, pm1):\n"
        "    return p + 0.1 * (p - pm1)  # smdiv damping\n",
        encoding="utf-8",
    )
    proof = _run(tmp_path, "diagnostic_stabilizer_provenance_scanner.py", input_path=source)
    assert "classification_counts" in proof["measurements"]
