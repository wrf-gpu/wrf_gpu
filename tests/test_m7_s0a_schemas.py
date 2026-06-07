from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuwrf.io.proof_schemas import (
    AIFSIngestManifest,
    OperationalOutput,
    OperationalScheduler,
    OperationalStatus,
    StationObservationSourceManifest,
    schema_for_artifact,
    validate_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
_AIFS_INGEST_MANIFEST = ROOT / "data/manifests/aifs_ingest_v0.json"


def _aifs_cited_paths_present() -> bool:
    """The AIFS-ingest manifest cites Gen2 WPS-case artifacts (per-case
    namelist.wps / ungrib gribs / met_em) that live on the workstation /mnt
    corpus and are purged when a WPS case directory is cleaned up. The
    existence cross-check below is skipped when those cited artifacts are gone."""
    try:
        data = json.loads(_AIFS_INGEST_MANIFEST.read_text())
        return all(Path(p).exists() for p in data.get("artifact_paths", []))
    except Exception:
        return False


@pytest.mark.skipif(
    not _aifs_cited_paths_present(),
    reason="AIFS-ingest manifest cites purged Gen2 WPS-case artifacts (namelist.wps / ungrib gribs)",
)
def test_aifs_ingest_manifest_validates_and_cites_existing_gen2_paths():
    manifest = ROOT / "data/manifests/aifs_ingest_v0.json"

    data = AIFSIngestManifest.validate_file(manifest)

    assert data["strategy"] == "reuse_gen2_wps_v0"
    assert data["frequency_hours"] == 6
    assert data["valid_time_window"]["required_steps_h_v0"] == [0, 6, 12, 18, 24]
    for artifact_path in data["artifact_paths"]:
        assert Path(artifact_path).exists(), artifact_path


def test_station_source_manifest_validates_and_keeps_access_statuses_honest():
    manifest = ROOT / "data/manifests/station_obs_sources_v0.json"

    data = StationObservationSourceManifest.validate_file(manifest)

    statuses = {source["source_id"]: source["status"] for source in data["sources"]}
    assert statuses["metar_canary_airports"] == "AVAILABLE"
    assert statuses["aemet_conventional_canary"] == "PARTIAL"
    assert statuses["grafcan_sitcan"] == "PARTIAL"
    assert data["binding_policy"]["operational_claim_rule"].startswith("M7 cannot claim operational validation")
    for artifact_path in data["artifact_paths"]:
        assert Path(artifact_path).exists(), artifact_path


def test_registry_validates_new_manifest_filenames():
    aifs_path = ROOT / "data/manifests/aifs_ingest_v0.json"
    station_path = ROOT / "data/manifests/station_obs_sources_v0.json"

    assert schema_for_artifact(aifs_path) is AIFSIngestManifest
    assert schema_for_artifact(station_path) is StationObservationSourceManifest
    assert validate_artifact(aifs_path)["status"] == "AVAILABLE"
    assert validate_artifact(station_path)["status"] == "PARTIAL"


def test_operational_output_round_trip_schema(tmp_path: Path):
    payload = {
        "cycle_id": "20260520_18z",
        "status": "PUBLISHED",
        "init_time_utc": "2026-05-20T18:00:00Z",
        "valid_times_utc": ["2026-05-20T18:00:00Z", "2026-05-21T18:00:00Z"],
        "domains": {"d02": {"dx_m": 3000, "shape": [66, 159]}},
        "fields": {"T2": {"units": "K", "dimensions": ["south_north", "west_east"]}},
        "output_formats": ["netcdf_hourly", "zarr_cycle"],
        "root_path": "/mnt/data/wrf_gpu2/operational/m7/cycles/20260520_18z",
        "attrs": {"model": "gpuwrf", "schema": "OperationalOutput"},
        "artifact_paths": ["artifacts/m7/postprocess/product_manifest.json"],
    }
    path = tmp_path / "operational_output.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert OperationalOutput.validate_file(path)["fields"]["T2"]["units"] == "K"


def test_operational_status_accepts_running_null_exit_code():
    payload = {
        "cycle_id": "20260520_18z",
        "status": "RUNNING",
        "init_time_utc": "2026-05-20T18:00:00Z",
        "updated_utc": "2026-05-21T06:00:00Z",
        "wall_clock_s": 120.5,
        "exit_code": None,
        "current_step": "forecast",
        "alert_flags": [],
        "artifact_index_path": "artifacts/m7/prologue/proof_index.json",
        "last_good_cycle_id": None,
    }

    assert OperationalStatus.validate_dict(payload)["exit_code"] is None


def test_operational_status_rejects_wrong_alert_type():
    payload = {
        "cycle_id": "20260520_18z",
        "status": "FAILED",
        "init_time_utc": "2026-05-20T18:00:00Z",
        "updated_utc": "2026-05-21T06:00:00Z",
        "wall_clock_s": 120.5,
        "exit_code": 1,
        "current_step": "forecast",
        "alert_flags": "GPU_OOM",
        "artifact_index_path": "artifacts/m7/prologue/proof_index.json",
        "last_good_cycle_id": "20260519_18z",
    }

    with pytest.raises(TypeError, match="alert_flags"):
        OperationalStatus.validate_dict(payload)


def test_operational_scheduler_schema_validates_daily_18z_contract():
    payload = {
        "scheduler_id": "m7-v0-daily-18z",
        "status": "DRAFT",
        "cycle_hour_utc": 18,
        "timezone": "UTC",
        "aifs_poll_start_utc": "01:25",
        "aifs_poll_timeout_utc": "05:25",
        "forecast_hours_minimum": 24,
        "locks": {"single_machine_lock": "/mnt/data/wrf_gpu2/operational/m7/cycle.lock"},
        "retention_policy": {"netcdf_days": 14, "zarr_days": 90},
        "failure_states": ["AIFS_LATE", "FAILED", "STALE_PUBLISHED"],
        "artifact_paths": ["artifacts/m7/monitoring/ops_status_schema.json"],
    }

    assert OperationalScheduler.validate_dict(payload)["cycle_hour_utc"] == 18


def test_operational_scheduler_rejects_string_cycle_hour():
    payload = {
        "scheduler_id": "m7-v0-daily-18z",
        "status": "DRAFT",
        "cycle_hour_utc": "18",
        "timezone": "UTC",
        "aifs_poll_start_utc": "01:25",
        "aifs_poll_timeout_utc": "05:25",
        "forecast_hours_minimum": 24,
        "locks": {},
        "retention_policy": {},
        "failure_states": [],
        "artifact_paths": [],
    }

    with pytest.raises(TypeError, match="cycle_hour_utc"):
        OperationalScheduler.validate_dict(payload)
