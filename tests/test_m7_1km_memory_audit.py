"""Static tests for the M7 1 km memory audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "scripts" / "m7_1km_memory_audit.py"

spec = importlib.util.spec_from_file_location("m7_1km_memory_audit", AUDIT_PATH)
assert spec is not None
m7_1km_memory_audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m7_1km_memory_audit)


def _fake_grid_shape() -> dict:
    return {
        "derived_full_1km": {
            "mass_shape": [44, 198, 477],
            "source_3km_wrfout_path": "/tmp/fake/wrfout_d02",
        }
    }


def test_full_1km_shape_derivation_matches_sprint_objective() -> None:
    shapes = m7_1km_memory_audit.staggered_shapes(nz=44, ny=198, nx=477)

    assert shapes["mass"] == [44, 198, 477]
    assert shapes["u"] == [44, 198, 478]
    assert shapes["v"] == [44, 199, 477]
    assert shapes["w"] == [45, 198, 477]
    assert shapes["boundary_mass"] == [1, 4, 44, 478]


def test_static_model_tracks_state_contract_field_count() -> None:
    payload = m7_1km_memory_audit.build_static_memory_model(_fake_grid_shape(), total_vram_bytes=32 * 1024**3)

    assert payload["field_count"] == len(m7_1km_memory_audit.State.__slots__)
    assert payload["field_count"] == len(m7_1km_memory_audit.STATE_FIELD_ORDER)
    assert payload["sanity_total_le_device_vram"] is True
    assert payload["total_state_bytes"] == payload["fields"][-1]["running_total_bytes"]
    assert {field["field"] for field in payload["fields"]} == set(m7_1km_memory_audit.State.__slots__)


def test_static_model_uses_precision_registry_for_known_fields() -> None:
    payload = m7_1km_memory_audit.build_static_memory_model(_fake_grid_shape(), total_vram_bytes=32 * 1024**3)
    fields = {field["field"]: field for field in payload["fields"]}

    assert fields["u"]["dtype"] == "float32"
    assert fields["theta"]["dtype"] == "float32"
    assert fields["p"]["dtype"] == "float64"
    assert fields["mu"]["dtype"] == "float64"
