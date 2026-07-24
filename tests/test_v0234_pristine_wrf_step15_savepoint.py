from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-17-v0234-pristine-wrf-step15-savepoint"
WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"


def test_observer_scope_is_literal_and_inert_by_default() -> None:
    source = (SPRINT / "wrf_patch/module_v0234_step15_savepoint.F").read_text()
    assert "enabled = .FALSE." in source
    assert "grid_id /= 3 .OR. itimestep /= 15" in source
    assert "DO j = 85, 90" in source
    assert "field(4,k,j)" in source
    assert "python_x 3" in source
    assert "python_y 84 85 86 87 88 89" in source
    assert "execute_command_line" not in source.lower()


def test_patch_binds_only_makefile_driver_and_new_module() -> None:
    patch = (SPRINT / "wrf_patch/step15-observer.patch").read_text()
    headers = [line for line in patch.splitlines() if line.startswith("diff --git ")]
    assert headers == [
        "diff --git a/phys/Makefile b/phys/Makefile",
        "diff --git a/phys/module_microphysics_driver.F b/phys/module_microphysics_driver.F",
    ]
    assert "CALL mp_gt_driver" in patch
    assert patch.index("v0234_capture_mp8('in'") < patch.index("CALL mp_gt_driver")
    assert patch.index("v0234_capture_mp8('out'") > patch.index("CALL mp_gt_driver")


def test_representation_budgets_are_predeclared_and_not_tolerance_fit() -> None:
    budget = json.loads((SPRINT / "representation-budgets.json").read_text())
    assert budget["declared_before_observation"] is True
    assert budget["input_budget"]["rule"].startswith("max(8 binary32 ULP")
    assert budget["output_budget"]["rule"].startswith("max(64 binary32 ULP")
    assert budget["predicate_budget"].startswith("exact boolean equality")
    assert budget["observer_identity"].startswith("all NetCDF variables")


def test_comparator_discards_payload_before_science_on_identity_red() -> None:
    source = (SPRINT / "compare-step15-savepoint.py").read_text()
    identity_gate = source.index('if not identity["pass"]:')
    first_payload_read = source.index('wrf_in = {name: load_payload')
    assert identity_gate < first_payload_read
    blocked_region = source[identity_gate:first_payload_read]
    assert "STEP15_SAVEPOINT_IDENTITY_BLOCKED" in blocked_region
    assert '"scientific_payload_read": False' in source[:identity_gate]
    assert 'path.name: sha256(path) for path in savepoint_files' not in source[:identity_gate]


def test_comparator_helpers_import_without_reading_external_payload() -> None:
    path = SPRINT / "compare-step15-savepoint.py"
    spec = importlib.util.spec_from_file_location("v0234_step15_compare", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.np.asarray([1.0, 0.0], dtype=module.np.float64)
    actual = reference.astype(module.np.float32).astype(module.np.float64)
    result = module.field_compare(actual, reference, 8, 1.0e-12)
    assert result["pass"] is True


def test_wrf_authority_commit_contains_bound_thompson_driver() -> None:
    assert subprocess.check_output(["git", "-C", str(WRF), "rev-parse", WRF_COMMIT], text=True).strip() == WRF_COMMIT
    body = subprocess.check_output(
        ["git", "-C", str(WRF), "show", f"{WRF_COMMIT}:phys/module_microphysics_driver.F"],
        text=True,
    )
    assert "CASE (THOMPSON)" in body
    assert "CALL mp_gt_driver" in body
    assert "ITIMESTEP=itimestep" in body
