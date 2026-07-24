from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v0234_pristine_pbl_reference.py"
SPEC = importlib.util.spec_from_file_location("v0234_pristine_pbl_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def _write_case(root: Path, names: tuple[str, ...], *, complete: bool) -> tuple[Path, Path]:
    arrays = {name: np.full((2, 3), index + 0.25, dtype="<f8") for index, name in enumerate(names)}
    archive = root / "reference.npz"
    np.savez(archive, **arrays)
    provenance = {}
    for name in names:
        if name in reference.RESIDUALS:
            provenance[name] = {
                "synthetic": False,
                "source": "source_ordered_authentic_wrf_substitution",
                "residual_stage": reference.RESIDUAL_STAGE[name],
                "wrf_edges": ["sealed_substitution_operand", "sealed_target"],
            }
        else:
            provenance[name] = {
                "synthetic": False,
                "source": "authenticated_gpu_capture_and_frozen_formula",
                "wrf_edges": [],
            }
    value = {
        "schema": "wrfgpu2-v0234-pristine-pbl-cpu-reference-v1",
        "archive_path": str(archive.resolve()),
        "archive_sha256": reference.sha256_file(archive),
        "array_order": list(names),
        "arrays": {name: reference.array_record(arrays[name]) for name in names},
        "provenance": provenance,
        "reference_status": "AUTHENTIC_28_OF_28" if complete else "HOLD_5_OF_28",
        "newly_materialized_arrays": list(reference.MISSING) if complete else [],
        "missing_arrays": [] if complete else list(reference.MISSING),
        "comparator_dispatch_ready": complete,
    }
    value["canonical_payload_sha256"] = reference.canonical_without_self(value)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    return manifest.resolve(), archive.resolve()


@pytest.fixture(autouse=True)
def _small_shapes(monkeypatch):
    monkeypatch.setattr(reference, "EXPECTED_SHAPES", {name: (2, 3) for name in reference.REQUIRED})


def test_import_is_backend_dark_and_inventory_is_exact() -> None:
    assert not any(name == "jax" or name.startswith(("jax.", "gpuwrf.")) for name in sys.modules)
    assert len(reference.REQUIRED) == 28
    assert len(reference.AVAILABLE) == 5
    assert len(reference.MISSING) == 23
    assert set(reference.AVAILABLE).isdisjoint(reference.MISSING)
    assert set(reference.REQUIRED) == set(reference.AVAILABLE) | set(reference.MISSING)


def test_partial_5_of_28_is_honest_hold_and_not_complete(tmp_path: Path) -> None:
    manifest, archive = _write_case(tmp_path, reference.AVAILABLE, complete=False)
    result = reference.validate_reference(manifest, archive, allow_partial=True)
    assert result["reference_status"] == "HOLD_5_OF_28"
    with pytest.raises(reference.ReferenceError, match="REFERENCE_INVENTORY"):
        reference.validate_reference(manifest, archive)


def test_complete_28_of_28_independent_validation(tmp_path: Path) -> None:
    manifest, archive = _write_case(tmp_path, reference.REQUIRED, complete=True)
    result = reference.validate_reference(manifest, archive)
    assert result["array_count"] == 28
    assert result["reference_status"] == "AUTHENTIC_28_OF_28"
    assert result["independent_validation"] is True


def test_canonical_sorted_json_preserves_explicit_array_order(tmp_path: Path) -> None:
    manifest, archive = _write_case(tmp_path, reference.REQUIRED, complete=True)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    result = reference.validate_reference(manifest, archive)
    assert result["array_order"] == list(reference.REQUIRED)


@pytest.mark.parametrize("mutation", ["extra", "order", "payload", "shape", "dtype", "nonfinite"])
def test_archive_identity_shape_dtype_order_and_payload_drift_refuse(tmp_path: Path, mutation: str) -> None:
    manifest, archive = _write_case(tmp_path, reference.REQUIRED, complete=True)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    with np.load(archive, allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    if mutation == "extra":
        arrays["extra"] = np.zeros((2, 3), dtype="<f8")
    elif mutation == "order":
        arrays = {name: arrays[name] for name in reversed(tuple(arrays))}
    elif mutation == "payload":
        arrays[reference.REQUIRED[0]][0, 0] += 1.0
    elif mutation == "shape":
        arrays[reference.REQUIRED[0]] = np.zeros((3, 2), dtype="<f8")
    elif mutation == "dtype":
        arrays[reference.REQUIRED[0]] = arrays[reference.REQUIRED[0]].astype("<f4")
    else:
        arrays[reference.REQUIRED[0]][0, 0] = np.nan
    np.savez(archive, **arrays)
    value["archive_sha256"] = reference.sha256_file(archive)
    if mutation in {"shape", "dtype", "nonfinite"}:
        value["arrays"][reference.REQUIRED[0]] = reference.array_record(arrays[reference.REQUIRED[0]]) if mutation != "nonfinite" else value["arrays"][reference.REQUIRED[0]]
    value["canonical_payload_sha256"] = reference.canonical_without_self(value)
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(reference.ReferenceError):
        reference.validate_reference(manifest, archive)


def test_wrf_edge_to_non_residual_refuses(tmp_path: Path) -> None:
    manifest, archive = _write_case(tmp_path, reference.REQUIRED, complete=True)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["provenance"]["surface_hfx"]["wrf_edges"] = ["forbidden"]
    value["canonical_payload_sha256"] = reference.canonical_without_self(value)
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(reference.ReferenceError, match="WRF_EDGE_TO_NONRESIDUAL"):
        reference.validate_reference(manifest, archive)


@pytest.mark.parametrize("field", reference.RESIDUALS)
def test_residual_source_order_drift_refuses(tmp_path: Path, field: str) -> None:
    manifest, archive = _write_case(tmp_path, reference.REQUIRED, complete=True)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["provenance"][field]["residual_stage"] = "wrong"
    value["canonical_payload_sha256"] = reference.canonical_without_self(value)
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(reference.ReferenceError, match="RESIDUAL_STAGE_DRIFT"):
        reference.validate_reference(manifest, archive)


def test_source_authority_binds_exact_formula_symbols() -> None:
    result = reference.source_authority()
    assert set(result["files"]) == set(reference.SOURCE_SHA256)
    assert set(result["symbols"]) == {
        "_mym_turbulence", "_solve_tridiagonal", "_step_mynn_pbl_impl_with_pblh",
        "_apply_mean_tendencies", "_diffusion_solve_with_mf",
    }


def test_script_contains_no_gpu_query_or_wrf_mpi_launch() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "nvidia-smi" not in source
    assert "mpirun" not in source
    assert "wrf.exe" not in source


def test_materializer_loads_only_cpu_safe_sealed_grid() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    materializer = source[source.index("def materialize("):source.index("def parser(")]
    assert 'Gen2Run(INPUT_DIR).grid("d03").as_grid_spec()' in materializer
    assert "nested_pipeline._load_domains" not in materializer
    assert "State.zeros" in materializer  # retained rationale for the fail-closed boundary


def test_surface_and_mixing_residuals_emit_preregistered_order() -> None:
    target = {"rublten": np.asarray([1.0]), "rvblten": np.asarray([2.0])}
    surface = {"rublten": np.asarray([3.0]), "rvblten": np.asarray([5.0])}
    mixing = {"rublten": np.asarray([7.0]), "rvblten": np.asarray([11.0])}
    result = reference.ordered_surface_mixing_residuals(surface, mixing, target)
    assert tuple(result) == reference.MISSING[17:21]
    assert [value.item() for value in result.values()] == [2.0, 3.0, 6.0, 9.0]
