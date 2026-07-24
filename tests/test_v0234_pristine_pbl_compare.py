from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v0234_pristine_pbl_compare.py"
SPEC = importlib.util.spec_from_file_location("v0234_pristine_pbl_compare", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def _case(root: Path, names: tuple[str, ...], complete: bool) -> tuple[Path, Path]:
    arrays = {name: np.full((2, 3), index + 1.0, dtype="<f8") for index, name in enumerate(names)}
    archive = (root / "reference.npz").resolve()
    np.savez(archive, **arrays)
    provenance = {}
    for name in names:
        item = {"synthetic": False, "source": "test", "wrf_edges": []}
        if name in compare.reference.RESIDUALS:
            item.update({
                "residual_stage": compare.reference.RESIDUAL_STAGE[name],
                "wrf_edges": ["sealed_operand", "sealed_target"],
            })
        provenance[name] = item
    value = {
        "schema": "wrfgpu2-v0234-pristine-pbl-cpu-reference-v1",
        "archive_path": str(archive),
        "archive_sha256": compare.reference.sha256_file(archive),
        "array_order": list(names),
        "arrays": {name: compare.reference.array_record(arrays[name]) for name in names},
        "provenance": provenance,
        "reference_status": "AUTHENTIC_28_OF_28" if complete else "HOLD_5_OF_28",
        "newly_materialized_arrays": list(compare.reference.MISSING) if complete else [],
        "missing_arrays": [] if complete else list(compare.reference.MISSING),
        "comparator_dispatch_ready": complete,
    }
    value["canonical_payload_sha256"] = compare.reference.canonical_without_self(value)
    manifest = (root / "manifest.json").resolve()
    manifest.write_text(json.dumps(value), encoding="utf-8")
    return manifest, archive


@pytest.fixture(autouse=True)
def _small_shapes(monkeypatch):
    monkeypatch.setattr(
        compare.reference,
        "EXPECTED_SHAPES",
        {name: (2, 3) for name in compare.reference.REQUIRED},
    )


def test_import_is_backend_dark_and_comparator_is_exactly_hash_bound() -> None:
    assert not any(name == "jax" or name.startswith(("jax.", "gpuwrf.")) for name in sys.modules)
    assert compare.comparator_authority()["sha256"] == compare.COMPARATOR_SHA256


def test_partial_reference_is_hold_and_never_dispatches(monkeypatch, tmp_path: Path) -> None:
    manifest, archive = _case(tmp_path, compare.reference.AVAILABLE, False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("subprocess dispatch forbidden")

    monkeypatch.setattr(compare.subprocess, "run", forbidden)
    result = compare.admission(manifest, archive, partial_expected=True)
    assert result["verdict"] == "REFERENCE_BUNDLE_REQUIRED_HOLD_5_OF_28"
    assert result["dispatch_ready"] is False


def test_complete_reference_builds_exact_legacy_schema_adapters(monkeypatch, tmp_path: Path) -> None:
    manifest, archive = _case(tmp_path, compare.reference.REQUIRED, True)
    monkeypatch.setattr(compare.reference, "dump_authority", lambda deep: {"deep": deep})
    run, adapter = compare.build_adapters(manifest, archive)
    assert run["schema"] == "wrfgpu2-mynn-sp2-run-proof-v1"
    assert run["verdict"] == "WRF_DISCRIMINATOR_CAPTURED_COMPARISON_PENDING"
    assert adapter["schema"] == "wrfgpu2-mynn-sp2-gpu-reference-v1"
    assert adapter["bundle_sha256"] == compare.reference.sha256_file(archive)
    assert set(adapter["arrays"]) == set(compare.reference.REQUIRED)


def test_reference_or_comparator_identity_drift_refuses(monkeypatch, tmp_path: Path) -> None:
    manifest, archive = _case(tmp_path, compare.reference.REQUIRED, True)
    monkeypatch.setattr(compare.reference, "dump_authority", lambda deep: {"deep": deep})
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["comparator_dispatch_ready"] = False
    value["canonical_payload_sha256"] = compare.reference.canonical_without_self(value)
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(compare.reference.ReferenceError, match="COMPLETE_STATUS_DRIFT"):
        compare.build_adapters(manifest, archive)


def test_wrapper_contains_no_scientific_algebra_or_runtime_launcher() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("nvidia-smi", "mpirun", "wrf.exe", "reconstruct_system", "solve_tridiagonal"):
        assert forbidden not in source
