from __future__ import annotations

import zipfile
from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest

from gpuwrf.validation.savepoint_io import read_savepoint, write_savepoint
from gpuwrf.validation.savepoint_schema import Savepoint, SavepointMetadata, VariableMetadata


def _metadata(array: np.ndarray) -> SavepointMetadata:
    return SavepointMetadata(
        run_id="test-run",
        wrf_version="WRF-test",
        wrf_commit="abc123",
        namelist_hash="def456",
        source_path="synthetic://test",
        domain_index=2,
        tier="column",
        operator="coefficient_construction",
        boundary="coefficient_construction",
        dt_seconds=6.0,
        rk_stage_index=1,
        acoustic_substep_index=1,
        map_factors={"msftx": "unity", "msfty": "unity"},
        vertical_grid={"kind": "hybrid_eta", "nz": array.shape[0]},
        variables={
            "theta": VariableMetadata(
                name="theta",
                dtype=str(array.dtype),
                shape=array.shape,
                stagger="mass",
                units="K",
                provenance="unit-test",
            )
        },
    )


def test_savepoint_roundtrip_preserves_metadata_and_arrays(tmp_path):
    theta = np.arange(12, dtype=np.float64).reshape(3, 2, 2) + 300.0
    savepoint = Savepoint(metadata=_metadata(theta), arrays={"theta": theta})
    path = tmp_path / "savepoint.npz"

    write_savepoint(path, savepoint)
    loaded = read_savepoint(path)

    assert loaded.metadata == savepoint.metadata
    np.testing.assert_array_equal(loaded.arrays["theta"], theta)


def test_tampered_shape_raises_clear_error(tmp_path):
    theta = np.arange(12, dtype=np.float64).reshape(3, 2, 2) + 300.0
    metadata = _metadata(theta)
    bad_metadata = replace(
        metadata,
        variables={
            "theta": replace(metadata.variables["theta"], shape=(99, 2, 2)),
        },
    )
    path = tmp_path / "tampered.npz"
    write_savepoint(path, Savepoint(metadata=metadata, arrays={"theta": theta}))

    with zipfile.ZipFile(path, "a") as archive:
        buffer = BytesIO()
        np.save(buffer, np.frombuffer(str(bad_metadata.to_json()).encode(), dtype=np.uint8))
        archive.writestr(
            "__metadata_json__.npy",
            buffer.getvalue(),
        )

    with pytest.raises(ValueError, match="invalid savepoint metadata JSON|shape mismatch"):
        read_savepoint(path)
