from __future__ import annotations

import h5py
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
        operator="calc_coef_w",
        boundary="calc_coef_w_post",
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
    path = tmp_path / "savepoint.h5"

    write_savepoint(path, savepoint)
    loaded = read_savepoint(path)

    assert loaded.metadata == savepoint.metadata
    np.testing.assert_array_equal(loaded.arrays["theta"], theta)


def test_tampered_shape_raises_clear_error(tmp_path):
    theta = np.arange(12, dtype=np.float64).reshape(3, 2, 2) + 300.0
    metadata = _metadata(theta)
    path = tmp_path / "tampered.h5"
    write_savepoint(path, Savepoint(metadata=metadata, arrays={"theta": theta}))

    with h5py.File(path, "a") as handle:
        handle["fields/theta"][0, 0, 0] += 1.0

    with pytest.raises(ValueError, match="tamper detection"):
        read_savepoint(path)
