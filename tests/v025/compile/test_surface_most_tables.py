"""Pin the GPU reference constants used by CPU and GPU AOT lowering."""
import hashlib
from importlib.resources import files
import io

import numpy as np

from gpuwrf.physics import surface_layer


def test_surface_most_asset_and_runtime_bytes_match_gpu_reference():
    raw = files("gpuwrf.physics").joinpath("surface_most_tables_v1.npz").read_bytes()
    assert len(raw) == 33118
    assert hashlib.sha256(raw).hexdigest() == (
        "5be823f1befe1c5d1d4fb2c19275577174b423a710a1dc434b78b58ded2e2570"
    )
    expected = {
        "_PSIM_STAB_TABLE": "1d39d1cc5d905f4612ba86d5d0d5912943d012ec6e7104b900505e91b8b3c65a",
        "_PSIH_STAB_TABLE": "ea1b7db9f1df26348957b096a8724bbf346d4137ace37f0b3405be826cfef316",
        "_PSIM_UNSTAB_TABLE": "c5b91c6313edc2f8583e8982a02810691618b6a3d3439f99262f1a828a892d61",
        "_PSIH_UNSTAB_TABLE": "a250bf0b8b62c1335ba2cf97c89597d390e8fff54aea12aaf94ef9d9067c4308",
    }
    with np.load(io.BytesIO(raw), allow_pickle=False) as tables:
        assert set(tables.files) == set(expected)
        for name, digest in expected.items():
            array = tables[name]
            assert array.shape == (surface_layer.SFCLAYREV_TABLE_N + 1,)
            assert array.dtype == np.float64
            assert np.isfinite(array).all()
            assert hashlib.sha256(array.tobytes()).hexdigest() == digest
            assert np.asarray(getattr(surface_layer, name)).tobytes() == array.tobytes()
