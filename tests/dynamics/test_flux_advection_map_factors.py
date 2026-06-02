"""P0-6 CPU oracle for WRF map-factor terms in flux-form advection."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_oracle_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "proofs" / "p0_6" / "map_factor_advection_oracle.py"
    spec = importlib.util.spec_from_file_location("p0_6_map_factor_advection_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flux_advection_map_factor_oracle_passes():
    """JAX operators match independent NumPy WRF formula transcriptions."""

    report = _load_oracle_module().build_report(write_json=False)
    assert report["status"] == "PASS", report["results"]
