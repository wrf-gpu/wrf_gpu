"""Frozen pristine-WRF KF gate for the shared REAL-table path."""
import importlib.util
from pathlib import Path

import pytest


def _oracle_module():
    path = Path(__file__).resolve().parents[2] / 'test_kf_cumulus_oracle.py'
    spec = importlib.util.spec_from_file_location('b_phys_kf_oracle', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resident_tables_against_frozen_wrf_oracle(monkeypatch, tmp_path, cpu_pallas_interpret):
    monkeypatch.setenv('GPUWRF_KF_RESIDENT_TABLES', '1')
    monkeypatch.setenv('GPUWRF_WRITE_PROOFS', '1')
    oracle = _oracle_module()
    oracle.JAX_PROOF = str(tmp_path / 'kf_wrf_parity.json')
    # Uses four independent Fortran columns and the original frozen gates:
    # tendencies0.2%, precipitation0.3%, categorical cloud levels unchanged.
    oracle.test_jax_vs_oracle()


def test_wrf_real_table_storage_is_immutable():
    import numpy as np
    from gpuwrf.physics import cumulus_kf_tables as tables

    for table in (tables.TTAB_R4, tables.QSTAB_R4, tables.THE0K_R4):
        assert table.dtype == np.float32
        assert not table.flags.writeable
