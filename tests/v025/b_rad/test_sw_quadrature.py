"""Full independent WRF SW heating/flux gate for fused adding-method sweep."""

import os

import jax

from gpuwrf.kernels import rad_sw_quadrature
from gpuwrf.kernels.rad_sw_quadrature import vertical_quadrature
from gpuwrf.physics import rrtmg_sw as sw
from gpuwrf.validation.tier1_rrtmg import run_tier1_sw
from gpuwrf.validation.rrtmg_intermediate_oracles import run_intermediate_validation


def test_fused_sw_full_wrf_oracle(monkeypatch, tmp_path):
    interpret = os.environ.get('B_RAD_PALLAS_INTERPRET', '1') == '1'
    if not interpret:
        assert jax.devices()[0].platform == 'gpu'
    monkeypatch.setattr(sw, '_FUSED_QUADRATURE', True)
    monkeypatch.setattr(rad_sw_quadrature, 'vertical_quadrature',
        lambda *a: vertical_quadrature(*a, interpret=interpret))
    sw.solve_rrtmg_sw_column.clear_cache()
    try:
        record = run_tier1_sw(tmp_path / 'sw_fused.json')
        assert record['pass'], record
        if interpret:
            # The intermediate helper deliberately selects CPU internally.
            intermediate = run_intermediate_validation(tmp_path / 'intermediate.json', tmp_path / 'bands.json')
            assert intermediate['pass'], intermediate
    finally:
        sw.solve_rrtmg_sw_column.clear_cache()
