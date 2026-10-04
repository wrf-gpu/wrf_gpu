"""All/clear broadband flux gate captured from the actual pristine WRF driver."""

import json
import os
from pathlib import Path

import jax
import numpy as np
import pytest

from gpuwrf.kernels import rad_lw_transfer
from gpuwrf.physics import rrtmg_lw as lw
from gpuwrf.validation.tier1_rrtmg import load_lw_fixture_state

FIXTURE = Path(__file__).parent / 'data/lw_driver_clear_flux.json'


@pytest.mark.parametrize('fused', [False, True])
def test_lw_all_clear_fluxes_against_pristine_driver(monkeypatch, tmp_path, fused):
    truth = json.loads(FIXTURE.read_text())
    state, _ = load_lw_fixture_state()
    interpret = os.environ.get('B_RAD_PALLAS_INTERPRET', '1') == '1'
    if not interpret:
        assert jax.devices()[0].platform == 'gpu'
    original = rad_lw_transfer.lw_band_fluxes
    monkeypatch.setattr(rad_lw_transfer, 'lw_band_fluxes',
                        lambda *a: original(*a, interpret=interpret))
    # The public fused flag also gates the retained recurrence's semantic fix.
    # Disable only dispatch to check that retained path against the same driver.
    monkeypatch.setattr(lw, '_CLEAR_SKY_COLUMN_CLOUD', True)
    monkeypatch.setattr(lw, '_FUSED_TRANSFER', fused)
    lw.solve_rrtmg_lw_column.clear_cache()
    try:
        result = lw.solve_rrtmg_lw_column(state, with_clear_sky=True)
        expected = np.asarray(truth['fluxes'])
        metrics = {}
        for i, name in enumerate(truth['fields']):
            actual = np.asarray(getattr(result, name))
            error = np.abs(actual - expected[..., i])
            allowed = truth['flux_tolerance']['atol'] + truth['flux_tolerance']['rtol'] * np.abs(expected[..., i])
            metrics[name] = dict(max_abs=float(error.max()), pass_gate=bool(np.all(error <= allowed)),
                                 finite=bool(np.isfinite(actual).all()))
        (tmp_path / 'clear_driver_gate.json').write_text(json.dumps(dict(
            backend=jax.devices()[0].platform, fused=fused, fields=metrics), indent=2)+'\n')
        assert all(v['finite'] and v['pass_gate'] for v in metrics.values()), metrics
    finally:
        lw.solve_rrtmg_lw_column.clear_cache()
