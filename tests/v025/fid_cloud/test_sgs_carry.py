"""MYNN clouds reach the next WRF radiation call across the RK handoff.

The input is a frozen real WN3 cloud column. This is a plumbing gate with
exact comparisons; coupled CPU-WRF comparisons remain the fidelity gate.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.physics import mynn_pbl as mynn
from gpuwrf.runtime import operational_mode as op


FIXTURES = Path(__file__).parent / 'fixtures'
SGS = ('qsq', 'qc_bl', 'qi_bl', 'cldfra_bl')


def _cloud_state():
    path = FIXTURES / 'wn3_cloud_column.npz'
    meta = json.loads((FIXTURES / 'wn3_cloud_column.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['fixture_sha256']
    with np.load(path) as f:
        a = {name: jnp.asarray(f[name]) for name in f.files}
    shape = a['T'].shape
    grid = SimpleNamespace(nz=shape[0], ny=shape[1], nx=shape[2])
    fields = {name: jnp.zeros(s, DEFAULT_DTYPES.dtype_for(name))
              for name, s in _state_field_shapes(grid).items()}
    fields.update(u=a['U'], v=a['V'], w=a['W'], theta=a['T'] + 300.,
                  p_total=a['P'] + a['PB'], p_perturbation=a['P'],
                  ph_total=a['PH'] + a['PHB'], ph_perturbation=a['PH'],
                  qv=a['QVAPOR'], qc=a['QCLOUD'], qi=a['QICE'],
                  qr=a['QRAIN'], qs=a['QSNOW'], qg=a['QGRAUP'], qke=a['QKE'],
                  t_skin=a['TSK'], ustar=a['UST'],
                  xland=jnp.ones(shape[1:]), roughness_m=jnp.full(shape[1:], .1),
                  rhosfc=jnp.ones(shape[1:]), pblh=a['PBLH'])
    state = State(**fields)
    # Specific source fluxes are not the numerical gate here; using WRF's
    # measured surface fluxes keeps the actual MYNN call physically active.
    column = pc._mynn_column_from_state(state, None)
    rho0 = pc._from_columns(column.rho)[0]
    return state.replace(rhosfc=rho0,
                         theta_flux=a['HFX'] / (rho0 * mynn.CP_MYNN),
                         qv_flux=a['QFX'] / rho0)


def test_two_step_mynn_rk_radiation_cloud_chain(monkeypatch):
    monkeypatch.setenv('GPUWRF_MYNN_SGS_CLOUD', '1')
    monkeypatch.setenv('GPUWRF_MYNN_CLOUDMIX', '1')
    monkeypatch.setenv('GPUWRF_MYNN_FP32_COLUMNS', '1')
    monkeypatch.setattr(mynn, '_MYNN_SGS_CLOUD', True)
    # The cloud carry seam is independent of plume transport; this CPU gate
    # exercises the real MYNN condensation/variance/mean producer without EDMF.
    monkeypatch.setattr(pc, '_MYNN_EDMF', False)
    mynn.step_mynn_pbl_column_with_pblh.clear_cache()
    state = _cloud_state()
    original_qsq = np.asarray(state.qsq).copy()
    try:
        for step in (1, 2):
            produced = pc.mynn_adapter(state, 18., None)
            assert float(jnp.max(produced.cldfra_bl)) > .1
            assert float(jnp.max(produced.qc_bl)) > 0.
            assert float(jnp.max(produced.qsq)) > 0.
            # RK has retained the old SGS leaves; physics must replace them.
            carried = op._apply_physics_non_dry_updates(state, state, produced)
            for name in SGS:
                np.testing.assert_array_equal(getattr(carried, name), getattr(produced, name))
            next_mynn = pc._mynn_column_from_state(carried, None)
            np.testing.assert_array_equal(next_mynn.qsq, pc._to_columns(produced.qsq))
            sw, lw, *_ = pc._rrtmg_column_inputs(
                carried, None, time_utc=datetime(2026, 2, 28, tzinfo=timezone.utc),
                lead_seconds=step * 18.)
            expected_cf = pc._to_columns(produced.cldfra_bl)
            np.testing.assert_array_equal(sw.cloud_fraction, expected_cf)
            np.testing.assert_array_equal(lw.cloud_fraction, expected_cf)
            qc = pc._to_columns(carried.qc)
            expected_qc = qc + jnp.where((qc < 1.e-6) & (expected_cf > .001),
                                        pc._to_columns(produced.qc_bl), 0.)
            np.testing.assert_array_equal(sw.qc, expected_qc)
            np.testing.assert_array_equal(lw.qc, expected_qc)
            state = carried
        assert not np.array_equal(np.asarray(state.qsq), original_qsq)
    finally:
        mynn.step_mynn_pbl_column_with_pblh.clear_cache()


@pytest.mark.parametrize('name', SGS)
def test_sgs_handoff_replaces_stale_rk_values(name):
    state = _cloud_state()
    old = jnp.full_like(getattr(state, name), .125)
    new = jnp.full_like(old, .25)
    reference = state.replace(**{name: old})
    produced = state.replace(**{name: new})
    # Distinct stale dynamics and reference values distinguish replacement
    # from both omission and adding a physics delta to stale RK data.
    dynamics = state.replace(**{name: jnp.full_like(old, .5)})
    result = op._apply_physics_non_dry_updates(dynamics, reference, produced)
    np.testing.assert_array_equal(getattr(result, name), new)
