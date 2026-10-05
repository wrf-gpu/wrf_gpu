"""Unused schemes stay unloaded; first traced use initializes real constants."""
import os
from pathlib import Path
import subprocess
import sys


def test_first_use_initializes_bmj_and_aerosol_tables_without_tracer_leaks():
    root = Path(__file__).resolve().parents[3]
    code = r'''
import importlib.util
import sys
from pathlib import Path
import jax
import numpy as np
import gpuwrf.integration.nested_pipeline
from gpuwrf.coupling import physics_couplers as couplers, scan_adapters

assert 'gpuwrf.physics.cumulus_bmj' not in sys.modules
assert 'gpuwrf.physics.thompson_aero_column' not in sys.modules
assert 'gpuwrf.physics.thompson_aero_tables' not in sys.modules

# Exercise first use INSIDE tracing, where an unguarded lazy import would
# otherwise retain Tracers in the scheme's module-level table globals.
from types import SimpleNamespace
seed_bmj = jax.jit(lambda theta: scan_adapters.initial_bmj_carry(SimpleNamespace(theta=theta)))
cldefi = seed_bmj(np.zeros((4, 2, 3), np.float64))
from gpuwrf.physics import cumulus_bmj
np.testing.assert_array_equal(np.asarray(cldefi), np.full((2, 3), cumulus_bmj.AVGEFI))
assert not any(isinstance(value, jax.core.Tracer) for value in vars(cumulus_bmj).values())
assert scan_adapters.step_bmj_column is cumulus_bmj.step_bmj_column
spy = lambda shape: np.full(shape, 0.75)
scan_adapters.initial_bmj_cldefi = spy
np.testing.assert_array_equal(scan_adapters.initial_bmj_carry(SimpleNamespace(theta=np.zeros((4, 2, 3)))), np.full((2, 3), 0.75))
assert scan_adapters.initial_bmj_cldefi is spy

spec = importlib.util.spec_from_file_location('aerosol_fixture',
    Path(sys.argv[1]) / 'tests/test_v016_thompson_aero_threading.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
state = fixture._tiny_state(ny=2, nx=2)
assert 'gpuwrf.physics.thompson_aero_column' not in sys.modules
seed_aerosol = jax.jit(couplers.thompson_aero_coldstart_init)
seeded = seed_aerosol(state)
assert np.isfinite(np.asarray(seeded.nwfa)).all() and np.asarray(seeded.nwfa).max() > 0
assert np.isfinite(np.asarray(seeded.nifa)).all() and np.asarray(seeded.nifa).max() > 0
from gpuwrf.physics import thompson_aero_tables, thompson_aero_column
assert couplers.step_thompson_aero_column_with_precip is thompson_aero_column.step_thompson_aero_column_with_precip
assert not any(isinstance(value, jax.core.Tracer) for value in thompson_aero_tables.THOMPSON_AERO_TABLES)
again = seed_aerosol(seeded)
np.testing.assert_array_equal(np.asarray(again.nwfa), np.asarray(seeded.nwfa))
np.testing.assert_array_equal(np.asarray(again.nifa), np.asarray(seeded.nifa))
column = couplers._thompson_aero_column_from_state(seeded)
assert column.qv.shape == (2, 2, state.theta.shape[0])
for module in (couplers, scan_adapters):
    try:
        getattr(module, 'unknown_scheme_attribute')
    except AttributeError:
        pass
    else:
        raise AssertionError('unknown module attribute must fail')
print('unused imports absent; selected BMJ/mp28 first-JIT initialization and repeat exact')
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0",
               GPUWRF_FAST_DEFAULTS="0", PYTHONPATH=str(root / "src"))
    result = subprocess.run([sys.executable, "-c", code, str(root)],
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "first-JIT initialization and repeat exact" in result.stdout
