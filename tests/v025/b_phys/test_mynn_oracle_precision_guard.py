"""The historical Thompson preflight must accept WRF REAL diagnostics only."""
import ast
from pathlib import Path

import jax.numpy as jnp

from gpuwrf.contracts.precision import MYNN_DIAGNOSTIC_LEAVES, SURFACE_LAYER_CARRY_LEAVES, STATE_FIELD_ORDER
from gpuwrf.runtime.operational_mode import _enforce_operational_precision
from test_mynn_held_diagnostics import fixture


def _preflight_low(state, *, mynn=MYNN_DIAGNOSTIC_LEAVES):
    path = Path(__file__).resolve().parents[1] / 'b_thompson/oracle_reference/p0l3b7c_runner.py'
    tree = ast.parse(path.read_text())
    expressions = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'low' for t in n.targets)]
    assert len(expressions) == 1
    namespace = {'state': state, 'jnp': jnp, 'STATE_FIELD_ORDER': STATE_FIELD_ORDER,
                 'SURFACE_LAYER_CARRY_LEAVES': SURFACE_LAYER_CARRY_LEAVES,
                 'MYNN_DIAGNOSTIC_LEAVES': mynn}
    return eval(compile(ast.Expression(expressions[0]), str(path), 'eval'), namespace)


def test_thompson_fp64_guard_accepts_real_mynn_and_rejects_real_prognostic():
    _, _, state = fixture()
    wide = _enforce_operational_precision(state, force_fp64=True)
    assert _preflight_low(wide) == []
    assert set(_preflight_low(wide, mynn=())) == {'el_pbl', 'maxmf', 'maxwidth', 'ztop_plume'}
    poisoned = wide.replace(_cast=False, theta=jnp.asarray(wide.theta, jnp.float32))
    assert _preflight_low(poisoned) == ['theta']
