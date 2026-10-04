"""WRF declarations constrain working types independently of output storage.

Reference: pristine module_mp_thompson.F:1580-1620 and :403 (cse).
These are type-contract checks, not independent physics fidelity evidence.
"""
import jax
import jax.numpy as jnp
import pytest

from gpuwrf.physics import thompson_column as tc


@pytest.mark.parametrize("function,names,real,double", [
    ("_rain_distribution", ("rr", "nr", "lamr", "ilamr", "mvd_r", "n0_r", "active"),
     ("rr", "nr", "mvd_r"), ("lamr", "ilamr", "n0_r")),
    ("_cloud_distribution", ("rc", "lamc", "xDc", "mvd_c", "active"),
     ("rc", "xDc", "mvd_c"), ("lamc",)),
    ("_ice_distribution", ("ri", "ni", "lami", "ilami", "xDi", "xmi", "active"),
     ("ri", "ni", "xDi", "xmi"), ("lami", "ilami")),
    ("_graupel_distribution", ("rg", "ng", "lamg", "ilamg", "n0_g", "active"),
     ("rg", "ng"), ("lamg", "ilamg", "n0_g")),
])
def test_distribution_declarations(monkeypatch, function, names, real, double):
    monkeypatch.setenv("GPUWRF_THOMPSON_NATIVE_REAL", "1")
    q = jnp.full((1, 44), 1.e-4, jnp.float32)
    number = jnp.full_like(q, 1.e5)
    rho = jnp.ones_like(q)
    args = (q, rho) if function == "_cloud_distribution" else (q, number, rho)
    outputs = jax.eval_shape(getattr(tc, function), *args)
    types = {name: out.dtype for name, out in zip(names, outputs)}
    assert all(types[name] == jnp.float32 for name in real), types
    assert all(types[name] == jnp.float64 for name in double), types


def test_real_snow_order_from_table(monkeypatch):
    monkeypatch.setenv("GPUWRF_THOMPSON_NATIVE_REAL", "1")
    moment = jnp.full((1, 44), 1.e-4, jnp.float32)
    tempc = jnp.full_like(moment, -10.)
    result = jax.eval_shape(lambda m, t: tc._snow_moment(tc.THOMPSON_TABLES.cse[12], m, t,
                                                      tc.THOMPSON_TABLES), moment, tempc)
    assert result.dtype == jnp.float32
