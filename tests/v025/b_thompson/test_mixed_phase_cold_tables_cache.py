"""F0 release blocker (integrate 07:31Z, FINDINGS E206): GPUWRF_THOMPSON_MIXED_PHASE_WRF must survive several programs
in one process. The WRF cold-collection tables were an lru_cache of jnp arrays built on first use; the first use sat
inside the d01 jit trace, so the cache held a tracer and the fused d02/d03 trace died with UnexpectedTracerError.
The cache now holds host NumPy fp64 only; thompson_column._mixed_phase_cold_tables builds fp64 jnp arrays per call
and refuses to run without x64 instead of truncating the WRF DOUBLE tables.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import phys_thompson_full as full
from gpuwrf.physics import thompson_column as tc
from gpuwrf.physics import thompson_tables as tt

C24 = dict(GPUWRF_THOMPSON_NATIVE_REAL="1", GPUWRF_THOMPSON_COLUMN_SED="1", GPUWRF_THOMPSON_SED_FP32="1",
           GPUWRF_THOMPSON_COLUMN_LAYOUT="1", GPUWRF_THOMPSON_FULL_COLUMN="1",
           GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT="1", GPUWRF_THOMPSON_SED_PREP_FUSED="0",
           GPUWRF_THOMPSON_IMPLICIT_SED="0", GPUWRF_THOMPSON_MIXED_PHASE_WRF="1")


@pytest.fixture
def first_use_in_a_trace(monkeypatch):
    """C24 + the flag, and an EMPTY table cache so the first load happens inside the first trace (the F0 order)."""
    for key, value in C24.items():
        monkeypatch.setenv(key, value)
    tt.load_wrf_cold_collection_tables.cache_clear()
    jax.clear_caches()
    yield
    tt.load_wrf_cold_collection_tables.cache_clear()
    jax.clear_caches()


def _state(ncol, nz=12):
    """Mixed-phase column (snow + supercooled rain + riming cloud at 263 K), so the cold tables are read."""
    shape = (ncol, nz)
    f = lambda v: jnp.full(shape, v, jnp.float32)  # noqa: E731
    T, p = f(263.0), f(7.0e4)
    qv = (0.95 * tc.saturation_mixing_ratio_liquid(p, T)).astype(jnp.float32)
    return tc.ThompsonColumnState(qv=qv, qc=f(2.0e-4), qr=f(1.0e-4), qi=f(1.0e-5), qs=f(2.0e-4), qg=f(1.0e-4),
                                  Ni=f(1.0e4), Nr=f(1.0e3), Ns=f(0.0), Ng=f(0.0), T=T, p=p,
                                  rho=tc.density_from_pressure_temperature(p, T, qv), dz=f(250.0), w=f(0.0))


def _finite(tree):
    return all(bool(np.isfinite(np.asarray(leaf)).all()) for leaf in jax.tree.leaves(tree))


def test_two_programs_in_one_process_full_column(first_use_in_a_trace):
    """Pallas full column: program 1 loads the tables inside its trace, program 2 (another shape, like the fused
    d02/d03 after d01) must not see a leaked tracer."""
    step = lambda s: full.full_column(s, 18.0, interpret=True)  # noqa: E731
    first = jax.jit(step)(_state(2))
    second = jax.jit(step)(_state(3))
    assert _finite(first) and _finite(second)


def test_two_programs_in_one_process_python_body(first_use_in_a_trace, monkeypatch):
    """Same with the Python body (FULL_COLUMN=0), which calls _mixed_phase_cold_tables directly in the trace."""
    monkeypatch.setenv("GPUWRF_THOMPSON_FULL_COLUMN", "0")
    body = lambda s: tc._thompson_source_sink_body(s, 18.0, False, sediment=False)  # noqa: E731
    first = jax.jit(body)(_state(2))
    second = jax.jit(body)(_state(3))
    assert _finite(first) and _finite(second)


def test_cache_holds_host_float64_only(first_use_in_a_trace, monkeypatch):
    """After a traced use the cache holds read-only host NumPy fp64 arrays, never a jax value."""
    monkeypatch.setenv("GPUWRF_THOMPSON_FULL_COLUMN", "0")
    jax.jit(lambda s: tc._thompson_source_sink_body(s, 18.0, False, sediment=False))(_state(2))
    host = tt.load_wrf_cold_collection_tables()
    assert all(type(t) is np.ndarray and t.dtype == np.float64 and not t.flags.writeable for t in host), \
        [(n, type(t).__name__, getattr(t, "dtype", None)) for n, t in zip(host._fields, host)]


def test_traced_tables_are_wrf_double(first_use_in_a_trace):
    """The tables reaching the source stage are fp64 inside a trace (WRF DOUBLE), with the asset values."""
    seen = {}

    def probe(x):
        tables = tc._mixed_phase_cold_tables()
        seen.update({n: t.dtype for n, t in zip(tables._fields, tables)})
        return x + tables.tcg_racg[4, 4, 4, 4] + tables.tpi_qcfz[3, 3]

    out = jax.jit(probe)(jnp.float64(0.0))
    host = tt.load_wrf_cold_collection_tables()
    assert set(seen.values()) == {jnp.dtype(jnp.float64)}, seen
    assert float(out) == float(host.tcg_racg[4, 4, 4, 4] + host.tpi_qcfz[3, 3])


def test_without_x64_the_tables_refuse_instead_of_truncating(first_use_in_a_trace):
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="jax_enable_x64"):
            tc._mixed_phase_cold_tables()
