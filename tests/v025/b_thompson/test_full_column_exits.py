"""Full-column kernel exits and DOUBLE pow products (review-thfast advisory; CPU, Pallas interpret).

Exit parity runs in a child process with XLA's algebraic simplifier disabled: algsimp
reassociates divides (A/(B/C) -> A*C/B) across whatever the lax.cond boundary leaves
fusable, so default XLA:CPU differs by an ulp between exit and no-exit graphs (TH06) while
the GPU Triton lowering evaluates the jaxpr literally. Literal CPU evaluation is the
comparison that matches the GPU semantics (and keeps the mutant checks load-bearing).
"""
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import phys_thompson_full as full
from gpuwrf.physics import thompson_column as tc

C24 = dict(GPUWRF_THOMPSON_NATIVE_REAL="1", GPUWRF_THOMPSON_COLUMN_SED="1", GPUWRF_THOMPSON_SED_FP32="1",
           GPUWRF_THOMPSON_COLUMN_LAYOUT="1", GPUWRF_THOMPSON_FULL_COLUMN="1",
           GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT="1", GPUWRF_THOMPSON_SED_PREP_FUSED="0")
LITERAL_XLA = "--xla_disable_hlo_passes=algsimp"


def _columns():
    x = np.load(Path(__file__).with_name("fixtures") / "exit_columns.npz")
    return tc.ThompsonColumnState(**{k: jnp.asarray(x[k]) for k in tc.ThompsonColumnState.__slots__})


def _run(out_dir, exits, mutant=""):
    out = Path(out_dir) / f"leaves_{exits}_{mutant or 'none'}.npz"
    env = {**os.environ, **C24, "JAX_PLATFORMS": "cpu",
           "GPUWRF_THOMPSON_FULL_COLUMN_WARM_EXIT": exits, "GPUWRF_THOMPSON_FULL_COLUMN_SED_EXIT": exits,
           "XLA_FLAGS": f"{os.environ.get('XLA_FLAGS', '')} {LITERAL_XLA}".strip(),
           "PYTHONPATH": os.pathsep.join([str(Path(full.__file__).parents[2]), os.environ.get("PYTHONPATH", "")])}
    subprocess.run([sys.executable, __file__, str(out), mutant], env=env, check=True, timeout=1500)
    with np.load(out) as leaves:
        return [leaves[f"a{i}"] for i in range(len(leaves.files))]


def _diff(a, b):
    return [i for i, (x, y) in enumerate(zip(a, b)) if x.tobytes() != y.tobytes()]


@pytest.fixture(scope="module")
def no_exit_reference(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("exits"), "0")


def test_exits_are_bitwise_on_rain_cloud_dry_and_adversarial_columns(no_exit_reference, tmp_path):
    assert not _diff(no_exit_reference, _run(tmp_path, "1"))


def _warm_exit_without_cloud_term(env, tc_module, valid):
    """Mutant: the warm exit tests rain only, so cloud-only columns lose autoconversion."""
    warm_fn, apply_warm = env["_warm_rain_collection"], env["_apply_warm_rain_rates"]

    def warm(state, dt, tables=env["THOMPSON_TABLES"]):
        zero64 = jnp.zeros_like(state.qc, dtype=jnp.float64)
        return jax.lax.cond(full._any(valid & (state.qr > tc_module.R1)), lambda: warm_fn(state, dt, tables),
                            lambda: apply_warm(state, dt, zero64, zero64, zero64,
                                               jnp.zeros_like(state.qc, dtype=jnp.float32)))

    env["_warm_rain_collection"] = warm


def _sed_exit_without_snow_term(env, tc_module, valid):
    """Mutant: the sedimentation exit ignores snow, so a snow-only column stops falling."""
    sed_fn, sediment = env["_sedimentation"], env["_sediment_with_speeds"]

    def sedimentation(state, dt, vts_boost=None, cloud_sed_on=None, cloud_rho=None):
        zero = jnp.zeros_like(state.qr, dtype=jnp.float64)
        active = full._any(valid & ((state.qr > tc_module.R1) | (state.qi > tc_module.R1) | (state.qg > tc_module.R1)))
        return jax.lax.cond(active, lambda: sed_fn(state, dt, vts_boost, cloud_sed_on, cloud_rho),
                            lambda: sediment(state, dt, (zero,) * 7, cloud_sed_on, cloud_rho))

    env["_sedimentation"] = sedimentation


MUTANTS = {"warm_no_cloud": ("_inactive_warm_branches", _warm_exit_without_cloud_term),
           "sed_no_snow": ("_inactive_sedimentation", _sed_exit_without_snow_term)}


@pytest.mark.parametrize("mutant", sorted(MUTANTS))
def test_exit_predicate_terms_are_load_bearing(no_exit_reference, tmp_path, mutant):
    assert _diff(no_exit_reference, _run(tmp_path, "1", mutant))


@pytest.mark.parametrize("exponent", [1.0, 2.0, 3.0, 4.0, -4.0, -5.0])
def test_dpow_products_only_for_native_real_f64_integral(monkeypatch, exponent):
    x = jnp.asarray([1.0e-3, 0.37, 2.5, 4.1e3, 1.9e5], jnp.float64)
    monkeypatch.setenv("GPUWRF_THOMPSON_NATIVE_REAL", "1")
    jaxpr = str(jax.make_jaxpr(lambda v: tc._dpow(v, exponent))(x))
    assert "integer_pow" in jaxpr and " pow " not in jaxpr
    np.testing.assert_allclose(np.asarray(tc._dpow(x, exponent)), np.asarray(x) ** exponent, rtol=4e-16 * 4)
    assert " pow " in str(jax.make_jaxpr(lambda v: tc._dpow(v, exponent))(x.astype(jnp.float32)))
    monkeypatch.setenv("GPUWRF_THOMPSON_DPOW_PRODUCTS", "0")
    assert " pow " in str(jax.make_jaxpr(lambda v: tc._dpow(v, exponent))(x))
    monkeypatch.setenv("GPUWRF_THOMPSON_DPOW_PRODUCTS", "1")
    monkeypatch.setenv("GPUWRF_THOMPSON_NATIVE_REAL", "0")
    assert " pow " in str(jax.make_jaxpr(lambda v: tc._dpow(v, exponent))(x))


def test_dpow_keeps_non_integral_exponents(monkeypatch):
    monkeypatch.setenv("GPUWRF_THOMPSON_NATIVE_REAL", "1")
    x = jnp.asarray([0.5, 2.0], jnp.float64)
    assert " pow " in str(jax.make_jaxpr(lambda v: tc._dpow(v, 2.5))(x))


if __name__ == "__main__":  # child of _run: literal XLA flags are fixed at backend start
    out_path, mutant_name = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
    if mutant_name:
        target, mutant_fn = MUTANTS[mutant_name]
        setattr(full, target, mutant_fn)
    result, ppt = full.full_column(_columns(), 18.0, interpret=True)
    np.savez(out_path, **{f"a{i}": np.asarray(v) for i, v in enumerate(jax.tree.leaves((result, ppt)))})
