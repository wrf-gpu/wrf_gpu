"""On-the-fly NSSL init tables vs the pristine nssl_2mom_init dump (size, sum and every 997th element):
tabqvs/tabqis/dtabqvs/dtabqis (1e6 entries), gamxinflu (gaminterp nodes), ciacr/qiacr/ziacrratio."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_io as oio  # noqa: E402

from gpuwrf.physics.nssl2mom import mathfun, satfun  # noqa: E402
from gpuwrf.physics.nssl2mom.constants import get_constants  # noqa: E402
from gpuwrf.physics.nssl2mom.indices import FP32, FP64  # noqa: E402

PREC = {"fp32": FP32, "fp64": FP64}


def large(mode):
    return oio.load_case(mode, 1)["module_vars_large"]


def rel(a, b, atol=1e-30):
    """max relative error ignoring |diff| <= atol (XLA:CPU flushes fp32 subnormals, E117)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.where(np.abs(a - b) <= atol, 0, np.abs(a - b) / np.maximum(np.abs(b), 1e-300))))


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
@pytest.mark.parametrize("name", ["tabqvs", "tabqis", "dtabqvs", "dtabqis"])
def test_sat_tables(mode, name):
    C, R = get_constants(mode), PREC[mode].R
    ref = large(mode)[name]
    l = jnp.arange(1, C.nqsat + 1, dtype=jnp.int32)
    vals = np.asarray(getattr(satfun, name)(l, C, R), np.float64)
    tol = 1e-13 if mode == "fp64" else 3e-7  # fp32: XLA vs glibc expf/logf ulps
    assert ref["size"] == vals.size
    assert rel(vals[::997], ref["stride997"]) <= tol * (1 if mode == "fp64" else 8)
    assert abs(vals.sum() - ref["sum"]) <= tol * abs(ref["sum"])


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
def test_gamxinflu_nodes(mode):
    """gaminterp recomputes the gamxinflu nodes; check them against the init table samples."""
    C, R = get_constants(mode), PREC[mode].R
    ref = large(mode)["gamxinflu"]
    ni, nj = C.nqiacrratio + 1, C.nqiacralpha - C.ialpstart + 1
    idx = np.arange(0, ref["size"], 997)
    i = idx % ni
    j = (idx // ni) % nj + C.ialpstart
    lu = (idx // (ni * nj)) % 12 + 1
    ilh = idx // (ni * nj * 12) + 1
    got = np.zeros(idx.size)
    for lu_v in range(1, 13):
        for ilh_v in (1, 2):
            m = (lu == lu_v) & (ilh == ilh_v)
            if not m.any():
                continue
            if lu_v == 12 and ilh_v == 2:
                got[m] = 0.0  # never set by init (static zero)
                continue
            got[m] = np.asarray(mathfun._gamxinflu_node(jnp.asarray(i[m]), jnp.asarray(j[m]), lu_v, ilh_v, R,
                                                         C.bx[7], C.bx[8]))
    tol = 1e-12 if mode == "fp64" else 2e-6
    refv = np.asarray(ref["stride997"])
    bad = np.abs(got - refv) > tol * np.abs(refv) + 1e-300
    assert not bad.any(), (int(bad.sum()), idx[bad][:5], got[bad][:5], refv[bad][:5])


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
@pytest.mark.parametrize("name", ["ciacrratio", "qiacrratio", "ziacrratio"])
def test_iacr_tables(mode, name):
    C, R = get_constants(mode), PREC[mode].R
    ref = large(mode)[name]
    ni = C.nqiacrratio + 1
    idx = np.arange(0, ref["size"], 997)
    got = np.asarray(mathfun.iacr_table(name, idx % ni, idx // ni + C.ialpstart, R), np.float64)
    tol = 1e-13 if mode == "fp64" else 3e-6
    assert rel(got, ref["stride997"]) <= tol


def test_mutant_gamxinf_sensitive():
    """E39: a perturbed incomplete-gamma series tolerance must break the node check."""
    C = get_constants("fp64")
    ref = np.asarray(large("fp64")["gamxinflu"]["stride997"])
    v = float(mathfun._gamxinflu_node(jnp.asarray(997 % 401), jnp.asarray(997 // 401 - 19), 1, 1, jnp.float64, 0.6, 0.593))
    assert abs(v - ref[1]) <= 1e-12 * abs(ref[1])
    assert abs(v * (1 + 1e-9) - ref[1]) > 1e-12 * abs(ref[1])
