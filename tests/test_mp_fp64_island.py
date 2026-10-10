"""v0.3.4 (o1-nlbind): fp64 islands let the fp64-parity MP adapters run under the REAL32 carry.

Through the CLI, mp=1/2/3/4/6/10/13/14/16/97 crashed at TRACE under the release defaults
(f32 State leaves + f64 metrics change their scan/cond/while carry dtypes); <= v0.3.3
this was masked because the CLI never bound mp_physics.  The island widens the State
to f64 for the call and writes every leaf back at its carry dtype; on an fp64 State
(legacy path) it is the identity.  Trace-only (eval_shape / make_jaxpr): no compile.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import pytest

from gpuwrf.coupling import scan_adapters as sa
from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state


def _states():
    grid = build_flat_grid(nx=6, ny=5, nz=8, dx_m=3000.0)
    s64 = build_neutral_state(grid)
    narrow = {n: getattr(s64, n).astype(jnp.float32) for n in s64.__slots__
              if getattr(getattr(s64, n, None), "dtype", None) == jnp.float64}
    return s64, s64.replace(_cast=False, **narrow), set(narrow)


def _canon(jaxpr) -> str:
    text = re.sub(r"name_and_src_info=[^\]\n]*", "", str(jaxpr))
    return re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", text)


@pytest.mark.parametrize("table,code", [("MP", c) for c in sa.FP64_ISLAND_MP_CODES]
                         + [("PBL", c) for c in sa.FP64_ISLAND_PBL_CODES])
def test_island_runs_on_real32_state_and_keeps_carry_dtypes(table, code):
    s64, s32, narrow = _states()
    island = getattr(sa, f"{table}_SCAN_ADAPTERS")[code]
    raw = island.__wrapped__
    with pytest.raises(TypeError, match="must have equal"):
        jax.eval_shape(lambda s: raw(s, 18.0), s32)  # the v0.3.3-masked crash
    out = jax.eval_shape(lambda s: island(s, 18.0), s32)
    assert all(getattr(out, n).dtype == jnp.float32 for n in narrow)
    # leaves the scheme does not touch keep their identity (spec-zone change detection)
    seen = {}

    def probe(s):
        res = island(s, 18.0)
        seen.update(same={n for n in narrow if getattr(res, n) is getattr(s, n)})
        return res

    jax.eval_shape(probe, s32)
    assert ({"u", "v", "w"} if table == "MP" else {"w"}) <= seen["same"]  # PBL mixes u/v
    # legacy State (fp64 theta): the island is the identity program
    assert s64.theta.dtype == jnp.float64
    assert _canon(jax.make_jaxpr(lambda s: island(s, 18.0))(s64)) == _canon(
        jax.make_jaxpr(lambda s: raw(s, 18.0))(s64))
