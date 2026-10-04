"""S2: the B36 nest scalar clock (lead, dtbc) is WRF REAL; only bdy_interp1's rdt is REAL*8.

Counts f64 compute equations in the lowered value/rate helper: the REAL*8 rdt
island (one multiply plus its broadcast reciprocal) is the only allowed f64
arithmetic.  Reverting the clock to f64 raises the count (review-hsclk).
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling.boundary_apply import _scalar_record_value_rate_real4

_F64_COMPUTE = re.compile(r"= stablehlo\.(?!constant|convert)\w+.*tensor<[^>]*f64>")


def test_nest_clock_has_only_the_rdt_double_island():
    records = jnp.asarray(np.ones((2, 4, 5, 3, 7), dtype=np.float32))
    lowered = jax.jit(
        lambda r, lead: _scalar_record_value_rate_real4(r, lead, 54.0)
    ).lower(records, jnp.float64(18.0))
    f64_eqns = [line for line in lowered.as_text().splitlines() if _F64_COMPUTE.search(line)]
    assert len(f64_eqns) <= 2, f64_eqns
