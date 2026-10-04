"""Control/branch regression. CPU-WRF evidence is wn3_lower_boundary_gate.py."""
from dataclasses import dataclass
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.io.lower_boundary import (
    LowerBoundary, apply_lower_boundary, load_lower_boundary, lower_boundary_history,
)
from gpuwrf.io.scheme_catalog import SupportStatus, classify_control


@dataclass(frozen=True)
class Surface:
    t_skin: object
    xland: object

    def replace(self, **kw):
        from dataclasses import replace
        return replace(self, **kw)


@dataclass(frozen=True)
class Carry:
    state: Surface
    noahmp_land: object = None

    def replace(self, **kw):
        from dataclasses import replace
        return replace(self, **kw)


def boundary():
    sst = jnp.asarray([[[280., 290., 250., 350.]], [[285., 295., 249., 351.]]], jnp.float32)
    return LowerBoundary(jnp.asarray([0, 400], jnp.int32), sst,
                         jnp.zeros_like(sst), jnp.zeros_like(sst), jnp.full_like(sst, .08))


def test_aux4_before_solve_phase_and_resumed_step():
    b = boundary()
    # One-based step401 begins at21600s, while step400/output21600s uses record0.
    for own_step, expected in ((0, 280.), (400, 280.), (401, 285.), (800, 285.)):
        assert float(lower_boundary_history(b, own_step)["SST"][0, 0]) == expected


def test_wrf_open_interval_and_land_mask_without_clamping():
    carry = Carry(Surface(jnp.full((1, 4), 299.), jnp.asarray([[2., 1., 2., 2.]])))
    for step, expected in ((400, 280.), (401, 285.)):
        actual, fields = apply_lower_boundary(carry, boundary(), jnp.asarray(step))
        np.testing.assert_array_equal(actual.state.t_skin, [[expected, 299., 299., 299.]])
        # Input fields still carry 250/350 or249/351: a WRF branch, no repair.
        np.testing.assert_array_equal(fields["SST"], boundary().sst[int(step > 400)])


def test_off_is_identity_and_never_opens_aux4(tmp_path):
    marker = object()
    assert apply_lower_boundary(marker, None, 1) == (marker, None)
    assert load_lower_boundary(tmp_path / "absent", {"physics": {"sst_update": 0}}, "d03",
                               run_start=datetime(2026, 2, 28), dt_s=6, shape=(2, 3)) is None
    assert lower_boundary_history(None, 100) == {}


def test_supported_namelist_values_and_invalid_value_fail_closed():
    for value in (0, 1):
        assert classify_control("sst_update", value).status == SupportStatus.IMPLEMENTED
    assert classify_control("sst_update", 2).status == SupportStatus.RECOGNIZED_FAIL_CLOSED
