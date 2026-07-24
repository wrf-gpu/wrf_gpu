"""Adversarial gates for the v0.23.4 final Ni fix: the nested in-loop
``spec_bdyupdate_ph`` base operand.

The falsified `60659a2e` wiring passed the PRE-``advance_w`` work array as the
output base of ``spec_bdyupdate_ph_tendency_inloop``, so the returned ``ph``
work array discarded every interior ``advance_w`` geopotential update.  The
interior ``ph`` froze bit-for-bit for the entire run (retained step-0 versus
step-200 carries: interior identical, only ring 0 changed), the acoustic
w<->ph implicit coupling broke, and the run detonated to non-finite Thompson
``Ni`` at the first interior scan cell (0, 1, 1) by the first 00:20 output.

These tests bind the corrected two-operand contract and turn red on ANY
substitution of the rejected single-base formula, at both the helper and the
``acoustic_substep_core`` call-site level.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling.boundary_apply import (
    BoundaryConfig,
    spec_bdyupdate_ph_tendency_inloop,
)
from gpuwrf.dynamics.core import acoustic
from scripts import v0234_1500_scientific_rca_source_oracle as oracle


CFG = BoundaryConfig(
    spec_bdy_width=5,
    spec_zone=1,
    relax_zone=4,
    update_cadence_s=18.0,
    force_geopotential=False,
    nested_frozen_wrf_boundary_bundle=True,
)


def _operands(seed: int, nzp1: int = 5, ny: int = 9, nx: int = 11):
    rng = np.random.default_rng(seed)
    ph_before = rng.normal(size=(nzp1, ny, nx))
    ph_advanced = ph_before + 0.5 + rng.normal(size=(nzp1, ny, nx))
    ph_save = 50.0 + rng.normal(size=(nzp1, ny, nx))
    ph_tend = rng.normal(size=(nzp1, ny, nx))
    mu_tend = rng.normal(size=(ny, nx))
    muts = 9000.0 + rng.normal(size=(ny, nx))
    c1f = np.linspace(0.2, 1.0, nzp1)
    c2f = np.linspace(2.0, 5.0, nzp1)
    return ph_before, ph_advanced, ph_save, ph_tend, mu_tend, muts, c1f, c2f


def _call(spec_zone, *arrays, dts=0.6):
    ph_before, ph_advanced, ph_save, ph_tend, mu_tend, muts, c1f, c2f = arrays
    return np.asarray(
        spec_bdyupdate_ph_tendency_inloop(
            jnp.asarray(ph_advanced),
            jnp.asarray(ph_before),
            jnp.asarray(ph_tend),
            jnp.asarray(ph_save),
            jnp.asarray(mu_tend),
            jnp.asarray(muts),
            jnp.asarray(c1f),
            jnp.asarray(c2f),
            dts,
            CFG,
            spec_zone=spec_zone,
        )
    )


def _ring_mask(ny: int, nx: int, zone: int) -> np.ndarray:
    mask = np.zeros((ny, nx), dtype=bool)
    for b in range(zone):
        mask[b, :] = True
        mask[ny - 1 - b, :] = True
        mask[:, b] = True
        mask[:, nx - 1 - b] = True
    return mask


def test_interior_is_the_advanced_field_and_ring_walks_from_pre_advance() -> None:
    arrays = _operands(7011)
    ph_before, ph_advanced, ph_save, ph_tend, mu_tend, muts, c1f, c2f = arrays
    dts = 0.6
    for zone in (1, 2):
        actual = _call(zone, *arrays, dts=dts)
        ny, nx = ph_before.shape[1:]
        ring = _ring_mask(ny, nx, zone)
        expected_ring = oracle.wrf_spec_ph_update(
            ph_before,
            ph_save,
            ph_tend,
            mu_tend[None, :, :],
            muts[None, :, :],
            c1f[:, None, None],
            c2f[:, None, None],
            dts,
        )
        # Ring: the WRF walk computed from the PRE-advance (loop-carried) ring.
        np.testing.assert_allclose(
            actual[:, ring], expected_ring[:, ring], rtol=3.0e-15, atol=3.0e-13
        )
        # Interior: bit-exactly the advance_w output, never the pre-advance
        # array (the rejected formula froze the interior geopotential).
        np.testing.assert_array_equal(actual[:, ~ring], ph_advanced[:, ~ring])
        assert np.all(np.abs(actual[:, ~ring] - ph_before[:, ~ring]) > 1.0e-10)


def test_rejected_single_base_formula_is_detected() -> None:
    """A regression that walks the ring but freezes the interior must fail."""

    arrays = _operands(4177)
    ph_before, ph_advanced = arrays[0], arrays[1]
    actual = _call(1, *arrays)
    rejected = _call(1, ph_before, ph_before, *arrays[2:])
    ny, nx = ph_before.shape[1:]
    ring = _ring_mask(ny, nx, 1)
    # Same ring algebra in both calls...
    np.testing.assert_array_equal(actual[:, ring], rejected[:, ring])
    # ...but the corrected interior must move with advance_w while the
    # rejected wiring keeps the pre-advance interior.
    assert np.all(np.abs(actual[:, ~ring] - rejected[:, ~ring]) > 1.0e-10)
    np.testing.assert_array_equal(rejected[:, ~ring], ph_before[:, ~ring])


def test_acoustic_call_site_feeds_advanced_base_and_pre_advance_ring_source() -> None:
    """AST gate: the nested 3b call must be
    ``ph_next = spec_bdyupdate_ph_tendency_inloop(ph_next, state_for_w.ph, ...)``.

    Passing anything other than the freshly advanced ``ph_next`` as the base
    reintroduces the interior-freeze blocker; passing anything other than the
    loop-carried ``state_for_w.ph`` as the ring source breaks the WRF
    loop-bound exclusion semantics.
    """

    source = textwrap.dedent(inspect.getsource(acoustic.acoustic_substep_core))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "spec_bdyupdate_ph_tendency_inloop"
    ]
    assert len(calls) == 1, "exactly one nested in-loop ph spec update call"
    call = calls[0]
    base = call.args[0]
    ring_source = call.args[1]
    assert isinstance(base, ast.Name) and base.id == "ph_next", (
        "the base operand must be the freshly advanced ph_next"
    )
    assert (
        isinstance(ring_source, ast.Attribute)
        and ring_source.attr == "ph"
        and isinstance(ring_source.value, ast.Name)
        and ring_source.value.id == "state_for_w"
    ), "the ring source must be the pre-advance loop-carried state_for_w.ph"


def test_two_positional_operand_signature_rejects_old_call_shape() -> None:
    """The rejected 9-positional call shape must not silently misbind."""

    arrays = _operands(902)
    ph_before, _, ph_save, ph_tend, mu_tend, muts, c1f, c2f = arrays
    with pytest.raises(TypeError):
        spec_bdyupdate_ph_tendency_inloop(  # noqa: PLE1120 - deliberate
            jnp.asarray(ph_before),
            jnp.asarray(ph_tend),
            jnp.asarray(ph_save),
            jnp.asarray(mu_tend),
            jnp.asarray(muts),
            jnp.asarray(c1f),
            jnp.asarray(c2f),
            0.6,
            CFG,
        )


def test_corner_ownership_matches_wrf_single_write() -> None:
    """Y-side rows own corners; every ring cell is written exactly once with
    the same pointwise target, so double writes cannot diverge."""

    arrays = _operands(5150)
    ph_before, ph_advanced, ph_save, ph_tend, mu_tend, muts, c1f, c2f = arrays
    actual = _call(1, *arrays)
    expected_ring = oracle.wrf_spec_ph_update(
        ph_before,
        ph_save,
        ph_tend,
        mu_tend[None, :, :],
        muts[None, :, :],
        c1f[:, None, None],
        c2f[:, None, None],
        0.6,
    )
    for corner in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        np.testing.assert_allclose(
            actual[:, corner[0], corner[1]],
            expected_ring[:, corner[0], corner[1]],
            rtol=3.0e-15,
            atol=3.0e-13,
        )
