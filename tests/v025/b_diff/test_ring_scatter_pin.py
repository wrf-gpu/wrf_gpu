"""ring_select.scatter_mask pins its operand and result row-major on nested-child steps (layout only).

Under the strict-guards default the z-window scatter pulled the RK fields to a z-fastest layout
(deviceless GPU-faithful compile of the PROD d02 step: 498 transposes vs 230 with legacy guards;
233 with the pin). On the root step the same pin flips the carry layout the other way (d01 733 ->
866), so only nested-child steps (ring_select.nested_step(True), set by the step builder) pin.
The value test is bitwise in both contexts; the layout test lowers for CUDA in this CPU process
(E115) and is deletion-sensitive (the unpinned helper fails it).
"""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import ring_select


def _case(dtype, shape=(5, 9, 11), width=2, seed=0):
    rng = np.random.default_rng(seed)
    field = rng.standard_normal(shape).astype(dtype)
    target = rng.standard_normal(shape).astype(dtype)
    return field, target, ring_select.ring_mask(shape[-2], shape[-1], width)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(5, 9, 11), (9, 11)])
@pytest.mark.parametrize("nested", [False, True])
def test_scatter_mask_values_bitwise(dtype, shape, nested):
    field, target, mask = _case(dtype, shape)
    expected = np.where(mask, target, field)
    with ring_select.nested_step(nested):
        eager = np.asarray(ring_select.scatter_mask(jnp.asarray(field), jnp.asarray(target), mask))
        jitted = np.asarray(jax.jit(lambda f, t: ring_select.scatter_mask(f, t, mask))(field, target))
    assert eager.dtype == jitted.dtype == np.dtype(dtype)
    assert eager.tobytes() == expected.tobytes() and jitted.tobytes() == expected.tobytes()


def _row_major_pinned_scatter(text):
    """True when the scatter's operand comes from a row-major LayoutConstraint and its result feeds one."""
    pins = {m.group(1): m.group(2) for m in re.finditer(
        r"(%[\w#]+) = stablehlo\.custom_call @LayoutConstraint\((%[\w#]+)\).*?result_layouts = \[dense<\[2, 1, 0\]>", text)}
    scatter = re.search(r'(%[\w#]+) = "stablehlo\.scatter"\((%[\w#]+),', text)
    if scatter is None:
        return False
    return scatter.group(2) in pins and scatter.group(1) in pins.values()


def _lowered_text(helper_patch=None, nested=True):
    field, target, mask = _case(np.float32)
    spec = jax.ShapeDtypeStruct(field.shape, jnp.float32)
    original = ring_select._row_major
    if helper_patch is not None:
        ring_select._row_major = helper_patch
    try:
        fn = lambda f, t: ring_select.scatter_mask(f, t, mask)  # fresh function per call (E140)
        with ring_select.nested_step(nested):
            return jax.jit(fn).trace(spec, spec).lower(lowering_platforms=("cuda",)).as_text()
    finally:
        ring_select._row_major = original


def test_scatter_mask_lowers_with_row_major_pins_for_cuda():
    assert jax.devices()[0].platform == "cpu"
    assert _row_major_pinned_scatter(_lowered_text())


def test_layout_check_is_deletion_sensitive():
    assert not _row_major_pinned_scatter(_lowered_text(helper_patch=lambda value: value))


def test_root_step_is_not_pinned():
    text = _lowered_text(nested=False)
    assert "@LayoutConstraint" not in text and '"stablehlo.scatter"' in text


def test_step_builder_marks_nested_children():
    import inspect
    from gpuwrf.runtime import operational_mode as op
    source = inspect.getsource(op._advance_chunk_fori)
    assert "nested_step(_acoustic_lateral_bc_flags(namelist)[2])" in source
