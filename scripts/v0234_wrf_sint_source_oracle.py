#!/usr/bin/env python3
"""Independent NumPy oracle for tracked WRF v4.7.1 ``share/sint.F``.

This module intentionally imports no ``gpuwrf`` code.  It is a second,
source-literal implementation used to judge the JAX candidate, never the other
way around.  Source authority is commit
``f52c197ed39d12e087d02c50f412d90d418f6186``.
"""

from __future__ import annotations

import numpy as np


def _limited(stencil: np.ndarray, a: float | np.ndarray) -> np.ndarray:
    """Literal DONOR/TR4/OV/UN block from ``sint.F:241-328``."""

    dtype = stencil.dtype
    f = dtype.type
    zero, one = f(0.0), f(1.0)
    one12, one24, ep = one / f(12.0), one / f(24.0), f(1.0e-10)
    a = np.asarray(a, dtype=dtype)
    ym2, ym1, y0, yp1, yp2 = (stencil[..., index] for index in range(5))
    sign_a = np.where(a >= zero, one, -one)
    fl0 = (ym1 * np.maximum(zero, sign_a) - y0 * np.minimum(zero, sign_a)) * a
    fl1 = (y0 * np.maximum(zero, sign_a) - yp1 * np.minimum(zero, sign_a)) * a
    w = y0 - (fl1 - fl0)
    mxm = np.maximum(np.maximum(ym1, y0), np.maximum(yp1, w))
    mn = np.minimum(np.minimum(ym1, y0), np.minimum(yp1, w))

    def tr4(v_m1, v_0, v_p1, v_p2):
        return (
            a * one12 * (f(7.0) * (v_p1 + v_0) - (v_p2 + v_m1))
            - a * a * one24 * (f(15.0) * (v_p1 - v_0) - (v_p2 - v_m1))
            - a * a * a * one12 * ((v_p1 + v_0) - (v_p2 + v_m1))
            + a * a * a * a * one24 * (f(3.0) * (v_p1 - v_0) - (v_p2 - v_m1))
        )

    f0 = tr4(ym2, ym1, y0, yp1) - fl0
    f1 = tr4(ym1, y0, yp1, yp2) - fl1
    ov = (mxm - w) / (-np.minimum(zero, f1) + np.maximum(zero, f0) + ep)
    un = (w - mn) / (np.maximum(zero, f1) - np.minimum(zero, f0) + ep)
    f0 = np.maximum(zero, f0) * np.minimum(one, ov) + np.minimum(zero, f0) * np.minimum(one, un)
    f1 = np.maximum(zero, f1) * np.minimum(one, un) + np.minimum(zero, f1) * np.minimum(one, ov)
    return w - (f1 - f0)


def _offset(subcell: int, ratio: int, staggered: bool, dtype: np.dtype) -> float:
    """Literal ``XIG``/``XJG`` formula from ``sint.F:253-264``."""

    f = dtype.type
    stagger_offset = f(1.0) if staggered and ratio % 2 == 0 else f(0.0)
    return float(
        (f(ratio) - f(1.0) - stagger_offset) / f(2 * ratio)
        - f(subcell) / f(ratio)
    )


def sint_full(
    coarse: np.ndarray,
    *,
    ratio: int,
    i_parent_start: int,
    j_parent_start: int,
    child_ny: int,
    child_nx: int,
    xstag: bool = False,
    ystag: bool = False,
) -> np.ndarray:
    """Evaluate full WRF SINT on one 2-D/3-D parent field.

    The loop/index mapping is the literal ``bdy_interp1`` mapping:
    ``ci=ipos+(ni1-1)/nri`` and ``IIM=ip+1+jp*nri``.  No clipping is performed;
    invalid nest starts fail naturally instead of changing the source stencil.
    """

    source = np.asarray(coarse)
    if source.ndim not in (2, 3):
        raise ValueError(f"expected 2-D/3-D parent field, got {source.shape}")
    was_2d = source.ndim == 2
    src = source[None, ...] if was_2d else source
    rr = int(ratio)
    i0 = int(i_parent_start) - 1
    j0 = int(j_parent_start) - 1
    # ``bdy_interp1`` first maps a destination index to the shifted SINT work
    # index (NI=NI1-IOFF / NJ=NJ1-JOFF); inverting that map for a requested
    # destination gives NI1-1 = destination_i + IOFF.  This is independent of
    # the even-ratio RIOFF/RJOFF used later in XIG/XJG.
    ioff = max((rr - 1) // 2, 1) if bool(xstag) else 0
    joff = max((rr - 1) // 2, 1) if bool(ystag) else 0
    fine_x = np.arange(int(child_nx), dtype=np.int64) + ioff
    fine_y = np.arange(int(child_ny), dtype=np.int64) + joff
    center_x = i0 + fine_x // rr
    center_y = j0 + fine_y // rr
    ax = np.asarray(
        [_offset(int(index % rr), rr, bool(xstag), src.dtype) for index in fine_x],
        dtype=src.dtype,
    )
    ay = np.asarray(
        [_offset(int(index % rr), rr, bool(ystag), src.dtype) for index in fine_y],
        dtype=src.dtype,
    )
    offsets = np.arange(-2, 3, dtype=np.int64)
    x_stencil = np.take(src, center_x[:, None] + offsets[None], axis=2)
    x_value = _limited(x_stencil, ax[None, None, :])
    y_stencil = np.take(x_value, center_y[:, None] + offsets[None], axis=1)
    y_stencil = np.moveaxis(y_stencil, 2, -1)
    out = _limited(y_stencil, ay[None, :, None])
    return out[0] if was_2d else out


def wrf_sides(field: np.ndarray, *, width: int, side_len: int) -> np.ndarray:
    """Independent W/E/S/N boundary-record layout used by ``bdy_interp1``."""

    value = np.asarray(field)
    if value.ndim == 2:
        value = value[None, ...]
    z_len, ny, nx = value.shape
    strips = (
        np.moveaxis(value[:, :, :width], 2, 0),
        np.moveaxis(value[:, :, nx - width :][:, :, ::-1], 2, 0),
        np.moveaxis(value[:, :width, :], 1, 0),
        np.moveaxis(value[:, ny - width :, :][:, ::-1, :], 1, 0),
    )
    result = np.zeros((4, int(width), z_len, int(side_len)), dtype=value.dtype)
    for side, strip in enumerate(strips):
        result[side, : strip.shape[0], :, : strip.shape[2]] = strip
    return result


__all__ = ["sint_full", "wrf_sides"]
