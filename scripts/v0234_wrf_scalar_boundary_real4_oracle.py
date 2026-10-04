#!/usr/bin/env python3
"""REAL4 caller adapter for the v0234 nested scalar boundary source oracle.

WRF declares the moist/scalar boundary records, the coupled scalar and the
half-level mass REAL (RWORDSIZE=4).  ``share/interp_fcn.F::bdy_interp1``
subtracts the REAL records and multiplies by a REAL*8 reciprocal interval;
assignment rounds the tendency back to REAL.  This module re-binds the
unchanged source-literal tendency body to those REAL operations (NumPy binary32
arithmetic: every operation separately rounded, no fused multiply-add).

B36 uses it for the Thompson qc/qr/qi/qs/qg/Ni/Nr families; QV keeps the
fp64 algebra of ``v0234_wrf_scalar_boundary_oracle``.
"""

from __future__ import annotations

import math
from types import FunctionType

import numpy as np

import v0234_wrf_scalar_boundary_oracle as _source


def boundary_rate(records, lead_seconds, cadence_s):
    records = np.asarray(records, np.float32)
    if records.shape[0] < 2:
        return np.zeros_like(records[0])
    lower = min(
        max(int(math.ceil(float(lead_seconds) / float(cadence_s))) - 1, 0),
        records.shape[0] - 2,
    )
    delta = records[lower + 1] - records[lower]
    return (delta.astype(np.float64) * (1.0 / float(cadence_s))).astype(np.float32)


def boundary_value(records, lead_seconds, cadence_s):
    records = np.asarray(records, np.float32)
    lower = min(
        max(int(math.ceil(float(lead_seconds) / float(cadence_s))) - 1, 0),
        max(records.shape[0] - 2, 0),
    )
    dtbc = np.float32(
        min(max(float(lead_seconds) - lower * float(cadence_s), 0.0), float(cadence_s))
    )
    return records[lower] + dtbc * boundary_rate(records, lead_seconds, cadence_s)


def couple_real4(scalar, mu_total, c1h, c2h):
    """REAL ``scalar*(c1h*mut+c2h)`` with each operation rounded."""

    c1 = np.asarray(c1h, np.float32)[:, None, None]
    c2 = np.asarray(c2h, np.float32)[:, None, None]
    mass = c1 * np.asarray(mu_total, np.float32)[None] + c2
    return np.asarray(scalar, np.float32) * mass


_body = FunctionType(
    _source.scalar_boundary_tendency.__code__,
    dict(_source.__dict__, boundary_value=boundary_value, boundary_rate=boundary_rate),
)
_body.__kwdefaults__ = dict(_source.scalar_boundary_tendency.__kwdefaults__ or {})


def scalar_boundary_tendency(scalar, mu_total, c1h, c2h, records, **kwargs):
    return _body(
        *(np.asarray(v, np.float32) for v in (scalar, mu_total, c1h, c2h, records)),
        **kwargs,
    )
