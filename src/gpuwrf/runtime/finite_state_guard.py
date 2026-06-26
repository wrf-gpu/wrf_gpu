"""Fail-fast finite-state guard for forecast chunk/output boundaries.

``GPUWRF_FINITE_CHECK`` is enabled by default. Set it to ``0``/``false``/``off``
only for explicit max-performance experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np


_FALSEY = {"0", "false", "no", "off", ""}
_TRUTHY = {"1", "true", "yes", "on"}

# State fields that can evolve during the forecast. Boundary/static geography
# leaves are intentionally excluded; they are input-validation territory.
PROGNOSTIC_STATE_FIELDS: tuple[str, ...] = (
    "u",
    "v",
    "w",
    "theta",
    "qv",
    "p_total",
    "p",
    "p_perturbation",
    "ph_total",
    "ph",
    "ph_perturbation",
    "mu_total",
    "mu",
    "mu_perturbation",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
    "Ni",
    "Nr",
    "Ns",
    "Ng",
    "Nc",
    "Nn",
    "qke",
    "qsq",
    "qc_bl",
    "qi_bl",
    "cldfra_bl",
    "qh",
    "Nh",
    "qvolg",
    "qvolh",
    "nwfa",
    "nifa",
    "rain_acc",
    "rainc_acc",
    "snow_acc",
    "graupel_acc",
    "ice_acc",
    "hail_acc",
    "ustar",
    "theta_flux",
    "qv_flux",
    "tau_u",
    "tau_v",
    "rhosfc",
    "fltv",
    "t_skin",
    "soil_moisture",
)

_LEGACY_TOTAL_ALIASES = {"p": "p_total", "ph": "ph_total", "mu": "mu_total"}


@dataclass(frozen=True)
class NonFiniteLocation:
    """First detected non-finite prognostic value."""

    field: str
    domain: str
    step: int
    level: int | None
    index: tuple[int, ...]
    sim_time_s: float | None = None


class NonFiniteStateError(RuntimeError):
    """Raised when the forecast state first contains NaN/Inf at a boundary."""

    def __init__(self, location: NonFiniteLocation) -> None:
        self.location = location
        self.field = location.field
        self.domain = location.domain
        self.step = location.step
        self.level = location.level
        self.index = location.index
        self.sim_time_s = location.sim_time_s
        sim = "" if location.sim_time_s is None else f" sim_time_s={location.sim_time_s:.6g}"
        super().__init__(
            "non-finite prognostic state detected: "
            f"domain={location.domain} field={location.field} "
            f"level={location.level} step={location.step}{sim} "
            f"first_index={location.index}"
        )


def finite_check_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Resolve ``GPUWRF_FINITE_CHECK``; default is enabled/fail-fast."""

    env = os.environ if environ is None else environ
    raw = env.get("GPUWRF_FINITE_CHECK")
    if raw is None:
        return True
    value = raw.strip().lower()
    if value in _FALSEY:
        return False
    if value in _TRUTHY:
        return True
    return True


def first_nonfinite_state_location(
    state: Any,
    *,
    domain: str,
    step: int,
    sim_time_s: float | None = None,
    fields: tuple[str, ...] = PROGNOSTIC_STATE_FIELDS,
) -> NonFiniteLocation | None:
    """Return the first non-finite prognostic field location, or ``None``.

    The success path performs one device-side ``isfinite(...).all()`` reduction
    per floating field and transfers only scalar booleans. If a field fails, only
    that field's first bad flat index is transferred for the diagnostic.
    """

    seen: set[str] = set()
    for field in fields:
        alias_target = _LEGACY_TOTAL_ALIASES.get(field)
        if alias_target is not None and alias_target in seen and hasattr(state, alias_target):
            continue
        if field in seen or not hasattr(state, field):
            continue
        seen.add(field)
        value = getattr(state, field)
        if value is None or not _is_floating_array(value):
            continue
        if _all_finite(value):
            continue
        index = _first_bad_index(value)
        level = index[0] if len(index) >= 3 else None
        return NonFiniteLocation(
            field=field,
            domain=str(domain),
            step=int(step),
            level=None if level is None else int(level),
            index=tuple(int(i) for i in index),
            sim_time_s=None if sim_time_s is None else float(sim_time_s),
        )
    return None


def assert_state_finite_at_boundary(
    state: Any,
    *,
    domain: str,
    step: int,
    sim_time_s: float | None = None,
    enabled: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Fail fast if a boundary/output-cadence state contains NaN/Inf."""

    if enabled is None:
        enabled = finite_check_enabled(environ)
    if not enabled:
        return
    location = first_nonfinite_state_location(
        state, domain=domain, step=step, sim_time_s=sim_time_s
    )
    if location is not None:
        raise NonFiniteStateError(location)


def _is_floating_array(value: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        try:
            dtype = np.asarray(value).dtype
        except Exception:
            return False
    return bool(np.issubdtype(np.dtype(dtype), np.inexact))


def _all_finite(value: Any) -> bool:
    if isinstance(value, jax.Array):
        return bool(np.asarray(jnp.all(jnp.isfinite(value))))
    array = np.asarray(value)
    return bool(np.isfinite(array).all())


def _first_bad_index(value: Any) -> tuple[int, ...]:
    if isinstance(value, jax.Array):
        bad = jnp.ravel(~jnp.isfinite(value))
        flat = int(np.asarray(jnp.argmax(bad)))
        return tuple(int(i) for i in np.unravel_index(flat, tuple(int(d) for d in value.shape)))
    array = np.asarray(value)
    flat_indices = np.flatnonzero(~np.isfinite(array))
    if flat_indices.size == 0:
        return ()
    return tuple(int(i) for i in np.unravel_index(int(flat_indices[0]), array.shape))


__all__ = [
    "NonFiniteLocation",
    "NonFiniteStateError",
    "PROGNOSTIC_STATE_FIELDS",
    "assert_state_finite_at_boundary",
    "finite_check_enabled",
    "first_nonfinite_state_location",
]
