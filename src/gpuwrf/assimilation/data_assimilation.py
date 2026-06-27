"""WRF-referenced DFI and FDDA nudging primitives.

This module intentionally implements the resident-array part of WRF data
assimilation, not WRF's I/O stack.  Callers provide already-decoded analysis or
observation targets on the model grid; the kernels compute the per-second
target-minus-state tendencies that WRF's ``fddagd`` / ``fddaobs`` drivers feed
into the RK tendency merge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.dynamics.core.rk_addtend_dry import DryPhysicsTendencies


_TARGET_FIELDS: tuple[str, ...] = ("u", "v", "theta", "qv", "ph", "mu")
_DFI_FILTER_FIELDS: tuple[str, ...] = (
    "u",
    "v",
    "w",
    "theta",
    "qv",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
    "p_total",
    "p_perturbation",
    "ph_total",
    "ph_perturbation",
    "mu_total",
    "mu_perturbation",
    "qke",
)


def _replace_state(state, /, **updates):
    try:
        return state.replace(_cast=False, **updates)
    except TypeError:
        return state.replace(**updates)


def _maybe_array(value):
    return None if value is None else jnp.asarray(value)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NudgingComponent:
    """One grid/obs/spectral nudging source.

    ``*_old`` and ``*_new`` are WRF's bracketing analysis fields.  ``mode`` is:
    ``"analysis"`` for grid FDDA, ``"obs"`` for already-objective-analyzed gridded
    observation increments, and ``"spectral"`` for WRF-style large-scale nudging of
    low horizontal wavenumbers.  Coefficients are per-second nudging strengths
    (WRF ``guv``, ``gt``, ``gq``, ``gph``; ``gmu`` is a port-local extension for
    dry-column mass targets).
    """

    u_old: object = None
    u_new: object = None
    v_old: object = None
    v_new: object = None
    theta_old: object = None
    theta_new: object = None
    qv_old: object = None
    qv_new: object = None
    ph_old: object = None
    ph_new: object = None
    mu_old: object = None
    mu_new: object = None
    u_weight: object = None
    v_weight: object = None
    theta_weight: object = None
    qv_weight: object = None
    ph_weight: object = None
    mu_weight: object = None
    guv: float = 0.0
    gt: float = 0.0
    gq: float = 0.0
    gph: float = 0.0
    gmu: float = 0.0
    analysis_interval_s: float = 3600.0
    end_seconds: float = -1.0
    ramp_seconds: float = 0.0
    mode: str = "analysis"
    target_policy: str = "interpolate"
    x_wavenum: int = 0
    y_wavenum: int = 0

    def tree_flatten(self):
        children = tuple(getattr(self, name) for name in self._array_fields())
        aux = (
            float(self.guv),
            float(self.gt),
            float(self.gq),
            float(self.gph),
            float(self.gmu),
            float(self.analysis_interval_s),
            float(self.end_seconds),
            float(self.ramp_seconds),
            str(self.mode),
            str(self.target_policy),
            int(self.x_wavenum),
            int(self.y_wavenum),
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            guv,
            gt,
            gq,
            gph,
            gmu,
            analysis_interval_s,
            end_seconds,
            ramp_seconds,
            mode,
            target_policy,
            x_wavenum,
            y_wavenum,
        ) = aux
        values = dict(zip(cls._array_fields(), children))
        return cls(
            **values,
            guv=guv,
            gt=gt,
            gq=gq,
            gph=gph,
            gmu=gmu,
            analysis_interval_s=analysis_interval_s,
            end_seconds=end_seconds,
            ramp_seconds=ramp_seconds,
            mode=mode,
            target_policy=target_policy,
            x_wavenum=x_wavenum,
            y_wavenum=y_wavenum,
        )

    @staticmethod
    def _array_fields() -> tuple[str, ...]:
        fields: list[str] = []
        for name in _TARGET_FIELDS:
            fields.extend((f"{name}_old", f"{name}_new", f"{name}_weight"))
        return tuple(fields)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DataAssimilationConfig:
    """Optional resident DA bundle attached to ``OperationalNamelist``."""

    analysis: NudgingComponent | None = None
    observation: NudgingComponent | None = None
    spectral: NudgingComponent | None = None

    def tree_flatten(self):
        return (self.analysis, self.observation, self.spectral), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)


@dataclass(frozen=True)
class DigitalFilterConfig:
    """DFI coefficient controls matching WRF ``dfcoef`` inputs."""

    half_window_steps: int
    dt_s: float
    cutoff_s: float
    filter_id: int = 1


@dataclass(frozen=True)
class NudgingRates:
    """Uncoupled per-second DA rates on each field's native grid."""

    u: object = None
    v: object = None
    theta: object = None
    qv: object = None
    ph: object = None
    mu: object = None


def _component_enabled(component: NudgingComponent | None) -> bool:
    if component is None:
        return False
    return any(
        float(value) > 0.0
        for value in (component.guv, component.gt, component.gq, component.gph, component.gmu)
    )


def _interp_coef(component: NudgingComponent, lead_seconds) -> jax.Array:
    if component.target_policy == "old":
        return jnp.asarray(0.0, dtype=jnp.float64)
    if component.target_policy == "new":
        return jnp.asarray(1.0, dtype=jnp.float64)
    interval = max(float(component.analysis_interval_s), 1.0e-12)
    lead = jnp.asarray(lead_seconds, dtype=jnp.float64)
    old = jnp.floor(lead / interval) * interval
    return jnp.clip((lead - old) / interval, 0.0, 1.0)


def _time_factor(component: NudgingComponent, lead_seconds) -> jax.Array:
    end = float(component.end_seconds)
    if end < 0.0:
        return jnp.asarray(1.0, dtype=jnp.float64)
    lead = jnp.asarray(lead_seconds, dtype=jnp.float64)
    ramp = float(component.ramp_seconds)
    if ramp == 0.0:
        return jnp.where(lead <= end, 1.0, 0.0)
    width = abs(ramp)
    actual_end = end + width if ramp > 0.0 else end
    ramp_start = end if ramp > 0.0 else end - width
    full = lead < ramp_start
    in_ramp = (lead >= ramp_start) & (lead <= actual_end)
    return jnp.where(full, 1.0, jnp.where(in_ramp, (actual_end - lead) / width, 0.0))


def _field_coef(component: NudgingComponent, name: str) -> float:
    if name in {"u", "v"}:
        return float(component.guv)
    if name == "theta":
        return float(component.gt)
    if name == "qv":
        return float(component.gq)
    if name == "ph":
        return float(component.gph)
    if name == "mu":
        return float(component.gmu)
    raise KeyError(name)


def _target(component: NudgingComponent, name: str, coef: jax.Array):
    old = _maybe_array(getattr(component, f"{name}_old"))
    new = _maybe_array(getattr(component, f"{name}_new"))
    if old is None and new is None:
        return None
    if old is None:
        old = new
    if new is None:
        new = old
    return old * (1.0 - coef) + new * coef


def _weight(component: NudgingComponent, name: str, target: jax.Array) -> jax.Array:
    value = _maybe_array(getattr(component, f"{name}_weight"))
    if value is None:
        return jnp.asarray(1.0, dtype=target.dtype)
    return value.astype(target.dtype)


def _spectral_lowpass(diff: jax.Array, *, x_wavenum: int, y_wavenum: int) -> jax.Array:
    if int(x_wavenum) <= 0 and int(y_wavenum) <= 0:
        return diff
    nx = int(diff.shape[-1])
    ny = int(diff.shape[-2])
    spec = jnp.fft.fftn(diff, axes=(-2, -1))
    kx = jnp.fft.fftfreq(nx) * nx
    ky = jnp.fft.fftfreq(ny) * ny
    keep_x = jnp.abs(kx) <= int(x_wavenum) if int(x_wavenum) > 0 else jnp.ones_like(kx, dtype=bool)
    keep_y = jnp.abs(ky) <= int(y_wavenum) if int(y_wavenum) > 0 else jnp.ones_like(ky, dtype=bool)
    mask = keep_y[:, None] & keep_x[None, :]
    return jnp.fft.ifftn(spec * mask, axes=(-2, -1)).real.astype(diff.dtype)


def _component_rates(state, component: NudgingComponent, lead_seconds) -> NudgingRates:
    coef = _interp_coef(component, lead_seconds)
    tfac = _time_factor(component, lead_seconds)
    updates: dict[str, jax.Array | None] = {}
    spectral = component.mode == "spectral"
    for name in _TARGET_FIELDS:
        strength = _field_coef(component, name)
        target = _target(component, name, coef)
        if strength <= 0.0 or target is None or not hasattr(state, name):
            updates[name] = None
            continue
        current = jnp.asarray(getattr(state, name), dtype=target.dtype)
        diff = target - current
        if spectral:
            diff = _spectral_lowpass(
                diff,
                x_wavenum=int(component.x_wavenum),
                y_wavenum=int(component.y_wavenum),
            )
        updates[name] = jnp.asarray(strength, dtype=diff.dtype) * _weight(component, name, target) * tfac * diff
    return NudgingRates(**updates)


def _add_optional(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _add_rates(a: NudgingRates, b: NudgingRates) -> NudgingRates:
    return NudgingRates(**{
        name: _add_optional(getattr(a, name), getattr(b, name))
        for name in _TARGET_FIELDS
    })


def data_assimilation_rates(state, config: DataAssimilationConfig | None, lead_seconds) -> NudgingRates:
    """Return combined analysis/obs/spectral per-second nudging rates."""

    if config is None:
        return NudgingRates()
    rates = NudgingRates()
    for component in (config.analysis, config.observation, config.spectral):
        if _component_enabled(component):
            rates = _add_rates(rates, _component_rates(state, component, lead_seconds))
    return rates


def _x_face_average_2d(field: jax.Array) -> jax.Array:
    padded = jnp.pad(field, ((0, 0), (1, 1)), mode="edge")
    return 0.5 * (padded[:, :-1] + padded[:, 1:])


def _y_face_average_2d(field: jax.Array) -> jax.Array:
    padded = jnp.pad(field, ((1, 1), (0, 0)), mode="edge")
    return 0.5 * (padded[:-1, :] + padded[1:, :])


def data_assimilation_dry_tendencies(state, rates: NudgingRates, metrics) -> DryPhysicsTendencies:
    """Convert uncoupled DA rates to WRF RK ``*_tendf`` channels."""

    mu = jnp.asarray(state.mu_total)
    mass_h = metrics.c1h[:, None, None] * mu[None, :, :] + metrics.c2h[:, None, None]
    ru = None
    rv = None
    t = None
    ph = None
    mu_t = None
    if rates.u is not None:
        muu = _x_face_average_2d(mu)
        mass_u = metrics.c1h[:, None, None] * muu[None, :, :] + metrics.c2h[:, None, None]
        ru = mass_u * jnp.asarray(rates.u)
    if rates.v is not None:
        muv = _y_face_average_2d(mu)
        mass_v = metrics.c1h[:, None, None] * muv[None, :, :] + metrics.c2h[:, None, None]
        rv = mass_v * jnp.asarray(rates.v)
    if rates.theta is not None:
        t = mass_h * jnp.asarray(rates.theta)
    if rates.ph is not None:
        mass_f = metrics.c1f[:, None, None] * mu[None, :, :] + metrics.c2f[:, None, None]
        ph = mass_f * jnp.asarray(rates.ph)
    if rates.mu is not None:
        mu_t = jnp.asarray(rates.mu)
    return DryPhysicsTendencies(
        ru_tendf=ru,
        rv_tendf=rv,
        ph_tendf=ph,
        t_tendf=t,
        mu_tendf=mu_t,
    )


def add_dry_physics_tendencies(
    base: DryPhysicsTendencies,
    extra: DryPhysicsTendencies,
) -> DryPhysicsTendencies:
    """Add two optional ``DryPhysicsTendencies`` bundles field by field."""

    return DryPhysicsTendencies(**{
        name: _add_optional(getattr(base, name), getattr(extra, name))
        for name in base.__dataclass_fields__
    })


def apply_nudging_rates(
    state,
    rates: NudgingRates,
    dt_s: float,
    *,
    fields: Iterable[str] = _TARGET_FIELDS,
):
    """Apply uncoupled nudging rates directly to a state.

    The operational scan uses this only for qv, because u/v/theta/ph/mu are routed
    through WRF's RK dry-tendency channels.  Tests use the broader field list to
    prove the target-pull invariant without running the full dycore.
    """

    updates: dict[str, jax.Array] = {}
    for name in fields:
        rate = getattr(rates, name)
        if rate is None or not hasattr(state, name):
            continue
        updates[name] = jnp.asarray(getattr(state, name)) + float(dt_s) * jnp.asarray(rate)
    return state if not updates else _replace_state(state, **updates)


def _window_values(filter_id: int, nsteps: int) -> np.ndarray:
    if nsteps == 0:
        return np.ones(1, dtype=np.float64)
    n = np.arange(nsteps + 1, dtype=np.float64)
    if filter_id == 0:
        return np.ones(nsteps + 1, dtype=np.float64)
    if filter_id == 1:
        x = np.pi * n / float(nsteps + 1)
        window = np.ones(nsteps + 1, dtype=np.float64)
        if nsteps:
            window[1:] = np.sin(x[1:]) / x[1:]
        return window
    if filter_id == 2:
        return 0.54 + 0.46 * np.cos(np.pi * n / float(nsteps))
    if filter_id == 3:
        return 0.42 + 0.5 * np.cos(np.pi * n / float(nsteps)) + 0.08 * np.cos(2.0 * np.pi * n / float(nsteps))
    raise NotImplementedError("DFI filters 0=uniform, 1=Lanczos, 2=Hamming, 3=Blackman are implemented")


def dfi_filter_coefficients(config: DigitalFilterConfig) -> jax.Array:
    """Return WRF ``dfcoef`` half-window coefficients ``h[0..n]``.

    The normalization matches WRF ``NORMLZ``: ``h0 + 2*sum(h[1:]) == 1`` for the
    symmetric two-sided filter.
    """

    nsteps = int(config.half_window_steps)
    if nsteps < 0:
        raise ValueError("half_window_steps must be non-negative")
    dt = abs(float(config.dt_s))
    cutoff = float(config.cutoff_s)
    if cutoff <= 0.0:
        raise ValueError("cutoff_s must be positive")
    n = np.arange(nsteps + 1, dtype=np.float64)
    window = _window_values(int(config.filter_id), nsteps)
    omega_c = 2.0 * np.pi / cutoff
    sinc = np.empty(nsteps + 1, dtype=np.float64)
    sinc[0] = omega_c * dt / np.pi
    if nsteps:
        sinc[1:] = np.sin(n[1:] * omega_c * dt) / (n[1:] * np.pi)
    h = sinc * window
    denom = h[0] + 2.0 * np.sum(h[1:])
    if abs(denom) <= 1.0e-30:
        raise ValueError("DFI filter normalization is singular")
    return jnp.asarray(h / denom, dtype=jnp.float64)


def _state_weighted_add(acc, state, weight: jax.Array, fields: Iterable[str]):
    updates: dict[str, jax.Array] = {}
    for name in fields:
        if not hasattr(acc, name) or not hasattr(state, name):
            continue
        a = getattr(acc, name)
        b = getattr(state, name)
        if a is None or b is None:
            continue
        updates[name] = jnp.asarray(a, dtype=jnp.float64) + weight * jnp.asarray(b, dtype=jnp.float64)
    return _replace_state(acc, **updates)


def _zero_filter_fields(state, fields: Iterable[str]):
    updates = {
        name: jnp.zeros_like(getattr(state, name), dtype=jnp.float64)
        for name in fields
        if hasattr(state, name) and getattr(state, name) is not None
    }
    return _replace_state(state, **updates)


def digital_filter_initialize(
    state,
    advance_fn: Callable[[object], object],
    config: DigitalFilterConfig,
    *,
    backward_fn: Callable[[object], object] | None = None,
    fields: Iterable[str] = _DFI_FILTER_FIELDS,
):
    """Run a small DFI launch and return the filtered initial state.

    With ``backward_fn`` this is a two-sided symmetric accumulation.  Without it,
    the available forward samples are renormalized; that is the practical
    forward-DFI launch path used by the GPU port until reverse integration is
    wired.
    """

    h = dfi_filter_coefficients(config)
    fields = tuple(fields)
    acc = _zero_filter_fields(state, fields)
    norm = jnp.asarray(0.0, dtype=jnp.float64)

    acc = _state_weighted_add(acc, state, h[0], fields)
    norm = norm + h[0]

    fwd = state
    for step in range(1, int(config.half_window_steps) + 1):
        fwd = advance_fn(fwd)
        acc = _state_weighted_add(acc, fwd, h[step], fields)
        norm = norm + h[step]

    if backward_fn is not None:
        bck = state
        for step in range(1, int(config.half_window_steps) + 1):
            bck = backward_fn(bck)
            acc = _state_weighted_add(acc, bck, h[step], fields)
            norm = norm + h[step]

    updates: dict[str, jax.Array] = {}
    for name in fields:
        if hasattr(acc, name) and getattr(acc, name) is not None:
            updates[name] = jnp.asarray(getattr(acc, name)) / norm
    return _replace_state(state, **updates)
