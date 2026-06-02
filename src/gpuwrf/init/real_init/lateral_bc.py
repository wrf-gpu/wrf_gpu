"""S3 (Opus) — lateral boundary value + tendency generation (wrfbdy-equiv).

FROZEN ENTRY SIGNATURE. Reproduces real_em.F::assemble_output (main/real_em.F:680
-1240) over the forcing intervals. For each interval the forcing met_em column is
run through the SAME S1+S2 init (vinterp + hydrostatic + surface) to get a full
state at that valid time, then:

  1. COUPLE each prognostic field with the same hybrid-coordinate mass term
     WRF uses in ``dyn_em/module_big_step_utilities_em.F::couple``:
     U/V use staggered dry mass plus ``C1H/C2H`` and map factors; T/QV use
     mass-point ``C1H/C2H``; PH uses full-level ``C1F/C2F``. ``T_B*`` stores
     WRF's ``THM`` boundary variable (moist theta by default), not raw ``T``.
     MU is stuffed uncoupled as ``MU_2``.
  2. STUFF the coupled value into the per-side boundary frames (stuff_bdy):
     XS = first ``spec_bdy_width`` i-columns, XE = last, YS/YE the j-rows; for
     U-stagger ide->ide, V-stagger jde->jde (share/module_bc.F:2934-3100).
  3. TENDENCY between consecutive intervals (stuff_bdytend_new,
     share/module_bc.F:2893, used at real_em.F:1123-1163):
         tend = (coupled_{n+1} - coupled_{n}) / interval_seconds      (:2938)
     The first interval's coupled value is the specified VALUE (the ``_bxs`` etc
     in LateralBC.values); the tendency is stored in ``_btxs`` etc.

NOTE the coupling needs MU/MUB/MSF from S1+S2 at EACH forcing time, so S3 calls
the S1/S2 entry points (a true data dependency on their *interfaces*, which are
frozen here). S3 develops against the frozen stubs / the real.exe wrfbdy oracle;
it does NOT need S1/S2 *implementations* merged to start (it can mock the per-
time state from the oracle wrfinput at each interval during development).

Oracle: wrfbdy U_BXS/.../U_BTXS/... etc for d01 (the parent carries LBC) across
the ≥10 cases; tols ``types.WRFBDY_TOLS`` on the coupled values + tendencies.

FILE OWNERSHIP: this file is S3's exclusive file. Do not edit types.py, driver.py,
or any S1/S2 file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gpuwrf.init.real_init.types import LateralBC, RealInitConfig
from gpuwrf.init.real_init.types import RVOVRD, T0
from gpuwrf.init.metgrid_schema import MetEmArtifact


_SIDE_VALUE_KEYS = ("bxs", "bxe", "bys", "bye")
_SIDE_TENDENCY_KEYS = ("btxs", "btxe", "btys", "btye")
_OPTIONAL_MOIST_FIELDS = ("qc", "qr", "qi", "qs", "qg")


@dataclass(frozen=True)
class _ForcingSnapshot:
    """Initialized real.exe forcing-time state in WRF array order."""

    valid_time: str
    u: np.ndarray
    v: np.ndarray
    thm: np.ndarray
    qv: np.ndarray
    ph: np.ndarray
    mu: np.ndarray
    mub: np.ndarray
    mapfac_uy: np.ndarray
    mapfac_vx: np.ndarray
    c1h: np.ndarray
    c2h: np.ndarray
    c1f: np.ndarray
    c2f: np.ndarray
    qc: np.ndarray | None = None
    qr: np.ndarray | None = None
    qi: np.ndarray | None = None
    qs: np.ndarray | None = None
    qg: np.ndarray | None = None


def generate_lateral_bc(
    config: RealInitConfig,
    forcing_sequence: Sequence[MetEmArtifact],
) -> LateralBC:
    """Builds wrfbdy-equivalent specified values + tendencies.

    ``forcing_sequence`` is the time-ordered list of met_em artifacts at the
    forcing intervals (e.g. AIFS 6-hourly). Each is initialized to a full state
    via the S1/S2 entry points, coupled, stuffed, and differenced. Returns the
    per-side value + tendency frames in :class:`LateralBC`.
    """

    if len(forcing_sequence) < 2:
        raise ValueError("lateral boundary generation requires at least two forcing times")
    if config.spec_bdy_width <= 0:
        raise ValueError("spec_bdy_width must be positive")
    if config.interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    snapshots = _prepare_snapshots(config, forcing_sequence)
    coupled = [_couple_snapshot(snapshot) for snapshot in snapshots]
    fields = _ordered_field_names(coupled)

    n_intervals = len(coupled) - 1
    value_accum: dict[str, dict[str, list[np.ndarray]]] = {
        field: {side: [] for side in _SIDE_VALUE_KEYS} for field in fields
    }
    tendency_accum: dict[str, dict[str, list[np.ndarray]]] = {
        field: {side: [] for side in _SIDE_TENDENCY_KEYS} for field in fields
    }

    for interval_index in range(n_intervals):
        current = coupled[interval_index]
        next_ = coupled[interval_index + 1]
        for field in fields:
            values = _stuff_bdy(current[field], config.spec_bdy_width)
            tendencies = _stuff_bdy(
                (next_[field] - current[field]) / float(config.interval_seconds),
                config.spec_bdy_width,
            )
            for side_key, array in values.items():
                value_accum[field][side_key].append(array)
            for value_key, tendency_key in zip(_SIDE_VALUE_KEYS, _SIDE_TENDENCY_KEYS, strict=True):
                tendency_accum[field][tendency_key].append(tendencies[value_key])

    values = {
        field: {
            side_key: np.stack(arrays, axis=0)
            for side_key, arrays in sides.items()
        }
        for field, sides in value_accum.items()
    }
    tendencies = {
        field: {
            side_key: np.stack(arrays, axis=0)
            for side_key, arrays in sides.items()
        }
        for field, sides in tendency_accum.items()
    }

    return LateralBC(
        values=values,
        tendencies=tendencies,
        spec_bdy_width=int(config.spec_bdy_width),
        bdyfrq_seconds=float(config.interval_seconds),
        valid_times=tuple(snapshot.valid_time for snapshot in snapshots[:-1]),
        coupled_field_names=tuple(fields),
    )


def _prepare_snapshots(
    config: RealInitConfig,
    forcing_sequence: Sequence[MetEmArtifact],
) -> list[_ForcingSnapshot]:
    if all(isinstance(item, _ForcingSnapshot) or _looks_initialized(item) for item in forcing_sequence):
        return [_snapshot_from_initialized(item) for item in forcing_sequence]

    from gpuwrf.init.real_init import base_state
    from gpuwrf.init.real_init import hydrostatic
    from gpuwrf.init.real_init import surface_init
    from gpuwrf.init.real_init import vertical_coord
    from gpuwrf.init.real_init import vinterp

    vcoord = vertical_coord.compute_vertical_coord(config)
    snapshots: list[_ForcingSnapshot] = []
    for metem in forcing_sequence:
        surface = surface_init.compute_surface_init(config, metem)
        base = base_state.compute_base_state(config, vcoord, surface.hgt)
        seed = vinterp.vertical_interpolate(config, vcoord, metem)
        dynamics = hydrostatic.balance(config, vcoord, base, seed)
        snapshots.append(
            _snapshot_from_parts(
                valid_time=metem.valid_time,
                dynamics=dynamics,
                base=base,
                surface=surface,
                vcoord=vcoord,
            )
        )
    return snapshots


def _looks_initialized(item: object) -> bool:
    return (
        hasattr(item, "dynamics")
        and hasattr(item, "base")
        and hasattr(item, "surface")
        and hasattr(item, "vcoord")
    ) or (
        hasattr(item, "u")
        and hasattr(item, "v")
        and hasattr(item, "ph")
        and hasattr(item, "mu")
        and hasattr(item, "mub")
    )


def _snapshot_from_initialized(item: object) -> _ForcingSnapshot:
    if isinstance(item, _ForcingSnapshot):
        return item
    if hasattr(item, "dynamics") and hasattr(item, "base") and hasattr(item, "surface"):
        vcoord = getattr(item, "vcoord", None)
        if vcoord is None:
            raise ValueError("initialized forcing objects must carry vcoord")
        valid_time = str(getattr(item, "valid_time", getattr(item, "init_time", "")))
        return _snapshot_from_parts(
            valid_time=valid_time,
            dynamics=getattr(item, "dynamics"),
            base=getattr(item, "base"),
            surface=getattr(item, "surface"),
            vcoord=vcoord,
        )

    return _ForcingSnapshot(
        valid_time=str(getattr(item, "valid_time", "")),
        u=_array_attr(item, "u"),
        v=_array_attr(item, "v"),
        thm=_thm_from_direct_snapshot(item),
        qv=_array_attr(item, "qv"),
        ph=_array_attr(item, "ph"),
        mu=_array_attr(item, "mu"),
        mub=_array_attr(item, "mub"),
        mapfac_uy=_array_attr(item, "mapfac_uy"),
        mapfac_vx=_array_attr(item, "mapfac_vx"),
        c1h=_array_attr(item, "c1h"),
        c2h=_array_attr(item, "c2h"),
        c1f=_array_attr(item, "c1f"),
        c2f=_array_attr(item, "c2f"),
        qc=_optional_array_attr(item, "qc"),
        qr=_optional_array_attr(item, "qr"),
        qi=_optional_array_attr(item, "qi"),
        qs=_optional_array_attr(item, "qs"),
        qg=_optional_array_attr(item, "qg"),
    )


def _snapshot_from_parts(
    *,
    valid_time: str,
    dynamics: Any,
    base: Any,
    surface: Any,
    vcoord: Any,
) -> _ForcingSnapshot:
    qv = _array_attr(dynamics, "qv")
    theta = _array_attr(dynamics, "theta")
    thm = _optional_array_attr(dynamics, "thm")
    if thm is None:
        thm = (theta + T0) * (1.0 + RVOVRD * qv) - T0

    return _ForcingSnapshot(
        valid_time=str(valid_time),
        u=_array_attr(dynamics, "u"),
        v=_array_attr(dynamics, "v"),
        thm=thm,
        qv=qv,
        ph=_array_attr(dynamics, "ph"),
        mu=_array_attr(dynamics, "mu"),
        mub=_array_attr(base, "mub"),
        mapfac_uy=_array_attr(surface, "mapfac_uy"),
        mapfac_vx=_array_attr(surface, "mapfac_vx"),
        c1h=_array_attr(vcoord, "c1h"),
        c2h=_array_attr(vcoord, "c2h"),
        c1f=_array_attr(vcoord, "c1f"),
        c2f=_array_attr(vcoord, "c2f"),
        qc=_optional_array_attr(dynamics, "qc"),
        qr=_optional_array_attr(dynamics, "qr"),
        qi=_optional_array_attr(dynamics, "qi"),
        qs=_optional_array_attr(dynamics, "qs"),
        qg=_optional_array_attr(dynamics, "qg"),
    )


def _thm_from_direct_snapshot(item: object) -> np.ndarray:
    thm = _optional_array_attr(item, "thm")
    if thm is not None:
        return thm
    theta = _optional_array_attr(item, "theta")
    qv = _optional_array_attr(item, "qv")
    if theta is None or qv is None:
        raise AttributeError("initialized snapshots must carry either thm or both theta and qv")
    return (theta + T0) * (1.0 + RVOVRD * qv) - T0


def _array_attr(item: object, name: str) -> np.ndarray:
    if isinstance(item, dict):
        value = item[name]
    else:
        value = getattr(item, name)
    return np.asarray(value, dtype=np.float64)


def _optional_array_attr(item: object, name: str) -> np.ndarray | None:
    if isinstance(item, dict):
        value = item.get(name)
    else:
        value = getattr(item, name, None)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _ordered_field_names(coupled: Sequence[dict[str, np.ndarray]]) -> list[str]:
    fields = ["u", "v", "t", "ph", "qv", "mu"]
    for optional in _OPTIONAL_MOIST_FIELDS:
        if all(optional in item for item in coupled):
            fields.append(optional)
    return fields


def _couple_snapshot(snapshot: _ForcingSnapshot) -> dict[str, np.ndarray]:
    mu = np.asarray(snapshot.mu, dtype=np.float64)
    mub = np.asarray(snapshot.mub, dtype=np.float64)
    c1h = _as_1d(snapshot.c1h, "c1h")
    c2h = _as_1d(snapshot.c2h, "c2h")
    c1f = _as_1d(snapshot.c1f, "c1f")
    c2f = _as_1d(snapshot.c2f, "c2f")
    total_mass = mu + mub
    mass_u, mass_v = _staggered_total_mass(total_mass)
    mass_h = c1h[:, None, None] * total_mass[None, :, :] + c2h[:, None, None]
    mass_f = c1f[:, None, None] * total_mass[None, :, :] + c2f[:, None, None]

    coupled = {
        "u": np.asarray(snapshot.u, dtype=np.float64)
        * (c1h[:, None, None] * mass_u[None, :, :] + c2h[:, None, None])
        / np.asarray(snapshot.mapfac_uy, dtype=np.float64)[None, :, :],
        "v": np.asarray(snapshot.v, dtype=np.float64)
        * (c1h[:, None, None] * mass_v[None, :, :] + c2h[:, None, None])
        / np.asarray(snapshot.mapfac_vx, dtype=np.float64)[None, :, :],
        "t": np.asarray(snapshot.thm, dtype=np.float64) * mass_h,
        "ph": np.asarray(snapshot.ph, dtype=np.float64) * mass_f,
        "qv": np.asarray(snapshot.qv, dtype=np.float64) * mass_h,
        "mu": mu.copy(),
    }
    for optional in _OPTIONAL_MOIST_FIELDS:
        value = getattr(snapshot, optional)
        if value is not None:
            coupled[optional] = np.asarray(value, dtype=np.float64) * mass_h

    _validate_coupled_shapes(coupled, snapshot)
    return coupled


def _as_1d(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape {array.shape}")
    return array


def _staggered_total_mass(total_mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if total_mass.ndim != 2:
        raise ValueError(f"MU/MUB must be 2D; got total mass shape {total_mass.shape}")
    ny, nx = total_mass.shape
    mass_u = np.empty((ny, nx + 1), dtype=np.float64)
    mass_u[:, 0] = total_mass[:, 0]
    mass_u[:, 1:nx] = 0.5 * (total_mass[:, 1:] + total_mass[:, :-1])
    mass_u[:, nx] = total_mass[:, nx - 1]

    mass_v = np.empty((ny + 1, nx), dtype=np.float64)
    mass_v[0, :] = total_mass[0, :]
    mass_v[1:ny, :] = 0.5 * (total_mass[1:, :] + total_mass[:-1, :])
    mass_v[ny, :] = total_mass[ny - 1, :]
    return mass_u, mass_v


def _validate_coupled_shapes(
    coupled: dict[str, np.ndarray],
    snapshot: _ForcingSnapshot,
) -> None:
    nz, ny, nx_plus_one = coupled["u"].shape
    if coupled["v"].shape != (nz, ny + 1, nx_plus_one - 1):
        raise ValueError(f"V shape {coupled['v'].shape} is inconsistent with U shape {coupled['u'].shape}")
    if coupled["t"].shape != (nz, ny, nx_plus_one - 1):
        raise ValueError(f"T/THM shape {coupled['t'].shape} is inconsistent with U shape {coupled['u'].shape}")
    if coupled["qv"].shape != coupled["t"].shape:
        raise ValueError(f"QV shape {coupled['qv'].shape} does not match T shape {coupled['t'].shape}")
    if coupled["ph"].shape != (nz + 1, ny, nx_plus_one - 1):
        raise ValueError(f"PH shape {coupled['ph'].shape} is inconsistent with mass shape {coupled['t'].shape}")
    if coupled["mu"].shape != (ny, nx_plus_one - 1):
        raise ValueError(f"MU shape {coupled['mu'].shape} is inconsistent with mass shape {coupled['t'].shape}")
    if np.asarray(snapshot.mapfac_uy).shape != coupled["u"].shape[1:]:
        raise ValueError("mapfac_uy shape must match U horizontal staggering")
    if np.asarray(snapshot.mapfac_vx).shape != coupled["v"].shape[1:]:
        raise ValueError("mapfac_vx shape must match V horizontal staggering")


def _stuff_bdy(field: np.ndarray, width: int) -> dict[str, np.ndarray]:
    array = np.asarray(field, dtype=np.float64)
    if array.ndim == 3:
        if width > array.shape[2] or width > array.shape[1]:
            raise ValueError(f"spec_bdy_width={width} exceeds 3D field horizontal shape {array.shape}")
        return {
            "bxs": np.moveaxis(array[:, :, :width], 2, 0).copy(),
            "bxe": np.moveaxis(array[:, :, -width:][:, :, ::-1], 2, 0).copy(),
            "bys": np.moveaxis(array[:, :width, :], 1, 0).copy(),
            "bye": np.moveaxis(array[:, -width:, :][:, ::-1, :], 1, 0).copy(),
        }
    if array.ndim == 2:
        if width > array.shape[1] or width > array.shape[0]:
            raise ValueError(f"spec_bdy_width={width} exceeds 2D field shape {array.shape}")
        return {
            "bxs": array[:, :width].T.copy(),
            "bxe": array[:, -width:][:, ::-1].T.copy(),
            "bys": array[:width, :].copy(),
            "bye": array[-width:, :][::-1, :].copy(),
        }
    raise ValueError(f"boundary field must be 2D or 3D; got shape {array.shape}")
