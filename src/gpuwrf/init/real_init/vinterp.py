"""S1 (Opus) — vertical interpolation of met_em columns onto model eta.

FROZEN ENTRY SIGNATURE. Implements the real.exe vertical-interpolation part
(module_initialize_real.F:1450-2809): build dry pressure column, find p_top,
integrate moisture, compute MU0, then vert_interp every atmos field from the
metgrid isobaric levels (grid%pd_gc) to the model dry-pressure target (grid%pb).

Key faithful steps (the implementer reproduces vert_interp / integ_moist /
p_dts / p_dry / lagrange logic — module_initialize_real.F:5590 vert_interp,
:6967 integ_moist, :6764 p_dts, :6710 p_dry, :7163 rh_to_mxrat):
  * find_p_top from the metgrid PRES (:1279) -> grid%p_top (capped at config).
  * integ_moist: column-integrate qv to get dry-pressure column pd_gc + intq (:1450).
  * p_dts -> MU0 (full dry column mass) from intq/psfc/p_top (:1475).
  * target dry pressure pb on half levels: p_dry(mu0,znw,p_top,...) (:1701).
  * vert_interp each of {ght->ph0, qv, t/theta, p, u, v, + hydrometeors}
    from pd_gc to pb with the configured interp_type/lagrange_order/extrap.
  * The vertical index order FLIPS metgrid (sfc=0..top) to model order here.

Output: a DynamicsInit with u/v/w/theta/qv/mu/mu0 and the *interpolated* p/ph
seeds; the FINAL hydrostatically-balanced p/ph/al/alt/p_hyd are produced by
hydrostatic.py (which takes this DynamicsInit + the BaseStateColumns). This lane
delivers the pre-balance interpolated state; hydrostatic.py finishes it.

Oracle: indirect — its product feeds hydrostatic.py, whose output (T/U/V/QVAPOR/
P/PH/MU) is compared to wrfinput. The lane's own unit test should check a single
column vert_interp against a hand-computed Lagrange result.

FILE OWNERSHIP: S1 exclusive.
"""

from __future__ import annotations

import numpy as np

from gpuwrf.init.real_init.types import (
    CP,
    G,
    P1000MB,
    R_D,
    RD_OVER_CP,
    T0,
    DynamicsInit,
    RealInitConfig,
    VerticalCoord1D,
)
from gpuwrf.init.metgrid_schema import MetEmArtifact


def vertical_interpolate(
    config: RealInitConfig,
    vcoord: VerticalCoord1D,
    metem: MetEmArtifact,
) -> DynamicsInit:
    """Interpolates met_em atmospheric columns onto the model eta column.

    Consumes the FROZEN v0.3.0 ``MetEmArtifact`` (the metgrid-equivalent input).
    Returns a DynamicsInit whose p/ph are interpolation seeds; the final
    balanced p/ph/al/alt/p_hyd come from :func:`hydrostatic.balance`.
    """

    arrays = metem.arrays
    required = ("TT", "UU", "VV", "GHT", "SPECHUMD", "PRES", "PSFC", "HGT_M")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"MetEmArtifact missing S1 fields: {missing}")

    temp_gc = _arr(arrays["TT"])
    u_gc = _arr(arrays["UU"])
    v_gc = _arr(arrays["VV"])
    ght_gc = _arr(arrays["GHT"])
    pres_gc = _arr(arrays["PRES"])
    psfc_gc = _arr(arrays["PSFC"])
    terrain = _arr(arrays["HGT_M"])

    sh_gc = _arr(arrays["SPECHUMD"])
    qv_gc = _specific_humidity_to_mixing_ratio(sh_gc, pres_gc)
    rh_gc = _relative_humidity_from_mixing_ratio(qv_gc, temp_gc, pres_gc)
    pd_gc, intq = _integ_moist(qv_gc, pres_gc, temp_gc, ght_gc)
    psfc = _surface_pressure(arrays, pres_gc, ght_gc, terrain, psfc_gc)
    mu0 = psfc - intq - float(config.p_top_pa)

    p_dry_half = _p_dry(mu0, vcoord, config, full_levels=False)
    p_dry_full = _p_dry(mu0, vcoord, config, full_levels=True)

    rh = _vert_interp(
        rh_gc,
        pd_gc,
        p_dry_half,
        var_type="Q",
        interp_type=2,
        lagrange_order=1,
        extrap_type=2,
        force_sfc_in_vinterp=1,
    )
    temp = _vert_interp(
        temp_gc,
        pd_gc,
        p_dry_half,
        var_type="T",
        interp_type=2,
        lagrange_order=1,
        extrap_type=2,
        force_sfc_in_vinterp=1,
    )
    p_seed_full = _vert_interp(
        pres_gc,
        pd_gc,
        p_dry_half,
        var_type="T",
        interp_type=1,
        lagrange_order=1,
        extrap_type=2,
        force_sfc_in_vinterp=1,
    )
    qv = _mixing_ratio_from_rh_liquid(rh, temp, p_seed_full)
    theta = temp * (P1000MB / p_seed_full) ** RD_OVER_CP - T0

    u = _vert_interp(
        u_gc,
        pd_gc,
        p_dry_half,
        var_type="U",
        interp_type=2,
        lagrange_order=1,
        extrap_type=2,
        force_sfc_in_vinterp=1,
    )
    v = _vert_interp(
        v_gc,
        pd_gc,
        p_dry_half,
        var_type="V",
        interp_type=2,
        lagrange_order=1,
        extrap_type=2,
        force_sfc_in_vinterp=1,
    )

    # WRF interpolates GHT to full eta levels before the later hydrostatic
    # branch overwrites PH. Preserve the seed as total geopotential here.
    ght_for_interp = ght_gc.copy()
    pd_for_ght = pd_gc.copy()
    if "PMSL" in arrays:
        pd_for_ght[0] = _arr(arrays["PMSL"]) - (pres_gc[0] - pd_gc[0])
        ght_for_interp[0] = 0.0
    ght_seed = _vert_interp(
        ght_for_interp,
        pd_for_ght,
        p_dry_full,
        var_type="Z",
        interp_type=2,
        lagrange_order=1,
        extrap_type=1,
        force_sfc_in_vinterp=0,
    )

    ny, nx = terrain.shape
    zeros_mass = np.zeros((config.nz, ny, nx), dtype=np.float64)
    return DynamicsInit(
        u=u,
        v=v,
        w=np.zeros((config.nz + 1, ny, nx), dtype=np.float64),
        theta=theta,
        qv=qv,
        mu=np.zeros((ny, nx), dtype=np.float64),
        mu0=mu0,
        p=p_seed_full,
        ph=ght_seed * G,
        al=zeros_mass.copy(),
        alt=zeros_mass.copy(),
        p_hyd=p_seed_full.copy(),
    )


def _arr(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _specific_humidity_to_mixing_ratio(sh_gc: np.ndarray, pres_gc: np.ndarray) -> np.ndarray:
    sh = sh_gc.copy()
    if sh.shape[0] < 2:
        raise ValueError("SPECHUMD must include surface plus isobaric levels")
    # real.exe replaces a missing/invalid surface specific humidity with the
    # closest pressure level before converting to mixing ratio.
    if np.nanmin(sh[0]) < 1.0e-6:
        nearest = 1 if np.nanmin(pres_gc[-1]) < np.nanmin(pres_gc[1]) else pres_gc.shape[0] - 1
        sh[0] = sh[nearest]
    return sh / (1.0 - sh)


def _relative_humidity_from_mixing_ratio(
    qv: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    sat_vap_pres_mb = 0.6112 * 10.0 * np.exp(
        17.67 * (temperature - 273.15) / (temperature - 29.65)
    )
    vap_pres_mb = qv * pressure / 100.0 / (qv + 0.622)
    return np.where(sat_vap_pres_mb > 0.0, (vap_pres_mb / sat_vap_pres_mb) * 100.0, 0.0)


def _mixing_ratio_from_rh_liquid(
    rh: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    rh_bounded = np.minimum(np.maximum(rh, 0.0), 100.0)
    es = (
        0.01
        * rh_bounded
        * 0.6112
        * 10.0
        * np.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))
    )
    q = np.where(
        es >= pressure / 100.0,
        1.0e-6,
        np.maximum(0.622 * es / (pressure / 100.0 - es), 1.0e-6),
    )
    q = np.where((pressure < 10000.0) & (q > 1.0e-5), 3.0e-6, q)
    q = np.where((pressure < 110000.0) & (q < 1.0e-6), 1.0e-6, q)
    return q


def _surface_pressure(
    arrays: dict[str, np.ndarray],
    pres_gc: np.ndarray,
    ght_gc: np.ndarray,
    terrain: np.ndarray,
    psfc_gc: np.ndarray,
) -> np.ndarray:
    if "PMSL" not in arrays or "SOILHGT" not in arrays:
        return psfc_gc.copy()
    return _sfcprs3(ght_gc, pres_gc, terrain, _arr(arrays["PMSL"]))


def _sfcprs3(height: np.ndarray, pressure: np.ndarray, terrain: np.ndarray, slp: np.ndarray) -> np.ndarray:
    nlev, ny, nx = pressure.shape
    out = np.empty((ny, nx), dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            ter = terrain[j, i]
            if ter < 50.0:
                out[j, i] = slp[j, i] + (
                    (pressure[1, j, i] - pressure[2, j, i])
                    / (height[1, j, i] - height[2, j, i])
                    * ter
                )
                continue

            found = False
            for k in range(1, nlev - 2):
                if height[k, j, i] <= ter and height[k + 1, j, i] > ter:
                    out[j, i] = _log_interp_z(
                        height[k, j, i],
                        height[k + 1, j, i],
                        ter,
                        pressure[k, j, i],
                        pressure[k + 1, j, i],
                    )
                    found = True
                    break
            if found:
                continue

            if slp[j, i] >= pressure[1, j, i]:
                out[j, i] = _log_interp_z(0.0, height[2, j, i], ter, slp[j, i], pressure[2, j, i])
                continue

            for k in range(1, nlev - 3):
                if slp[j, i] >= pressure[k + 1, j, i] and slp[j, i] < pressure[k, j, i]:
                    out[j, i] = _log_interp_z(
                        0.0,
                        height[k + 1, j, i],
                        ter,
                        slp[j, i],
                        pressure[k + 1, j, i],
                    )
                    found = True
                    break
            if not found:
                raise ValueError(f"could not compute surface pressure at i={i}, j={j}")
    return out


def _log_interp_z(zl: float, zu: float, zm: float, pl: float, pu: float) -> float:
    return float(np.exp((np.log(pl) * (zm - zu) + np.log(pu) * (zl - zm)) / (zl - zu)))


def _integ_moist(
    q_in: np.ndarray,
    p_in: np.ndarray,
    t_in: np.ndarray,
    ght_in: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nlev, ny, nx = p_in.shape
    pd_out = np.empty_like(p_in, dtype=np.float64)
    intq = np.zeros((ny, nx), dtype=np.float64)
    upside_down = bool(p_in[1, 0, 0] < p_in[-1, 0, 0])

    for j in range(ny):
        for i in range(nx):
            if upside_down:
                p = np.concatenate(([p_in[0, j, i]], p_in[:0:-1, j, i]))
                t = np.concatenate(([t_in[0, j, i]], t_in[:0:-1, j, i]))
                q = np.concatenate(([q_in[0, j, i]], q_in[:0:-1, j, i]))
                ght = np.concatenate(([ght_in[0, j, i]], ght_in[:0:-1, j, i]))
            else:
                p = p_in[:, j, i]
                t = t_in[:, j, i]
                q = q_in[:, j, i]
                ght = ght_in[:, j, i]

            psfc = p[0]
            level_above = -1
            if p[1] < psfc:
                level_above = 1
            else:
                for k in range(1, nlev - 1):
                    if (p[k] - psfc >= 0.0) and (p[k + 1] - psfc < 0.0):
                        level_above = k + 1
                        break
            if level_above < 0:
                raise ValueError(f"could not find level above surface at i={i}, j={j}")

            pd = np.empty(nlev, dtype=np.float64)
            column_intq = 0.0
            pd[-1] = p[-1]
            for k in range(nlev - 2, level_above - 1, -1):
                rhobar = 0.5 * (p[k] / (R_D * t[k]) + p[k + 1] / (R_D * t[k + 1]))
                qbar = 0.5 * (q[k] + q[k + 1])
                dz = ght[k + 1] - ght[k]
                column_intq += G * qbar * rhobar / (1.0 + qbar) * dz
                pd[k] = p[k] - column_intq

            if (
                p[level_above - 1] - psfc >= 0.0
                and p[level_above] - psfc < 0.0
                and level_above > 0
            ):
                rhobar = 0.5 * (psfc / (R_D * t[0]) + p[level_above] / (R_D * t[level_above]))
                qbar = 0.5 * (q[0] + q[level_above])
                dz = ght[level_above] - ght[0]
                if dz > 0.1:
                    column_intq += G * qbar * rhobar / (1.0 + qbar) * dz
                for k in range(level_above - 1, 0, -1):
                    pd[k] = p[k] - column_intq
            pd[0] = psfc - column_intq

            intq[j, i] = column_intq
            if upside_down:
                pd_out[0, j, i] = pd[0]
                pd_out[1:, j, i] = pd[:0:-1]
            else:
                pd_out[:, j, i] = pd
    return pd_out, intq


def _p_dry(
    mu0: np.ndarray,
    vcoord: VerticalCoord1D,
    config: RealInitConfig,
    *,
    full_levels: bool,
) -> np.ndarray:
    if full_levels:
        return (
            vcoord.c3f[:, None, None] * mu0[None, :, :]
            + vcoord.c4f[:, None, None]
            + float(config.p_top_pa)
        )
    return (
        vcoord.c3h[:, None, None] * mu0[None, :, :]
        + vcoord.c4h[:, None, None]
        + float(config.p_top_pa)
    )


def _vert_interp(
    field: np.ndarray,
    pressure_mass: np.ndarray,
    target_pressure_mass: np.ndarray,
    *,
    var_type: str,
    interp_type: int,
    lagrange_order: int,
    extrap_type: int,
    force_sfc_in_vinterp: int,
) -> np.ndarray:
    pressure, target = _stagger_pressures(pressure_mass, target_pressure_mass, var_type)
    if field.shape[1:] != pressure.shape[1:]:
        raise ValueError(
            f"field shape {field.shape} incompatible with {var_type} pressure {pressure.shape}"
        )

    _, ny, nx = field.shape
    ntarget = target.shape[0]
    out = np.empty((ntarget, ny, nx), dtype=np.float64)
    flip = bool(pressure[1, 0, 0] < pressure[-1, 0, 0])
    for j in range(ny):
        for i in range(nx):
            out[:, j, i] = _interp_column(
                field[:, j, i],
                pressure[:, j, i],
                target[:, j, i],
                var_type=var_type,
                interp_type=interp_type,
                lagrange_order=lagrange_order,
                extrap_type=extrap_type,
                force_sfc_in_vinterp=force_sfc_in_vinterp,
                flip=flip,
            )
    return out


def _stagger_pressures(
    pressure_mass: np.ndarray,
    target_pressure_mass: np.ndarray,
    var_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    if var_type == "U":
        p = np.empty((pressure_mass.shape[0], pressure_mass.shape[1], pressure_mass.shape[2] + 1))
        pn = np.empty((target_pressure_mass.shape[0], target_pressure_mass.shape[1], target_pressure_mass.shape[2] + 1))
        p[:, :, 0] = pressure_mass[:, :, 0]
        p[:, :, 1:-1] = 0.5 * (pressure_mass[:, :, 1:] + pressure_mass[:, :, :-1])
        p[:, :, -1] = pressure_mass[:, :, -1]
        pn[:, :, 0] = target_pressure_mass[:, :, 0]
        pn[:, :, 1:-1] = 0.5 * (target_pressure_mass[:, :, 1:] + target_pressure_mass[:, :, :-1])
        pn[:, :, -1] = target_pressure_mass[:, :, -1]
        return p, pn
    if var_type == "V":
        p = np.empty((pressure_mass.shape[0], pressure_mass.shape[1] + 1, pressure_mass.shape[2]))
        pn = np.empty((target_pressure_mass.shape[0], target_pressure_mass.shape[1] + 1, target_pressure_mass.shape[2]))
        p[:, 0, :] = pressure_mass[:, 0, :]
        p[:, 1:-1, :] = 0.5 * (pressure_mass[:, 1:, :] + pressure_mass[:, :-1, :])
        p[:, -1, :] = pressure_mass[:, -1, :]
        pn[:, 0, :] = target_pressure_mass[:, 0, :]
        pn[:, 1:-1, :] = 0.5 * (target_pressure_mass[:, 1:, :] + target_pressure_mass[:, :-1, :])
        pn[:, -1, :] = target_pressure_mass[:, -1, :]
        return p, pn
    return pressure_mass, target_pressure_mass


def _interp_column(
    field: np.ndarray,
    pressure: np.ndarray,
    target_pressure: np.ndarray,
    *,
    var_type: str,
    interp_type: int,
    lagrange_order: int,
    extrap_type: int,
    force_sfc_in_vinterp: int,
    flip: bool,
) -> np.ndarray:
    p = pressure.copy()
    f = field.copy()
    if flip:
        p[1:] = p[:0:-1]
        f[1:] = f[:0:-1]

    ko_above = None
    for ko in range(1, len(p)):
        if p[0] > p[ko]:
            ko_above = ko
            break
    if ko_above is None:
        raise ValueError("could not identify first pressure level above surface")

    ordered_p, ordered_f = _ordered_column(
        p,
        f,
        target_pressure,
        ko_above,
        force_sfc_in_vinterp=force_sfc_in_vinterp,
        zap_close_levels=500.0,
    )
    x = np.log(ordered_p) if interp_type != 1 else ordered_p
    target_x = np.log(target_pressure) if interp_type != 1 else target_pressure
    return _lagrange_setup(
        x,
        ordered_f,
        target_x,
        var_type=var_type,
        interp_type=interp_type,
        lagrange_order=lagrange_order,
        extrap_type=extrap_type,
    )


def _ordered_column(
    p: np.ndarray,
    f: np.ndarray,
    target_pressure: np.ndarray,
    ko_above: int,
    *,
    force_sfc_in_vinterp: int,
    zap_close_levels: float,
) -> tuple[np.ndarray, np.ndarray]:
    ordered_p: list[float] = []
    ordered_f: list[float] = []
    zap_below = 0

    if ko_above > 1:
        for ko in range(1, ko_above):
            ordered_p.append(float(p[ko]))
            ordered_f.append(float(f[ko]))
        if ordered_p and ordered_p[-1] - p[0] < zap_close_levels:
            ordered_p.pop()
            ordered_f.pop()
            zap_below = 1
        ordered_p.append(float(p[0]))
        ordered_f.append(float(f[0]))

        knext = ko_above
        if force_sfc_in_vinterp > 0:
            target_idx = force_sfc_in_vinterp - 1
            for ko in range(ko_above, len(p)):
                if p[ko] <= target_pressure[target_idx]:
                    knext = ko
                    break
        kst = knext + 1 if ordered_p[-1] - p[knext] < zap_close_levels else knext
        for ko in range(kst, len(p)):
            ordered_p.append(float(p[ko]))
            ordered_f.append(float(f[ko]))
    else:
        ordered_p.append(float(p[0]))
        ordered_f.append(float(f[0]))
        knext = 1
        if force_sfc_in_vinterp > 0:
            target_idx = force_sfc_in_vinterp - 1
            for ko in range(1, len(p)):
                if p[ko] <= target_pressure[target_idx]:
                    knext = ko
                    break
        for ko in range(knext, len(p)):
            if ordered_p[-1] - p[ko] < zap_close_levels and ko < len(p) - 1:
                continue
            ordered_p.append(float(p[ko]))
            ordered_f.append(float(f[ko]))

    del zap_below  # kept explicit to mirror the WRF branch naming.
    return np.asarray(ordered_p, dtype=np.float64), np.asarray(ordered_f, dtype=np.float64)


def _lagrange_setup(
    all_x: np.ndarray,
    all_y: np.ndarray,
    target_x: np.ndarray,
    *,
    var_type: str,
    interp_type: int,
    lagrange_order: int,
    extrap_type: int,
) -> np.ndarray:
    if all_x.size < lagrange_order + 1:
        raise ValueError("interpolating order is too large for input pressure column")
    if lagrange_order < 1:
        raise ValueError("lagrange_order must be >= 1")

    out = np.empty_like(target_x, dtype=np.float64)
    vboundb = 4
    vboundt = 0
    for target_loop, tx in enumerate(target_x):
        loc_left = None
        for loop in range(all_x.size - 1):
            if (tx - all_x[loop]) * (tx - all_x[loop + 1]) <= 0.0:
                loc_left = loop
                break

        if loc_left is None:
            if tx > all_x[0]:
                out[target_loop] = _extrapolate_below_ground(
                    all_x,
                    all_y,
                    tx,
                    var_type=var_type,
                    interp_type=interp_type,
                    extrap_type=extrap_type,
                )
                continue
            raise ValueError("could not find trapping pressure levels")

        if lagrange_order == 1:
            out[target_loop] = _lagrange_interp(
                all_x[loc_left : loc_left + 2], all_y[loc_left : loc_left + 2], tx
            )
            continue

        if lagrange_order % 2 != 0:
            half = (lagrange_order + 1) // 2 - 1
            start = loc_left - half
            end = start + lagrange_order + 1
            if start < 0 or end > all_x.size:
                raise ValueError("odd-order interpolation stencil outside column")
            out[target_loop] = _lagrange_interp(all_x[start:end], all_y[start:end], tx)
            continue

        if (
            target_loop + 1 >= 1 + vboundb
            and target_loop + 1 <= target_x.size - vboundt
        ):
            first_ok = loc_left + lagrange_order + 1 <= all_x.size
            second_ok = loc_left - 1 >= 0 and loc_left + lagrange_order <= all_x.size
            if first_ok and second_ok:
                y1 = _lagrange_interp(
                    all_x[loc_left : loc_left + lagrange_order + 1],
                    all_y[loc_left : loc_left + lagrange_order + 1],
                    tx,
                )
                y2 = _lagrange_interp(
                    all_x[loc_left - 1 : loc_left + lagrange_order],
                    all_y[loc_left - 1 : loc_left + lagrange_order],
                    tx,
                )
                out[target_loop] = 0.5 * (y1 + y2)
            elif first_ok:
                out[target_loop] = _lagrange_interp(
                    all_x[loc_left : loc_left + lagrange_order + 1],
                    all_y[loc_left : loc_left + lagrange_order + 1],
                    tx,
                )
            elif second_ok:
                out[target_loop] = _lagrange_interp(
                    all_x[loc_left - 1 : loc_left + lagrange_order],
                    all_y[loc_left - 1 : loc_left + lagrange_order],
                    tx,
                )
            else:
                raise ValueError("quadratic interpolation stencil outside column")
        else:
            out[target_loop] = _lagrange_interp(
                all_x[loc_left : loc_left + 2], all_y[loc_left : loc_left + 2], tx
            )
    return out


def _extrapolate_below_ground(
    all_x: np.ndarray,
    all_y: np.ndarray,
    target_x: float,
    *,
    var_type: str,
    interp_type: int,
    extrap_type: int,
) -> float:
    if interp_type == 1:
        all_x_full = all_x
        target_x_full = target_x
    else:
        all_x_full = np.exp(all_x)
        target_x_full = float(np.exp(target_x))

    if extrap_type == 1 and var_type == "T":
        temp_1 = all_y[0] * (all_x_full[0] / P1000MB) ** (R_D / CP)
        return float(temp_1 * (P1000MB / target_x_full) ** (R_D / CP))
    if extrap_type == 2 and var_type == "T":
        crc_const1 = 11880.516
        crc_const2 = 0.1902632
        crc_const3 = 0.0065
        depth_of_extrap_in_p = target_x_full - all_x_full[0]
        avg_of_extrap_p = 0.5 * (target_x_full + all_x_full[0])
        temp_start = all_y[0] * (all_x_full[0] / P1000MB) ** (R_D / CP)
        dhdp = crc_const1 * crc_const2 * (avg_of_extrap_p / 100.0) ** (crc_const2 - 1.0)
        dh = dhdp * (depth_of_extrap_in_p / 100.0)
        dt = dh * crc_const3
        return float((temp_start + dt) * (P1000MB / target_x_full) ** (R_D / CP))
    if extrap_type == 3 and var_type == "T":
        return float(all_y[0])
    if extrap_type == 1:
        return float(
            (
                all_y[1] * (target_x - all_x[2])
                + all_y[2] * (all_x[1] - target_x)
            )
            / (all_x[1] - all_x[2])
        )
    if extrap_type == 2:
        return float(all_y[0])
    raise ValueError("unsupported extrapolation option")


def _lagrange_interp(x: np.ndarray, y: np.ndarray, target_x: float) -> float:
    px = 0.0
    for i in range(x.size):
        numer = 1.0
        denom = 1.0
        for k in range(x.size):
            if k == i:
                continue
            numer *= target_x - x[k]
            denom *= x[i] - x[k]
        if denom != 0.0:
            px += y[i] * numer / denom
    return float(px)
