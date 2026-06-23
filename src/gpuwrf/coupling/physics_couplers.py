"""Type-checked adapters from the coupled State pytree to M5 column kernels.

The persistent state layout stays ADR-002 SoA. These wrappers only create
transient column views with vertical as the last axis because the M5 Thompson,
MYNN, and RRTMG kernels are column-batched in that convention.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import NamedTuple

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State
from gpuwrf.physics.mynn_pbl import (
    MynnPBLColumnState,
    _MYNN_COLUMN_TILE_COLS,
    _MYNN_COLUMN_TILING,
    _pad_columns_leaf,
    _scatter_columns_leaf,
    _slice_columns_leaf,
    mynn_coldstart_init_columns,
    step_mynn_pbl_column,
    step_mynn_pbl_column_with_pblh,
)
from gpuwrf.physics.mynn_sgs_cloud import sgs_cloud_enabled
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
from gpuwrf.physics.gwd_gwdo import GWDOColumnState, GWDOStatics, gwdo_columns
from gpuwrf.physics.surface_layer import surface_layer, surface_layer_with_diagnostics
from gpuwrf.physics.ra_sw_dudhia import (
    DudhiaSWColumnState,
    solve_dudhia_sw_column,
)
from gpuwrf.physics.ra_sw_gsfc import (
    GsfcSWColumnState,
    solve_gsfc_sw_column,
)
from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState, solve_rrtmg_lw_column
from gpuwrf.physics.ra_lw_rrtm import RRTMLWColumnState
from gpuwrf.physics.ra_lw_rrtm_jax import solve_rrtm_lw_column_jax
from gpuwrf.physics.ra_lw_hs import (
    HeldSuarezColumnState,
    solve_held_suarez_column,
)
from gpuwrf.physics.rrtmg_sw import (
    RRTMGSWColumnState,
    RRTMGSWTopographyState,
    solve_rrtmg_sw_column,
)
from gpuwrf.physics.thompson_column import (
    ThompsonColumnState,
    density_from_pressure_temperature,
    step_thompson_column,
    step_thompson_column_with_precip,
)
from gpuwrf.physics.thompson_aero_column import (
    NA_CCN0,
    NA_CCN1,
    NA_IN0,
    NA_IN1,
    ThompsonAeroColumnState,
    apply_surface_aerosol_emission,
    step_thompson_aero_column_with_precip,
)


P0_PA = 100000.0
R_D_OVER_CP = 287.0 / 1004.0
GRAVITY_M_S2 = 9.80665
WRF_PHYSICS_G_M_S2 = 9.81
WRF_RV_OVER_RD = 461.6 / 287.0
DEG_TO_RAD = 3.141592653589793 / 180.0
MINUTES_PER_DAY = 1440.0

# RRTMG-SW built-in solar constant (W/m^2). WRF `module_ra_rrtmg_sw.F:115`
# `rrsw_scon = 1.36822e3`; the per-band solar source `sfluxzen` integrates to
# this value, so WRF rescales every band by `solvar(ib) = scon / rrsw_scon`
# (`:9869`). The GPU kernel applies the identical multiplier via
# `RRTMGSWColumnState.solar_source_scale` (rrtmg_sw.py).
RRSW_SCON = 1368.22

# MODIFIED_IGBP_MODIS_NOAH LANDUSE.TBL summer values:
# columns ALBD (%), SFEM. Index 0 is a water fallback for legacy zero LU_INDEX
# analytic states; WRF category indices are used directly for entries 1..61.
_MODIS_NOAH_ALBEDO = jnp.asarray(
    (
        0.08,
        0.12,
        0.12,
        0.14,
        0.16,
        0.13,
        0.22,
        0.20,
        0.22,
        0.20,
        0.19,
        0.14,
        0.17,
        0.15,
        0.18,
        0.55,
        0.25,
        0.08,
        0.15,
        0.15,
        0.25,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
    ),
    dtype=jnp.float64,
)
_MODIS_NOAH_EMISSIVITY = jnp.asarray(
    (
        0.98,
        0.95,
        0.95,
        0.94,
        0.93,
        0.97,
        0.93,
        0.95,
        0.93,
        0.92,
        0.96,
        0.95,
        0.985,
        0.88,
        0.98,
        0.95,
        0.90,
        0.98,
        0.93,
        0.92,
        0.90,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.97,
        0.97,
        0.97,
        0.97,
        0.97,
        0.97,
        0.97,
        0.97,
        0.97,
        0.97,
        0.97,
    ),
    dtype=jnp.float64,
)

# Existing production callers have no model-time argument yet. This fallback is
# deterministic and keeps legacy no-time call sites in the same SW magnitude
# range until a future interface sprint threads model time through the scan.
_LEGACY_RRTMG_TIME_UTC = datetime(2000, 5, 21, 17, 30, tzinfo=timezone.utc)


class _SurfaceColumnState(NamedTuple):
    """Column-oriented view consumed by `surface_layer.surface_layer`."""

    u: object
    v: object
    theta: object
    qv: object
    p: object
    dz: object
    t_skin: object
    soil_moisture: object
    xland: object
    lakemask: object
    mavail: object
    roughness_m: object
    ustar: object
    t_air: object = None
    psfc: object = None
    rho: object = None
    # RETIRED (v0.9.0): the WRF-faithful MYNN-SL 2-m T2 diagnostic is
    # ``THGB + DTG*PSIT2/PSIT``; over LAND real WRF overwrites it with the Noah-MP LSM
    # value ``T2 = FVEG*T2MV + (1-FVEG)*T2MB``. That overwrite is now done FAITHFULLY
    # from the genuine Noah-MP T2MV/T2MB diagnostics in the coupler
    # (noahmp_surface_hook.overlay_noahmp_land_diagnostics; proofs/v090/noahmp_t2mb_parity.json),
    # so the earlier opt-in empirical bare-ground stand-in has been removed and this
    # surface-layer module is the pure module_sf_mynn.F 2-m diagnostic. ``lsm_t2_diag``
    # is now INERT (kept only so legacy constructors do not break) — surface_layer.py no
    # longer reads it.
    lsm_t2_diag: bool = False


class ThompsonTendencySideChannel(NamedTuple):
    """Microphysical species-tendency oracle emitted by the Thompson adapter.

    Arrays are in State orientation `(z, y, x)` except `column_water_tendency`,
    which is the vertical-mean total-water tendency on `(y, x)`.
    """

    qv: object
    qc: object
    qr: object
    qi: object
    qs: object
    qg: object
    column_water_tendency: object
    precip_out_tendency: object


class SurfaceMynnDiagnostics(NamedTuple):
    """B2 operational surface/PBL diagnostics (coupler_interface.md §4).

    Side-channel only — NOT prognostic State leaves. All fields are mass-point
    2-D ``(ny, nx)``. ``hfx``/``lh`` in W m^-2 (upward positive); ``t2`` in K;
    ``u10``/``v10`` in m s^-1; ``pblh`` in m; ``ustar`` in m s^-1. ``hfx``/``lh``
    are the WRF-form W m^-2 fluxes from the revised surface layer; the kinematic
    ``theta_flux``/``qv_flux`` written to State are HFX/(rho*cpm) and QFX/rho.
    """

    hfx: object
    lh: object
    pblh: object
    t2: object
    u10: object
    v10: object
    ustar: object


class MynnPBLSourceLeaves(NamedTuple):
    """MYNN adapter output plus raw WRF PBL source leaves.

    ``rublten``/``rvblten`` are the raw A-grid (mass-point) momentum source
    rates ``(u_out - u_in)/dt`` exactly as WRF's MYNN driver writes RUBLTEN /
    RVBLTEN before ``phy_tend`` mass-couples them (module_em.F:2381) and
    ``update_phy_ten`` face-averages them into ``ru_tendf``/``rv_tendf``
    (phys/module_physics_addtendc.F add_a2c_u/add_a2c_v).
    """

    state: State
    rthblten: jax.Array
    rqvblten: jax.Array
    rublten: jax.Array
    rvblten: jax.Array


class RRTMGRadiationDiagnostics(NamedTuple):
    """Surface radiation diagnostics emitted by the RRTMG adapter inputs."""

    surface_albedo: object
    surface_emissivity: object
    coszen: object
    swdown: object
    swnorm: object
    swup: object
    swup_topographic: object
    glw: object
    glw_up: object
    topographic_correction_factor: object
    shadow_mask: object
    # B1 (v0.12.0) top-of-atmosphere all-sky flux slices, mass-point (ny, nx).
    # These are the ``[..., -1]`` (model top) interface fluxes the RRTMG SW/LW
    # column solvers already compute; surfaced here for the wrfout TOA flux vars
    # (SWDNT/SWUPT/LWDNT/LWUPT/OLR).
    sw_toa_down: object
    sw_toa_up: object
    lw_toa_down: object
    lw_toa_up: object
    # v0.13.0 clear-sky (cloud-free) SW/LW fluxes from the WRF second clear-sky
    # radiative-transfer pass (RRTMG `pbbcd/pbbcu`, `totdclfl/totuclfl`), surfaced
    # for the WRF ``...C`` wrfout vars.  ``None`` unless ``with_clear_sky=True`` is
    # threaded into the diagnostics call.  Top == model-top interface, bot ==
    # surface: SWUPTC/SWDNTC/SWUPBC/SWDNBC + LWUPTC/LWDNTC/LWUPBC/LWDNBC.
    sw_clear_toa_down: object = None
    sw_clear_toa_up: object = None
    sw_clear_sfc_down: object = None
    sw_clear_sfc_up: object = None
    lw_clear_toa_down: object = None
    lw_clear_toa_up: object = None
    lw_clear_sfc_down: object = None
    lw_clear_sfc_up: object = None


class SolarGeometry(NamedTuple):
    """WRF radiation-driver solar geometry shared by RRTMG and topo shading."""

    coszen: object
    declination_rad: object
    hour_angle_rad: object


class RRTMGRadiationStatic(NamedTuple):
    """Per-run WRF radiation grid fields on mass points ``(ny, nx)``."""

    xlat_deg: object
    xlong_deg: object
    terrain_height_m: object
    slope_rad: object
    slope_azimuth_rad: object
    sina: object
    cosa: object


def _to_columns(field):
    """Moves state vertical axis from leading `(z, y, x)` to trailing `(..., z)`."""

    return jnp.moveaxis(field, 0, -1)


def _from_columns(field):
    """Moves column-kernel vertical axis from trailing `(..., z)` to leading z."""

    return jnp.moveaxis(field, -1, 0)


def _u_mass(state: State):
    """Collocates x-face wind to mass points."""

    return 0.5 * (state.u[:, :, :-1] + state.u[:, :, 1:])


def _v_mass(state: State):
    """Collocates y-face wind to mass points."""

    return 0.5 * (state.v[:, :-1, :] + state.v[:, 1:, :])


def _w_mass(state: State):
    """Collocates vertical-face wind to mass points."""

    return 0.5 * (state.w[:-1, :, :] + state.w[1:, :, :])


def _mass_to_u_face(field):
    """Reconstruct x-face wind from a mass-point wind field, non-periodic edges.

    Gate-1 decision #4: the prior periodic ``jnp.roll`` reconstruction wrapped the
    last interior column onto the first u-face, which is wrong for the Canary
    domain (NOT periodic). Interior faces are the centred average of the two
    adjacent mass cells; the two domain-edge faces (x=0 and x=nx) are filled by
    one-sided extrapolation from the nearest interior mass cell (zero-gradient at
    the wall), so no cross-domain wrap occurs.

    B4 SEAM: this zero-gradient edge fill is a placeholder. When B4 lateral
    boundaries are wired, the two edge u-faces inside the relaxation/specified
    zone are OWNED by ``apply_lateral_boundaries`` (it runs AFTER mynn_adapter in
    the bundle, operational_mode.py:1453) and will overwrite them with the
    wrfbdy-driven values. MYNN must not assume periodicity here; it only provides
    a finite interior-consistent guess that B4 then corrects at the edge. See
    coupler_interface.md §6 item 4.

    ``field`` is mass-point ``(nz, ny, nx)``; returns u-face ``(nz, ny, nx+1)``.
    """

    interior = 0.5 * (field[:, :, :-1] + field[:, :, 1:])  # (nz, ny, nx-1)
    left = field[:, :, :1]   # zero-gradient extrapolation to the x=0 wall face
    right = field[:, :, -1:]  # zero-gradient extrapolation to the x=nx wall face
    return jnp.concatenate((left, interior, right), axis=2)


def _mass_to_v_face(field):
    """Reconstruct y-face wind from a mass-point wind field, non-periodic edges.

    Same non-periodic treatment as :func:`_mass_to_u_face` on the y axis (Gate-1
    decision #4). Interior y-faces are centred averages; the y=0 and y=ny wall
    faces use zero-gradient extrapolation. B4 SEAM: the edge v-faces are corrected
    by ``apply_lateral_boundaries`` after MYNN.

    ``field`` is mass-point ``(nz, ny, nx)``; returns v-face ``(nz, ny+1, nx)``.
    """

    interior = 0.5 * (field[:, :-1, :] + field[:, 1:, :])  # (nz, ny-1, nx)
    bottom = field[:, :1, :]
    top = field[:, -1:, :]
    return jnp.concatenate((bottom, interior, top), axis=1)


def _mass_to_w_face(field):
    """Maps mass-point vertical velocity back to rigid vertical faces."""

    interior = 0.5 * (field[:-1, :, :] + field[1:, :, :])
    return jnp.concatenate((field[:1, :, :], interior, field[-1:, :, :]), axis=0)


def _temperature_from_theta(theta, p):
    """Converts potential temperature to temperature for column physics."""

    exner = (jnp.maximum(p, 1.0) / P0_PA) ** R_D_OVER_CP
    return theta.astype(p.dtype) * exner


def _theta_from_temperature(T, p, dtype):
    """Converts temperature back to potential temperature at the coupling boundary."""

    exner = (jnp.maximum(p, 1.0) / P0_PA) ** R_D_OVER_CP
    return (T / jnp.maximum(exner, 1.0e-12)).astype(dtype)


def _field_dtype(field: str):
    """Returns the frozen M6 storage dtype for a State field."""

    return DEFAULT_DTYPES.dtype_for(field)


def _output_dtype(state: State, field: str):
    """Return the dtype the adapter must WRITE for ``field`` on this State.

    fp32-defeat fix (Sprint coupler-fp64 / GPT P0-1): the physics adapters used
    to cast every reassembled field back to the *frozen* ADR-007 perf matrix
    (``_field_dtype``), which pins ``theta``/``qv``/``u``/``v``/hydrometeors to
    fp32.  Under ``force_fp64`` the *state* fields arrive as fp64 (the
    operational precision enforcement upcast them), but that frozen-matrix cast
    silently downcast every physics-coupled field to fp32 *inside* the timestep
    -- so the physics tendencies (and the theta/qv/wind they advance) were
    computed and stored in fp32 even on the "fp64" path; the end-of-step
    re-upcast only re-widened already-truncated values.

    The dtype an adapter writes must therefore track the *live* state field, not
    the frozen matrix: fp64 when ``force_fp64`` has upcast the carry, fp32 in the
    default mixed-precision perf mode.  ``getattr(state, field).dtype`` is exactly
    that contract -- it preserves the existing perf-matrix behaviour byte-for-byte
    (the field is still fp32 there) while keeping force_fp64 truly fp64 through
    Thompson/MYNN/RRTMG.  Equivalent to letting ``state.replace`` default-cast to
    the current dtype, but made explicit so the intermediate math (e.g. the
    ``_theta_from_temperature`` round trip) also stays fp64 rather than being
    truncated before the write.
    """

    return getattr(state, field).dtype


def _coerce_datetime_utc(time_utc) -> datetime:
    """Normalize accepted host-side time values to a timezone-aware UTC datetime."""

    if time_utc is None:
        return _LEGACY_RRTMG_TIME_UTC
    if isinstance(time_utc, datetime):
        value = time_utc
    elif isinstance(time_utc, date):
        value = datetime(time_utc.year, time_utc.month, time_utc.day)
    else:
        text = str(time_utc).strip().replace("Z", "+00:00")
        value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _time_utc_parts(time_utc) -> tuple[float, float]:
    """Return WRF-style Julian day and UTC minutes since midnight."""

    value = _coerce_datetime_utc(time_utc)
    julian = float(value.timetuple().tm_yday)
    minute = (
        float(value.hour) * 60.0
        + float(value.minute)
        + float(value.second) / 60.0
        + float(value.microsecond) / 60_000_000.0
    )
    return julian, minute


def time_utc_clock_base(time_utc):
    """Host-side (``julian``, ``utc_minute``) pair for ``time_utc`` as fp64 arrays.

    This is the v0.20.0 #91 cross-date-cache-hit seam: callers OUTSIDE ``jax.jit``
    (the operational host loop, the M9 snapshot, the daily/nested pipelines) call
    this ONCE per run and thread the resulting pair into the jitted scan as a
    TRACED argument (``clock_base``).  Because the values then enter the compiled
    program as runtime inputs -- not Python-float literals baked at trace time --
    the radiation HLO is IDENTICAL for every forecast date, so the persistent
    compile cache HITS across dates with no config.  The numbers are byte-for-byte
    the same as the old host-extracted ``_time_utc_parts`` floats; only their
    *binding time* (runtime vs trace) changes, so this is numerically inert.

    Returns ``(julian, utc_minute)`` as 0-D ``float64`` ``jnp`` arrays.
    """

    julian, minute = _time_utc_parts(time_utc)
    return (
        jnp.asarray(julian, dtype=jnp.float64),
        jnp.asarray(minute, dtype=jnp.float64),
    )


def _resolve_clock_parts(time_utc, clock_base):
    """Return the ``(julian, utc_minute)`` used by the solar-geometry helpers.

    ``clock_base`` (a TRACED ``(julian, utc_minute)`` pair built outside jit by
    :func:`time_utc_clock_base`) overrides the host extraction so the compiled
    HLO is date-independent.  ``clock_base=None`` keeps the legacy behaviour: the
    host-side Python floats from ``time_utc`` (used by idealized cases, tests, and
    any caller that has not yet threaded the traced base).
    """

    if clock_base is not None:
        julian, utc_minute = clock_base
        return julian, utc_minute
    return _time_utc_parts(time_utc)


def _compute_coszen(lat, lon, time_utc, lead_seconds=0.0, *, clock_base=None):
    """Compute WRF-style cosine solar zenith from lat/lon and the forecast clock.

    Faithful transcription of `module_radiation_driver.F` `radconst`
    (`:3594-3636`, 23.5 deg obliquity + vernal-equinox solar longitude) and
    `calc_coszen` (`:3639-3666`, equation-of-time + hour-angle), verified
    line-for-line against the pristine WRF v4 source.

    Model time is THREADED, not fixed. WRF's `calc_coszen` uses
    `tloctm = gmt + xt24/60 + xlon/15` with `xt24 = mod(xtime,1440) + eot`,
    where `xtime` is *minutes since simulation start* and `gmt` is the start
    hour. We carry the static base instant in `time_utc` (-> Julian day +
    absolute UTC minutes-of-day = `gmt*60 + xtime_at_base`) and add the elapsed
    forecast `lead_seconds`. `lead_seconds` may be a traced JAX value, so the
    diurnal cycle advances correctly *inside* `jax.lax.scan` — no fixed-time
    fallback once a caller threads the step lead through.

    `time_utc=None` is the LEGACY no-clock path (kept only for old call sites
    that never plumbed a clock); it pins a deterministic mid-day base. Callers
    on the operational diurnal path pass the real init instant + lead.
    """

    julian, utc_minute = _resolve_clock_parts(time_utc, clock_base)
    lat_rad = jnp.asarray(lat, dtype=jnp.float64) * DEG_TO_RAD
    lon_deg = jnp.asarray(lon, dtype=jnp.float64)
    lead_minutes = jnp.asarray(lead_seconds, dtype=jnp.float64) / 60.0
    # Absolute UTC minutes-of-(base)-day advanced by elapsed forecast minutes,
    # then rolled into Julian-day for the declination/EOT terms so multi-day
    # forecasts keep the seasonal term correct.
    abs_minute = utc_minute + lead_minutes
    day_advance = jnp.floor(abs_minute / MINUTES_PER_DAY)
    julian_now = julian + day_advance

    obecl = 23.5 * DEG_TO_RAD
    sxlong_day = jnp.where(julian_now >= 80.0, julian_now - 80.0, julian_now + 285.0)
    sxlong = sxlong_day * (360.0 / 365.0) * DEG_TO_RAD
    declin = jnp.arcsin(jnp.sin(obecl) * jnp.sin(sxlong))

    da = 2.0 * jnp.pi * (julian_now - 1.0) / 365.0
    eot = (
        0.000075
        + 0.001868 * jnp.cos(da)
        - 0.032077 * jnp.sin(da)
        - 0.014615 * jnp.cos(2.0 * da)
        - 0.04089 * jnp.sin(2.0 * da)
    ) * 229.18
    xt24 = jnp.mod(abs_minute, MINUTES_PER_DAY) + eot
    local_time_h = xt24 / 60.0 + lon_deg / 15.0
    hour_angle = 15.0 * (local_time_h - 12.0) * DEG_TO_RAD
    coszen = jnp.sin(lat_rad) * jnp.sin(declin) + jnp.cos(lat_rad) * jnp.cos(declin) * jnp.cos(hour_angle)
    return jnp.clip(coszen, -1.0, 1.0)


def _compute_solar_geometry(lat, lon, time_utc, lead_seconds=0.0, *, clock_base=None) -> SolarGeometry:
    """Return the same WRF solar geometry as :func:`_compute_coszen`.

    `declination_rad` and `hour_angle_rad` are also required by WRF's
    topographic-shadow and slope-radiation paths.
    """

    julian, utc_minute = _resolve_clock_parts(time_utc, clock_base)
    lat_rad = jnp.asarray(lat, dtype=jnp.float64) * DEG_TO_RAD
    lon_deg = jnp.asarray(lon, dtype=jnp.float64)
    lead_minutes = jnp.asarray(lead_seconds, dtype=jnp.float64) / 60.0
    abs_minute = utc_minute + lead_minutes
    day_advance = jnp.floor(abs_minute / MINUTES_PER_DAY)
    julian_now = julian + day_advance

    obecl = 23.5 * DEG_TO_RAD
    sxlong_day = jnp.where(julian_now >= 80.0, julian_now - 80.0, julian_now + 285.0)
    sxlong = sxlong_day * (360.0 / 365.0) * DEG_TO_RAD
    declin = jnp.arcsin(jnp.sin(obecl) * jnp.sin(sxlong))

    da = 2.0 * jnp.pi * (julian_now - 1.0) / 365.0
    eot = (
        0.000075
        + 0.001868 * jnp.cos(da)
        - 0.032077 * jnp.sin(da)
        - 0.014615 * jnp.cos(2.0 * da)
        - 0.04089 * jnp.sin(2.0 * da)
    ) * 229.18
    xt24 = jnp.mod(abs_minute, MINUTES_PER_DAY) + eot
    local_time_h = xt24 / 60.0 + lon_deg / 15.0
    hour_angle = 15.0 * (local_time_h - 12.0) * DEG_TO_RAD
    coszen = (
        jnp.sin(lat_rad) * jnp.sin(declin)
        + jnp.cos(lat_rad) * jnp.cos(declin) * jnp.cos(hour_angle)
    )
    return SolarGeometry(
        coszen=jnp.clip(coszen, -1.0, 1.0),
        declination_rad=declin,
        hour_angle_rad=hour_angle,
    )


def _fortran_int(value):
    """WRF/Fortran `INT(real)` truncates toward zero."""

    return jnp.trunc(value).astype(jnp.int32)


def wrf_radiation_slope_aspect_from_terrain(
    terrain_height_m,
    *,
    dx_m: float,
    dy_m: float,
    msftx=None,
    msfty=None,
    sina=None,
    cosa=None,
) -> tuple[jax.Array, jax.Array]:
    """Compute WRF `start_em.F` slope and slope azimuth for slope radiation."""

    terrain = jnp.asarray(terrain_height_m, dtype=jnp.float64)
    ny, nx = terrain.shape
    shape = terrain.shape
    dtype = terrain.dtype
    msftx_arr = jnp.ones(shape, dtype=dtype) if msftx is None else jnp.asarray(msftx, dtype=dtype)
    msfty_arr = jnp.ones(shape, dtype=dtype) if msfty is None else jnp.asarray(msfty, dtype=dtype)
    sina_arr = jnp.zeros(shape, dtype=dtype) if sina is None else jnp.asarray(sina, dtype=dtype)
    cosa_arr = jnp.ones(shape, dtype=dtype) if cosa is None else jnp.asarray(cosa, dtype=dtype)

    west = jnp.concatenate((terrain[:, :1], terrain[:, :-1]), axis=1)
    east = jnp.concatenate((terrain[:, 1:], terrain[:, -1:]), axis=1)
    south = jnp.concatenate((terrain[:1, :], terrain[:-1, :]), axis=0)
    north = jnp.concatenate((terrain[1:, :], terrain[-1:, :]), axis=0)
    denom_x = jnp.full((nx,), 2.0, dtype=dtype)
    denom_y = jnp.full((ny,), 2.0, dtype=dtype)
    denom_x = denom_x.at[0].set(1.0).at[-1].set(1.0)
    denom_y = denom_y.at[0].set(1.0).at[-1].set(1.0)

    hx = (east - west) * msftx_arr / (float(dx_m) * denom_x[None, :])
    hy = (north - south) * msfty_arr / (float(dy_m) * denom_y[:, None])
    slope = jnp.arctan(jnp.sqrt(hx * hx + hy * hy))
    flat = slope < 1.0e-4
    raw_azi = jnp.arctan2(hx, hy) + jnp.pi
    rotation = jnp.where(cosa_arr >= 0.0, jnp.arcsin(sina_arr), jnp.pi - jnp.arcsin(sina_arr))
    slope_azimuth = raw_azi - rotation
    return jnp.where(flat, 0.0, slope), jnp.where(flat, 0.0, slope_azimuth)


def build_radiation_static_from_wrf_fields(
    xlat_deg,
    xlong_deg,
    terrain_height_m,
    *,
    dx_m: float,
    dy_m: float,
    msftx=None,
    msfty=None,
    sina=None,
    cosa=None,
) -> RRTMGRadiationStatic:
    """Build the per-run RRTMG terrain-radiation static bundle from WRF fields."""

    terrain = jnp.asarray(terrain_height_m, dtype=jnp.float64)
    shape = terrain.shape
    dtype = terrain.dtype
    slope, slope_azimuth = wrf_radiation_slope_aspect_from_terrain(
        terrain,
        dx_m=float(dx_m),
        dy_m=float(dy_m),
        msftx=msftx,
        msfty=msfty,
        sina=sina,
        cosa=cosa,
    )
    return RRTMGRadiationStatic(
        xlat_deg=jnp.asarray(xlat_deg, dtype=dtype),
        xlong_deg=jnp.asarray(xlong_deg, dtype=dtype),
        terrain_height_m=terrain,
        slope_rad=slope,
        slope_azimuth_rad=slope_azimuth,
        sina=jnp.zeros(shape, dtype=dtype) if sina is None else jnp.asarray(sina, dtype=dtype),
        cosa=jnp.ones(shape, dtype=dtype) if cosa is None else jnp.asarray(cosa, dtype=dtype),
    )


def _cast_radiation_static(static: RRTMGRadiationStatic, dtype) -> RRTMGRadiationStatic:
    return RRTMGRadiationStatic(
        xlat_deg=jnp.asarray(static.xlat_deg, dtype=dtype),
        xlong_deg=jnp.asarray(static.xlong_deg, dtype=dtype),
        terrain_height_m=jnp.asarray(static.terrain_height_m, dtype=dtype),
        slope_rad=jnp.asarray(static.slope_rad, dtype=dtype),
        slope_azimuth_rad=jnp.asarray(static.slope_azimuth_rad, dtype=dtype),
        sina=jnp.asarray(static.sina, dtype=dtype),
        cosa=jnp.asarray(static.cosa, dtype=dtype),
    )


def _radiation_static_for_grid(
    surface_shape: tuple[int, int],
    grid: GridSpec | None,
    radiation_static: RRTMGRadiationStatic | None,
    dtype,
) -> RRTMGRadiationStatic | None:
    if radiation_static is not None:
        return _cast_radiation_static(radiation_static, dtype)
    if grid is None:
        return None
    lat, lon = _grid_lat_lon(surface_shape, grid, dtype)
    return _cast_radiation_static(
        build_radiation_static_from_wrf_fields(
            lat,
            lon,
            grid.terrain_height,
            dx_m=float(grid.projection.dx_m),
            dy_m=float(grid.projection.dy_m),
        ),
        dtype,
    )


def _wrf_topographic_shadow_mask(
    terrain_height_m,
    *,
    latitude_deg,
    coszen,
    declination_rad,
    hour_angle_rad,
    sina,
    cosa,
    dx_m: float,
    dy_m: float,
    shadow_length_m: float,
):
    """WRF `module_radiation_driver.F:toposhad` local terrain ray scan."""

    terrain = jnp.asarray(terrain_height_m, dtype=jnp.float64)
    ny, nx = terrain.shape
    if ny == 0 or nx == 0:
        return jnp.zeros_like(terrain, dtype=jnp.int32)
    max_steps = int(float(shadow_length_m) / float(dx_m) + 1.0)
    if max_steps <= 0:
        return jnp.zeros_like(terrain, dtype=jnp.int32)

    lat = jnp.asarray(latitude_deg, dtype=jnp.float64) * DEG_TO_RAD
    csza = jnp.asarray(coszen, dtype=jnp.float64)
    declin = jnp.asarray(declination_rad, dtype=jnp.float64)
    hrang = jnp.asarray(hour_angle_rad, dtype=jnp.float64)
    sina_arr = jnp.asarray(sina, dtype=jnp.float64)
    cosa_arr = jnp.asarray(cosa, dtype=jnp.float64)
    daylight = csza >= 1.0e-2

    denom = jnp.maximum(jnp.sin(jnp.arccos(jnp.clip(csza, -1.0, 1.0))) * jnp.cos(lat), 1.0e-12)
    argu = jnp.clip((csza * jnp.sin(lat) - jnp.sin(declin)) / denom, -1.0, 1.0)
    acos_argu = jnp.arccos(argu)
    sol_azi = jnp.where(jnp.sin(hrang) >= 0.0, acos_argu, -acos_argu) + jnp.pi
    sol_azi = jnp.where(cosa_arr >= 0.0, sol_azi + jnp.arcsin(sina_arr), sol_azi + jnp.pi - jnp.arcsin(sina_arr))

    yy = jnp.arange(ny, dtype=jnp.int32)[:, None]
    xx = jnp.arange(nx, dtype=jnp.int32)[None, :]
    y_float = yy.astype(jnp.float64)
    x_float = xx.astype(jnp.float64)
    h0 = terrain
    shadowed = jnp.zeros((ny, nx), dtype=bool)

    branch_n = (sol_azi > 1.75 * jnp.pi) | (sol_azi < 0.25 * jnp.pi)
    branch_e = (~branch_n) & (sol_azi < 0.75 * jnp.pi)
    branch_s = (~branch_n) & (~branch_e) & (sol_azi < 1.25 * jnp.pi)
    branch_w = (~branch_n) & (~branch_e) & (~branch_s)

    def x_interp_shadow(row, x_real, dxabs, branch):
        i1 = _fortran_int(x_real)
        i2 = i1 + 1
        wgt = x_real - i1.astype(jnp.float64)
        valid = daylight & branch & (row >= 0) & (row < ny) & (i1 >= 0) & (i2 < nx)
        row_c = jnp.clip(row, 0, ny - 1)
        i1_c = jnp.clip(i1, 0, nx - 1)
        i2_c = jnp.clip(i2, 0, nx - 1)
        h = wgt * terrain[row_c, i2_c] + (1.0 - wgt) * terrain[row_c, i1_c]
        topoelev = jnp.arctan((h - h0) / jnp.maximum(dxabs, 1.0e-6))
        return valid & (jnp.sin(topoelev) >= csza)

    def y_interp_shadow(col, y_real, dxabs, branch):
        j1 = _fortran_int(y_real)
        j2 = j1 + 1
        wgt = y_real - j1.astype(jnp.float64)
        valid = daylight & branch & (col >= 0) & (col < nx) & (j1 >= 0) & (j2 < ny)
        col_c = jnp.clip(col, 0, nx - 1)
        j1_c = jnp.clip(j1, 0, ny - 1)
        j2_c = jnp.clip(j2, 0, ny - 1)
        h = wgt * terrain[j2_c, col_c] + (1.0 - wgt) * terrain[j1_c, col_c]
        topoelev = jnp.arctan((h - h0) / jnp.maximum(dxabs, 1.0e-6))
        return valid & (jnp.sin(topoelev) >= csza)

    for step in range(1, max_steps + 1):
        step_f = float(step)
        tan_azi = jnp.tan(sol_azi)
        tan_east_west = jnp.tan(0.5 * jnp.pi + sol_azi)

        row_n = yy + step
        x_n = x_float + step_f * tan_azi
        dxabs_n = jnp.sqrt((float(dy_m) * step_f) ** 2 + (float(dx_m) * (x_n - x_float)) ** 2)
        shadowed = shadowed | x_interp_shadow(row_n, x_n, dxabs_n, branch_n)

        col_e = xx + step
        y_e = y_float - step_f * tan_east_west
        dxabs_e = jnp.sqrt((float(dx_m) * step_f) ** 2 + (float(dy_m) * (y_e - y_float)) ** 2)
        shadowed = shadowed | y_interp_shadow(col_e, y_e, dxabs_e, branch_e)

        row_s = yy - step
        x_s = x_float - step_f * tan_azi
        dxabs_s = jnp.sqrt((float(dy_m) * step_f) ** 2 + (float(dx_m) * (x_s - x_float)) ** 2)
        shadowed = shadowed | x_interp_shadow(row_s, x_s, dxabs_s, branch_s)

        col_w = xx - step
        y_w = y_float + step_f * tan_east_west
        dxabs_w = jnp.sqrt((float(dx_m) * step_f) ** 2 + (float(dy_m) * (y_w - y_float)) ** 2)
        shadowed = shadowed | y_interp_shadow(col_w, y_w, dxabs_w, branch_w)

    return jnp.where(daylight & shadowed, 1, 0).astype(jnp.int32)


def _rrtmg_topography_state(
    static: RRTMGRadiationStatic | None,
    grid: GridSpec | None,
    geometry: SolarGeometry,
    *,
    slope_rad: int = 0,
    topo_shading: int = 0,
    shadow_length_m: float = 25000.0,
) -> RRTMGSWTopographyState | None:
    if static is None or int(slope_rad) != 1:
        return None
    dtype = jnp.asarray(static.xlat_deg).dtype
    if int(topo_shading) == 1 and grid is not None:
        shadow_mask = _wrf_topographic_shadow_mask(
            static.terrain_height_m,
            latitude_deg=static.xlat_deg,
            coszen=geometry.coszen,
            declination_rad=geometry.declination_rad,
            hour_angle_rad=geometry.hour_angle_rad,
            sina=static.sina,
            cosa=static.cosa,
            dx_m=float(grid.projection.dx_m),
            dy_m=float(grid.projection.dy_m),
            shadow_length_m=float(shadow_length_m),
        )
    else:
        shadow_mask = jnp.zeros_like(static.slope_rad, dtype=jnp.int32)
    return RRTMGSWTopographyState(
        latitude_deg=static.xlat_deg.astype(dtype),
        declination_rad=jnp.asarray(geometry.declination_rad, dtype=dtype),
        hour_angle_rad=jnp.asarray(geometry.hour_angle_rad, dtype=dtype),
        slope_rad=static.slope_rad.astype(dtype),
        slope_azimuth_rad=static.slope_azimuth_rad.astype(dtype),
        shadow_mask=shadow_mask,
    )


def _solar_source_scale_for_time(time_utc=None, lead_seconds=0.0, *, clock_base=None):
    """WRF `solvar(ib) = scon / rrsw_scon` per-band SW source multiplier.

    Faithful transcription of `module_radiation_driver.F` `radconst`
    (`:3629-3634`) composed with `module_ra_rrtmg_sw.F` (`:9869`):

        solcon = 1370. * ECCFAC                       (radconst :3634)
        ECCFAC = 1.000110 + 0.034221*cos(RJUL)        (Paltridge & Platt 1976
               + 0.001280*sin(RJUL)                    earth-sun eccentricity)
               + 0.000719*cos(2*RJUL) + 0.000077*sin(2*RJUL)
        RJUL   = JULIAN * (360/365) * DEGRAD          (radconst :3630-3631)
        scon   = solcon                               (rrtmg_sw.F :10872, obscur=0)
        solvar = scon / rrsw_scon                      (rrtmg_sw.F :9869)

    This is a pure function of the run *date* (the fractional day-of-year),
    NOT replay output: COSZEN cancels out exactly. The clear-sky oracle's
    `--source-scale wrf-toa` value `SWDNTC/(COSZEN*1368.22)` reduces
    algebraically to this same constant because WRF's clear-sky TOA-down
    `SWDNTC = solcon * COSZEN`; the oracle merely *measured* what this formula
    *computes* from the date. We compute it directly so the operational path
    needs no WRF field.

    `JULIAN` follows the same 0-based fractional day-of-year convention WRF's
    radiation driver carries in `grid%julian` (Jan 1 00z = 0.0), reusing the
    clock that `_compute_coszen` already threads through `jax.lax.scan`. The
    scale advances correctly across a multi-day forecast.
    """

    return _solcon_for_time(time_utc, lead_seconds, clock_base=clock_base) / RRSW_SCON


def _solcon_for_time(time_utc=None, lead_seconds=0.0, *, clock_base=None):
    """WRF ``radconst`` date-adjusted total solar constant ``solcon`` (W m^-2).

    Faithful transcription of ``module_radiation_driver.F`` ``radconst``
    (``:3629-3634``)::

        solcon = 1370. * ECCFAC                       (radconst :3634)
        ECCFAC = 1.000110 + 0.034221*cos(RJUL)        (Paltridge & Platt 1976
               + 0.001280*sin(RJUL)                    earth-sun eccentricity)
               + 0.000719*cos(2*RJUL) + 0.000077*sin(2*RJUL)
        RJUL   = JULIAN * (360/365) * DEGRAD          (radconst :3630-3631)

    This is the SAME ``solcon`` the WRF radiation driver passes into BOTH the
    RRTMG SW source normalisation (where it appears as ``solcon/rrsw_scon``,
    :func:`_solar_source_scale_for_time`) and the Dudhia ``SWRAD`` call
    (``SOLCON`` argument). Both radiation families therefore share the identical
    date-of-year eccentricity factor; the Dudhia kernel multiplies it by
    ``coszen`` to form the TOA-down flux (``SOLTOP=SOLCON``, ``SDOWN(1)=SOLTOP*XMU``).
    Pure function of the run date; no replay output.
    """

    julian, utc_minute = _resolve_clock_parts(time_utc, clock_base)
    lead_minutes = jnp.asarray(lead_seconds, dtype=jnp.float64) / 60.0
    # WRF grid%julian is 0-based (Jan 1 00z -> 0.0); _time_utc_parts returns the
    # 1-based tm_yday, so subtract one day and add the fractional time-of-day.
    julian_now = (
        (julian - 1.0)
        + (jnp.asarray(utc_minute, dtype=jnp.float64) + lead_minutes) / MINUTES_PER_DAY
    )
    rjul = julian_now * (360.0 / 365.0) * DEG_TO_RAD
    eccfac = (
        1.000110
        + 0.034221 * jnp.cos(rjul)
        + 0.001280 * jnp.sin(rjul)
        + 0.000719 * jnp.cos(2.0 * rjul)
        + 0.000077 * jnp.sin(2.0 * rjul)
    )
    return 1370.0 * eccfac


def _grid_lat_lon(surface_shape: tuple[int, int], grid: GridSpec | None, dtype):
    """Build a deterministic mass-grid lat/lon approximation from GridSpec metadata."""

    if grid is None:
        return (
            jnp.zeros(surface_shape, dtype=dtype),
            jnp.zeros(surface_shape, dtype=dtype),
        )
    projection = grid.projection
    ny, nx = surface_shape
    y = jnp.arange(ny, dtype=jnp.float64) - (float(ny) - 1.0) / 2.0
    x = jnp.arange(nx, dtype=jnp.float64) - (float(nx) - 1.0) / 2.0
    lat_step = float(projection.dy_m) / 111_320.0
    lon_scale = jnp.maximum(111_320.0 * jnp.cos(float(projection.lat_0) * DEG_TO_RAD), 1.0)
    lon_step = float(projection.dx_m) / lon_scale
    lat = float(projection.lat_0) + y[:, None] * lat_step
    lon = float(projection.lon_0) + x[None, :] * lon_step
    return (
        jnp.broadcast_to(lat, surface_shape).astype(dtype),
        jnp.broadcast_to(lon, surface_shape).astype(dtype),
    )


def _surface_radiation_properties(state: State, land_state=None):
    """Return RRTMG surface albedo/emissivity, using prognostic land values when present."""

    lu = jnp.clip(jnp.asarray(state.lu_index, dtype=jnp.int32), 0, _MODIS_NOAH_ALBEDO.shape[0] - 1)
    fallback_albedo = jnp.take(_MODIS_NOAH_ALBEDO, lu).astype(state.t_skin.dtype)
    fallback_emissivity = jnp.take(_MODIS_NOAH_EMISSIVITY, lu).astype(state.t_skin.dtype)
    if land_state is None or not hasattr(land_state, "albedo") or not hasattr(land_state, "emiss"):
        return fallback_albedo, fallback_emissivity
    is_land = jnp.asarray(state.xland) < 1.5
    land_albedo = jnp.asarray(land_state.albedo, dtype=state.t_skin.dtype)
    land_emissivity = jnp.asarray(land_state.emiss, dtype=state.t_skin.dtype)
    valid_albedo = (land_albedo >= 0.0) & (land_albedo <= 1.0)
    valid_emissivity = (land_emissivity > 0.0) & (land_emissivity <= 1.0)
    albedo = jnp.where(is_land & valid_albedo, land_albedo, fallback_albedo)
    emissivity = jnp.where(is_land & valid_emissivity, land_emissivity, fallback_emissivity)
    return albedo, emissivity


def _rrtmg_column_inputs(
    state: State,
    grid: GridSpec | None,
    *,
    time_utc=None,
    lead_seconds=0.0,
    clock_base=None,
    radiation_static: RRTMGRadiationStatic | None = None,
    topo_shading: int = 0,
    slope_rad: int = 0,
    shadow_length_m: float = 25000.0,
    land_state=None,
) -> tuple[RRTMGSWColumnState, RRTMGLWColumnState, object, object, SolarGeometry, RRTMGSWTopographyState | None]:
    """Build SW/LW RRTMG column states and expose shared surface fields."""

    if getattr(grid, "metrics", None) is not None:
        # WRF's phy_prep decouples theta_m back to dry th_phy before radiation.
        theta = jnp.asarray(state.theta, dtype=jnp.float64) / (
            1.0 + WRF_RV_OVER_RD * jnp.asarray(state.qv, dtype=jnp.float64)
        )
    else:
        theta = jnp.asarray(state.theta)
    T = _temperature_from_theta(theta, jnp.asarray(state.p))
    p_columns = _to_columns(state.p)
    qv_columns = _to_columns(state.qv)
    qc_columns = _to_columns(state.qc)
    qi_columns = _to_columns(state.qi)
    qs_columns = _to_columns(state.qs)
    qg_columns = _to_columns(state.qg)
    cloud_fraction = _cloud_fraction_columns(state)
    if sgs_cloud_enabled():
        # WRF icloud_bl=1 (Registry default; both CPU-truth regions run it):
        # radiation sees the MYNN subgrid BL clouds. module_radiation_driver.F
        # :1404-1431 -- after the first step CLDFRA := CLDFRA_BL everywhere
        # (mym_condensation CASE(2) folds resolved qc/qi/qs into its cloud
        # fraction via the rh/q1 boosts, so it IS the full-column fraction),
        # and the SGS condensate is added ONLY where resolved hydrometeors are
        # absent (qc<1e-6 / qi<1e-8 and cldfra_bl>0.001). WRF saves/restores
        # qc/qi around radiation (lines 1217/3266), i.e. the merge is a
        # radiation-input view only -- exactly what these local columns are.
        cldfra_bl_cols = _to_columns(state.cldfra_bl)
        qc_bl_cols = _to_columns(state.qc_bl)
        qi_bl_cols = _to_columns(state.qi_bl)
        not_first = jnp.asarray(lead_seconds, dtype=jnp.float64) > 0.0
        cloud_fraction = jnp.where(
            not_first, jnp.clip(cldfra_bl_cols, 0.0, 1.0), cloud_fraction
        )
        bl_active = cldfra_bl_cols > 0.001
        qc_columns = qc_columns + jnp.where(
            (qc_columns < 1.0e-6) & bl_active, qc_bl_cols, 0.0
        )
        qi_columns = qi_columns + jnp.where(
            (qi_columns < 1.0e-8) & bl_active, qi_bl_cols, 0.0
        )
    dz = _column_dz_from_state(state, grid)
    rho = _to_columns(_rho_from_state(state))
    surface_shape = state.t_skin.shape
    surface_albedo, surface_emissivity = _surface_radiation_properties(state, land_state=land_state)
    static = _radiation_static_for_grid(surface_shape, grid, radiation_static, state.t_skin.dtype)
    if static is None:
        lat, lon = _grid_lat_lon(surface_shape, grid, state.t_skin.dtype)
    else:
        lat, lon = static.xlat_deg, static.xlong_deg
    geometry = _compute_solar_geometry(lat, lon, time_utc, lead_seconds, clock_base=clock_base)
    geometry = SolarGeometry(
        coszen=geometry.coszen.astype(state.t_skin.dtype),
        declination_rad=geometry.declination_rad,
        hour_angle_rad=geometry.hour_angle_rad.astype(state.t_skin.dtype),
    )
    topography = _rrtmg_topography_state(
        static,
        grid,
        geometry,
        slope_rad=slope_rad,
        topo_shading=topo_shading,
        shadow_length_m=shadow_length_m,
    )
    # WRF `solvar = scon/rrsw_scon` per-band SW source normalization, computed
    # from the run date (function-of-JULIAN, not replay output). Closes the GPU
    # per-band source sum to WRF's date-adjusted solar constant.
    solar_source_scale = _solar_source_scale_for_time(
        time_utc, lead_seconds, clock_base=clock_base
    ).astype(state.t_skin.dtype)

    sw_state = RRTMGSWColumnState(
        _to_columns(T),
        p_columns,
        qv_columns,
        qc_columns,
        qi_columns,
        qs_columns,
        qg_columns,
        cloud_fraction,
        surface_albedo,
        geometry.coszen,
        dz,
        rho,
        solar_source_scale=solar_source_scale,
    )
    lw_state = RRTMGLWColumnState(
        _to_columns(T),
        p_columns,
        qv_columns,
        qc_columns,
        qi_columns,
        qs_columns,
        qg_columns,
        cloud_fraction,
        state.t_skin,
        surface_emissivity,
        dz,
        rho,
    )
    return sw_state, lw_state, surface_albedo, surface_emissivity, geometry, topography


def _bottom_column(surface_field, template_columns):
    """Place a 2-D surface field into the lowest slot of a column array."""

    value = jnp.asarray(surface_field).astype(template_columns.dtype)
    return jnp.zeros_like(template_columns).at[..., 0].set(value)


def _surface_flux_column_inputs(state: State, theta_columns):
    """Return MYNN bottom-boundary columns sourced from the surface adapter."""

    return (
        _bottom_column(state.theta_flux, theta_columns),
        _bottom_column(state.qv_flux, theta_columns),
        _bottom_column(state.tau_u, theta_columns),
        _bottom_column(state.tau_v, theta_columns),
    )


def _rho_from_state(state: State):
    """Builds the density diagnostic required by column physics kernels."""

    T = _temperature_from_theta(state.theta, state.p)
    return density_from_pressure_temperature(state.p, T, state.qv)


def _column_dz_from_state(state: State, grid: GridSpec | None):
    """Returns terrain-following layer thickness from geopotential interfaces."""

    del grid
    interface_height_m = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz = jnp.maximum(interface_height_m[1:, :, :] - interface_height_m[:-1, :, :], 1.0)
    return _to_columns(dz)


def _surface_dz_from_state(state: State):
    """WRF `phy_prep` surface-layer `dz8w` using physics `g=9.81`."""

    interface_height_m = state.ph.astype(jnp.float64) / WRF_PHYSICS_G_M_S2
    dz = jnp.maximum(interface_height_m[1:, :, :] - interface_height_m[:-1, :, :], 1.0)
    return _to_columns(dz)


def _total_or_legacy_field(state: State, total_name: str, legacy_name: str, dtype):
    """Read authoritative c2 total fields; legacy aliases are synced at State init."""

    del legacy_name
    return jnp.asarray(getattr(state, total_name), dtype=dtype)


def _wrf_hydrostatic_pressure_from_state(state: State, metrics):
    """Reconstruct WRF `phy_prep` `p_hyd`/`psfc` for surface physics."""

    dtype = jnp.float32
    mut = _total_or_legacy_field(state, "mu_total", "mu", dtype)
    c1h = jnp.asarray(metrics.c1h, dtype=dtype)
    c2h = jnp.asarray(metrics.c2h, dtype=dtype)
    dnw = jnp.asarray(metrics.dnw, dtype=dtype)
    p_top = jnp.reshape(jnp.asarray(metrics.p_top, dtype=dtype), ())
    qtot = sum(
        jnp.asarray(getattr(state, field), dtype=dtype)
        for field in ("qv", "qc", "qr", "qi", "qs", "qg")
    )

    nz = int(state.theta.shape[0])
    next_face = jnp.broadcast_to(p_top, mut.shape).astype(dtype)
    faces_top_to_bottom = [next_face]
    for k in range(nz - 1, -1, -1):
        mass = c1h[k] * mut + c2h[k]
        next_face = (next_face - (1.0 + qtot[k]) * mass * dnw[k]).astype(dtype)
        faces_top_to_bottom.append(next_face)

    faces = jnp.stack(tuple(reversed(faces_top_to_bottom)), axis=0)
    p_hyd = (0.5 * (faces[:-1, :, :] + faces[1:, :, :])).astype(jnp.float64)
    psfc = faces[0, :, :].astype(jnp.float64)
    return p_hyd, psfc


def _wrf_phy_prep_rho_from_state(state: State, metrics):
    """Reconstruct WRF ``phy_prep`` density passed to surface/PBL physics.

    WRF passes ``rho = (1+QVAPOR)/ALT`` where ``ALT`` is the full inverse
    density diagnosed by the dycore. The live State does not carry ``ALT``, but
    for the hypsometric-opt=2 nested path it is recoverable from total
    geopotential and hybrid pressure faces with the same float32 arithmetic used
    by WRF's live arrays.
    """

    dtype = jnp.float32
    mut = _total_or_legacy_field(state, "mu_total", "mu", dtype)
    ph_total = _total_or_legacy_field(state, "ph_total", "ph", dtype)
    qv = jnp.asarray(state.qv, dtype=dtype)
    c3h = jnp.asarray(metrics.c3h, dtype=dtype)
    c4h = jnp.asarray(metrics.c4h, dtype=dtype)
    c3f = jnp.asarray(metrics.c3f, dtype=dtype)
    c4f = jnp.asarray(metrics.c4f, dtype=dtype)
    p_top = jnp.reshape(jnp.asarray(metrics.p_top, dtype=dtype), ())

    p_up = c3f[1:, None, None] * mut[None, :, :] + c4f[1:, None, None] + p_top
    p_down = c3f[:-1, None, None] * mut[None, :, :] + c4f[:-1, None, None] + p_top
    p_mid = c3h[:, None, None] * mut[None, :, :] + c4h[:, None, None] + p_top
    dph = ph_total[1:, :, :] - ph_total[:-1, :, :]
    alt = dph / p_mid / jnp.log(p_down / p_up)
    rho = (1.0 + qv) / alt
    return rho.astype(jnp.float64)


def _cloud_fraction_columns(state: State):
    """Builds a bounded diagnostic cloud fraction from hydrometeor occupancy."""

    condensate = state.qc + state.qi + state.qs + state.qg
    return _to_columns(jnp.clip(condensate * 1.0e5, 0.0, 1.0))


def _thompson_column_from_state(state: State, grid: GridSpec | None = None) -> ThompsonColumnState:
    """Build the column-kernel input view for Thompson microphysics.

    Carries snow/graupel number (``Ns``/``Ng``), layer thickness (``dz``, from
    geopotential interfaces) and vertical velocity (``w``, mass-point) so the
    full WRF ``mp_gt_driver`` column kernel — including sedimentation — can run.
    """

    T = _temperature_from_theta(state.theta, state.p)
    rho = density_from_pressure_temperature(state.p, T, state.qv)
    # _column_dz_from_state already returns columns (trailing z); the others are
    # converted here. dz must be at least 1 m to keep the flux finite.
    dz_columns = _column_dz_from_state(state, grid)
    return ThompsonColumnState(
        _to_columns(state.qv),
        _to_columns(state.qc),
        _to_columns(state.qr),
        _to_columns(state.qi),
        _to_columns(state.qs),
        _to_columns(state.qg),
        _to_columns(state.Ni),
        _to_columns(state.Nr),
        _to_columns(T),
        _to_columns(state.p),
        _to_columns(rho),
        Ns=_to_columns(state.Ns),
        Ng=_to_columns(state.Ng),
        dz=dz_columns,
        w=_to_columns(_w_mass(state)),
    )


def _state_from_thompson_output(state: State, out: ThompsonColumnState, precip=None) -> State:
    """Reassemble a State from Thompson column-kernel output.

    When ``precip`` (per-channel surface accumulation, mm) is supplied, the
    precip accumulators are advanced with a per-step ``+=`` (Gate-1 decision #3,
    coupler_interface.md §6.3): rain_acc<-pptrain, snow_acc<-pptsnow,
    graupel_acc<-pptgraul, ice_acc<-pptice.
    """

    # Write at the LIVE state field dtype so force_fp64 stays truly fp64 through
    # the microphysics (fp32-defeat fix; see _output_dtype). The theta round
    # trip and every hydrometeor tendency are computed in fp64 when the carry is
    # fp64; in the default perf matrix the live dtype is still fp32 (unchanged).
    theta = _theta_from_temperature(_from_columns(out.T), state.p, _output_dtype(state, "theta"))
    updates = dict(
        theta=theta,
        qv=_from_columns(out.qv).astype(_output_dtype(state, "qv")),
        qc=_from_columns(out.qc).astype(_output_dtype(state, "qc")),
        qr=_from_columns(out.qr).astype(_output_dtype(state, "qr")),
        qi=_from_columns(out.qi).astype(_output_dtype(state, "qi")),
        qs=_from_columns(out.qs).astype(_output_dtype(state, "qs")),
        qg=_from_columns(out.qg).astype(_output_dtype(state, "qg")),
        Ni=_from_columns(out.Ni).astype(_output_dtype(state, "Ni")),
        Nr=_from_columns(out.Nr).astype(_output_dtype(state, "Nr")),
        Ns=_from_columns(out.Ns).astype(_output_dtype(state, "Ns")),
        Ng=_from_columns(out.Ng).astype(_output_dtype(state, "Ng")),
    )
    if precip is not None:
        # precip values are surface (ny, nx) in mm; State accumulators are (ny, nx).
        # Accumulators are fp64-locked in both modes (PRECISION_MATRIX), so the
        # live dtype is fp64 here -- the += already runs in fp64.
        updates["rain_acc"] = (jnp.asarray(state.rain_acc, dtype=jnp.float64) + precip["rain"]).astype(_output_dtype(state, "rain_acc"))
        updates["snow_acc"] = (jnp.asarray(state.snow_acc, dtype=jnp.float64) + precip["snow"]).astype(_output_dtype(state, "snow_acc"))
        updates["graupel_acc"] = (jnp.asarray(state.graupel_acc, dtype=jnp.float64) + precip["graupel"]).astype(_output_dtype(state, "graupel_acc"))
        updates["ice_acc"] = (jnp.asarray(state.ice_acc, dtype=jnp.float64) + precip["ice"]).astype(_output_dtype(state, "ice_acc"))
    return state.replace(**updates)


def _thompson_tendency_side_channel(
    state: State,
    out: ThompsonColumnState,
    dt: float,
) -> ThompsonTendencySideChannel:
    """Return water-species tendencies independent of precipitation accumulators."""

    inv_dt = 1.0 / float(dt)
    tendencies = {
        field: (
            jnp.asarray(_from_columns(getattr(out, field)), dtype=jnp.float64)
            - jnp.asarray(getattr(state, field), dtype=jnp.float64)
        )
        * inv_dt
        for field in ("qv", "qc", "qr", "qi", "qs", "qg")
    }
    column_water_tendency = jnp.mean(sum(tendencies.values()), axis=0)
    precip_out_tendency = jnp.zeros_like(column_water_tendency)
    return ThompsonTendencySideChannel(
        qv=tendencies["qv"],
        qc=tendencies["qc"],
        qr=tendencies["qr"],
        qi=tendencies["qi"],
        qs=tendencies["qs"],
        qg=tendencies["qg"],
        column_water_tendency=column_water_tendency,
        precip_out_tendency=precip_out_tendency,
    )


def thompson_adapter(state: State, dt: float, grid: GridSpec | None = None, *, return_tendencies: bool = False):
    """Slice state to Thompson inputs, call the kernel, and reassemble State.

    Advances all hydrometeor mixing ratios (qv,qc,qr,qi,qs,qg), all number
    concentrations (Ni,Nr,Ns,Ng), and `theta` (latent heat), runs sedimentation,
    and accumulates surface precipitation into rain/snow/graupel/ice accumulators
    via a per-step `+=` (Gate-1 decision #3).

    `return_tendencies=True` exposes the water-budget side channel while existing
    coupled-driver calls keep the original State-only API. The adapter is a no-op
    on columns that are physically inactive (handled inside the column kernel's
    thermodynamic-admissibility gate).
    """

    column = _thompson_column_from_state(state, grid)
    out, precip = step_thompson_column_with_precip(column, dt, debug=False)
    next_state = _state_from_thompson_output(state, out, precip)
    if return_tendencies:
        return next_state, _thompson_tendency_side_channel(state, out, dt)
    return next_state


def thompson_adapter_with_tendencies(state: State, dt: float) -> tuple[State, ThompsonTendencySideChannel]:
    """Explicit Thompson side-channel wrapper for validation call sites."""

    return thompson_adapter(state, dt, return_tendencies=True)


# ------------------------------------------------------------------------------
# v0.17 ADR-032 graupel/hail microphysics family -- fail-closed adapter slot
# ------------------------------------------------------------------------------

# The hail microphysics family this State substrate unblocks (mp_physics ids).
HAIL_MP_FAMILY: frozenset[int] = frozenset({7, 17, 18, 19, 21, 22, 24, 26, 27, 38})


def hail_mp_adapter(state: State, dt: float, *, mp_physics: int, grid: GridSpec | None = None) -> State:
    """Single fail-closed slot for the hail-heavy microphysics family (ADR-032).

    The State substrate (``qh``/``Nh``/``qvolg``/``qvolh``) is wired end-to-end
    -- precision matrix, registry, wrfout/wrfrst I/O, restart roundtrip,
    flux-form advection, scan increment carry and nest feedback -- but the
    SCHEME KERNELS (WSM7=24, WDM7=26, UDM=27, Goddard-4ice/NUWRF=7, NSSL=17-22,
    Thompson-graupel/hail=38) are NOT yet ported. This adapter is the single
    place a future worker drops the real column kernel; until then it fails
    closed loudly. It is NEVER reached by the operational scan, because none of
    ``HAIL_MP_FAMILY`` is in ``runtime.operational_mode._SCAN_WIRED_OPTIONS``
    nor in ``contracts.physics_registry.ACCEPTED_MP_PHYSICS`` -- so a hail mp id
    fails closed in the namelist validator / dispatcher long before any scan
    step, and this stub cannot silently produce a wrong result.
    """

    del state, dt, grid
    raise NotImplementedError(
        f"hail microphysics family (mp_physics={int(mp_physics)}) has its prognostic "
        "State substrate wired (ADR-032: qh/Nh/qvolg/qvolh leaves, I/O, restart, "
        "advection, feedback), but the scheme column kernel is not yet ported. "
        "Supported microphysics remain the v0.6.0 set (mp_physics in "
        "{0,1,2,3,4,6,8,10,14,16}); WSM7/WDM7/UDM/Goddard-4ice/NSSL/"
        "Thompson-graupel-hail will be enabled by a future worker that replaces "
        "this adapter body and adds the id to ACCEPTED_MP_PHYSICS / "
        "_SCAN_WIRED_OPTIONS / scheme_catalog._IMPLEMENTED."
    )


# ------------------------------------------------------------------------------
# v0.16 aerosol-aware Thompson (mp_physics=28) coupling
# ------------------------------------------------------------------------------


def _mass_level_height_columns(state: State):
    """WRF ``phy_prep`` mass-level geometric height z (m MSL) as columns.

    WRF hands ``thompson_init``/``mp_gt_driver`` the phy_prep height field
    z = (PH+PHB)/g averaged to mass levels with the physics g=9.81
    (module_model_constants.F). ``state.ph`` is the TOTAL geopotential on
    interfaces (legacy alias of ``ph_total``), so the mass-level height is the
    mean of the two bounding interface heights. Returns ``(ny, nx, nz)``.
    """

    z_face = state.ph.astype(jnp.float64) / WRF_PHYSICS_G_M_S2  # (nz+1, ny, nx)
    z_mass = 0.5 * (z_face[:-1, :, :] + z_face[1:, :, :])
    return _to_columns(z_mass)


def _thompson_aero_column_from_state(state: State, grid: GridSpec | None = None) -> ThompsonAeroColumnState:
    """Build the column-kernel input view for aerosol-aware Thompson (mp=28).

    Mirrors :func:`_thompson_column_from_state` and additionally threads the
    aerosol-aware prognostics: cloud droplet number ``Nc`` and the
    water-/ice-friendly aerosol numbers ``nwfa``/``nifa`` (per kg).
    """

    T = _temperature_from_theta(state.theta, state.p)
    rho = density_from_pressure_temperature(state.p, T, state.qv)
    dz_columns = _column_dz_from_state(state, grid)
    return ThompsonAeroColumnState(
        _to_columns(state.qv),
        _to_columns(state.qc),
        _to_columns(state.qr),
        _to_columns(state.qi),
        _to_columns(state.qs),
        _to_columns(state.qg),
        _to_columns(state.Ni),
        _to_columns(state.Nr),
        _to_columns(state.Nc),
        _to_columns(state.nwfa),
        _to_columns(state.nifa),
        _to_columns(T),
        _to_columns(state.p),
        _to_columns(rho),
        Ns=_to_columns(state.Ns),
        Ng=_to_columns(state.Ng),
        dz=dz_columns,
        w=_to_columns(_w_mass(state)),
    )


def _aerosol_surface_emission_columns(state: State, grid: GridSpec | None = None):
    """WRF ``thompson_init`` fake surface aerosol emission (nwfa2d, nifa2d).

    Inline jnp transcription of the ``nwfa2d`` closed form in
    :func:`gpuwrf.physics.thompson_aero_column.climatological_aerosol_profiles`
    (module_mp_thompson.F:493-558, use_aero_icbc=.false. path) so the jitted
    timestep loop never crosses to the host (the frozen helper is NumPy).
    ``nifa2d`` is zero in WRF for this path. Returns per-kg-per-second columns
    shaped ``(ny, nx)`` (the column batch shape).
    """

    del grid
    hgt = _mass_level_height_columns(state)  # (ny, nx, nz), m MSL
    h0 = hgt[..., 0]
    h_01 = jnp.where(
        h0 <= 1000.0,
        0.8,
        jnp.where(h0 >= 2500.0, 0.01, 0.8 * jnp.cos(h0 * 0.001 - 1.0)),
    )
    ni_ccn3 = -1.0 * jnp.log(NA_CCN1 / NA_CCN0) / h_01
    # Level 1 uses the LEVEL-2 height offset (WRF lines 508, 546).
    dz1 = hgt[..., 1] - h0
    nwfa1 = NA_CCN1 + NA_CCN0 * jnp.exp(-(dz1 / 1000.0) * ni_ccn3)
    z1 = jnp.maximum(dz1, 1.0)
    nwfa2d = nwfa1 * 0.000196 * (50.0 / z1)
    return nwfa2d, jnp.zeros_like(nwfa2d)


def _climatological_aerosol_profile_columns(state: State, grid: GridSpec | None = None):
    """Inline jnp ``climatological_aerosol_profiles`` 3-D nwfa/nifa (per kg).

    Faithful transcription of the frozen NumPy helper in
    ``thompson_aero_column`` (module_mp_thompson.F:493-558) on the mass-level
    heights derived from State geopotential. Returns ``(nwfa, nifa)`` columns
    shaped ``(ny, nx, nz)``.
    """

    del grid
    hgt = _mass_level_height_columns(state)  # (ny, nx, nz), m MSL
    h0 = hgt[..., :1]
    h_01 = jnp.where(
        h0 <= 1000.0,
        0.8,
        jnp.where(h0 >= 2500.0, 0.01, 0.8 * jnp.cos(h0 * 0.001 - 1.0)),
    )
    ni_ccn3 = -1.0 * jnp.log(NA_CCN1 / NA_CCN0) / h_01
    ni_in3 = -1.0 * jnp.log(NA_IN1 / NA_IN0) / h_01
    dz_agl = hgt - h0
    # Level 1 uses the LEVEL-2 height offset (WRF lines 508, 546).
    dz1 = hgt[..., 1:2] - h0
    dz_eff = jnp.concatenate([dz1, dz_agl[..., 1:]], axis=-1)
    nwfa = NA_CCN1 + NA_CCN0 * jnp.exp(-(dz_eff / 1000.0) * ni_ccn3)
    nifa = NA_IN1 + NA_IN0 * jnp.exp(-(dz_eff / 1000.0) * ni_in3)
    return nwfa, nifa


def thompson_aero_coldstart_init(state: State, grid: GridSpec | None = None) -> State:
    """Cold-start nwfa/nifa from the WRF climatological aerosol profiles.

    WRF ``thompson_init`` (use_aero_icbc=.false. equivalent) self-initializes
    QNWFA/QNIFA from boundary-layer-following exponential climatologies when
    the inputs carry no aerosol state. The implementation is pure JAX so it can
    be staged by the operational forecast JIT without a host scalar transfer.
    A state that already carries a real (restart/forced) nwfa field is
    preserved, mirroring WRF's is_aerosol_aware input check.
    """

    state = state.ensure_conditional_leaves(mp_physics=28)
    existing_nwfa = jnp.asarray(state.nwfa)
    existing_nifa = jnp.asarray(state.nifa)
    nwfa_cols, nifa_cols = _climatological_aerosol_profile_columns(state, grid)
    seeded_nwfa = _from_columns(nwfa_cols).astype(_output_dtype(state, "nwfa"))
    seeded_nifa = _from_columns(nifa_cols).astype(_output_dtype(state, "nifa"))
    has_aerosol_input = jnp.max(jnp.abs(existing_nwfa)) > 0.0
    return state.replace(
        nwfa=jnp.where(has_aerosol_input, existing_nwfa, seeded_nwfa).astype(_output_dtype(state, "nwfa")),
        nifa=jnp.where(has_aerosol_input, existing_nifa, seeded_nifa).astype(_output_dtype(state, "nifa")),
    )


def _state_from_thompson_aero_output(state: State, out: ThompsonAeroColumnState, precip=None) -> State:
    """Reassemble a State from aerosol-aware Thompson column-kernel output.

    Extends :func:`_state_from_thompson_output` with the aerosol-aware
    prognostics (Nc, nwfa, nifa). Precip accumulators advance with the same
    per-step ``+=`` convention (rain/snow/graupel/ice; the aero ``cloudw``
    surface-sedimentation channel is NOT an accumulator -- WRF RAINNCV is
    rain+snow+graupel+ice, proofs/v016/thompson_aero_savepoint_parity.py).
    """

    theta = _theta_from_temperature(_from_columns(out.T), state.p, _output_dtype(state, "theta"))
    updates = dict(
        theta=theta,
        qv=_from_columns(out.qv).astype(_output_dtype(state, "qv")),
        qc=_from_columns(out.qc).astype(_output_dtype(state, "qc")),
        qr=_from_columns(out.qr).astype(_output_dtype(state, "qr")),
        qi=_from_columns(out.qi).astype(_output_dtype(state, "qi")),
        qs=_from_columns(out.qs).astype(_output_dtype(state, "qs")),
        qg=_from_columns(out.qg).astype(_output_dtype(state, "qg")),
        Ni=_from_columns(out.Ni).astype(_output_dtype(state, "Ni")),
        Nr=_from_columns(out.Nr).astype(_output_dtype(state, "Nr")),
        Ns=_from_columns(out.Ns).astype(_output_dtype(state, "Ns")),
        Ng=_from_columns(out.Ng).astype(_output_dtype(state, "Ng")),
        Nc=_from_columns(out.Nc).astype(_output_dtype(state, "Nc")),
        nwfa=_from_columns(out.nwfa).astype(_output_dtype(state, "nwfa")),
        nifa=_from_columns(out.nifa).astype(_output_dtype(state, "nifa")),
    )
    if precip is not None:
        updates["rain_acc"] = (jnp.asarray(state.rain_acc, dtype=jnp.float64) + precip["rain"]).astype(_output_dtype(state, "rain_acc"))
        updates["snow_acc"] = (jnp.asarray(state.snow_acc, dtype=jnp.float64) + precip["snow"]).astype(_output_dtype(state, "snow_acc"))
        updates["graupel_acc"] = (jnp.asarray(state.graupel_acc, dtype=jnp.float64) + precip["graupel"]).astype(_output_dtype(state, "graupel_acc"))
        updates["ice_acc"] = (jnp.asarray(state.ice_acc, dtype=jnp.float64) + precip["ice"]).astype(_output_dtype(state, "ice_acc"))
    return state.replace(**updates)


def thompson_aero_adapter(state: State, dt: float, grid: GridSpec | None = None, *, return_tendencies: bool = False):
    """Slice state to mp=28 inputs, call the aero kernel, reassemble State.

    Mirrors :func:`thompson_adapter` (mp=8) with the aerosol-aware additions:
    threads Nc/nwfa/nifa through the WRF-parity-gated aero column kernel
    (proofs/v016/thompson_aero_savepoint_parity.json), then applies the WRF
    fake surface aerosol emission (module_mp_thompson.F:1317-1326) computed
    inline in jnp -- no host transfer inside the timestep loop.
    """

    state = state.ensure_conditional_leaves(mp_physics=28)
    column = _thompson_aero_column_from_state(state, grid)
    out, precip = step_thompson_aero_column_with_precip(column, float(dt), debug=False)
    nwfa2d, nifa2d = _aerosol_surface_emission_columns(state, grid)
    out = apply_surface_aerosol_emission(out, nwfa2d, nifa2d, float(dt))
    next_state = _state_from_thompson_aero_output(state, out, precip)
    if return_tendencies:
        return next_state, _thompson_tendency_side_channel(state, out, dt)
    return next_state


def _surface_fluxes_from_state(state: State) -> SurfaceFluxes:
    """Read the surface-flux handles ``surface_adapter`` wrote earlier in the chain.

    MYNN consumes the WRF revised surface layer's kinematic fluxes (the FROZEN
    surface→MYNN hand-off, coupler_interface.md §3). The kernel applies them as
    its bottom boundary condition inside the implicit vertical solve — no separate
    bottom-BC pass is needed (that would double-count the surface flux).
    """

    return SurfaceFluxes(
        ustar=jnp.asarray(state.ustar, dtype=jnp.float64),
        theta_flux=jnp.asarray(state.theta_flux, dtype=jnp.float64),
        qv_flux=jnp.asarray(state.qv_flux, dtype=jnp.float64),
        tau_u=jnp.asarray(state.tau_u, dtype=jnp.float64),
        tau_v=jnp.asarray(state.tau_v, dtype=jnp.float64),
        rhosfc=jnp.asarray(state.rhosfc, dtype=jnp.float64),
        fltv=jnp.asarray(state.fltv, dtype=jnp.float64),
        # WRF land/sea mask drives the mym_length CASE(1) land/water branch
        # (elt_max + el(k) hurricane taper). Marine columns (xland=2) use the
        # faithful elt_max=350 vs 400 over land.
        xland=jnp.asarray(state.xland, dtype=jnp.float64),
    )


def _mynn_column_uses_wrf_phy_prep(grid: GridSpec | None) -> bool:
    """Whether MYNN can use the same WRF ``phy_prep`` fields as the live driver."""

    return getattr(grid, "metrics", None) is not None


def _mynn_column_from_state(state: State, grid: GridSpec | None) -> MynnPBLColumnState:
    """Build the MYNN column-kernel input view from State (mass-point winds).

    The live WRF path stores theta_m in the model state when ``use_theta_m=1``,
    but ``module_bl_mynnedmf_driver`` receives dry ``th_phy`` plus hydrostatic
    ``p_phy``/``rho``/``dz8w`` from ``phy_prep``. Mirror that grid-backed
    contract here; analytic callers without metrics keep the historical direct
    state view.
    """

    if _mynn_column_uses_wrf_phy_prep(grid):
        metrics = grid.metrics
        theta = jnp.asarray(state.theta, dtype=jnp.float64) / (
            1.0 + WRF_RV_OVER_RD * jnp.asarray(state.qv, dtype=jnp.float64)
        )
        p, _psfc = _wrf_hydrostatic_pressure_from_state(state, metrics)
        rho = _wrf_phy_prep_rho_from_state(state, metrics)
        dz_columns = _surface_dz_from_state(state)
    else:
        theta = jnp.asarray(state.theta, dtype=jnp.float64)
        p = jnp.asarray(state.p, dtype=jnp.float64)
        rho = _rho_from_state(state)
        dz_columns = _column_dz_from_state(state, grid)

    rho_columns = _to_columns(rho)
    zeros = jnp.zeros_like(rho_columns)
    return MynnPBLColumnState(
        _to_columns(_u_mass(state)),
        _to_columns(_v_mass(state)),
        _to_columns(_w_mass(state)),
        _to_columns(theta),
        _to_columns(state.qv),
        0.5 * _to_columns(state.qke),  # tke = qke/2
        _to_columns(p),
        rho_columns,
        dz_columns,
        zeros,  # km (kernel output)
        zeros,  # kh (kernel output)
        zeros,  # el (kernel output)
        qc=_to_columns(state.qc),
        qi=_to_columns(state.qi),
        qs=_to_columns(state.qs),
        qsq=_to_columns(state.qsq),
    )


def _add_a2c_u_increment(u_face: jax.Array, du_mass: jax.Array) -> jax.Array:
    """Add a mass-point u increment to the C-grid u-faces, WRF ``add_a2c_u`` style.

    WRF source anchor: ``phys/module_physics_addtendc.F:2531-2582`` (``add_a2c_u``)
    adds ``0.5*(RUBLTEN(i)+RUBLTEN(i-1))`` -- the centred A-grid->C-grid average of
    the PBL momentum *increment* -- onto the EXISTING C-grid u tendency, leaving the
    dynamics' large-scale wind exactly as the dycore produced it. The interior u-face
    at x-index ``i`` (Python ``i``, between mass cells ``i-1`` and ``i``) gets
    ``0.5*(du(i-1)+du(i))``; the two domain-edge faces (x=0, x=nx) are NOT updated
    (``i_start=MAX(ids+1,its)``, ``i_end=MIN(ide-1,ite)`` for specified/nested) and are
    OWNED by ``apply_lateral_boundaries`` downstream.

    ``u_face`` is C-grid ``(nz, ny, nx+1)``; ``du_mass`` is ``(nz, ny, nx)``.
    """

    # interior faces 1..nx-1 = avg of the two adjacent mass-cell increments.
    # Cast to the (gated) u-face dtype before the scatter: with qke promoted to
    # fp64 (qke-fp64-fix), the MYNN length-scale/diffusivity and hence the column
    # u output promote to fp64, so ``du_mass`` arrives fp64 while the C-grid
    # u-face stays fp32-gated. The increment is intentionally STORED at the
    # face's gated precision (u is FP32_GATED); the explicit cast keeps that
    # contract crisp and avoids the implicit-scatter-downcast warning (a future
    # JAX hard error). No numerical change vs the prior implicit downcast.
    interior = (0.5 * (du_mass[:, :, :-1] + du_mass[:, :, 1:])).astype(u_face.dtype)
    du_face = jnp.zeros_like(u_face)
    du_face = du_face.at[:, :, 1:-1].set(interior)  # edges (0, nx) stay 0
    return u_face + du_face


def _add_a2c_v_increment(v_face: jax.Array, dv_mass: jax.Array) -> jax.Array:
    """Add a mass-point v increment to the C-grid v-faces, WRF ``add_a2c_v`` style.

    Same as :func:`_add_a2c_u_increment` on the y axis (WRF ``add_a2c_v``): interior
    y-faces get ``0.5*(dv(j-1)+dv(j))``; the y=0 / y=ny edge faces are left for the
    lateral-boundary pass.  ``v_face`` is ``(nz, ny+1, nx)``; ``dv_mass`` is
    ``(nz, ny, nx)``.
    """

    # Cast to the gated v-face dtype before the scatter (see _add_a2c_u_increment:
    # qke-fp64 promotes the column v output to fp64; v stays FP32_GATED on the
    # C-grid). No numerical change vs the prior implicit downcast.
    interior = (0.5 * (dv_mass[:, :-1, :] + dv_mass[:, 1:, :])).astype(v_face.dtype)  # (nz, ny-1, nx)
    dv_face = jnp.zeros_like(v_face)
    dv_face = dv_face.at[:, 1:-1, :].set(interior)
    return v_face + dv_face


def _state_from_mynn_output(
    state: State, out: MynnPBLColumnState, *, theta_output_is_dry: bool = False
) -> State:
    """Reassemble State from MYNN column output, WRF-faithful incremental coupling.

    WRF couples PBL momentum by ADDING the A-grid PBL increment (RUBLTEN/RVBLTEN),
    averaged to the C-grid faces, onto the existing C-grid wind tendency
    (``phys/module_physics_addtendc.F::add_a2c_u/add_a2c_v``).  It never replaces
    the full C-grid wind with a mass->face reconstruction.

    The previous code did ``u = _mass_to_u_face(u_mass_after_mynn)`` -- a
    face->mass->face round trip of the WHOLE field every step.  That round trip is
    NOT identity: it spuriously re-interpolates (smooths/shifts) the dynamics'
    near-surface u/v each step.  The kinematic terrain-following surface w boundary
    condition (``advance_w.py``: ``w_surface = msftx*rdx*dht*u_low3 + ...``) reads
    those low-level winds, so the per-step reconstruction error -- amplified by
    ``rdx ~ 1/dx`` over the steep Canary d02 terrain -- seeded a LINEARLY RAMPING
    spurious surface w (w@k0: 42->1147 m/s over 24h; ``proofs/stability/`` 2026-05-30
    localization).  With MYNN off the same dycore is stable (w~14 m/s @ mid-level).

    The fix: form the MYNN increment on mass points (``Δ = out - input_mass``) and
    add ONLY that increment, A2C-averaged, onto the dynamics' ORIGINAL C-grid faces.
    The large-scale C-grid winds (and hence the surface-w BC) are then preserved
    exactly; only the genuine PBL friction increment is applied -- exactly WRF's
    contract.  theta/qv/qke live on the mass grid where read-back is identity, so
    those keep their direct writes (the increment form is mathematically identical
    there).  w is untouched (MYNN does not solve it).  LIVE-dtype writes keep
    force_fp64 truly fp64 through the PBL solve (fp32-defeat fix; see _output_dtype).
    """

    # MYNN reads the input winds via _u_mass/_v_mass; the increment it produced is
    # the difference between its output mass winds and that SAME input mass wind.
    du_mass = _from_columns(out.u) - _u_mass(state)
    dv_mass = _from_columns(out.v) - _v_mass(state)
    u_new = _add_a2c_u_increment(state.u, du_mass).astype(_output_dtype(state, "u"))
    v_new = _add_a2c_v_increment(state.v, dv_mass).astype(_output_dtype(state, "v"))
    qv_new = _from_columns(out.qv).astype(_output_dtype(state, "qv"))
    theta_new = _from_columns(out.theta)
    if theta_output_is_dry:
        theta_new = theta_new * (1.0 + WRF_RV_OVER_RD * jnp.asarray(qv_new, jnp.float64))
    return state.replace(
        u=u_new,
        v=v_new,
        theta=theta_new.astype(_output_dtype(state, "theta")),
        qv=qv_new,
        qke=(2.0 * _from_columns(out.tke)).astype(_output_dtype(state, "qke")),
        # v0.15 MYNN SGS-cloud chain: persist the closure-2.6 prognostic
        # total-water variance and the mym_condensation/DMP subgrid cloud the
        # icloud_bl radiation merge consumes. With the chain disabled these are
        # zeros-in/zeros-out (no behavior change).
        qsq=_from_columns(out.qsq).astype(_output_dtype(state, "qsq")),
        qc_bl=_from_columns(out.qc_bl).astype(_output_dtype(state, "qc_bl")),
        qi_bl=_from_columns(out.qi_bl).astype(_output_dtype(state, "qi_bl")),
        cldfra_bl=_from_columns(out.cldfra_bl).astype(_output_dtype(state, "cldfra_bl")),
    )


def _mynn_dx(grid: GridSpec | None) -> float:
    """Grid spacing (m) sizing the MYNN-EDMF updraft area/excess, WRF ``dx``.

    WRF's ``DMP_mf`` scales the updraft area and the excess-buoyancy closure with
    the horizontal grid spacing (``module_bl_mynnedmf.F`` ``edmf`` block). The
    MYNN-EDMF lane's :func:`mynn_edmf.dmp_mf_columns` takes ``dx`` as a static
    arg; pull it from the projection (same accessor as :func:`_grid_lat_lon`),
    defaulting to the public-entry default when no grid metadata is present.
    """

    if grid is None:
        return 1000.0
    return float(grid.projection.dx_m)


def _flatten_columns_to_batch(tree, ny: int, nx: int):
    """Collapse a ``(ny, nx, ...)`` MYNN column pytree to a single batch axis.

    The MYNN-EDMF mass-flux entry (:func:`mynn_edmf.dmp_mf_columns`) is a
    ``jax.vmap`` over a SINGLE leading column dimension — its contract is profiles
    ``(B, nz)`` / surface scalars ``(B,)`` / interfaces ``(B, nz+1)``. The
    operational coupler builds the column view with ``_to_columns`` as
    ``(ny, nx, nz)`` (two spatial leading axes). The plain eddy-diffusion solve is
    leading-axis agnostic so it tolerated the extra axis, but the EDMF vmap would
    leave a residual ``nx`` axis inside the per-column kernel (shape clash
    ``zw[:-1]`` vs ``dz``). Flatten the ``(ny, nx)`` spatial grid to ``B=ny*nx``
    before the kernel and restore it after; this is the natural "batch of
    independent columns" contract the kernel documents.
    """

    return jax.tree_util.tree_map(
        lambda a: a.reshape((ny * nx,) + a.shape[2:]) if a.ndim >= 2 else a, tree
    )


def _unflatten_batch_to_columns(tree, ny: int, nx: int):
    """Inverse of :func:`_flatten_columns_to_batch`: ``(ny*nx, ...) -> (ny, nx, ...)``."""

    return jax.tree_util.tree_map(
        lambda a: a.reshape((ny, nx) + a.shape[1:]) if a.ndim >= 1 else a, tree
    )


def _mynn_coldstart_init_columns_tiled(
    column_b: MynnPBLColumnState,
    ust_b: jax.Array,
    dx: float,
    xland_b: jax.Array,
    *,
    rmol_init,
) -> tuple[jax.Array, jax.Array]:
    """Run WRF MYNN cold-start QKE over bounded column tiles.

    The cold-start initializer is per-column, but the dense BouLac path inside it
    materializes ``(ncol, nz, nz)`` scratch. On AC1_FIT d03 that full-domain init
    can trip XLA autotune allocations before the forecast starts. Reusing the
    production MYNN column tile width keeps the exact dense algorithm and only
    changes the independent column batch size.
    """

    profile = jnp.asarray(column_b.theta)
    tiling_active = (
        _MYNN_COLUMN_TILING
        and _MYNN_COLUMN_TILE_COLS > 0
        and profile.ndim == 2
        and int(profile.shape[0]) > _MYNN_COLUMN_TILE_COLS
    )
    if not tiling_active:
        return mynn_coldstart_init_columns(
            column_b, ust_b, dx, xland_b, rmol_init=rmol_init
        )

    ncol = int(profile.shape[0])
    tile_cols = int(_MYNN_COLUMN_TILE_COLS)
    n_tiles = (ncol + tile_cols - 1) // tile_cols
    padded_ncol = n_tiles * tile_cols

    padded_column = jax.tree_util.tree_map(
        lambda arr: _pad_columns_leaf(arr, ncol=ncol, padded_ncol=padded_ncol),
        column_b,
    )
    ust_pad = _pad_columns_leaf(ust_b, ncol=ncol, padded_ncol=padded_ncol)
    xland_pad = _pad_columns_leaf(xland_b, ncol=ncol, padded_ncol=padded_ncol)
    rmol_pad = (
        None
        if rmol_init is None
        else _pad_columns_leaf(rmol_init, ncol=ncol, padded_ncol=padded_ncol)
    )
    init_qke = jnp.zeros((padded_ncol, int(profile.shape[1])), dtype=profile.dtype)
    init_pblh = jnp.zeros((padded_ncol,), dtype=profile.dtype)

    def body(carry, tile_index):
        start = tile_index * tile_cols
        tile_column = jax.tree_util.tree_map(
            lambda arr: _slice_columns_leaf(arr, start, tile_cols, padded_ncol),
            padded_column,
        )
        tile_ust = _slice_columns_leaf(ust_pad, start, tile_cols, padded_ncol)
        tile_xland = _slice_columns_leaf(xland_pad, start, tile_cols, padded_ncol)
        tile_rmol = (
            None
            if rmol_pad is None
            else _slice_columns_leaf(rmol_pad, start, tile_cols, padded_ncol)
        )
        tile_qke, tile_pblh = mynn_coldstart_init_columns(
            tile_column, tile_ust, dx, tile_xland, rmol_init=tile_rmol
        )
        qke_acc, pblh_acc = carry
        return (
            _scatter_columns_leaf(qke_acc, tile_qke, start),
            _scatter_columns_leaf(pblh_acc, tile_pblh, start),
        ), None

    (qke, pblh), _ = jax.lax.scan(
        body, (init_qke, init_pblh), jnp.arange(n_tiles, dtype=jnp.int32)
    )
    return qke[:ncol], pblh[:ncol]


# MYNN-EDMF mass-flux nonlocal transport (WRF ``bl_mynn_edmf=1``).
# The Canary namelist selects MYNN (``bl_pbl_physics=5``) and relies on WRF's
# Registry defaults: ``bl_mynn_edmf=1`` and ``bl_mynn_edmf_mom=1``. The
# operational path therefore keeps scalar and momentum mass-flux transport on.
_MYNN_EDMF = True


def mynn_adapter(
    state: State, dt: float, grid: GridSpec | None = None, *, first_timestep=False
) -> State:
    """Advance the MYNN PBL using the surface fluxes ``surface_adapter`` wrote.

    THIN adapter: builds the column view, hands the FROZEN surface→MYNN flux
    contract to the kernel (which applies it as the implicit bottom BC), and
    reassembles State with non-periodic C-grid wind reconstruction.

    The MYNN-EDMF mass-flux arrays (``s_aw``/``s_awu``/``s_awv``/``s_awqv``/
    ``s_awthl``; scalar arrays verified <0.5% vs pristine WRF ``DMP_mf`` in
    ``proofs/mynn_edmf``) are gated by :data:`_MYNN_EDMF`. The column view is
    flattened to the kernel's single-batch contract for the EDMF vmap (see
    :func:`_flatten_columns_to_batch`).
    """

    state = _mynn_state_with_first_call_qke(state, grid, first_timestep)
    column = _mynn_column_from_state(state, grid)
    surface = _surface_fluxes_from_state(state)
    ny, nx = column.theta.shape[0], column.theta.shape[1]
    column_b = _flatten_columns_to_batch(column, ny, nx)
    surface_b = _flatten_columns_to_batch(surface, ny, nx)
    out_b = step_mynn_pbl_column(
        column_b, dt, debug=False, surface=surface_b, edmf=_MYNN_EDMF, dx=_mynn_dx(grid)
    )
    out = _unflatten_batch_to_columns(out_b, ny, nx)
    return _state_from_mynn_output(
        state, out, theta_output_is_dry=_mynn_column_uses_wrf_phy_prep(grid)
    )


def mynn_coldstart_qke_from_state(
    state: State, grid: GridSpec | None = None, rmol_init=None
) -> jax.Array:
    """WRF MYNN first-call cold-start qke initialization from ``State``.

    Builds the same column view the operational MYNN adapter consumes and runs
    the faithful ``module_bl_mynnedmf.F`` ``initflag>0``/``INITIALIZE_QKE``
    transcription (:func:`gpuwrf.physics.mynn_pbl.mynn_coldstart_init_columns`):
    driver taper pre-seed -> frozen ``GET_PBLH``/``SCALE_AWARE`` -> 5-pass
    level-2 equilibrium ``mym_initialize`` iteration. This is an INIT-TIME
    construction (no timestep-loop work): in statically unstable initial layers
    the level-2 equilibrium gives O(0.1-10 m^2/s^2) qke, which the prior
    taper-only seed missed by 3-5 orders of magnitude — the root cause of the
    Step-1 MYNN source outputs being ~10x weaker than WRF (v0.14
    ``proofs/v014/mynn_driver_source_output_fix``).

    Returns the ``(nz, ny, nx)`` qke field WRF's first MYNN call initializes.
    """

    column = _mynn_column_from_state(state, grid)
    ny, nx = column.theta.shape[0], column.theta.shape[1]
    column_b = _flatten_columns_to_batch(column, ny, nx)
    ust_b = jnp.asarray(state.ustar, dtype=jnp.float64).reshape(ny * nx)
    xland_b = jnp.asarray(state.xland, dtype=jnp.float64).reshape(ny * nx)
    rmol_b = None if rmol_init is None else jnp.asarray(rmol_init, dtype=jnp.float64).reshape(ny * nx)
    qke_b, _pblh = _mynn_coldstart_init_columns_tiled(
        column_b, ust_b, _mynn_dx(grid), xland_b, rmol_init=rmol_b
    )
    return _from_columns(_unflatten_batch_to_columns(qke_b, ny, nx))


# WRF module_bl_mynnedmf.F:623: INITIALIZE_QKE = MAXVAL(qke) < 0.0002 -- the
# scheme cold-starts the TKE state ONLY when the incoming field carries no real
# turbulence.  Mirrored here so a mid-run re-init that loads a spun-up QKE
# (e.g. the Switzerland h36 wrfout re-init, max qke ~25 m^2/s^2) keeps it,
# exactly like WRF's INITIALIZE_QKE=.FALSE. branch.  v0.14 venting-residual
# sprint: the unconditional seed overrode the loaded h36 QKE and inflated the
# PBL sources 2-5x vs the WRF-native implied truth
# (proofs/v014/switzerland_uv_lane_contributors).
_MYNN_QKE_INIT_THRESHOLD = 0.0002


def _mynn_state_with_first_call_qke(
    state: State, grid: GridSpec | None, first_timestep
) -> State:
    """Apply WRF's first MYNN ``mym_initialize`` ordering after surface fluxes."""

    qke_live = jnp.asarray(state.qke)
    needs_init = jnp.max(qke_live) < _MYNN_QKE_INIT_THRESHOLD
    if isinstance(first_timestep, bool):
        if not first_timestep:
            return state
        qke_seed = jnp.where(
            needs_init, mynn_coldstart_qke_from_state(state, grid), qke_live
        )
    else:
        flag = jnp.asarray(first_timestep, dtype=bool) & needs_init

        def seed(_unused):
            return mynn_coldstart_qke_from_state(state, grid)

        qke_seed = jax.lax.cond(flag, seed, lambda _unused: qke_live, None)
    return state.replace(qke=qke_seed.astype(_output_dtype(state, "qke")))


def mynn_adapter_with_source_leaves(
    state: State, dt: float, grid: GridSpec | None = None, *, first_timestep=False
) -> MynnPBLSourceLeaves:
    """Advance MYNN and expose raw WRF MYNN source tendencies.

    WRF's MYNN path writes ``RTHBLTEN``/``RQVBLTEN`` from the scheme-local
    post-solve deltas divided by ``dt`` before ``module_em`` mass-couples them.
    The operational adapter already computes the same post-solve theta/qv; this
    helper returns those raw source rates so the runtime can build the WRF
    ``DryPhysicsTendencies.t_tendf`` source without treating an aggregate
    multi-scheme state delta as a dry source.
    """

    state = _mynn_state_with_first_call_qke(state, grid, first_timestep)
    column = _mynn_column_from_state(state, grid)
    surface = _surface_fluxes_from_state(state)
    ny, nx = column.theta.shape[0], column.theta.shape[1]
    column_b = _flatten_columns_to_batch(column, ny, nx)
    surface_b = _flatten_columns_to_batch(surface, ny, nx)
    out_b = step_mynn_pbl_column(
        column_b, dt, debug=False, surface=surface_b, edmf=_MYNN_EDMF, dx=_mynn_dx(grid)
    )
    out = _unflatten_batch_to_columns(out_b, ny, nx)
    theta_after = _from_columns(out.theta)
    theta_before = _from_columns(column.theta)
    qv_after = _from_columns(out.qv)
    rthblten = ((theta_after - theta_before) / float(dt)).astype(
        _output_dtype(state, "theta")
    )
    rqvblten = ((qv_after - jnp.asarray(state.qv, jnp.float64)) / float(dt)).astype(
        _output_dtype(state, "qv")
    )
    # Raw WRF A-grid momentum sources: the SAME mass-point increments
    # _state_from_mynn_output A2C-couples into the C-grid winds, divided by dt
    # (WRF MYNN driver RUBLTEN/RVBLTEN semantics).
    rublten = ((_from_columns(out.u) - _u_mass(state)) / float(dt)).astype(
        _output_dtype(state, "u")
    )
    rvblten = ((_from_columns(out.v) - _v_mass(state)) / float(dt)).astype(
        _output_dtype(state, "v")
    )
    return MynnPBLSourceLeaves(
        state=_state_from_mynn_output(
            state, out, theta_output_is_dry=_mynn_column_uses_wrf_phy_prep(grid)
        ),
        rthblten=rthblten,
        rqvblten=rqvblten,
        rublten=rublten,
        rvblten=rvblten,
    )


def mynn_adapter_with_diagnostics(
    state: State, dt: float, grid: GridSpec | None = None
) -> tuple[State, object]:
    """``mynn_adapter`` plus the PBLH operational diagnostic (mass-point 2-D)."""

    column = _mynn_column_from_state(state, grid)
    surface = _surface_fluxes_from_state(state)
    ny, nx = column.theta.shape[0], column.theta.shape[1]
    column_b = _flatten_columns_to_batch(column, ny, nx)
    surface_b = _flatten_columns_to_batch(surface, ny, nx)
    out_b, pblh_b = step_mynn_pbl_column_with_pblh(
        column_b, dt, debug=False, surface=surface_b, edmf=_MYNN_EDMF, dx=_mynn_dx(grid)
    )
    out = _unflatten_batch_to_columns(out_b, ny, nx)
    pblh = pblh_b.reshape((ny, nx) + pblh_b.shape[1:])
    return (
        _state_from_mynn_output(
            state, out, theta_output_is_dry=_mynn_column_uses_wrf_phy_prep(grid)
        ),
        pblh,
    )


def _surface_column_view(state: State, grid: GridSpec | None = None) -> _SurfaceColumnState:
    """Build the column-oriented view consumed by the WRF revised surface layer."""

    metrics = getattr(grid, "metrics", None) if grid is not None else None
    if metrics is not None:
        # WRF `phy_prep` converts in-memory theta_m back to dry `th_phy` for
        # physics when `use_theta_m=1`, while `surface_driver` passes hydrostatic
        # `P_PHY=grid%p_hyd` but retains `t_phy` computed from nonhydrostatic
        # `p+pb`. The v0.14 live-nest path stores theta_m in State.theta.
        dry_theta = jnp.asarray(state.theta, dtype=jnp.float64) / (
            1.0 + WRF_RV_OVER_RD * jnp.asarray(state.qv, dtype=jnp.float64)
        )
        p_hyd, psfc = _wrf_hydrostatic_pressure_from_state(state, metrics)
        rho = _wrf_phy_prep_rho_from_state(state, metrics)
        t_air = _temperature_from_theta(dry_theta, jnp.asarray(state.p, dtype=jnp.float64))
        theta = _to_columns(dry_theta)
        p = _to_columns(p_hyd)
        dz = _surface_dz_from_state(state)
    else:
        theta = _to_columns(state.theta)
        p = _to_columns(state.p)
        dz = _column_dz_from_state(state, None)
        t_air = None
        psfc = None
        rho = None

    return _SurfaceColumnState(
        u=_to_columns(_u_mass(state)),
        v=_to_columns(_v_mass(state)),
        theta=theta,
        qv=_to_columns(state.qv),
        p=p,
        dz=dz,
        t_skin=state.t_skin,
        soil_moisture=state.soil_moisture,
        xland=state.xland,
        lakemask=state.lakemask,
        mavail=state.mavail,
        roughness_m=state.roughness_m,
        ustar=state.ustar,
        t_air=_to_columns(t_air) if t_air is not None else None,
        psfc=psfc,
        rho=_to_columns(rho) if rho is not None else None,
    )


def surface_adapter(state: State, dt: float, grid: GridSpec | None = None, *, first_timestep=False) -> State:
    """Run the WRF revised surface layer and store its surface-flux handles.

    THIN adapter: the algebra lives in ``physics.surface_layer`` (a faithful port
    of ``sf_sfclayrev_run``). Writes only the B2 flux handles
    (coupler_interface.md §3); the operational diagnostics (HFX/LH/T2/U10/V10)
    are exposed separately via :func:`surface_layer_diagnostics`.
    """

    del dt
    flux = surface_layer(_surface_column_view(state, grid), first_timestep=first_timestep)
    # Surface flux handles are fp64-locked in PRECISION_MATRIX, so the live
    # dtype is fp64 in both modes; written via _output_dtype for one consistent
    # adapter-write contract (fp32-defeat fix; see _output_dtype).
    return state.replace(
        ustar=flux.ustar.astype(_output_dtype(state, "ustar")),
        theta_flux=flux.theta_flux.astype(_output_dtype(state, "theta_flux")),
        qv_flux=flux.qv_flux.astype(_output_dtype(state, "qv_flux")),
        tau_u=flux.tau_u.astype(_output_dtype(state, "tau_u")),
        tau_v=flux.tau_v.astype(_output_dtype(state, "tau_v")),
        rhosfc=flux.rhosfc.astype(_output_dtype(state, "rhosfc")),
        fltv=flux.fltv.astype(_output_dtype(state, "fltv")),
    )


def surface_layer_diagnostics(state: State, grid: GridSpec | None = None) -> SurfaceMynnDiagnostics:
    """Return B2 operational surface/PBL diagnostics without changing State.

    HFX/LH/T2/U10/V10/ustar come from the revised surface layer; PBLH is the
    MYNN-diagnosed PBL height. Side-channel only (coupler_interface.md §4): no
    prognostic State leaves are written. Call on a State whose surface-flux
    handles have already been written by ``surface_adapter`` (so MYNN sees the
    real fluxes when diagnosing PBLH)."""

    diag = surface_layer_with_diagnostics(_surface_column_view(state, grid))
    column = _mynn_column_from_state(state, grid)
    surface = _surface_fluxes_from_state(state)
    _out, pblh = step_mynn_pbl_column_with_pblh(column, 1.0, debug=False, surface=surface)
    return SurfaceMynnDiagnostics(
        hfx=diag.hfx,
        lh=diag.lh,
        pblh=pblh,
        t2=diag.t2,
        u10=diag.u10,
        v10=diag.v10,
        ustar=diag.fluxes.ustar,
    )


def rrtmg_radiation_diagnostics(
    state: State,
    grid: GridSpec | None = None,
    *,
    time_utc=None,
    lead_seconds=0.0,
    clock_base=None,
    radiation_static: RRTMGRadiationStatic | None = None,
    topo_shading: int = 0,
    slope_rad: int = 0,
    shadow_length_m: float = 25000.0,
    land_state=None,
    with_clear_sky: bool = False,
) -> RRTMGRadiationDiagnostics:
    """Return surface RRTMG radiation diagnostics without changing State.

    `time_utc` is the simulation init instant; `lead_seconds` (may be traced)
    is elapsed forecast time, so SWDOWN/GLW/coszen follow the real forecast
    clock (diurnal cycle) at the M9 I/O cadence.

    When ``with_clear_sky`` is True the SW/LW solvers also run the WRF clear-sky
    (cloud-free) radiative-transfer pass and the ``...C`` clear-sky surface/TOA
    fluxes are populated.  The all-sky outputs are byte-identical either way.
    """

    sw_state, lw_state, surface_albedo, surface_emissivity, geometry, topography = _rrtmg_column_inputs(
        state,
        grid,
        time_utc=time_utc,
        lead_seconds=lead_seconds,
        clock_base=clock_base,
        radiation_static=radiation_static,
        topo_shading=topo_shading,
        slope_rad=slope_rad,
        shadow_length_m=shadow_length_m,
        land_state=land_state,
    )
    sw = solve_rrtmg_sw_column(sw_state, debug=False, topography=topography, with_clear_sky=with_clear_sky)
    lw = solve_rrtmg_lw_column(lw_state, debug=False, with_clear_sky=with_clear_sky)
    shadow_mask = (
        jnp.zeros_like(surface_albedo, dtype=jnp.int32)
        if topography is None
        else topography.shadow_mask
    )
    if with_clear_sky:
        sw_clear_toa_down = sw.clear_flux_down[..., -1]
        sw_clear_toa_up = sw.clear_flux_up[..., -1]
        sw_clear_sfc_down = sw.clear_flux_down[..., 0]
        sw_clear_sfc_up = sw.clear_flux_up[..., 0]
        lw_clear_toa_down = lw.clear_flux_down[..., -1]
        lw_clear_toa_up = lw.clear_flux_up[..., -1]
        lw_clear_sfc_down = lw.clear_flux_down[..., 0]
        lw_clear_sfc_up = lw.clear_flux_up[..., 0]
    else:
        sw_clear_toa_down = sw_clear_toa_up = sw_clear_sfc_down = sw_clear_sfc_up = None
        lw_clear_toa_down = lw_clear_toa_up = lw_clear_sfc_down = lw_clear_sfc_up = None
    return RRTMGRadiationDiagnostics(
        surface_albedo=surface_albedo,
        surface_emissivity=surface_emissivity,
        coszen=geometry.coszen,
        swdown=sw.surface_down,
        swnorm=sw.surface_down_topographic,
        swup=sw.surface_up,
        swup_topographic=sw.surface_up_topographic,
        glw=lw.surface_down,
        glw_up=lw.surface_up,
        topographic_correction_factor=sw.topographic_correction_factor,
        shadow_mask=shadow_mask,
        # B1: top-of-atmosphere all-sky flux slices (model-top interface). WRF
        # SWDNT/SWUPT == SW top down/up; LWDNT/LWUPT == LW top down/up; OLR == LWUPT.
        sw_toa_down=sw.toa_down,
        sw_toa_up=sw.toa_up,
        lw_toa_down=lw.toa_down,
        lw_toa_up=lw.toa_up,
        sw_clear_toa_down=sw_clear_toa_down,
        sw_clear_toa_up=sw_clear_toa_up,
        sw_clear_sfc_down=sw_clear_sfc_down,
        sw_clear_sfc_up=sw_clear_sfc_up,
        lw_clear_toa_down=lw_clear_toa_down,
        lw_clear_toa_up=lw_clear_toa_up,
        lw_clear_sfc_down=lw_clear_sfc_down,
        lw_clear_sfc_up=lw_clear_sfc_up,
    )


def rrtmg_adapter(
    state: State,
    dt: float,
    grid: GridSpec | None = None,
    *,
    time_utc=None,
    lead_seconds=0.0,
    apply_seconds: float | None = None,
    radiation_static: RRTMGRadiationStatic | None = None,
    topo_shading: int = 0,
    slope_rad: int = 0,
    shadow_length_m: float = 25000.0,
    land_state=None,
) -> State:
    """Run SW and LW RRTMG column kernels and apply their temperature tendency.

    Model time is threaded: `time_utc` is the init instant and `lead_seconds`
    (may be a traced JAX value inside the operational scan) is elapsed forecast
    time. The solar-zenith / diurnal forcing therefore responds to the actual
    forecast clock; there is NO fixed-time fallback on the diurnal path once a
    caller passes the step lead.

    B3 CADENCE SCALING (coupling fix 2026-05-30): the SW/LW kernels return the
    instantaneous radiative heating RATE (K s^-1). WRF computes this rate once per
    `radt` interval and then applies the persisted RTHRATEN tendency at EVERY
    dynamics step over that whole interval (module_radiation_driver.F; the rate is
    held constant between radiation calls). When the operational scan only invokes
    this adapter once per `radiation_cadence_steps`, the heating must be integrated
    over the WHOLE cadence interval, not a single dynamics `dt`, or the radiative
    forcing is delivered `radiation_cadence_steps`x too weak. `apply_seconds` is
    that interval (= cadence_steps * dt); it defaults to `dt` for legacy
    every-step callers. (The previous `dt * heating_rate` was the artifact behind
    the B3 isolation-ladder anomaly the dycore-realinit frontrunner flagged: at
    cadence it under-heated; applied every step at full rate it could over-heat.)
    """

    seconds = float(dt) if apply_seconds is None else float(apply_seconds)
    T = _temperature_from_theta(state.theta, state.p)
    sw_state, lw_state, _, _, _, topography = _rrtmg_column_inputs(
        state,
        grid,
        time_utc=time_utc,
        lead_seconds=lead_seconds,
        radiation_static=radiation_static,
        topo_shading=topo_shading,
        slope_rad=slope_rad,
        shadow_length_m=shadow_length_m,
        land_state=land_state,
    )
    sw = solve_rrtmg_sw_column(sw_state, debug=False, topography=topography)
    lw = solve_rrtmg_lw_column(lw_state, debug=False)
    T_next = T + seconds * _from_columns(sw.heating_rate + lw.heating_rate)
    # LIVE-dtype write keeps force_fp64 fp64 through radiation (fp32-defeat fix;
    # see _output_dtype). The fp32 RRTMG band optics are an intrinsic property of
    # the RRTMG kernel (WRF uses r4 there too); the heating-rate ADD onto T and
    # the theta round trip run in fp64 when the carry is fp64.
    return state.replace(theta=_theta_from_temperature(T_next, state.p, _output_dtype(state, "theta")))


def rrtmg_theta_tendency(
    state: State,
    grid: GridSpec | None = None,
    *,
    time_utc=None,
    lead_seconds=0.0,
    clock_base=None,
    radiation_static: RRTMGRadiationStatic | None = None,
    topo_shading: int = 0,
    slope_rad: int = 0,
    shadow_length_m: float = 25000.0,
    land_state=None,
) -> "jnp.ndarray":
    """Return the WRF ``RTHRATEN`` radiative potential-temperature tendency (K/s).

    This is the HELD-RATE primitive behind the WRF-faithful radiation cadence
    (Sprint coupler-fp64 FIX #2 / GPT P0-2).  WRF recomputes ``RTHRATEN`` only
    once per ``radt`` interval (``module_radiation_driver.F`` run_param gate,
    :1111-1127) and then ADDS it into the theta tendency at EVERY dynamics step
    over that interval (``phy_ra_ten`` in ``module_physics_addtendc.F:131-229``,
    fed by ``module_first_rk_step_part2.F:392-394``).  The lumped
    ``dt*cadence*rate``-at-one-step alternative is NOT equivalent: the
    intervening dynamics/microphysics/PBL would see a different temperature
    trajectory.

    The returned 3-D field is on the (nz, ny, nx) mass grid and matches the
    state's theta dtype (fp64 under force_fp64).  Computing the tendency in
    theta-space (not T-space) means the per-step application is the exact WRF
    ``theta += dt * RTHRATEN`` -- no extra exner round trip per step.
    """

    T = _temperature_from_theta(state.theta, state.p)
    sw_state, lw_state, _, _, _, topography = _rrtmg_column_inputs(
        state,
        grid,
        time_utc=time_utc,
        lead_seconds=lead_seconds,
        clock_base=clock_base,
        radiation_static=radiation_static,
        topo_shading=topo_shading,
        slope_rad=slope_rad,
        shadow_length_m=shadow_length_m,
        land_state=land_state,
    )
    sw = solve_rrtmg_sw_column(sw_state, debug=False, topography=topography)
    lw = solve_rrtmg_lw_column(lw_state, debug=False)
    heating_rate_T = _from_columns(sw.heating_rate + lw.heating_rate)  # dT/dt (K/s)
    # Convert the temperature heating rate to a theta tendency via the exner
    # factor (theta = T / exner; for fixed pressure d(theta)/dt = (dT/dt)/exner).
    exner = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    rthraten = heating_rate_T / jnp.maximum(exner, 1.0e-12)
    return rthraten.astype(_output_dtype(state, "theta"))


def rrtmg_sw_theta_tendency(
    state: State,
    grid: GridSpec | None = None,
    *,
    time_utc=None,
    lead_seconds=0.0,
    clock_base=None,
    radiation_static: RRTMGRadiationStatic | None = None,
    topo_shading: int = 0,
    slope_rad: int = 0,
    shadow_length_m: float = 25000.0,
    land_state=None,
) -> "jnp.ndarray":
    """Return the RRTMG shortwave-ONLY ``RTHRATEN`` (K/s).

    The SW half of :func:`rrtmg_theta_tendency`, used by the dispatch when the LW
    scheme is selected independently (e.g. ``ra_sw=4`` paired with the classic
    RRTM LW ``ra_lw=1``). Reuses the shared RRTMG column-input assembler + SW
    kernel (including topo-shading / slope-rad); only the SW heating rate is
    converted to a theta tendency.
    """

    sw_state, _, _, _, _, topography = _rrtmg_column_inputs(
        state,
        grid,
        time_utc=time_utc,
        lead_seconds=lead_seconds,
        clock_base=clock_base,
        radiation_static=radiation_static,
        topo_shading=topo_shading,
        slope_rad=slope_rad,
        shadow_length_m=shadow_length_m,
        land_state=land_state,
    )
    sw = solve_rrtmg_sw_column(sw_state, debug=False, topography=topography)
    heating_rate_T = _from_columns(sw.heating_rate)  # dT/dt (K/s)
    exner = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    rthraten = heating_rate_T / jnp.maximum(exner, 1.0e-12)
    return rthraten.astype(_output_dtype(state, "theta"))


# --------------------------------------------------------------------------- #
# Dudhia shortwave (``ra_sw_physics=1``) HELD-RATE theta-tendency coupler.
# --------------------------------------------------------------------------- #
def _dudhia_sw_column_inputs(
    state: State,
    grid: GridSpec | None,
    *,
    time_utc=None,
    lead_seconds=0.0,
    clock_base=None,
    radiation_static: RRTMGRadiationStatic | None = None,
    land_state=None,
) -> tuple[DudhiaSWColumnState, SolarGeometry]:
    """Build the Dudhia SW column-kernel input view from operational ``State``.

    Mirrors :func:`_rrtmg_column_inputs` but assembles the inputs the Stephens-1984
    broadband shortwave kernel (``module_ra_sw.F:SWPARA``) consumes: layer
    temperature ``T`` (K), pressure ``p`` (Pa), the six moisture species, layer
    thickness ``dz`` (m), per-column cosine-zenith ``coszen``, surface ``albedo``
    and the date-adjusted total solar constant ``solcon`` (W m^-2). The hydrometeor
    species are passed through unchanged (the kernel forms its own cloud
    liquid-water path); cloud fraction is NOT consumed by Dudhia. The kernel works
    on flat ``(ncol=ny*nx, nz)`` columns, so the ``(nz, ny, nx)`` mass fields are
    reshaped to the kernel's batch contract here.

    ``solcon`` is the SAME WRF ``radconst`` date-of-year eccentricity-scaled solar
    constant the RRTMG path uses (:func:`_solcon_for_time`); the Dudhia driver
    passes ``SOLCON`` straight into ``SWRAD`` and the kernel multiplies it by
    ``coszen`` (``SOLTOP=SOLCON``) to form the TOA-down flux.
    """

    T = _temperature_from_theta(state.theta, state.p)
    nz, ny, nx = state.theta.shape
    ncol = ny * nx

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(ncol, nz)

    dz_cols = _column_dz_from_state(state, grid).reshape(ncol, nz)
    surface_shape = state.t_skin.shape
    surface_albedo, _ = _surface_radiation_properties(state, land_state=land_state)
    static = _radiation_static_for_grid(surface_shape, grid, radiation_static, state.t_skin.dtype)
    if static is None:
        lat, lon = _grid_lat_lon(surface_shape, grid, state.t_skin.dtype)
    else:
        lat, lon = static.xlat_deg, static.xlong_deg
    geometry = _compute_solar_geometry(lat, lon, time_utc, lead_seconds, clock_base=clock_base)
    geometry = SolarGeometry(
        coszen=geometry.coszen.astype(state.t_skin.dtype),
        declination_rad=geometry.declination_rad,
        hour_angle_rad=geometry.hour_angle_rad.astype(state.t_skin.dtype),
    )
    solcon = _solcon_for_time(time_utc, lead_seconds, clock_base=clock_base).astype(state.t_skin.dtype)
    solcon = jnp.broadcast_to(solcon, surface_shape).reshape(ncol)

    column = DudhiaSWColumnState(
        T=_cols(T),
        p=_cols(state.p),
        qv=_cols(state.qv),
        qc=_cols(state.qc),
        qr=_cols(state.qr),
        qi=_cols(state.qi),
        qs=_cols(state.qs),
        qg=_cols(state.qg),
        dz=dz_cols,
        coszen=geometry.coszen.reshape(ncol),
        albedo=jnp.asarray(surface_albedo).reshape(ncol),
        solcon=solcon,
    )
    return column, geometry


def dudhia_sw_theta_tendency(
    state: State,
    grid: GridSpec | None = None,
    *,
    time_utc=None,
    lead_seconds=0.0,
    clock_base=None,
    radiation_static: RRTMGRadiationStatic | None = None,
    land_state=None,
) -> "jnp.ndarray":
    """Return the Dudhia (``ra_sw_physics=1``) shortwave ``RTHRATEN`` (K/s).

    Shortwave-only HELD-RATE primitive, the Dudhia analogue of
    :func:`rrtmg_theta_tendency`. WRF's ``SWRAD`` accumulates the SW heating into
    ``RTHRATEN`` as ``RTHRATEN += TTEN1D/pi3D`` (``module_ra_sw.F``): the
    kernel returns the per-layer TEMPERATURE heating rate ``dT/dt`` (K/s) and the
    theta tendency is that rate divided by the Exner factor. The operational scan
    holds this rate over the ``radt`` interval (the same cadence machinery the
    RRTMG path uses) and ADDS it into theta at every dynamics step.

    This coupler is SHORTWAVE-ONLY; the longwave heating still comes from the
    operational RRTMG-LW path (the dispatch adds the two RTHRATEN contributions),
    exactly as WRF runs the SW and LW radiation drivers independently.

    The returned 3-D field is on the ``(nz, ny, nx)`` mass grid and matches the
    state's theta dtype (fp64 under ``force_fp64``).
    """

    nz, ny, nx = state.theta.shape
    column, _ = _dudhia_sw_column_inputs(
        state,
        grid,
        time_utc=time_utc,
        lead_seconds=lead_seconds,
        clock_base=clock_base,
        radiation_static=radiation_static,
        land_state=land_state,
    )
    out = solve_dudhia_sw_column(column)
    # (ncol, nz) heating rate -> (nz, ny, nx) dT/dt (K/s).
    heating_rate_T = jnp.moveaxis(out.heating_rate.reshape(ny, nx, nz), -1, 0)
    # T-space heating rate -> theta tendency via the Exner factor
    # (theta = T/exner; for fixed pressure d(theta)/dt = (dT/dt)/exner).
    exner = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    rthraten = heating_rate_T / jnp.maximum(exner, 1.0e-12)
    return rthraten.astype(_output_dtype(state, "theta"))


# --------------------------------------------------------------------------- #
# GSFC (Chou-Suarez) shortwave (``ra_sw_physics=2``) HELD-RATE theta-tendency  #
# coupler. SW-only; the dispatch adds the operational RRTMG/classic-RRTM LW.   #
# --------------------------------------------------------------------------- #
def _gsfc_sw_column_inputs(
    state: State,
    grid: GridSpec | None,
    *,
    time_utc=None,
    lead_seconds=0.0,
    clock_base=None,
    radiation_static: RRTMGRadiationStatic | None = None,
    land_state=None,
) -> tuple[GsfcSWColumnState, SolarGeometry]:
    """Build the GSFC SW column-kernel input view from operational ``State``.

    Mirrors :func:`_dudhia_sw_column_inputs` but assembles the inputs the
    Chou-Suarez multi-band kernel (``module_ra_gsfcsw.F:GSFCSWRAD``) consumes:
    layer ``T`` (K), pressure ``p`` (Pa), interface pressure ``p8w`` (Pa,
    reconstructed WRF-faithfully via :func:`_interface_pressure_from_state`), the
    six moisture species, a diagnostic cloud fraction, per-column cosine-zenith,
    surface ``albedo`` and the date-adjusted total solar constant ``solcon``.
    The ozone climatology is selected from the grid CENTER latitude
    (``grid.projection.lat_0``) and the time-of-year Julian day, exactly as
    WRF's ``gsfc_swinit(cen_lat)`` + ``GSFCSWRAD`` ``iprof`` block.

    The kernel works on flat ``(ncol=ny*nx, nz)`` columns, so the
    ``(nz, ny, nx)`` mass fields are reshaped to the kernel's batch contract.
    """

    T = _temperature_from_theta(state.theta, state.p)
    nz, ny, nx = state.theta.shape
    ncol = ny * nx

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(ncol, nz)

    dz_cols = _column_dz_from_state(state, grid).reshape(ncol, nz)
    # _interface_pressure_from_state returns COLUMN form (ny, nx, nz+1) with the
    # vertical axis trailing and index 0 = surface; flatten to (ncol, nz+1).
    p8w_cols = _interface_pressure_from_state(state).reshape(ncol, nz + 1)
    cloud_fraction = _cloud_fraction_columns(state).reshape(ncol, nz)

    surface_shape = state.t_skin.shape
    surface_albedo, _ = _surface_radiation_properties(state, land_state=land_state)
    static = _radiation_static_for_grid(surface_shape, grid, radiation_static, state.t_skin.dtype)
    if static is None:
        lat, lon = _grid_lat_lon(surface_shape, grid, state.t_skin.dtype)
    else:
        lat, lon = static.xlat_deg, static.xlong_deg
    geometry = _compute_solar_geometry(lat, lon, time_utc, lead_seconds, clock_base=clock_base)
    geometry = SolarGeometry(
        coszen=geometry.coszen.astype(state.t_skin.dtype),
        declination_rad=geometry.declination_rad,
        hour_angle_rad=geometry.hour_angle_rad.astype(state.t_skin.dtype),
    )
    solcon = _solcon_for_time(time_utc, lead_seconds, clock_base=clock_base).astype(state.t_skin.dtype)
    solcon = jnp.broadcast_to(solcon, surface_shape).reshape(ncol)

    # NOTE (#91): ``julian`` here selects the GSFC ozone climatology band
    # ``iprof`` (a discrete int 1-5) via ``_select_iprof`` -> a STATIC array
    # index, so the GSFC SW HLO (ra_sw_physics=2, NON-default) still varies with
    # the season band. The default operational path is RRTMG (ra_sw=4/ra_lw=4)
    # and is fully date-independent via clock_base; the GSFC ozone-band index is
    # a documented residual on the non-default path (would need a traced gather
    # over the small ozone table to remove).
    julian, _ = _time_utc_parts(time_utc)
    # Grid CENTER latitude for the ozone profile (WRF gsfc_swinit cen_lat). The
    # projection lat_0 is the canonical grid center; falls back to 0 (tropical)
    # when no grid is supplied (bare-kernel callers inject their own).
    center_lat = float(grid.projection.lat_0) if grid is not None else 0.0

    column = GsfcSWColumnState(
        T=_cols(T),
        p=_cols(state.p),
        p8w=p8w_cols,
        qv=_cols(state.qv),
        qc=_cols(state.qc),
        qr=_cols(state.qr),
        qi=_cols(state.qi),
        qs=_cols(state.qs),
        qg=_cols(state.qg),
        dz=dz_cols,
        cldfra=cloud_fraction,
        coszen=geometry.coszen.reshape(ncol),
        albedo=jnp.asarray(surface_albedo).reshape(ncol),
        solcon=solcon,
        julday=int(julian),
        center_lat=center_lat,
        f_qi=True,
        warm_rain=False,
    )
    return column, geometry


def gsfc_sw_theta_tendency(
    state: State,
    grid: GridSpec | None = None,
    *,
    time_utc=None,
    lead_seconds=0.0,
    clock_base=None,
    radiation_static: RRTMGRadiationStatic | None = None,
    land_state=None,
) -> "jnp.ndarray":
    """Return the GSFC (``ra_sw_physics=2``) shortwave ``RTHRATEN`` (K/s).

    Shortwave-only HELD-RATE primitive, the GSFC analogue of
    :func:`dudhia_sw_theta_tendency`. WRF's ``GSFCSWRAD`` accumulates the SW
    heating into ``RTHRATEN`` as ``RTHRATEN += max(TTEN,0)/pi3D``: the kernel
    returns the (already non-negative) per-layer TEMPERATURE heating rate
    ``dT/dt`` (K/s) and the theta tendency is that rate divided by the Exner
    factor. The operational scan holds this rate over the ``radt`` interval (the
    same cadence machinery the RRTMG path uses) and ADDS it into theta at every
    dynamics step.

    This coupler is SHORTWAVE-ONLY; the longwave heating still comes from the
    operational RRTMG-LW / classic-RRTM-LW path (the dispatch adds the two
    RTHRATEN contributions), exactly as WRF runs the SW and LW radiation drivers
    independently.

    The returned 3-D field is on the ``(nz, ny, nx)`` mass grid and matches the
    state's theta dtype (fp64 under ``force_fp64``).
    """

    nz, ny, nx = state.theta.shape
    column, _ = _gsfc_sw_column_inputs(
        state,
        grid,
        time_utc=time_utc,
        lead_seconds=lead_seconds,
        clock_base=clock_base,
        radiation_static=radiation_static,
        land_state=land_state,
    )
    out = solve_gsfc_sw_column(column)
    # (ncol, nz) heating rate -> (nz, ny, nx) dT/dt (K/s).
    heating_rate_T = jnp.moveaxis(out.heating_rate.reshape(ny, nx, nz), -1, 0)
    exner = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    rthraten = heating_rate_T / jnp.maximum(exner, 1.0e-12)
    return rthraten.astype(_output_dtype(state, "theta"))


def rrtmg_lw_theta_tendency(
    state: State,
    grid: GridSpec | None = None,
    *,
    time_utc=None,
    lead_seconds=0.0,
    clock_base=None,
    radiation_static: RRTMGRadiationStatic | None = None,
    land_state=None,
) -> "jnp.ndarray":
    """Return the RRTMG longwave-ONLY ``RTHRATEN`` (K/s).

    The LW half of :func:`rrtmg_theta_tendency`, used by the Dudhia-SW dispatch so
    that ``ra_sw_physics=1`` runs Dudhia shortwave + RRTMG longwave (WRF runs the
    SW and LW drivers independently). Reuses the RRTMG column-input assembler and
    LW kernel; only the LW heating rate is converted to a theta tendency.
    """

    _, lw_state, _, _, _, _ = _rrtmg_column_inputs(
        state,
        grid,
        time_utc=time_utc,
        lead_seconds=lead_seconds,
        clock_base=clock_base,
        radiation_static=radiation_static,
        land_state=land_state,
    )
    lw = solve_rrtmg_lw_column(lw_state, debug=False)
    heating_rate_T = _from_columns(lw.heating_rate)  # dT/dt (K/s)
    exner = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    rthraten = heating_rate_T / jnp.maximum(exner, 1.0e-12)
    return rthraten.astype(_output_dtype(state, "theta"))


# --------------------------------------------------------------------------- #
# Classic RRTM longwave (``ra_lw_physics=1``) HELD-RATE theta-tendency coupler.
# --------------------------------------------------------------------------- #
def _interface_temperature_from_state(state: State, T):
    """WRF ``t8w`` (temperature at w-levels) reconstructed from mass-point ``T``.

    Mirrors the WRF model convention (and the pristine-WRF RRTM oracle driver):
    the surface interface is the skin temperature ``t_skin``, interior interfaces
    are the arithmetic mean of the two bounding mass-layer temperatures, and the
    model-top interface is linearly extrapolated from the two topmost mass layers.
    ``T`` is mass-point ``(nz, ny, nx)``; returns interface columns ``(ny, nx, nz+1)``.
    """

    nz = T.shape[0]
    interior = 0.5 * (T[:-1] + T[1:])                      # (nz-1, ny, nx)
    surface = jnp.asarray(state.t_skin).astype(T.dtype)[None]  # (1, ny, nx)
    top = (T[-1] + 0.5 * (T[-1] - T[-2]))[None]            # (1, ny, nx)
    t8w = jnp.concatenate([surface, interior, top], axis=0)  # (nz+1, ny, nx)
    return _to_columns(t8w)


def _rrtm_lw_column_inputs(
    state: State,
    grid: GridSpec | None,
    *,
    land_state=None,
) -> RRTMLWColumnState:
    """Build the classic-RRTM LW column-kernel input view from operational ``State``.

    The traceable RRTM LW kernel (:func:`gpuwrf.physics.ra_lw_rrtm_jax.solve_rrtm_lw_column_jax`)
    consumes layer ``T``/``p`` plus the six moisture species, a diagnostic cloud
    fraction, layer thickness ``dz`` and density ``rho``, AND the interface
    temperature ``t8w`` / interface pressure ``p8w`` (``nz+1`` entries), with the
    LSM surface emissivity ``emiss`` and skin temperature ``tsk``. ``t8w``/``p8w``
    are reconstructed WRF-faithfully from the State here (the State does not carry
    the w-level fields explicitly), exactly as the SW coupler reconstructs ``dz`` /
    ``solcon``. The kernel works on flat ``(ncol=ny*nx, nz)`` columns.
    """

    T = _temperature_from_theta(state.theta, state.p)
    nz, ny, nx = state.theta.shape
    ncol = ny * nx

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(ncol, nz)

    dz_cols = _column_dz_from_state(state, grid).reshape(ncol, nz)
    rho_cols = _to_columns(_rho_from_state(state)).reshape(ncol, nz)
    cloud_fraction = _cloud_fraction_columns(state).reshape(ncol, nz)
    # ``_interface_pressure_from_state`` / ``_interface_temperature_from_state``
    # already return COLUMN-form ``(ny, nx, nz+1)`` (the vertical axis is trailing),
    # so flatten the leading 2-D grid directly to ``(ncol, nz+1)``.
    p8w_cols = _interface_pressure_from_state(state).reshape(ncol, nz + 1)
    t8w_cols = _interface_temperature_from_state(state, T).reshape(ncol, nz + 1)

    _, surface_emissivity = _surface_radiation_properties(state, land_state=land_state)

    # F1: plumb the grid's REAL model-top pressure so the kernel sizes its
    # above-model-top RRTM buffer (nbuf=nint(p_top_mb/4), WRF module_ra_rrtm.F:6781)
    # to THIS grid's top -- not the hardcoded 5000 Pa. It is a STATIC Python float
    # (it sets traced-array shapes), constant-folded into the trace. ``grid is None``
    # (proof / bare-kernel callers) keeps the legacy 5000-Pa fallback (None).
    top_pressure_pa = None
    if grid is not None:
        top_pressure_pa = float(grid.vertical.top_pressure_pa)

    return RRTMLWColumnState(
        T=_cols(T),
        t8w=t8w_cols,
        p=_cols(state.p),
        p8w=p8w_cols,
        qv=_cols(state.qv),
        qc=_cols(state.qc),
        qr=_cols(state.qr),
        qi=_cols(state.qi),
        qs=_cols(state.qs),
        qg=_cols(state.qg),
        cloud_fraction=cloud_fraction,
        dz=dz_cols,
        rho=rho_cols,
        emiss=jnp.asarray(surface_emissivity).reshape(ncol),
        tsk=jnp.asarray(state.t_skin).reshape(ncol),
        top_pressure_pa=top_pressure_pa,
    )


def rrtm_lw_theta_tendency(
    state: State,
    grid: GridSpec | None = None,
    *,
    time_utc=None,
    lead_seconds=0.0,
    radiation_static: RRTMGRadiationStatic | None = None,
    land_state=None,
) -> "jnp.ndarray":
    """Return the classic-RRTM (``ra_lw_physics=1``) longwave ``RTHRATEN`` (K/s).

    Longwave-only HELD-RATE primitive, the classic-RRTM analogue of
    :func:`rrtmg_lw_theta_tendency`. The AER 16-band k-distribution kernel returns
    the per-layer TEMPERATURE heating rate ``dT/dt`` (K/s) and the theta tendency
    is that rate divided by the Exner factor (WRF ``RRTMLWRAD``:
    ``RTHRATEN += TTEN/pi``). The operational scan holds this rate over the
    ``radt`` interval and ADDS it into theta at every dynamics step (shared cadence
    machinery, identical to the RRTMG-LW path).

    This coupler is LONGWAVE-ONLY; the shortwave heating comes from the chosen SW
    scheme (Dudhia or RRTMG), exactly as WRF runs the SW and LW drivers
    independently. The kernel is JIT/vmap-traceable, so this coupler rides the
    device ``jax.lax.scan`` radiation slot with no host callbacks.

    The returned 3-D field is on the ``(nz, ny, nx)`` mass grid and matches the
    state's theta dtype (fp64 under ``force_fp64``).
    """

    del radiation_static  # classic RRTM LW geometry is date-independent (no zenith)
    nz, ny, nx = state.theta.shape
    column = _rrtm_lw_column_inputs(state, grid, land_state=land_state)
    out = solve_rrtm_lw_column_jax(column)
    # (ncol, nz) heating rate -> (nz, ny, nx) dT/dt (K/s).
    heating_rate_T = jnp.moveaxis(out.heating_rate.reshape(ny, nx, nz), -1, 0)
    # T-space heating rate -> theta tendency via the Exner factor
    # (theta = T/exner; for fixed pressure d(theta)/dt = (dT/dt)/exner).
    exner = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    rthraten = heating_rate_T / jnp.maximum(exner, 1.0e-12)
    return rthraten.astype(_output_dtype(state, "theta"))


# --------------------------------------------------------------------------- #
# Held-Suarez idealized radiation (``ra_lw_physics=31``) HELD-RATE theta         #
# tendency coupler. COMBINED LW+SW: HSRAD supplies the entire radiative          #
# tendency (WRF makes no separate SW call for this scheme), so the dispatch      #
# requires ra_sw_physics=0 and uses this coupler as the sole radiation source.   #
# --------------------------------------------------------------------------- #
def held_suarez_theta_tendency(
    state: State,
    grid: GridSpec | None = None,
    *,
    time_utc=None,
    lead_seconds=0.0,
    radiation_static: RRTMGRadiationStatic | None = None,
    land_state=None,
) -> "jnp.ndarray":
    """Return the Held-Suarez (``ra_lw_physics=31``) radiative ``RTHRATEN`` (K/s).

    Combined idealized LW+SW HELD-RATE primitive (``module_ra_hs.F:HSRAD``): a
    Newtonian relaxation of temperature toward an analytic radiative-equilibrium
    profile. WRF accumulates ``RTHRATEN += t_tend/pi`` where ``t_tend`` is the
    per-layer kinetic-temperature relaxation rate (K/s); this coupler runs the
    column kernel and divides the returned ``dT/dt`` by the Exner factor, exactly
    as the Dudhia/GSFC/RRTM held-rate couplers do.

    Unlike the band-model radiation schemes, Held-Suarez consumes NO solar
    geometry, moisture, cloud, albedo, or ozone -- only layer ``T``, layer
    pressure, the surface interface pressure (``p8w(i,1,j)``) and the column
    latitude (preferring the authoritative ``radiation_static.xlat_deg`` when a
    per-run static bundle is supplied, else the GridSpec lat approximation).
    ``time_utc``/``lead_seconds`` are unused (the forcing is time-independent).

    This is a COMBINED LW+SW scheme; the radiation dispatch fail-closes any SW
    selection when ``ra_lw_physics=31`` (HSRAD is the only radiative call), so
    there is no second SW contribution to add. The returned 3-D field is on the
    ``(nz, ny, nx)`` mass grid and matches the state's theta dtype (fp64 under
    ``force_fp64``).
    """

    del time_utc, lead_seconds  # Held-Suarez forcing is time-independent.
    nz, ny, nx = state.theta.shape
    ncol = ny * nx

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(ncol, nz)

    T = _temperature_from_theta(state.theta, state.p)
    # Surface interface pressure p8w(i,1,j): index 0 of the reconstructed columns.
    p8w_cols = _interface_pressure_from_state(state).reshape(ncol, nz + 1)
    psfc = p8w_cols[:, 0]

    surface_shape = state.t_skin.shape
    static = _radiation_static_for_grid(surface_shape, grid, radiation_static, state.t_skin.dtype)
    if static is None:
        lat, _lon = _grid_lat_lon(surface_shape, grid, state.t_skin.dtype)
    else:
        lat = static.xlat_deg
    lat_deg = jnp.asarray(lat).reshape(ncol)

    column = HeldSuarezColumnState(
        T=_cols(T),
        p=_cols(state.p),
        psfc=psfc,
        lat_deg=lat_deg,
    )
    out = solve_held_suarez_column(column)
    # (ncol, nz) dT/dt -> (nz, ny, nx); theta tendency = rate / Exner.
    heating_rate_T = jnp.moveaxis(out.heating_rate.reshape(ny, nx, nz), -1, 0)
    exner = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    rthraten = heating_rate_T / jnp.maximum(exner, 1.0e-12)
    return rthraten.astype(_output_dtype(state, "theta"))


# --------------------------------------------------------------------------- #
# Orographic gravity-wave drag (GWDO, ``gwd_opt=1``) adapter.
# --------------------------------------------------------------------------- #
def build_gwdo_statics_from_wrf_fields(
    var2d,
    con,
    oa1,
    oa2,
    oa3,
    oa4,
    ol1,
    ol2,
    ol3,
    ol4,
    *,
    dx_m: float,
    sina=None,
    cosa=None,
    dtype=jnp.float64,
) -> GWDOStatics:
    """Assemble the per-run GWDO sub-grid orography statics from WRF fields.

    The ten 2-D statics are the geo_em / ``wrfinput`` fields carried by the
    port's ``init/metgrid_schema.py`` (Registry package ``gwd_used_1``):

        var2d <- ``VAR`` (std dev of subgrid orography, m)
        con   <- ``CON`` (orographic convexity -> WRF ``oc1``)
        oa1..4 <- ``OA1..4``;  ol1..4 <- ``OL1..4``

    Inputs are mass-point 2-D ``(ny, nx)``; the bundle is flattened to the
    kernel's ``(B=ny*nx,)`` batch contract. ``sina``/``cosa`` default to the
    unrotated grid (0/1). ``dx_m`` is the (uniform) grid spacing in metres.
    """

    def flat(field, default=None):
        if field is None:
            arr = jnp.full(shape, float(default), dtype=dtype)
        else:
            arr = jnp.asarray(field, dtype=dtype)
        return arr.reshape((-1,))

    var2d = jnp.asarray(var2d, dtype=dtype)
    shape = var2d.shape
    return GWDOStatics(
        var=var2d.reshape((-1,)),
        oc1=flat(con),
        oa1=flat(oa1),
        oa2=flat(oa2),
        oa3=flat(oa3),
        oa4=flat(oa4),
        ol1=flat(ol1),
        ol2=flat(ol2),
        ol3=flat(ol3),
        ol4=flat(ol4),
        sina=flat(sina, 0.0),
        cosa=flat(cosa, 1.0),
        dxmeter=jnp.full((var2d.size,), float(dx_m), dtype=dtype),
    )


def _interface_pressure_from_state(state: State):
    """WRF ``p8w`` (full pressure at w-levels) reconstructed from mass-point ``p``.

    GWDO consumes interface pressure only through the layer mass
    ``del(k)=prsi(k)-prsi(k+1)`` and the column-fraction normaliser ``delks``;
    it never differences a single interface against a level. The interface
    pressures are the standard log-linear midpoints of the mass-point pressures,
    with the surface (k=0) and model-top (k=K) faces extrapolated log-linearly
    from the two nearest mass levels (matching WRF ``p8w`` at the boundaries to
    within hydrostatic round-off). Pressure decreases monotonically with height,
    so the reconstruction preserves ``del(k) > 0``.

    ``state.p`` is mass-point ``(nz, ny, nx)``; returns interface columns
    ``(ny, nx, nz+1)``.
    """

    p = state.p.astype(jnp.float64)  # (nz, ny, nx)
    logp = jnp.log(jnp.maximum(p, 1.0))
    # interior interfaces: geometric mean of adjacent mass levels
    interior = jnp.exp(0.5 * (logp[:-1] + logp[1:]))  # (nz-1, ny, nx)
    # surface face: log-linear extrapolation below level 0 using levels 0,1
    bottom = jnp.exp(1.5 * logp[0] - 0.5 * logp[1])[None]  # (1, ny, nx)
    # top face: log-linear extrapolation above the top level using K-2,K-1
    top = jnp.exp(1.5 * logp[-1] - 0.5 * logp[-2])[None]
    prsi = jnp.concatenate([bottom, interior, top], axis=0)  # (nz+1, ny, nx)
    # guard strict monotone decrease so del(k) stays positive.
    prsi = jnp.maximum(prsi, 1.0)
    return _to_columns(prsi)


def gwdo_adapter(
    state: State,
    dt: float,
    statics: GWDOStatics,
    grid: GridSpec | None = None,
) -> State:
    """Apply orographic gravity-wave drag + flow-blocking (``gwd_opt=1``).

    Thin adapter: builds the GWDO column view from State (mass-point winds, T,
    qv, mid/interface pressure, exner, geopotential height), runs the faithful
    :func:`gpuwrf.physics.gwd_gwdo.gwdo_columns` kernel, and adds the resulting
    A-grid wind tendency increment onto the C-grid faces using the same WRF
    ``add_a2c_u/v`` averaging as the MYNN coupler (:func:`_add_a2c_u_increment`).
    Momentum-only: theta/qv/qke/w are untouched (GWDO produces no heating).

    ``statics`` is the per-run :class:`GWDOStatics` bundle
    (:func:`build_gwdo_statics_from_wrf_fields`).
    """

    T = _temperature_from_theta(state.theta, state.p)
    exner = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    # geopotential height of mass-points (m): average of the two bounding faces.
    z_face = state.ph.astype(jnp.float64) / GRAVITY_M_S2  # (nz+1, ny, nx)
    z_mass = 0.5 * (z_face[:-1] + z_face[1:])

    u_mass = _u_mass(state)
    v_mass = _v_mass(state)
    ny, nx = state.theta.shape[1], state.theta.shape[2]

    column = GWDOColumnState(
        uproj=_to_columns(u_mass).reshape((ny * nx, -1)),
        vproj=_to_columns(v_mass).reshape((ny * nx, -1)),
        t1=_to_columns(T).reshape((ny * nx, -1)),
        q1=_to_columns(state.qv).reshape((ny * nx, -1)),
        prsl=_to_columns(state.p).reshape((ny * nx, -1)),
        prsi=_interface_pressure_from_state(state).reshape((ny * nx, -1)),
        prslk=_to_columns(exner).reshape((ny * nx, -1)),
        zl=_to_columns(z_mass).reshape((ny * nx, -1)),
    )
    out = gwdo_columns(column, statics, dt)

    # column tendency (m/s^2) -> wind increment over the step on mass points.
    du_mass = _from_columns(out.rublten.reshape((ny, nx, -1))) * dt  # (nz, ny, nx)
    dv_mass = _from_columns(out.rvblten.reshape((ny, nx, -1))) * dt
    u_new = _add_a2c_u_increment(state.u, du_mass).astype(_output_dtype(state, "u"))
    v_new = _add_a2c_v_increment(state.v, dv_mass).astype(_output_dtype(state, "v"))
    return state.replace(u=u_new, v=v_new)


__all__ = [
    "GWDOStatics",
    "RRTMGRadiationDiagnostics",
    "RRTMGRadiationStatic",
    "SurfaceMynnDiagnostics",
    "ThompsonTendencySideChannel",
    "_compute_coszen",
    "_compute_solar_geometry",
    "build_radiation_static_from_wrf_fields",
    "build_gwdo_statics_from_wrf_fields",
    "wrf_radiation_slope_aspect_from_terrain",
    "gwdo_adapter",
    "mynn_adapter",
    "mynn_adapter_with_source_leaves",
    "mynn_coldstart_qke_from_state",
    "mynn_adapter_with_diagnostics",
    "rrtmg_radiation_diagnostics",
    "rrtmg_adapter",
    "rrtmg_theta_tendency",
    "rrtmg_lw_theta_tendency",
    "rrtmg_sw_theta_tendency",
    "rrtm_lw_theta_tendency",
    "dudhia_sw_theta_tendency",
    "gsfc_sw_theta_tendency",
    "held_suarez_theta_tendency",
    "surface_adapter",
    "surface_layer_diagnostics",
    "thompson_adapter",
    "thompson_adapter_with_tendencies",
    "HAIL_MP_FAMILY",
    "hail_mp_adapter",
    "thompson_aero_adapter",
    "thompson_aero_coldstart_init",
]
