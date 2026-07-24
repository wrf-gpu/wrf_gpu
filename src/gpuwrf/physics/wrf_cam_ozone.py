"""WRF CAM monthly ozone interpolation used by ``o3input=2``.

This is a direct, float32 transcription of the WRF chain
``oznini -> ozn_time_int -> ozn_p_int``.  The source tables are extracted
verbatim from WRF's ``run/ozone*.formatted`` files into a deterministic,
hash-pinned runtime asset.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OZONE_ASSET = ROOT / "data" / "fixtures" / "wrf-cam-ozone-v1.npz"
OZONE_ASSET_SHA256 = "621981243488618988b5d8b5926e981c5658d4a82ebabdf8c680874f3a3fec95"
_DATE_OZ = np.asarray(
    [16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350],
    dtype=np.float32,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_wrf_cam_ozone_asset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the immutable WRF ozone table asset."""

    if _sha256(OZONE_ASSET) != OZONE_ASSET_SHA256:
        raise RuntimeError(f"WRF CAM ozone asset hash drift: {OZONE_ASSET}")
    with np.load(OZONE_ASSET, allow_pickle=False) as archive:
        if set(archive.files) != {"latitude_deg", "pressure_pa", "ozone_vmr"}:
            raise RuntimeError(f"WRF CAM ozone asset schema drift: {archive.files}")
        latitude = np.asarray(archive["latitude_deg"], dtype=np.float32)
        pressure = np.asarray(archive["pressure_pa"], dtype=np.float32)
        ozone = np.asarray(archive["ozone_vmr"], dtype=np.float32)
    if latitude.shape != (64,) or pressure.shape != (59,) or ozone.shape != (12, 64, 59):
        raise RuntimeError(
            "WRF CAM ozone asset shape drift: "
            f"latitude={latitude.shape}, pressure={pressure.shape}, ozone={ozone.shape}"
        )
    if not np.all(np.diff(latitude) > 0.0) or not np.all(np.diff(pressure) > 0.0):
        raise RuntimeError("WRF CAM ozone coordinates must be strictly increasing")
    if not all(np.isfinite(value).all() for value in (latitude, pressure, ozone)):
        raise RuntimeError("WRF CAM ozone asset contains non-finite values")
    return latitude, pressure, ozone


def wrf_ozn_time_indices_factors(
    julian_day_1based,
    utc_minute,
    lead_seconds=0.0,
):
    """Return WRF ``ozn_time_int`` month indices and interpolation factors.

    ``julian_day_1based`` is the host clock's day-of-year (Jan 1 == 1).  WRF's
    grid clock is zero-based and ``ozn_time_int`` immediately adds one, so this
    representation reaches the source expression without a lossy round trip.
    All arithmetic intentionally uses WRF REAL/float32.
    """

    dtype = jnp.float32
    day = jnp.asarray(julian_day_1based, dtype=dtype) + (
        jnp.asarray(utc_minute, dtype=dtype)
        + jnp.asarray(lead_seconds, dtype=dtype) / jnp.asarray(60.0, dtype=dtype)
    ) / jnp.asarray(1440.0, dtype=dtype)
    ijul = day.astype(jnp.int32)
    fraction = day - ijul.astype(dtype)
    ijul = jnp.mod(ijul, jnp.asarray(365, dtype=jnp.int32))
    ijul = jnp.where(ijul == 0, jnp.asarray(365, dtype=jnp.int32), ijul)
    day = fraction + ijul.astype(dtype)

    dates = jnp.asarray(_DATE_OZ, dtype=dtype)
    after = dates > day
    month_plus = jnp.where(jnp.any(after), jnp.argmax(after), 0).astype(jnp.int32)
    month_minus = jnp.where(month_plus > 0, month_plus - 1, 11).astype(jnp.int32)
    cday_plus = jnp.take(dates, month_plus)
    cday_minus = jnp.take(dates, month_minus)

    december_january = month_plus == 0
    delta_cyclic = cday_plus + jnp.asarray(365.0, dtype=dtype) - cday_minus
    factor1_cyclic = jnp.where(
        day > cday_plus,
        (cday_plus + jnp.asarray(365.0, dtype=dtype) - day) / delta_cyclic,
        (cday_plus - day) / delta_cyclic,
    )
    factor2_cyclic = jnp.where(
        day > cday_plus,
        (day - cday_minus) / delta_cyclic,
        (day + jnp.asarray(365.0, dtype=dtype) - cday_minus) / delta_cyclic,
    )
    delta = cday_plus - cday_minus
    factor1 = jnp.where(december_january, factor1_cyclic, (cday_plus - day) / delta)
    factor2 = jnp.where(december_january, factor2_cyclic, (day - cday_minus) / delta)
    return month_minus, month_plus, factor1.astype(dtype), factor2.astype(dtype), day


def wrf_cam_ozone_profile(
    latitude_deg,
    pressure_pa,
    *,
    julian_day_1based,
    utc_minute,
    lead_seconds=0.0,
):
    """Interpolate WRF CAM ozone VMR to mass-grid columns.

    Parameters follow the production radiation view: ``latitude_deg`` has the
    column leading shape and ``pressure_pa`` has that shape plus a final
    bottom-to-top layer axis.  The result is WRF REAL/float32 O3 volume mixing
    ratio with the same shape as ``pressure_pa``.
    """

    latitude_table_np, pressure_table_np, ozone_table_np = _load_wrf_cam_ozone_asset()
    dtype = jnp.float32
    latitude_table = jnp.asarray(latitude_table_np, dtype=dtype)
    pressure_table = jnp.asarray(pressure_table_np, dtype=dtype)
    ozone_table = jnp.asarray(ozone_table_np, dtype=dtype)
    latitude = jnp.asarray(latitude_deg, dtype=dtype)
    pressure = jnp.asarray(pressure_pa, dtype=dtype)
    if pressure.ndim != latitude.ndim + 1 or pressure.shape[:-1] != latitude.shape:
        raise ValueError(
            "WRF CAM ozone expects pressure shape latitude.shape + (nz,), got "
            f"latitude={latitude.shape}, pressure={pressure.shape}"
        )

    # module_ra_cam_support.F::lin_interpol2, including endpoint extrapolation.
    latitude_index = jnp.searchsorted(latitude_table, latitude, side="left") - 1
    latitude_index = jnp.clip(latitude_index, 0, latitude_table.shape[0] - 2)
    latitude0 = jnp.take(latitude_table, latitude_index)
    latitude1 = jnp.take(latitude_table, latitude_index + 1)
    ozone0 = jnp.take(ozone_table, latitude_index, axis=1)
    ozone1 = jnp.take(ozone_table, latitude_index + 1, axis=1)
    slope = (ozone1 - ozone0) / (latitude1 - latitude0)[None, ..., None]
    ozone_monthly = ozone0 + slope * (latitude - latitude0)[None, ..., None]

    month_minus, month_plus, factor1, factor2, _ = wrf_ozn_time_indices_factors(
        julian_day_1based,
        utc_minute,
        lead_seconds,
    )
    ozone_time = (
        jnp.take(ozone_monthly, month_minus, axis=0) * factor1
        + jnp.take(ozone_monthly, month_plus, axis=0) * factor2
    ).astype(dtype)

    # module_radiation_driver.F::ozn_p_int, pointwise equivalent to its
    # top-down loop and carried ``kupper`` search accelerator.
    pressure_index = jnp.searchsorted(pressure_table, pressure, side="left") - 1
    pressure_index = jnp.clip(pressure_index, 0, pressure_table.shape[0] - 2)
    pressure0 = jnp.take(pressure_table, pressure_index)
    pressure1 = jnp.take(pressure_table, pressure_index + 1)
    ozone_p0 = jnp.take_along_axis(ozone_time, pressure_index, axis=-1)
    ozone_p1 = jnp.take_along_axis(ozone_time, pressure_index + 1, axis=-1)
    dpu = pressure - pressure0
    dpl = pressure1 - pressure
    interpolated = (ozone_p0 * dpl + ozone_p1 * dpu) / (dpl + dpu)
    above = jnp.take(ozone_time, 0, axis=-1)[..., None] * pressure / pressure_table[0]
    below = jnp.take(ozone_time, -1, axis=-1)[..., None]
    return jnp.where(
        pressure < pressure_table[0],
        above,
        jnp.where(pressure > pressure_table[-1], below, interpolated),
    ).astype(dtype)


__all__ = [
    "OZONE_ASSET",
    "OZONE_ASSET_SHA256",
    "wrf_cam_ozone_profile",
    "wrf_ozn_time_indices_factors",
]
