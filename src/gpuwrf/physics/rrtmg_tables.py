"""Device-resident spectral tables for the M5-S3 RRTMG kernels."""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import NamedTuple

from jax import config
import jax.numpy as jnp
import numpy as np


config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
TABLE_ASSET = ROOT / "data" / "fixtures" / "rrtmg-tables-v1.npz"

ASSET_TABLE_NAMES = (
    "sw_band_weights",
    "sw_absorption_coefficients",
    "sw_rayleigh_coefficients",
    "sw_cloud_liquid_extinction",
    "sw_cloud_ice_extinction",
    "lw_band_weights",
    "lw_absorption_coefficients",
    "lw_cloud_absorption",
    "gas_vmr_defaults",
    "cloud_optical_defaults",
)


class RRTMGTableBundle(NamedTuple):
    """Runtime table bundle passed as JAX array leaves to fused kernels."""

    sw_band_weights: jnp.ndarray
    sw_absorption_coefficients: jnp.ndarray
    sw_rayleigh_coefficients: jnp.ndarray
    sw_cloud_liquid_extinction: jnp.ndarray
    sw_cloud_ice_extinction: jnp.ndarray
    lw_band_weights: jnp.ndarray
    lw_absorption_coefficients: jnp.ndarray
    lw_cloud_absorption: jnp.ndarray
    gas_vmr_defaults: jnp.ndarray
    cloud_optical_defaults: jnp.ndarray


def asset_sha256(path: Path = TABLE_ASSET) -> str:
    """Returns the SHA-256 digest for the extracted table asset."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _load_npz(path: str) -> dict[str, np.ndarray]:
    """Loads the extracted RRTMG table asset once per interpreter."""

    asset = Path(path)
    if not asset.exists():
        raise FileNotFoundError(f"missing RRTMG table asset: {asset}")
    with np.load(asset, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name], dtype=np.float64) for name in ASSET_TABLE_NAMES}


def load_rrtmg_tables(path: Path = TABLE_ASSET) -> RRTMGTableBundle:
    """Loads RRTMG lookup arrays as JAX leaves, not closed-over constants."""

    arrays = _load_npz(str(path))
    return RRTMGTableBundle(
        sw_band_weights=jnp.asarray(arrays["sw_band_weights"], dtype=jnp.float64),
        sw_absorption_coefficients=jnp.asarray(arrays["sw_absorption_coefficients"], dtype=jnp.float64),
        sw_rayleigh_coefficients=jnp.asarray(arrays["sw_rayleigh_coefficients"], dtype=jnp.float64),
        sw_cloud_liquid_extinction=jnp.asarray(arrays["sw_cloud_liquid_extinction"], dtype=jnp.float64),
        sw_cloud_ice_extinction=jnp.asarray(arrays["sw_cloud_ice_extinction"], dtype=jnp.float64),
        lw_band_weights=jnp.asarray(arrays["lw_band_weights"], dtype=jnp.float64),
        lw_absorption_coefficients=jnp.asarray(arrays["lw_absorption_coefficients"], dtype=jnp.float64),
        lw_cloud_absorption=jnp.asarray(arrays["lw_cloud_absorption"], dtype=jnp.float64),
        gas_vmr_defaults=jnp.asarray(arrays["gas_vmr_defaults"], dtype=jnp.float64),
        cloud_optical_defaults=jnp.asarray(arrays["cloud_optical_defaults"], dtype=jnp.float64),
    )


RRTMG_TABLES = load_rrtmg_tables()


TABLE_SOURCE_LINES = {
    "sw_band_count": "module_ra_rrtmg_sw.F:31-37",
    "sw_driver": "module_ra_rrtmg_sw.F:10034-10100",
    "sw_driver_calls_core": "module_ra_rrtmg_sw.F:11462-11484",
    "sw_data_open": "module_ra_rrtmg_sw.F:11667-11685",
    "sw_data_records": "module_ra_rrtmg_sw.F:11705-11710",
    "lw_band_count": "module_ra_rrtmg_lw.F:76-82",
    "lw_driver": "module_ra_rrtmg_lw.F:11570-11607",
    "lw_driver_calls_core": "module_ra_rrtmg_lw.F:12768-12778",
    "lw_data_open": "module_ra_rrtmg_lw.F:13046-13067",
    "lw_data_records": "module_ra_rrtmg_lw.F:13085-13090",
}
