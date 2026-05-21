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
    "sw_reference_pressure_pa",
    "sw_preflog",
    "sw_tref",
    "sw_gpoint_mask",
    "sw_gpoint_weights",
    "sw_absorption_coefficients",
    "sw_rayleigh_coefficients",
    "sw_cloud_liquid_extinction",
    "sw_cloud_ice_extinction",
    "sw_cloud_liquid_ssa",
    "sw_cloud_ice_ssa",
    "sw_cloud_liquid_asymmetry",
    "sw_cloud_ice_asymmetry",
    "sw_nspa",
    "sw_nspb",
    "sw_absa",
    "sw_absb",
    "sw_selfref",
    "sw_forref",
    "sw_sfluxref",
    "sw_rayl",
    "sw_rayl_scalar",
    "sw_rayla",
    "sw_raylb",
    "sw_abs_ch4",
    "sw_abs_o3a",
    "sw_abs_o3b",
    "sw_abs_h2o",
    "sw_abs_co2",
    "sw_strrat",
    "sw_layreffr",
    "sw_givfac",
    "sw_scalekur",
    "lw_band_weights",
    "lw_reference_pressure_pa",
    "lw_preflog",
    "lw_tref",
    "lw_gpoint_mask",
    "lw_gpoint_weights",
    "lw_absorption_coefficients",
    "lw_cloud_absorption",
    "lw_totplnk",
    "lw_totplk16",
    "lw_delwave",
    "gas_vmr_defaults",
    "cloud_optical_defaults",
)


class RRTMGTableBundle(NamedTuple):
    """Runtime table bundle passed as JAX array leaves to fused kernels."""

    sw_band_weights: jnp.ndarray
    sw_reference_pressure_pa: jnp.ndarray
    sw_preflog: jnp.ndarray
    sw_tref: jnp.ndarray
    sw_gpoint_mask: jnp.ndarray
    sw_gpoint_weights: jnp.ndarray
    sw_absorption_coefficients: jnp.ndarray
    sw_rayleigh_coefficients: jnp.ndarray
    sw_cloud_liquid_extinction: jnp.ndarray
    sw_cloud_ice_extinction: jnp.ndarray
    sw_cloud_liquid_ssa: jnp.ndarray
    sw_cloud_ice_ssa: jnp.ndarray
    sw_cloud_liquid_asymmetry: jnp.ndarray
    sw_cloud_ice_asymmetry: jnp.ndarray
    sw_nspa: jnp.ndarray
    sw_nspb: jnp.ndarray
    sw_absa: jnp.ndarray
    sw_absb: jnp.ndarray
    sw_selfref: jnp.ndarray
    sw_forref: jnp.ndarray
    sw_sfluxref: jnp.ndarray
    sw_rayl: jnp.ndarray
    sw_rayl_scalar: jnp.ndarray
    sw_rayla: jnp.ndarray
    sw_raylb: jnp.ndarray
    sw_abs_ch4: jnp.ndarray
    sw_abs_o3a: jnp.ndarray
    sw_abs_o3b: jnp.ndarray
    sw_abs_h2o: jnp.ndarray
    sw_abs_co2: jnp.ndarray
    sw_strrat: jnp.ndarray
    sw_layreffr: jnp.ndarray
    sw_givfac: jnp.ndarray
    sw_scalekur: jnp.ndarray
    lw_band_weights: jnp.ndarray
    lw_reference_pressure_pa: jnp.ndarray
    lw_preflog: jnp.ndarray
    lw_tref: jnp.ndarray
    lw_gpoint_mask: jnp.ndarray
    lw_gpoint_weights: jnp.ndarray
    lw_absorption_coefficients: jnp.ndarray
    lw_cloud_absorption: jnp.ndarray
    lw_totplnk: jnp.ndarray
    lw_totplk16: jnp.ndarray
    lw_delwave: jnp.ndarray
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
        sw_reference_pressure_pa=jnp.asarray(arrays["sw_reference_pressure_pa"], dtype=jnp.float64),
        sw_preflog=jnp.asarray(arrays["sw_preflog"], dtype=jnp.float64),
        sw_tref=jnp.asarray(arrays["sw_tref"], dtype=jnp.float64),
        sw_gpoint_mask=jnp.asarray(arrays["sw_gpoint_mask"], dtype=jnp.float64),
        sw_gpoint_weights=jnp.asarray(arrays["sw_gpoint_weights"], dtype=jnp.float64),
        sw_absorption_coefficients=jnp.asarray(arrays["sw_absorption_coefficients"], dtype=jnp.float64),
        sw_rayleigh_coefficients=jnp.asarray(arrays["sw_rayleigh_coefficients"], dtype=jnp.float64),
        sw_cloud_liquid_extinction=jnp.asarray(arrays["sw_cloud_liquid_extinction"], dtype=jnp.float64),
        sw_cloud_ice_extinction=jnp.asarray(arrays["sw_cloud_ice_extinction"], dtype=jnp.float64),
        sw_cloud_liquid_ssa=jnp.asarray(arrays["sw_cloud_liquid_ssa"], dtype=jnp.float64),
        sw_cloud_ice_ssa=jnp.asarray(arrays["sw_cloud_ice_ssa"], dtype=jnp.float64),
        sw_cloud_liquid_asymmetry=jnp.asarray(arrays["sw_cloud_liquid_asymmetry"], dtype=jnp.float64),
        sw_cloud_ice_asymmetry=jnp.asarray(arrays["sw_cloud_ice_asymmetry"], dtype=jnp.float64),
        sw_nspa=jnp.asarray(arrays["sw_nspa"], dtype=jnp.int32),
        sw_nspb=jnp.asarray(arrays["sw_nspb"], dtype=jnp.int32),
        sw_absa=jnp.asarray(arrays["sw_absa"], dtype=jnp.float64),
        sw_absb=jnp.asarray(arrays["sw_absb"], dtype=jnp.float64),
        sw_selfref=jnp.asarray(arrays["sw_selfref"], dtype=jnp.float64),
        sw_forref=jnp.asarray(arrays["sw_forref"], dtype=jnp.float64),
        sw_sfluxref=jnp.asarray(arrays["sw_sfluxref"], dtype=jnp.float64),
        sw_rayl=jnp.asarray(arrays["sw_rayl"], dtype=jnp.float64),
        sw_rayl_scalar=jnp.asarray(arrays["sw_rayl_scalar"], dtype=jnp.float64),
        sw_rayla=jnp.asarray(arrays["sw_rayla"], dtype=jnp.float64),
        sw_raylb=jnp.asarray(arrays["sw_raylb"], dtype=jnp.float64),
        sw_abs_ch4=jnp.asarray(arrays["sw_abs_ch4"], dtype=jnp.float64),
        sw_abs_o3a=jnp.asarray(arrays["sw_abs_o3a"], dtype=jnp.float64),
        sw_abs_o3b=jnp.asarray(arrays["sw_abs_o3b"], dtype=jnp.float64),
        sw_abs_h2o=jnp.asarray(arrays["sw_abs_h2o"], dtype=jnp.float64),
        sw_abs_co2=jnp.asarray(arrays["sw_abs_co2"], dtype=jnp.float64),
        sw_strrat=jnp.asarray(arrays["sw_strrat"], dtype=jnp.float64),
        sw_layreffr=jnp.asarray(arrays["sw_layreffr"], dtype=jnp.int32),
        sw_givfac=jnp.asarray(arrays["sw_givfac"], dtype=jnp.float64),
        sw_scalekur=jnp.asarray(arrays["sw_scalekur"], dtype=jnp.float64),
        lw_band_weights=jnp.asarray(arrays["lw_band_weights"], dtype=jnp.float64),
        lw_reference_pressure_pa=jnp.asarray(arrays["lw_reference_pressure_pa"], dtype=jnp.float64),
        lw_preflog=jnp.asarray(arrays["lw_preflog"], dtype=jnp.float64),
        lw_tref=jnp.asarray(arrays["lw_tref"], dtype=jnp.float64),
        lw_gpoint_mask=jnp.asarray(arrays["lw_gpoint_mask"], dtype=jnp.float64),
        lw_gpoint_weights=jnp.asarray(arrays["lw_gpoint_weights"], dtype=jnp.float64),
        lw_absorption_coefficients=jnp.asarray(arrays["lw_absorption_coefficients"], dtype=jnp.float64),
        lw_cloud_absorption=jnp.asarray(arrays["lw_cloud_absorption"], dtype=jnp.float64),
        lw_totplnk=jnp.asarray(arrays["lw_totplnk"], dtype=jnp.float64),
        lw_totplk16=jnp.asarray(arrays["lw_totplk16"], dtype=jnp.float64),
        lw_delwave=jnp.asarray(arrays["lw_delwave"], dtype=jnp.float64),
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
    "sw_gpoint_reduction": "module_ra_rrtmg_sw.F:4763-4784",
    "sw_gpoint_groups": "module_ra_rrtmg_sw.F:4927-5027",
    "sw_cloud_delta_scaling": "module_ra_rrtmg_sw.F:2388-2428",
    "sw_setcoef": "module_ra_rrtmg_sw.F:2843-3099",
    "sw_taumol": "module_ra_rrtmg_sw.F:3190-4653",
    "lw_band_count": "module_ra_rrtmg_lw.F:76-82",
    "lw_driver": "module_ra_rrtmg_lw.F:11570-11607",
    "lw_driver_calls_core": "module_ra_rrtmg_lw.F:12768-12778",
    "lw_data_open": "module_ra_rrtmg_lw.F:13046-13067",
    "lw_data_records": "module_ra_rrtmg_lw.F:13085-13090",
    "lw_gpoint_reduction": "module_ra_rrtmg_lw.F:8073-8104",
    "lw_gpoint_groups": "module_ra_rrtmg_lw.F:8244-8315",
    "lw_cloud_absorption": "module_ra_rrtmg_lw.F:2997-3018",
    "lw_setcoef_planck": "module_ra_rrtmg_lw.F:3556-3921",
    "lw_rtrnmc_planck_source": "module_ra_rrtmg_lw.F:3270-3340",
}
