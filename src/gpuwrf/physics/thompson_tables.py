"""Device-resident WRF Thompson lookup tables exported from the Fortran module."""

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
TABLE_ASSET = ROOT / "data" / "fixtures" / "thompson-tables-v1.npz"


ASSET_TABLE_NAMES = (
    "r_c",
    "r_i",
    "r_r",
    "r_s",
    "r_g",
    "n0r_exp",
    "n0g_exp",
    "nt_i",
    "nt_in",
    "dr",
    "dc",
    "t_nc",
    "t_Efrw",
    "t_Efsw",
    "tps_iaus",
    "tni_iaus",
    "tpi_ide",
    "tpi_qrfz",
    "tpg_qrfz",
    "tni_qrfz",
    "tnr_qrfz",
    "snow_sa",
    "snow_sb",
    "cse",
    "csg",
    "graupel_cge",
    "graupel_cgg",
    "am_g",
    "av_g",
    "bv_g",
    "rho_g",
)


class ThompsonTableBundle(NamedTuple):
    """Runtime table bundle used by the JAX Thompson lookup paths."""

    t_Efrw: jnp.ndarray
    iaus: jnp.ndarray
    qrfz: jnp.ndarray
    snow_sa: jnp.ndarray
    snow_sb: jnp.ndarray
    cse: jnp.ndarray
    # v0.15 riming: snow-collecting-cloud-water collection efficiency
    # (WRF table_Efsw, indexed (snow-bin, cloud-mvd-micron)).
    t_Efsw: jnp.ndarray


R_R_FIRST = 9.999999974752427e-07
R_I_FIRST = 1.000000013351432e-10
NT_I_FIRST = 1.0
DR_FIRST = 5.11646483279726e-05
DR_LAST = 0.004886186104161879
N_R_TABLE = 37
N_R1_TABLE = 37
N_I_TABLE = 64
N_I1_TABLE = 55
N_TC_TABLE = 45
N_IN_TABLE = 55
N_EFRW_R = 100
N_EFRW_C = 100


def asset_sha256(path: Path = TABLE_ASSET) -> str:
    """Returns the table asset digest pinned in the fixture manifest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _load_npz(path: str) -> dict[str, np.ndarray]:
    """Loads the extracted table asset once per interpreter."""

    asset = Path(path)
    if not asset.exists():
        raise FileNotFoundError(f"missing Thompson table asset: {asset}")
    with np.load(asset, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name], dtype=np.float64) for name in ASSET_TABLE_NAMES}


def load_thompson_tables(path: Path = TABLE_ASSET) -> ThompsonTableBundle:
    """Loads WRF tables as JAX arrays at module import time."""

    arrays = _load_npz(str(path))
    iaus = np.stack((arrays["tps_iaus"], arrays["tni_iaus"], arrays["tpi_ide"]), axis=-1)
    default_in_index = 27
    qrfz = np.stack(
        (
            arrays["tpi_qrfz"][:, :, :, default_in_index],
            arrays["tpg_qrfz"][:, :, :, default_in_index],
            arrays["tni_qrfz"][:, :, :, default_in_index],
            arrays["tnr_qrfz"][:, :, :, default_in_index],
        ),
        axis=-1,
    ).reshape(-1, 4)
    return ThompsonTableBundle(
        t_Efrw=jnp.asarray(arrays["t_Efrw"], dtype=jnp.float64),
        iaus=jnp.asarray(iaus, dtype=jnp.float64),
        qrfz=jnp.asarray(qrfz, dtype=jnp.float64),
        snow_sa=jnp.asarray(arrays["snow_sa"], dtype=jnp.float64),
        snow_sb=jnp.asarray(arrays["snow_sb"], dtype=jnp.float64),
        cse=jnp.asarray(arrays["cse"], dtype=jnp.float64),
        t_Efsw=jnp.asarray(arrays["t_Efsw"], dtype=jnp.float64),
    )


THOMPSON_TABLES = load_thompson_tables()


# ---------------------------------------------------------------------------
# v0.15 cold-collection lookup tables (rain-collecting-snow qr_acr_qs,
# rain-collecting-graupel qr_acr_qg, Bigg rain-freezing freezeH2O).  Extracted
# bit-exact from the pristine WRF .dat tables via
# proofs/v015/cold_collection_oracle/extract_collision_tables.py.  These are the
# Fortran-computed table contents, not a recomputation.
# ---------------------------------------------------------------------------
COLD_TABLE_ASSET = ROOT / "data" / "fixtures" / "thompson-cold-collection-v1.npz"

COLD_TABLE_NAMES = (
    # qr_acr_qs: (ntb_s, ntb_t, ntb_r1, ntb_r)
    "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
    "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
    "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
    # qr_acr_qg (mp8 effective idx_bg1=5 read from WRF's dimNRHG=1 allocation):
    # (ntb_g1, ntb_g, ntb_r1, ntb_r). Cold-branch subset only (tcg_racg is
    # warm-rcg only; Bigg rain-freezing qrfz tables are already in
    # thompson-tables-v1.npz).
    "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
    # freezeH2O cloud-water-freezing planes for non-aerosol mp8:
    # (ntb_c, ntb_tc), with default idx_n(Nt_c=100e6 m^-3) and idx_IN.
    "tpi_qcfz", "tni_qcfz",
)


class ColdCollectionTables(NamedTuple):
    """rain-snow / rain-graupel WRF cold-collection lookup tables (fp64)."""

    tcs_racs1: jnp.ndarray
    tmr_racs1: jnp.ndarray
    tcs_racs2: jnp.ndarray
    tmr_racs2: jnp.ndarray
    tcr_sacr1: jnp.ndarray
    tms_sacr1: jnp.ndarray
    tcr_sacr2: jnp.ndarray
    tms_sacr2: jnp.ndarray
    tnr_racs1: jnp.ndarray
    tnr_racs2: jnp.ndarray
    tnr_sacr1: jnp.ndarray
    tnr_sacr2: jnp.ndarray
    tmr_racg: jnp.ndarray
    tcr_gacr: jnp.ndarray
    tnr_racg: jnp.ndarray
    tnr_gacr: jnp.ndarray
    tpi_qcfz: jnp.ndarray
    tni_qcfz: jnp.ndarray


def cold_tables_available(path: Path = COLD_TABLE_ASSET) -> bool:
    """True when the cold-collection fixture is present (lane is gated off it)."""

    return Path(path).exists()


@lru_cache(maxsize=1)
def load_cold_collection_tables(path: str = str(COLD_TABLE_ASSET)) -> ColdCollectionTables:
    """Loads the cold-collection tables as fp64 JAX arrays (once per interpreter)."""

    with np.load(Path(path), allow_pickle=False) as loaded:
        arrays = {name: jnp.asarray(loaded[name], dtype=jnp.float64) for name in COLD_TABLE_NAMES}
    return ColdCollectionTables(**arrays)


COLD_COLLECTION_TABLES = (
    load_cold_collection_tables() if cold_tables_available() else None
)


TABLE_SOURCE_LINES = {
    "t_Efrw": "module_mp_thompson.F.pre:4921-4977",
    "t_Efsw": "module_mp_thompson.F.pre:4985-5021",
    "tps_iaus": "module_mp_thompson.F.pre:4870-4913",
    "tni_iaus": "module_mp_thompson.F.pre:4870-4913",
    "tpi_ide": "module_mp_thompson.F.pre:4870-4913",
    "tpi_qrfz": "module_mp_thompson.F.pre:4664-4855",
    "tpg_qrfz": "module_mp_thompson.F.pre:4664-4855",
    "tni_qrfz": "module_mp_thompson.F.pre:4664-4855",
    "tnr_qrfz": "module_mp_thompson.F.pre:4664-4855",
    "tpi_qcfz": "module_mp_thompson.F.pre:4664-4855",
    "tni_qcfz": "module_mp_thompson.F.pre:4664-4855",
    "snow_moments": "module_mp_thompson.F.pre:2093-2191",
    "graupel_moments": "module_mp_thompson.F.pre:760-770",
}
