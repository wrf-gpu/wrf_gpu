"""Device-resident lookup tables for aerosol-aware Thompson (mp_physics=28).

Loads two assets:
  * ``thompson-aero-tables-v1.npz`` (built by ``scripts/build_thompson_aero_tables.py``):
    CCN activation table (verbatim WRF ``CCN_ACTIVATE.BIN``), droplet-evap
    number table ``tnc_wev``, heterogeneous cloud-freezing tables
    ``tpi/tni_qcfz``, and the variable-``nu_c`` cloud gamma families.
  * the existing mp=8 asset ``thompson-tables-v1.npz`` for the FULL 4-D
    rain-freezing tables (``t*_qrfz`` with the N_IN axis the mp=8 kernel
    slices away) and the graupel gamma table entries the aerosol-scavenging
    terms need.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import NamedTuple

from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.thompson_tables import TABLE_ASSET as MP8_ASSET

configure_jax_x64()

ROOT = Path(__file__).resolve().parents[3]
AERO_ASSET = ROOT / "data" / "fixtures" / "thompson-aero-tables-v1.npz"

# Table dimensions (module_mp_thompson.F header).
N_ARC, N_ARW, N_ART = 7, 9, 7
NBC = 100
N_C_TABLE = 37
N_TC_TABLE = 45
N_IN_TABLE = 55

# Fixed mean-aerosol-radius / hygroscopicity indices (module_mp_thompson.F:5229-5230).
_L_RADIUS = 3
_M_KAPPA = 2


class ThompsonAeroTableBundle(NamedTuple):
    """Runtime aerosol-aware table bundle (all jnp arrays, fp64)."""

    # CCN activation: slice (Na, Ww, Tk) at l=3, m=2 plus log-axis vectors.
    ccn_act: jnp.ndarray          # (7, 9, 7)
    ta_na: jnp.ndarray            # (7,) per-cc axis
    ta_ww: jnp.ndarray            # (9,) m/s axis
    # Droplet evaporation number table (D-bin, r_c, t_Nc), flattened.
    tnc_wev: jnp.ndarray          # (100*37*100,)
    # Heterogeneous cloud freezing (r_c, t_Nc-bin, -T, N_IN) mass/number, flat.
    qcfz: jnp.ndarray             # (37*100*45*55, 2)
    # Rain freezing with the full N_IN axis (r_r, N0r, -T, N_IN) x 4, flat.
    qrfz4: jnp.ndarray            # (37*37*45*55, 4)
    # Variable-nu_c cloud gamma families (index 0 == nu_c=1).
    cce: jnp.ndarray              # (5, 15)
    ccg: jnp.ndarray              # (5, 15)
    ocg1: jnp.ndarray             # (15,)
    ocg2: jnp.ndarray             # (15,)
    # Cloud bin axes for the evap/freeze table indexes.
    t_nc1: jnp.ndarray            # scalar t_Nc(1)
    nic1: jnp.ndarray             # scalar log(t_Nc(nbc)/t_Nc(1))


def asset_sha256(path: Path = AERO_ASSET) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _load(path: str) -> dict[str, np.ndarray]:
    asset = Path(path)
    if not asset.exists():
        raise FileNotFoundError(
            f"missing Thompson aero table asset: {asset} "
            "(run scripts/build_thompson_aero_tables.py)"
        )
    with np.load(asset, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def load_thompson_aero_tables(path: Path = AERO_ASSET) -> ThompsonAeroTableBundle:
    arrays = _load(str(path))
    with np.load(MP8_ASSET, allow_pickle=False) as mp8:
        qrfz4 = np.stack(
            (
                np.asarray(mp8["tpi_qrfz"], dtype=np.float64),
                np.asarray(mp8["tpg_qrfz"], dtype=np.float64),
                np.asarray(mp8["tni_qrfz"], dtype=np.float64),
                np.asarray(mp8["tnr_qrfz"], dtype=np.float64),
            ),
            axis=-1,
        ).reshape(-1, 4)

    ccn_full = np.asarray(arrays["tnccn_act"], dtype=np.float64)
    ccn_act = ccn_full[:, :, :, _L_RADIUS - 1, _M_KAPPA - 1]
    qcfz = np.stack(
        (
            np.asarray(arrays["tpi_qcfz"], dtype=np.float64),
            np.asarray(arrays["tni_qcfz"], dtype=np.float64),
        ),
        axis=-1,
    ).reshape(-1, 2)
    return ThompsonAeroTableBundle(
        ccn_act=jnp.asarray(ccn_act),
        ta_na=jnp.asarray(np.asarray(arrays["ta_Na"], dtype=np.float64)),
        ta_ww=jnp.asarray(np.asarray(arrays["ta_Ww"], dtype=np.float64)),
        tnc_wev=jnp.asarray(np.asarray(arrays["tnc_wev"], dtype=np.float64).reshape(-1)),
        qcfz=jnp.asarray(qcfz),
        qrfz4=jnp.asarray(qrfz4),
        cce=jnp.asarray(np.asarray(arrays["cce"], dtype=np.float64)),
        ccg=jnp.asarray(np.asarray(arrays["ccg"], dtype=np.float64)),
        ocg1=jnp.asarray(np.asarray(arrays["ocg1"], dtype=np.float64)),
        ocg2=jnp.asarray(np.asarray(arrays["ocg2"], dtype=np.float64)),
        t_nc1=jnp.asarray(float(np.asarray(arrays["t_Nc"])[0])),
        nic1=jnp.asarray(float(np.asarray(arrays["nic1"]))),
    )


THOMPSON_AERO_TABLES = load_thompson_aero_tables()
