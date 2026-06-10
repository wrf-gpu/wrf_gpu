"""v0.14 regression: Noah-MP WATER soil/veg category gather must be 1-BASED.

Root cause of the v0.14 d01 LU16 preflight blocker (51 nonfinite land cells,
all ISLTYP=1 sand): ``water_hydro._category_index`` gathered the frozen
1-based parameter tables (axis-0 length ncat+1 with an ALL-ZERO dummy row 0,
``tables._parse_soilparm``) with ``category - 1``. Sand (ISLTYP=1) read the
zero row -> SMCMAX=0 -> smc/smcmax = inf -> NaN soil moisture on the first
step; every other soil category silently ran WATER with the PREVIOUS
category's hydraulic parameters. WRF's TRANSFER_MP_PARAMETERS indexes
``BEXP_TABLE(SOILTYPE)`` directly (1-based), as do ``noahmp_driver._gather_vec``
and the phenology gathers.

These tests pin (a) the gather row identity against the driver's gather,
(b) finiteness of a one-step WATER advance on the EXACT failing configuration
(warm dry sand below SMCDRY, barren veg 16, no precip, small ground evap).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic
from gpuwrf.physics.noahmp.noahmp_driver import _gather_vec
from gpuwrf.physics.noahmp.types import NoahMPEtFluxes, NoahMPForcing
from gpuwrf.physics.noahmp.water_hydro import _soil_param, noahmp_water_hydro


class _Params:
    """Synthetic 1-based soil tables (row 0 = WRF parser's all-zero dummy row).

    Rows 1..5 carry distinctive values so a +-1 index shift is detectable.
    Row 1 mirrors SOILPARM.TBL STAS sand. Table length is deliberately != NSOIL
    (real STAS tables are length 20) so ``_soil_param`` takes the category-gather
    branch, exactly as production does.
    """

    bexp = jnp.array([0.0, 2.79, 4.26, 4.74, 5.33, 3.86])
    smcmax = jnp.array([0.0, 0.339, 0.421, 0.434, 0.476, 0.484])
    smcref = jnp.array([0.0, 0.236, 0.383, 0.383, 0.360, 0.383])
    smcwlt = jnp.array([0.0, 0.010, 0.028, 0.047, 0.084, 0.061])
    smcdry = jnp.array([0.0, 0.010, 0.028, 0.047, 0.084, 0.061])
    dksat = jnp.array([0.0, 4.66e-05, 1.41e-05, 5.23e-06, 2.81e-06, 2.18e-06])
    dwsat = jnp.array([0.0, 2.65e-05, 5.14e-06, 8.05e-06, 2.39e-05, 1.43e-05])
    psisat = jnp.array([0.0, 0.069, 0.036, 0.141, 0.759, 0.955])
    quartz = jnp.array([0.0, 0.92, 0.82, 0.60, 0.25, 0.40])
    ch2op = jnp.array([0.0] + [0.1] * 20)   # veg table: row 0 dummy, rows 1.. = 0.1
    slope = jnp.array([0.0, 0.1, 0.6, 1.0])  # GENPARM 1-based slope table


def _grid(ny=2, nx=3):
    return ny, nx


def _static(isltyp_val: int, ivgtyp_val: int = 16, ny=2, nx=3) -> NoahMPStatic:
    dzs = jnp.array([0.10, 0.30, 0.60, 1.00])
    zsoil = -jnp.cumsum(dzs)
    return NoahMPStatic(
        ivgtyp=jnp.full((ny, nx), ivgtyp_val, dtype=jnp.int32),
        isltyp=jnp.full((ny, nx), isltyp_val, dtype=jnp.int32),
        xland=jnp.ones((ny, nx)),
        landmask=jnp.ones((ny, nx)),
        lakemask=jnp.zeros((ny, nx)),
        lu_index=jnp.full((ny, nx), ivgtyp_val, dtype=jnp.int32),
        tbot=jnp.full((ny, nx), 293.0),
        dzs=dzs, zsoil=zsoil,
        lat=jnp.full((ny, nx), 22.0), dx_m=9000.0,
        parameters=_Params(),
        shdmax=jnp.zeros((ny, nx)),     # barren
        shdfac=jnp.zeros((ny, nx)),
    )


def _land_state(smois_val: float, ny=2, nx=3) -> NoahMPLandState:
    soil = lambda v: jnp.full((NSOIL, ny, nx), v)  # noqa: E731
    surf = lambda v: jnp.full((ny, nx), v)         # noqa: E731
    dzs = np.array([0.10, 0.30, 0.60, 1.00])
    zsnso = jnp.concatenate(
        [jnp.zeros((NSNOW, ny, nx)),
         jnp.broadcast_to(jnp.asarray(-np.cumsum(dzs))[:, None, None], (NSOIL, ny, nx))],
        axis=0)
    return NoahMPLandState(
        tslb=soil(300.0), smois=soil(smois_val), sh2o=soil(smois_val),
        smcwtd=surf(smois_val),
        isnow=jnp.zeros((ny, nx), dtype=jnp.int32),
        tsno=jnp.zeros((NSNOW, ny, nx)),
        snice=jnp.zeros((NSNOW, ny, nx)), snliq=jnp.zeros((NSNOW, ny, nx)),
        zsnso=zsnso, snowh=surf(0.0), sneqv=surf(0.0), sneqvo=surf(0.0),
        tauss=surf(0.0), albold=surf(0.65),
        tv=surf(300.0), tg=surf(300.0), tah=surf(300.0), eah=surf(2000.0),
        canliq=surf(0.0), canice=surf(0.0), fwet=surf(0.0),
        lai=surf(0.0), sai=surf(0.0), cm=surf(1e-4), ch=surf(1e-4),
        t_skin=surf(300.0), qsfc=surf(0.0), znt=surf(0.05), emiss=surf(0.97),
        albedo=surf(0.3), sfcrunoff=surf(0.0), udrunoff=surf(0.0),
    )


def _forcing(ny=2, nx=3) -> NoahMPForcing:
    z = jnp.zeros((ny, nx))
    return NoahMPForcing(
        sfctmp=jnp.full((ny, nx), 299.0), sfcprs=jnp.full((ny, nx), 1.0e5),
        psfc=jnp.full((ny, nx), 1.0e5), uu=z + 3.0, vv=z + 1.0,
        qair=jnp.full((ny, nx), 0.008), qc=z,
        soldn=z + 50.0, lwdn=z + 380.0,
        prcpconv=z, prcpnonc=z, prcpsnow=z, prcpgrpl=z, prcphail=z,
        cosz=z + 0.1, zlvl=jnp.full((ny, nx), 28.0),
        julian=jnp.asarray(120.75), yearlen=jnp.asarray(365.0),
    )


def _et(edir: float, ny=2, nx=3) -> NoahMPEtFluxes:
    z = jnp.zeros((ny, nx))
    btrani = jnp.zeros((NSOIL, ny, nx)).at[0].set(1.0)
    return NoahMPEtFluxes(
        ecan=z, etran=z, edir=z + edir, qseva=z + edir, btrani=btrani,
        qsnow=z, qmelt=z, imelt=jnp.zeros((NSNOW + NSOIL, ny, nx)),
    )


def test_soil_param_gather_matches_driver_1based_gather():
    """WATER's category gather must read the SAME table row as the driver's
    1-based gather (TRANSFER_MP_PARAMETERS: BEXP_TABLE(SOILTYPE)) for every
    category, including sand=1 (NOT the all-zero dummy row 0)."""
    template = jnp.zeros((NSOIL, 2, 3))
    for cat in (1, 2, 3, 4, 5):
        st = _static(cat)
        for name in ("bexp", "smcmax", "smcwlt", "dksat", "dwsat"):
            got = np.asarray(_soil_param(_Params(), name, st, template, 0.0))
            want = np.asarray(_gather_vec(getattr(_Params(), name), st.isltyp))
            assert np.allclose(got, np.broadcast_to(want, got.shape)), (
                f"{name} cat={cat}: water gathered {got[0,0,0]}, driver {want[0,0]}")
    # sand explicitly: must be the real sand row, not the zero dummy row
    st1 = _static(1)
    smcmax = np.asarray(_soil_param(_Params(), "smcmax", st1, template, 0.0))
    assert np.allclose(smcmax, 0.339), smcmax


def test_water_hydro_dry_sand_below_smcdry_stays_finite():
    """The exact v0.14 d01 failing configuration: ISLTYP=1 sand, IVGTYP=16
    barren, warm, SMOIS=0.0098 (below sand SMCDRY=0.010), no precip, small
    positive ground evaporation. One WATER step must stay finite (was NaN in
    all 4 layers via the zero dummy row before the 1-based gather fix)."""
    smois0 = 0.0098
    land = _land_state(smois0)
    out = noahmp_water_hydro(land, _forcing(), _static(1), _et(1.2e-07), 18.0)
    smc = np.asarray(out.smois)
    sh2o = np.asarray(out.sh2o)
    assert np.all(np.isfinite(smc)), smc
    assert np.all(np.isfinite(sh2o)), sh2o
    assert np.all(np.isfinite(np.asarray(out.sfcrunoff)))
    assert np.all(np.isfinite(np.asarray(out.udrunoff)))
    # physical sanity: bounded by porosity, non-negative, and the column only
    # LOSES water (evap sink, no precip): drawdown <= sink + drainage.
    assert np.all(smc >= 0.0) and np.all(smc <= 0.339 + 1e-12)
    assert np.all(smc[0] <= smois0 + 1e-12)


def test_water_hydro_all_soil_categories_finite():
    """Sweep every STAS soil category id over the synthetic 3-row table: any
    future index shift puts SOME category on the zero row and trips this."""
    for cat in (1, 2, 3, 4, 5):
        out = noahmp_water_hydro(
            _land_state(0.02), _forcing(), _static(cat), _et(1.0e-07), 18.0)
        assert np.all(np.isfinite(np.asarray(out.smois))), f"cat={cat}"
        assert np.all(np.isfinite(np.asarray(out.sh2o))), f"cat={cat}"
