"""CPU checks for WRF Noah-MP vegetation-root table wiring."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from gpuwrf.contracts.noahmp_state import NoahMPStatic
from gpuwrf.physics.noahmp.noahmp_driver import build_energy_params
from gpuwrf.physics.noahmp.tables import MBAND, MSC, NMONTH, NoahMPParameters


def _veg_col(values: list[float]) -> jnp.ndarray:
    out = np.zeros(len(values) + 1, dtype=np.float64)
    out[1:] = np.asarray(values, dtype=np.float64)
    return jnp.asarray(out)


def _veg_band(nveg: int, value: float) -> jnp.ndarray:
    out = np.zeros((nveg + 1, MBAND), dtype=np.float64)
    out[1:, :] = value
    return jnp.asarray(out)


def _veg_monthly(nveg: int, value: float) -> jnp.ndarray:
    out = np.zeros((nveg + 1, NMONTH), dtype=np.float64)
    out[1:, :] = value
    return jnp.asarray(out)


def _soil_col(value: float, nsoil_cat: int = 2) -> jnp.ndarray:
    out = np.zeros(nsoil_cat + 1, dtype=np.float64)
    out[1:] = value
    return jnp.asarray(out)


def _params_with_nroot(nroot_by_veg: list[float]) -> NoahMPParameters:
    nveg = len(nroot_by_veg)
    veg = lambda value: _veg_col([value] * nveg)  # noqa: E731

    return NoahMPParameters(
        rhol=_veg_band(nveg, 0.10),
        rhos=_veg_band(nveg, 0.08),
        taul=_veg_band(nveg, 0.06),
        taus=_veg_band(nveg, 0.04),
        xl=veg(1.0),
        z0mvt=veg(0.12),
        hvt=veg(1.5),
        hvb=veg(0.1),
        dleaf=veg(0.04),
        rc=veg(0.0),
        den=veg(0.0),
        cwpvt=veg(18.0),
        saim=_veg_monthly(nveg, 0.2),
        laim=_veg_monthly(nveg, 1.0),
        sla=veg(0.0),
        ch2op=veg(0.1),
        nroot=_veg_col(nroot_by_veg),
        mfsno=veg(2.5),
        scffac=veg(1.0),
        rsmin=veg(100.0),
        rsmax=veg(5000.0),
        rgl=veg(30.0),
        hs=veg(50.0),
        topt=veg(298.0),
        bp=veg(0.01),
        mp=veg(9.0),
        c3psn=veg(1.0),
        kc25=veg(40.0),
        akc=veg(2.1),
        ko25=veg(30000.0),
        ako=veg(1.2),
        vcmx25=veg(50.0),
        avcmx=veg(2.4),
        qe25=veg(0.06),
        aqe=veg(1.8),
        folnmx=veg(3.0),
        bexp=_soil_col(4.5),
        smcmax=_soil_col(0.45),
        smcref=_soil_col(0.30),
        smcwlt=_soil_col(0.10),
        smcdry=_soil_col(0.05),
        dksat=_soil_col(1.0e-6),
        dwsat=_soil_col(1.0e-5),
        psisat=_soil_col(0.2),
        quartz=_soil_col(0.4),
        albsat=jnp.ones((MSC + 1, MBAND), dtype=jnp.float64) * 0.15,
        albdry=jnp.ones((MSC + 1, MBAND), dtype=jnp.float64) * 0.30,
        csoil=jnp.asarray(2.0e6),
        zbot=jnp.asarray(-8.0),
        czil=jnp.asarray(0.1),
        refdk=jnp.asarray(2.0e-6),
        refkdt=jnp.asarray(3.0),
        frzk=jnp.asarray(0.15),
        slope=jnp.asarray([0.0, 0.1], dtype=jnp.float64),
        eg=jnp.asarray([0.97, 0.98], dtype=jnp.float64),
        omegas=jnp.asarray([0.8, 0.4], dtype=jnp.float64),
        betads=jnp.asarray(0.5),
        betais=jnp.asarray(0.5),
        swemx=jnp.asarray(1.0),
        z0sno=jnp.asarray(0.002),
        ssi=jnp.asarray(0.03),
        snow_ret_fac=jnp.asarray(0.1),
        snow_emis=jnp.asarray(0.99),
        iswater=17,
        isbarren=16,
        isice=15,
        iscrop=12,
        isurban=13,
        tau0=jnp.asarray(1.0e6),
        grain_growth=jnp.asarray(0.0),
        extra_growth=jnp.asarray(0.0),
        dirt_soot=jnp.asarray(0.0),
    )


def test_build_energy_params_uses_per_cell_wrf_nroot_map() -> None:
    """NROOT must be gathered by IVGTYP, not collapsed to grid max per cell."""

    ivgtyp = jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32)
    static = NoahMPStatic(
        ivgtyp=ivgtyp,
        isltyp=jnp.ones_like(ivgtyp),
        xland=jnp.ones_like(ivgtyp, dtype=jnp.float64),
        landmask=jnp.ones_like(ivgtyp, dtype=jnp.float64),
        lakemask=jnp.zeros_like(ivgtyp, dtype=jnp.float64),
        lu_index=ivgtyp,
        tbot=jnp.ones_like(ivgtyp, dtype=jnp.float64) * 290.0,
        dzs=jnp.asarray([0.1, 0.3, 0.6, 1.0], dtype=jnp.float64),
        zsoil=jnp.asarray([-0.1, -0.4, -1.0, -2.0], dtype=jnp.float64),
        lat=jnp.ones_like(ivgtyp, dtype=jnp.float64) * 28.0,
        dx_m=3000.0,
        parameters=_params_with_nroot([1.0, 4.0, 3.0, 2.0]),
    )

    energy, _rad = build_energy_params(static, ivgtyp.shape)

    assert energy.nroot == 4
    assert np.array_equal(
        np.asarray(energy.nroot_cell),
        np.asarray([[1, 4], [3, 2]], dtype=np.int32),
    )
    assert len(np.unique(np.asarray(energy.nroot_cell))) > 1
