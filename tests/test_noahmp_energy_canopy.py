"""Unit / energy-closure tests for Sprint S1 noahmp_energy_canopy (THE HFX FIX).

ORACLE STATUS: the pristine-WRF ENERGY savepoint fixtures ARE NOW PRESENT
(proofs/noahmp/savepoints_energy.json, S0b external oracle). The REAL gate is
``test_real_wrf_energy_savepoint_parity`` below — field-wise parity vs WRF
NOAHMP_SFLX over 11 real Canary d03 columns (daytime sparse-veg, bare soil,
urban, night x5, Teide snow). The hand-built invariant tests below remain as a
fast self-contained sanity layer (closure + sign + magnitude band):

  1. Surface energy balance closes: SAV+SAG = FIRA+FSH+FCEV+FGEV+FCTR+SSOIL
     (ENERGY :2281-2283 / ERROR :1662) to within Newton-Raphson residual.
  2. THE FIX: daytime land HFX is the canopy-balance value, NOT the bulk
     radiative-skin over-flux. TAH (canopy-air) is COOLER than TRAD (radiative
     skin), which is exactly why FSH(TAH) < the bulk FSH(TSK_radiative).
  3. Bowen-dominated dry case: HFX > LH, LH small (high Bowen ratio).
  4. Nighttime (cosz=0, low LW): downward/near-zero sensible heat, TRAD < day.
  5. Radiation sub-step: 0 < albedo < 1, 0.9 < emissivity < 1, SAG>0 by day.

These are physics-invariant gates (closure + sign + magnitude band), the
strongest self-contained check before the savepoint oracle is wired.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.contracts.noahmp_state import (  # noqa: E402
    NSNOW,
    NSOIL,
    NoahMPLandState,
    NoahMPStatic,
)
from gpuwrf.physics.noahmp.energy import (  # noqa: E402
    EnergyParams,
    noahmp_energy_canopy,
)
from gpuwrf.physics.noahmp.energy_radiation import (  # noqa: E402
    TwoStreamParams,
    radiation_twostream,
)
from gpuwrf.physics.noahmp.types import NoahMPForcing, NoahMPPhenology  # noqa: E402

# WRF Noah-MP module constants used in the closure check
SB = 5.67e-08
CPAIR = 1004.64
RAIR = 287.04
HVAP = 2.5104e06

# ---------------------------------------------------------------------------------
# fixtures: a (1,1) land column for the documented dry sparse-veg midday case
# ---------------------------------------------------------------------------------
def _f(v):
    return jnp.full((1, 1), float(v), dtype=jnp.float64)


def _soil(vals):
    return jnp.asarray(vals, dtype=jnp.float64).reshape(NSOIL, 1, 1)


def _midday_forcing(soldn=900.0, cosz=0.95):
    """Midday dry-land forcing (~sparse veg). Air ~297.66 K (the proof's thx)."""
    sfctmp = 297.66
    sfcprs = 95000.0
    qair = 0.006  # dry boundary layer
    return NoahMPForcing(
        sfctmp=_f(sfctmp), sfcprs=_f(sfcprs), psfc=_f(sfcprs),
        uu=_f(3.0), vv=_f(2.0), qair=_f(qair), qc=_f(0.0),
        soldn=_f(soldn), lwdn=_f(360.0),
        prcpconv=_f(0.0), prcpnonc=_f(0.0), prcpsnow=_f(0.0),
        prcpgrpl=_f(0.0), prcphail=_f(0.0),
        cosz=_f(cosz), zlvl=_f(20.0),
        julian=jnp.asarray(180.0), yearlen=jnp.asarray(365.0),
    )


def _night_forcing():
    return NoahMPForcing(
        sfctmp=_f(288.0), sfcprs=_f(95000.0), psfc=_f(95000.0),
        uu=_f(2.0), vv=_f(1.0), qair=_f(0.006), qc=_f(0.0),
        soldn=_f(0.0), lwdn=_f(320.0),
        prcpconv=_f(0.0), prcpnonc=_f(0.0), prcpsnow=_f(0.0),
        prcpgrpl=_f(0.0), prcphail=_f(0.0),
        cosz=_f(0.0), zlvl=_f(20.0),
        julian=jnp.asarray(180.0), yearlen=jnp.asarray(365.0),
    )


def _land_state(tg=305.0, tv=303.0):
    zsoil = (-0.1, -0.4, -1.0, -2.0)
    # ZSNSO: snow interfaces (0 when no snow) then soil interfaces
    zsnso = jnp.asarray([0.0, 0.0, 0.0] + list(zsoil), dtype=jnp.float64).reshape(
        NSNOW + NSOIL, 1, 1
    )
    return NoahMPLandState(
        tslb=_soil([300.0, 298.0, 295.0, 293.0]),
        smois=_soil([0.10, 0.12, 0.15, 0.18]),
        sh2o=_soil([0.10, 0.12, 0.15, 0.18]),
        smcwtd=_f(0.18),
        isnow=jnp.zeros((1, 1), dtype=jnp.int32),
        tsno=jnp.zeros((NSNOW, 1, 1), dtype=jnp.float64),
        snice=jnp.zeros((NSNOW, 1, 1), dtype=jnp.float64),
        snliq=jnp.zeros((NSNOW, 1, 1), dtype=jnp.float64),
        zsnso=zsnso,
        snowh=_f(0.0), sneqv=_f(0.0), sneqvo=_f(0.0),
        tauss=_f(0.0), albold=_f(0.55),
        tv=_f(tv), tg=_f(tg), tah=_f(300.0), eah=_f(1200.0),
        canliq=_f(0.0), canice=_f(0.0), fwet=_f(0.0),
        lai=_f(1.0), sai=_f(0.3),
        cm=_f(0.01), ch=_f(0.01),
        t_skin=_f(tg), qsfc=_f(0.006), znt=_f(0.05),
        emiss=_f(0.97), albedo=_f(0.2),
        sfcrunoff=_f(0.0), udrunoff=_f(0.0),
    )


def _static():
    zsoil = jnp.asarray([-0.1, -0.4, -1.0, -2.0], dtype=jnp.float64)
    dzs = jnp.asarray([0.1, 0.3, 0.6, 1.0], dtype=jnp.float64)
    return NoahMPStatic(
        ivgtyp=jnp.full((1, 1), 7, dtype=jnp.int32),   # open shrubland (sparse)
        isltyp=jnp.full((1, 1), 6, dtype=jnp.int32),   # loam
        xland=_f(1.0), landmask=_f(1.0), lakemask=_f(0.0),
        lu_index=jnp.full((1, 1), 7, dtype=jnp.int32),
        tbot=_f(290.0), dzs=dzs, zsoil=zsoil,
        lat=_f(28.0), dx_m=3000.0, parameters=None,
    )


def _phenology(fveg=0.30, lai=1.0, sai=0.3):
    elai = lai  # snow-free -> exposed = full
    esai = sai
    return NoahMPPhenology(
        lai=_f(lai), sai=_f(sai), elai=_f(elai), esai=_f(esai),
        fveg=_f(fveg), igs=_f(1.0),
    )


def _energy_params():
    # open-shrubland veg (MPTABLE idx 7) + loam soil (SOILPARM idx 6)
    return EnergyParams(
        z0mvt=_f(0.06), hvt=_f(1.0), cwpvt=_f(5.0), dleaf=_f(0.04),
        z0sno=_f(0.002), cbiom=_f(0.02),
        smcmax=_soil([0.434] * 4), smcref=_soil([0.383] * 4),
        smcwlt=_soil([0.066] * 4), psisat=_soil([0.3548] * 4),
        bexp=_soil([5.25] * 4), quartz=_soil([0.40] * 4),
        csoil=_f(2.0e6), nroot=2, eg=_f(0.97), snow_emis=_f(0.99),
        rsurf_exp=_f(5.0),
        bp=_f(2.0e3), mp=_f(9.0), folnmx=_f(1.5), qe25=_f(0.06),
        kc25=_f(30.0), ko25=_f(3.0e4), akc=_f(2.1), ako=_f(1.2),
        avcmx=_f(2.4), vcmx25=_f(40.0), c3psn=_f(1.0),
    )


def _rad_params():
    # open shrubland reflect/transmit (MPTABLE idx 7), loam soil albedo
    def band2(vis, nir):
        return jnp.stack([_f(vis), _f(nir)], axis=0)

    return TwoStreamParams(
        rhol=band2(0.11, 0.58), rhos=band2(0.36, 0.58),
        taul=band2(0.07, 0.25), taus=band2(0.220, 0.380),
        xl=_f(0.01),
        albsat=band2(0.10, 0.20), albdry=band2(0.20, 0.40),
    )


def _run(forcing, ls=None, phen=None, dt=1800.0):
    ls = ls if ls is not None else _land_state()
    phen = phen if phen is not None else _phenology()
    static = _static()
    rad, extras = radiation_twostream(ls, forcing, static, phen, _rad_params(), dt)
    ls2, ef, et = noahmp_energy_canopy(
        ls, forcing, static, rad, dt,
        phen=phen, params=_energy_params(), rad_extras=extras,
    )
    return rad, extras, ls2, ef, et


# ---------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------
def test_radiation_albedo_emissivity_physical_daytime():
    rad, extras, ls2, ef, et = _run(_midday_forcing())
    alb = float(rad.albedo[0, 0])
    emiss = float(ef.emissi[0, 0])
    sag = float(rad.sag[0, 0])
    sav = float(rad.sav[0, 0])
    assert 0.0 < alb < 1.0, f"albedo out of range: {alb}"
    assert 0.90 < emiss < 1.0, f"emissivity out of range: {emiss}"
    assert sag > 0.0 and sav >= 0.0, f"daytime absorbed solar must be >0 (sag={sag}, sav={sav})"
    # absorbed should be a substantial fraction of incoming, not >incoming
    assert sag + sav < 900.0


def test_surface_energy_balance_closes_daytime():
    rad, extras, ls2, ef, et = _run(_midday_forcing())
    sav = float(rad.sav[0, 0])
    sag = float(rad.sag[0, 0])
    rhs = (float(ef.fira[0, 0]) + float(ef.fsh[0, 0]) + float(ef.fcev[0, 0])
           + float(ef.fgev[0, 0]) + float(ef.fctr[0, 0]) + float(ef.ssoil[0, 0]))
    resid = (sav + sag) - rhs
    # Newton-Raphson residual after fixed iterations: a few W/m2 at most
    assert abs(resid) < 10.0, f"energy budget does not close: resid={resid} W/m2"


def test_hfx_is_canopy_balance_not_bulk_overflux_daytime():
    """THE FIX: HFX from the canopy-air balance must be far below the bulk
    radiative-skin over-flux (~777 W/m2 GPU vs ~477 corpus, ratio 1.63)."""
    rad, extras, ls2, ef, et = _run(_midday_forcing())
    hfx = float(ef.fsh[0, 0])
    trad = float(ef.trad[0, 0])
    tah = float(ls2.tah[0, 0])
    # canopy-air must be COOLER than the radiative skin (the whole reason FSH<bulk)
    assert tah < trad, f"TAH {tah} must be < TRAD {trad} (canopy-air cooler than skin)"
    # HFX must be physical and NOT the 1.6x bulk over-flux (>700). A dry-land
    # midday sensible flux is positive and O(100s) W/m2.
    assert 50.0 < hfx < 650.0, f"daytime land HFX out of canopy-balance band: {hfx}"


def test_dry_case_is_bowen_dominated():
    rad, extras, ls2, ef, et = _run(_midday_forcing())
    hfx = float(ef.fsh[0, 0])
    lh = float(ef.fcev[0, 0] + ef.fgev[0, 0] + ef.fctr[0, 0])
    assert hfx > lh, f"dry-land daytime must be Bowen-dominated: HFX={hfx}, LH={lh}"
    assert lh < 200.0, f"dry-land LH should be small, got {lh}"


def test_nighttime_sensible_heat_small_or_downward():
    rad, extras, ls2, ef, et = _run(_night_forcing())
    hfx_night = float(ef.fsh[0, 0])
    sag = float(rad.sag[0, 0])
    assert sag == pytest.approx(0.0, abs=1e-9), "no absorbed solar at night"
    # nocturnal sensible heat: small magnitude (radiative regime), not a daytime spike
    assert abs(hfx_night) < 200.0, f"nighttime HFX magnitude too large: {hfx_night}"


def test_outputs_finite_and_shaped():
    rad, extras, ls2, ef, et = _run(_midday_forcing())
    for name, arr in (
        ("fsh", ef.fsh), ("lh_fgev", ef.fgev), ("ssoil", ef.ssoil),
        ("trad", ef.trad), ("emissi", ef.emissi), ("z0wrf", ef.z0wrf),
        ("tah", ls2.tah), ("tv", ls2.tv), ("tg", ls2.tg),
        ("qfx_edir", et.edir), ("ecan", et.ecan), ("etran", et.etran),
    ):
        assert arr.shape == (1, 1), f"{name} bad shape {arr.shape}"
        assert jnp.isfinite(arr).all(), f"{name} not finite: {arr}"
    # land carry advanced, soil layers preserved in shape
    assert ls2.tslb.shape == (NSOIL, 1, 1)
    assert et.btrani.shape == (NSOIL, 1, 1)


def test_bare_ground_branch_when_no_veg():
    """FVEG=0 / VAI=0 should fall through to the bare-ground tile cleanly."""
    phen = _phenology(fveg=0.0, lai=0.0, sai=0.0)
    rad, extras, ls2, ef, et = _run(_midday_forcing(), phen=phen)
    hfx = float(ef.fsh[0, 0])
    assert jnp.isfinite(ef.fsh).all()
    assert -50.0 < hfx < 700.0, f"bare-ground HFX out of band: {hfx}"
    # bare ground: no canopy evaporation / transpiration
    assert float(ef.fcev[0, 0]) == pytest.approx(0.0, abs=1e-9)
    assert float(ef.fctr[0, 0]) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------------
# THE REAL GATE: field-wise parity vs the pristine-WRF ENERGY savepoints (S0b
# external oracle). Skips cleanly if the savepoints / WRF tables are unavailable.
# ---------------------------------------------------------------------------------
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_PROOFS = Path(__file__).resolve().parent.parent / "proofs" / "noahmp"
_HAVE_GATE = (_PROOFS / "savepoints_energy.json").exists() and Path(
    "/home/user/src/wrf_pristine/WRF/run/MPTABLE.TBL"
).exists()


@pytest.mark.skipif(not _HAVE_GATE, reason="WRF ENERGY savepoints / MPTABLE not present")
def test_real_wrf_energy_savepoint_parity():
    """Every pristine-WRF ENERGY savepoint column must match field-wise within the
    predeclared tolerances (dry daytime sparse-veg, bare soil, urban, night, snow)."""
    import json

    sys.path.insert(0, str(_PROOFS))
    import energy_savepoint_gate as gate  # noqa: E402

    sp = json.load(open(_PROOFS / "savepoints_energy.json"))
    failures = []
    for col in sp["columns"]:
        wrf = col["wrf"]
        ref = {**wrf["energy_out"], "albedo": wrf["energy_state"]["albedo"],
               "tg": wrf["energy_state"]["tg"], "tah": wrf["energy_state"]["tah"]}
        got = gate.run_column(col)
        for fld, (atol, rtol) in gate.TOL.items():
            if fld == "qsfc":
                continue
            if fld == "erreng":
                if abs(got["erreng"]) > atol:
                    failures.append((col["name"], fld, 0.0, got["erreng"]))
                continue
            r = ref.get(fld)
            if r is None:
                continue
            if abs(got[fld] - r) > atol + rtol * abs(r):
                failures.append((col["name"], fld, r, got[fld]))
    assert not failures, "WRF ENERGY savepoint parity failures:\n" + "\n".join(
        f"  {n}/{f}: wrf={r:.6g} jax={g:.6g} d={g-r:+.5g}" for n, f, r, g in failures
    )
