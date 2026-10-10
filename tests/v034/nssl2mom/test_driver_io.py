"""NSSL mp=18 driver plumbing vs the pristine-WRF oracle: pack/t7/denscale (S0), precip binding
(S2 xfall -> RAINNCV...), de-scale/unpack (S5 -> *_OUT).  fp64 port vs fp64 oracle (tight) and
fp32 port vs fp32 oracle."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_io as oio  # noqa: E402

from gpuwrf.physics.nssl2mom import driver  # noqa: E402
from gpuwrf.physics.nssl2mom.constants import get_constants  # noqa: E402
from gpuwrf.physics.nssl2mom.indices import FP32, FP64, LR, LS, LH, LHL, NA  # noqa: E402

PREC = {"fp32": FP32, "fp64": FP64}
TOL = {"fp64": 1e-12, "fp32": 2e-6}
IN_NAMES = {"th": "TH", "qv": "QV", "qc": "QC", "qr": "QR", "qi": "QI", "qs": "QS", "qg": "QH",
            "qh": "QHL", "qndrop": "CCW", "qnr": "CRW", "qni": "CCI", "qns": "CSW", "qng": "CHW",
            "qnh": "CHL", "qnn": "CN", "qvolg": "VHW", "qvolh": "VHL"}


def fields_in(sp):
    col = sp["columns"]
    f = {k: np.array(col[v + "_IN"]) for k, v in IN_NAMES.items()}
    f.update(pii=np.array(col["PII"]), p=np.array(col["P"]), w=np.array(col["W"]),
             dz=np.array(col["DZ"]), rho=np.array(col["DN"]))
    return f


def relerr(a, b, atol=1e-30):
    """max relative error; differences below atol are ignored (XLA:CPU flushes fp32 subnormals,
    E117 — WRF keeps them; 1e-30 is far below every physical threshold of the scheme)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    den = np.maximum(np.abs(b), 1e-300)
    return float(np.max(np.where(np.abs(a - b) <= atol, 0.0, np.abs(a - b) / den)))


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
@pytest.mark.parametrize("case", oio.CASES)
def test_pack_t7_denscale(mode, case):
    sp = oio.load_case(mode, case)
    C = get_constants(mode)
    an, aux = driver.pack(fields_in(sp), sp["scalars"]["ITIMESTEP"], C, PREC[mode])
    ref = oio.stage_an(sp, "S0")
    for il in range(1, NA + 1):
        assert relerr(an[il], ref[il]) <= TOL[mode], (il, relerr(an[il], ref[il]))
    for name, key in (("t0", "t0"), ("t7", "t7"), ("t00", "t00"), ("t77", "t77"), ("pn", "pn"), ("dn", "dn1")):
        e = relerr(aux[name], oio.stage_col(sp, "S0", key))
        # t7 = exp(12.96*(ssival-1)-0.639): fp32 libm (XLA vs glibc expf) ulps amplified ~10x
        tol = 2e-5 if (mode == "fp32" and name == "t7") else TOL[mode]
        assert e <= tol, (name, e)


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
@pytest.mark.parametrize("case", oio.CASES)
def test_precip_binding(mode, case):
    sp = oio.load_case(mode, case)
    C = get_constants(mode)
    xf = sp["stages"]["S2"]["xfall"]
    xfall = {il: np.asarray(xf[il - 1], PREC[mode].nR) for il in (LR, LS, LH, LHL)}
    dn1 = np.asarray(oio.stage_col(sp, "S0", "dn1")[0], PREC[mode].nR)
    p = driver.bind_precip(xfall, dn1, sp["scalars"]["DT"], C, PREC[mode])
    for k in ("rainncv", "snowncv", "grplncv", "hailncv", "sr"):
        assert relerr(p[k], sp["scalars"][k.upper()]) <= TOL[mode], (k, float(p[k]), sp["scalars"][k.upper()])


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
@pytest.mark.parametrize("case", oio.CASES)
def test_unpack(mode, case):
    sp = oio.load_case(mode, case)
    C = get_constants(mode)
    an = np.asarray(oio.stage_an(sp, "S5"), PREC[mode].nR)
    dn = np.asarray(oio.stage_col(sp, "S0", "dn1"), PREC[mode].nR)
    out = driver.unpack(an, dn, C, PREC[mode])
    for k, v in IN_NAMES.items():
        e = relerr(out[k], sp["columns"][v + "_OUT"])
        assert e <= TOL[mode], (k, e)


def test_mutant_t7_sensitive():
    """Deletion check: the nucleation-rate exponent constant matters (E39)."""
    sp = oio.load_case("fp64", 3)
    C = get_constants("fp64")
    _an, aux = driver.pack(fields_in(sp), 1, C, FP64)
    ref = oio.stage_col(sp, "S0", "t7")
    assert np.max(ref) > 0
    assert relerr(aux["t7"] * 1.001, ref) > 1e-4
