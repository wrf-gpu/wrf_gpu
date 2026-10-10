"""Faithful JAX port of the WRF RUC land-surface model (``sf_surface_physics=3``).

Source: unmodified WRF v4 ``phys/module_sf_ruclsm.F`` (``LSMRUC`` driver and its
internal ``SOILVEGIN``, ``SFCTMP``, ``SOIL``, ``SNOWSOIL``, ``SOILTEMP``,
``SNOWTEMP``, ``SOILMOIST``, ``SOILPROP``, ``TRANSF``, ``VILKA``, ``QSN``) plus
``RUCLSMINIT`` and the ``RUCLSM_SOILVEGPARM`` table reader.

Design (wave O1, lane o1-ruc):

* column-vectorised: every per-point scalar of the Fortran is a ``(ncol,)`` array,
  every soil profile ``(ncol, nzs)``; all ``if`` branches are ``jnp.where`` so the
  step is jit-traceable (no host round trips, no data-dependent Python control);
* dtype-generic: the step runs in the dtype of the inputs.  WRF builds RUC in
  REAL (fp32); the fp64 path exists for parity against the fp64 oracle build;
* literal WRF operation order (left-to-right Fortran evaluation, constant
  sub-expressions evaluated in the working precision, E133);
* WRF quirks are reproduced on purpose and named where they occur (e.g. SFCEVP
  is accumulated twice per step, ``lsmruc`` :1095 and :1116).

Supported: land points (dominant or mosaic land-use/soil), snow (thin / one- /
two-layer, melt iteration, bottom melt, compaction, canopy interception),
frozen soil, water points.  Fail-closed (flagged in ``unsupported``): sea ice
(``xice >= xice_threshold`` on a land point, WRF ``SICE``/``SNOWSEAICE``),
lake model points, ``spp_lsm=1``.

WRF undefined behaviour that this port must resolve deterministically
(documented, oracle-checked):

* ``nroot+1 > nzs``: evergreen/mixed forests (``IFORTBL <= 2``) on the 6-level
  RUC grid select ``nroot=6`` and WRF then reads ``zshalf(7)`` out of bounds.
  ``RucConfig.oob_root_policy`` decides (default ``"fail"`` -> flagged
  unsupported; the 9-level HRRR grid is unaffected).
* thin snow (``snhei < snth``) leaves ``ilnb`` uninitialised in ``SNOWTEMP``;
  the end-of-step ``tsnav`` then uses the one-layer form (``ilnb <= 1``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

NLUS = 50
NSLTYPE = 30
ISNCOVR_OPT = 2
ISNCOND_OPT = 2
VILKA_MAX_ITERS = 40

# module_model_constants (REAL parameters)
_RHOWATER = 1000.0
_PICONST = 3.1415926535897932384626433
_P1000MB = 100000.0

#: carry (state) fields of one RUC column, scalars
STATE_SCALARS = (
    "soilt", "soilt1", "tsnav", "snow", "snowh", "snowc", "canwat", "alb", "emiss",
    "znt", "z0", "lai", "mavail", "vegfra", "snoalb", "qvg", "qsg", "qcg", "dew",
    "qsfc", "chklowq", "hfx", "qfx", "lh", "grdflx", "sfcrunoff", "udrunoff",
    "acrunoff", "sfcexc", "sfcevp", "smavail", "smmax", "snowfallac", "acsnow",
    "snom", "rhosnf", "precipfr",
)
#: carry (state) soil-profile fields (ncol, nzs)
STATE_PROFILES = ("tso", "soilmois", "sh2o", "smfr3d", "keepfr3dflag")
#: per-step forcing (level-1 atmosphere, radiation, exchange coefficients, precip)
FORCING_FIELDS = (
    "t3d", "qv3d", "qc3d", "p8w", "rho3d", "z3d", "glw", "gsw", "chs", "flqc",
    "flhc", "rainbl", "rainncv", "snowncv", "graupelncv", "frzfrac",
)
#: static per-column fields
STATIC_FIELDS = ("ivgtyp", "isltyp", "xland", "xice", "tbot", "shdmin", "shdmax", "albbck")


# ---------------------------------------------------------------------------
# Tables (RUCLSM_SOILVEGPARM)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"'[^']*'|[^,\s]+")


def _tokens(line: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(line) if not t.startswith("'")]


@dataclass(frozen=True)
class RucTables:
    """WRF RUC parameter tables, parsed exactly like ``RUCLSM_SOILVEGPARM``.

    Arrays are 1-based in WRF; here index ``k-1`` holds category ``k``.  Entries
    beyond the table size stay zero (Fortran module arrays are zero-initialised).
    Values are kept as the decimal text parsed to float64 and cast to the
    working dtype at use (Fortran list-directed read of the same text).
    """

    mminluruc: str
    lucats: int
    slcats: int
    albtbl: np.ndarray
    z0tbl: np.ndarray
    lemitbl: np.ndarray
    pctbl: np.ndarray
    shdtbl: np.ndarray
    ifortbl: np.ndarray
    rstbl: np.ndarray
    rgltbl: np.ndarray
    hstbl: np.ndarray
    snuptbl: np.ndarray
    laitbl: np.ndarray
    maxalb: np.ndarray
    bb: np.ndarray
    drysmc: np.ndarray
    hc: np.ndarray
    maxsmc: np.ndarray
    refsmc: np.ndarray
    satpsi: np.ndarray
    satdk: np.ndarray
    satdw: np.ndarray
    wltsmc: np.ndarray
    qtz: np.ndarray
    topt_data: float
    cmcmax_data: float
    cfactr_data: float
    rsmax_data: float
    bare: int
    natural: int
    crop: int
    urban: int


def mminlu_to_ruc(mminlu: str) -> str:
    """``RUCLSMINIT`` land-use dataset mapping."""

    m = str(mminlu).strip()
    if m == "USGS":
        return "USGS-RUC"
    if m in ("MODIS", "MODIFIED_IGBP_MODIS_NOAH"):
        return "MODI-RUC"
    raise ValueError(
        f"RUC LSM supports MMINLU 'USGS' or 'MODIFIED_IGBP_MODIS_NOAH'/'MODIS' only "
        f"(WRF RUCLSMINIT maps nothing else; got {mminlu!r})"
    )


def load_ruc_tables(mminluruc: str = "USGS-RUC", run_dir: str | Path | None = None,
                    mminsl: str = "STAS-RUC") -> RucTables:
    """Parse ``VEGPARM.TBL`` / ``SOILPARM.TBL`` like WRF ``RUCLSM_SOILVEGPARM``."""

    if run_dir is None:
        from gpuwrf.config.paths import wrf_run_dir

        run_dir = wrf_run_dir()
    run_dir = Path(run_dir)
    veg = (run_dir / "VEGPARM.TBL").read_text().splitlines()
    try:
        start = next(i for i, ln in enumerate(veg) if ln[:8].strip() == mminluruc and ln.strip() == mminluruc)
    except StopIteration as exc:  # WRF: wrf_error_fatal
        raise ValueError(f"Land Use Dataset '{mminluruc}' not found in VEGPARM.TBL") from exc
    lucats = int(_tokens(veg[start + 1])[0])
    arr = {k: np.zeros(NLUS) for k in (
        "albtbl", "z0tbl", "lemitbl", "pctbl", "shdtbl", "rstbl", "rgltbl", "hstbl",
        "snuptbl", "laitbl", "maxalb")}
    ifortbl = np.zeros(NLUS, dtype=np.int32)
    for lc in range(lucats):
        t = _tokens(veg[start + 2 + lc])
        (_, albtbl, z0tbl, lemitbl, pctbl, shdtbl, ifor, rstbl, rgltbl, hstbl, snuptbl,
         laitbl, maxalb) = t[:13]
        for name, val in (("albtbl", albtbl), ("z0tbl", z0tbl), ("lemitbl", lemitbl),
                          ("pctbl", pctbl), ("shdtbl", shdtbl), ("rstbl", rstbl),
                          ("rgltbl", rgltbl), ("hstbl", hstbl), ("snuptbl", snuptbl),
                          ("laitbl", laitbl), ("maxalb", maxalb)):
            arr[name][lc] = float(val)
        ifortbl[lc] = int(ifor)
    tail = veg[start + 2 + lucats:]

    def _after(label: str) -> str:
        idx = next(i for i, ln in enumerate(tail) if ln.strip() == label)
        return _tokens(tail[idx + 1])[0]

    soil = (run_dir / "SOILPARM.TBL").read_text().splitlines()
    try:
        sstart = next(i for i, ln in enumerate(soil) if ln.strip() == mminsl)
    except StopIteration as exc:
        raise ValueError("INCONSISTENT OR MISSING SOILPARM FILE") from exc
    slcats = int(_tokens(soil[sstart + 1])[0])
    sarr = {k: np.zeros(NSLTYPE) for k in (
        "bb", "drysmc", "hc", "maxsmc", "refsmc", "satpsi", "satdk", "satdw", "wltsmc", "qtz")}
    for lc in range(slcats):
        t = _tokens(soil[sstart + 2 + lc])
        for name, val in zip(("bb", "drysmc", "hc", "maxsmc", "refsmc", "satpsi", "satdk",
                              "satdw", "wltsmc", "qtz"), t[1:11]):
            sarr[name][lc] = float(val)
    return RucTables(
        mminluruc=mminluruc, lucats=lucats, slcats=slcats, ifortbl=ifortbl,
        topt_data=float(_after("TOPT_DATA")), cmcmax_data=float(_after("CMCMAX_DATA")),
        cfactr_data=float(_after("CFACTR_DATA")), rsmax_data=float(_after("RSMAX_DATA")),
        bare=int(_after("BARE")), natural=int(_after("NATURAL")), crop=int(_after("CROP")),
        urban=int(_after("URBAN")), **arr, **sarr,
    )


# ---------------------------------------------------------------------------
# Configuration / vertical grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RucConfig:
    """Static (trace-time) RUC configuration of one domain."""

    dt: float
    zs: tuple[float, ...]
    dtype: str = "float32"
    frpcpn: bool = False
    myj: bool = False
    mosaic_lu: int = 0
    mosaic_soil: int = 0
    rdlai2d: bool = False
    iswater: int = 16
    isice: int = 24
    xice_threshold: float = 0.5
    # constants as passed by the WRF surface driver (cp, rcp, g, xlv, stbolt)
    cp: float = 1004.5
    rovcp: float = 287.0 / 1004.5
    g0: float = 9.81
    lv: float = 2.5e6
    stbolt: float = 5.67051e-8
    #: how to resolve WRF's zshalf(nroot+1) out-of-bounds read (see module doc):
    #: "fail" -> column flagged unsupported; "bottom" -> use zs(nzs)
    oob_root_policy: str = "fail"

    @property
    def nzs(self) -> int:
        return len(self.zs)

    @property
    def np_dtype(self):
        return np.dtype(self.dtype)


def _c(value, dtype) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


@dataclass(frozen=True)
class _Grid:
    zsmain: np.ndarray
    zshalf: np.ndarray
    dtdzs: np.ndarray
    dtdzs2: np.ndarray


def _grid(cfg: RucConfig) -> _Grid:
    """LSMRUC :699-729 in the working precision."""

    dt_ = cfg.np_dtype
    nzs = cfg.nzs
    zsmain = np.zeros(nzs, dtype=dt_)
    zshalf = np.zeros(nzs, dtype=dt_)
    for k in range(2, nzs + 1):
        zsmain[k - 1] = _c(cfg.zs[k - 1], dt_)
        zshalf[k - 1] = _c(0.5, dt_) * (zsmain[k - 2] + zsmain[k - 1])
    dtdzs = np.zeros(2 * (nzs - 2), dtype=dt_)
    dtdzs2 = np.zeros(nzs, dtype=dt_)
    delt = _c(cfg.dt, dt_)
    for k in range(2, nzs):
        k1 = 2 * k - 3
        k2 = k1 + 1
        x = delt / _c(2.0, dt_) / (zshalf[k] - zshalf[k - 1])
        dtdzs[k1 - 1] = x / (zsmain[k - 1] - zsmain[k - 2])
        dtdzs2[k - 2] = x
        dtdzs[k2 - 1] = x / (zsmain[k] - zsmain[k - 1])
    return _Grid(zsmain, zshalf, dtdzs, dtdzs2)


def tbq_table(dtype) -> np.ndarray:
    """LSMRUC :458-475: saturation table, ``cq`` accumulated by repeated +0.05."""

    dt_ = np.dtype(dtype)
    cq = _c(173.15, dt_) - _c(0.05, dt_)
    r61 = _c(6.1153, dt_) * _c(0.62198, dt_)
    out = np.zeros(5001, dtype=dt_)
    c05 = _c(0.05, dt_)
    c27315 = _c(273.15, dt_)
    c2965 = _c(29.65, dt_)
    c1767 = _c(17.67, dt_)
    c22514 = _c(22.514, dt_)
    c615e3 = _c(6.15e3, dt_)
    for k in range(5001):
        cq = (cq + c05).astype(dt_)
        arg_v = (c1767 * (cq - c27315) / (cq - c2965)).astype(dt_)
        arg_i = (c22514 - c615e3 / cq).astype(dt_)
        # libm exp in the working precision (float64 exp rounded once)
        evs = np.asarray(np.exp(np.float64(arg_v)), dtype=dt_)
        eis = np.asarray(np.exp(np.float64(arg_i)), dtype=dt_)
        out[k] = r61 * evs if cq >= c27315 else r61 * eis
    return out


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

W = jnp.where


def _take(table: np.ndarray, idx, dtype):
    tab = jnp.asarray(np.asarray(table, dtype=dtype))
    return jnp.take(tab, jnp.clip(idx, 1, tab.shape[0]) - 1)


def _take_i(table: np.ndarray, idx):
    tab = jnp.asarray(np.asarray(table, dtype=np.int32))
    return jnp.take(tab, jnp.clip(idx, 1, tab.shape[0]) - 1)


def qsn(tn, tbq):
    """WRF ``QSN`` (table interpolation with clamping)."""

    dt_ = tn.dtype
    r = (tn - 173.15) / 0.05 + 1.0
    i = jnp.trunc(r).astype(jnp.int32)
    lo = i < 1
    i = W(lo, 1, i)
    r = W(lo, jnp.asarray(1.0, dt_), r)
    hi = i > 5000
    i = W(hi, 5000, i)
    r = W(hi, jnp.asarray(5001.0, dt_), r)
    t = jnp.asarray(tbq)
    r1 = jnp.take(t, i - 1)
    r2 = r - i.astype(dt_)
    return (jnp.take(t, i) - r1) * r2 + r1


def vilka(tn, d1, d2, pp, tbq):
    """WRF ``VILKA``: tabulated solution of the surface energy budget.

    Returns ``(qs, ts, ok)``; ``ok`` is False where WRF would stop with
    'crash in surface energy budget' or the iteration did not converge.
    """

    dt_ = tn.dtype
    t = jnp.asarray(tbq)
    c05 = 0.05

    def tt(i):
        return jnp.take(t, jnp.clip(i, 1, 5001) - 1)

    i = jnp.trunc((tn - 173.15) / c05 + 1.0).astype(jnp.int32)
    t1 = 173.1 + i.astype(dt_) * c05
    f1 = t1 + d1 * tt(i) - d2
    i1 = jnp.trunc(i.astype(dt_) - f1 / (c05 + d1 * (tt(i + 1) - tt(i)))).astype(jnp.int32)
    i = i1
    ok = (i <= 5000) & (i >= 1)
    rn0 = jnp.zeros_like(tn)

    def body(_, carry):
        i, t1, rn, done, ok = carry
        active = (~done) & ok
        i1 = i
        t1n = 173.1 + i.astype(dt_) * c05
        f1 = t1n + d1 * tt(i) - d2
        rnn = f1 / (c05 + d1 * (tt(i + 1) - tt(i)))
        inew = i - jnp.trunc(rnn).astype(jnp.int32)
        oknew = (inew <= 5000) & (inew >= 1)
        conv = inew == i1
        return (
            W(active, inew, i),
            W(active, t1n, t1),
            W(active, rnn, rn),
            done | (active & conv),
            ok & W(active, oknew, True),
        )

    i, t1, rn, done, ok = lax.fori_loop(
        0, VILKA_MAX_ITERS, body, (i, t1, rn0, jnp.zeros(tn.shape, bool), ok)
    )
    ts = t1 - c05 * rn
    qs = (tt(i) + (tt(i) - tt(i + 1)) * rn) / pp
    return qs, ts, ok & done


# ---------------------------------------------------------------------------
# SOILVEGIN
# ---------------------------------------------------------------------------


def soilvegin(cfg: RucConfig, tab: RucTables, *, ivgtyp, isltyp, vegfra, shdmin, shdmax,
              znt, lai, landusef=None, soilctop=None):
    dt_ = cfg.np_dtype
    lu_n = tab.ifortbl.shape[0]
    iforest = _take_i(tab.ifortbl, ivgtyp)
    factor = W(
        (shdmax - shdmin) < 1.0,
        jnp.asarray(1.0, dt_),
        1.0 - jnp.maximum(0.0, jnp.minimum(1.0, (vegfra - shdmin) / jnp.maximum(1.0, shdmax - shdmin))),
    )

    def today(k):  # category k (1-based): (laitoday, znttoday) per column
        ifk = int(tab.ifortbl[k - 1])
        lai_k = _c(tab.laitbl[k - 1], dt_)
        if ifk == 1:
            dlai = np.minimum(_c(0.2, dt_), _c(0.8, dt_) * lai_k)
        elif ifk in (2, 7):
            dlai = np.minimum(_c(0.5, dt_), _c(0.8, dt_) * lai_k)
        elif ifk == 3:
            dlai = np.minimum(_c(0.45, dt_), _c(0.8, dt_) * lai_k)
        elif ifk == 4:
            dlai = np.minimum(_c(0.75, dt_), _c(0.8, dt_) * lai_k)
        elif ifk == 5:
            dlai = np.minimum(_c(0.86, dt_), _c(0.8, dt_) * lai_k)
        else:
            dlai = _c(0.0, dt_)
        z0k = _c(tab.z0tbl[k - 1], dt_)
        if k != cfg.iswater:
            laik = lai_k - dlai * factor
            zntk = z0k - 0.125 * factor if ifk == 7 else jnp.broadcast_to(jnp.asarray(z0k), factor.shape)
        else:
            laik = jnp.broadcast_to(jnp.asarray(lai_k), factor.shape)
            zntk = znt
        return laik, zntk

    if cfg.mosaic_lu == 1:
        nlcat = landusef.shape[-1]
        lb = 5.0
        area = jnp.zeros_like(vegfra)
        emiss = jnp.zeros_like(vegfra)
        zn = jnp.zeros_like(vegfra)
        lai_m = jnp.zeros_like(vegfra)
        pc = jnp.zeros_like(vegfra)
        for k in range(1, nlcat + 1):
            frac = landusef[:, k - 1]
            laik, zntk = today(k) if k <= lu_n else (jnp.zeros_like(vegfra), jnp.zeros_like(vegfra))
            area = area + frac
            emiss = emiss + _c(tab.lemitbl[k - 1] if k <= NLUS else 0.0, dt_) * frac
            zn = zn + frac / jnp.log(lb / zntk) ** 2.0
            lai_m = lai_m + laik * frac
            pc = pc + _c(tab.pctbl[k - 1] if k <= NLUS else 0.0, dt_) * frac
        area = W(area > 1.0, jnp.asarray(1.0, dt_), area)
        bad_area = area <= 0.0
        emiss = emiss / area
        znt_out = lb / jnp.exp(jnp.sqrt(1.0 / zn))
        lai_out = lai if cfg.rdlai2d else lai_m / area
        pc = pc / area
    else:
        bad_area = jnp.zeros(vegfra.shape, bool)
        emiss = _take(tab.lemitbl, ivgtyp, dt_)
        pc = _take(tab.pctbl, ivgtyp, dt_)
        lai_tab = jnp.zeros_like(vegfra)
        znt_tab = jnp.zeros_like(vegfra)
        for k in range(1, lu_n + 1):
            if k > tab.lucats and tab.z0tbl[k - 1] == 0.0 and tab.laitbl[k - 1] == 0.0 and k != cfg.iswater:
                continue
            laik, zntk = today(k)
            sel = ivgtyp == k
            lai_tab = W(sel, laik, lai_tab)
            znt_tab = W(sel, zntk, znt_tab)
        znt_out = znt_tab
        lai_out = lai if cfg.rdlai2d else lai_tab

    names = ("hc", "bb", "maxsmc", "drysmc", "satdk", "satpsi", "refsmc", "wltsmc", "qtz")
    if cfg.mosaic_soil == 1:
        nscat = soilctop.shape[-1]
        acc = {n: jnp.zeros_like(vegfra) for n in ("rhocs", "bclh", "dqm", "ksat", "psis", "qmin", "ref", "wilt", "qwrtz")}
        area_s = jnp.zeros_like(vegfra)
        for k in range(1, nscat + 1):
            if k == 14:
                continue
            f = soilctop[:, k - 1]
            g = {n: _c(getattr(tab, n)[k - 1] if k <= NSLTYPE else 0.0, dt_) for n in names}
            area_s = area_s + f
            acc["rhocs"] = acc["rhocs"] + g["hc"] * _c(1.0e6, dt_) * f
            acc["bclh"] = acc["bclh"] + g["bb"] * f
            acc["dqm"] = acc["dqm"] + (g["maxsmc"] - g["drysmc"]) * f
            acc["ksat"] = acc["ksat"] + g["satdk"] * f
            acc["psis"] = acc["psis"] - g["satpsi"] * f
            acc["qmin"] = acc["qmin"] + g["drysmc"] * f
            acc["ref"] = acc["ref"] + g["refsmc"] * f
            acc["wilt"] = acc["wilt"] + g["wltsmc"] * f
            acc["qwrtz"] = acc["qwrtz"] + g["qtz"] * f
        area_s = W(area_s > 1.0, jnp.asarray(1.0, dt_), area_s)
        dom = _dominant_soil(tab, isltyp, dt_, water_zero=False)
        use_dom = area_s <= 0.0
        soil = {n: W(use_dom, dom[n], acc[n] / area_s) for n in acc}
    else:
        soil = _dominant_soil(tab, isltyp, dt_, water_zero=True)
    return dict(iforest=iforest, emiss=emiss, pc=pc, znt=znt_out, lai=lai_out,
                bad_area=bad_area, **soil)


def _dominant_soil(tab, isltyp, dt_, *, water_zero: bool):
    hc = _take(tab.hc, isltyp, dt_)
    out = dict(
        rhocs=hc * _c(1.0e6, dt_),
        bclh=_take(tab.bb, isltyp, dt_),
        dqm=_take(tab.maxsmc, isltyp, dt_) - _take(tab.drysmc, isltyp, dt_),
        ksat=_take(tab.satdk, isltyp, dt_),
        psis=-_take(tab.satpsi, isltyp, dt_),
        qmin=_take(tab.drysmc, isltyp, dt_),
        ref=_take(tab.refsmc, isltyp, dt_),
        wilt=_take(tab.wltsmc, isltyp, dt_),
        qwrtz=_take(tab.qtz, isltyp, dt_),
    )
    if water_zero:
        wat = isltyp == 14
        out = {k: W(wat, jnp.zeros_like(v), v) for k, v in out.items()}
    return out


# ---------------------------------------------------------------------------
# soil physics building blocks
# ---------------------------------------------------------------------------


def _freeze_partition(tso, soilmois, keepfr, smfrkeep, p, riw, keep_coef):
    """SOIL :2474-2496 / SNOWSOIL :3442-3464 liquid/ice partition per level.

    ``keep_coef`` is ``riw`` in SOIL and ``rhoice*1.e-3`` evaluated as
    ``soilice*rhoice*1.e-3`` in SNOWSOIL's first pass (``None``).
    """

    dqm, qmin, psis, bclh = p["dqm"], p["qmin"], p["psis"], p["bclh"]
    xlmelt = 3.35e5
    tln = jnp.log(tso / 273.15)
    frozen = tln < 0.0
    liq = (dqm[:, None] + qmin[:, None]) * (
        (xlmelt * (tso - 273.15) / tso / 9.81 / psis[:, None]) ** (-1.0 / bclh[:, None])
    ) - qmin[:, None]
    liq = jnp.maximum(0.0, liq)
    liq = jnp.minimum(liq, soilmois)
    ice = (soilmois - liq) / riw
    kf = keepfr == 1.0
    ice_k = jnp.minimum(ice, smfrkeep)
    if keep_coef is None:
        liq_k = jnp.maximum(0.0, soilmois - ice_k * 900.0 * 1.0e-3)
    else:
        liq_k = jnp.maximum(0.0, soilmois - ice_k * riw)
    ice = W(kf, ice_k, ice)
    liq = W(kf, liq_k, liq)
    soilice = W(frozen, ice, 0.0)
    soiliqw = W(frozen, liq, soilmois)
    return soilice, soiliqw


def _mid_partition(tso, soilmois, keepfr, smfrkeep, p, riw):
    """SOIL :2498-2529 (layer-mid values for SOILPROP)."""

    dqm, qmin, psis, bclh = p["dqm"], p["qmin"], p["psis"], p["bclh"]
    xlmelt = 3.35e5
    tav = 0.5 * (tso[:, :-1] + tso[:, 1:])
    soilmoism = 0.5 * (soilmois[:, :-1] + soilmois[:, 1:])
    tavln = jnp.log(tav / 273.15)
    frozen = tavln < 0.0
    liqm0 = (dqm[:, None] + qmin[:, None]) * (
        (xlmelt * (tav - 273.15) / tav / 9.81 / psis[:, None]) ** (-1.0 / bclh[:, None])
    ) - qmin[:, None]
    fwsat = dqm[:, None] - liqm0
    lwsat = liqm0 + qmin[:, None]
    liqm = jnp.minimum(jnp.maximum(0.0, liqm0), soilmoism)
    icem = (soilmoism - liqm) / riw
    kf = keepfr[:, :-1] == 1.0
    icem_k = jnp.minimum(icem, 0.5 * (smfrkeep[:, :-1] + smfrkeep[:, 1:]))
    liqm_k = jnp.maximum(0.0, soilmoism - icem_k * riw)
    icem = W(kf, icem_k, icem)
    liqm = W(kf, liqm_k, liqm)
    fwsat = W(kf, dqm[:, None] - liqm_k, fwsat)
    lwsat = W(kf, liqm_k + qmin[:, None], lwsat)
    soilicem = W(frozen, icem, 0.0)
    soiliqwm = W(frozen, liqm, soilmoism)
    lwsat = W(frozen, lwsat, dqm[:, None] + qmin[:, None])
    fwsat = W(frozen, fwsat, 0.0)
    return tav, soilmoism, soilicem, soiliqwm, fwsat, lwsat


def soilprop(cfg, p, *, fwsat, lwsat, tav, keepfr, soilmois, soiliqw, soilice, soilmoism,
             soiliqwm, soilicem, riw, cvw, ci, kqwrtz=7.7, kice=2.2, kwt=0.57):
    """WRF ``SOILPROP``; returns (thdif, diffu, hydro, cap), each (ncol, nzs)."""

    cp, g0_p = cfg.cp, cfg.g0
    dqm, qmin, psis, bclh, ksat = p["dqm"], p["qmin"], p["psis"], p["bclh"], p["ksat"]
    qwrtz, rhocs = p["qwrtz"], p["rhocs"]
    xlmelt = 3.35e5
    kzero = 2.0
    ws = dqm + qmin
    x1 = xlmelt / (g0_p * psis)
    x2 = x1 / bclh * ws
    x4 = (bclh + 1.0) / bclh
    gamd = (1.0 - ws) * 2700.0
    kdry = (0.135 * gamd + 64.7) / (2700.0 - 0.947 * gamd)
    kq = jnp.power(jnp.asarray(kqwrtz, qwrtz.dtype), qwrtz)
    kas = W(qwrtz > 0.2,
            kq * jnp.power(jnp.asarray(kzero, qwrtz.dtype), 1.0 - qwrtz),
            kq * jnp.power(jnp.asarray(3.0, qwrtz.dtype), 1.0 - qwrtz))

    wsb, qminb, dqmb, psisb, bclhb, ksatb = (v[:, None] for v in (ws, qmin, dqm, psis, bclh, ksat))
    tn = tav - 273.15
    keep_m = keepfr[:, :-1]
    detal = W(
        (soilicem != 0.0) & (tn < 0.0),
        273.15 * x2[:, None] / (tav * tav) * jnp.power(tav / (x1[:, None] * tn), x4[:, None]),
        0.0,
    )
    detal = W((soilicem != 0.0) & (tn < 0.0) & (keep_m == 1.0), 0.0, detal)
    kasat = (jnp.power(kas[:, None], 1.0 - wsb) * jnp.power(jnp.asarray(kice, tav.dtype), fwsat)
             * jnp.power(jnp.asarray(kwt, tav.dtype), lwsat))
    x5 = (soilmoism + qminb) / wsb
    sr = jnp.maximum(0.101, x5)
    ke = W(soilicem == 0.0, jnp.log10(sr) + 1.0, x5)
    kjpl = ke * (kasat - kdry[:, None]) + kdry[:, None]
    cap_m = ((1.0 - wsb) * rhocs[:, None] + (soiliqwm + qminb) * cvw + soilicem * ci
             + (dqmb - soilmoism) * cp * 1.2 - detal * 1.0e3 * xlmelt)
    a = riw * soilicem
    h = jnp.maximum(0.0, (soilmoism + qminb - a) / jnp.maximum(1.0e-8, wsb - a))
    facd = W(a != 0.0, 1.0 - a / jnp.maximum(1.0e-8, soilmoism), 1.0)
    ame = jnp.maximum(1.0e-8, wsb - riw * soilicem)
    diffu_m = (-bclhb * ksatb * psisb / ame * jnp.power(wsb / ame, 3.0)
               * jnp.power(h, bclhb + 2.0) * facd)
    diffu_m = W((wsb - a) < 0.12, 0.0, diffu_m)
    thdif_m = kjpl / cap_m

    zcol = jnp.zeros_like(tav[:, :1])
    thdif = jnp.concatenate([thdif_m, zcol], axis=1)
    diffu = jnp.concatenate([diffu_m, zcol], axis=1)
    cap = jnp.concatenate([cap_m, zcol], axis=1)

    fach = W(soilice != 0.0, 1.0 - riw * soilice / jnp.maximum(1.0e-8, soilmois), 1.0)
    am = jnp.maximum(1.0e-8, wsb - riw * soilice)
    hydro = jnp.minimum(ksatb, ksatb / am * jnp.power(soiliqw / am, 2.0 * bclhb + 2.0) * fach)
    hydro = W(hydro < 1.0e-10, 0.0, hydro)
    hydro = W((wsb - riw * soilice) < 0.12, 0.0, hydro)
    return thdif, diffu, hydro, cap


def transf(cfg, tab, p, grid, *, nroot, soiliqw, tabs, lai, gswin, pc, iland):
    """WRF ``TRANSF``: root-zone transpiration weights (ncol, nzs) and their sum."""

    dt_ = cfg.np_dtype
    qmin, ref, wilt = p["qmin"], p["ref"], p["wilt"]
    nzs = cfg.nzs
    zsh = _zshalf_ext(cfg, grid)
    cols = []
    for k in range(nzs):
        totliq = soiliqw[:, k] + qmin
        did = zsh[1] if k == 0 else zsh[k + 1] - zsh[k]
        full = (totliq > ref) if k == 0 else (totliq >= ref)
        tk = W(full, jnp.asarray(did, dt_),
               W(totliq <= wilt, 0.0, (totliq - wilt) / (ref - wilt) * did))
        cols.append(W(k < nroot, tk, 0.0))
    pctot = W(lai > 4.0, 0.8, pc)
    ftem = W(tabs <= 302.15,
             1.0 / (1.0 + jnp.exp(-0.41 * (tabs - 282.05))),
             1.0 / (1.0 + jnp.exp(0.5 * (tabs - 314.0))))
    cmin = _c(1.0, dt_) / _c(tab.rsmax_data, dt_)
    rst = _take(tab.rstbl, iland, dt_)
    rgl = _take(tab.rgltbl, iland, dt_)
    cmax = W(lai > 1.0, lai / rst, 1.0 / rst)
    fsol = W(gswin < rgl, 1.0 / (1.0 + jnp.exp(-0.034 * (gswin - 3.5))), 1.0)
    totcnd = (cmin + (cmax - cmin) * pctot * ftem * fsol) / cmax
    transum = jnp.zeros_like(tabs)
    out = []
    for k in range(nzs):
        tk = W(k < nroot, jnp.maximum(cmin, cols[k] * totcnd), 0.0)
        transum = W(k < nroot, transum + tk, transum)
        out.append(tk)
    return jnp.stack(out, axis=1), transum


def _zshalf_ext(cfg, grid):
    """zshalf with one extra entry for the WRF zshalf(nroot+1) read.

    WRF reads past the array when nroot == nzs (module doc); the pad is zs(nzs) for
    ``oob_root_policy="bottom"`` and irrelevant for "fail" (column flagged unsupported).
    """

    del cfg
    pad = grid.zsmain[-1]
    return np.concatenate([grid.zshalf, np.asarray([pad], dtype=grid.zshalf.dtype)])


def _zroot(cfg, grid, nroot):
    zsh = jnp.asarray(_zshalf_ext(cfg, grid))
    return jnp.take(zsh, nroot)  # zshalf(nroot+1) 1-based


def _soil_tridiag_t(grid, thdif, tso):
    """SOILTEMP/SNOWTEMP :4705-4717 coefficients (cotso, rhtso) as lists (1-based k)."""

    nzs = tso.shape[1]
    cotso = [None] * (nzs + 1)
    rhtso = [None] * (nzs + 1)
    cotso[1] = jnp.zeros_like(tso[:, 0])
    rhtso[1] = tso[:, nzs - 1]
    for k in range(1, nzs - 1):
        kn = nzs - k
        k1 = 2 * kn - 3
        x1 = grid.dtdzs[k1 - 1] * thdif[:, kn - 2]
        x2 = grid.dtdzs[k1] * thdif[:, kn - 1]
        ft = tso[:, kn - 1] + x1 * (tso[:, kn - 2] - tso[:, kn - 1]) - x2 * (tso[:, kn - 1] - tso[:, kn])
        denom = 1.0 + x1 + x2 - x2 * cotso[k]
        cotso[k + 1] = x1 / denom
        rhtso[k + 1] = (ft + x2 * rhtso[k]) / denom
    return cotso, rhtso


def soiltemp(cfg, grid, tbq, *, nroot, delt, conflx, prcpms, rainf, patm, tabs, qvatm,
             emiss, rnet, qkms, tkms, rho, vegfrac, thdif, cap, drycan, wetcan, transum,
             mavail, soilres, tso, soilt, qvg, qsg, qcg):
    """WRF ``SOILTEMP`` (snow-free surface energy budget + soil heat diffusion)."""

    nzs = cfg.nzs
    cp, xlv, stbolt, cvw = cfg.cp, cfg.lv, cfg.stbolt, 4.183e6
    dzstop = 1.0 / (grid.zsmain[1] - grid.zsmain[0])
    qgold = qvg
    cotso, rhtso = _soil_tridiag_t(grid, thdif, tso)
    rhcs = cap[:, 0]
    h = mavail
    trans = transum * drycan / _zroot(cfg, grid, nroot)
    can = wetcan + trans
    umveg = (1.0 - vegfrac) * soilres
    d1 = cotso[nzs - 1]
    d2 = rhtso[nzs - 1]
    tn = soilt
    d9 = thdif[:, 0] * rhcs * dzstop
    d10 = tkms * cp * rho
    r211 = 0.5 * conflx / delt
    r21 = r211 * cp * rho
    r22 = 0.5 / (thdif[:, 0] * delt * (dzstop * dzstop))
    tn2 = tn * tn
    r6 = emiss * stbolt * 0.5 * (tn2 * tn2)
    r7 = r6 / tn
    d11 = rnet + r6
    tdenom = d9 * (1.0 - d1 + r22) + d10 + r21 + r7 + rainf * cvw * prcpms
    fkq = qkms * rho
    r210 = r211 * rho
    c = vegfrac * fkq * can
    cc = c * xlv / tdenom
    aa = xlv * (fkq * umveg + r210) / tdenom
    bb = (d10 * tabs + r21 * tn + xlv * (qvatm * (fkq * umveg + c) + r210 * qvg) + d11
          + d9 * (d2 + r22 * tn) + rainf * cvw * prcpms * jnp.maximum(273.15, tabs)) / tdenom
    aa1 = aa + cc
    pp = patm * 1.0e3
    aa1 = aa1 / pp
    qs1a, ts1a, ok_a = vilka(tn, aa1, bb, pp, tbq)
    tx2 = qvatm * (1.0 - h)
    q1a = tx2 + h * qs1a
    to100 = q1a < qs1a
    bb2 = bb - aa * tx2
    aa2 = (aa * h + cc) / pp
    qs1b, ts1b, ok_b = vilka(tn, aa2, bb2, pp, tbq)
    q1b = tx2 + h * qs1b
    back90 = q1b >= qs1b
    # 90 reached directly or from 100
    qs1 = W(to100, qs1b, qs1a)
    ts1 = W(to100, ts1b, ts1a)
    q1 = W(to100, q1b, q1a)
    at90 = (~to100) | back90
    qvg_n = W(at90, qs1, q1)
    qsg_n = qs1
    qcg_n = W(at90, jnp.maximum(0.0, q1 - qs1), 0.0)
    ok = W(to100, ok_b, ok_a)
    soilt_n = ts1
    tso_cols = [ts1]
    for k in range(2, nzs + 1):
        kk = nzs - k + 1
        tso_cols.append(rhtso[kk] + cotso[kk] * tso_cols[-1])
    tso_n = jnp.stack(tso_cols, axis=1)
    x = ((cp * rho * r211 + rhcs * grid.zsmain[1] * 0.5 / delt) * (soilt_n - tn)
         + xlv * rho * r211 * (qvg_n - qgold))
    x = x - rainf * cvw * prcpms * (jnp.maximum(273.15, tabs) - soilt_n)
    return dict(tso=tso_n, soilt=soilt_n, qvg=qvg_n, qsg=qsg_n, qcg=qcg_n, x=x, ok=ok)


def soilmoist(cfg, grid, *, delt, diffu, hydro, qsg, qvg, qcg, qcatm, qvatm, prcp, qkms,
              transp, drip, dew, smelt, soilice, vegfrac, snowfrac, soilres, dqm, qmin,
              ref, ksat, ras, riw, soilmois):
    """WRF ``SOILMOIST`` (Richards equation + infiltration/runoff)."""

    del riw
    nzs = cfg.nzs
    zsmain, zshalf, dtdzs, dtdzs2 = grid.zsmain, grid.zshalf, grid.dtdzs, grid.dtdzs2
    cosmc = [None] * (nzs + 1)
    rhsmc = [None] * (nzs + 1)
    cosmc[1] = jnp.zeros_like(qsg)
    rhsmc[1] = soilmois[:, nzs - 1]
    for k in range(1, nzs - 1):
        kn = nzs - k
        k1 = 2 * kn - 3
        x4 = 2.0 * dtdzs[k1 - 1] * diffu[:, kn - 2]
        x2 = 2.0 * dtdzs[k1] * diffu[:, kn - 1]
        q4 = x4 + hydro[:, kn - 2] * dtdzs2[kn - 2]
        q2 = x2 - hydro[:, kn] * dtdzs2[kn - 2]
        denom = 1.0 + x2 + x4 - q2 * cosmc[k]
        cosmc[k + 1] = q4 / denom
        rhsmc[k + 1] = (soilmois[:, kn - 1] + q2 * rhsmc[k]
                        + transp[:, kn - 1] / (zshalf[kn] - zshalf[kn - 1]) * delt) / denom

    trans = transp[:, 0]
    umveg = (1.0 - vegfrac) * soilres
    runoff = jnp.zeros_like(qsg)
    runoff2 = jnp.zeros_like(qsg)
    dzs = zsmain[1]
    r1 = cosmc[nzs - 1]
    r2 = rhsmc[nzs - 1]
    r3 = diffu[:, 0] / dzs
    r4 = r3 + hydro[:, 0] * 0.5
    r5 = r3 - hydro[:, 1] * 0.5
    r6 = qkms * ras
    totliq = prcp - drip / delt - umveg * dew * ras - smelt
    flx = totliq

    cvfrz = 3.0
    refkdt = 3.0
    refdk = 3.4341e-6
    delt1 = delt / 86400.0
    f1max = dqm * zshalf[1]
    f1 = f1max * (1.0 - soilmois[:, 0] / dqm)
    dice = soilice[:, 0] * zshalf[1]
    fd = f1
    for k in range(2, nzs):
        dz = zshalf[k] - zshalf[k - 1]
        dice = dice + dz * soilice[:, k - 1]
        fkmax = dqm * dz
        fk = fkmax * (1.0 - soilmois[:, k - 1] / dqm)
        fd = fd + fk
    kdt = refkdt * ksat / refdk
    val = 1.0 - jnp.exp(-kdt * delt1)
    ddt = fd * val
    px = -totliq * delt
    px = W(px < 0.0, 0.0, px)
    infmax1 = W(px > 0.0, (px * (ddt / (px + ddt))) / delt, 0.0)
    frzx = 0.15 * ((dqm + qmin) / ref) * (np.asarray(0.412, qsg.dtype) / np.asarray(0.468, qsg.dtype))
    acrt = cvfrz * frzx / dice
    ssum = 1.0
    ssum = ssum + jnp.power(acrt, cvfrz - 1.0) / 2.0
    ssum = ssum + jnp.power(acrt, cvfrz - 2.0) / 1.0
    fcr = W(dice > 1.0e-2, 1.0 - jnp.exp(-acrt) * ssum, 1.0)
    infmax1 = infmax1 * fcr
    infmax = jnp.maximum(infmax1, hydro[:, 0] * soilmois[:, 0])
    infmax = jnp.minimum(infmax, -totliq)
    over = -totliq > infmax
    runoff = W(over, -totliq - infmax, runoff)
    flx = W(over, -infmax, flx)
    infiltrp = flx
    r7 = 0.5 * dzs / delt
    r4 = r4 + r7
    flx = flx - soilmois[:, 0] * r7
    r8 = umveg * r6 * (1.0 - snowfrac)
    qtot = qvatm + qcatm
    r9 = trans
    r10 = qtot - qsg
    ev = r10 <= 0.0
    den_e = r4 - r5 * r1 - r10 * r8 / (ref - qmin)
    qq = W(ev, (r5 * r2 - flx + r9) / den_e,
           (r2 * r5 - flx + r8 * (qtot - qcg - qvg) + r9) / (r4 - r1 * r5))
    flxsat = W(ev, -dqm * den_e + r5 * r2 + r9,
               -dqm * (r4 - r1 * r5) + r2 * r5 + r8 * (qtot - qvg - qcg) + r9)
    neg = qq < 0.0
    sat = (~neg) & (qq > dqm)
    top = W(neg, 1.0e-8, W(sat, dqm, jnp.minimum(dqm, jnp.maximum(1.0e-8, qq))))
    runoff = W(sat, runoff + (flxsat - flx), runoff)
    cols = [top]
    for k in range(2, nzs + 1):
        kk = nzs - k + 1
        qq = cosmc[kk] * cols[-1] + rhsmc[kk]
        neg = qq < 0.0
        sat = (~neg) & (qq > dqm)
        thick = (zsmain[k - 1] - zshalf[k - 1]) if k == nzs else (zshalf[k] - zshalf[k - 1])
        runoff2 = W(sat, runoff2 + ((qq - dqm) * thick) / delt, runoff2)
        cols.append(W(neg, 1.0e-8, W(sat, dqm, jnp.minimum(dqm, jnp.maximum(1.0e-8, qq)))))
    soilmois_n = jnp.stack(cols, axis=1)
    mavail = jnp.maximum(0.00001, jnp.minimum(1.0, (soilmois_n[:, 0] / (ref - qmin) * (1.0 - snowfrac) + 1.0 * snowfrac)))
    return dict(soilmois=soilmois_n, mavail=mavail, runoff1=runoff, runoff2=runoff2, infiltrp=infiltrp)


def _keepfr_update(keepfr, soilice, tso, told, soilmois, smold):
    upd = W((tso > told) & (soilmois > smold), 1.0, 0.0)
    return W(soilice > 0.0, upd, keepfr)




# ---------------------------------------------------------------------------
# SOIL (snow-free column)
# ---------------------------------------------------------------------------


def soil(cfg, tab, grid, tbq, p, *, iland, nroot, delt, conflx, prcpms, rainf, patm, qvatm,
         qcatm, gswin, emiss, rnet, qkms, tkms, pc, cst, drip, infwater, rho, vegfrac, lai,
         tabs, sat, cn, soilmois, tso, smfrkeep, keepfr, soilt, qvg, qsg, qcg, mavail):
    """WRF ``SOIL``.  Returns a dict of the INOUT/OUT arguments."""

    cp, xlv, rovcp = cfg.cp, cfg.lv, cfg.rovcp
    dqm, qmin, ref, ksat = p["dqm"], p["qmin"], p["ref"], p["ksat"]
    dt_ = cfg.np_dtype
    rhoice = 900.0
    ci = _c(rhoice, dt_) * _c(2100.0, dt_)
    cvw = 4.183e6
    ras = rho * 1.0e-3
    riw = _c(rhoice, dt_) * _c(1.0e-3, dt_)
    dzstop = _c(1.0, dt_) / (grid.zsmain[1] - grid.zsmain[0])

    soilice, soiliqw = _freeze_partition(tso, soilmois, keepfr, smfrkeep, p, riw, riw)
    tav, soilmoism, soilicem, soiliqwm, fwsat, lwsat = _mid_partition(tso, soilmois, keepfr, smfrkeep, p, riw)
    smfrkeep = W(soilice > 0.0, soilice, soilmois / riw)
    thdif, diffu, hydro, cap = soilprop(
        cfg, p, fwsat=fwsat, lwsat=lwsat, tav=tav, keepfr=keepfr, soilmois=soilmois,
        soiliqw=soiliqw, soilice=soilice, soilmoism=soilmoism, soiliqwm=soiliqwm,
        soilicem=soilicem, riw=riw, cvw=cvw, ci=ci)

    wetcan = jnp.minimum(0.25, jnp.power(jnp.maximum(0.0, cst / sat), cn))
    drycan = 1.0 - wetcan
    tranf, transum = transf(cfg, tab, p, grid, nroot=nroot, soiliqw=soiliqw, tabs=tabs,
                            lai=lai, gswin=gswin, pc=pc, iland=iland)
    told, smold = tso, soilmois
    fc = jnp.maximum(qmin, ref * 0.5)
    fex_fc = jnp.maximum(jnp.minimum(1.0, (soilmois[:, 0] + qmin) / fc), 0.01)
    one_m_cos = 1.0 - jnp.cos(_PICONST * fex_fc)
    soilres = W(((soilmois[:, 0] + qmin) > fc) | ((qvatm - qvg) > 0.0), 1.0,
                0.25 * (one_m_cos * one_m_cos))
    st = soiltemp(cfg, grid, tbq, nroot=nroot, delt=delt, conflx=conflx, prcpms=prcpms,
                  rainf=rainf, patm=patm, tabs=tabs, qvatm=qvatm, emiss=emiss, rnet=rnet,
                  qkms=qkms, tkms=tkms, rho=rho, vegfrac=vegfrac, thdif=thdif, cap=cap,
                  drycan=drycan, wetcan=wetcan, transum=transum, mavail=mavail,
                  soilres=soilres, tso=tso, soilt=soilt, qvg=qvg, qsg=qsg, qcg=qcg)
    tso, soilt, qvg, qsg, qcg, x = st["tso"], st["soilt"], st["qvg"], st["qsg"], st["qcg"], st["x"]

    cnd = qvatm >= qsg
    dew = W(cnd, qkms * (qvatm - qsg), 0.0)
    zroot = _zroot(cfg, grid, nroot)
    nzs = cfg.nzs
    tcols = []
    ett1 = jnp.zeros_like(qvatm)
    for k in range(nzs):
        tr = vegfrac * ras * qkms * (qvatm - qsg) * tranf[:, k] * drycan / zroot
        tr = W(tr > 0.0, 0.0, tr)
        inroot = k < nroot
        tr = W(inroot & (~cnd), tr, 0.0)
        ett1 = W(inroot & (~cnd), ett1 - tr, ett1)
        tcols.append(tr)
    transp = jnp.stack(tcols, axis=1)

    soilice, soiliqw = _freeze_partition(tso, soilmois, keepfr, smfrkeep, p, riw, riw)
    sm = soilmoist(cfg, grid, delt=delt, diffu=diffu, hydro=hydro, qsg=qsg, qvg=qvg, qcg=qcg,
                   qcatm=qcatm, qvatm=qvatm, prcp=-infwater, qkms=qkms, transp=transp,
                   drip=drip, dew=dew, smelt=0.0, soilice=soilice, vegfrac=vegfrac,
                   snowfrac=0.0, soilres=soilres, dqm=dqm, qmin=qmin, ref=ref, ksat=ksat,
                   ras=ras, riw=riw, soilmois=soilmois)
    soilmois = sm["soilmois"]
    keepfr = _keepfr_update(keepfr, soilice, tso, told, soilmois, smold)

    hft = -tkms * cp * rho * (tabs - soilt)
    hfx = -tkms * cp * rho * (tabs - soilt) * jnp.power(
        (_c(_P1000MB, dt_) * _c(0.00001, dt_)) / patm, rovcp)
    q1 = -qkms * ras * (qvatm - qsg)
    cond = q1 <= 0.0
    if cfg.myj:
        eeta_c = -qkms * ras * (qvatm / (1.0 + qvatm) - qsg / (1.0 + qsg)) * 1.0e3
        cst_c = cst - eeta_c * delt * vegfrac
    else:
        eeta_c = -rho * dew
        cst_c = cst + delt * dew * ras * vegfrac
    qfx_c = xlv * eeta_c
    eeta_c = -rho * dew
    edir1 = -soilres * (1.0 - vegfrac) * qkms * ras * (qvatm - qvg)
    ec1 = q1 * wetcan * vegfrac
    cst_e = jnp.maximum(0.0, cst - ec1 * delt)
    if cfg.myj:
        eeta_e = -soilres * qkms * ras * (qvatm / (1.0 + qvatm) - qvg / (1.0 + qvg)) * 1.0e3
    else:
        eeta_e = (edir1 + ec1 + ett1) * 1.0e3
    qfx_e = xlv * eeta_e
    eeta_e = (edir1 + ec1 + ett1) * 1.0e3
    eeta = W(cond, eeta_c, eeta_e)
    qfx = W(cond, qfx_c, qfx_e)
    cst = W(cond, cst_c, cst_e)
    edir1 = W(cond, 0.0, edir1)
    ec1 = W(cond, 0.0, ec1)
    ett1 = W(cond, 0.0, ett1)
    s = thdif[:, 0] * cap[:, 0] * dzstop * (tso[:, 0] - tso[:, 1])
    fltot = rnet - hft - xlv * eeta - s - x
    return dict(soilmois=soilmois, tso=tso, smfrkeep=smfrkeep, keepfr=keepfr, dew=dew,
                soilt=soilt, qvg=qvg, qsg=qsg, qcg=qcg, edir1=edir1, ec1=ec1, ett1=ett1,
                eeta=eeta, qfx=qfx, hfx=hfx, s=s, fltot=fltot, runoff1=sm["runoff1"],
                runoff2=sm["runoff2"], mavail=sm["mavail"], soilice=soilice,
                soiliqw=soiliqw, infiltr=sm["infiltrp"], cst=cst, ok=st["ok"])


# ---------------------------------------------------------------------------
# SNOWTEMP
# ---------------------------------------------------------------------------


def _thdifsn(rhosn, rhocsn, newsnow, rhonewsn, snhei):
    """SNOWTEMP :5050-5072 (isncond_opt=2, Sturm et al. 1997)."""

    keff = W((rhosn < 156.0) | ((newsnow > 0.0) & (rhonewsn < 156.0)),
             0.023 + 0.234 * rhosn * 1.0e-3,
             0.138 - 1.01 * rhosn * 1.0e-3 + 3.233 * (rhosn * rhosn) * 1.0e-6)
    return W((newsnow <= 0.0) & (snhei > 1.0) & (rhosn > 250.0), 4.431718e-7, keff / rhocsn * 1.0)


def snowtemp(cfg, grid, tbq, *, nroot, delt, conflx, snwe, snwepr, snhei, newsnow, snowfrac,
             beta, deltsn, snth, rhosn, rhonewsn, meltfactor, prcpms, rainf, patm, tabs,
             qvatm, glw, emiss, rnet, qkms, tkms, rho, vegfrac, thdif, cap, drycan, wetcan,
             tranf, transum, tso, soilt, soilt1, tsnav, qvg, qsg, qcg):
    """WRF ``SNOWTEMP`` incl. the snow-melt second iteration (goto 212)."""

    del glw, tsnav
    nzs = cfg.nzs
    dt_ = cfg.np_dtype
    cp, stbolt = cfg.cp, cfg.stbolt
    xlvm = _c(cfg.lv, dt_) + _c(3.35e5, dt_)
    cvw = 4.183e6
    xlmelt = 3.35e5
    rhocsn = 2090.0 * rhosn
    rhonewcsn = 2090.0 * rhonewsn
    thdifsn = _thdifsn(rhosn, rhocsn, newsnow, rhonewsn, snhei)
    ras = rho * 1.0e-3
    dzstop = _c(1.0, dt_) / (grid.zsmain[1] - grid.zsmain[0])
    qgold = qvg
    zero = jnp.zeros_like(soilt)
    cotso, rhtso = _soil_tridiag_t(grid, thdif, tso)
    nzs1 = nzs - 1

    deep = snhei >= snth
    one_l = deep & (snhei <= deltsn + snth)
    two_l = deep & (~one_l)
    thin = (snhei < snth) & (snhei > 0.0)

    # --- one-layer snow
    snprim1 = jnp.maximum(snth, snhei)
    xsn1l = delt / 2.0 / (grid.zshalf[1] + 0.5 * snprim1)
    x1sn1l = xsn1l / snprim1 * thdifsn
    x2 = grid.dtdzs[0] * thdif[:, 0]
    ft1 = tso[:, 0] + x1sn1l * (soilt - tso[:, 0]) - x2 * (tso[:, 0] - tso[:, 1])
    den1 = 1.0 + x1sn1l + x2 - x2 * cotso[nzs1]
    cot_n1 = x1sn1l / den1
    rht_n1 = (ft1 + x2 * rhtso[nzs1]) / den1
    # --- two-layer snow
    xsn2 = delt / 2.0 / (0.5 * deltsn)
    xsn2b = delt / 2.0 / (grid.zshalf[1] + 0.5 * (snhei - deltsn))
    x1sn2 = xsn2 / deltsn * thdifsn
    x1sn2b = xsn2b / (snhei - deltsn) * thdifsn
    ft2 = tso[:, 0] + x1sn2b * (soilt1 - tso[:, 0]) - x2 * (tso[:, 0] - tso[:, 1])
    den2 = 1.0 + x1sn2b + x2 - x2 * cotso[nzs1]
    cot_n2 = x1sn2b / den2
    rht_n2 = (ft2 + x2 * rhtso[nzs1]) / den2
    ftsnow = soilt1 + x1sn2 * (soilt - soilt1) - x1sn2b * (soilt1 - tso[:, 0])
    densn = 1.0 + x1sn2 + x1sn2b - x1sn2b * cot_n2
    cotsn2 = x1sn2 / densn
    rhtsn2 = (ftsnow + x1sn2b * rht_n2) / densn
    # --- thin snow combined with top soil layer
    snprim3 = snhei + grid.zsmain[1]
    fsn3 = snhei / snprim3
    fso3 = 1.0 - fsn3
    xsn3 = delt / 2.0 / ((grid.zshalf[2] - grid.zsmain[1]) + 0.5 * snprim3)
    x1sn3 = xsn3 / snprim3 * (fsn3 * thdifsn + fso3 * thdif[:, 0])
    x2b = grid.dtdzs[1] * thdif[:, 1]
    ft3 = tso[:, 1] + x1sn3 * (soilt - tso[:, 1]) - x2b * (tso[:, 1] - tso[:, 2])
    den3 = 1.0 + x1sn3 + x2b - x2b * cotso[nzs - 2]
    cot_n13 = x1sn3 / den3
    rht_n13 = (ft3 + x2b * rhtso[nzs - 2]) / den3

    cot_nzs = W(one_l, cot_n1, W(two_l, cot_n2, W(thin, cot_n13, zero)))
    rht_nzs = W(one_l, rht_n1, W(two_l, rht_n2, W(thin, rht_n13, zero)))
    cot_nzs1 = W(thin, cot_n13, cotso[nzs1])
    rht_nzs1 = W(thin, rht_n13, rhtso[nzs1])
    cotsn = W(two_l, cotsn2, cot_nzs)
    rhtsn = W(two_l, rhtsn2, rht_nzs)
    snprim = W(one_l, snprim1, W(two_l, deltsn, W(thin, snprim3, zero)))
    fsn = W(thin, fsn3, 1.0)
    fso = W(thin, fso3, 0.0)
    ilnb = W(one_l, 1, W(two_l, 2, 0))
    tsob0 = W(one_l, tso[:, 0], W(two_l, soilt1, W(thin, tso[:, 1], zero)))
    cotso_k = list(cotso) + [None]
    rhtso_k = list(rhtso) + [None]
    cotso_k[nzs] = cot_nzs
    rhtso_k[nzs] = rht_nzs
    cotso_k[nzs1] = cot_nzs1
    rhtso_k[nzs1] = rht_nzs1

    rhcs = cap[:, 0]
    h = 1.0
    trans = transum * drycan / _zroot(cfg, grid, nroot)
    can = wetcan + trans
    umveg = 1.0 - vegfrac
    d1 = cot_nzs1
    d2 = rht_nzs1
    tn = soilt
    d9 = thdif[:, 0] * rhcs * dzstop
    d10 = tkms * cp * rho
    r211 = 0.5 * conflx / delt
    r21 = r211 * cp * rho
    r22 = 0.5 / (thdif[:, 0] * delt * (dzstop * dzstop))
    tn2 = tn * tn
    r6 = emiss * stbolt * 0.5 * (tn2 * tn2)
    r7 = r6 / tn
    d11 = rnet + r6
    d1sn = W(deep, W(one_l, cot_nzs, cotsn), W(thin, d1, d1))
    d2sn = W(deep, W(one_l, rht_nzs, rhtsn), W(thin, d2, d2))
    d9sn = W(deep, thdifsn * rhocsn / snprim,
             W(thin, (fsn * thdifsn * rhocsn + fso * thdif[:, 0] * rhcs) / snprim, d9))
    r22sn = W(deep, snprim * snprim * 0.5 / (thdifsn * delt),
              W(thin, snprim * snprim * 0.5 / ((fsn * thdifsn + fso * thdif[:, 0]) * delt), r22))
    epot0 = -qkms * (qvatm - qgold)
    pp = patm * 1.0e3

    def energy_pass(snoh, beta, nmelt, snwe):
        tdenom = (d9sn * (1.0 - d1sn + r22sn) + d10 + r21 + r7 + rainf * cvw * prcpms
                  + rhonewcsn * newsnow / delt)
        fkq = qkms * rho
        r210 = r211 * rho
        c = vegfrac * fkq * can
        cc = c * xlvm / tdenom
        aa = xlvm * (beta * fkq * umveg + r210) / tdenom
        bb = (d10 * tabs + r21 * tn + xlvm * (qvatm * (beta * fkq * umveg + c) + r210 * qgold)
              + d11 + d9sn * (d2sn + r22sn * tn)
              + rainf * cvw * prcpms * jnp.maximum(273.15, tabs)
              + rhonewcsn * newsnow / delt * jnp.minimum(273.15, tabs)) / tdenom
        aa1 = aa + cc
        aa1 = aa1 / pp
        bb = bb - snoh / tdenom
        qs1a, ts1a, ok_a = vilka(tn, aa1, bb, pp, tbq)
        tx2 = qvatm * (1.0 - h)
        q1a = tx2 + h * qs1a
        to100 = q1a < qs1a
        bb2 = bb - aa * tx2
        aa2 = (aa * h + cc) / pp
        qs1b, ts1b, ok_b = vilka(tn, aa2, bb2, pp, tbq)
        q1b = tx2 + h * qs1b
        back90 = q1b > qs1b
        qs1 = W(to100, qs1b, qs1a)
        ts1 = W(to100, ts1b, ts1a)
        q1 = W(to100, q1b, q1a)
        at90 = (~to100) | back90
        qvg_n = W(at90, qs1, q1)
        qsg_n = qs1
        qcg_n = W(at90, jnp.maximum(0.0, q1 - qs1), 0.0)
        soilt_n = ts1
        soilt_n = W(nmelt & (snowfrac == 1.0) & (snwe > 0.0) & (soilt_n > 273.15),
                    jnp.minimum(273.15, soilt_n), soilt_n)
        # snow / top soil temperatures
        s1_two = jnp.minimum(273.15, rhtsn + cotsn * soilt_n)
        t1_two = rht_nzs + cot_nzs * s1_two
        t1_one = rht_nzs + cot_nzs * soilt_n
        t2_thin = rht_nzs1 + cot_nzs1 * soilt_n
        t1_thin = t2_thin + (soilt_n - t2_thin) * fso
        tso1 = W(two_l, t1_two, W(one_l, t1_one, W(thin, t1_thin, soilt_n)))
        soilt1_n = W(two_l, s1_two, W(one_l, t1_one, W(thin, t1_thin, soilt_n)))
        tsob = W(two_l, s1_two, W(one_l, t1_one, W(thin, t2_thin, tso1)))
        clamp = nmelt & (snowfrac == 1.0)
        soilt1_n = W(clamp, jnp.minimum(273.15, soilt1_n), soilt1_n)
        tso1 = W(clamp, jnp.minimum(273.15, tso1), tso1)
        tsob = W(clamp, jnp.minimum(273.15, tsob), tsob)
        # profile: thin -> k=3..nzs from tso(2); else k=2..nzs from tso(1)
        cols_a = [tso1]
        for k in range(2, nzs + 1):
            kk = nzs - k + 1
            cols_a.append(rhtso_k[kk] + cotso_k[kk] * cols_a[-1])
        cols_b = [tso1, t2_thin]
        for k in range(3, nzs + 1):
            kk = nzs - k + 1
            cols_b.append(rhtso_k[kk] + cotso_k[kk] * cols_b[-1])
        tso_n = jnp.stack([W(thin, b, a) for a, b in zip(cols_a, cols_b)], axis=1)
        return dict(qvg=qvg_n, qsg=qsg_n, qcg=qcg_n, soilt=soilt_n, soilt1=soilt1_n,
                    tso=tso_n, tsob=tsob, ok=W(to100, ok_b, ok_a), r210=r210)

    false = jnp.zeros(soilt.shape, bool)
    p1 = energy_pass(zero, beta, false, snwe)

    # ---- melt branch (evaluated on pass-1 results)
    melt = (p1["soilt"] > 273.15) & (beta == 1.0) & (snhei > 0.0)
    soiltfrac = snowfrac * 273.15 + (1.0 - snowfrac) * p1["soilt"]
    qsg_m = jnp.minimum(p1["qsg"], qsn(soiltfrac, tbq) / pp)
    qvg_m = snowfrac * qsg_m + (1.0 - snowfrac) * p1["qvg"]
    epot_m = -qkms * (qvatm - qsg_m)
    q1m = epot_m * ras
    condm = q1m <= 0.0
    dew_m = -epot_m
    qfx_c = -xlvm * rho * dew_m
    zroot = _zroot(cfg, grid, nroot)
    ett1 = zero
    tcols = []
    for k in range(nzs):
        tr = -vegfrac * q1m * tranf[:, k] * drycan / zroot
        inroot = k < nroot
        tr = W(inroot, tr, 0.0)
        ett1 = W(inroot, ett1 - tr, ett1)
        tcols.append(tr)
    edir1 = q1m * umveg * beta
    ec1 = q1m * wetcan * vegfrac
    eeta_e = (edir1 + ec1 + ett1) * 1.0e3
    qfx_e = xlvm * eeta_e
    qfx_m = W(condm, qfx_c, qfx_e)
    hfx_m = -d10 * (tabs - soiltfrac)
    soh = W(deep, thdifsn * rhocsn * (soiltfrac - p1["tsob"]) / snprim,
            (fsn * thdifsn * rhocsn + fso * thdif[:, 0] * rhcs) * (soiltfrac - p1["tsob"]) / snprim)
    x_m = (r21 + d9sn * r22sn) * (soiltfrac - tn) + xlvm * p1["r210"] * (qvg_m - qgold)
    snoh_m = (rnet - qfx_m - hfx_m - soh - x_m
              + rhonewcsn * newsnow / delt * (jnp.minimum(273.15, tabs) - soiltfrac)
              + rainf * cvw * prcpms * (jnp.maximum(273.15, tabs) - soiltfrac))
    snoh_m = jnp.maximum(0.0, snoh_m)
    smelt_m = snoh_m / xlmelt * 1.0e-3
    allevap = (epot_m > 0.0) & (snwepr <= epot_m * ras * delt)
    beta_a = snwepr / (epot_m * ras * delt)
    smelt_a = jnp.minimum(smelt_m, snwepr / delt - beta_a * epot_m * ras)
    smelt_b = jnp.maximum(0.0, smelt_m)
    eggl = ((rhosn < 350.0) | ((newsnow > 0.0) & (rhonewsn < 450.0))) & (p1["soilt"] < 283.0)
    smelt_b = W(eggl, jnp.minimum(smelt_b, delt / 60.0 * 5.6e-8 * meltfactor
                                  * jnp.maximum(1.0, p1["soilt"] - 273.15)), smelt_b)
    rr = jnp.maximum(0.0, snwepr / delt - beta * epot_m * ras)
    over = smelt_b > rr
    smelt_b = W(over, jnp.minimum(smelt_b, rr), smelt_b)
    smelt_m = W(allevap, smelt_a, smelt_b)
    beta_m = W(allevap, beta_a, beta)
    snwe_m = W(allevap | over, 0.0, snwe)
    snoh_m = smelt_m * xlmelt * 1.0e3
    rsmfrac = jnp.minimum(0.18, jnp.maximum(0.08, snwepr / 0.10 * 0.13))
    rsm_m = W(smelt_m > 0.0,
              W((snhei > 0.01) & (rhosn < 350.0), rsmfrac * smelt_m * delt, 0.0), 0.0)
    smelt_m = W(rsm_m > 0.0, jnp.maximum(0.0, smelt_m - rsm_m / delt), smelt_m)
    snwe_m = W(snwe_m > 0.0, jnp.maximum(0.0, snwepr - (smelt_m + beta_m * epot_m * ras) * delt), snwe_m)
    # ---- no-melt branch
    epot_n = -qkms * (qvatm - p1["qsg"])
    snwe_n = W((snhei != 0.0) & (beta == 1.0), jnp.maximum(0.0, snwepr - beta * epot_n * ras * delt), 0.0)

    # ---- second energy pass for melt columns (goto 212)
    p2 = energy_pass(snoh_m, beta_m, melt, snwe_m)
    pick = lambda a, b: W(melt, a, b)  # noqa: E731
    qvg_o = pick(p2["qvg"], p1["qvg"])
    qsg_o = pick(p2["qsg"], p1["qsg"])
    qcg_o = pick(p2["qcg"], p1["qcg"])
    soilt_o = pick(p2["soilt"], p1["soilt"])
    soilt1_o = pick(p2["soilt1"], p1["soilt1"])
    tso_o = W(melt[:, None], p2["tso"], p1["tso"])
    tsob_o = pick(p2["tsob"], p1["tsob"])
    ok = pick(p2["ok"] & p1["ok"], p1["ok"])
    smelt = pick(smelt_m, zero)
    snoh = pick(snoh_m, zero)
    beta_o = pick(beta_m, beta)
    snwe_o = pick(snwe_m, snwe_n)
    rsm = pick(rsm_m, zero)
    epot = pick(epot_m, epot_n)
    dew_o = pick(dew_m, zero)

    # ---- 220: density update by retained melt water
    upd = (smelt > 0.0) & (rsm > 0.0) & (snwe_o > rsm)
    xsn = (rhosn * (snwe_o - rsm) + 1.0e3 * rsm) / snwe_o
    rhosn_o = W(upd, jnp.minimum(jnp.maximum(58.8, xsn), 500.0), rhosn)
    rhocsn_o = W(upd, 2090.0 * rhosn_o, rhocsn)
    thdifsn_o = W(upd, _thdifsn(rhosn_o, rhocsn_o, newsnow, rhonewsn, snhei), thdifsn)
    # heat flux into the snow / soil
    snflx = W(deep, thdifsn_o * rhocsn_o * (soilt_o - tsob_o) / snprim,
              W(thin, (fsn * thdifsn_o * rhocsn_o + fso * thdif[:, 0] * rhcs) * (soilt_o - tsob_o) / snprim,
                d9sn * (soilt_o - tsob_o)))
    s = d9 * (tso_o[:, 0] - tso_o[:, 1])
    snhei_o = snwe_o * 1.0e3 / rhosn_o
    # bottom melt
    bm = (tso_o[:, 0] > 273.15) & (snhei_o > 0.0)
    hsn = W(snhei_o > deltsn + snth, snhei_o - deltsn, snhei_o)
    sfrac_b = snowfrac * 273.15 + (1.0 - snowfrac) * tso_o[:, 0]
    snohg = (tso_o[:, 0] - sfrac_b) * (cap[:, 0] * grid.zshalf[1] + rhocsn_o * 0.5 * hsn) / delt
    snohg = jnp.maximum(0.0, snohg)
    smeltg = snohg / xlmelt * 1.0e-3
    smeltg = W(((rhosn_o < 350.0) | ((newsnow > 0.0) & (rhonewsn < 450.0))) & (soilt_o < 283.0),
               jnp.minimum(smeltg, 5.8e-9), smeltg)
    smeltg = jnp.minimum(smeltg, snwe_o / delt)
    snwe_b = jnp.maximum(0.0, snwe_o - smeltg * delt)
    snhei_b = snwe_b * 1.0e3 / rhosn_o
    snwe_o = W(bm, snwe_b, snwe_o)
    snhei_o = W(bm, snhei_b, snhei_o)
    smelt = W(bm, smelt + smeltg, smelt)
    tso1 = W(bm & (snhei_b > 0.0), sfrac_b, tso_o[:, 0])
    tso_o = tso_o.at[:, 0].set(tso1)
    r210 = r211 * rho
    x = (r21 + d9sn * r22sn) * (soilt_o - tn) + xlvm * r210 * (qsg_o - qgold)
    x = (x - rhonewcsn * newsnow / delt * (jnp.minimum(273.15, tabs) - soilt_o)
         - rainf * cvw * prcpms * (jnp.maximum(273.15, tabs) - soilt_o))
    tsnav = W(snhei_o > 0.0,
              W(ilnb > 1,
                0.5 / snhei_o * ((soilt_o + soilt1_o) * deltsn + (soilt1_o + tso_o[:, 0]) * (snhei_o - deltsn)) - 273.15,
                0.5 * (soilt_o + tso_o[:, 0]) - 273.15),
              soilt_o - 273.15)
    return dict(snwe=snwe_o, snhei=snhei_o, beta=beta_o, rhosn=rhosn_o, rsm=rsm, tso=tso_o,
                soilt=soilt_o, soilt1=soilt1_o, tsnav=tsnav, qvg=qvg_o, qsg=qsg_o, qcg=qcg_o,
                smelt=smelt, snoh=snoh, snflx=snflx, s=s, x=x, dew=dew_o, epot=epot, ok=ok,
                nmelt=melt, bottom_melt=bm)


# ---------------------------------------------------------------------------
# SNOWSOIL
# ---------------------------------------------------------------------------


def snowsoil(cfg, tab, grid, tbq, p, *, iland, nroot, meltfactor, rhonewsn, delt, conflx,
             prcpms, rainf, newsnow, snhei, snwe, snowfrac, rhosn, patm, qvatm, qcatm,
             gswin, emiss, rnet, qkms, tkms, pc, cst, infwater, rho, vegfrac, lai, tabs,
             sat, cn, soilmois, tso, smfrkeep, keepfr, soilt, soilt1, tsnav, qvg, qsg, qcg,
             snom, mavail):
    """WRF ``SNOWSOIL`` (snow-covered column)."""

    dt_ = cfg.np_dtype
    cp, xlv, rovcp = cfg.cp, cfg.lv, cfg.rovcp
    dqm, qmin, ref, ksat = p["dqm"], p["qmin"], p["ref"], p["ksat"]
    cvw = 4.183e6
    xlvm = _c(xlv, dt_) + _c(3.35e5, dt_)
    deltsn = (_c(0.05, dt_) * _c(1.0e3, dt_)) / rhosn
    snth = (_c(0.01, dt_) * _c(1.0e3, dt_)) / rhosn
    deltsn = W((snhei >= deltsn + snth) & (snhei - deltsn - snth < snth), 0.5 * (snhei - snth), deltsn)
    ci = _c(900.0, dt_) * _c(2100.0, dt_)
    ras = rho * 1.0e-3
    riw = _c(900.0, dt_) * _c(1.0e-3, dt_)

    soilice, soiliqw = _freeze_partition(tso, soilmois, keepfr, smfrkeep, p, riw, None)
    tav, soilmoism, soilicem, soiliqwm, fwsat, lwsat = _mid_partition(tso, soilmois, keepfr, smfrkeep, p, riw)
    smfrkeep = W(soilice > 0.0, soilice, soilmois / riw)
    thdif, diffu, hydro, cap = soilprop(
        cfg, p, fwsat=fwsat, lwsat=lwsat, tav=tav, keepfr=keepfr, soilmois=soilmois,
        soiliqw=soiliqw, soilice=soilice, soilmoism=soilmoism, soiliqwm=soiliqwm,
        soilicem=soilicem, riw=riw, cvw=cvw, ci=ci)

    fq = qkms
    umveg = 1.0 - vegfrac
    epot = -fq * (qvatm - qsg)
    snwepr = snwe
    epdt = epot * ras * delt * umveg
    allev = (epdt > 0.0) & (snwepr <= epdt)
    beta = W(allev, snwepr / jnp.maximum(1.0e-8, epdt), 1.0)
    snwe = W(allev, 0.0, snwe)
    wetcan = jnp.minimum(0.25, jnp.power(jnp.maximum(0.0, cst / sat), cn))
    drycan = 1.0 - wetcan
    tranf, transum = transf(cfg, tab, p, grid, nroot=nroot, soiliqw=soiliqw, tabs=tabs,
                            lai=lai, gswin=gswin, pc=pc, iland=iland)
    told, smold = tso, soilmois
    st = snowtemp(cfg, grid, tbq, nroot=nroot, delt=delt, conflx=conflx, snwe=snwe,
                  snwepr=snwepr, snhei=snhei, newsnow=newsnow, snowfrac=snowfrac, beta=beta,
                  deltsn=deltsn, snth=snth, rhosn=rhosn, rhonewsn=rhonewsn,
                  meltfactor=meltfactor, prcpms=prcpms, rainf=rainf, patm=patm, tabs=tabs,
                  qvatm=qvatm, glw=None, emiss=emiss, rnet=rnet, qkms=qkms, tkms=tkms,
                  rho=rho, vegfrac=vegfrac, thdif=thdif, cap=cap, drycan=drycan,
                  wetcan=wetcan, tranf=tranf, transum=transum, tso=tso, soilt=soilt,
                  soilt1=soilt1, tsnav=tsnav, qvg=qvg, qsg=qsg, qcg=qcg)
    tso, soilt, soilt1, qvg, qsg, qcg = st["tso"], st["soilt"], st["soilt1"], st["qvg"], st["qsg"], st["qcg"]
    snwe, snhei, beta, rhosn, smelt, snoh = st["snwe"], st["snhei"], st["beta"], st["rhosn"], st["smelt"], st["snoh"]

    epot = -fq * (qvatm - qsg)
    ev = epot > 0.0
    zroot = _zroot(cfg, grid, nroot)
    ett1 = jnp.zeros_like(qvatm)
    tcols = []
    for k in range(cfg.nzs):
        tr = vegfrac * ras * fq * (qvatm - qsg) * tranf[:, k] * drycan / zroot
        inroot = (k < nroot) & ev
        tr = W(inroot, tr, 0.0)
        ett1 = W(inroot, ett1 - tr, ett1)
        tcols.append(tr)
    transp = jnp.stack(tcols, axis=1)
    dew = W(ev, 0.0, -epot)

    soilice, soiliqw = _freeze_partition(tso, soilmois, keepfr, smfrkeep, p, riw, riw)
    sm = soilmoist(cfg, grid, delt=delt, diffu=diffu, hydro=hydro, qsg=qsg, qvg=qvg, qcg=qcg,
                   qcatm=qcatm, qvatm=qvatm, prcp=-infwater, qkms=qkms, transp=transp,
                   drip=0.0, dew=0.0, smelt=smelt, soilice=soilice, vegfrac=vegfrac,
                   snowfrac=snowfrac, soilres=1.0, dqm=dqm, qmin=qmin, ref=ref, ksat=ksat,
                   ras=ras, riw=riw, soilmois=soilmois)
    soilmois = sm["soilmois"]
    tsnav = W(snhei == 0.0, soilt - 273.15, st["tsnav"])
    snom = snom + smelt * delt * 1.0e3
    keepfr = _keepfr_update(keepfr, soilice, tso, told, soilmois, smold)
    hfx = -tkms * cp * rho * (tabs - soilt) * jnp.power(
        (_c(_P1000MB, dt_) * _c(0.00001, dt_)) / patm, rovcp)
    hft = -tkms * cp * rho * (tabs - soilt)
    q1 = -fq * ras * (qvatm - qsg)
    cond = q1 < 0.0
    dew_c = qkms * (qvatm - qsg)
    if cfg.myj:
        eeta_c = -qkms * ras * (qvatm / (1.0 + qvatm) - qsg / (1.0 + qsg)) * 1.0e3
        cst_c = cst - eeta_c * delt * vegfrac
    else:
        eeta_c = -rho * dew_c
        cst_c = cst + delt * dew_c * ras * vegfrac
    qfx_c = xlvm * eeta_c
    eeta_c = -rho * dew_c
    edir1 = q1 * umveg * beta
    ec1 = q1 * wetcan * vegfrac
    cst_e = jnp.maximum(0.0, cst - ec1 * delt)
    if cfg.myj:
        eeta_e = -(qkms * ras * (qvatm / (1.0 + qvatm) - qsg / (1.0 + qsg)) * 1.0e3) * beta
    else:
        eeta_e = (edir1 + ec1 + ett1) * 1.0e3
    qfx_e = xlvm * eeta_e
    eeta_e = (edir1 + ec1 + ett1) * 1.0e3
    eeta = W(cond, eeta_c, eeta_e)
    qfx = W(cond, qfx_c, qfx_e)
    cst = W(cond, cst_c, cst_e)
    dew = W(cond, dew_c, dew)
    edir1 = W(cond, 0.0, edir1)
    ec1 = W(cond, 0.0, ec1)
    ett1 = W(cond, 0.0, ett1)
    s = st["snflx"]
    fltot = rnet - hft - xlvm * eeta - s - snoh - st["x"]
    return dict(soilmois=soilmois, tso=tso, smfrkeep=smfrkeep, keepfr=keepfr, dew=dew,
                soilt=soilt, soilt1=soilt1, tsnav=tsnav, qvg=qvg, qsg=qsg, qcg=qcg,
                smelt=smelt, snoh=snoh, snflx=st["snflx"], snom=snom, edir1=edir1, ec1=ec1,
                ett1=ett1, eeta=eeta, qfx=qfx, hfx=hfx, s=s, fltot=fltot,
                runoff1=sm["runoff1"], runoff2=sm["runoff2"], mavail=sm["mavail"],
                soilice=soilice, soiliqw=soiliqw, infiltr=sm["infiltrp"], snwe=snwe,
                snhei=snhei, rhosn=rhosn, cst=cst, ok=st["ok"], nmelt=st["nmelt"],
                bottom_melt=st["bottom_melt"])


# ---------------------------------------------------------------------------
# SFCTMP (land, no sea ice)
# ---------------------------------------------------------------------------


def sfctmp(cfg, tab, grid, tbq, p, *, conflx, nroot, meltfactor, iland, ivgtyp, prcpms,
           newsnms, snwe, snhei, snowfrac, rhosn, rhosnfall, snowrat, grauprat, icerat,
           curat, patm, tabs, qvatm, qcatm, rho, glw, gsw, emiss, qkms, tkms, pc, mavail,
           cst, vegfra, alb, znt, alb_snow, alb_snow_free, lai, sat, cn, soilm1d, ts1d,
           smfrkeep, keepfr, soilt, soilt1, tsnav, dew, qvg, qsg, qcg, snom, snowfallac):
    """WRF ``SFCTMP`` for land points (seaice < 0.5)."""

    dt_ = cfg.np_dtype
    delt = _c(cfg.dt, dt_)
    stbolt = cfg.stbolt
    c1sn, c2sn = 0.026, 21.0
    isice = cfg.isice
    rhowater = _c(_RHOWATER, dt_)
    zero = jnp.zeros_like(soilt)

    snhei_crit = _c(0.01601, dt_) * rhowater / rhosn
    snhei_crit_newsn = _c(0.0005, dt_) * rhowater / rhosn
    snowfrac = W(snhei == 0.0, 0.0, snowfrac)
    vegfrac = 0.01 * vegfra
    gswnew = gsw
    gswin = gsw / (1.0 - alb)
    albice = alb_snow_free
    emissn = 0.98
    emiss_snowfree = _take(tab.lemitbl, ivgtyp, dt_)

    # snow compaction (Koren et al. 1999)
    cmp = snhei > (_c(0.0081, dt_) * _c(1.0e3, dt_)) / rhosn
    bsn = delt / 3600.0 * c1sn * jnp.exp(0.08 * jnp.minimum(0.0, tsnav) - c2sn * rhosn * 1.0e-3)
    arg = bsn * snwe * 100.0
    xsn = rhosn * (jnp.exp(arg) - 1.0) / arg
    rhosn = W(cmp & ~(arg < 1.0e-4), jnp.minimum(jnp.maximum(58.8, xsn), 500.0), rhosn)

    snow_mosaic = W(snowfrac < 0.75, 1.0, 0.0)
    newsn = newsnms * delt
    has_new = newsn > 0.0
    rhonewsn_f = jnp.minimum(125.0, 1000.0 / jnp.maximum(8.0, 17.0 * jnp.tanh((276.65 - tabs) * 0.15)))
    rhonewgr = jnp.minimum(500.0, rhowater / jnp.maximum(2.0, 3.5 * jnp.tanh((274.15 - tabs) * 0.3333)))
    rhosnfall_f = jnp.minimum(500.0, jnp.maximum(
        58.8, rhonewsn_f * snowrat + rhonewgr * grauprat + rhonewsn_f * icerat + rhonewgr * curat))
    rhonewsn = W(has_new, rhosnfall_f, 100.0)
    rhosnfall = W(has_new, rhosnfall_f, rhosnfall)
    xsn2 = (rhosn * snwe + rhonewsn * newsn) / (snwe + newsn)
    rhosn = W(has_new, jnp.minimum(jnp.maximum(58.8, xsn2), 500.0), rhosn)
    rainf = W(prcpms != 0.0, 1.0, 0.0)

    vg = vegfrac > 0.01
    e = 1.0 - jnp.exp(-0.5 * lai)
    interw = W(vg, 0.25 * delt * prcpms * e * vegfrac, 0.0)
    intersn = W(vg, 0.25 * newsn * e * vegfrac, 0.0)
    infwater = W(vg, prcpms - interw / delt, prcpms)
    intwratio = W(vg & ((interw + intersn) > 0.0), interw / (interw + intersn), 0.0)
    dd1 = cst + interw + intersn
    drip = W(vg & (dd1 > sat), dd1 - sat, 0.0)
    cst = W(vg, W(dd1 > sat, sat, dd1), 0.0)

    snwe_n = jnp.maximum(0.0, snwe + newsn - intersn)
    dripliq = drip * intwratio
    mos_drip = has_new & (drip > 0.0) & (snow_mosaic == 1.0)
    snwe_n = W(has_new & (drip > 0.0), W(snow_mosaic == 1.0, snwe_n + (drip - dripliq), snwe_n + drip), snwe_n)
    infwater = W(mos_drip, infwater + dripliq, infwater)
    snwe = W(has_new, snwe_n, snwe)
    snhei = W(has_new, snwe * rhowater / rhosn, snhei)
    newsn = W(has_new, newsn * rhowater / rhonewsn, newsn)

    # ------------------------------------------------------------- snow on the ground
    snow = snhei > 0.0
    znt_in = znt
    sf1 = jnp.minimum(1.0, snhei / (2.0 * snhei_crit))
    sf2 = jnp.tanh(snhei / (2.5 * jnp.minimum(0.2, znt) * (rhosn / rhonewsn)))
    snowfrac_s = 0.5 * (sf1 + sf2)
    snowfracnewsn = W(newsn > 0.0, jnp.minimum(1.0, snowfallac * 1.0e-3 / snhei_crit_newsn), 0.0)
    snowfrac_s = W(ivgtyp == tab.urban, jnp.minimum(0.75, snowfrac_s), snowfrac_s)
    snow_mosaic = W(snow & (snowfrac_s < 0.75), 1.0, snow_mosaic)
    keep = W((snowfracnewsn > 0.99) & (rhosnfall < 450.0), 1.0, 0.0)
    snow_mosaic = W(snow & (keep == 1.0), 0.0, snow_mosaic)
    z0i = _c(tab.z0tbl[isice - 1], dt_)
    zc = (newsn == 0.0) & (znt <= 0.2) & (ivgtyp != isice)
    znt_s = W(zc & (snhei <= 2.0 * znt), 0.55 * znt + 0.45 * z0i,
              W(zc & (snhei > 2.0 * znt) & (snhei <= 4.0 * znt), 0.2 * znt + 0.8 * z0i,
                W(zc & (snhei > 4.0 * znt), jnp.broadcast_to(jnp.asarray(z0i), znt.shape), znt)))
    mos = snow_mosaic == 1.0
    albsn_m = W((keep > 0.9) & (alb_snow < 0.4), 0.7, alb_snow)
    albsn_n = jnp.maximum(keep * alb_snow, jnp.minimum(alb_snow_free + (alb_snow - alb_snow_free) * snowfrac_s, alb_snow))
    albsn_n = W((newsn > 0.0) & (keep > 0.9) & (albsn_n < 0.4), 0.7, albsn_n)
    albsn = W(mos, albsn_m, albsn_n)
    emiss_s = W(mos, jnp.asarray(emissn, dt_) + zero, jnp.maximum(
        keep * emissn, jnp.minimum(emiss_snowfree + (emissn - emiss_snowfree) * snowfrac_s, emissn)))
    alb_s = W((albsn < 0.4) | (keep == 1.0), albsn,
              jnp.minimum(albsn, jnp.maximum(albsn - 0.1 * (soilt - 263.15) / (_c(273.15, dt_) - _c(263.15, dt_)) * albsn,
                                             albsn - 0.05)))

    # snow-free part of a partially snow covered column (snow_mosaic) and no-snow columns
    t3 = stbolt * soilt * soilt * soilt
    upflux = t3 * soilt
    gsw_mos = gswin * (1.0 - alb_snow_free)
    rnet_mos = gsw_mos + emiss_snowfree * (glw - upflux)
    rnet_nos = gswnew + emiss * (glw - upflux)
    use_mos = snow & mos
    soil_in_gsw = W(use_mos, gsw_mos, gswnew)
    soil_emiss = W(use_mos, emiss_snowfree, emiss)
    soil_rnet = W(use_mos, rnet_mos, rnet_nos)
    soil_drip = W(snow, 0.0, drip)
    s_out = soil(cfg, tab, grid, tbq, p, iland=W(use_mos, ivgtyp, iland), nroot=nroot,
                 delt=delt, conflx=conflx, prcpms=prcpms, rainf=rainf, patm=patm, qvatm=qvatm,
                 qcatm=qcatm, gswin=gswin, emiss=soil_emiss, rnet=soil_rnet, qkms=qkms,
                 tkms=tkms, pc=pc, cst=cst, drip=soil_drip, infwater=infwater, rho=rho,
                 vegfrac=vegfrac, lai=lai, tabs=tabs, sat=sat, cn=cn, soilmois=soilm1d,
                 tso=ts1d, smfrkeep=smfrkeep, keepfr=keepfr, soilt=soilt, qvg=qvg, qsg=qsg,
                 qcg=qcg, mavail=mavail)
    del soil_in_gsw

    # snow part
    gswnew_s = gswin * (1.0 - alb_s)
    rnet_s = gswnew_s + emiss_s * (glw - upflux)
    snfr = W(mos, 1.0, snowfrac_s)
    w_out = snowsoil(cfg, tab, grid, tbq, p, iland=jnp.full_like(ivgtyp, isice), nroot=nroot,
                     meltfactor=meltfactor, rhonewsn=rhonewsn, delt=delt, conflx=conflx,
                     prcpms=prcpms, rainf=rainf, newsnow=newsn, snhei=snhei, snwe=snwe,
                     snowfrac=snfr, rhosn=rhosn, patm=patm, qvatm=qvatm, qcatm=qcatm,
                     gswin=gswin, emiss=emiss_s, rnet=rnet_s, qkms=qkms, tkms=tkms, pc=pc,
                     cst=cst, infwater=infwater, rho=rho, vegfrac=vegfrac, lai=lai, tabs=tabs,
                     sat=sat, cn=cn, soilmois=soilm1d, tso=ts1d, smfrkeep=smfrkeep,
                     keepfr=keepfr, soilt=soilt, soilt1=soilt1, tsnav=tsnav, qvg=qvg,
                     qsg=qsg, qcg=qcg, snom=snom, mavail=mavail)

    sfw = snowfrac_s
    one = 1.0 - sfw

    def blend(a, b):  # snow-free a, snow b
        if a.ndim == 2:
            return a * one[:, None] + b * sfw[:, None]
        return a * one + b * sfw

    def sel(no_snow, snow_pure, snow_mos):
        if no_snow.ndim == 2:
            return W(snow[:, None], W(mos[:, None], snow_mos, snow_pure), no_snow)
        return W(snow, W(mos, snow_mos, snow_pure), no_snow)

    keep_s = W((sfw > 0.5)[:, None], w_out["keepfr"], s_out["keepfr"])
    out = {}
    out["soilm1d"] = sel(s_out["soilmois"], w_out["soilmois"], blend(s_out["soilmois"], w_out["soilmois"]))
    out["ts1d"] = sel(s_out["tso"], w_out["tso"], blend(s_out["tso"], w_out["tso"]))
    out["smfrkeep"] = sel(s_out["smfrkeep"], w_out["smfrkeep"], blend(s_out["smfrkeep"], w_out["smfrkeep"]))
    out["keepfr"] = sel(s_out["keepfr"], w_out["keepfr"], W(snow[:, None] & mos[:, None], keep_s, keep_s))
    out["soilice"] = sel(s_out["soilice"], w_out["soilice"], blend(s_out["soilice"], w_out["soilice"]))
    out["soiliqw"] = sel(s_out["soiliqw"], w_out["soiliqw"], blend(s_out["soiliqw"], w_out["soiliqw"]))
    for k in ("dew", "soilt", "qvg", "qsg", "qcg", "cst", "eeta", "qfx", "hfx", "s",
              "runoff1", "runoff2", "infiltr"):
        out[k] = sel(s_out[k], w_out[k], blend(s_out[k], w_out[k]))
    out["mavail"] = sel(s_out["mavail"], w_out["mavail"], s_out["mavail"] * one + 1.0 * sfw)
    out["soilt1"] = W(snow, w_out["soilt1"], soilt1)
    out["tsnav"] = W(snow, w_out["tsnav"], tsnav)
    out["smelt"] = W(snow, w_out["smelt"], 0.0)
    out["snom"] = W(snow, w_out["snom"], snom)
    out["snwe"] = W(snow, w_out["snwe"], snwe)
    snhei_o = W(snow, w_out["snhei"], snhei)
    rhosn_o = W(snow, w_out["rhosn"], rhosn)
    alb_w = alb_s
    alb_mos = jnp.maximum(keep * alb_w, jnp.minimum(alb_snow_free + (alb_w - alb_snow_free) * sfw, alb_w))
    emiss_mos = jnp.maximum(keep * emissn, jnp.minimum(emiss_snowfree + (emissn - emiss_snowfree) * sfw, emissn))
    alb_o = W(mos, alb_mos, alb_w)
    emiss_o = W(mos, emiss_mos, emiss_s)
    # end of the snow branch
    gone = snhei_o == 0.0
    alb_o = W(gone, alb_snow_free, alb_o)
    sf1b = jnp.minimum(1.0, snhei_o / (2.0 * snhei_crit))
    sf2b = jnp.tanh(snhei_o / (2.5 * jnp.minimum(0.2, znt_s) * (rhosn_o / rhonewsn)))
    snowfrac_e = W(gone, snowfrac_s, 0.5 * (sf1b + sf2b))
    snowfrac_e = W(ivgtyp == tab.urban, jnp.minimum(0.75, snowfrac_e), snowfrac_e)
    out["snowfallac"] = W(snow, snowfallac + newsn * 1.0e3, snowfallac)
    out["snhei"] = snhei_o
    out["rhosn"] = rhosn_o
    out["snowfrac"] = W(snow, snowfrac_e, snowfrac)
    out["alb"] = W(snow, alb_o, alb)
    out["emiss"] = W(snow, emiss_o, emiss)
    out["znt"] = W(snow, znt_s, znt_in)
    out["rhosnfall"] = rhosnfall
    out["ok"] = W(snow, w_out["ok"] & W(mos, s_out["ok"], True), s_out["ok"])
    out["nmelt"] = snow & w_out["nmelt"]
    out["bottom_melt"] = snow & w_out["bottom_melt"]
    return out


# ---------------------------------------------------------------------------
# LSMRUC driver step
# ---------------------------------------------------------------------------


def _root_depth(cfg, iforest):
    """LSMRUC :783-813: nroot and meltfactor per column."""

    zs = cfg.zs
    def first(th):
        for k in range(2, cfg.nzs + 1):
            if zs[k - 1] >= th:
                return k
        return 4
    n_open = first(0.4)
    n_forest = first(1.1)
    nroot = W(iforest > 2, n_open, n_forest).astype(jnp.int32)
    meltfactor = W(iforest > 2, 2.0, 0.85)
    return nroot, meltfactor


def lsmruc_step(state: Mapping, forcing: Mapping, static: Mapping, cfg: RucConfig,
                tab: RucTables, ktau, *, landusef=None, soilctop=None):
    """One call of WRF ``LSMRUC`` on a batch of columns.

    ``state``: dict with :data:`STATE_SCALARS` ``(ncol,)`` and :data:`STATE_PROFILES`
    ``(ncol, nzs)``; ``forcing``: :data:`FORCING_FIELDS` ``(ncol,)``; ``static``:
    :data:`STATIC_FIELDS`; ``ktau``: WRF ``itimestep`` (1 triggers the first-step
    initialisation block).  Returns the new state dict plus ``unsupported`` (bool) and
    ``vilka_ok`` diagnostics.
    """

    dt_ = cfg.np_dtype
    grid = _grid(cfg)
    tbq = tbq_table(dt_)
    nzs = cfg.nzs
    s = {k: jnp.asarray(v, dt_) for k, v in state.items()}
    f = {k: jnp.asarray(v, dt_) for k, v in forcing.items()}
    ivgtyp = jnp.asarray(static["ivgtyp"], jnp.int32)
    isltyp = jnp.asarray(static["isltyp"], jnp.int32)
    xland = jnp.asarray(static["xland"], dt_)
    xice = jnp.asarray(static["xice"], dt_)
    tbot = jnp.asarray(static["tbot"], dt_)
    shdmin = jnp.asarray(static["shdmin"], dt_)
    shdmax = jnp.asarray(static["shdmax"], dt_)
    albbck = jnp.asarray(static["albbck"], dt_)
    delt = _c(cfg.dt, dt_)
    first = jnp.asarray(ktau) == 1

    soilt, tso = s["soilt"], s["tso"]
    # ---- ktau == 1 initialisation (LSMRUC :481-565)
    keepfr3d = W(first, 0.0, s["keepfr3dflag"])
    bad1 = (s["soilt1"] < 170.0) | (s["soilt1"] > 400.0)
    soilt1 = W(first & bad1, W(s["snowc"] > 0.0, 0.5 * (soilt + tso[:, 0]), tso[:, 0]), s["soilt1"])
    tsnav = W(first, 0.5 * (soilt + tso[:, 0]) - 273.15, s["tsnav"])
    patmb0 = f["p8w"] * 1.0e-2
    qsg = W(first, qsn(soilt, tbq) / patmb0, s["qsg"])
    qcg = W(first & ((s["qcg"] < 0.0) | (s["qcg"] > 0.1)), f["qc3d"], s["qcg"])
    qvg = W(first & ((s["qvg"] <= 0.0) | (s["qvg"] > 0.1)), qsg * s["mavail"], s["qvg"])
    qsfc = W(first, qvg / (1.0 + qvg), s["qsfc"])
    snom = W(first, 0.0, s["snom"])
    snowfallac = W(first, 0.0, s["snowfallac"])
    rhosnf = W(first, -1.0e3, s["rhosnf"])
    dew = W(first, 0.0, s["dew"])
    sfcrunoff = W(first, 0.0, s["sfcrunoff"])
    udrunoff = W(first, 0.0, s["udrunoff"])
    acrunoff = W(first, 0.0, s["acrunoff"])
    chklowq = W(first, 1.0, s["chklowq"])

    tabs, qvatm, qcatm = f["t3d"], f["qv3d"], f["qc3d"]
    patm = f["p8w"] * 1.0e-5
    conflx = f["z3d"] * 0.5
    rho = f["rho3d"]
    rainbl = f["rainbl"]
    zero = jnp.zeros_like(tabs)
    if cfg.frpcpn:
        rainncv, frzfrac = f["rainncv"], f["frzfrac"]
        prcpncliq = rainncv * (1.0 - frzfrac)
        prcpncfr = rainncv * frzfrac
        c1 = (frzfrac > 0.0) & (tabs < 273.0)
        cold = tabs < 273.0
        conv = rainbl - rainncv
        prcpculiq = W(c1, jnp.maximum(0.0, conv * (1.0 - frzfrac)), W(cold, 0.0, jnp.maximum(0.0, conv)))
        prcpcufr = W(c1, jnp.maximum(0.0, conv * frzfrac), W(cold, jnp.maximum(0.0, conv), 0.0))
        prcpms = (prcpncliq + prcpculiq) / delt * 1.0e-3
        newsnms = (prcpncfr + prcpcufr) / delt * 1.0e-3
        graupamt = f["graupelncv"]
        tot = prcpncfr + prcpcufr
        pos = tot > 0.0
        snowrat = W(pos, jnp.minimum(1.0, jnp.maximum(0.0, f["snowncv"] / tot)), 0.0)
        grauprat = W(pos, jnp.minimum(1.0, jnp.maximum(0.0, graupamt / tot)), 0.0)
        icerat = W(pos, jnp.minimum(1.0, jnp.maximum(0.0, (prcpncfr - f["snowncv"] - graupamt) / tot)), 0.0)
        curat = W(pos, jnp.minimum(1.0, jnp.maximum(0.0, prcpcufr / tot)), 0.0)
    else:
        cold = tabs <= 273.15
        prcpms = W(cold, 0.0, rainbl / delt * 1.0e-3)
        newsnms = W(cold, rainbl / delt * 1.0e-3, 0.0)
        snowrat = W(cold, 1.0, 0.0)
        grauprat = icerat = curat = zero
    precipfr = newsnms * delt * 1.0e3

    mavail_in = s["mavail"]
    if cfg.myj:
        qkms = f["chs"]
        tkms = f["chs"]
    else:
        qkms = f["flqc"] / rho / mavail_in
        tkms = f["flhc"] / rho / (cfg.cp * (1.0 + 0.84 * qvatm))
    snwe = s["snow"] * 1.0e-3
    snhei = s["snowh"]
    canwatr = s["canwat"] * 1.0e-3
    snowfrac = s["snowc"]
    rhosnfall = rhosnf
    rhosn = W((s["snow"] > 0.0) & (s["snowh"] > 0.0), s["snow"] / s["snowh"], 300.0)

    sv = soilvegin(cfg, tab, ivgtyp=ivgtyp, isltyp=isltyp, vegfra=s["vegfra"], shdmin=shdmin,
                   shdmax=shdmax, znt=s["znt"], lai=s["lai"], landusef=landusef, soilctop=soilctop)
    p = {k: sv[k] for k in ("rhocs", "bclh", "dqm", "ksat", "psis", "qmin", "ref", "wilt", "qwrtz")}
    znt, lai, emissl, pc = sv["znt"], sv["lai"], sv["emiss"], sv["pc"]
    cn = _c(tab.cfactr_data, dt_)
    sat = 5.0e-4
    nroot, meltfactor = _root_depth(cfg, sv["iforest"])

    water = (xland - 1.5) >= 0.0
    seaice = xice >= cfg.xice_threshold
    oob = (nroot + 1) > nzs
    unsupported = (~water) & (seaice | (oob if cfg.oob_root_policy == "fail" else False) | sv["bad_area"])

    qmin, dqm, ref = p["qmin"], p["dqm"], p["ref"]
    soilm1d = jnp.minimum(jnp.maximum(0.0, s["soilmois"] - qmin[:, None]), dqm[:, None])
    soiliqw0 = jnp.minimum(jnp.maximum(0.0, s["sh2o"] - qmin[:, None]), soilm1d)
    del soiliqw0
    lmavail = jnp.maximum(0.00001, jnp.minimum(1.0, soilm1d[:, 0] / (ref - qmin)))

    o = sfctmp(cfg, tab, grid, tbq, p, conflx=conflx, nroot=nroot, meltfactor=meltfactor,
               iland=ivgtyp, ivgtyp=ivgtyp, prcpms=prcpms, newsnms=newsnms, snwe=snwe,
               snhei=snhei, snowfrac=snowfrac, rhosn=rhosn, rhosnfall=rhosnfall,
               snowrat=snowrat, grauprat=grauprat, icerat=icerat, curat=curat, patm=patm,
               tabs=tabs, qvatm=qvatm, qcatm=qcatm, rho=rho, glw=f["glw"], gsw=f["gsw"],
               emiss=emissl, qkms=qkms, tkms=tkms, pc=pc, mavail=lmavail, cst=canwatr,
               vegfra=s["vegfra"], alb=s["alb"], znt=znt, alb_snow=s["snoalb"],
               alb_snow_free=albbck, lai=lai, sat=sat, cn=cn, soilm1d=soilm1d, ts1d=tso,
               smfrkeep=s["smfr3d"], keepfr=keepfr3d, soilt=soilt, soilt1=soilt1,
               tsnav=tsnav, dew=dew, qvg=qvg, qsg=qsg, qcg=qcg, snom=snom,
               snowfallac=snowfallac)
    soilm1d = o["soilm1d"]
    if cfg.mosaic_lu == 1:
        factor = jnp.maximum(0.0, jnp.minimum(1.0, (s["vegfra"] - shdmin) / jnp.maximum(1.0, shdmax - shdmin)))
        lucrop = landusef[:, tab.crop - 1]
        lunat = landusef[:, tab.natural - 1]
        do_crop = ((lucrop > 0.0) | (lunat > 0.0)) & (factor > 0.75)
        cols = []
        for k in range(nzs):
            cropsm = 1.1 * p["wilt"] - qmin
            cropfr = jnp.minimum(1.0, lucrop + 0.4 * lunat)
            newsm = cropsm * cropfr + (1.0 - cropfr) * soilm1d[:, k]
            ck = do_crop & (k < nroot) & (soilm1d[:, k] < newsm)
            cols.append(W(ck, newsm, soilm1d[:, k]))
        soilm1d = jnp.stack(cols, axis=1)

    zsmain, zshalf = grid.zsmain, grid.zshalf
    smavail = jnp.zeros_like(tabs)
    smmax = jnp.zeros_like(tabs)
    for k in range(nzs - 1):
        dz = zshalf[k + 1] - zshalf[k]
        smavail = smavail + (qmin + soilm1d[:, k]) * dz
        smmax = smmax + (qmin + dqm) * dz
    dz = zsmain[nzs - 1] - zshalf[nzs - 1]
    smavail = smavail + (qmin + soilm1d[:, nzs - 1]) * dz
    smmax = smmax + (qmin + dqm) * dz
    sfcrunoff_l = sfcrunoff + o["runoff1"] * delt * 1000.0
    udrunoff_l = udrunoff + o["runoff2"] * delt * 1000.0
    acrunoff_l = acrunoff + o["runoff1"] * delt * 1000.0
    smavail = smavail * 1000.0
    smmax = smmax * 1000.0
    soilmois_l = soilm1d + qmin[:, None]
    sh2o_l = jnp.minimum(o["soiliqw"] + qmin[:, None], soilmois_l)
    tso_l = o["ts1d"].at[:, nzs - 1].set(tbot)
    patmb = f["p8w"] * 1.0e-2
    q2sat = qsn(tabs, tbq) / patmb
    qvg_l = o["qvg"]
    qsfc_l = qvg_l / (1.0 + qvg_l)
    chklowq_l = W((qvatm >= q2sat * 0.95) & (qvatm < qvg_l), 0.0, 1.0)
    emiss_l = W(s["snow"] == 0.0, _take(tab.lemitbl, ivgtyp, dt_), o["emiss"])
    sfcevp_l = s["sfcevp"] + o["eeta"] * delt
    snowc_l = W((o["snowfrac"] > 0.0) & (xice >= cfg.xice_threshold), o["snowfrac"] * xice, o["snowfrac"])
    sfcevp_l = sfcevp_l + o["eeta"] * delt  # WRF adds qfx*dt a second time (:1116)

    land = {
        "soilt": o["soilt"], "soilt1": o["soilt1"], "tsnav": o["tsnav"],
        "snow": o["snwe"] * 1000.0, "snowh": o["snhei"], "snowc": snowc_l,
        "canwat": o["cst"] * 1000.0, "alb": o["alb"], "emiss": emiss_l, "znt": o["znt"],
        "z0": o["znt"], "lai": lai, "mavail": o["mavail"], "vegfra": s["vegfra"],
        "snoalb": s["snoalb"], "qvg": qvg_l, "qsg": o["qsg"], "qcg": o["qcg"],
        "dew": o["dew"], "qsfc": qsfc_l, "chklowq": chklowq_l, "hfx": o["hfx"],
        "qfx": o["eeta"], "lh": o["qfx"], "grdflx": -1.0 * o["s"],
        "sfcrunoff": sfcrunoff_l, "udrunoff": udrunoff_l, "acrunoff": acrunoff_l,
        "sfcexc": tkms, "sfcevp": sfcevp_l, "smavail": smavail, "smmax": smmax,
        "snowfallac": o["snowfallac"], "acsnow": s["acsnow"], "snom": o["snom"],
        "rhosnf": o["rhosnfall"], "precipfr": precipfr,
        "tso": tso_l, "soilmois": soilmois_l, "sh2o": sh2o_l, "smfr3d": o["smfrkeep"],
        "keepfr3dflag": o["keepfr"],
    }
    # ---- water points (LSMRUC :828-856)
    qvg_w = qsn(soilt, tbq) / patmb
    ones_p = jnp.ones_like(tso)
    wat = {
        "soilt": soilt, "soilt1": soilt1, "tsnav": tsnav, "snow": zero, "snowh": zero,
        "snowc": zero, "canwat": s["canwat"], "alb": s["alb"], "emiss": s["emiss"],
        "znt": znt, "z0": s["z0"], "lai": lai, "mavail": mavail_in, "vegfra": s["vegfra"],
        "snoalb": s["snoalb"], "qvg": qvg_w, "qsg": qsg, "qcg": qcg, "dew": dew,
        "qsfc": qvg_w / (1.0 + qvg_w), "chklowq": jnp.ones_like(tabs), "hfx": s["hfx"],
        "qfx": s["qfx"], "lh": s["lh"], "grdflx": s["grdflx"], "sfcrunoff": sfcrunoff,
        "udrunoff": udrunoff, "acrunoff": acrunoff, "sfcexc": s["sfcexc"],
        "sfcevp": s["sfcevp"], "smavail": jnp.ones_like(tabs), "smmax": jnp.ones_like(tabs),
        "snowfallac": snowfallac, "acsnow": s["acsnow"], "snom": snom, "rhosnf": rhosnf,
        "precipfr": precipfr, "tso": jnp.broadcast_to(soilt[:, None], tso.shape),
        "soilmois": ones_p, "sh2o": ones_p, "smfr3d": s["smfr3d"], "keepfr3dflag": keepfr3d,
    }
    new = {}
    for k in land:
        a, b = wat[k], land[k]
        m = water[:, None] if b.ndim == 2 else water
        new[k] = W(m, a, b)
    new["unsupported"] = unsupported
    new["vilka_ok"] = water | o["ok"]
    new["surface_melt"] = (~water) & o["nmelt"]  # branch census: SNOWTEMP nmelt=1 iteration
    new["bottom_melt"] = (~water) & o["bottom_melt"]  # branch census: SNOWTEMP bottom melt (tso(1)>273.15)
    return new


# ---------------------------------------------------------------------------
# RUCLSMINIT (cold start, not restart)
# ---------------------------------------------------------------------------


def ruclsminit(tab: RucTables, *, tslb, smois, isltyp, ivgtyp, xice, dtype="float32"):
    """WRF ``RUCLSMINIT``: sh2o, smfr3d, mavail, znt from wrfinput TSLB/SMOIS."""

    dt_ = np.dtype(dtype)
    tslb = jnp.asarray(tslb, dt_)
    smois = jnp.asarray(smois, dt_)
    isltyp = jnp.asarray(isltyp, jnp.int32)
    ivgtyp = jnp.asarray(ivgtyp, jnp.int32)
    xice = jnp.asarray(xice, dt_)
    riw = _c(900.0, dt_) * _c(1.0e-3, dt_)
    xlmelt = 3.35e5
    znt = _take(tab.z0tbl, ivgtyp, dt_)
    dqm = _take(tab.maxsmc, isltyp, dt_) - _take(tab.drysmc, isltyp, dt_)
    ref = _take(tab.refsmc, isltyp, dt_)
    psis = -_take(tab.satpsi, isltyp, dt_)
    qmin = _take(tab.drysmc, isltyp, dt_)
    bclh = _take(tab.bb, isltyp, dt_)
    tln = jnp.log(tslb / 273.15)
    liq = (dqm[:, None] + qmin[:, None]) * jnp.power(
        xlmelt * (tslb - 273.15) / tslb / 9.81 / psis[:, None], -1.0 / bclh[:, None])
    liq = jnp.minimum(jnp.maximum(0.0, liq), smois)
    frozen = tln < 0.0
    sh2o_l = W(frozen, liq, smois)
    smfr_l = W(frozen, (smois - liq) / riw, 0.0)
    mav_l = jnp.maximum(0.00001, jnp.minimum(1.0, (smois[:, 0] - qmin) / (ref - qmin)))
    ice = xice > 0.0
    water = isltyp == 14
    sh2o = W(ice[:, None], 0.0, W(water[:, None], 1.0, sh2o_l))
    smfr3d = W(ice[:, None], 1.0, W(water[:, None], 0.0, smfr_l))
    mavail = W(ice | water, 1.0, mav_l)
    if bool(jnp.any(isltyp < 1)):
        raise ValueError("module_sf_ruclsm.f: lsminit: out of range value of isltyp")
    return dict(sh2o=sh2o, smfr3d=smfr3d, mavail=mavail, znt=znt)


__all__ = [
    "FORCING_FIELDS", "RucConfig", "RucTables", "STATE_PROFILES", "STATE_SCALARS",
    "STATIC_FIELDS", "load_ruc_tables", "lsmruc_step", "mminlu_to_ruc", "qsn",
    "ruclsminit", "tbq_table", "vilka",
]
