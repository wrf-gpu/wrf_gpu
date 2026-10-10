"""WRF RUC multi-layer soil land-surface model (``sf_surface_physics=3``) -- v0.18 API.

Compatibility front end kept for the v0.17/v0.18 single-column callers
(``ruc_columns``/``ruc_column``).  The physics is the faithful JAX port of the
unmodified WRF LSMRUC in :mod:`gpuwrf.physics.ruclsm` (the earlier approximate
no-snow port that lived here reached only ~1e-4 relative agreement and is gone).

The v0.17 oracle (``proofs/v017/oracle/ruclsm``) drives LSMRUC with the literal
``ROVCP=0.2857`` and the non-standard 6-level grid ``0/.05/.2/.4/1/2 m``; this
front end keeps those defaults so its historical savepoint still matches.  The
operational path is :mod:`gpuwrf.coupling.ruc_surface_hook`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import jax.numpy as jnp

from gpuwrf.physics.ruclsm import RucConfig, load_ruc_tables, lsmruc_step

RUC_ORACLE_DIR = "proofs/v017/oracle/ruclsm"
RUC_SAVEPOINT = "proofs/v017/savepoints/ruclsm/fp64/ruclsm_fp64.json"

RUC_NUM_SOIL_LAYERS = 6
V017_ZS = (0.0, 0.05, 0.20, 0.40, 1.00, 2.00)
V017_ROVCP = 0.2857


class RucLandState(NamedTuple):
    """Minimal RUC column carry of the v0.18 API (see ``coupling.ruc_surface_hook``)."""

    soilt: "object"
    tso: "object"
    soilmois: "object"
    sh2o: "object"
    smfr3d: "object"
    keepfr3dflag: "object"
    snow: "object"
    snowh: "object"
    qsfc: "object"

    def replace(self, **updates) -> "RucLandState":
        return self._replace(**updates)


def ruc_columns(
    *,
    gsw,
    glw,
    emiss,
    tabs,
    qv,
    qc,
    rho,
    p8w,
    z3d,
    rainbl,
    vegfra,
    flhc,
    flqc,
    tbot,
    xland,
    mavail,
    ivgtyp,
    isltyp,
    soilt,
    tso,
    soilmois,
    sh2o,
    snow=None,
    snowh=None,
    alb=None,
    dt: float = 180.0,
    nsteps: int = 6,
    zs=V017_ZS,
    rovcp: float = V017_ROVCP,
    dtype: str = "float64",
):
    """Run ``nsteps`` LSMRUC calls (ktau=1..nsteps) over a batch of land columns."""

    dt_ = np.dtype(dtype)

    def a(x):
        return np.asarray(x, dtype=dt_)

    gsw = a(gsw)
    n = gsw.shape[0]
    z = np.zeros(n, dt_)
    soilt = a(soilt)
    soilmois = a(soilmois)
    state = dict(
        soilt=soilt, soilt1=soilt, tsnav=soilt - dt_.type(273.15),
        snow=z if snow is None else a(snow), snowh=z if snowh is None else a(snowh), snowc=z,
        canwat=z, alb=np.full(n, 0.18, dt_) if alb is None else a(alb), emiss=a(emiss),
        znt=np.full(n, 0.05, dt_), z0=np.full(n, 0.05, dt_), lai=np.full(n, 2.0, dt_),
        mavail=a(mavail), vegfra=a(vegfra), snoalb=np.full(n, 0.70, dt_), qvg=z, qsg=z, qcg=z,
        dew=z, qsfc=z, chklowq=np.ones(n, dt_), hfx=z, qfx=z, lh=z, grdflx=z, sfcrunoff=z,
        udrunoff=z, acrunoff=z, sfcexc=z, sfcevp=z, smavail=z, smmax=z, snowfallac=z, acsnow=z,
        snom=z, rhosnf=np.full(n, -1.0e3, dt_), precipfr=z, tso=a(tso), soilmois=soilmois,
        sh2o=a(sh2o), smfr3d=np.zeros_like(soilmois), keepfr3dflag=np.zeros_like(soilmois),
    )
    forcing = dict(t3d=a(tabs), qv3d=a(qv), qc3d=a(qc), p8w=a(p8w), rho3d=a(rho), z3d=a(z3d),
                   glw=a(glw), gsw=gsw, chs=z, flqc=a(flqc), flhc=a(flhc), rainbl=a(rainbl),
                   rainncv=a(rainbl), snowncv=z, graupelncv=z, frzfrac=z)
    static = dict(ivgtyp=np.asarray(ivgtyp).astype(np.int32), isltyp=np.asarray(isltyp).astype(np.int32),
                  xland=a(xland), xice=z, tbot=a(tbot), shdmin=np.ones(n, dt_),
                  shdmax=np.full(n, 80.0, dt_), albbck=np.full(n, 0.18, dt_))
    cfg = RucConfig(dt=float(dt), zs=tuple(float(v) for v in zs), dtype=dtype, rovcp=float(rovcp))
    tab = load_ruc_tables()
    for ktau in range(1, int(nsteps) + 1):
        out = lsmruc_step(state, forcing, static, cfg, tab, ktau)
        if bool(jnp.any(out["unsupported"])):
            raise NotImplementedError("RUC sea-ice columns are not ported (fail closed)")
        state = {k: out[k] for k in state}
    return state


def ruc_column(**kwargs):
    """Single-column convenience wrapper around :func:`ruc_columns`."""

    batched = {}
    for key, value in kwargs.items():
        if key in {"dt", "nsteps", "zs", "rovcp", "dtype"}:
            batched[key] = value
        else:
            batched[key] = np.asarray(value, dtype=np.float64)[None, ...]
    out = ruc_columns(**batched)
    return {key: value[0] for key, value in out.items()}


__all__ = [
    "RUC_ORACLE_DIR",
    "RUC_SAVEPOINT",
    "RUC_NUM_SOIL_LAYERS",
    "RucLandState",
    "ruc_column",
    "ruc_columns",
]
