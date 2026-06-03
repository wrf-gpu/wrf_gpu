"""Grell-Freitas scale-aware cumulus for WRF ``cu_physics=3``.

This module exposes:

- ``grell_freitas_column`` — a faithful single-column port of the WRF
  ``cu_physics=3`` call path ``GFDRV`` -> ``cup_gf`` (deep) + ``cup_gf_sh``
  (shallow), delegating to :mod:`gpuwrf.physics._gf_reference`. The reference is
  a line-faithful NumPy translation of pristine ``module_cu_gf_deep.F`` /
  ``module_cu_gf_sh.F`` / ``module_cu_gf_wrfdrv.F`` (no clamps, no masks, no
  reduced-tendency candidate). It reproduces the WRF-module savepoints across
  the deep / shallow / non-triggering / scale-aware (coarse, fine) regimes.
- ``grell_freitas_scale_factor`` — the WRF scale-dependence factor ``sig``.
- ``grell_freitas_step`` — the v0.6.0 frozen-interface adapter returning a
  ``PhysicsStepResult``.

Status / provenance
-------------------
The GF closure ensemble is inherently sequential (level-by-level plume /
downdraft / 16-member dynamic-control). The savepoint gate runs a single column
on CPU, so the faithful artifact here is a sequential NumPy reference. A
vectorized JAX hot-path is a separate optimization sprint (recorded for the
manager) and is not needed for the v0.6.0 equivalence claim.

``cugd_*`` carry provenance: pristine WRF ``GFDRV`` does not accept or update
``cugd_*`` arrays — those are wired through ``G3DRV``/spread in WRF, not the
``GFDRV`` call path used by ``cu_physics=3``. The v0.6.0 adapter therefore
carries ``cugd_*`` as zero/diagnostic only; the correct integration carry is the
combined ``RTHCUTEN/RQVCUTEN/RQCCUTEN/RQICUTEN`` tendencies + ``RAINCV`` (see the
handoff note).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from gpuwrf.contracts.physics_interfaces import (
    PhysicsCarry,
    PhysicsDiagnostics,
    PhysicsStepResult,
    PhysicsTendency,
)
from gpuwrf.physics import _gf_reference as _ref

G = 9.81
CP = 1004.0
R_D = 287.0
XLV = 2.5e6
FRH_THRESH = 0.9
RH_TRIGGER = 0.97
ENTR_RATE_LAND = 7.0e-5
ENTR_RATE_WATER = 7.0e-5
ENTR_RATE_MID = 1.0e-4

CARRY_KEYS = (
    "cugd_qvten",
    "cugd_tten",
    "cugd_qvtens",
    "cugd_ttens",
    "cugd_qcten",
    "xmb_shallow",
    "k22_shallow",
    "kbcon_shallow",
    "ktop_shallow",
)


def saturation_mixing_ratio(p_pa, t_k):
    """Liquid saturation mixing ratio (diagnostic helper)."""
    p_pa = np.asarray(p_pa, dtype=np.float64)
    t_k = np.asarray(t_k, dtype=np.float64)
    es = 611.2 * np.exp(17.67 * (t_k - 273.15) / (t_k - 29.65))
    es = np.minimum(es, 0.95 * p_pa)
    return 0.622 * es / (p_pa - es)


def grell_freitas_scale_factor(dx, *, csum=0.0, xland=1.0, imid=0):
    """WRF GF scale-aware factor ``sig=(1-frh)^2`` (module_cu_gf_deep.F)."""
    dx = float(dx)
    csum = float(csum)
    xland = float(xland)
    entr_rate = ENTR_RATE_LAND - min(20.0, csum) * 3.0e-6
    if xland == 0.0:
        entr_rate = ENTR_RATE_WATER
    if int(imid) == 1:
        entr_rate = ENTR_RATE_MID
    radius = 0.2 / entr_rate
    frh = min(1.0, np.pi * radius * radius / (dx * dx))
    if frh > FRH_THRESH:
        frh = FRH_THRESH
    return (1.0 - frh) ** 2


def grell_freitas_column(
    t,
    qv,
    p,
    dz,
    rho,
    w,
    *,
    dt,
    dx,
    pi_exner=None,
    u=None,
    v=None,
    rthblten=None,
    rqvblten=None,
    kpbl=5,
    hfx=0.0,
    qfx=0.0,
    xland=1.0,
    ht=0.0,
    csum=0.0,
):
    """Run one faithful Grell-Freitas column (deep + shallow), WRF cu_physics=3.

    Inputs use WRF GF driver units: ``t`` K, ``qv`` kg/kg, ``p`` Pa, ``dz`` m,
    ``rho`` kg/m3, ``w`` m/s, ``dt`` s, ``dx`` m. ``u``/``v`` are mass-point
    winds (m/s); ``rthblten``/``rqvblten`` are the PBL forcing tendencies that
    GFDRV folds into the forced sounding (default zero). Returned tendency names
    mirror WRF's cumulus-driver arrays. ``csum`` is unused by ``cu_physics=3``
    (GFDRV passes 0) and is accepted only for signature stability.
    """
    del csum  # GFDRV standalone path uses csum=0.

    t = np.asarray(t, dtype=np.float64)
    qv = np.asarray(qv, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    kx = t.shape[0]

    if pi_exner is None:
        pi_exner = (p / 1.0e5) ** (R_D / CP)
    pi_exner = np.asarray(pi_exner, dtype=np.float64)

    zeros = np.zeros(kx, dtype=np.float64)
    u = zeros if u is None else np.asarray(u, dtype=np.float64)
    v = zeros if v is None else np.asarray(v, dtype=np.float64)
    rthblten = zeros if rthblten is None else np.asarray(rthblten, dtype=np.float64)
    rqvblten = zeros if rqvblten is None else np.asarray(rqvblten, dtype=np.float64)

    out = _ref.gfdrv(
        t, qv, p, pi_exner, dz, rho, u, v, w, rthblten, rqvblten,
        dt=float(dt), dx=float(dx), hfx=float(hfx), qfx=float(qfx),
        kpbl=int(kpbl), xland=float(xland), ht=float(ht),
        ishallow_g3=1, ichoice=0,
    )

    sig = grell_freitas_scale_factor(dx, csum=0.0, xland=xland)
    # scale_normalized: diagnostic only (relative to the 9 km undamped factor).
    sig9 = grell_freitas_scale_factor(9000.0, csum=0.0, xland=xland)
    scale_normalized = min(1.0, np.sqrt(sig / sig9)) if sig9 > 0 else 0.0

    rthcuten = out["RTHCUTEN"]
    rqvcuten = out["RQVCUTEN"]
    rqccuten = out["RQCCUTEN"]
    rqicuten = out["RQICUTEN"]
    rqrcuten = np.zeros(kx, dtype=np.float64)
    rqscuten = np.zeros(kx, dtype=np.float64)

    trigger_deep = bool(out["KTOP_DEEP"] > 0 and out["RAINCV"] > 0.0)
    trigger_shallow = bool(out["XMB_SHALLOW"] > 0.0 or out["KTOP_SHALLOW"] > 0)

    qv_sink_mm = -float(np.sum(rqvcuten * rho * dz) * dt)

    return {
        "RTHCUTEN": rthcuten,
        "RQVCUTEN": rqvcuten,
        "RQCCUTEN": rqccuten,
        "RQRCUTEN": rqrcuten,
        "RQICUTEN": rqicuten,
        "RQSCUTEN": rqscuten,
        "RAINCV": np.float64(out["RAINCV"]),
        "PRATEC": np.float64(out["PRATEC"]),
        "KTOP_DEEP": np.int32(out["KTOP_DEEP"]),
        "XMB_SHALLOW": np.float64(out["XMB_SHALLOW"]),
        "K22_SHALLOW": np.int32(out["K22_SHALLOW"]),
        "KBCON_SHALLOW": np.int32(out["KBCON_SHALLOW"]),
        "KTOP_SHALLOW": np.int32(out["KTOP_SHALLOW"]),
        "SCALE_FACTOR": np.float64(sig),
        "SCALE_NORMALIZED": np.float64(scale_normalized),
        "TRIGGER_DEEP": trigger_deep,
        "TRIGGER_SHALLOW": trigger_shallow,
        "QVSINK_MM": np.float64(qv_sink_mm),
        "IERR_DEEP": np.int32(out["IERR_DEEP"]),
        "IERR_SHALLOW": np.int32(out["IERR_SHALLOW"]),
    }


# The reference port is sequential NumPy on CPU; there is no JIT wrapper. Kept
# as an alias so existing imports of ``grell_freitas_column_jit`` resolve.
grell_freitas_column_jit = grell_freitas_column


def grell_freitas_step(
    state: Mapping[str, object],
    carry: Mapping[str, object] | None = None,
    *,
    dt,
    dx,
    kpbl=5,
    hfx=0.0,
    qfx=0.0,
    xland=1.0,
    ht=0.0,
) -> PhysicsStepResult:
    """Frozen-interface adapter returning ``PhysicsStepResult``.

    ``state`` may provide either ``t`` or ``theta`` plus ``pi``. Required arrays:
    ``qv``, ``p``, ``dz``, ``rho``, ``w``. Optional: ``u``, ``v``,
    ``rthblten``, ``rqvblten`` (PBL forcing folded into the GF forced sounding).
    """
    del carry
    pi = state.get("pi")
    if "t" in state:
        t = state["t"]
    else:
        if pi is None:
            raise ValueError("state must provide either 't' or both 'theta' and 'pi'")
        t = np.asarray(state["theta"]) * np.asarray(pi)
    out = grell_freitas_column(
        t,
        state["qv"],
        state["p"],
        state["dz"],
        state["rho"],
        state["w"],
        dt=dt,
        dx=dx,
        pi_exner=pi,
        u=state.get("u"),
        v=state.get("v"),
        rthblten=state.get("rthblten"),
        rqvblten=state.get("rqvblten"),
        kpbl=kpbl,
        hfx=hfx,
        qfx=qfx,
        xland=xland,
        ht=ht,
    )

    zeros_3d = np.zeros_like(np.asarray(out["RTHCUTEN"]))
    # cugd_* are NOT updated by the GFDRV call path (see module docstring); the
    # integration-relevant carry is the combined cumulus tendencies + RAINCV.
    cumulus_carry = {
        "cugd_qvten": zeros_3d,
        "cugd_tten": zeros_3d,
        "cugd_qvtens": zeros_3d,
        "cugd_ttens": zeros_3d,
        "cugd_qcten": zeros_3d,
        "xmb_shallow": out["XMB_SHALLOW"],
        "k22_shallow": out["K22_SHALLOW"],
        "kbcon_shallow": out["KBCON_SHALLOW"],
        "ktop_shallow": out["KTOP_SHALLOW"],
    }
    diagnostics = {
        "raincv": out["RAINCV"],
        "pratec": out["PRATEC"],
        "ktop_deep": out["KTOP_DEEP"],
        "scale_factor": out["SCALE_FACTOR"],
        "scale_normalized": out["SCALE_NORMALIZED"],
        "trigger_deep": out["TRIGGER_DEEP"],
        "trigger_shallow": out["TRIGGER_SHALLOW"],
        "qv_sink_mm": out["QVSINK_MM"],
        "rthcuten": out["RTHCUTEN"],
        "rqvcuten": out["RQVCUTEN"],
        "rqccuten": out["RQCCUTEN"],
        "rqrcuten": out["RQRCUTEN"],
        "rqicuten": out["RQICUTEN"],
        "rqscuten": out["RQSCUTEN"],
    }
    tendency = PhysicsTendency(
        state_tendencies={
            "theta": out["RTHCUTEN"],
            "qv": out["RQVCUTEN"],
            "qc": out["RQCCUTEN"],
            "qr": out["RQRCUTEN"],
            "qi": out["RQICUTEN"],
            "qs": out["RQSCUTEN"],
        },
        accumulator_increments={"rainc_acc": out["RAINCV"]},
        diagnostics=diagnostics,
    )
    tendency.validate_keys()
    return PhysicsStepResult(
        tendency=tendency,
        carry=PhysicsCarry(cumulus=cumulus_carry),
        diagnostics=PhysicsDiagnostics(cumulus=diagnostics),
    )


__all__ = [
    "CARRY_KEYS",
    "FRH_THRESH",
    "RH_TRIGGER",
    "grell_freitas_column",
    "grell_freitas_column_jit",
    "grell_freitas_scale_factor",
    "grell_freitas_step",
    "saturation_mixing_ratio",
]
