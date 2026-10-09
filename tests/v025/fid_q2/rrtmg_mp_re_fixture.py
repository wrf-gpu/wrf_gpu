"""RE01: pristine WRF V4.7.1 RRTMG with the TRUE PROD/WN3 caller flags on real WN3 0227 columns (reader + bounds).

Generator and caller audit: proofs/rrtmg_mp_re/README.md.  Every arm is the unchanged pristine RRTMG_LWRAD/RRTMG_SWRAD
(libwrflib.a) on the same real operands (186 CPU-WRF 0227 d02/d03 columns at tau 12/18/24/30/36, 44 levels):

  A  WRF truth: has_reqc/i/s = 1 (use_mp_re=1 default with mp8 + ra 4) fed Thompson calc_effectRad radii, cldovrlp = 2
  B  has_req = 0 (use_mp_re = 0): WRF's own 5/10/10 um with inflglw 2 / iceflglw 3 (NOT 10/30/75)
  C  has_req = 1 with constant 10/30/75 um (the port's fixed cldprmc radii), cldovrlp = 2
  D  C with cldovrlp = 1 (random overlap, the port's only McICA path) = the port today
  E  A with cldovrlp = 1 = MP radii wired, overlap still random

Keys
  in_*   caller operands: t (t_phy), p (p_phy, the MP/calc_effectRad pressure), pi (pi_phy), th, rho (grid%rho),
         p_hyd / p_hyd_w (radiation P / P8W, phy_prep REAL recursion), t8w, dz8w (nz+1, top 0), qv, qc, qi, ni, qs, qr, qg,
         qc_rad / qi_rad (radiation_driver icloud_bl merge view), cldfra, tsk, emiss, albedo, xland, xlat, xlong, snow,
         xice, coszen_hist (history COSZEN = the tau-30 min call), julian, julday, gmt, tau, p_top, p_d01, lat_d01
  col_*  per column: solcon, declin, coszen (radiation call at tau: calc_coszen(xtime+radt/2)), o3 (o3rad VMR: CAM
         climatology on the d01 PARENT column, copied level-wise to the nest = Registry rdf=(p2c)), re_cloud / re_ice /
         re_snow [m] (calc_effectRad + mp_gt_driver clamps; Nc is unused because is_aerosol_aware = .false.)
  {A..E}_*  hlw, hlwc, hsw, hswc [K/s, theta tendency = RTHRATEN*]; lwup, lwupc, lwdn, lwdnc, swup, swupc, swdn, swdnc
         [W/m2, nz+2 interfaces as WRF stores them (k = kts..kte+2): 0 = surface (p_hyd_w[0]), nz = model top (p_top),
         nz+1 = the FIRST of RRTMG's 13 buffer interfaces above p_top (deltap 4 hPa), NOT the TOA -> gates compare
         interfaces 0..nz (LEVEL_SLICE); TOA values are the *t scalars];
         scalars glw, olr, lwcf, lwupt, lwuptc, lwdnt, lwdntc, lwupb, lwupbc, lwdnb, lwdnbc, gsw, swcf, swupt, swuptc,
         swdnt, swdntc, swupb, swupbc, swdnb (SWDOWN before slope/shading), swdnbc, swddir, swddni, swddif, swdownc, coszr
  category, domain, ji
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).with_name("fixtures") / "rrtmg_mp_re_0227_v1.npz"
SHA256 = "bba1a3e080010dbadbe39c48d07113858220d104faf18bb72f0735a5ee541ac0"
ARMS = ("A", "B", "C", "D", "E")
# Frozen tier-1 RRTMG bounds (fixtures/manifests/analytic-rrtmg-{lw,sw}-column-v1.yaml): abs + rel * |ref|.
FLUX_BOUND = (1.0, 0.05)
HEATING_BOUND = (1.0e-4, 0.05)
SCALAR_FLUXES = ("glw", "olr", "lwupt", "lwuptc", "lwdnb", "lwupb", "gsw", "swupt", "swdnb", "swupb", "swddir", "swddif")
LEVEL_FLUXES = ("lwup", "lwdn", "swup", "swdn")
LEVEL_SLICE = slice(0, -1)  # surface .. model-top interfaces; WRF's last stored index is inside the buffer layer
HEATING = ("hlw", "hsw")


def load(path: Path = FIXTURE) -> dict[str, np.ndarray]:
    """Loads the registered fixture after checking its content hash."""

    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{path}: sha256 {digest} is not the registered RE01 fixture {SHA256}")
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def bound_ratio(reference, candidate, bound) -> np.ndarray:
    """|candidate - reference| / (abs + rel * |reference|); > 1 means outside the frozen bound."""

    ref = np.asarray(reference, np.float64)
    return np.abs(np.asarray(candidate, np.float64) - ref) / (bound[0] + bound[1] * np.abs(ref))


def columns_outside(fx: dict[str, np.ndarray], candidate: dict[str, np.ndarray], reference_arm: str = "A") -> np.ndarray:
    """Per-column flag: any provided output outside the frozen bounds vs the reference arm.

    ``candidate`` maps output keys (``glw``, ``swdnb``, ``hlw``, ``lwup`` ...) to arrays shaped like the fixture's.
    A non-finite candidate or reference value marks its column outside (a NaN ratio never compares > 1);
    a candidate whose shape differs from the reference is refused.
    """

    out = np.zeros(fx["category"].shape, bool)
    for key, value in candidate.items():
        bound = HEATING_BOUND if key in HEATING or key in ("hlwc", "hswc") else FLUX_BOUND
        reference = fx[f"{reference_arm}_{key}"]
        value = np.asarray(value)
        if value.shape != reference.shape:
            raise ValueError(f"{key}: candidate shape {value.shape} != fixture shape {reference.shape}")
        if key.rstrip("c") in LEVEL_FLUXES:
            reference, value = reference[:, LEVEL_SLICE], value[:, LEVEL_SLICE]
        ratio = bound_ratio(reference, value, bound)
        bad = (ratio > 1) | ~np.isfinite(ratio) | ~np.isfinite(value) | ~np.isfinite(reference)
        out |= bad.reshape(bad.shape[0], -1).any(1)
    return out


def arm_outputs(fx: dict[str, np.ndarray], arm: str) -> dict[str, np.ndarray]:
    """The gate's comparison set of one arm (scalar fluxes, level fluxes, heating)."""

    return {key: fx[f"{arm}_{key}"] for key in SCALAR_FLUXES + LEVEL_FLUXES + HEATING}
